"""Tests for compliance_check tool."""
from __future__ import annotations

import pytest
import httpx
import respx

from ghostfolio_agent.tools.compliance_check import ComplianceCheckTool


MOCK_REPORT_RESPONSE = {
    "xRay": {
        "categories": [
            {
                "key": "diversification",
                "name": "Diversification",
                "rules": [
                    {
                        "key": "currency-diversification",
                        "name": "Currency Diversification",
                        "isActive": True,
                        "value": True,
                        "evaluation": "At least 2 currencies",
                    },
                    {
                        "key": "portfolio-concentration",
                        "name": "Portfolio Concentration",
                        "isActive": True,
                        "value": False,
                        "evaluation": "Single position exceeds 50%",
                        "configuration": {
                            "threshold": {"max": 1, "min": 0, "step": 0.01, "unit": "%"},
                            "thresholdMax": 0.5,
                        },
                    },
                ],
            },
            {
                "key": "emergency-fund",
                "name": "Emergency Fund",
                "rules": [
                    {
                        "key": "has-emergency-fund",
                        "name": "Emergency Fund",
                        "isActive": True,
                        "value": True,
                        "evaluation": "Emergency fund is sufficient",
                    },
                ],
            },
            {
                "key": "fees",
                "name": "Fees",
                "rules": [
                    {
                        "key": "low-fees",
                        "name": "Low Fees",
                        "isActive": False,
                        "value": None,
                    },
                ],
            },
        ],
        "statistics": {
            "rulesActiveCount": 3,
            "rulesFulfilledCount": 2,
        },
    }
}

MOCK_DETAILS_RESPONSE = {
    "holdings": {
        "AAPL": {
            "assetClass": "EQUITY",
            "currency": "USD",
            "valueInPercentage": 0.6,
        },
        "BND": {
            "assetClass": "FIXED_INCOME",
            "currency": "USD",
            "valueInPercentage": 0.3,
        },
        "SIE.DE": {
            "assetClass": "EQUITY",
            "currency": "EUR",
            "valueInPercentage": 0.1,
        },
    },
    "markets": {
        "developedMarkets": {"valueInPercentage": 0.85},
        "emergingMarkets": {"valueInPercentage": 0.15},
    },
    "marketsAdvanced": {
        "northAmerica": {"valueInPercentage": 0.70},
        "europe": {"valueInPercentage": 0.15},
        "asiaPacific": {"valueInPercentage": 0.05},
        "japan": {"valueInPercentage": 0.03},
        "emergingMarkets": {"valueInPercentage": 0.07},
    },
    "accounts": {
        "acc-1": {"name": "Brokerage", "valueInPercentage": 0.8},
        "acc-2": {"name": "Crypto", "valueInPercentage": 0.2},
    },
    "summary": {
        "cash": 5000,
        "fees": 42.5,
        "emergencyFund": {"total": 10000},
        "dividendInBaseCurrency": 350,
    },
}


@pytest.fixture
def tool():
    return ComplianceCheckTool()


@pytest.mark.asyncio
@respx.mock
async def test_returns_compliance_report(tool):
    """Tool should return structured compliance results with composition."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    assert result["tool_name"] == "compliance_check"
    assert "rules" in result["result"]
    assert "portfolio_composition" in result["result"]
    assert "summary" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_shows_passing_and_failing_rules(tool):
    """Tool should clearly indicate which rules pass and which fail."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    rules_by_name = {r["name"]: r for r in result["result"]["rules"]}
    assert rules_by_name["Currency Diversification"]["status"] == "PASS"
    assert rules_by_name["Portfolio Concentration"]["status"] == "FAIL"


@pytest.mark.asyncio
@respx.mock
async def test_includes_summary(tool):
    """Tool should include a human-readable summary."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    assert "2 of 3 rules passed" in result["result"]["summary"]
    assert "1 failed" in result["result"]["summary"]


@pytest.mark.asyncio
@respx.mock
async def test_includes_thresholds(tool):
    """Tool should include threshold targets when available."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    concentration = next(
        r for r in result["result"]["rules"] if r["name"] == "Portfolio Concentration"
    )
    assert "target" in concentration
    assert "50.0%" in concentration["target"]


@pytest.mark.asyncio
@respx.mock
async def test_ignores_boolean_thresholds(tool):
    """Tool should NOT produce a target when thresholdMax is a boolean (real API behavior)."""
    boolean_report = {
        "xRay": {
            "categories": [
                {
                    "key": "risk",
                    "name": "Risk",
                    "rules": [
                        {
                            "key": "equity-cluster",
                            "name": "Equity Cluster Risk",
                            "isActive": True,
                            "value": False,
                            "evaluation": "The equity contribution (85.6%) exceeds 82.0%",
                            "configuration": {
                                "threshold": {"max": 1, "min": 0, "step": 0.01, "unit": "%"},
                                "thresholdMax": True,
                            },
                        },
                    ],
                },
            ],
            "statistics": {"rulesActiveCount": 1, "rulesFulfilledCount": 0},
        }
    }
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=boolean_report)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    rule = result["result"]["rules"][0]
    assert rule["name"] == "Equity Cluster Risk"
    assert rule["status"] == "FAIL"
    assert "target" not in rule  # boolean thresholds should be ignored
    assert "85.6%" in rule["evaluation"]  # evaluation string has the real data


@pytest.mark.asyncio
@respx.mock
async def test_skips_inactive_rules(tool):
    """Tool should skip inactive rules."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    rule_names = [r["name"] for r in result["result"]["rules"]]
    assert "Low Fees" not in rule_names


@pytest.mark.asyncio
@respx.mock
async def test_composition_asset_classes(tool):
    """Tool should return asset class allocation."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    ac = result["result"]["portfolio_composition"]["asset_classes"]
    assert "EQUITY" in ac
    assert "FIXED_INCOME" in ac
    assert ac["EQUITY"] == "70.0%"  # 60% + 10%
    assert ac["FIXED_INCOME"] == "30.0%"


@pytest.mark.asyncio
@respx.mock
async def test_composition_currencies(tool):
    """Tool should return currency allocation."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    currencies = result["result"]["portfolio_composition"]["currencies"]
    assert "USD" in currencies
    assert "EUR" in currencies


@pytest.mark.asyncio
@respx.mock
async def test_composition_regions(tool):
    """Tool should return regional market allocation."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    regions = result["result"]["portfolio_composition"]["regions"]
    assert "North America" in regions
    assert "Europe" in regions
    assert regions["North America"] == "70.0%"


@pytest.mark.asyncio
@respx.mock
async def test_composition_accounts(tool):
    """Tool should return account allocation."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    accounts = result["result"]["portfolio_composition"]["accounts"]
    assert "Brokerage" in accounts
    assert accounts["Brokerage"] == "80.0%"


@pytest.mark.asyncio
@respx.mock
async def test_composition_summary_metrics(tool):
    """Tool should include cash, fees, emergency fund in summary."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt")

    s = result["result"]["portfolio_composition"]["summary"]
    assert s["cash"] == 5000
    assert s["fees_total"] == 42.5
    assert s["emergency_fund"] == 10000


@pytest.mark.asyncio
@respx.mock
async def test_returns_error_on_401(tool):
    """Tool should return error dict on unauthorized."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(401, json={"message": "Unauthorized"})
    )

    result = await tool.execute(jwt="bad-jwt")

    assert "error" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_graceful_when_details_fails(tool):
    """Tool should still return rules even if details endpoint fails."""
    respx.get("http://localhost:3333/api/v1/portfolio/report").mock(
        return_value=httpx.Response(200, json=MOCK_REPORT_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    result = await tool.execute(jwt="fake-jwt")

    assert result["result"]["rules"]
    assert result["result"]["portfolio_composition"] == {}


@pytest.mark.asyncio
async def test_tool_metadata(tool):
    """Tool should have correct name and description."""
    assert tool.name == "compliance_check"
    assert "compliance" in tool.description.lower()
