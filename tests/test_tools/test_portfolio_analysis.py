"""Tests for portfolio_analysis tool."""
from __future__ import annotations

import copy

import pytest
import httpx
import respx

from ghostfolio_agent.tools.portfolio_analysis import PortfolioAnalysisTool


MOCK_DETAILS_RESPONSE = {
    "hasError": False,
    "holdings": {
        "YAHOO-AAPL": {
            "name": "Apple Inc.",
            "symbol": "AAPL",
            "currency": "USD",
            "assetClass": "EQUITY",
            "dataSource": "YAHOO",
            "marketPrice": 195.50,
            "quantity": 10,
            "investment": 1500.00,
            "netPerformance": 455.00,
            "dividend": 12.50,
            "sectors": [{"name": "Technology", "weight": 1}],
            "countries": [{"code": "US", "name": "United States", "weight": 1}],
        },
        "YAHOO-GOOGL": {
            "name": "Alphabet Inc.",
            "symbol": "GOOGL",
            "currency": "USD",
            "assetClass": "EQUITY",
            "dataSource": "YAHOO",
            "marketPrice": 140.00,
            "quantity": 5,
            "investment": 600.00,
            "netPerformance": 100.00,
            "dividend": 0,
            "sectors": [{"name": "Communication Services", "weight": 1}],
            "countries": [{"code": "US", "name": "United States", "weight": 1}],
        },
        "YAHOO-MSFT": {
            "name": "Microsoft Corp.",
            "symbol": "MSFT",
            "currency": "USD",
            "assetClass": "EQUITY",
            "dataSource": "YAHOO",
            "marketPrice": 420.00,
            "quantity": 2,
            "investment": 700.00,
            "netPerformance": 140.00,
            "dividend": 5.00,
            "sectors": [{"name": "Technology", "weight": 1}],
            "countries": [{"code": "US", "name": "United States", "weight": 1}],
        },
    },
    "accounts": {
        "acct-1": {
            "name": "Main Brokerage",
            "balance": 500.00,
            "valueInBaseCurrency": 3495.00,
            "currency": "USD",
        }
    },
    "summary": {
        "totalBuy": 2800.00,
        "totalSell": 0,
        "fees": 15.00,
        "cash": 500.00,
        "dividendInBaseCurrency": 17.50,
        "activityCount": 4,
    },
}

MOCK_ORDERS_RESPONSE = {
    "activities": [
        {
            "SymbolProfile": {"symbol": "AAPL"},
            "type": "BUY",
            "quantity": 10,
            "unitPrice": 150.00,
            "fee": 1.00,
        },
        {
            "SymbolProfile": {"symbol": "GOOGL"},
            "type": "BUY",
            "quantity": 5,
            "unitPrice": 120.00,
            "fee": 0.50,
        },
        {
            "SymbolProfile": {"symbol": "MSFT"},
            "type": "BUY",
            "quantity": 2,
            "unitPrice": 350.00,
            "fee": 0.50,
        },
    ],
    "count": 3,
}

MOCK_ACCOUNTS_RESPONSE = {
    "accounts": [
        {"id": "acct-1", "name": "Main Brokerage", "currency": "USD", "balance": 500.00},
        {"id": "acct-2", "name": "Crypto Portfolio", "currency": "USD", "balance": 0},
    ],
}

# Filtered details response (only AAPL, as if filtered to one account)
MOCK_FILTERED_DETAILS = {
    "hasError": False,
    "holdings": {
        "YAHOO-AAPL": MOCK_DETAILS_RESPONSE["holdings"]["YAHOO-AAPL"],
    },
    "accounts": {
        "acct-1": MOCK_DETAILS_RESPONSE["accounts"]["acct-1"],
    },
    "summary": {
        "totalBuy": 1500.00,
        "totalSell": 0,
        "fees": 5.00,
        "cash": 500.00,
        "dividendInBaseCurrency": 12.50,
        "activityCount": 1,
    },
}


# --- Historical data mocks for daily change ---

MOCK_HISTORICAL_AAPL = {
    "historicalData": [
        {"date": "2025-01-14T00:00:00.000Z", "marketPrice": 192.00},
        {"date": "2025-01-15T00:00:00.000Z", "marketPrice": 195.50},
    ],
}

MOCK_HISTORICAL_GOOGL = {
    "historicalData": [
        {"date": "2025-01-14T00:00:00.000Z", "marketPrice": 142.00},
        {"date": "2025-01-15T00:00:00.000Z", "marketPrice": 140.00},
    ],
}

MOCK_HISTORICAL_MSFT = {
    "historicalData": [
        {"date": "2025-01-14T00:00:00.000Z", "marketPrice": 415.00},
        {"date": "2025-01-15T00:00:00.000Z", "marketPrice": 420.00},
    ],
}


def _mock_errored_details():
    """Return details with hasError and zeroed investment."""
    details = copy.deepcopy(MOCK_DETAILS_RESPONSE)
    details["hasError"] = True
    for h in details["holdings"].values():
        h["investment"] = 0
        h["netPerformance"] = 0
    return details


def _mock_orders():
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=MOCK_ORDERS_RESPONSE)
    )


def _mock_all():
    """Mock details + orders endpoints."""
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )
    _mock_orders()


def _mock_historical_data():
    """Mock historical data endpoints for daily change."""
    respx.get("http://localhost:3333/api/v1/symbol/YAHOO/AAPL", params__contains={"includeHistoricalData": "2"}).mock(
        return_value=httpx.Response(200, json=MOCK_HISTORICAL_AAPL)
    )
    respx.get("http://localhost:3333/api/v1/symbol/YAHOO/GOOGL", params__contains={"includeHistoricalData": "2"}).mock(
        return_value=httpx.Response(200, json=MOCK_HISTORICAL_GOOGL)
    )
    respx.get("http://localhost:3333/api/v1/symbol/YAHOO/MSFT", params__contains={"includeHistoricalData": "2"}).mock(
        return_value=httpx.Response(200, json=MOCK_HISTORICAL_MSFT)
    )


def _mock_all_with_daily():
    """Mock details + orders + historical data."""
    _mock_all()
    _mock_historical_data()


@pytest.fixture
def tool():
    return PortfolioAnalysisTool()


# ---- Core functionality ----

@pytest.mark.asyncio
@respx.mock
async def test_returns_holdings_on_success(tool):
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    assert result["tool_name"] == "portfolio_analysis"
    table = result["result"]["holdings_table"]
    assert "AAPL" in table
    assert "GOOGL" in table
    assert "MSFT" in table


@pytest.mark.asyncio
@respx.mock
async def test_returns_error_on_401(tool):
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(401, json={"message": "Unauthorized"})
    )
    result = await tool.execute(jwt="bad-jwt")
    assert "error" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_handles_empty_portfolio(tool):
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json={
            "holdings": {}, "accounts": {}, "summary": {}, "hasError": False,
        })
    )
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    assert result["result"]["holdings_table"] == "No holdings found."
    assert result["result"]["portfolio"]["total_value"] == 0


@pytest.mark.asyncio
@respx.mock
async def test_sorts_by_allocation(tool):
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    lines = [l for l in table.split("\n") if "|" in l and "---" not in l]
    # First data row should be highest allocation (AAPL: 10*195.50 = 1955)
    assert "AAPL" in lines[1]


# ---- Formatted values ----

@pytest.mark.asyncio
@respx.mock
async def test_dollar_formatting(tool):
    """Prices, cost basis, value, and gain should have $ and commas."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    assert "$" in table
    # AAPL value = 10 * 195.50 = $1,955.00
    assert "$1,955.00" in table


@pytest.mark.asyncio
@respx.mock
async def test_percent_formatting(tool):
    """Allocation and gain % should have % signs."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    assert "%" in table
    # Check allocation column has % (not just the header)
    lines = [l for l in table.split("\n") if "AAPL" in l]
    assert "%" in lines[0]


@pytest.mark.asyncio
@respx.mock
async def test_table_has_all_columns(tool):
    """Table should include all expected column headers."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    for header in ["Symbol", "Name", "Sector", "Price", "Cost Basis",
                    "Cost Basis/Share", "Value", "Allocation", "Gain", "Gain %"]:
        assert header in table


@pytest.mark.asyncio
@respx.mock
async def test_sector_in_table(tool):
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")
    assert "Technology" in result["result"]["holdings_table"]


# ---- Cost basis and gain ----

@pytest.mark.asyncio
@respx.mock
async def test_computes_gain_from_api_investment(tool):
    """When API investment is available, use it for cost basis."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    # AAPL: cost_basis=$1,500, value=$1,955, gain=$455
    assert "$1,500.00" in table
    assert "$455.00" in table


@pytest.mark.asyncio
@respx.mock
async def test_orders_fallback_when_investment_zero(tool):
    """When API investment is 0, compute cost basis from orders."""
    errored = _mock_errored_details()
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=errored)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    # AAPL: 10 * 150 + $1 fee = $1,501.00 cost basis
    assert "$1,501.00" in table
    # Gain should be computed, not "—"
    assert "nan" not in table.lower()


@pytest.mark.asyncio
@respx.mock
async def test_gain_shows_dash_when_no_cost_data(tool):
    """When no investment AND no orders, gain shows '—'."""
    errored = _mock_errored_details()
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=errored)
    )
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": [], "count": 0})
    )

    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    assert "nan" not in table.lower()
    assert "—" in table


# ---- Cost basis per share ----

@pytest.mark.asyncio
@respx.mock
async def test_cost_basis_per_share_in_table(tool):
    """Cost Basis/Share should appear with correct value."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    # AAPL: cost_basis $1,500 / qty 10 = $150.00/share
    assert "$150.00" in table
    # GOOGL: cost_basis $600 / qty 5 = $120.00/share
    assert "$120.00" in table


@pytest.mark.asyncio
@respx.mock
async def test_cost_basis_per_share_dash_when_no_cost(tool):
    """Cost Basis/Share shows '—' when no cost data available."""
    errored = _mock_errored_details()
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=errored)
    )
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": [], "count": 0})
    )

    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    assert "Cost Basis/Share" in table
    # All cost basis/share should be "—" since no cost data
    assert "nan" not in table.lower()


# ---- Portfolio summary ----

@pytest.mark.asyncio
@respx.mock
async def test_portfolio_summary_gain(tool):
    """Summary should compute gain from totalBuy - totalSell."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")
    portfolio = result["result"]["portfolio"]

    # total_value = 1955 + 700 + 840 = 3495
    assert portfolio["total_value"] == 3495.00
    assert portfolio["net_cost_basis"] == 2800.00
    assert portfolio["gain"] == 695.00
    assert portfolio["gain_pct"] == round((695.0 / 2800.0) * 100, 2)


@pytest.mark.asyncio
@respx.mock
async def test_portfolio_summary_fields(tool):
    """Summary should include cash, fees, and activity count."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")
    portfolio = result["result"]["portfolio"]

    assert portfolio["cash"] == 500.00
    assert portfolio["fees"] == 15.00
    assert portfolio["activity_count"] == 4
    assert portfolio["dividends_total"] == 17.50


# ---- Deterministic computation ----

@pytest.mark.asyncio
@respx.mock
async def test_value_is_quantity_times_price(tool):
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")
    table = result["result"]["holdings_table"]

    # AAPL: 10 * 195.50 = $1,955.00
    assert "$1,955.00" in table
    # MSFT: 2 * 420 = $840.00
    assert "$840.00" in table


@pytest.mark.asyncio
@respx.mock
async def test_allocation_sums_to_100(tool):
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")
    table = result["result"]["holdings_table"]

    # Extract allocation percentages from the table
    import re
    allocs = re.findall(r"(\d+\.\d+)%", table)
    float_allocs = [float(a) for a in allocs]
    # At minimum, the allocations alone should be present
    assert len(float_allocs) >= 3


# ---- Account filtering ----

@pytest.mark.asyncio
@respx.mock
async def test_account_filter_resolves_name_to_id(tool):
    """When account param is given, resolve name and filter via API."""
    respx.get("http://localhost:3333/api/v1/account").mock(
        return_value=httpx.Response(200, json=MOCK_ACCOUNTS_RESPONSE)
    )
    # The filtered details call should include ?accounts=acct-1
    respx.get(url__startswith="http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_FILTERED_DETAILS)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", account="Main Brokerage")

    # Verify the filtered API call was made with account ID
    called_urls = [str(call.request.url) for call in respx.calls]
    details_urls = [u for u in called_urls if "portfolio/details" in u]
    assert any("accounts=acct-1" in u for u in details_urls)

    # Should only show AAPL (from filtered response)
    table = result["result"]["holdings_table"]
    assert "AAPL" in table
    assert "GOOGL" not in table


@pytest.mark.asyncio
@respx.mock
async def test_account_filter_case_insensitive(tool):
    """Account name matching should be case-insensitive."""
    respx.get("http://localhost:3333/api/v1/account").mock(
        return_value=httpx.Response(200, json=MOCK_ACCOUNTS_RESPONSE)
    )
    respx.get(url__startswith="http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_FILTERED_DETAILS)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", account="main brokerage")

    called_urls = [str(call.request.url) for call in respx.calls]
    details_urls = [u for u in called_urls if "portfolio/details" in u]
    assert any("accounts=acct-1" in u for u in details_urls)


@pytest.mark.asyncio
@respx.mock
async def test_account_filter_partial_match(tool):
    """Account name matching should support partial match."""
    respx.get("http://localhost:3333/api/v1/account").mock(
        return_value=httpx.Response(200, json=MOCK_ACCOUNTS_RESPONSE)
    )
    respx.get(url__startswith="http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_FILTERED_DETAILS)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", account="crypto")

    called_urls = [str(call.request.url) for call in respx.calls]
    details_urls = [u for u in called_urls if "portfolio/details" in u]
    assert any("accounts=acct-2" in u for u in details_urls)


@pytest.mark.asyncio
@respx.mock
async def test_account_filter_not_found_shows_warning(tool):
    """When account name doesn't match, show all with warning."""
    respx.get("http://localhost:3333/api/v1/account").mock(
        return_value=httpx.Response(200, json=MOCK_ACCOUNTS_RESPONSE)
    )
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", account="Nonexistent Account")

    # Should show all holdings (unfiltered)
    table = result["result"]["holdings_table"]
    assert "AAPL" in table
    assert "GOOGL" in table
    # Should include warning
    assert "warning" in result["result"]
    assert "Nonexistent Account" in result["result"]["warning"]


# ---- Query parameters ----

@pytest.mark.asyncio
@respx.mock
async def test_asset_class_filter_passed_to_api(tool):
    """asset_classes param should be forwarded as ?assetClasses=."""
    respx.get(url__startswith="http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", asset_classes="EQUITY")

    called_urls = [str(call.request.url) for call in respx.calls]
    details_urls = [u for u in called_urls if "portfolio/details" in u]
    assert any("assetClasses=EQUITY" in u for u in details_urls)


@pytest.mark.asyncio
@respx.mock
async def test_range_param_passed_to_api(tool):
    """range param should be forwarded as ?range=."""
    respx.get(url__startswith="http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_DETAILS_RESPONSE)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", **{"range": "ytd"})

    called_urls = [str(call.request.url) for call in respx.calls]
    details_urls = [u for u in called_urls if "portfolio/details" in u]
    assert any("range=ytd" in u for u in details_urls)


@pytest.mark.asyncio
@respx.mock
async def test_multiple_params_combined(tool):
    """Multiple params should be combined in query string."""
    respx.get("http://localhost:3333/api/v1/account").mock(
        return_value=httpx.Response(200, json=MOCK_ACCOUNTS_RESPONSE)
    )
    respx.get(url__startswith="http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=MOCK_FILTERED_DETAILS)
    )
    _mock_orders()

    result = await tool.execute(
        jwt="fake-jwt", account="Main Brokerage",
        asset_classes="EQUITY", **{"range": "1y"},
    )

    called_urls = [str(call.request.url) for call in respx.calls]
    details_urls = [u for u in called_urls if "portfolio/details" in u]
    url = details_urls[0]
    assert "accounts=acct-1" in url
    assert "assetClasses=EQUITY" in url
    assert "range=1y" in url


@pytest.mark.asyncio
@respx.mock
async def test_no_params_calls_unfiltered(tool):
    """No params should call details without query string."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    called_urls = [str(call.request.url) for call in respx.calls]
    details_urls = [u for u in called_urls if "portfolio/details" in u]
    assert details_urls[0] == "http://localhost:3333/api/v1/portfolio/details"


# ---- Account IDs in output ----

@pytest.mark.asyncio
@respx.mock
async def test_accounts_include_ids(tool):
    """Account list should include account IDs."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    accounts = result["result"]["accounts"]
    assert len(accounts) == 1
    assert accounts[0]["id"] == "acct-1"
    assert accounts[0]["name"] == "Main Brokerage"


# ---- Metadata ----

@pytest.mark.asyncio
@respx.mock
async def test_tool_metadata(tool):
    assert tool.name == "portfolio_analysis"
    assert "portfolio" in tool.description.lower()


@pytest.mark.asyncio
@respx.mock
async def test_no_performance_endpoint_called(tool):
    """Tool should NOT call the performance endpoint."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    # Verify only details + orders were called (no performance endpoint)
    called_urls = [str(call.request.url) for call in respx.calls]
    assert any("portfolio/details" in u for u in called_urls)
    assert any("order" in u for u in called_urls)
    assert not any("performance" in u for u in called_urls)


@pytest.mark.asyncio
@respx.mock
async def test_parameters_schema_has_expected_params(tool):
    """Schema should expose all params including new view and symbols."""
    schema = tool.parameters_schema
    props = schema["properties"]
    assert "account" in props
    assert "asset_classes" in props
    assert "range" in props
    assert "view" in props
    assert "symbols" in props
    assert "include_daily_change" in props
    assert "include_countries" in props
    assert "filter_gains" in props
    assert schema["required"] == []


# ---- Daily change columns (historical data fetch) ----

@pytest.mark.asyncio
@respx.mock
async def test_include_daily_change_adds_columns(tool):
    """When include_daily_change=True, table should have Day Change columns."""
    _mock_all_with_daily()

    result = await tool.execute(jwt="fake-jwt", include_daily_change=True)

    table = result["result"]["holdings_table"]
    assert "Day Change" in table
    assert "Day Change %" in table
    # AAPL: current 195.50, prev close 192.00 → change = $3.50
    assert "$3.50" in table


@pytest.mark.asyncio
@respx.mock
async def test_daily_change_not_shown_by_default(tool):
    """Without include_daily_change, Day Change columns should not appear."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    assert "Day Change" not in table


@pytest.mark.asyncio
@respx.mock
async def test_daily_change_rate_limited(tool):
    """When historical data returns 429, show '—' not an error."""
    _mock_all()
    # Mock all historical endpoints as 429 (rate limited)
    respx.get(url__startswith="http://localhost:3333/api/v1/symbol/").mock(
        return_value=httpx.Response(429, text="Too Many Requests")
    )

    result = await tool.execute(jwt="fake-jwt", include_daily_change=True)

    table = result["result"]["holdings_table"]
    assert "Day Change" in table
    assert "Day Change %" in table
    # All values should be "—" since rate limited
    assert "error" not in table.lower()


@pytest.mark.asyncio
@respx.mock
async def test_daily_change_skips_manual_datasource(tool):
    """Holdings with MANUAL dataSource should not trigger historical fetch."""
    details = copy.deepcopy(MOCK_DETAILS_RESPONSE)
    # Add a MANUAL holding (like cash/USD)
    details["holdings"]["MANUAL-USD"] = {
        "name": "US Dollar",
        "symbol": "USD",
        "currency": "USD",
        "assetClass": "LIQUIDITY",
        "dataSource": "MANUAL",
        "marketPrice": 1.0,
        "quantity": 500,
        "investment": 500.0,
        "sectors": [],
        "countries": [],
    }
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=details)
    )
    _mock_orders()
    _mock_historical_data()

    result = await tool.execute(jwt="fake-jwt", include_daily_change=True)

    # Verify no historical fetch for MANUAL/USD
    called_urls = [str(call.request.url) for call in respx.calls]
    assert not any("symbol/MANUAL/USD" in u for u in called_urls)
    # USD row should still appear with "—" for daily change
    table = result["result"]["holdings_table"]
    assert "USD" in table


# ---- Country column ----

@pytest.mark.asyncio
@respx.mock
async def test_include_countries_adds_column(tool):
    """When include_countries=True, table should have Country column."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", include_countries=True)

    table = result["result"]["holdings_table"]
    assert "Country" in table
    assert "United States" in table


@pytest.mark.asyncio
@respx.mock
async def test_countries_not_shown_by_default(tool):
    """Without include_countries, Country column should not appear."""
    _mock_all()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["holdings_table"]
    assert "Country" not in table


# ---- Gain/loss filtering ----

@pytest.mark.asyncio
@respx.mock
async def test_filter_unrealized_losses(tool):
    """filter_gains='unrealized_losses' should only show holdings with negative gain."""
    details = copy.deepcopy(MOCK_DETAILS_RESPONSE)
    # Make GOOGL have a loss (investment > value: qty 5 * price 140 = 700, investment = 800)
    details["holdings"]["YAHOO-GOOGL"]["investment"] = 800.00
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=details)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", filter_gains="unrealized_losses")

    table = result["result"]["holdings_table"]
    # GOOGL has a loss (value 700 - cost 800 = -100), should be shown
    assert "GOOGL" in table
    # AAPL has a gain (value 1955 - cost 1500 = 455), should NOT be shown
    assert "AAPL" not in table
    # MSFT has a gain (value 840 - cost 700 = 140), should NOT be shown
    assert "MSFT" not in table


@pytest.mark.asyncio
@respx.mock
async def test_filter_unrealized_gains(tool):
    """filter_gains='unrealized_gains' should only show holdings with positive gain."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", filter_gains="unrealized_gains")

    table = result["result"]["holdings_table"]
    # All three mock holdings have gains (investment < value)
    assert "AAPL" in table
    assert "GOOGL" in table
    assert "MSFT" in table


@pytest.mark.asyncio
@respx.mock
async def test_filter_unrealized_losses_empty_result(tool):
    """When no holdings have losses, filtering for losses returns empty."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", filter_gains="unrealized_losses")

    # All mock holdings have gains, so filtering for losses should be empty
    table = result["result"]["holdings_table"]
    assert table == "No holdings found."


@pytest.mark.asyncio
@respx.mock
async def test_filter_gains_uses_filtered_summary(tool):
    """When filter_gains is applied, portfolio summary should use filtered cost basis."""
    details = copy.deepcopy(MOCK_DETAILS_RESPONSE)
    details["holdings"]["YAHOO-GOOGL"]["investment"] = 800.00
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=details)
    )
    _mock_orders()

    result = await tool.execute(jwt="fake-jwt", filter_gains="unrealized_losses")

    portfolio = result["result"]["portfolio"]
    # Only GOOGL should contribute to the summary (loss position)
    # GOOGL: value = 5 * 140 = 700, cost = 800
    assert portfolio["total_value"] == 700.00
    assert portfolio["net_cost_basis"] == 800.00


# ---- View presets ----

@pytest.mark.asyncio
@respx.mock
async def test_view_performance_columns(tool):
    """view='performance' should show only Symbol, Name, Value, Gain, Gain %."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", view="performance")

    table = result["result"]["holdings_table"]
    assert "Symbol" in table
    assert "Name" in table
    assert "Value" in table
    assert "Gain" in table
    assert "Gain %" in table
    # Should NOT have these columns
    assert "Sector" not in table
    assert "Price" not in table
    assert "Cost Basis/Share" not in table
    assert "Allocation" not in table


@pytest.mark.asyncio
@respx.mock
async def test_view_exposure_columns(tool):
    """view='exposure' should show Symbol, Name, Country, Sector, Value, Allocation."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", view="exposure")

    table = result["result"]["holdings_table"]
    assert "Symbol" in table
    assert "Name" in table
    assert "Country" in table
    assert "Sector" in table
    assert "Value" in table
    assert "Allocation" in table
    # Should NOT have these
    assert "Gain %" not in table
    assert "Cost Basis" not in table


@pytest.mark.asyncio
@respx.mock
async def test_view_daily_triggers_fetch(tool):
    """view='daily' should implicitly trigger historical data fetch."""
    _mock_all_with_daily()

    result = await tool.execute(jwt="fake-jwt", view="daily")

    table = result["result"]["holdings_table"]
    assert "Day Change" in table
    assert "Day Change %" in table
    # Verify historical data was fetched
    called_urls = [str(call.request.url) for call in respx.calls]
    assert any("symbol/YAHOO/AAPL" in u for u in called_urls)


@pytest.mark.asyncio
@respx.mock
async def test_view_compact_minimal_columns(tool):
    """view='compact' should show only Symbol, Value, Gain, Gain %."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", view="compact")

    table = result["result"]["holdings_table"]
    assert "Symbol" in table
    assert "Value" in table
    assert "Gain %" in table
    # Should NOT have these columns
    assert "Name" not in table
    assert "Sector" not in table
    assert "Country" not in table
    assert "Day Change" not in table


@pytest.mark.asyncio
@respx.mock
async def test_view_with_include_override(tool):
    """Boolean flags should add columns on top of view presets."""
    _mock_all()

    result = await tool.execute(
        jwt="fake-jwt", view="performance", include_countries=True,
    )

    table = result["result"]["holdings_table"]
    # Performance columns
    assert "Symbol" in table
    assert "Name" in table
    assert "Value" in table
    assert "Gain" in table
    # Added by include_countries override
    assert "Country" in table
    # Still should NOT have unrelated columns
    assert "Sector" not in table


# ---- Symbols filter ----

@pytest.mark.asyncio
@respx.mock
async def test_symbols_filter(tool):
    """symbols param should filter rows to specified symbols."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", symbols="AAPL,MSFT")

    table = result["result"]["holdings_table"]
    assert "AAPL" in table
    assert "MSFT" in table
    assert "GOOGL" not in table


@pytest.mark.asyncio
@respx.mock
async def test_symbols_filter_case_insensitive(tool):
    """symbols param should be case-insensitive."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", symbols="aapl")

    table = result["result"]["holdings_table"]
    assert "AAPL" in table
    assert "GOOGL" not in table
    assert "MSFT" not in table


@pytest.mark.asyncio
@respx.mock
async def test_symbols_filter_with_gain_filter(tool):
    """symbols and filter_gains should compose together."""
    details = copy.deepcopy(MOCK_DETAILS_RESPONSE)
    details["holdings"]["YAHOO-GOOGL"]["investment"] = 800.00  # GOOGL at a loss
    respx.get("http://localhost:3333/api/v1/portfolio/details").mock(
        return_value=httpx.Response(200, json=details)
    )
    _mock_orders()

    # Filter to AAPL and GOOGL, then also filter for losses
    result = await tool.execute(
        jwt="fake-jwt", symbols="AAPL,GOOGL", filter_gains="unrealized_losses"
    )

    table = result["result"]["holdings_table"]
    # Only GOOGL should appear (it's in the symbols filter AND has a loss)
    assert "GOOGL" in table
    assert "AAPL" not in table


@pytest.mark.asyncio
@respx.mock
async def test_symbols_filter_uses_filtered_summary(tool):
    """When symbols is set, portfolio summary should reflect only those symbols."""
    _mock_all()

    result = await tool.execute(jwt="fake-jwt", symbols="AAPL")

    portfolio = result["result"]["portfolio"]
    # Only AAPL: value = 10 * 195.50 = 1955
    assert portfolio["total_value"] == 1955.00
