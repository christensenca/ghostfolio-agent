"""Tests for tax_estimate tool."""
from __future__ import annotations

import pytest
import httpx
import respx

from ghostfolio_agent.tools.tax_estimate import TaxEstimateTool


# --- Mock data ---
# BUY AAPL: 10 @ $150 = $1500 (avg cost = $150/share)
# BUY AAPL: 10 @ $200 = $2000 (avg cost = (1500+2000)/20 = $175/share)
# SELL AAPL: 5 @ $195 = $975 (cost basis = 5 * $175 = $875, gain = $975 - $875 - $9.99 = $90.01)
# BUY BTC: 0.1 @ $50000 = $5000 (avg cost = $50000/BTC)
# SELL BTC: 0.05 @ $60000 = $3000 (cost basis = 0.05 * $50000 = $2500, gain = $3000 - $2500 - $5 = $495)
# DIVIDEND AAPL: 20 * $0.50 = $10
# DIVIDEND AAPL (next year): 20 * $0.60 = $12

MOCK_ORDERS_RESPONSE = {
    "activities": [
        {
            "date": "2024-01-15T00:00:00Z",
            "type": "BUY",
            "currency": "USD",
            "quantity": 10,
            "unitPrice": 150.00,
            "valueInBaseCurrency": 1500.00,
            "feeInBaseCurrency": 9.99,
            "account": {"name": "Brokerage"},
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
        },
        {
            "date": "2024-03-01T00:00:00Z",
            "type": "BUY",
            "currency": "USD",
            "quantity": 10,
            "unitPrice": 200.00,
            "valueInBaseCurrency": 2000.00,
            "feeInBaseCurrency": 9.99,
            "account": {"name": "Brokerage"},
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
        },
        {
            "date": "2024-06-15T00:00:00Z",
            "type": "SELL",
            "currency": "USD",
            "quantity": 5,
            "unitPrice": 195.00,
            "valueInBaseCurrency": 975.00,
            "feeInBaseCurrency": 9.99,
            "account": {"name": "Brokerage"},
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
        },
        {
            "date": "2024-02-01T00:00:00Z",
            "type": "BUY",
            "currency": "USD",
            "quantity": 0.1,
            "unitPrice": 50000.00,
            "valueInBaseCurrency": 5000.00,
            "feeInBaseCurrency": 2.00,
            "account": {"name": "Crypto"},
            "SymbolProfile": {"symbol": "bitcoin", "name": "Bitcoin"},
        },
        {
            "date": "2024-08-01T00:00:00Z",
            "type": "SELL",
            "currency": "USD",
            "quantity": 0.05,
            "unitPrice": 60000.00,
            "valueInBaseCurrency": 3000.00,
            "feeInBaseCurrency": 5.00,
            "account": {"name": "Crypto"},
            "SymbolProfile": {"symbol": "bitcoin", "name": "Bitcoin"},
        },
        {
            "date": "2024-07-15T00:00:00Z",
            "type": "DIVIDEND",
            "currency": "USD",
            "quantity": 20,
            "unitPrice": 0.50,
            "valueInBaseCurrency": 10.00,
            "feeInBaseCurrency": 0,
            "account": {"name": "Brokerage"},
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
        },
        {
            "date": "2025-01-15T00:00:00Z",
            "type": "DIVIDEND",
            "currency": "USD",
            "quantity": 20,
            "unitPrice": 0.60,
            "valueInBaseCurrency": 12.00,
            "feeInBaseCurrency": 0,
            "account": {"name": "Brokerage"},
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
        },
    ],
}


def _mock_orders():
    """Set up respx mock for the order endpoint."""
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=MOCK_ORDERS_RESPONSE)
    )


@pytest.fixture
def tool():
    return TaxEstimateTool()


# --- Basic output structure ---


@pytest.mark.asyncio
@respx.mock
async def test_returns_tax_estimate(tool):
    """Tool should return structured tax estimate."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    assert result["tool_name"] == "tax_estimate"
    assert "summary" in result["result"]
    assert "by_year" in result["result"]
    assert "by_symbol" in result["result"]
    assert "sells_table" in result["result"]
    assert "dividends_table" in result["result"]
    assert "disclaimer" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_includes_disclaimer(tool):
    """Tool MUST include a tax disclaimer."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    assert "not tax advice" in result["result"]["disclaimer"].lower()


# --- Capital gains (average cost basis) ---


@pytest.mark.asyncio
@respx.mock
async def test_aapl_gain_uses_avg_cost(tool):
    """AAPL gain should use average cost of $175/share."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    by_sym = result["result"]["by_symbol"]
    # avg cost = (1500 + 2000) / 20 = 175
    # cost basis = 5 * 175 = 875
    # gain = 975 - 875 - 9.99 = 90.01
    assert "AAPL" in by_sym
    assert by_sym["AAPL"]["realized_gains"] == 90.01


@pytest.mark.asyncio
@respx.mock
async def test_btc_gain_uses_avg_cost(tool):
    """Bitcoin gain should use average cost of $50000/BTC."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    by_sym = result["result"]["by_symbol"]
    # avg cost = 5000 / 0.1 = 50000
    # cost basis = 0.05 * 50000 = 2500
    # gain = 3000 - 2500 - 5 = 495
    assert "bitcoin" in by_sym
    assert by_sym["bitcoin"]["realized_gains"] == 495.0


@pytest.mark.asyncio
@respx.mock
async def test_total_realized_gains(tool):
    """Total gains should be sum of all symbol gains."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    summary = result["result"]["summary"]
    # AAPL: 90.01 + BTC: 495.0 = 585.01
    assert summary["total_realized_gains"] == 585.01
    assert summary["sell_count"] == 2


# --- Dividends ---


@pytest.mark.asyncio
@respx.mock
async def test_total_dividends(tool):
    """Total dividends should sum all dividend payments."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    summary = result["result"]["summary"]
    # 10 + 12 = 22
    assert summary["total_dividends"] == 22.0
    assert summary["dividend_count"] == 2


@pytest.mark.asyncio
@respx.mock
async def test_dividend_by_symbol(tool):
    """by_symbol should include dividend income."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    aapl = result["result"]["by_symbol"]["AAPL"]
    assert aapl["dividend_income"] == 22.0
    assert aapl["dividend_count"] == 2


# --- Fees ---


@pytest.mark.asyncio
@respx.mock
async def test_total_fees(tool):
    """Total fees should include all transaction fees."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    # 9.99 + 9.99 + 9.99 + 2.00 + 5.00 + 0 + 0 = 36.97
    assert result["result"]["summary"]["total_fees"] == 36.97


# --- Per-year breakdown ---


@pytest.mark.asyncio
@respx.mock
async def test_by_year_breakdown(tool):
    """by_year should have entries for each year with taxable events."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    by_year = result["result"]["by_year"]
    years = {e["year"] for e in by_year}
    assert 2024 in years
    assert 2025 in years


@pytest.mark.asyncio
@respx.mock
async def test_by_year_2024_values(tool):
    """2024 should have both sells and one dividend."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    yr_2024 = next(e for e in result["result"]["by_year"] if e["year"] == 2024)
    assert yr_2024["realized_gains"] == 585.01
    assert yr_2024["dividend_income"] == 10.0
    assert yr_2024["sell_count"] == 2


@pytest.mark.asyncio
@respx.mock
async def test_by_year_2025_values(tool):
    """2025 should have only dividend, no sells."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    yr_2025 = next(e for e in result["result"]["by_year"] if e["year"] == 2025)
    assert yr_2025["realized_gains"] == 0
    assert yr_2025["dividend_income"] == 12.0
    assert yr_2025["sell_count"] == 0


@pytest.mark.asyncio
@respx.mock
async def test_by_year_sorted_descending(tool):
    """by_year should be sorted newest first."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    years = [e["year"] for e in result["result"]["by_year"]]
    assert years == sorted(years, reverse=True)


# --- Markdown tables ---


@pytest.mark.asyncio
@respx.mock
async def test_sells_table_is_markdown(tool):
    """Sells table should be markdown with expected columns."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["sells_table"]
    assert "|" in table
    for col in ("Date", "Symbol", "Proceeds", "Avg Cost", "Cost Basis", "Gain/Loss", "Fee"):
        assert col in table


@pytest.mark.asyncio
@respx.mock
async def test_sells_table_has_dollar_signs(tool):
    """Sells table should have dollar formatting."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    assert "$" in result["result"]["sells_table"]


@pytest.mark.asyncio
@respx.mock
async def test_dividends_table_is_markdown(tool):
    """Dividends table should be markdown with expected columns."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["dividends_table"]
    assert "|" in table
    for col in ("Date", "Symbol", "Per Share", "Amount"):
        assert col in table


@pytest.mark.asyncio
@respx.mock
async def test_dividends_table_contains_all_dividends(tool):
    """Dividends table should contain all dividend payments."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["dividends_table"]
    assert "AAPL" in table
    # Both dates should appear
    assert "2024-07-15" in table
    assert "2025-01-15" in table


# --- Filters ---


@pytest.mark.asyncio
@respx.mock
async def test_account_filter(tool):
    """Account filter should restrict to matching account."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt", account="Crypto")

    summary = result["result"]["summary"]
    # Only BTC sell: gain = 495
    assert summary["total_realized_gains"] == 495.0
    assert summary["sell_count"] == 1
    assert summary["dividend_count"] == 0


@pytest.mark.asyncio
@respx.mock
async def test_account_filter_case_insensitive(tool):
    """Account filter should be case-insensitive."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt", account="crypto")

    assert result["result"]["summary"]["sell_count"] == 1


@pytest.mark.asyncio
@respx.mock
async def test_symbol_filter(tool):
    """Symbol filter should restrict to matching symbol."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt", symbol="AAPL")

    summary = result["result"]["summary"]
    assert summary["total_realized_gains"] == 90.01
    assert summary["sell_count"] == 1
    assert summary["dividend_count"] == 2


@pytest.mark.asyncio
@respx.mock
async def test_year_filter(tool):
    """Year filter should restrict to that year."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt", year=2024)

    summary = result["result"]["summary"]
    assert summary["sell_count"] == 2
    assert summary["dividend_count"] == 1  # Only 2024 dividend


@pytest.mark.asyncio
@respx.mock
async def test_year_filter_no_match_shows_warning(tool):
    """Non-matching year filter should show warning and return all."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt", year=2020)

    assert "warning" in result["result"]
    assert result["result"]["summary"]["sell_count"] == 2


@pytest.mark.asyncio
@respx.mock
async def test_filters_applied_echoed(tool):
    """filters_applied should echo back active filters."""
    _mock_orders()
    result = await tool.execute(jwt="fake-jwt", account="Crypto", year=2024)

    fa = result["result"]["filters_applied"]
    assert fa["account"] == "Crypto"
    assert fa["year"] == 2024
    assert fa["symbol"] is None


# --- Error handling ---


@pytest.mark.asyncio
@respx.mock
async def test_returns_error_on_401(tool):
    """Tool should return error dict on unauthorized."""
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(401, json={"message": "Unauthorized"})
    )
    result = await tool.execute(jwt="bad-jwt")

    assert "error" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_handles_no_transactions(tool):
    """Tool should handle empty activities."""
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": []})
    )
    result = await tool.execute(jwt="fake-jwt")

    assert result["result"]["summary"]["total_realized_gains"] == 0
    assert result["result"]["summary"]["sell_count"] == 0


@pytest.mark.asyncio
@respx.mock
async def test_handles_no_sells(tool):
    """Tool should handle only buys and dividends (no sells)."""
    no_sells = {
        "activities": [
            {
                "date": "2024-01-15T00:00:00Z",
                "type": "BUY",
                "currency": "USD",
                "quantity": 10,
                "unitPrice": 150,
                "valueInBaseCurrency": 1500,
                "feeInBaseCurrency": 5,
                "account": {"name": "Brokerage"},
                "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
            },
            {
                "date": "2024-07-15T00:00:00Z",
                "type": "DIVIDEND",
                "currency": "USD",
                "quantity": 10,
                "unitPrice": 0.50,
                "valueInBaseCurrency": 5,
                "feeInBaseCurrency": 0,
                "account": {"name": "Brokerage"},
                "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
            },
        ]
    }
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=no_sells)
    )
    result = await tool.execute(jwt="fake-jwt")

    assert result["result"]["summary"]["total_realized_gains"] == 0
    assert result["result"]["summary"]["sell_count"] == 0
    assert result["result"]["summary"]["total_dividends"] == 5.0
    assert "No sell transactions" in result["result"]["sells_table"]


@pytest.mark.asyncio
@respx.mock
async def test_sells_with_no_buys_zero_cost(tool):
    """Sells for symbols with no buys should have zero cost basis."""
    sells_only = {
        "activities": [
            {
                "date": "2024-06-15T00:00:00Z",
                "type": "SELL",
                "currency": "USD",
                "quantity": 5,
                "unitPrice": 100,
                "valueInBaseCurrency": 500,
                "feeInBaseCurrency": 2,
                "account": {"name": "Brokerage"},
                "SymbolProfile": {"symbol": "XYZ", "name": "XYZ Corp"},
            },
        ]
    }
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=sells_only)
    )
    result = await tool.execute(jwt="fake-jwt")

    # gain = 500 - 0 - 2 = 498
    assert result["result"]["summary"]["total_realized_gains"] == 498.0


# --- Edge cases ---


@pytest.mark.asyncio
@respx.mock
async def test_handles_missing_account(tool):
    """Activities without account should use N/A."""
    no_account = {
        "activities": [
            {
                "date": "2024-06-15T00:00:00Z",
                "type": "SELL",
                "currency": "USD",
                "quantity": 1,
                "unitPrice": 100,
                "valueInBaseCurrency": 100,
                "feeInBaseCurrency": 0,
                "SymbolProfile": {"symbol": "TEST", "name": "Test"},
            }
        ]
    }
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=no_account)
    )
    result = await tool.execute(jwt="fake-jwt")

    assert "N/A" in result["result"]["sells_table"]


@pytest.mark.asyncio
@respx.mock
async def test_fetches_all_transactions_range_max(tool):
    """Tool should always fetch range=max for accurate cost basis."""
    route = respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=MOCK_ORDERS_RESPONSE)
    )
    await tool.execute(jwt="fake-jwt")

    assert route.called
    assert "range=max" in str(route.calls[0].request.url)


# --- Metadata ---


@pytest.mark.asyncio
async def test_tool_metadata(tool):
    """Tool should have correct name and description."""
    assert tool.name == "tax_estimate"
    assert "tax" in tool.description.lower()


@pytest.mark.asyncio
async def test_parameters_schema(tool):
    """Schema should include filter parameters."""
    props = tool.parameters_schema["properties"]
    assert "account" in props
    assert "symbol" in props
    assert "year" in props
