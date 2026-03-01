"""Tests for transaction_categorize tool."""
from __future__ import annotations

import pytest
import httpx
import respx

from ghostfolio_agent.tools.transaction_categorize import TransactionCategorizeTool


MOCK_ORDERS_RESPONSE = {
    "count": 5,
    "activities": [
        {
            "id": "order-1",
            "date": "2024-06-15T00:00:00Z",
            "type": "BUY",
            "currency": "USD",
            "quantity": 10,
            "unitPrice": 150.25,
            "fee": 9.99,
            "value": 1502.50,
            "valueInBaseCurrency": 1502.50,
            "feeInBaseCurrency": 9.99,
            "comment": "Weekly DCA",
            "account": {"id": "acct-1", "name": "Main Brokerage"},
            "SymbolProfile": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "assetClass": "EQUITY",
                "currency": "USD",
            },
        },
        {
            "id": "order-2",
            "date": "2024-07-01T00:00:00Z",
            "type": "SELL",
            "currency": "USD",
            "quantity": 5,
            "unitPrice": 160.00,
            "fee": 9.99,
            "value": 800.00,
            "valueInBaseCurrency": 800.00,
            "feeInBaseCurrency": 9.99,
            "comment": "Taking profits",
            "account": {"id": "acct-1", "name": "Main Brokerage"},
            "SymbolProfile": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "assetClass": "EQUITY",
                "currency": "USD",
            },
        },
        {
            "id": "order-3",
            "date": "2024-07-15T00:00:00Z",
            "type": "DIVIDEND",
            "currency": "USD",
            "quantity": 10,
            "unitPrice": 0.96,
            "fee": 0,
            "value": 9.60,
            "valueInBaseCurrency": 9.60,
            "feeInBaseCurrency": 0,
            "comment": "Quarterly dividend",
            "account": {"id": "acct-1", "name": "Main Brokerage"},
            "SymbolProfile": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "assetClass": "EQUITY",
                "currency": "USD",
            },
        },
        {
            "id": "order-4",
            "date": "2024-08-01T00:00:00Z",
            "type": "BUY",
            "currency": "USD",
            "quantity": 3,
            "unitPrice": 420.00,
            "fee": 4.99,
            "value": 1260.00,
            "valueInBaseCurrency": 1260.00,
            "feeInBaseCurrency": 4.99,
            "comment": "",
            "account": {"id": "acct-1", "name": "Main Brokerage"},
            "SymbolProfile": {
                "symbol": "MSFT",
                "name": "Microsoft Corp.",
                "assetClass": "EQUITY",
                "currency": "USD",
            },
        },
        {
            "id": "order-5",
            "date": "2024-08-15T00:00:00Z",
            "type": "BUY",
            "currency": "EUR",
            "quantity": 50,
            "unitPrice": 45.00,
            "fee": 5.00,
            "value": 2250.00,
            "valueInBaseCurrency": 2430.00,
            "feeInBaseCurrency": 5.40,
            "comment": "ETF purchase",
            "account": {"id": "acct-2", "name": "Crypto Portfolio"},
            "SymbolProfile": {
                "symbol": "VWCE.DE",
                "name": "Vanguard FTSE All-World",
                "assetClass": "EQUITY",
                "currency": "EUR",
            },
        },
    ],
}


def _mock_both():
    """Set up respx mock for the order endpoint."""
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=MOCK_ORDERS_RESPONSE)
    )


@pytest.fixture
def tool():
    return TransactionCategorizeTool()


# --- Basic output structure ---


@pytest.mark.asyncio
@respx.mock
async def test_returns_categorized_transactions(tool):
    """Tool should categorize transactions by type."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    assert result["tool_name"] == "transaction_categorize"
    by_type = result["result"]["by_type"]
    assert "BUY" in by_type
    assert "SELL" in by_type
    assert "DIVIDEND" in by_type


@pytest.mark.asyncio
@respx.mock
async def test_includes_summary_stats(tool):
    """Tool should include summary statistics."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    summary = result["result"]["summary"]
    assert summary["total_count"] == 5
    assert summary["total_fees"] > 0
    assert summary["total_value"] > 0
    assert "earliest" in summary["date_range"]
    assert "latest" in summary["date_range"]


@pytest.mark.asyncio
@respx.mock
async def test_by_type_counts(tool):
    """by_type should have correct counts and values."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    by_type = result["result"]["by_type"]
    assert by_type["BUY"]["count"] == 3
    assert by_type["SELL"]["count"] == 1
    assert by_type["DIVIDEND"]["count"] == 1
    assert by_type["BUY"]["total_fees"] > 0


@pytest.mark.asyncio
@respx.mock
async def test_by_symbol_summary(tool):
    """by_symbol should group transactions per symbol."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    by_symbol = result["result"]["by_symbol"]
    assert "AAPL" in by_symbol
    assert "MSFT" in by_symbol
    assert by_symbol["AAPL"]["count"] == 3  # BUY + SELL + DIVIDEND
    assert sorted(by_symbol["AAPL"]["types"]) == ["BUY", "DIVIDEND", "SELL"]


@pytest.mark.asyncio
@respx.mock
async def test_by_account_summary(tool):
    """by_account should group transactions per account."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    by_account = result["result"]["by_account"]
    assert "Main Brokerage" in by_account
    assert "Crypto Portfolio" in by_account
    assert by_account["Main Brokerage"]["count"] == 4


@pytest.mark.asyncio
@respx.mock
async def test_monthly_breakdown(tool):
    """monthly should have entries sorted descending."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    monthly = result["result"]["monthly"]
    assert len(monthly) >= 2
    # Sorted descending
    months = [m["month"] for m in monthly]
    assert months == sorted(months, reverse=True)
    # Each entry has count
    for entry in monthly:
        assert "month" in entry
        assert "count" in entry


# --- Markdown table ---


@pytest.mark.asyncio
@respx.mock
async def test_transactions_table_is_markdown(tool):
    """Result should contain a markdown table."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    assert "|" in table
    assert "Date" in table


@pytest.mark.asyncio
@respx.mock
async def test_table_has_all_columns(tool):
    """Markdown table should have all expected columns."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    for col in ("Date", "Type", "Symbol", "Name", "Account", "Qty", "Unit Price", "Value", "Fee"):
        assert col in table


@pytest.mark.asyncio
@respx.mock
async def test_all_transactions_in_table(tool):
    """All transactions should appear in the table (no 10-per-type cap)."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    assert "AAPL" in table
    assert "MSFT" in table
    assert "VWCE.DE" in table


@pytest.mark.asyncio
@respx.mock
async def test_table_sorted_date_descending(tool):
    """Table rows should be sorted by date descending."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    lines = [l for l in table.split("\n") if l.strip() and not l.startswith("|:")]
    # Skip header and separator, get data rows
    data_rows = lines[2:]
    dates = []
    for row in data_rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if cells:
            dates.append(cells[0])
    assert dates == sorted(dates, reverse=True)


@pytest.mark.asyncio
@respx.mock
async def test_dollar_formatting_in_table(tool):
    """Dollar values should have $ formatting."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    assert "$" in table


@pytest.mark.asyncio
@respx.mock
async def test_symbols_in_table(tool):
    """Symbols should appear in the markdown table."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    assert "AAPL" in table
    assert "MSFT" in table


# --- Filters ---


@pytest.mark.asyncio
@respx.mock
async def test_account_filter(tool):
    """Filtering by account should only include matching transactions."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", account="Crypto")

    table = result["result"]["transactions_table"]
    assert "VWCE.DE" in table
    assert "AAPL" not in table
    assert result["result"]["summary"]["total_count"] == 1


@pytest.mark.asyncio
@respx.mock
async def test_account_filter_case_insensitive(tool):
    """Account filter should be case-insensitive."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", account="crypto portfolio")

    assert result["result"]["summary"]["total_count"] == 1


@pytest.mark.asyncio
@respx.mock
async def test_account_filter_partial_match(tool):
    """Account filter should do partial matching."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", account="Main")

    assert result["result"]["summary"]["total_count"] == 4


@pytest.mark.asyncio
@respx.mock
async def test_account_filter_not_found_shows_warning(tool):
    """Non-matching account filter should show warning and return all."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", account="Nonexistent")

    assert result["result"]["summary"]["total_count"] == 5
    assert "warning" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_symbol_filter(tool):
    """Filtering by symbol should only include that symbol."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", symbol="AAPL")

    assert result["result"]["summary"]["total_count"] == 3
    table = result["result"]["transactions_table"]
    assert "MSFT" not in table


@pytest.mark.asyncio
@respx.mock
async def test_symbol_filter_case_insensitive(tool):
    """Symbol filter should be case-insensitive."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", symbol="aapl")

    assert result["result"]["summary"]["total_count"] == 3


@pytest.mark.asyncio
@respx.mock
async def test_type_filter(tool):
    """Filtering by type should only include that type."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", type="BUY")

    assert result["result"]["summary"]["total_count"] == 3
    assert "BUY" in result["result"]["by_type"]
    assert "SELL" not in result["result"]["by_type"]


@pytest.mark.asyncio
@respx.mock
async def test_type_filter_multiple(tool):
    """Filtering by multiple types should include all matching."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", type="BUY,DIVIDEND")

    assert result["result"]["summary"]["total_count"] == 4


@pytest.mark.asyncio
@respx.mock
async def test_filters_applied_in_output(tool):
    """filters_applied should echo back active filters."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", account="Main", symbol="AAPL")

    fa = result["result"]["filters_applied"]
    assert fa["account"] == "Main"
    assert fa["symbol"] == "AAPL"
    assert fa["type"] is None


# --- Client-side date range and asset class filters ---


@pytest.mark.asyncio
@respx.mock
async def test_range_always_fetches_max(tool):
    """Range filtering is client-side; API always gets range=max."""
    route = respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=MOCK_ORDERS_RESPONSE)
    )
    await tool.execute(jwt="fake-jwt", range="1y")

    assert route.called
    assert "range=max" in str(route.calls[0].request.url)


@pytest.mark.asyncio
@respx.mock
async def test_range_filters_by_date_client_side(tool):
    """Range=1m should only keep transactions from the last 30 days."""
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y-%m-%dT00:00:00Z")
    old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%dT00:00:00Z")

    mock_data = {
        "activities": [
            {
                "date": today,
                "type": "BUY",
                "quantity": 10,
                "unitPrice": 100,
                "feeInBaseCurrency": 1,
                "valueInBaseCurrency": 1000,
                "currency": "USD",
                "SymbolProfile": {"symbol": "AAPL", "name": "Apple", "assetClass": "EQUITY"},
                "account": {"name": "Main"},
            },
            {
                "date": old_date,
                "type": "BUY",
                "quantity": 5,
                "unitPrice": 200,
                "feeInBaseCurrency": 2,
                "valueInBaseCurrency": 1000,
                "currency": "USD",
                "SymbolProfile": {"symbol": "GOOGL", "name": "Google", "assetClass": "EQUITY"},
                "account": {"name": "Main"},
            },
        ]
    }
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=mock_data)
    )
    result = await tool.execute(jwt="fake-jwt", range="1m")

    assert result["result"]["summary"]["total_count"] == 1


@pytest.mark.asyncio
@respx.mock
async def test_asset_classes_filtered_client_side(tool):
    """Asset classes should be filtered client-side."""
    route = respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=MOCK_ORDERS_RESPONSE)
    )
    result = await tool.execute(jwt="fake-jwt", asset_classes="EQUITY")

    assert route.called
    # Should not pass assetClasses to API
    assert "assetClasses" not in str(route.calls[0].request.url)
    # All results should be EQUITY (our mock has EQUITY entries)
    assert result["result"]["summary"]["total_count"] > 0


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
async def test_handles_empty_transactions(tool):
    """Tool should handle no transactions gracefully."""
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json={"count": 0, "activities": []})
    )
    result = await tool.execute(jwt="fake-jwt")

    assert result["tool_name"] == "transaction_categorize"
    assert result["result"]["summary"]["total_count"] == 0
    assert result["result"]["by_type"] == {}
    assert "No transactions" in result["result"]["transactions_table"]


# --- Edge cases ---


@pytest.mark.asyncio
@respx.mock
async def test_handles_missing_account(tool):
    """Activities without account should show N/A."""
    no_account_response = {
        "activities": [
            {
                "date": "2024-06-15T00:00:00Z",
                "type": "BUY",
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
        return_value=httpx.Response(200, json=no_account_response)
    )
    result = await tool.execute(jwt="fake-jwt")

    assert "N/A" in result["result"]["transactions_table"]


@pytest.mark.asyncio
@respx.mock
async def test_handles_missing_comment(tool):
    """Activities without comment should not error."""
    no_comment_response = {
        "activities": [
            {
                "date": "2024-06-15T00:00:00Z",
                "type": "BUY",
                "currency": "USD",
                "quantity": 1,
                "unitPrice": 100,
                "valueInBaseCurrency": 100,
                "feeInBaseCurrency": 0,
                "account": {"name": "Test Account"},
                "SymbolProfile": {"symbol": "TEST", "name": "Test"},
            }
        ]
    }
    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json=no_comment_response)
    )
    result = await tool.execute(jwt="fake-jwt")

    assert result["result"]["summary"]["total_count"] == 1


# --- Metadata ---


@pytest.mark.asyncio
async def test_tool_metadata(tool):
    """Tool should have correct name and description."""
    assert tool.name == "transaction_categorize"
    assert "transaction" in tool.description.lower()


@pytest.mark.asyncio
async def test_parameters_schema_has_expected_params(tool):
    """Schema should include all filter parameters."""
    props = tool.parameters_schema["properties"]
    assert "account" in props
    assert "symbol" in props
    assert "type" in props
    assert "asset_classes" in props
    assert "range" in props
    assert "limit" in props
    assert "sort_by" in props
    assert "format" in props


# --- Limit ---


def _count_data_rows(table: str) -> int:
    """Count data rows in a markdown table (excludes header and separator)."""
    lines = table.strip().split("\n")
    # Data rows are lines starting with | that are NOT the separator (contains ---)
    return sum(
        1 for l in lines
        if l.strip().startswith("|") and "---" not in l and l != lines[0]
    )


@pytest.mark.asyncio
@respx.mock
async def test_limit_restricts_table_rows(tool):
    """Limit should restrict the number of rows in the table."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", limit=2)

    table = result["result"]["transactions_table"]
    assert _count_data_rows(table) == 2


@pytest.mark.asyncio
@respx.mock
async def test_limit_does_not_affect_summary(tool):
    """Limit should not affect summary stats (computed from full data)."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", limit=1)

    assert result["result"]["summary"]["total_count"] == 5


@pytest.mark.asyncio
@respx.mock
async def test_limit_larger_than_data(tool):
    """Limit larger than total rows should show all rows."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", limit=100)

    table = result["result"]["transactions_table"]
    assert "AAPL" in table
    assert "MSFT" in table
    assert "VWCE.DE" in table


# --- Sort ---


@pytest.mark.asyncio
@respx.mock
async def test_sort_by_value_descending(tool):
    """sort_by=value should sort by value descending."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", sort_by="value")

    table = result["result"]["transactions_table"]
    lines = [l for l in table.split("\n") if l.strip() and not l.startswith("|:")]
    data_rows = lines[2:]
    values = []
    for row in data_rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if cells:
            val_str = cells[7].replace("$", "").replace(",", "")  # Value column
            values.append(float(val_str))
    assert values == sorted(values, reverse=True)


@pytest.mark.asyncio
@respx.mock
async def test_sort_by_symbol_ascending(tool):
    """sort_by=symbol should sort alphabetically ascending."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", sort_by="symbol")

    table = result["result"]["transactions_table"]
    lines = [l for l in table.split("\n") if l.strip() and not l.startswith("|:")]
    data_rows = lines[2:]
    symbols = []
    for row in data_rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if cells:
            symbols.append(cells[2])  # Symbol column
    assert symbols == sorted(symbols)


@pytest.mark.asyncio
@respx.mock
async def test_sort_by_default_is_date(tool):
    """Default sort should be date descending."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    fa = result["result"]["filters_applied"]
    assert fa["sort_by"] == "date"


# --- Format ---


@pytest.mark.asyncio
@respx.mock
async def test_format_summary_omits_table(tool):
    """format=summary should not include transactions_table."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", format="summary")

    assert "transactions_table" not in result["result"]
    assert "summary" in result["result"]
    assert "by_type" in result["result"]
    assert "by_symbol" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_format_table_omits_breakdowns(tool):
    """format=table should include table and summary but not by_symbol etc."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", format="table")

    assert "transactions_table" in result["result"]
    assert "summary" in result["result"]
    assert "by_symbol" not in result["result"]
    assert "by_account" not in result["result"]
    assert "monthly" not in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_format_both_includes_everything(tool):
    """format=both (default) should include table and all summaries."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", format="both")

    assert "transactions_table" in result["result"]
    assert "summary" in result["result"]
    assert "by_type" in result["result"]
    assert "by_symbol" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_format_default_is_both(tool):
    """Default format should be 'both'."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    assert result["result"]["filters_applied"]["format"] == "both"
    assert "transactions_table" in result["result"]
    assert "by_type" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_filters_applied_includes_new_params(tool):
    """filters_applied should echo limit, sort_by, format."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt", limit=5, sort_by="value", format="table")

    fa = result["result"]["filters_applied"]
    assert fa["limit"] == 5
    assert fa["sort_by"] == "value"
    assert fa["format"] == "table"


# --- Auto-truncate ---


@pytest.mark.asyncio
@respx.mock
async def test_auto_truncate_large_table(tool):
    """Tables with >20 rows should auto-truncate with a footer."""
    # Build 25 activities
    activities = []
    for i in range(25):
        activities.append({
            "date": "2024-{m:02d}-{d:02d}T00:00:00Z".format(m=(i % 12) + 1, d=(i % 28) + 1),
            "type": "BUY",
            "currency": "USD",
            "quantity": 1,
            "unitPrice": 100 + i,
            "valueInBaseCurrency": 100 + i,
            "feeInBaseCurrency": 1,
            "account": {"name": "Test"},
            "SymbolProfile": {"symbol": "TEST{i}".format(i=i), "name": "Test {i}".format(i=i)},
        })

    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": activities})
    )
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    assert "Showing 20 of 25 transactions" in table
    assert "limit" in table.lower()


@pytest.mark.asyncio
@respx.mock
async def test_no_truncation_under_limit(tool):
    """Tables with <=20 rows should not show truncation footer."""
    _mock_both()
    result = await tool.execute(jwt="fake-jwt")

    table = result["result"]["transactions_table"]
    assert "Showing" not in table


@pytest.mark.asyncio
@respx.mock
async def test_explicit_limit_overrides_auto_truncate(tool):
    """Explicit limit should take priority over auto-truncate."""
    activities = []
    for i in range(25):
        activities.append({
            "date": "2024-{m:02d}-{d:02d}T00:00:00Z".format(m=(i % 12) + 1, d=(i % 28) + 1),
            "type": "BUY",
            "currency": "USD",
            "quantity": 1,
            "unitPrice": 100,
            "valueInBaseCurrency": 100,
            "feeInBaseCurrency": 0,
            "account": {"name": "Test"},
            "SymbolProfile": {"symbol": "T", "name": "Test"},
        })

    respx.get("http://localhost:3333/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": activities})
    )
    result = await tool.execute(jwt="fake-jwt", limit=5)

    table = result["result"]["transactions_table"]
    assert _count_data_rows(table) == 5
    # Summary should still reflect all 25
    assert result["result"]["summary"]["total_count"] == 25
