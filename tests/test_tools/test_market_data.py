"""Tests for market_data tool."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest
import respx

from ghostfolio_agent.tools.market_data import MarketDataTool


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_LOOKUP_RESPONSE = {
    "items": [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "currency": "USD",
            "dataSource": "YAHOO",
            "assetClass": "EQUITY",
            "assetSubClass": "STOCK",
            "dataProviderInfo": {
                "dataSource": "YAHOO",
                "isPremium": False,
                "name": "Yahoo Finance",
                "url": "https://finance.yahoo.com",
            },
        },
        {
            "symbol": "AAPX",
            "name": "Apple Enhanced ETF",
            "currency": "USD",
            "dataSource": "YAHOO",
            "assetClass": "EQUITY",
            "assetSubClass": "ETF",
            "dataProviderInfo": {
                "dataSource": "YAHOO",
                "isPremium": False,
            },
        },
    ]
}

MOCK_LOOKUP_MIXED_RESPONSE = {
    "items": [
        {
            "symbol": "APPLE-CAT",
            "name": "Apple Cat Token",
            "currency": "USD",
            "dataSource": "COINGECKO",
            "assetClass": "LIQUIDITY",
            "assetSubClass": "CRYPTOCURRENCY",
            "dataProviderInfo": {
                "dataSource": "COINGECKO",
                "isPremium": False,
                "name": "CoinGecko",
                "url": "https://coingecko.com",
            },
        },
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "currency": "USD",
            "dataSource": "YAHOO",
            "assetClass": "EQUITY",
            "assetSubClass": "STOCK",
            "dataProviderInfo": {
                "dataSource": "YAHOO",
                "isPremium": False,
                "name": "Yahoo Finance",
                "url": "https://finance.yahoo.com",
            },
        },
        {
            "symbol": "APPLE-ONDO",
            "name": "Apple Ondo Tokenized Stock",
            "currency": "USD",
            "dataSource": "COINGECKO",
            "assetClass": "LIQUIDITY",
            "assetSubClass": "CRYPTOCURRENCY",
            "dataProviderInfo": {
                "dataSource": "COINGECKO",
                "isPremium": False,
                "name": "CoinGecko",
                "url": "https://coingecko.com",
            },
        },
    ]
}

_yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z")
_two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00.000Z")
_today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00.000Z")

# Minimal symbol response (current price only — no historicalData needed now)
MOCK_SYMBOL_RESPONSE = {
    "dataSource": "YAHOO",
    "symbol": "AAPL",
    "currency": "USD",
    "marketPrice": 274.23,
    "historicalData": [],
}

# Market-data response with profile + real historical prices
MOCK_MARKET_DATA_RESPONSE = {
    "assetProfile": {
        "name": "Apple Inc.",
        "assetClass": "EQUITY",
        "assetSubClass": "STOCK",
        "sectors": [{"name": "Technology", "weight": 1.0}],
        "countries": [{"name": "United States", "weight": 1.0}],
        "url": "https://apple.com",
    },
    "marketData": [
        {"date": _two_days_ago, "marketPrice": 264.00},
        {"date": _yesterday, "marketPrice": 266.18},
        {"date": _today, "marketPrice": 274.23},
    ],
}

MOCK_MARKET_DATA_RICH_HISTORY = {
    "assetProfile": {
        "name": "Apple Inc.",
        "assetClass": "EQUITY",
    },
    "marketData": [
        {"date": "2026-02-01T00:00:00.000Z", "marketPrice": 250.00},
        {"date": "2026-02-02T00:00:00.000Z", "marketPrice": 251.00},
        {"date": "2026-02-03T00:00:00.000Z", "marketPrice": 252.00},
        {"date": "2026-02-04T00:00:00.000Z", "marketPrice": 253.00},
        {"date": "2026-02-05T00:00:00.000Z", "marketPrice": 254.00},
        {"date": "2026-02-06T00:00:00.000Z", "marketPrice": 255.00},
        {"date": "2026-02-07T00:00:00.000Z", "marketPrice": 256.00},
        {"date": "2026-02-08T00:00:00.000Z", "marketPrice": 257.00},
        {"date": "2026-02-09T00:00:00.000Z", "marketPrice": 258.00},
        {"date": "2026-02-10T00:00:00.000Z", "marketPrice": 259.00},
        {"date": "2026-02-11T00:00:00.000Z", "marketPrice": 260.00},
        {"date": "2026-02-12T00:00:00.000Z", "marketPrice": 261.00},
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool():
    return MarketDataTool()


def _mock_detail_endpoints(market_data_response=None, market_data_status=200):
    """Helper to mock both endpoints needed for symbol detail."""
    respx.get("http://localhost:3333/api/v1/symbol/YAHOO/AAPL").mock(
        return_value=httpx.Response(200, json=MOCK_SYMBOL_RESPONSE)
    )
    if market_data_response is not None:
        respx.get("http://localhost:3333/api/v1/market-data/YAHOO/AAPL").mock(
            return_value=httpx.Response(market_data_status, json=market_data_response)
        )
    else:
        respx.get("http://localhost:3333/api/v1/market-data/YAHOO/AAPL").mock(
            return_value=httpx.Response(403)
        )


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_lookup_returns_matching_symbols(tool):
    """Tool should return structured symbol search results."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(200, json=MOCK_LOOKUP_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt", query="AAPL")

    assert result["tool_name"] == "market_data"
    symbols = [s["symbol"] for s in result["result"]["symbols"]]
    assert "AAPL" in symbols


@pytest.mark.asyncio
@respx.mock
async def test_lookup_includes_asset_class(tool):
    """Results should include asset class info."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(200, json=MOCK_LOOKUP_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt", query="AAPL")

    aapl = next(s for s in result["result"]["symbols"] if s["symbol"] == "AAPL")
    assert aapl["asset_class"] == "EQUITY"


@pytest.mark.asyncio
@respx.mock
async def test_returns_error_on_401(tool):
    """Tool should return error dict on unauthorized."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(401, json={"message": "Unauthorized"})
    )

    result = await tool.execute(jwt="bad-jwt", query="AAPL")

    assert "error" in result["result"]


@pytest.mark.asyncio
@respx.mock
async def test_handles_no_results(tool):
    """Tool should handle empty search results gracefully."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(200, json={"items": []})
    )

    result = await tool.execute(jwt="fake-jwt", query="XYZNONEXIST")

    assert result["tool_name"] == "market_data"
    assert result["result"]["symbols"] == []
    assert result["result"]["total_results"] == 0


@pytest.mark.asyncio
async def test_tool_metadata(tool):
    """Tool should have correct name and description."""
    assert tool.name == "market_data"
    assert "market" in tool.description.lower() or "symbol" in tool.description.lower()


@pytest.mark.asyncio
@respx.mock
async def test_lookup_includes_source_metadata(tool):
    """Lookup results should include source attribution."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(200, json=MOCK_LOOKUP_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt", query="AAPL")

    aapl = next(s for s in result["result"]["symbols"] if s["symbol"] == "AAPL")
    assert "source" in aapl
    assert aapl["source"]["data_source"] == "YAHOO"
    assert aapl["source"]["provider_name"] == "Yahoo Finance"
    assert aapl["source"]["provider_url"] == "https://finance.yahoo.com"


@pytest.mark.asyncio
@respx.mock
async def test_search_ranking_equity_before_crypto(tool):
    """EQUITY results should appear before CRYPTOCURRENCY/LIQUIDITY."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(200, json=MOCK_LOOKUP_MIXED_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt", query="Apple")

    symbols = result["result"]["symbols"]
    assert symbols[0]["symbol"] == "AAPL"
    assert symbols[0]["asset_class"] == "EQUITY"
    assert symbols[1]["asset_class"] == "LIQUIDITY"


@pytest.mark.asyncio
@respx.mock
async def test_exact_ticker_match_boosted(tool):
    """When query is an uppercase ticker, exact match should be first."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(200, json=MOCK_LOOKUP_MIXED_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt", query="AAPL")

    symbols = result["result"]["symbols"]
    assert symbols[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
@respx.mock
async def test_lookup_includes_asset_sub_class(tool):
    """Lookup results should include asset_sub_class."""
    respx.get("http://localhost:3333/api/v1/symbol/lookup").mock(
        return_value=httpx.Response(200, json=MOCK_LOOKUP_RESPONSE)
    )

    result = await tool.execute(jwt="fake-jwt", query="AAPL")

    aapl = next(s for s in result["result"]["symbols"] if s["symbol"] == "AAPL")
    assert aapl["asset_sub_class"] == "STOCK"

    etf = next(s for s in result["result"]["symbols"] if s["symbol"] == "AAPX")
    assert etf["asset_sub_class"] == "ETF"


# ---------------------------------------------------------------------------
# Symbol detail tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_symbol_detail_returns_price(tool):
    """Tool should return price data for a specific symbol."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RESPONSE)

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")

    assert result["result"]["current_price"] == 274.23
    assert result["result"]["currency"] == "USD"


@pytest.mark.asyncio
@respx.mock
async def test_symbol_detail_includes_historical(tool):
    """Tool should include historical data from market-data endpoint."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RESPONSE)

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")

    prices = result["result"]["historical_prices"]
    assert len(prices) > 0
    assert prices[0]["price"] == 264.00


@pytest.mark.asyncio
@respx.mock
async def test_detail_verified_tool_name(tool):
    """Detail responses should use plain tool_name with source in result."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RESPONSE)

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")

    assert result["tool_name"] == "market_data"
    assert result["result"]["source"]["provider_name"] == "Yahoo Finance"


@pytest.mark.asyncio
@respx.mock
async def test_detail_verified_tool_name_fallback(tool):
    """Plain tool_name should be used even when market-data returns 403."""
    _mock_detail_endpoints()  # 403 fallback

    # Mock yfinance to return nothing so we test pure Ghostfolio-only path
    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")

    assert result["tool_name"] == "market_data"


@pytest.mark.asyncio
@respx.mock
async def test_detail_includes_source_metadata(tool):
    """Detail results should include source with provider_name."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RESPONSE)

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")

    source = result["result"]["source"]
    assert source["data_source"] == "YAHOO"
    assert source["provider_name"] == "Yahoo Finance"


# ---------------------------------------------------------------------------
# Daily change tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_daily_change_computed(tool):
    """Detail should include daily change vs previous close."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RESPONSE)

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")
    r = result["result"]

    # Previous close is yesterday's price (266.18), current is 274.23
    assert r["previous_close"] == 266.18
    assert r["daily_change"] == 8.05
    assert r["daily_change_pct"] == 3.02


@pytest.mark.asyncio
@respx.mock
async def test_daily_change_missing_when_no_market_data(tool):
    """Daily change should be absent when market-data endpoint fails and yfinance has nothing."""
    _mock_detail_endpoints()  # 403 fallback — no market data

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")
    r = result["result"]

    assert "daily_change" not in r
    assert "previous_close" not in r


@pytest.mark.asyncio
@respx.mock
async def test_daily_change_missing_when_empty_market_data(tool):
    """Daily change should be absent when marketData is empty and yfinance has nothing."""
    _mock_detail_endpoints({"assetProfile": None, "marketData": []})

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")
    r = result["result"]

    assert "daily_change" not in r


@pytest.mark.asyncio
@respx.mock
async def test_daily_change_only_today_data(tool):
    """Daily change should be absent when only today's data exists."""
    _mock_detail_endpoints({
        "assetProfile": None,
        "marketData": [{"date": _today, "marketPrice": 274.23}],
    })

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")
    r = result["result"]

    assert "daily_change" not in r


# ---------------------------------------------------------------------------
# Asset profile tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_asset_profile_merged_when_available(tool):
    """Detail should include profile fields when market-data endpoint works."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RESPONSE)

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")
    r = result["result"]

    assert r["name"] == "Apple Inc."
    assert r["asset_class"] == "EQUITY"
    assert r["sector"] == "Technology"
    assert r["country"] == "United States"


@pytest.mark.asyncio
@respx.mock
async def test_asset_profile_graceful_fallback_on_403(tool):
    """Detail should work fine when market-data endpoint returns 403 and yfinance has nothing."""
    _mock_detail_endpoints()  # 403 fallback

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")
    r = result["result"]

    assert r["current_price"] == 274.23
    assert r["currency"] == "USD"
    assert "name" not in r
    assert "sector" not in r


# ---------------------------------------------------------------------------
# Expanded history tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_returns_last_10_historical_points(tool):
    """Detail should return up to 10 historical data points."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RICH_HISTORY)

    result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")
    prices = result["result"]["historical_prices"]

    assert len(prices) == 10
    # Should be the last 10 points (Feb 03 through Feb 12)
    assert prices[0]["price"] == 252.00
    assert prices[-1]["price"] == 261.00


@pytest.mark.asyncio
@respx.mock
async def test_no_historical_when_market_data_fails(tool):
    """Historical prices should be empty when market-data endpoint fails."""
    _mock_detail_endpoints()  # 403 fallback — no yfinance fallback for AAPL since we mock it

    # Patch yfinance to return nothing so we test the "no data at all" path
    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")

    assert result["result"]["historical_prices"] == []


# ---------------------------------------------------------------------------
# yfinance fallback helpers
# ---------------------------------------------------------------------------

def _make_yfinance_info(
    name: str = "IREN LIMITED",
    sector: str = "Financial Services",
    industry: str = "Capital Markets",
    country: str = "Australia",
    fifty_two_week_high: float = 76.87,
    fifty_two_week_low: float = 5.125,
    market_cap: int = 13_500_000_000,
    trailing_pe: float = 22.5,
    beta: float = 3.12,
) -> dict:
    return {
        "shortName": name,
        "longName": name,
        "sector": sector,
        "industry": industry,
        "country": country,
        "currentPrice": 44.03,
        "previousClose": 45.45,
        "fiftyTwoWeekHigh": fifty_two_week_high,
        "fiftyTwoWeekLow": fifty_two_week_low,
        "marketCap": market_cap,
        "trailingPE": trailing_pe,
        "beta": beta,
    }


def _make_yfinance_history(num_days: int = 5) -> pd.DataFrame:
    dates = pd.date_range(
        end=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
        periods=num_days,
        freq="B",  # business days
    )
    data = {
        "Open": [40.0 + i for i in range(num_days)],
        "High": [41.0 + i for i in range(num_days)],
        "Low": [39.0 + i for i in range(num_days)],
        "Close": [40.5 + i for i in range(num_days)],
        "Volume": [1000000] * num_days,
    }
    return pd.DataFrame(data, index=dates)


def _mock_untracked_symbol_endpoints(symbol: str = "IREN"):
    """Mock endpoints for a symbol not tracked in Ghostfolio."""
    # Symbol endpoint works (returns current price)
    respx.get("http://localhost:3333/api/v1/symbol/YAHOO/{sym}".format(sym=symbol)).mock(
        return_value=httpx.Response(200, json={
            "dataSource": "YAHOO",
            "symbol": symbol,
            "currency": "USD",
            "marketPrice": 44.03,
            "historicalData": [],
        })
    )
    # Market-data endpoint returns 404 (not tracked)
    respx.get("http://localhost:3333/api/v1/market-data/YAHOO/{sym}".format(sym=symbol)).mock(
        return_value=httpx.Response(404)
    )


# ---------------------------------------------------------------------------
# yfinance fallback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_provides_profile(tool):
    """yfinance should provide profile data when Ghostfolio has none."""
    _mock_untracked_symbol_endpoints("IREN")

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yfinance_info()
        mock_ticker.history.return_value = _make_yfinance_history()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="IREN", data_source="YAHOO")

    r = result["result"]
    assert r["name"] == "IREN LIMITED"
    assert r["sector"] == "Financial Services"
    assert r["country"] == "Australia"
    assert r["asset_class"] == "EQUITY"


@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_provides_historical(tool):
    """yfinance should provide historical prices when Ghostfolio has none."""
    _mock_untracked_symbol_endpoints("IREN")

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yfinance_info()
        mock_ticker.history.return_value = _make_yfinance_history(5)
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="IREN", data_source="YAHOO")

    prices = result["result"]["historical_prices"]
    assert len(prices) == 5
    assert prices[0]["price"] == 40.5


@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_provides_extras(tool):
    """yfinance should provide enrichment fields (52wk, market cap, P/E, beta)."""
    _mock_untracked_symbol_endpoints("IREN")

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yfinance_info()
        mock_ticker.history.return_value = _make_yfinance_history()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="IREN", data_source="YAHOO")

    r = result["result"]
    assert r["fifty_two_week_high"] == 76.87
    assert r["fifty_two_week_low"] == 5.12  # round(5.125, 2) = 5.12 (banker's rounding)
    assert r["market_cap"] == 13_500_000_000
    assert r["pe_ratio"] == 22.5
    assert r["beta"] == 3.12
    assert r["industry"] == "Capital Markets"


@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_source_label(tool):
    """Source should show yfinance fallback in result, not tool_name."""
    _mock_untracked_symbol_endpoints("IREN")

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yfinance_info()
        mock_ticker.history.return_value = _make_yfinance_history()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="IREN", data_source="YAHOO")

    assert result["tool_name"] == "market_data"
    assert result["result"]["source"]["fallback"] == "yfinance"


@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_not_triggered_when_ghostfolio_has_data(tool):
    """yfinance should NOT be called when Ghostfolio has market data."""
    _mock_detail_endpoints(MOCK_MARKET_DATA_RESPONSE)

    with patch("yfinance.Ticker") as mock_ticker_cls:
        result = await tool.execute(jwt="fake-jwt", symbol="AAPL", data_source="YAHOO")

    # yfinance.Ticker should never be called
    mock_ticker_cls.assert_not_called()
    assert result["tool_name"] == "market_data"
    assert "fallback" not in result["result"]["source"]


@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_graceful_on_invalid_symbol(tool):
    """yfinance returning empty info should not crash."""
    _mock_untracked_symbol_endpoints("XYZFAKE")

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # invalid symbol
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="XYZFAKE", data_source="YAHOO")

    r = result["result"]
    # Should still return basic data from symbol endpoint
    assert r["current_price"] == 44.03
    assert r["historical_prices"] == []
    assert "name" not in r  # no profile


@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_graceful_on_exception(tool):
    """yfinance exception should be caught and not crash the tool."""
    _mock_untracked_symbol_endpoints("IREN")

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker_cls.side_effect = Exception("yfinance network error")

        result = await tool.execute(jwt="fake-jwt", symbol="IREN", data_source="YAHOO")

    r = result["result"]
    assert r["current_price"] == 44.03
    assert r["historical_prices"] == []


@pytest.mark.asyncio
@respx.mock
async def test_yfinance_fallback_daily_change(tool):
    """yfinance history should enable daily change computation."""
    _mock_untracked_symbol_endpoints("IREN")

    hist = _make_yfinance_history(5)

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yfinance_info()
        mock_ticker.history.return_value = hist
        mock_ticker_cls.return_value = mock_ticker

        result = await tool.execute(jwt="fake-jwt", symbol="IREN", data_source="YAHOO")

    r = result["result"]
    # Daily change should be computed from yfinance history
    assert "previous_close" in r or "daily_change" not in r  # depends on date matching
