"""Market data tool — looks up symbols and fetches price data.

Features:
  - Symbol lookup with ranked results (EQUITY before crypto junk)
  - Symbol detail with current price, daily change, and asset profile
  - Source attribution via result.source dict (data_source, provider_name)
  - 10-point historical prices for trend context
  - yfinance fallback for untracked symbols (not in portfolio)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from ghostfolio_agent.tools.base import GhostfolioTool

logger = logging.getLogger(__name__)

# Priority for ranking search results — lower = shown first.
_ASSET_CLASS_PRIORITY = {
    "EQUITY": 0,
    "FIXED_INCOME": 1,
    "REAL_ESTATE": 2,
    "COMMODITY": 3,
}
# Anything not listed (crypto tokens, etc.) gets 99.

# Friendly names for data sources.
_SOURCE_DISPLAY_NAMES = {
    "YAHOO": "Yahoo Finance",
    "COINGECKO": "CoinGecko",
    "EOD_HISTORICAL_DATA": "EOD Historical Data",
    "FINANCIAL_MODELING_PREP": "Financial Modeling Prep",
    "GOOGLE_SHEETS": "Google Sheets",
    "MANUAL": "Manual",
}


class MarketDataTool(GhostfolioTool):
    @property
    def name(self) -> str:
        return "market_data"

    @property
    def description(self) -> str:
        return (
            "Looks up symbols and retrieves market data including current price, "
            "daily change, asset profile, and historical prices. Results include "
            "data source attribution. Use 'query' to search for a symbol by name, "
            "or 'symbol' + 'data_source' to get details for a specific symbol."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term to look up symbols (e.g. 'Apple', 'AAPL')",
                },
                "symbol": {
                    "type": "string",
                    "description": "Exact symbol to get details for (e.g. 'AAPL')",
                },
                "data_source": {
                    "type": "string",
                    "description": "Data source for symbol detail lookup (default: 'YAHOO')",
                },
            },
            "required": [],
        }

    async def execute(self, jwt: str, **kwargs) -> Dict[str, Any]:
        symbol = kwargs.get("symbol")
        data_source = kwargs.get("data_source", "YAHOO")
        query = kwargs.get("query")

        if symbol and not query:
            return await self._get_symbol_detail(jwt, data_source, symbol)

        if query:
            return await self._lookup_symbol(jwt, query)

        return {
            "tool_name": self.name,
            "result": {
                "error": "Please provide either 'query' to search or 'symbol' to get details."
            },
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def _lookup_symbol(self, jwt: str, query: str) -> Dict[str, Any]:
        response = await self._api_get(
            "/api/v1/symbol/lookup?query={q}".format(q=query), jwt
        )

        if response.status_code == 401:
            return {
                "tool_name": self.name,
                "result": {"error": "Unauthorized. Please check your authentication."},
            }

        if response.status_code != 200:
            return {
                "tool_name": self.name,
                "result": {
                    "error": "Failed to search symbols (HTTP {code}).".format(
                        code=response.status_code
                    )
                },
            }

        data = response.json()
        items = data.get("items", [])

        # Rank results: EQUITY/FIXED_INCOME first, crypto last.
        items.sort(
            key=lambda x: _ASSET_CLASS_PRIORITY.get(x.get("assetClass", ""), 99)
        )

        # If query looks like a ticker (all uppercase, no spaces), boost exact matches.
        if query == query.upper() and " " not in query:
            items.sort(
                key=lambda x: 0 if x.get("symbol", "").upper() == query.upper() else 1
            )

        symbols = []
        for item in items[:10]:
            dpi = item.get("dataProviderInfo") or {}
            symbols.append(
                {
                    "symbol": item.get("symbol", "N/A"),
                    "name": item.get("name", "Unknown"),
                    "asset_class": item.get("assetClass", "N/A"),
                    "asset_sub_class": item.get("assetSubClass", "N/A"),
                    "currency": item.get("currency", "N/A"),
                    "data_source": item.get("dataSource", "N/A"),
                    "source": {
                        "data_source": item.get("dataSource", "N/A"),
                        "provider_name": dpi.get("name"),
                        "provider_url": dpi.get("url"),
                    },
                }
            )

        return {
            "tool_name": self.name,
            "result": {"symbols": symbols, "total_results": len(items)},
        }

    # ------------------------------------------------------------------
    # Symbol detail
    # ------------------------------------------------------------------

    async def _get_symbol_detail(
        self, jwt: str, data_source: str, symbol: str
    ) -> Dict[str, Any]:
        # Primary endpoint: current price + currency
        path = "/api/v1/symbol/{ds}/{sym}?includeHistoricalData=0".format(
            ds=data_source, sym=symbol
        )
        response = await self._api_get(path, jwt)

        if response.status_code == 401:
            return {
                "tool_name": self.name,
                "result": {"error": "Unauthorized. Please check your authentication."},
            }

        if response.status_code != 200:
            return {
                "tool_name": self.name,
                "result": {
                    "error": "Failed to fetch symbol data (HTTP {code}).".format(
                        code=response.status_code
                    )
                },
            }

        data = response.json()
        current_price = round(data.get("marketPrice", 0), 2)

        # Secondary endpoint: asset profile + real historical prices
        profile, market_data = await self._fetch_market_data(jwt, data_source, symbol)

        # Fallback to yfinance when Ghostfolio has no market data (untracked symbol)
        used_yfinance = False
        yfinance_extras = {}
        if not profile and not market_data:
            yf_profile, yf_market_data, yf_extras = await self._fetch_yfinance_data(
                symbol
            )
            if yf_profile or yf_market_data:
                profile = yf_profile
                market_data = yf_market_data
                yfinance_extras = yf_extras or {}
                used_yfinance = True

        # --- Historical prices (last 10) ---
        historical_prices = []
        if market_data:
            for point in market_data[-10:]:
                price = point.get("marketPrice", 0)
                if price and price > 0:
                    historical_prices.append(
                        {
                            "date": (point.get("date") or "N/A")[:10],
                            "price": round(price, 2),
                        }
                    )

        # --- Daily change (current vs previous close) ---
        daily = self._compute_daily_change(current_price, market_data or [])

        source_name = _SOURCE_DISPLAY_NAMES.get(data_source, data_source)

        result = {
            "symbol": symbol,
            "data_source": data_source,
            "current_price": current_price,
            "currency": data.get("currency", "N/A"),
            "historical_prices": historical_prices,
            "source": {"data_source": data_source, "provider_name": source_name},
        }

        if used_yfinance:
            result["source"]["fallback"] = "yfinance"

        # Merge yfinance extras (52wk range, market cap, P/E, beta, industry)
        if yfinance_extras:
            result.update(yfinance_extras)

        # Merge daily change if available
        if daily:
            result["previous_close"] = daily["previous_close"]
            result["daily_change"] = daily["daily_change"]
            result["daily_change_pct"] = daily["daily_change_pct"]

        # Merge asset profile if available
        if profile:
            result["name"] = profile.get("name")
            result["asset_class"] = profile.get("assetClass")

            sectors = profile.get("sectors") or []
            if sectors:
                result["sector"] = sectors[0].get("name")

            countries = profile.get("countries") or []
            if countries:
                result["country"] = countries[0].get("name")

        return {"tool_name": self.name, "result": result}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_daily_change(
        self, current_price: float, market_data: list
    ) -> Dict[str, float] | None:
        """Compute daily change from current price vs previous close.

        Walks market_data backwards, skips today's entries, and uses
        the most recent prior entry as "previous close".
        """
        if not market_data or current_price <= 0:
            return None

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        previous_close = None
        for point in reversed(market_data):
            date_str = (point.get("date") or "")[:10]
            if date_str and date_str != today_str:
                price = point.get("marketPrice", 0)
                if price and price > 0:
                    previous_close = price
                    break

        if previous_close is None or previous_close <= 0:
            return None

        change = round(current_price - previous_close, 2)
        change_pct = round((change / previous_close) * 100, 2)

        return {
            "previous_close": round(previous_close, 2),
            "daily_change": change,
            "daily_change_pct": change_pct,
        }

    async def _fetch_market_data(
        self, jwt: str, data_source: str, symbol: str
    ) -> tuple[Dict[str, Any] | None, list | None]:
        """Fetch asset profile + historical prices from /api/v1/market-data.

        Returns (assetProfile, marketData) tuple. Either may be None on error.
        """
        path = "/api/v1/market-data/{ds}/{sym}".format(ds=data_source, sym=symbol)
        try:
            response = await self._api_get(path, jwt)
            if response.status_code != 200:
                return None, None
            data = response.json()
            return data.get("assetProfile"), data.get("marketData")
        except Exception:
            logger.debug("Failed to fetch market data for %s/%s", data_source, symbol)
            return None, None

    async def _fetch_yfinance_data(
        self, symbol: str
    ) -> tuple[Dict[str, Any] | None, list | None, Dict[str, Any] | None]:
        """Fallback: fetch data directly from Yahoo Finance via yfinance.

        Used when Ghostfolio's market-data endpoint has no data (untracked symbol).
        Returns (profile, market_data, extras) tuple matching _fetch_market_data's
        shape for the first two elements.
        """
        try:
            import yfinance
        except ImportError:
            logger.debug("yfinance not installed — skipping direct fallback")
            return None, None, None

        try:

            def _fetch():
                ticker = yfinance.Ticker(symbol)
                info = ticker.info or {}
                hist = ticker.history(period="1mo")
                return info, hist

            info, hist = await asyncio.to_thread(_fetch)

            # yfinance returns minimal/empty dict for invalid symbols
            if not info.get("shortName") and not info.get("longName"):
                return None, None, None

            # Build profile matching Ghostfolio's assetProfile shape
            profile: Dict[str, Any] = {
                "name": info.get("shortName") or info.get("longName"),
                "assetClass": "EQUITY",
                "sectors": (
                    [{"name": info["sector"]}] if info.get("sector") else []
                ),
                "countries": (
                    [{"name": info["country"]}] if info.get("country") else []
                ),
            }

            # Build market_data list matching Ghostfolio's marketData shape
            market_data: list = []
            if hist is not None and not hist.empty:
                for date_idx, row in hist.iterrows():
                    close = float(row["Close"])
                    if close > 0:
                        market_data.append(
                            {
                                "date": date_idx.strftime(
                                    "%Y-%m-%dT00:00:00.000Z"
                                ),
                                "marketPrice": round(close, 2),
                            }
                        )

            # Extra enrichment fields not available from Ghostfolio
            extras: Dict[str, Any] = {}
            for key, field in [
                ("fiftyTwoWeekHigh", "fifty_two_week_high"),
                ("fiftyTwoWeekLow", "fifty_two_week_low"),
                ("marketCap", "market_cap"),
                ("trailingPE", "pe_ratio"),
                ("beta", "beta"),
            ]:
                val = info.get(key)
                if val is not None:
                    extras[field] = (
                        round(val, 2) if isinstance(val, float) else val
                    )

            if info.get("industry"):
                extras["industry"] = info["industry"]

            return profile, market_data, extras

        except Exception:
            logger.debug(
                "yfinance fallback failed for %s", symbol, exc_info=True
            )
            return None, None, None
