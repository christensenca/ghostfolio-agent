"""Portfolio analysis tool — fetches holdings, allocation, and performance.

Uses two endpoints:
  - /api/v1/portfolio/details  → holdings, accounts, portfolio summary
  - /api/v1/order              → buy/sell orders for per-holding cost basis
  - /api/v1/symbol/{ds}/{sym}  → historical data for daily change (on demand)

All math happens in pandas. The LLM receives a single pre-formatted
markdown table it can reference directly.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ghostfolio_agent.tools.base import GhostfolioTool

# View presets: map preset name → list of display column names
VIEW_COLUMNS = {
    "full": [
        "Symbol", "Name", "Sector", "Shares", "Market Price",
        "Cost Basis/Share", "Cost Basis", "Value", "Allocation",
        "Gain", "Gain %",
    ],
    "performance": ["Symbol", "Name", "Value", "Gain", "Gain %"],
    "exposure": ["Symbol", "Name", "Country", "Sector", "Value", "Allocation"],
    "daily": [
        "Symbol", "Name", "Market Price", "Day Change", "Day Change %", "Value",
    ],
    "compact": ["Symbol", "Value", "Gain", "Gain %"],
}


def _fmt_dollar(val):
    """Format number as $1,234.56 or '—' for NaN."""
    if pd.isna(val):
        return "—"
    return "${:,.2f}".format(val)


def _fmt_pct(val):
    """Format number as 12.34% or '—' for NaN."""
    if pd.isna(val):
        return "—"
    return "{:.2f}%".format(val)


class PortfolioAnalysisTool(GhostfolioTool):
    @property
    def name(self) -> str:
        return "portfolio_analysis"

    @property
    def description(self) -> str:
        return (
            "Returns the user's portfolio holdings with prices, cost basis, "
            "values, allocation percentages, gain/loss, and sector breakdown. "
            "Also includes a portfolio-level performance summary. Use this "
            "when the user asks about their portfolio, holdings, allocations, "
            "or investment performance."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": (
                        "Filter holdings by account name (e.g. 'Roth IRA', 'Brokerage'). "
                        "Case-insensitive partial match. Omit to show all accounts."
                    ),
                },
                "asset_classes": {
                    "type": "string",
                    "description": (
                        "Comma-separated asset classes to filter "
                        "(e.g. 'EQUITY', 'EQUITY,CRYPTOCURRENCY'). Omit to show all."
                    ),
                },
                "range": {
                    "type": "string",
                    "description": (
                        "Date range for performance: "
                        "'max', '1y', 'ytd', '6m', '3m'. Default: 'max'."
                    ),
                },
                "view": {
                    "type": "string",
                    "enum": ["full", "performance", "exposure", "daily", "compact"],
                    "description": (
                        "Controls which columns to include. "
                        "'full' (default): Symbol, Name, Sector, Shares, Market Price, Cost Basis/Share, Cost Basis, Value, Allocation, Gain, Gain %. "
                        "'performance': Symbol, Name, Value, Gain, Gain %. "
                        "'exposure': Symbol, Name, Country, Sector, Value, Allocation. "
                        "'daily': Symbol, Name, Market Price, Day Change, Day Change %, Value. "
                        "'compact': Symbol, Value, Gain, Gain % (minimal for chaining)."
                    ),
                },
                "symbols": {
                    "type": "string",
                    "description": (
                        "Comma-separated symbols to include (e.g. 'AAPL,GOOGL'). "
                        "Case-insensitive. Omit to show all holdings."
                    ),
                },
                "include_daily_change": {
                    "type": "boolean",
                    "description": (
                        "When true, include daily price change columns (Day Change, "
                        "Day Change %) in the output table. Use when the user asks "
                        "about today's movers, what's up/down today."
                    ),
                },
                "include_countries": {
                    "type": "boolean",
                    "description": (
                        "When true, include a Country column in the output table. "
                        "Use when the user asks about geographic exposure, country "
                        "allocation, or wants to reduce exposure to a specific region."
                    ),
                },
                "filter_gains": {
                    "type": "string",
                    "enum": ["unrealized_losses", "unrealized_gains"],
                    "description": (
                        "Filter holdings by gain/loss status. 'unrealized_losses' "
                        "shows only holdings currently at a loss (useful for tax-loss "
                        "harvesting). 'unrealized_gains' shows only holdings at a profit."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, jwt: str, **kwargs) -> Dict[str, Any]:
        account_name = kwargs.get("account")
        asset_classes = kwargs.get("asset_classes")
        date_range = kwargs.get("range")
        view = kwargs.get("view", "full")
        symbols_param = kwargs.get("symbols")
        include_daily_change = kwargs.get("include_daily_change", False)
        include_countries = kwargs.get("include_countries", False)
        filter_gains = kwargs.get("filter_gains")

        # View presets implicitly enable features
        if view == "daily":
            include_daily_change = True
        if view == "exposure":
            include_countries = True

        # --- Step 1: Resolve account name → ID (if provided) ---
        account_warning = None
        query_parts = []

        if account_name:
            account_id = await self._resolve_account_id(jwt, account_name)
            if account_id:
                query_parts.append("accounts={v}".format(v=account_id))
            else:
                account_warning = (
                    "No account matching '{name}' found. Showing all accounts."
                    .format(name=account_name)
                )

        if asset_classes:
            query_parts.append("assetClasses={v}".format(v=asset_classes))
        if date_range:
            query_parts.append("range={v}".format(v=date_range))

        path = "/api/v1/portfolio/details"
        if query_parts:
            path = path + "?" + "&".join(query_parts)

        # --- Step 2: Fetch details ---
        details_resp = await self._api_get(path, jwt)

        if details_resp.status_code == 401:
            return {
                "tool_name": self.name,
                "result": {"error": "Unauthorized. Please check your authentication."},
            }

        if details_resp.status_code != 200:
            return {
                "tool_name": self.name,
                "result": {
                    "error": "Failed to fetch portfolio details (HTTP {code}).".format(
                        code=details_resp.status_code
                    )
                },
            }

        details = details_resp.json()
        holdings_map = details.get("holdings", {})

        # --- Step 3: Fetch orders for per-holding cost basis ---
        orders_cost_basis = await self._compute_cost_basis_from_orders(jwt)

        # --- Step 3b: Fetch daily changes (if needed) ---
        daily_changes: Dict[str, Tuple[float, float]] = {}
        if include_daily_change and holdings_map:
            daily_changes = await self._fetch_daily_changes(jwt, holdings_map)

        # --- Step 4: Build holdings DataFrame ---
        df = self._build_holdings_df(holdings_map, orders_cost_basis, daily_changes)

        # --- Step 4b: Apply symbols filter ---
        if symbols_param and not df.empty:
            symbol_list = [
                s.strip().upper() for s in symbols_param.split(",") if s.strip()
            ]
            df = df[df["symbol"].str.upper().isin(symbol_list)].reset_index(drop=True)

        # --- Step 4c: Apply gain/loss filter ---
        if filter_gains == "unrealized_losses":
            df = df[df["gain"] < 0].reset_index(drop=True)
        elif filter_gains == "unrealized_gains":
            df = df[df["gain"] > 0].reset_index(drop=True)

        # --- Step 5: Build portfolio summary ---
        is_filtered = bool(query_parts) or bool(filter_gains) or bool(symbols_param)
        portfolio = self._build_portfolio_summary(details, df, filtered=is_filtered)

        # --- Step 6: Build accounts list ---
        accounts = details.get("accounts", {})
        account_list = [
            {
                "id": acct_id,
                "name": acct.get("name", "Unknown"),
                "balance": round(acct.get("balance", 0), 2),
                "total_value": round(acct.get("valueInBaseCurrency", 0), 2),
                "currency": acct.get("currency", ""),
            }
            for acct_id, acct in accounts.items()
        ]

        # --- Step 7: Format output ---
        result = self._format_output(
            df, portfolio, account_list,
            view=view,
            include_daily_change=include_daily_change,
            include_countries=include_countries,
        )

        if account_warning:
            result["warning"] = account_warning

        return {"tool_name": self.name, "result": result}

    async def _fetch_daily_changes(
        self, jwt: str, holdings_map: dict
    ) -> Dict[str, Tuple[float, float]]:
        """Fetch previous close for each holding, compute daily change.

        Returns {symbol: (daily_change, daily_change_pct)}.
        Rate-limit resilient: individual failures return None, not crash.
        """

        async def _fetch_one(
            symbol: str, data_source: str, current_price: float
        ) -> Tuple[str, float | None, float | None]:
            try:
                path = "/api/v1/symbol/{ds}/{sym}?includeHistoricalData=2".format(
                    ds=data_source, sym=symbol
                )
                resp = await self._api_get(path, jwt)
                if resp.status_code != 200:
                    return symbol, None, None
                data = resp.json()
                hist = data.get("historicalData", [])
                # With includeHistoricalData=2, we get [previous_close, current].
                # Use the second-to-last entry as the previous close.
                if len(hist) >= 2:
                    prev = hist[-2].get("marketPrice", hist[-2].get("value", 0))
                    if prev > 0:
                        change = round(current_price - prev, 2)
                        pct = round((change / prev) * 100, 2)
                        return symbol, change, pct
                return symbol, None, None
            except Exception:
                return symbol, None, None  # Rate limited or network error

        tasks = []
        for h in holdings_map.values():
            sym = h.get("symbol", "")
            ds = h.get("dataSource", "YAHOO")
            price = h.get("marketPrice", 0)
            if sym and price > 0 and ds != "MANUAL":
                tasks.append(_fetch_one(sym, ds, price))

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks)
        return {
            sym: (chg, pct)
            for sym, chg, pct in results
            if chg is not None
        }

    async def _resolve_account_id(self, jwt: str, account_name: str) -> str | None:
        """Resolve a human account name to its UUID via /api/v1/account."""
        resp = await self._api_get("/api/v1/account", jwt)
        if resp.status_code != 200:
            return None

        data = resp.json()
        accounts = data.get("accounts", [])
        needle = account_name.lower()

        for acct in accounts:
            if needle in acct.get("name", "").lower():
                return acct.get("id")

        return None

    async def _compute_cost_basis_from_orders(self, jwt: str) -> dict[str, float]:
        """Fetch orders and compute per-symbol cost basis.

        Returns {symbol: remaining_cost} accounting for buys and sells.
        """
        resp = await self._api_get("/api/v1/order", jwt)
        if resp.status_code != 200:
            return {}

        activities = resp.json()
        if isinstance(activities, dict):
            activities = activities.get("activities", [])

        cost_by_symbol: dict[str, float] = {}
        qty_bought: dict[str, float] = {}

        for order in activities:
            symbol = order.get("SymbolProfile", {}).get("symbol", order.get("symbol", ""))
            order_type = order.get("type", "")
            quantity = float(order.get("quantity", 0))
            unit_price = float(order.get("unitPrice", 0))
            fee = float(order.get("fee", 0))

            if order_type == "BUY":
                cost_by_symbol[symbol] = cost_by_symbol.get(symbol, 0) + (quantity * unit_price) + fee
                qty_bought[symbol] = qty_bought.get(symbol, 0) + quantity
            elif order_type == "SELL":
                if symbol in qty_bought and qty_bought[symbol] > 0:
                    avg_cost = cost_by_symbol[symbol] / qty_bought[symbol]
                    cost_by_symbol[symbol] -= quantity * avg_cost
                    qty_bought[symbol] -= quantity

        return cost_by_symbol

    def _build_holdings_df(
        self,
        holdings_map: dict,
        orders_cost_basis: dict[str, float],
        daily_changes: Dict[str, Tuple[float, float]] | None = None,
    ) -> pd.DataFrame:
        """Build DataFrame from holdings + orders cost basis + daily changes."""
        if not holdings_map:
            return pd.DataFrame()

        daily_changes = daily_changes or {}

        rows = []
        for h in holdings_map.values():
            sectors = h.get("sectors") or []
            countries = h.get("countries") or []
            sym = h.get("symbol", "N/A")
            chg, chg_pct = daily_changes.get(sym, (None, None))
            rows.append({
                "symbol": sym,
                "name": h.get("name", "Unknown"),
                "sector": sectors[0].get("name", "—") if sectors else "—",
                "country": countries[0].get("name", "—") if countries else "—",
                "quantity": h.get("quantity", 0),
                "market_price": h.get("marketPrice", 0),
                "api_investment": h.get("investment", 0),
                "daily_change": chg,
                "daily_change_pct": chg_pct,
            })

        df = pd.DataFrame(rows)

        # Value = quantity × market_price
        df["value"] = (df["quantity"] * df["market_price"]).round(2)

        # Cost basis: prefer API investment, fall back to orders
        df["cost_basis"] = df["api_investment"].astype(float).round(2)
        if orders_cost_basis:
            no_cost = df["cost_basis"] <= 0
            if no_cost.any():
                df.loc[no_cost, "cost_basis"] = df.loc[no_cost, "symbol"].map(
                    lambda s: round(orders_cost_basis.get(s, 0), 2)
                )

        # Cost basis per share
        df["cost_basis_per_share"] = np.nan
        has_qty = df["quantity"] > 0
        if has_qty.any():
            df.loc[has_qty, "cost_basis_per_share"] = (
                df.loc[has_qty, "cost_basis"] / df.loc[has_qty, "quantity"]
            ).round(2)

        # Gain = value - cost_basis (only when cost_basis > 0)
        df["gain"] = np.nan
        df["gain_pct"] = np.nan
        has_cost = df["cost_basis"] > 0
        if has_cost.any():
            df.loc[has_cost, "gain"] = (
                df.loc[has_cost, "value"] - df.loc[has_cost, "cost_basis"]
            ).round(2)
            df.loc[has_cost, "gain_pct"] = (
                (df.loc[has_cost, "gain"] / df.loc[has_cost, "cost_basis"]) * 100
            ).round(2)

        # Allocation = value / total
        total_value = df["value"].sum()
        if total_value > 0:
            df["allocation_pct"] = ((df["value"] / total_value) * 100).round(2)
        else:
            df["allocation_pct"] = 0.0

        # Sort by allocation descending
        df = df.sort_values("allocation_pct", ascending=False).reset_index(drop=True)

        return df

    def _build_portfolio_summary(
        self, details: dict, df: pd.DataFrame, *, filtered: bool = False,
    ) -> Dict[str, Any]:
        """Build portfolio summary from details endpoint + computed DataFrame.

        When ``filtered`` is True the API summary fields (totalBuy/totalSell)
        are unreliable because they reflect the full portfolio. In that case
        we derive cost basis from the per-holding DataFrame instead.
        """
        summary = details.get("summary", {})
        total_value = float(df["value"].sum()) if not df.empty else 0.0

        if filtered and not df.empty:
            # Sum per-holding cost bases for an accurate filtered total
            net_cost_basis = float(df.loc[df["cost_basis"] > 0, "cost_basis"].sum())
        else:
            total_buy = summary.get("totalBuy", 0)
            total_sell = summary.get("totalSell", 0)
            net_cost_basis = total_buy - total_sell

        portfolio: Dict[str, Any] = {
            "total_value": round(total_value, 2),
            "net_cost_basis": round(net_cost_basis, 2),
            "fees": round(summary.get("fees", 0), 2),
            "activity_count": summary.get("activityCount", 0),
            "cash": round(summary.get("cash", 0), 2),
            "dividends_total": round(summary.get("dividendInBaseCurrency", 0), 2),
            "currency": "USD",
        }

        if not filtered:
            portfolio["total_buy"] = round(summary.get("totalBuy", 0), 2)
            portfolio["total_sell"] = round(summary.get("totalSell", 0), 2)

        if net_cost_basis > 0 and total_value > 0:
            gain = total_value - net_cost_basis
            portfolio["gain"] = round(gain, 2)
            portfolio["gain_pct"] = round((gain / net_cost_basis) * 100, 2)

        return portfolio

    def _format_output(
        self, df: pd.DataFrame, portfolio: dict, account_list: list,
        *, view: str = "full",
        include_daily_change: bool = False, include_countries: bool = False,
    ) -> Dict[str, Any]:
        """Format DataFrame into a single pre-formatted markdown table.

        Column selection is driven by the ``view`` preset. Boolean flags
        (``include_daily_change``, ``include_countries``) act as overrides
        that add columns on top of any view.
        """
        if df.empty:
            return {
                "portfolio": portfolio,
                "holdings_table": "No holdings found.",
                "accounts": account_list,
            }

        # Determine which columns this view needs
        view_cols = set(VIEW_COLUMNS.get(view, VIEW_COLUMNS["full"]))
        # Boolean overrides add columns on top of the view
        if include_daily_change:
            view_cols.update(["Day Change", "Day Change %"])
        if include_countries:
            view_cols.add("Country")

        # Build all possible columns, then select only those in view_cols
        all_cols: Dict[str, pd.Series] = {}
        all_cols["Symbol"] = df["symbol"]
        all_cols["Name"] = df["name"]
        all_cols["Sector"] = df["sector"]
        all_cols["Country"] = df["country"]
        all_cols["Shares"] = df["quantity"].map(
            lambda v: "{:.4f}".format(v).rstrip("0").rstrip(".") if pd.notna(v) else "—"
        )
        all_cols["Market Price"] = df["market_price"].map(_fmt_dollar)
        all_cols["Cost Basis/Share"] = df["cost_basis_per_share"].map(_fmt_dollar)
        all_cols["Cost Basis"] = df["cost_basis"].map(_fmt_dollar)
        all_cols["Value"] = df["value"].map(_fmt_dollar)
        all_cols["Allocation"] = df["allocation_pct"].map(_fmt_pct)
        all_cols["Gain"] = df["gain"].map(_fmt_dollar)
        all_cols["Gain %"] = df["gain_pct"].map(_fmt_pct)
        all_cols["Day Change"] = df["daily_change"].map(
            lambda v: _fmt_dollar(v) if pd.notna(v) else "—"
        )
        all_cols["Day Change %"] = df["daily_change_pct"].map(
            lambda v: _fmt_pct(v) if pd.notna(v) else "—"
        )

        # Maintain a stable column order
        ordered = [
            "Symbol", "Name", "Country", "Sector", "Shares",
            "Market Price", "Cost Basis/Share", "Cost Basis", "Value",
            "Allocation", "Gain", "Gain %", "Day Change", "Day Change %",
        ]

        display = pd.DataFrame({
            col: all_cols[col] for col in ordered if col in view_cols
        })

        md_table = display.to_markdown(index=False)

        return {
            "portfolio": portfolio,
            "holdings_table": md_table,
            "accounts": account_list,
        }
