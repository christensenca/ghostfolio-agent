"""Tax estimate tool — pandas-based capital gains and dividend analysis.

Fetches all transactions via /api/v1/order, computes average cost basis
per symbol, calculates realized gains on sells, and summarizes dividend
income. Groups results by year, symbol, and account.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ghostfolio_agent.tools.base import GhostfolioTool


def _fmt_dollar(val) -> str:
    """Format number as $1,234.56 or '—' for NaN."""
    if pd.isna(val):
        return "—"
    return "${:,.2f}".format(val)


class TaxEstimateTool(GhostfolioTool):
    @property
    def name(self) -> str:
        return "tax_estimate"

    @property
    def description(self) -> str:
        return (
            "Estimates tax-relevant events: realized capital gains/losses from "
            "sells (using average cost basis) and dividend income. Returns per-year "
            "summaries, sells table with gain/loss, dividends table, and breakdowns "
            "by symbol and account. Supports filtering by account, symbol, and year. "
            "Use this for tax questions, capital gains, dividend income, or fee deductions."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": (
                        "Filter by account name (e.g. 'Crypto Portfolio'). "
                        "Case-insensitive partial match."
                    ),
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "Filter by symbol (e.g. 'bitcoin', 'AAPL'). "
                        "Case-insensitive exact match."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": (
                        "Filter to a specific tax year (e.g. 2025). "
                        "Default: all years."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, jwt: str, **kwargs) -> Dict[str, Any]:
        # Always fetch all transactions for accurate cost basis
        response = await self._api_get("/api/v1/order?range=max", jwt)

        if response.status_code == 401:
            return {
                "tool_name": self.name,
                "result": {"error": "Unauthorized. Please check your authentication."},
            }
        if response.status_code != 200:
            return {
                "tool_name": self.name,
                "result": {
                    "error": "Failed to fetch transactions (HTTP {code}).".format(
                        code=response.status_code
                    )
                },
            }

        activities = response.json().get("activities", [])
        if not activities:
            return {
                "tool_name": self.name,
                "result": {
                    "summary": self._empty_summary(),
                    "disclaimer": self._disclaimer(),
                },
            }

        df = self._build_df(activities)

        # Compute average cost basis per symbol from ALL buys (before filtering)
        avg_cost = self._compute_avg_cost(df)

        # Apply filters
        account_filter = kwargs.get("account")
        symbol_filter = kwargs.get("symbol")
        year_filter = kwargs.get("year")
        df, warning = self._apply_filters(df, account_filter, symbol_filter, year_filter)

        # Separate taxable events
        sells = df[df["type"] == "SELL"].copy()
        dividends = df[df["type"] == "DIVIDEND"].copy()

        # Compute gains on sells using average cost
        if not sells.empty:
            sells = self._compute_gains(sells, avg_cost)

        # Build output
        summary = self._build_summary(sells, dividends, df)
        sells_table = self._format_sells_table(sells)
        dividends_table = self._format_dividends_table(dividends)
        by_year = self._build_by_year(sells, dividends, df)
        by_symbol = self._build_by_symbol(sells, dividends)

        result: Dict[str, Any] = {
            "summary": summary,
            "by_year": by_year,
            "by_symbol": by_symbol,
            "sells_table": sells_table,
            "dividends_table": dividends_table,
            "filters_applied": {
                "account": account_filter,
                "symbol": symbol_filter,
                "year": year_filter,
            },
            "disclaimer": self._disclaimer(),
        }
        if warning:
            result["warning"] = warning

        return {"tool_name": self.name, "result": result}

    @staticmethod
    def _disclaimer() -> str:
        return (
            "This is NOT tax advice. These figures are estimates based on "
            "average cost basis. Consult a qualified tax professional for "
            "accurate tax calculations."
        )

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "total_realized_gains": 0,
            "total_dividends": 0,
            "total_fees": 0,
            "sell_count": 0,
            "dividend_count": 0,
        }

    @staticmethod
    def _build_df(activities: list) -> pd.DataFrame:
        """Build DataFrame from raw API activities."""
        rows = []
        for act in activities:
            profile = act.get("SymbolProfile") or {}
            account = act.get("account") or {}
            rows.append({
                "date": (act.get("date") or "")[:10],
                "year": int((act.get("date") or "0000")[:4]),
                "type": act.get("type", "UNKNOWN"),
                "symbol": profile.get("symbol", "N/A"),
                "name": profile.get("name", "Unknown"),
                "account": account.get("name") or "N/A",
                "quantity": float(act.get("quantity", 0)),
                "unit_price": float(act.get("unitPrice", 0)),
                "fee": float(act.get("feeInBaseCurrency", 0)),
                "value": float(act.get("valueInBaseCurrency", 0)),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _compute_avg_cost(df: pd.DataFrame) -> Dict[str, float]:
        """Compute average cost per unit for each symbol from BUY transactions."""
        buys = df[df["type"] == "BUY"]
        if buys.empty:
            return {}

        avg_cost: Dict[str, float] = {}
        for symbol, group in buys.groupby("symbol"):
            total_value = group["value"].sum()
            total_qty = group["quantity"].sum()
            if total_qty > 0:
                avg_cost[symbol] = total_value / total_qty
        return avg_cost

    @staticmethod
    def _apply_filters(
        df: pd.DataFrame,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply client-side filters. Returns (filtered_df, warning_or_None)."""
        warning = None

        if account:
            needle = account.lower()
            mask = df["account"].str.lower().str.contains(needle, na=False)
            if mask.any():
                df = df[mask]
            else:
                warning = "No transactions matching account '{a}'. Showing all.".format(
                    a=account
                )

        if symbol:
            needle = symbol.upper()
            mask = df["symbol"].str.upper() == needle
            if mask.any():
                df = df[mask]
            else:
                msg = "No transactions matching symbol '{s}'. Showing all.".format(
                    s=symbol
                )
                warning = msg if warning is None else warning + " " + msg

        if year:
            mask = df["year"] == year
            if mask.any():
                df = df[mask]
            else:
                msg = "No transactions in year {y}. Showing all.".format(y=year)
                warning = msg if warning is None else warning + " " + msg

        return df.reset_index(drop=True), warning

    @staticmethod
    def _compute_gains(sells: pd.DataFrame, avg_cost: Dict[str, float]) -> pd.DataFrame:
        """Add cost_basis and gain columns to sells DataFrame."""
        sells["avg_cost_per_unit"] = sells["symbol"].map(
            lambda s: avg_cost.get(s, 0)
        )
        sells["cost_basis"] = sells["quantity"] * sells["avg_cost_per_unit"]
        sells["gain"] = sells["value"] - sells["cost_basis"] - sells["fee"]
        return sells

    @staticmethod
    def _build_summary(
        sells: pd.DataFrame, dividends: pd.DataFrame, all_df: pd.DataFrame
    ) -> dict:
        """Build top-level summary stats."""
        total_gains = round(float(sells["gain"].sum()), 2) if not sells.empty else 0
        total_divs = round(float(dividends["value"].sum()), 2) if not dividends.empty else 0
        total_fees = round(float(all_df["fee"].sum()), 2)

        return {
            "total_realized_gains": total_gains,
            "total_dividends": total_divs,
            "total_fees": total_fees,
            "sell_count": len(sells),
            "dividend_count": len(dividends),
        }

    @staticmethod
    def _build_by_year(
        sells: pd.DataFrame, dividends: pd.DataFrame, all_df: pd.DataFrame
    ) -> List[dict]:
        """Build per-year breakdown sorted descending."""
        years = set()
        if not sells.empty:
            years.update(sells["year"].unique())
        if not dividends.empty:
            years.update(dividends["year"].unique())

        if not years:
            return []

        result = []
        for yr in sorted(years, reverse=True):
            yr_sells = sells[sells["year"] == yr] if not sells.empty else sells
            yr_divs = dividends[dividends["year"] == yr] if not dividends.empty else dividends
            yr_all = all_df[all_df["year"] == yr]

            entry: Dict[str, Any] = {
                "year": int(yr),
                "realized_gains": round(float(yr_sells["gain"].sum()), 2) if not yr_sells.empty else 0,
                "dividend_income": round(float(yr_divs["value"].sum()), 2) if not yr_divs.empty else 0,
                "fees": round(float(yr_all["fee"].sum()), 2),
                "sell_count": len(yr_sells),
                "dividend_count": len(yr_divs),
            }
            result.append(entry)
        return result

    @staticmethod
    def _build_by_symbol(sells: pd.DataFrame, dividends: pd.DataFrame) -> dict:
        """Build per-symbol breakdown of gains and dividends."""
        symbols: Dict[str, dict] = {}

        if not sells.empty:
            for sym, group in sells.groupby("symbol"):
                symbols[sym] = {
                    "realized_gains": round(float(group["gain"].sum()), 2),
                    "sell_count": len(group),
                    "total_proceeds": round(float(group["value"].sum()), 2),
                    "total_cost_basis": round(float(group["cost_basis"].sum()), 2),
                }

        if not dividends.empty:
            for sym, group in dividends.groupby("symbol"):
                entry = symbols.get(sym, {})
                entry["dividend_income"] = round(float(group["value"].sum()), 2)
                entry["dividend_count"] = len(group)
                symbols[sym] = entry

        return symbols

    def _format_sells_table(self, sells: pd.DataFrame) -> str:
        """Build markdown table for sell transactions with gain/loss."""
        if sells.empty:
            return "No sell transactions found."

        display = pd.DataFrame()
        display["Date"] = sells["date"]
        display["Symbol"] = sells["symbol"]
        display["Account"] = sells["account"]
        display["Qty"] = sells["quantity"].map(lambda v: "{:.4g}".format(v))
        display["Sell Price"] = sells["unit_price"].map(_fmt_dollar)
        display["Proceeds"] = sells["value"].map(_fmt_dollar)
        display["Avg Cost"] = sells["avg_cost_per_unit"].map(_fmt_dollar)
        display["Cost Basis"] = sells["cost_basis"].map(_fmt_dollar)
        display["Gain/Loss"] = sells["gain"].map(_fmt_dollar)
        display["Fee"] = sells["fee"].map(_fmt_dollar)

        return display.to_markdown(index=False)

    def _format_dividends_table(self, dividends: pd.DataFrame) -> str:
        """Build markdown table for dividend payments."""
        if dividends.empty:
            return "No dividend payments found."

        display = pd.DataFrame()
        display["Date"] = dividends["date"]
        display["Symbol"] = dividends["symbol"]
        display["Account"] = dividends["account"]
        display["Qty"] = dividends["quantity"].map(lambda v: "{:.4g}".format(v))
        display["Per Share"] = dividends["unit_price"].map(_fmt_dollar)
        display["Amount"] = dividends["value"].map(_fmt_dollar)

        return display.to_markdown(index=False)
