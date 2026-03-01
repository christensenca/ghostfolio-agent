"""Transaction categorize tool — pandas-based order history with filtering.

Fetches /api/v1/order, builds a DataFrame, applies client-side filters,
and returns summaries (by type, symbol, account, month) plus a markdown table.
Supports limit, sort_by, and format parameters for output control.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ghostfolio_agent.tools.base import GhostfolioTool


def _fmt_dollar(val):
    """Format number as $1,234.56 or '—' for NaN/zero-ish."""
    if pd.isna(val):
        return "—"
    return "${:,.2f}".format(val)


def _truncate(val, max_len=20):
    """Truncate string with ellipsis."""
    if pd.isna(val) or not val:
        return "—"
    s = str(val)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


class TransactionCategorizeTool(GhostfolioTool):
    @property
    def name(self) -> str:
        return "transaction_categorize"

    @property
    def description(self) -> str:
        return (
            "Fetches the user's transaction/order history with full details "
            "including account, value, fees, and asset class. Supports filtering "
            "by account name, symbol, transaction type, asset class, and date range. "
            "Supports limit (number of rows), sort_by (date/value/fee/symbol), and "
            "format (table/summary/both) to control output. "
            "Use this when the user asks about their transactions, trades, purchases, "
            "sells, dividends, or activity history."
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
                        "Filter by symbol (e.g. 'VOO', 'bitcoin'). "
                        "Case-insensitive exact match."
                    ),
                },
                "type": {
                    "type": "string",
                    "description": (
                        "Filter by transaction type: BUY, SELL, DIVIDEND, "
                        "ITEM, FEE, INTEREST, LIABILITY, STAKE. "
                        "Comma-separated for multiple."
                    ),
                },
                "asset_classes": {
                    "type": "string",
                    "description": "Comma-separated asset classes (e.g. 'EQUITY,CRYPTOCURRENCY').",
                },
                "range": {
                    "type": "string",
                    "description": "Date range: 'max', '1d', '1w', '1m', '3m', '6m', '1y', 'ytd'.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of transactions to return in the table. "
                        "Use for 'last N transactions' queries. Default: all."
                    ),
                },
                "sort_by": {
                    "type": "string",
                    "description": (
                        "Sort transactions by: 'date' (default, newest first), "
                        "'value' (highest first), 'fee' (highest first), 'symbol' (A-Z)."
                    ),
                },
                "format": {
                    "type": "string",
                    "description": (
                        "Output format: 'table' (markdown table only), "
                        "'summary' (stats only, no table), 'both' (default, table + stats)."
                    ),
                },
            },
            "required": [],
        }

    _RANGE_DELTAS = {
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90),
        "6m": timedelta(days=182),
        "1y": timedelta(days=365),
        "ytd": None,  # special-cased below
    }

    def _filter_by_range(
        self, df: pd.DataFrame, range_str: Optional[str]
    ) -> pd.DataFrame:
        """Filter DataFrame by date range client-side."""
        if not range_str or range_str == "max" or df.empty:
            return df

        today = datetime.now().date()

        if range_str == "ytd":
            cutoff = datetime(today.year, 1, 1).date()
        else:
            delta = self._RANGE_DELTAS.get(range_str)
            if delta is None:
                return df  # unknown range, return all
            cutoff = today - delta

        cutoff_str = cutoff.isoformat()
        return df[df["date"] >= cutoff_str].reset_index(drop=True)

    async def execute(self, jwt: str, **kwargs) -> Dict[str, Any]:
        # Always fetch all transactions and filter client-side
        path = "/api/v1/order?range=max"

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
                    "error": "Failed to fetch transactions (HTTP {code}).".format(
                        code=response.status_code
                    )
                },
            }

        data = response.json()
        activities = data.get("activities", [])

        if not activities:
            return {
                "tool_name": self.name,
                "result": {
                    "summary": {"total_count": 0, "total_value": 0, "total_fees": 0},
                    "by_type": {},
                    "transactions_table": "No transactions found.",
                },
            }

        df = self._build_activities_df(activities)

        # Date range filter (client-side)
        df = self._filter_by_range(df, kwargs.get("range"))

        # Client-side filters
        df, warning = self._apply_filters(
            df,
            account=kwargs.get("account"),
            symbol=kwargs.get("symbol"),
            type_filter=kwargs.get("type"),
            asset_classes=kwargs.get("asset_classes"),
        )

        # Sort
        sort_by = kwargs.get("sort_by", "date")
        df = self._apply_sort(df, sort_by)

        filters_applied = {
            "account": kwargs.get("account"),
            "symbol": kwargs.get("symbol"),
            "type": kwargs.get("type"),
            "asset_classes": kwargs.get("asset_classes"),
            "range": kwargs.get("range"),
            "limit": kwargs.get("limit"),
            "sort_by": sort_by,
            "format": kwargs.get("format", "both"),
        }

        if df.empty:
            result: Dict[str, Any] = {
                "summary": {"total_count": 0, "total_value": 0, "total_fees": 0},
                "by_type": {},
                "transactions_table": "No transactions match the applied filters.",
                "filters_applied": filters_applied,
            }
            if warning:
                result["warning"] = warning
            return {"tool_name": self.name, "result": result}

        output_format = kwargs.get("format", "both")
        limit = kwargs.get("limit")

        # Build summaries from full (untruncated) data
        summaries = self._build_summary(df)

        # Apply limit for table display
        table_df = df.head(limit) if limit and limit > 0 else df
        table = self._format_table(table_df, total_count=len(df))

        if output_format == "summary":
            result = {
                **summaries,
                "filters_applied": filters_applied,
            }
        elif output_format == "table":
            result = {
                "summary": summaries["summary"],
                "transactions_table": table,
                "filters_applied": filters_applied,
            }
        else:  # "both" (default)
            result = {
                **summaries,
                "transactions_table": table,
                "filters_applied": filters_applied,
            }

        if warning:
            result["warning"] = warning

        return {"tool_name": self.name, "result": result}

    def _build_activities_df(self, activities: list) -> pd.DataFrame:
        """Build a DataFrame from raw API activities."""
        rows = []
        for act in activities:
            profile = act.get("SymbolProfile") or {}
            account = act.get("account") or {}
            rows.append(
                {
                    "date": (act.get("date") or "")[:10],
                    "type": act.get("type", "UNKNOWN"),
                    "symbol": profile.get("symbol", "N/A"),
                    "name": profile.get("name", "Unknown"),
                    "asset_class": profile.get("assetClass") or "",
                    "account": account.get("name") or "N/A",
                    "currency": act.get("currency", ""),
                    "quantity": float(act.get("quantity", 0)),
                    "unit_price": float(act.get("unitPrice", 0)),
                    "fee": float(act.get("feeInBaseCurrency", 0)),
                    "value": float(act.get("valueInBaseCurrency", 0)),
                    "comment": act.get("comment") or "",
                }
            )
        df = pd.DataFrame(rows)
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        return df

    def _apply_filters(
        self,
        df: pd.DataFrame,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        type_filter: Optional[str] = None,
        asset_classes: Optional[str] = None,
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

        if type_filter:
            types = [t.strip().upper() for t in type_filter.split(",")]
            mask = df["type"].isin(types)
            if mask.any():
                df = df[mask]
            else:
                msg = "No transactions matching type '{t}'. Showing all.".format(
                    t=type_filter
                )
                warning = msg if warning is None else warning + " " + msg

        if asset_classes:
            classes = [c.strip().upper() for c in asset_classes.split(",")]
            mask = df["asset_class"].str.upper().isin(classes)
            if mask.any():
                df = df[mask]
            else:
                msg = "No transactions matching asset class '{a}'. Showing all.".format(
                    a=asset_classes
                )
                warning = msg if warning is None else warning + " " + msg

        return df.reset_index(drop=True), warning

    @staticmethod
    def _apply_sort(df: pd.DataFrame, sort_by: Optional[str] = None) -> pd.DataFrame:
        """Sort DataFrame by the given column. Default: date descending."""
        sort_map = {
            "date": ("date", False),
            "value": ("value", False),
            "fee": ("fee", False),
            "symbol": ("symbol", True),
        }
        col, ascending = sort_map.get(sort_by or "date", ("date", False))
        return df.sort_values(col, ascending=ascending).reset_index(drop=True)

    def _build_summary(self, df: pd.DataFrame) -> dict:
        """Compute summary stats from the DataFrame."""
        summary = {
            "total_count": len(df),
            "total_value": round(float(df["value"].sum()), 2),
            "total_fees": round(float(df["fee"].sum()), 2),
            "date_range": {
                "earliest": df["date"].min(),
                "latest": df["date"].max(),
            },
        }

        # By type
        by_type: Dict[str, dict] = {}
        for t, group in df.groupby("type"):
            by_type[t] = {
                "count": len(group),
                "total_value": round(float(group["value"].sum()), 2),
                "total_fees": round(float(group["fee"].sum()), 2),
            }

        # By symbol
        by_symbol: Dict[str, dict] = {}
        for s, group in df.groupby("symbol"):
            by_symbol[s] = {
                "count": len(group),
                "total_value": round(float(group["value"].sum()), 2),
                "types": sorted(group["type"].unique().tolist()),
            }

        # By account
        by_account: Dict[str, dict] = {}
        for a, group in df.groupby("account"):
            by_account[a] = {
                "count": len(group),
                "total_value": round(float(group["value"].sum()), 2),
            }

        # Monthly breakdown
        monthly: List[dict] = []
        months = df.copy()
        months["month"] = months["date"].str[:7]
        for month, group in months.groupby("month", sort=False):
            entry: Dict[str, Any] = {"month": month, "count": len(group)}
            for typ in group["type"].unique():
                key = "{t}_value".format(t=typ.lower())
                entry[key] = round(float(group.loc[group["type"] == typ, "value"].sum()), 2)
            monthly.append(entry)
        monthly.sort(key=lambda x: x["month"], reverse=True)

        return {
            "summary": summary,
            "by_type": by_type,
            "by_symbol": by_symbol,
            "by_account": by_account,
            "monthly": monthly,
        }

    _MAX_TABLE_ROWS = 20

    def _format_table(
        self, df: pd.DataFrame, total_count: Optional[int] = None
    ) -> str:
        """Build a markdown table from the DataFrame.

        Auto-truncates to _MAX_TABLE_ROWS with a summary footer when the
        table exceeds that limit and no explicit limit was applied.
        """
        if df.empty:
            return "No transactions found."

        shown_df = df
        truncated = False
        remaining = 0

        if len(df) > self._MAX_TABLE_ROWS:
            shown_df = df.head(self._MAX_TABLE_ROWS)
            truncated = True
            remaining = len(df) - self._MAX_TABLE_ROWS

        display = pd.DataFrame()
        display["Date"] = shown_df["date"]
        display["Type"] = shown_df["type"]
        display["Symbol"] = shown_df["symbol"]
        display["Name"] = shown_df["name"].map(lambda v: _truncate(v, 20))
        display["Account"] = shown_df["account"]
        display["Qty"] = shown_df["quantity"].map(lambda v: "{:.4g}".format(v))
        display["Unit Price"] = shown_df["unit_price"].map(_fmt_dollar)
        display["Value"] = shown_df["value"].map(_fmt_dollar)
        display["Fee"] = shown_df["fee"].map(_fmt_dollar)

        table = display.to_markdown(index=False)

        if truncated:
            total = total_count if total_count is not None else len(df) + remaining
            table += "\n\n*Showing {shown} of {total} transactions. Use `limit` or filters to narrow results.*".format(
                shown=self._MAX_TABLE_ROWS, total=total
            )

        return table
