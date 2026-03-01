"""Compliance check tool — portfolio health rules with composition context.

Uses two endpoints:
  - /api/v1/portfolio/report   → X-Ray rules with pass/fail, thresholds
  - /api/v1/portfolio/details  → actual portfolio composition for context
"""
from __future__ import annotations

from typing import Any, Dict, List

from ghostfolio_agent.tools.base import GhostfolioTool


def _pct(value: float) -> str:
    """Format as '12.3%' or '—'."""
    if value is None:
        return "—"
    return "{:.1f}%".format(value * 100)


class ComplianceCheckTool(GhostfolioTool):
    @property
    def name(self) -> str:
        return "compliance_check"

    @property
    def description(self) -> str:
        return (
            "Runs a compliance check on the user's portfolio. Returns rule "
            "results (pass/fail with thresholds and current values) plus "
            "portfolio composition (asset class, market, currency, and account "
            "allocation percentages). Use this for diversification, "
            "concentration risk, emergency fund, and fee questions."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, jwt: str, **kwargs) -> Dict[str, Any]:
        report_resp = await self._api_get("/api/v1/portfolio/report", jwt)

        if report_resp.status_code == 401:
            return {
                "tool_name": self.name,
                "result": {"error": "Unauthorized. Please check your authentication."},
            }
        if report_resp.status_code != 200:
            return {
                "tool_name": self.name,
                "result": {
                    "error": "Failed to fetch compliance report (HTTP {code}).".format(
                        code=report_resp.status_code
                    )
                },
            }

        details_resp = await self._api_get("/api/v1/portfolio/details", jwt)
        details = details_resp.json() if details_resp.status_code == 200 else {}

        report = report_resp.json()
        x_ray = report.get("xRay", {})

        rules = self._build_rules(x_ray.get("categories", []))
        composition = self._build_composition(details)
        statistics = x_ray.get("statistics", {})

        active = statistics.get("rulesActiveCount", 0)
        fulfilled = statistics.get("rulesFulfilledCount", 0)
        failed = active - fulfilled

        return {
            "tool_name": self.name,
            "result": {
                "summary": "{fulfilled} of {active} rules passed{fail_note}".format(
                    fulfilled=fulfilled,
                    active=active,
                    fail_note=(
                        " ({failed} failed)".format(failed=failed)
                        if failed > 0
                        else ""
                    ),
                ),
                "rules": rules,
                "portfolio_composition": composition,
            },
        }

    @staticmethod
    def _is_numeric(val: Any) -> bool:
        """True for int/float, False for bool/None/str."""
        return isinstance(val, (int, float)) and not isinstance(val, bool)

    def _build_rules(self, categories: List[dict]) -> List[dict]:
        """Extract rules with thresholds and status.

        Note: The API returns thresholdMin/thresholdMax as booleans
        (enabled/disabled) for default rules. Only custom-configured
        rules have numeric values. The evaluation string always
        contains the actual numbers regardless.
        """
        rules: List[dict] = []
        for category in categories:
            for rule in category.get("rules", []):
                if not rule.get("isActive"):
                    continue

                passed = rule.get("value")
                if passed is True:
                    status = "PASS"
                elif passed is False:
                    status = "FAIL"
                else:
                    status = "N/A"

                entry: dict = {
                    "category": category.get("name", "Unknown"),
                    "name": rule.get("name", "Unknown"),
                    "status": status,
                    "evaluation": rule.get("evaluation", ""),
                }

                # Include threshold targets only when they are real numbers
                # (the API returns booleans for default rules)
                config = rule.get("configuration", {})
                threshold = config.get("threshold", {})
                if threshold:
                    unit = threshold.get("unit", "")
                    t_min = config.get("thresholdMin")
                    t_max = config.get("thresholdMax")
                    min_ok = self._is_numeric(t_min)
                    max_ok = self._is_numeric(t_max)

                    if min_ok and max_ok:
                        if unit == "%":
                            entry["target"] = "{min}% – {max}%".format(
                                min=round(t_min * 100, 1),
                                max=round(t_max * 100, 1),
                            )
                        else:
                            entry["target"] = "{min} – {max}".format(
                                min=t_min, max=t_max
                            )
                    elif max_ok:
                        if unit == "%":
                            entry["target"] = "max {max}%".format(
                                max=round(t_max * 100, 1)
                            )
                        else:
                            entry["target"] = "max {max}".format(max=t_max)
                    elif min_ok:
                        if unit == "%":
                            entry["target"] = "min {min}%".format(
                                min=round(t_min * 100, 1)
                            )
                        else:
                            entry["target"] = "min {min}".format(min=t_min)

                rules.append(entry)
        return rules

    def _build_composition(self, details: dict) -> dict:
        """Build portfolio composition from /portfolio/details."""
        if not details:
            return {}

        holdings = details.get("holdings", {})
        summary = details.get("summary", {})
        composition: dict = {}

        # Asset class allocation
        asset_classes: Dict[str, float] = {}
        for h in holdings.values():
            ac = h.get("assetClass", "OTHER") or "OTHER"
            asset_classes[ac] = asset_classes.get(ac, 0) + (
                h.get("valueInPercentage", 0) or 0
            )
        if asset_classes:
            composition["asset_classes"] = {
                k: _pct(v)
                for k, v in sorted(asset_classes.items(), key=lambda x: -x[1])
            }

        # Market allocation (economic)
        markets = details.get("markets", {})
        if markets:
            composition["markets"] = {}
            for key in ("developedMarkets", "emergingMarkets", "otherMarkets"):
                m = markets.get(key, {})
                pct = m.get("valueInPercentage")
                if pct is not None:
                    label = key.replace("Markets", "").replace("other", "Other")
                    composition["markets"][label] = _pct(pct)

        # Regional allocation
        markets_adv = details.get("marketsAdvanced", {})
        if markets_adv:
            composition["regions"] = {}
            region_labels = {
                "northAmerica": "North America",
                "europe": "Europe",
                "asiaPacific": "Asia-Pacific",
                "japan": "Japan",
                "emergingMarkets": "Emerging",
            }
            for key, label in region_labels.items():
                r = markets_adv.get(key, {})
                pct = r.get("valueInPercentage")
                if pct is not None:
                    composition["regions"][label] = _pct(pct)

        # Currency allocation (from holdings)
        currencies: Dict[str, float] = {}
        for h in holdings.values():
            cur = h.get("currency", "???")
            currencies[cur] = currencies.get(cur, 0) + (
                h.get("valueInPercentage", 0) or 0
            )
        if currencies:
            composition["currencies"] = {
                k: _pct(v)
                for k, v in sorted(currencies.items(), key=lambda x: -x[1])
            }

        # Account allocation
        accounts = details.get("accounts", {})
        if accounts:
            composition["accounts"] = {}
            for acct_id, acct in accounts.items():
                name = acct.get("name", acct_id)
                pct = acct.get("valueInPercentage")
                if pct is not None:
                    composition["accounts"][name] = _pct(pct)

        # Key summary metrics
        if summary:
            ef = summary.get("emergencyFund", {})
            composition["summary"] = {
                "cash": round(summary.get("cash", 0), 2),
                "fees_total": round(summary.get("fees", 0), 2),
                "emergency_fund": round(
                    ef.get("total", 0) if isinstance(ef, dict) else 0, 2
                ),
                "dividend_total": round(
                    summary.get("dividendInBaseCurrency", 0), 2
                ),
            }

        return composition
