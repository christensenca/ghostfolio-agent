"""Post-processing verification for agent responses.

Five verification systems run automatically after the agent produces an answer:
1. Response grounding — checks that numbers in the response trace back to tool data
2. Source attribution — extracts data provider names from tool results
3. Domain constraints — validates financial invariants on tool data
4. Output validation — checks response text for formatting/content issues
5. Confidence scoring — multi-factor score with domain/output penalties
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel

# Numbers that are too common to be meaningful for grounding checks.
_TRIVIAL_NUMBERS = {0, 1, 2, 3, 4, 5, 10, 12, 100}

# Tolerance for matching numbers (accounts for rounding differences).
_MATCH_TOLERANCE = 0.01  # 1%

# Friendly display names for tools (used as source labels).
_TOOL_SOURCE_NAMES = {
    "portfolio_analysis": "Portfolio",
    "market_data": "Market Data",
    "transaction_categorize": "Transactions",
    "tax_estimate": "Tax Data",
    "compliance_check": "Compliance",
}

# Data freshness thresholds (days).
_FRESH_THRESHOLD = 1
_STALE_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class GroundingResult(BaseModel):
    grounded: int
    ungrounded: int
    rate: float  # 0.0-1.0
    ungrounded_values: List[float] = []


class SourceInfo(BaseModel):
    name: str
    tool: str


class ConstraintViolation(BaseModel):
    tool: str
    rule: str
    severity: str  # "warning" | "error"
    detail: str


class VerificationResult(BaseModel):
    confidence: float
    confidence_label: str  # "high" | "medium" | "low"
    grounding: GroundingResult
    sources: List[SourceInfo]
    domain_violations: List[ConstraintViolation] = []
    output_warnings: List[str] = []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_result(tr: Dict[str, Any]) -> Any:
    """Parse tool result, handling JSON-encoded strings."""
    result_data = tr.get("result", {})
    if isinstance(result_data, str):
        try:
            return json.loads(result_data)
        except (json.JSONDecodeError, TypeError):
            return {}
    return result_data


def _base_tool_name(tr: Dict[str, Any]) -> str:
    """Extract tool name from a tool result dict."""
    return tr.get("tool_name", "unknown")


# ---------------------------------------------------------------------------
# 1. Response Grounding
# ---------------------------------------------------------------------------

# Matches numbers like: 1234, 1,234.56, $1,234.56, -42.5, 12.5%
_NUMBER_PATTERN = re.compile(
    r"(?<![a-zA-Z])"           # not preceded by a letter
    r"-?"                       # optional negative sign
    r"\$?"                      # optional dollar sign
    r"\d{1,3}(?:,\d{3})*"     # integer part with optional comma grouping
    r"(?:\.\d+)?"              # optional decimal part
    r"%?"                       # optional percent sign
    r"(?![a-zA-Z])"           # not followed by a letter
)


def _extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text."""
    numbers = []
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group()
        # Strip formatting characters
        cleaned = raw.replace("$", "").replace(",", "").replace("%", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def _flatten_numbers_from_data(data: Any) -> List[float]:
    """Recursively extract all numeric values from a nested structure."""
    numbers = []
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        numbers.append(float(data))
    elif isinstance(data, str):
        # Parse numbers from string values (tool results often contain
        # formatted strings or JSON-encoded data)
        numbers.extend(_extract_numbers(data))
    elif isinstance(data, dict):
        for v in data.values():
            numbers.extend(_flatten_numbers_from_data(v))
    elif isinstance(data, (list, tuple)):
        for item in data:
            numbers.extend(_flatten_numbers_from_data(item))
    return numbers


def _numbers_match(a: float, b: float) -> bool:
    """Check if two numbers match within tolerance."""
    if a == b:
        return True
    if b == 0:
        return abs(a) < 0.01
    return abs(a - b) / abs(b) <= _MATCH_TOLERANCE


def ground_response(
    answer: str,
    tool_results: List[Dict[str, Any]],
    user_message: str = "",
) -> GroundingResult:
    """Check that numbers in the response are grounded in tool data."""
    if not tool_results:
        return GroundingResult(grounded=0, ungrounded=0, rate=1.0)

    # Extract numbers from the response
    response_numbers = _extract_numbers(answer)
    if not response_numbers:
        # Tools were used but response has no verifiable numbers.
        # If all tools returned useful data, the lack of numbers is likely
        # expected (e.g. "symbol not found" or qualitative responses).
        # If tools failed/returned empty data, it's more suspicious.
        all_useful = all(_is_tool_result_useful(tr) for tr in tool_results)
        neutral = 0.8 if all_useful else 0.5
        return GroundingResult(grounded=0, ungrounded=0, rate=neutral)

    # Build the set of known numbers from all tool results
    known_numbers: List[float] = []
    for tr in tool_results:
        result_data = _parse_result(tr)
        known_numbers.extend(_flatten_numbers_from_data(result_data))

    # Also extract numbers from the user message (so we don't flag
    # numbers the user themselves mentioned)
    user_numbers = set()
    for n in _extract_numbers(user_message):
        user_numbers.add(n)

    grounded = 0
    ungrounded = 0
    ungrounded_values = []

    for num in response_numbers:
        # Skip trivial numbers
        if num in _TRIVIAL_NUMBERS:
            continue
        # Skip numbers from the user's message
        if num in user_numbers:
            continue

        # Check against known values
        is_grounded = any(_numbers_match(num, known) for known in known_numbers)
        if is_grounded:
            grounded += 1
        else:
            ungrounded += 1
            ungrounded_values.append(num)

    total = grounded + ungrounded
    rate = grounded / total if total > 0 else 1.0

    return GroundingResult(
        grounded=grounded,
        ungrounded=ungrounded,
        rate=rate,
        ungrounded_values=ungrounded_values,
    )


# ---------------------------------------------------------------------------
# 2. Source Attribution
# ---------------------------------------------------------------------------


def extract_sources(tool_results: List[Dict[str, Any]]) -> List[SourceInfo]:
    """Extract data source names from tool results."""
    sources: List[SourceInfo] = []
    seen = set()

    for tr in tool_results:
        base_name = _base_tool_name(tr)
        result_data = _parse_result(tr)

        if not isinstance(result_data, dict):
            continue

        # Check for explicit source info
        source = result_data.get("source", {})
        provider_name = None
        if isinstance(source, dict):
            provider_name = source.get("provider_name")

        # Use provider name if available, otherwise use tool display name
        display_name = provider_name or _TOOL_SOURCE_NAMES.get(
            base_name, base_name
        )
        key = display_name
        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(name=display_name, tool=base_name))

    return sources


# ---------------------------------------------------------------------------
# 3. Domain Constraint Validation
# ---------------------------------------------------------------------------


def validate_tool_results(
    tool_results: List[Dict[str, Any]],
) -> List[ConstraintViolation]:
    """Validate financial invariants on tool data (raw numeric fields only)."""
    violations: List[ConstraintViolation] = []

    for tr in tool_results:
        result_data = _parse_result(tr)
        if not isinstance(result_data, dict):
            continue

        tool_name = _base_tool_name(tr)

        if tool_name == "portfolio_analysis":
            violations.extend(_check_portfolio_constraints(result_data))
        elif tool_name == "market_data":
            violations.extend(_check_market_constraints(result_data))
        elif tool_name == "tax_estimate":
            violations.extend(_check_tax_constraints(result_data))
        elif tool_name == "transaction_categorize":
            violations.extend(_check_transaction_constraints(result_data))

    return violations


def _check_portfolio_constraints(
    result: Dict[str, Any],
) -> List[ConstraintViolation]:
    violations: List[ConstraintViolation] = []
    portfolio = result.get("portfolio", {})
    if not isinstance(portfolio, dict):
        return violations

    total_value = portfolio.get("total_value")
    cost_basis = portfolio.get("net_cost_basis")
    gain = portfolio.get("gain")
    fees = portfolio.get("fees")

    # total_value >= 0
    if isinstance(total_value, (int, float)) and total_value < 0:
        violations.append(ConstraintViolation(
            tool="portfolio_analysis",
            rule="total_value_non_negative",
            severity="error",
            detail="total_value is {v}, expected >= 0".format(v=total_value),
        ))

    # fees >= 0
    if isinstance(fees, (int, float)) and fees < 0:
        violations.append(ConstraintViolation(
            tool="portfolio_analysis",
            rule="fees_non_negative",
            severity="warning",
            detail="fees is {v}, expected >= 0".format(v=fees),
        ))

    # gain ≈ total_value - cost_basis (±$1 tolerance)
    if (
        isinstance(gain, (int, float))
        and isinstance(total_value, (int, float))
        and isinstance(cost_basis, (int, float))
        and cost_basis > 0
    ):
        expected_gain = total_value - cost_basis
        if abs(gain - expected_gain) > 1.0:
            violations.append(ConstraintViolation(
                tool="portfolio_analysis",
                rule="gain_formula_consistency",
                severity="warning",
                detail="gain={g}, but total_value - cost_basis = {e} (diff={d})".format(
                    g=round(gain, 2),
                    e=round(expected_gain, 2),
                    d=round(abs(gain - expected_gain), 2),
                ),
            ))

    return violations


def _check_market_constraints(
    result: Dict[str, Any],
) -> List[ConstraintViolation]:
    violations: List[ConstraintViolation] = []

    current_price = result.get("current_price")
    previous_close = result.get("previous_close")
    daily_change = result.get("daily_change")
    hist = result.get("historical_prices", [])
    high_52w = result.get("fifty_two_week_high")
    low_52w = result.get("fifty_two_week_low")

    # current_price >= 0
    if isinstance(current_price, (int, float)) and current_price < 0:
        violations.append(ConstraintViolation(
            tool="market_data",
            rule="price_non_negative",
            severity="error",
            detail="current_price is {p}".format(p=current_price),
        ))

    # daily_change ≈ current_price - previous_close (±$0.02)
    if (
        isinstance(daily_change, (int, float))
        and isinstance(current_price, (int, float))
        and isinstance(previous_close, (int, float))
    ):
        expected = round(current_price - previous_close, 2)
        if abs(daily_change - expected) > 0.02:
            violations.append(ConstraintViolation(
                tool="market_data",
                rule="daily_change_formula",
                severity="warning",
                detail="daily_change={dc}, but current - previous = {e}".format(
                    dc=daily_change, e=expected,
                ),
            ))

    # Historical prices: dates chronologically ordered
    if isinstance(hist, list) and len(hist) > 1:
        dates = [h.get("date", "") for h in hist if isinstance(h, dict)]
        for i in range(1, len(dates)):
            if dates[i] and dates[i - 1] and dates[i] < dates[i - 1]:
                violations.append(ConstraintViolation(
                    tool="market_data",
                    rule="historical_dates_ordered",
                    severity="warning",
                    detail="dates out of order: {a} > {b}".format(
                        a=dates[i - 1], b=dates[i],
                    ),
                ))
                break  # One violation is enough

    # 52-week range: low <= current <= high (5% tolerance for after-hours)
    if (
        isinstance(current_price, (int, float))
        and isinstance(high_52w, (int, float))
        and isinstance(low_52w, (int, float))
        and current_price > 0
    ):
        tolerance = 0.05
        if current_price > high_52w * (1 + tolerance):
            violations.append(ConstraintViolation(
                tool="market_data",
                rule="price_within_52w_range",
                severity="warning",
                detail="current_price {p} > 52w high {h} (with 5% tolerance)".format(
                    p=current_price, h=high_52w,
                ),
            ))
        if current_price < low_52w * (1 - tolerance):
            violations.append(ConstraintViolation(
                tool="market_data",
                rule="price_within_52w_range",
                severity="warning",
                detail="current_price {p} < 52w low {l} (with 5% tolerance)".format(
                    p=current_price, l=low_52w,
                ),
            ))

    return violations


def _check_tax_constraints(
    result: Dict[str, Any],
) -> List[ConstraintViolation]:
    violations: List[ConstraintViolation] = []
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return violations

    dividends = summary.get("total_dividends")
    fees = summary.get("total_fees")
    sell_count = summary.get("sell_count")
    dividend_count = summary.get("dividend_count")

    if isinstance(dividends, (int, float)) and dividends < 0:
        violations.append(ConstraintViolation(
            tool="tax_estimate",
            rule="dividends_non_negative",
            severity="error",
            detail="total_dividends is {v}".format(v=dividends),
        ))

    if isinstance(fees, (int, float)) and fees < 0:
        violations.append(ConstraintViolation(
            tool="tax_estimate",
            rule="fees_non_negative",
            severity="warning",
            detail="total_fees is {v}".format(v=fees),
        ))

    if isinstance(sell_count, (int, float)) and sell_count < 0:
        violations.append(ConstraintViolation(
            tool="tax_estimate",
            rule="sell_count_non_negative",
            severity="error",
            detail="sell_count is {v}".format(v=sell_count),
        ))

    if isinstance(dividend_count, (int, float)) and dividend_count < 0:
        violations.append(ConstraintViolation(
            tool="tax_estimate",
            rule="dividend_count_non_negative",
            severity="error",
            detail="dividend_count is {v}".format(v=dividend_count),
        ))

    return violations


def _check_transaction_constraints(
    result: Dict[str, Any],
) -> List[ConstraintViolation]:
    violations: List[ConstraintViolation] = []
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return violations

    fees = summary.get("total_fees")
    count = summary.get("total_count")

    if isinstance(fees, (int, float)) and fees < 0:
        violations.append(ConstraintViolation(
            tool="transaction_categorize",
            rule="fees_non_negative",
            severity="warning",
            detail="total_fees is {v}".format(v=fees),
        ))

    if isinstance(count, (int, float)) and count < 0:
        violations.append(ConstraintViolation(
            tool="transaction_categorize",
            rule="count_non_negative",
            severity="error",
            detail="total_count is {v}".format(v=count),
        ))

    return violations


# ---------------------------------------------------------------------------
# 4. Output Validation
# ---------------------------------------------------------------------------

# Catches inline math like "100 / 5 =" or "50000 * 0.03 ="
_INLINE_MATH_PATTERN = re.compile(r"\d+\s*[/\*×]\s*\d+(?:\.\d+)?\s*=")

# Internal tool names that should never appear in user-facing responses
_TOOL_NAMES = {
    "portfolio_analysis",
    "market_data",
    "transaction_categorize",
    "tax_estimate",
    "compliance_check",
    "calculate",
}

# Raw field names from tool schemas that indicate verbatim output dumping.
# These are internal API field names that should be reformatted for users.
_RAW_FIELD_NAMES = {
    "net_cost_basis",
    "gain_pct",
    "activity_count",
    "total_buy",
    "total_sell",
    "dividends_total",
    "feeInBaseCurrency",
    "valueInBaseCurrency",
    "assetClass",
    "asset_class",
    "unitPrice",
    "unit_price",
    "total_count",
    "total_fees",
    "sell_count",
    "dividend_count",
    "total_realized_gains",
    "current_price",
    "previous_close",
    "daily_change",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "historical_prices",
    "data_source",
    "provider_name",
}

# Multi-line JSON pattern: opening brace/bracket followed by quoted key on
# same or next line.  Catches `{\n  "key":` and `[{"key":` variants.
_MULTILINE_JSON_PATTERN = re.compile(
    r'[\{\[]\s*\n\s*"[a-z_]+":', re.IGNORECASE
)

# Repeated "key": value pattern — 3+ JSON-like key-value pairs in sequence
# indicates raw object dump (e.g. "total_value": 43946, "net_cost_basis": ...)
_REPEATED_KV_PATTERN = re.compile(
    r'("[a-z_]+":\s*(?:"[^"]*"|\d[\d.,]*|true|false|null|\[|\{)'
    r'[\s,]*){3,}',
    re.IGNORECASE,
)

# Severity tiers for output warnings — used in confidence penalty calculation
OUTPUT_WARNING_SEVERITY: Dict[str, float] = {
    "raw_json_leak": 0.15,
    "tool_name_leak": 0.10,
    "raw_field_leak": 0.10,
    "tool_internals_leak": 0.10,
    "missing_tax_disclaimer": 0.05,
    "inline_math_detected": 0.05,
}


def validate_output(
    answer: str,
    tool_results: List[Dict[str, Any]],
) -> List[str]:
    """Validate the LLM's response text for formatting/content issues."""
    warnings: List[str] = []

    # --- Raw JSON leak (improved) ---
    # Inline JSON: {"key": or [{"key":
    if '{"' in answer or '[{"' in answer or '"tool_name"' in answer:
        warnings.append("raw_json_leak")
    # Multi-line JSON blocks
    elif _MULTILINE_JSON_PATTERN.search(answer):
        warnings.append("raw_json_leak")
    # Repeated key-value pairs (3+)
    elif _REPEATED_KV_PATTERN.search(answer):
        warnings.append("raw_json_leak")

    # --- Tool name leak ---
    answer_lower = answer.lower()
    for name in _TOOL_NAMES:
        if name in answer_lower:
            warnings.append("tool_name_leak")
            break

    # --- Raw field name leak ---
    # 3+ internal field names in the response indicates verbatim dump
    field_hits = sum(1 for f in _RAW_FIELD_NAMES if f in answer)
    if field_hits >= 3:
        warnings.append("raw_field_leak")

    # --- Tool internals leak ---
    for term in ("tool_input", "tool_output", "result_data"):
        if term in answer:
            warnings.append("tool_internals_leak")
            break

    # --- Tax disclaimer check ---
    tax_used = any(
        _base_tool_name(tr) == "tax_estimate" for tr in tool_results
    )
    if tax_used and "not tax advice" not in answer_lower:
        warnings.append("missing_tax_disclaimer")

    # --- Inline math detection ---
    if _INLINE_MATH_PATTERN.search(answer):
        warnings.append("inline_math_detected")

    return warnings


# ---------------------------------------------------------------------------
# 5. Confidence Scoring
# ---------------------------------------------------------------------------


def _compute_freshness_score(tool_results: List[Dict[str, Any]]) -> float:
    """Score data freshness based on timestamps in tool results."""
    latest_date = None

    for tr in tool_results:
        result_data = _parse_result(tr)

        if not isinstance(result_data, dict):
            continue

        # Check historical_prices for freshness
        for key in ("historical_prices", "historicalData"):
            hist = result_data.get(key, [])
            if hist and isinstance(hist, list):
                last_entry = hist[-1]
                date_str = last_entry.get("date", "")
                if date_str:
                    try:
                        dt = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )
                        # Ensure timezone-aware for comparison with utcnow
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if latest_date is None or dt > latest_date:
                            latest_date = dt
                    except (ValueError, TypeError):
                        continue

    if latest_date is None:
        return 0.7  # No date info — unknown, not perfect

    age_days = (datetime.now(timezone.utc) - latest_date).days
    if age_days <= _FRESH_THRESHOLD:
        return 1.0
    if age_days <= _STALE_THRESHOLD:
        return 0.7
    return 0.3


def _is_tool_result_useful(tr: Dict[str, Any]) -> bool:
    """Check if a tool result contains actual useful data (not just empty/zero)."""
    result_data = _parse_result(tr)
    if not isinstance(result_data, dict):
        return False

    # Explicit error
    if "error" in result_data:
        return False

    tool_name = _base_tool_name(tr)

    # market_data: distinguish lookup vs detail responses
    if tool_name == "market_data":
        # Lookup response has "symbols" key — always useful (even if empty,
        # "not found" is a valid answer)
        if "symbols" in result_data:
            return True
        # Detail response: price=0 with no history = empty/failed
        if (
            result_data.get("current_price", 0) == 0
            and not result_data.get("historical_prices")
        ):
            return False

    # portfolio_analysis: "No holdings found."
    if tool_name == "portfolio_analysis":
        if result_data.get("holdings_table") == "No holdings found.":
            # If filters were applied, "no holdings" is a valid answer
            # (e.g. user asked about TSLA but doesn't own it)
            tool_input = tr.get("tool_input", {})
            has_filters = any(
                tool_input.get(f)
                for f in ("symbols", "account", "asset_classes", "filter_gains")
            )
            if has_filters:
                return True
            return False

    # calculate: all steps errored
    if tool_name == "calculate":
        calcs = result_data.get("calculations", [])
        if calcs and all("error" in c for c in calcs):
            return False

    return True


def _compute_tool_success_rate(tool_results: List[Dict[str, Any]]) -> float:
    """Compute fraction of tools that returned useful data."""
    if not tool_results:
        return 1.0

    useful = sum(1 for tr in tool_results if _is_tool_result_useful(tr))
    return useful / len(tool_results)


def compute_confidence(
    grounding: GroundingResult,
    tool_results: List[Dict[str, Any]],
    domain_violations: List[ConstraintViolation] | None = None,
    output_warnings: List[str] | None = None,
) -> float:
    """Compute multi-factor confidence score (0.0-1.0)."""
    domain_violations = domain_violations or []
    output_warnings = output_warnings or []

    grounding_score = grounding.rate
    tool_success = _compute_tool_success_rate(tool_results)
    freshness = _compute_freshness_score(tool_results)
    has_sources = 1.0 if tool_results else 0.5

    base_score = (
        0.40 * grounding_score
        + 0.30 * tool_success
        + 0.20 * freshness
        + 0.10 * has_sources
    )

    # Domain violations penalty: -0.1 per violation, clamped at 0.4
    domain_penalty = min(len(domain_violations) * 0.1, 0.4)

    # Output warnings penalty: severity-weighted, clamped at 0.3
    output_penalty = min(
        sum(OUTPUT_WARNING_SEVERITY.get(w, 0.05) for w in output_warnings),
        0.3,
    )

    score = base_score - domain_penalty - output_penalty
    return round(min(max(score, 0.0), 1.0), 2)


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def verify_response(
    answer: str,
    tool_results: List[Dict[str, Any]],
    user_message: str = "",
) -> VerificationResult | None:
    """Run all verification checks on an agent response.

    Returns None for pure conversational replies (no tools used, no numbers).
    """
    # Skip verification for pure conversation (no tools, no financial data)
    if not tool_results and not _extract_numbers(answer):
        return None

    grounding = ground_response(answer, tool_results, user_message)
    sources = extract_sources(tool_results)
    domain_violations = validate_tool_results(tool_results)
    output_warnings = validate_output(answer, tool_results)
    confidence = compute_confidence(
        grounding, tool_results, domain_violations, output_warnings,
    )

    return VerificationResult(
        confidence=confidence,
        confidence_label=_confidence_label(confidence),
        grounding=grounding,
        sources=sources,
        domain_violations=domain_violations,
        output_warnings=output_warnings,
    )
