"""Deterministic scoring functions for agent eval cases."""
from __future__ import annotations

import re
from typing import Any


def tool_used_score(tool_calls: list[dict], tool_name: str) -> float:
    """1.0 if the named tool was called, else 0.0."""
    for tc in tool_calls:
        if tc.get("tool_name") == tool_name:
            return 1.0
    return 0.0


def tool_not_used_score(tool_calls: list[dict], tool_name: str) -> float:
    """1.0 if the named tool was NOT called, else 0.0."""
    for tc in tool_calls:
        if tc.get("tool_name") == tool_name:
            return 0.0
    return 1.0


def contains_score(response: str, value: str) -> float:
    """1.0 if the response contains the value (case-insensitive), else 0.0."""
    return 1.0 if value.lower() in response.lower() else 0.0


def contains_any_score(response: str, value: str) -> float:
    """1.0 if the response contains ANY of the pipe-separated values (case-insensitive).

    Value format: 'word1|word2|word3'
    Example: "cannot|can't|unable to"
    """
    lower = response.lower()
    for alt in value.split("|"):
        if alt.strip().lower() in lower:
            return 1.0
    return 0.0


def not_contains_score(response: str, value: str) -> float:
    """1.0 if the response does NOT contain the value (case-insensitive), else 0.0."""
    return 0.0 if value.lower() in response.lower() else 1.0


def has_number_score(response: str) -> float:
    """1.0 if the response contains at least one number, else 0.0."""
    return 1.0 if re.search(r"\d+\.?\d*", response) else 0.0


def no_error_score(response: str) -> float:
    """1.0 if the response doesn't indicate an error, else 0.0."""
    error_indicators = [
        "i encountered an error",
        "error processing",
        "error: unauthorized",
        "failed to fetch",
        "system error",
        "an error occurred",
    ]
    lower = response.lower()
    for indicator in error_indicators:
        if indicator in lower:
            return 0.0
    return 1.0


def tool_param_equals_score(tool_calls: list[dict], value: str) -> float:
    """1.0 if a tool was called with a specific param=value, else 0.0.

    Value format: 'tool_name.param_name=expected_value'
    Example: 'portfolio_analysis.view=performance'
    """
    try:
        tool_part, expected_value = value.split("=", 1)
        tool_name, param_name = tool_part.rsplit(".", 1)
    except ValueError:
        return 0.0

    for tc in tool_calls:
        if tc.get("tool_name") != tool_name:
            continue
        tool_input = tc.get("tool_input", {})
        actual = str(tool_input.get(param_name, ""))
        if actual.lower() == expected_value.lower():
            return 1.0
    return 0.0


def tool_param_contains_score(tool_calls: list[dict], value: str) -> float:
    """1.0 if a tool's param value contains a substring, else 0.0.

    Value format: 'tool_name.param_name=substring'
    Example: 'portfolio_analysis.symbols=AAPL'
    """
    try:
        tool_part, substring = value.split("=", 1)
        tool_name, param_name = tool_part.rsplit(".", 1)
    except ValueError:
        return 0.0

    for tc in tool_calls:
        if tc.get("tool_name") != tool_name:
            continue
        tool_input = tc.get("tool_input", {})
        actual = str(tool_input.get(param_name, ""))
        if substring.lower() in actual.lower():
            return 1.0
    return 0.0


def no_tools_used_score(tool_calls: list[dict]) -> float:
    """1.0 if zero tools were called, else 0.0."""
    return 1.0 if len(tool_calls) == 0 else 0.0


def min_tool_calls_score(tool_calls: list[dict], min_count: str) -> float:
    """1.0 if at least min_count distinct tools were called, else 0.0."""
    distinct_tools = set(tc.get("tool_name") for tc in tool_calls)
    return 1.0 if len(distinct_tools) >= int(min_count) else 0.0


# Registry mapping assertion types to scorer functions
SCORERS = {
    "tool_used": lambda result, value: tool_used_score(result["tool_calls"], value),
    "tool_not_used": lambda result, value: tool_not_used_score(result["tool_calls"], value),
    "contains": lambda result, value: contains_score(result["message"], value),
    "contains_any": lambda result, value: contains_any_score(result["message"], value),
    "not_contains": lambda result, value: not_contains_score(result["message"], value),
    "has_number": lambda result, _: has_number_score(result["message"]),
    "no_error": lambda result, _: no_error_score(result["message"]),
    "tool_param_equals": lambda result, value: tool_param_equals_score(result["tool_calls"], value),
    "tool_param_contains": lambda result, value: tool_param_contains_score(result["tool_calls"], value),
    "no_tools_used": lambda result, _: no_tools_used_score(result["tool_calls"]),
    "min_tool_calls": lambda result, value: min_tool_calls_score(result["tool_calls"], value),
}


def evaluate_assertions(
    result: dict[str, Any], assertions: list[dict]
) -> list[dict[str, Any]]:
    """Run all assertions against an agent result.

    Args:
        result: Agent response dict with 'message', 'tool_calls', 'confidence'
        assertions: List of assertion dicts with 'type' and optional 'value'

    Returns:
        List of scored assertion results.
    """
    scored = []
    for assertion in assertions:
        a_type = assertion["type"]
        a_value = assertion.get("value", "")
        scorer = SCORERS.get(a_type)

        if scorer is None:
            scored.append({
                "type": a_type,
                "value": a_value,
                "score": 0.0,
                "error": f"Unknown assertion type: {a_type}",
            })
            continue

        try:
            score = scorer(result, a_value)
        except Exception as e:
            score = 0.0
            scored.append({
                "type": a_type,
                "value": a_value,
                "score": score,
                "error": str(e),
            })
            continue

        scored.append({
            "type": a_type,
            "value": a_value,
            "score": score,
        })

    return scored
