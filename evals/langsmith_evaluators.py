"""LangSmith evaluator wrappers around existing deterministic scorers.

Each evaluator receives a LangSmith Run (with agent outputs) and an Example
(with expected outputs from the dataset) and returns EvaluationResult(s).
"""
from __future__ import annotations

from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

from evals.scorers import (
    contains_any_score,
    contains_score,
    has_number_score,
    min_tool_calls_score,
    no_error_score,
    no_tools_used_score,
    not_contains_score,
    tool_not_used_score,
    tool_param_contains_score,
    tool_param_equals_score,
    tool_used_score,
)


def _get_agent_output(run: Run) -> dict:
    """Extract agent output from run.outputs, handling missing data."""
    outputs = run.outputs or {}
    return {
        "message": outputs.get("message", ""),
        "tool_calls": outputs.get("tool_calls", []),
        "confidence": outputs.get("confidence", 0.0),
    }


# ---------------------------------------------------------------------------
# Scorer dispatch — maps assertion type strings to scoring lambdas
# ---------------------------------------------------------------------------
_SCORER_DISPATCH = {
    "tool_used": lambda out, val: tool_used_score(out["tool_calls"], val),
    "tool_not_used": lambda out, val: tool_not_used_score(out["tool_calls"], val),
    "contains": lambda out, val: contains_score(out["message"], val),
    "contains_any": lambda out, val: contains_any_score(out["message"], val),
    "not_contains": lambda out, val: not_contains_score(out["message"], val),
    "has_number": lambda out, _: has_number_score(out["message"]),
    "no_error": lambda out, _: no_error_score(out["message"]),
    "tool_param_equals": lambda out, val: tool_param_equals_score(out["tool_calls"], val),
    "tool_param_contains": lambda out, val: tool_param_contains_score(out["tool_calls"], val),
    "no_tools_used": lambda out, _: no_tools_used_score(out["tool_calls"]),
    "min_tool_calls": lambda out, val: min_tool_calls_score(out["tool_calls"], val),
}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def assertion_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Run every assertion and return a single overall_pass score.

    Per-assertion details are included in the comment field so LangSmith
    shows one clean column instead of a sparse matrix.
    """
    agent_output = _get_agent_output(run)
    assertions = (example.outputs or {}).get("assertions", [])

    details = []
    scores = []

    for assertion in assertions:
        a_type = assertion["type"]
        a_value = assertion.get("value", "")
        label = "{t}:{v}".format(t=a_type, v=a_value) if a_value else a_type
        scorer = _SCORER_DISPATCH.get(a_type)

        if scorer is None:
            scores.append(0.0)
            details.append("FAIL {lab} (unknown type)".format(lab=label))
            continue

        try:
            score = scorer(agent_output, a_value)
        except Exception as exc:
            score = 0.0
            details.append("FAIL {lab} ({err})".format(lab=label, err=exc))
            scores.append(score)
            continue

        scores.append(score)
        status = "PASS" if score == 1.0 else "FAIL"
        details.append("{s} {lab}".format(s=status, lab=label))

    passed = len([s for s in scores if s == 1.0])
    total = len(scores)
    all_passed = passed == total and total > 0

    return EvaluationResult(
        key="overall_pass",
        score=1.0 if all_passed else 0.0,
        comment="{p}/{t} passed: {d}".format(
            p=passed, t=total, d=" | ".join(details)
        ),
    )


def tool_match_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Check if the exact set of tools called matches the expected set."""
    agent_output = _get_agent_output(run)
    expected_tools = set((example.outputs or {}).get("expected_tools", []))
    actual_tools = {tc.get("tool_name") for tc in agent_output.get("tool_calls", [])}

    # If no expected tools defined, consider it a match
    match = expected_tools == actual_tools if expected_tools else True

    return EvaluationResult(
        key="tool_match",
        score=1.0 if match else 0.0,
        comment="Expected: {e}, Got: {a}".format(
            e=sorted(expected_tools), a=sorted(actual_tools)
        ),
    )


def latency_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Score latency against rubric targets.

    - Single-tool / easy/medium: < 5 s
    - Multi-step / hard:          < 15 s
    """
    if run.end_time and run.start_time:
        latency = (run.end_time - run.start_time).total_seconds()
    else:
        latency = 0.0

    difficulty = (example.metadata or {}).get("difficulty", "easy")
    threshold = 15.0 if difficulty == "hard" else 5.0

    return EvaluationResult(
        key="latency",
        score=1.0 if latency <= threshold else 0.0,
        comment="{lat:.1f}s (threshold: {thr:.0f}s)".format(lat=latency, thr=threshold),
    )


def no_error_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Check that the agent response does not contain error indicators."""
    agent_output = _get_agent_output(run)
    score = no_error_score(agent_output["message"])
    return EvaluationResult(key="no_error", score=score)


def category_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Surface the example category as a categorical feedback column.

    Uses ``value`` (string) rather than ``score`` (numeric) so LangSmith
    treats it as a filterable/groupable categorical column in the
    Experiments UI.
    """
    category = (example.metadata or {}).get("category", "unknown")
    return EvaluationResult(key="category", value=category)


def difficulty_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Surface the example difficulty as a categorical feedback column."""
    difficulty = (example.metadata or {}).get("difficulty", "unknown")
    return EvaluationResult(key="difficulty", value=difficulty)
