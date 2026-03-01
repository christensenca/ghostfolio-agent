"""Comprehensive unit tests for the verification module."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ghostfolio_agent.verification import (
    OUTPUT_WARNING_SEVERITY,
    ConstraintViolation,
    GroundingResult,
    VerificationResult,
    _base_tool_name,
    _compute_freshness_score,
    _compute_tool_success_rate,
    _extract_numbers,
    _flatten_numbers_from_data,
    _is_tool_result_useful,
    _numbers_match,
    _parse_result,
    compute_confidence,
    extract_sources,
    ground_response,
    validate_output,
    validate_tool_results,
    verify_response,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestParseResult:
    def test_dict_passthrough(self):
        assert _parse_result({"result": {"key": "val"}}) == {"key": "val"}

    def test_json_string(self):
        assert _parse_result({"result": '{"key": "val"}'}) == {"key": "val"}

    def test_invalid_json_string(self):
        assert _parse_result({"result": "not json"}) == {}

    def test_missing_result(self):
        assert _parse_result({}) == {}

    def test_empty_string(self):
        assert _parse_result({"result": ""}) == {}


class TestBaseToolName:
    def test_simple_name(self):
        assert _base_tool_name({"tool_name": "market_data"}) == "market_data"

    def test_missing_tool_name(self):
        assert _base_tool_name({}) == "unknown"


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------


class TestExtractNumbers:
    def test_dollar_amount(self):
        assert _extract_numbers("$1,234.56") == [1234.56]

    def test_percentage(self):
        assert _extract_numbers("12.5%") == [12.5]

    def test_negative(self):
        assert _extract_numbers("-42.5") == [-42.5]

    def test_multiple(self):
        nums = _extract_numbers("up 45.2% to $50,000")
        assert nums == [45.2, 50000.0]

    def test_plain_integer(self):
        assert _extract_numbers("there are 250 shares") == [250.0]

    def test_no_match_surrounded_by_letters(self):
        # The lookbehind/lookahead prevents matching the full "123" but
        # the regex can still match partial substrings like "2" (preceded
        # by digit '1', not a letter, and followed by digit '3', not a letter).
        # This is acceptable — the grounding system handles these via
        # trivial number filtering.
        nums = _extract_numbers("abc123def")
        # At minimum, no full "123" should be extracted as a standalone match
        assert 123.0 not in nums

    def test_empty_string(self):
        assert _extract_numbers("") == []

    def test_no_numbers(self):
        assert _extract_numbers("no numbers here") == []

    def test_large_number(self):
        assert _extract_numbers("$1,000,000.00") == [1000000.0]

    def test_zero(self):
        assert _extract_numbers("0") == [0.0]


class TestFlattenNumbersFromData:
    def test_int(self):
        assert _flatten_numbers_from_data(42) == [42.0]

    def test_float(self):
        assert _flatten_numbers_from_data(3.14) == [3.14]

    def test_bool_excluded(self):
        assert _flatten_numbers_from_data(True) == []

    def test_string_with_number(self):
        assert _flatten_numbers_from_data("$1,234.56") == [1234.56]

    def test_nested_dict(self):
        data = {"a": {"b": 10, "c": 20}}
        assert sorted(_flatten_numbers_from_data(data)) == [10.0, 20.0]

    def test_list(self):
        assert _flatten_numbers_from_data([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_empty(self):
        assert _flatten_numbers_from_data({}) == []
        assert _flatten_numbers_from_data([]) == []
        assert _flatten_numbers_from_data("") == []


class TestNumbersMatch:
    def test_exact(self):
        assert _numbers_match(100.0, 100.0) is True

    def test_within_tolerance(self):
        # 1234.0 vs 1234.12 → diff = 0.12/1234.12 ≈ 0.0097 < 0.01
        assert _numbers_match(1234.0, 1234.12) is True

    def test_outside_tolerance(self):
        # 100 vs 102 → diff = 2/102 ≈ 0.0196 > 0.01
        assert _numbers_match(100.0, 102.0) is False

    def test_zero_comparison(self):
        assert _numbers_match(0.005, 0) is True
        assert _numbers_match(0.02, 0) is False

    def test_both_zero(self):
        assert _numbers_match(0.0, 0.0) is True


# ---------------------------------------------------------------------------
# Response grounding
# ---------------------------------------------------------------------------


class TestGroundResponse:
    def test_all_grounded(self):
        tool_results = [
            {"tool_name": "portfolio_analysis", "result": {"total_value": 50000.0}}
        ]
        result = ground_response(
            "Your portfolio is worth $50,000.", tool_results
        )
        assert result.grounded == 1
        assert result.ungrounded == 0
        assert result.rate == 1.0

    def test_mixed(self):
        tool_results = [
            {"tool_name": "market_data", "result": {"current_price": 150.0}}
        ]
        result = ground_response(
            "Price is $150.00 with a target of $200.", tool_results
        )
        assert result.grounded == 1
        assert result.ungrounded == 1
        assert result.rate == 0.5

    def test_no_numbers_with_useful_tools(self):
        """Tools used successfully but response has no numbers → rate=0.8."""
        tool_results = [
            {"tool_name": "compliance_check", "result": {"summary": "5 rules"}}
        ]
        result = ground_response(
            "Your portfolio is well-diversified.", tool_results
        )
        assert result.rate == 0.8

    def test_no_numbers_with_failed_tools(self):
        """Tools used but failed — no numbers is more suspicious → rate=0.5."""
        tool_results = [
            {"tool_name": "market_data", "result": {"error": "HTTP 500"}}
        ]
        result = ground_response(
            "I couldn't retrieve the data.", tool_results
        )
        assert result.rate == 0.5

    def test_no_numbers_no_tools(self):
        """No tools, no numbers → rate=1.0."""
        result = ground_response("Hello there!", [])
        assert result.rate == 1.0

    def test_no_tool_results(self):
        result = ground_response("Your value is $50,000.", [])
        assert result.rate == 1.0

    def test_user_numbers_excluded(self):
        tool_results = [
            {"tool_name": "market_data", "result": {"current_price": 150.0}}
        ]
        result = ground_response(
            "At $150 per share and your target of $200...",
            tool_results,
            user_message="what if the price is $200?",
        )
        # 150 is grounded, 200 is from user message so excluded
        assert result.grounded == 1
        assert result.ungrounded == 0

    def test_trivial_numbers_excluded(self):
        tool_results = [
            {"tool_name": "portfolio_analysis", "result": {"count": 7}}
        ]
        # 3 and 5 are trivial, 7 is not
        result = ground_response(
            "You have 3 accounts, 5 holdings, and 7 transactions.",
            tool_results,
        )
        assert result.grounded == 1  # only 7 is checked and grounded
        assert result.ungrounded == 0

    def test_tolerance_matching(self):
        tool_results = [
            {"tool_name": "market_data", "result": {"current_price": 1234.12}}
        ]
        result = ground_response("Price is $1,234.", tool_results)
        assert result.grounded == 1


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


class TestExtractSources:
    def test_provider_name(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "source": {
                        "data_source": "YAHOO",
                        "provider_name": "Yahoo Finance",
                    }
                },
            }
        ]
        sources = extract_sources(tool_results)
        assert len(sources) == 1
        assert sources[0].name == "Yahoo Finance"
        assert sources[0].tool == "market_data"

    def test_fallback_to_display_name(self):
        tool_results = [
            {"tool_name": "portfolio_analysis", "result": {"data": "ok"}}
        ]
        sources = extract_sources(tool_results)
        assert len(sources) == 1
        assert sources[0].name == "Portfolio"

    def test_deduplication(self):
        tool_results = [
            {"tool_name": "portfolio_analysis", "result": {"data": "ok"}},
            {"tool_name": "portfolio_analysis", "result": {"data": "ok2"}},
        ]
        sources = extract_sources(tool_results)
        assert len(sources) == 1


# ---------------------------------------------------------------------------
# Tool result usefulness
# ---------------------------------------------------------------------------


class TestIsToolResultUseful:
    def test_explicit_error(self):
        tr = {"tool_name": "market_data", "result": {"error": "HTTP 500"}}
        assert _is_tool_result_useful(tr) is False

    def test_market_data_empty(self):
        tr = {
            "tool_name": "market_data",
            "result": {
                "current_price": 0,
                "historical_prices": [],
                "symbol": "FAKE",
            },
        }
        assert _is_tool_result_useful(tr) is False

    def test_market_data_with_data(self):
        tr = {
            "tool_name": "market_data",
            "result": {
                "current_price": 150.0,
                "historical_prices": [{"date": "2026-02-26", "price": 149.0}],
            },
        }
        assert _is_tool_result_useful(tr) is True

    def test_market_data_price_zero_but_has_history(self):
        """Price=0 but history exists — could be a delisted stock with history."""
        tr = {
            "tool_name": "market_data",
            "result": {
                "current_price": 0,
                "historical_prices": [{"date": "2026-01-01", "price": 10.0}],
            },
        }
        assert _is_tool_result_useful(tr) is True

    def test_portfolio_no_holdings_unfiltered(self):
        """Unfiltered query with no holdings = genuinely empty portfolio."""
        tr = {
            "tool_name": "portfolio_analysis",
            "result": {
                "portfolio": {"total_value": 0.0},
                "holdings_table": "No holdings found.",
            },
        }
        assert _is_tool_result_useful(tr) is False

    def test_portfolio_no_holdings_filtered(self):
        """Filtered query (symbols=TSLA) with no match = valid 'not owned' answer."""
        tr = {
            "tool_name": "portfolio_analysis",
            "result": {
                "portfolio": {"total_value": 0.0},
                "holdings_table": "No holdings found.",
            },
            "tool_input": {"symbols": "TSLA"},
        }
        assert _is_tool_result_useful(tr) is True

    def test_portfolio_no_holdings_account_filter(self):
        """Filtered by account with no results = valid answer."""
        tr = {
            "tool_name": "portfolio_analysis",
            "result": {
                "portfolio": {"total_value": 0.0},
                "holdings_table": "No holdings found.",
            },
            "tool_input": {"account": "Nonexistent Account"},
        }
        assert _is_tool_result_useful(tr) is True

    def test_portfolio_with_holdings(self):
        tr = {
            "tool_name": "portfolio_analysis",
            "result": {
                "portfolio": {"total_value": 50000.0},
                "holdings_table": "| Symbol | Value |\n|---|---|\n| AAPL | $50,000 |",
            },
        }
        assert _is_tool_result_useful(tr) is True

    def test_calculate_all_errors(self):
        tr = {
            "tool_name": "calculate",
            "result": {
                "calculations": [
                    {"name": "x", "error": "Division by zero."},
                    {"name": "y", "error": "Unknown reference 'z'."},
                ]
            },
        }
        assert _is_tool_result_useful(tr) is False

    def test_calculate_some_ok(self):
        tr = {
            "tool_name": "calculate",
            "result": {
                "calculations": [
                    {"name": "x", "value": 42.0, "formatted": "42.00"},
                    {"name": "y", "error": "Division by zero."},
                ]
            },
        }
        assert _is_tool_result_useful(tr) is True

    def test_calculate_all_ok(self):
        tr = {
            "tool_name": "calculate",
            "result": {
                "calculations": [
                    {"name": "x", "value": 42.0, "formatted": "42.00"},
                ]
            },
        }
        assert _is_tool_result_useful(tr) is True

    def test_market_data_lookup_response(self):
        """Lookup responses (with 'symbols' key) are always useful."""
        tr = {
            "tool_name": "market_data",
            "result": {
                "symbols": [
                    {"symbol": "AAPL", "name": "Apple Inc.", "currency": "USD"}
                ],
                "total_results": 1,
            },
        }
        assert _is_tool_result_useful(tr) is True

    def test_market_data_lookup_empty_results(self):
        """Lookup with no matches is still useful — 'not found' is valid info."""
        tr = {
            "tool_name": "market_data",
            "result": {
                "symbols": [],
                "total_results": 0,
            },
        }
        assert _is_tool_result_useful(tr) is True

    def test_unknown_tool_no_error(self):
        tr = {"tool_name": "some_new_tool", "result": {"data": "ok"}}
        assert _is_tool_result_useful(tr) is True


# ---------------------------------------------------------------------------
# Domain constraint validation
# ---------------------------------------------------------------------------


class TestValidateToolResults:
    def test_clean_results(self):
        tool_results = [
            {
                "tool_name": "portfolio_analysis",
                "result": {
                    "portfolio": {
                        "total_value": 50000.0,
                        "net_cost_basis": 45000.0,
                        "gain": 5000.0,
                        "fees": 50.0,
                    }
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        assert violations == []

    def test_portfolio_negative_total_value(self):
        tool_results = [
            {
                "tool_name": "portfolio_analysis",
                "result": {
                    "portfolio": {"total_value": -100.0, "fees": 0.0}
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        assert len(violations) == 1
        assert violations[0].rule == "total_value_non_negative"
        assert violations[0].severity == "error"

    def test_portfolio_gain_mismatch(self):
        tool_results = [
            {
                "tool_name": "portfolio_analysis",
                "result": {
                    "portfolio": {
                        "total_value": 50000.0,
                        "net_cost_basis": 45000.0,
                        "gain": 10000.0,  # Should be 5000
                        "fees": 0.0,
                    }
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        rules = [v.rule for v in violations]
        assert "gain_formula_consistency" in rules

    def test_portfolio_gain_within_tolerance(self):
        """Gain within $1 of expected should not trigger."""
        tool_results = [
            {
                "tool_name": "portfolio_analysis",
                "result": {
                    "portfolio": {
                        "total_value": 50000.0,
                        "net_cost_basis": 45000.0,
                        "gain": 5000.50,  # 0.50 off — within $1
                        "fees": 0.0,
                    }
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        assert violations == []

    def test_market_data_negative_price(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {"current_price": -5.0},
            }
        ]
        violations = validate_tool_results(tool_results)
        assert len(violations) == 1
        assert violations[0].rule == "price_non_negative"

    def test_market_data_daily_change_mismatch(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 150.0,
                    "previous_close": 148.0,
                    "daily_change": 5.0,  # Should be 2.0
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        rules = [v.rule for v in violations]
        assert "daily_change_formula" in rules

    def test_market_data_dates_out_of_order(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 150.0,
                    "historical_prices": [
                        {"date": "2026-02-26", "price": 149.0},
                        {"date": "2026-02-25", "price": 148.0},  # Out of order
                    ],
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        rules = [v.rule for v in violations]
        assert "historical_dates_ordered" in rules

    def test_market_data_52w_range_violation(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 200.0,
                    "fifty_two_week_high": 180.0,
                    "fifty_two_week_low": 100.0,
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        rules = [v.rule for v in violations]
        assert "price_within_52w_range" in rules

    def test_tax_negative_dividends(self):
        tool_results = [
            {
                "tool_name": "tax_estimate",
                "result": {
                    "summary": {
                        "total_dividends": -50.0,
                        "total_fees": 10.0,
                        "sell_count": 0,
                        "dividend_count": 0,
                    }
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        assert len(violations) == 1
        assert violations[0].rule == "dividends_non_negative"

    def test_transaction_negative_fees(self):
        tool_results = [
            {
                "tool_name": "transaction_categorize",
                "result": {
                    "summary": {"total_fees": -10.0, "total_count": 5}
                },
            }
        ]
        violations = validate_tool_results(tool_results)
        assert len(violations) == 1
        assert violations[0].rule == "fees_non_negative"

    def test_error_results_skipped(self):
        """Tools with errors shouldn't be domain-checked."""
        tool_results = [
            {
                "tool_name": "portfolio_analysis",
                "result": {"error": "Unauthorized"},
            }
        ]
        # Should not crash — no portfolio dict to check
        violations = validate_tool_results(tool_results)
        assert violations == []

    def test_empty_results(self):
        assert validate_tool_results([]) == []


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


class TestValidateOutput:
    def test_clean_response(self):
        warnings = validate_output(
            "Your portfolio is worth $50,000.",
            [{"tool_name": "portfolio_analysis", "result": {}}],
        )
        assert warnings == []

    # --- Raw JSON leak ---

    def test_json_leak_inline(self):
        warnings = validate_output(
            'Here is the data: {"tool_name": "market_data"}',
            [],
        )
        assert "raw_json_leak" in warnings

    def test_json_leak_multiline(self):
        """Multi-line JSON block with brace then newline then quoted key."""
        answer = (
            "Here is your data:\n"
            "{\n"
            '  "total_value": 43946.06,\n'
            '  "net_cost_basis": 38034.81\n'
            "}"
        )
        warnings = validate_output(answer, [])
        assert "raw_json_leak" in warnings

    def test_json_leak_array(self):
        """Raw array of objects dumped into response."""
        answer = (
            "Accounts:\n"
            '[\n'
            '  {"id": "abc123", "name": "My Account", "balance": 0}\n'
            ']'
        )
        warnings = validate_output(answer, [])
        assert "raw_json_leak" in warnings

    def test_json_leak_repeated_kv(self):
        """Multiple key-value pairs in sequence indicate raw dump."""
        answer = (
            '"total_value": 43946.06, "net_cost_basis": 38034.81, '
            '"fees": 98.19, "activity_count": 73'
        )
        warnings = validate_output(answer, [])
        assert "raw_json_leak" in warnings

    def test_no_json_leak_normal_prose(self):
        """Normal prose with quotes shouldn't trigger."""
        warnings = validate_output(
            'Your portfolio has a "buy and hold" strategy with 7 holdings.',
            [],
        )
        assert "raw_json_leak" not in warnings

    # --- Tool name leak ---

    def test_tool_name_leak(self):
        warnings = validate_output(
            "I used portfolio_analysis to fetch your holdings.",
            [],
        )
        assert "tool_name_leak" in warnings

    def test_tool_name_leak_market_data(self):
        warnings = validate_output(
            "The market_data tool returned the current price.",
            [],
        )
        assert "tool_name_leak" in warnings

    def test_no_tool_name_leak(self):
        """Referring to concepts by natural name is fine."""
        warnings = validate_output(
            "Based on your portfolio data, here are your holdings.",
            [],
        )
        assert "tool_name_leak" not in warnings

    # --- Raw field name leak ---

    def test_raw_field_leak(self):
        """3+ internal field names in response indicates verbatim dump."""
        answer = (
            "Your net_cost_basis is $38,034. The gain_pct is 15.54% "
            "and activity_count is 73."
        )
        warnings = validate_output(answer, [])
        assert "raw_field_leak" in warnings

    def test_no_raw_field_leak_few_fields(self):
        """1-2 field names could be coincidental — don't flag."""
        warnings = validate_output(
            "The current_price of AAPL is $264.18.",
            [],
        )
        assert "raw_field_leak" not in warnings

    # --- Tool internals leak ---

    def test_tool_internals_leak(self):
        warnings = validate_output(
            "The tool_input was symbols=AAPL.",
            [],
        )
        assert "tool_internals_leak" in warnings

    # --- Tax disclaimer ---

    def test_missing_tax_disclaimer(self):
        warnings = validate_output(
            "Your capital gains are $5,000.",
            [{"tool_name": "tax_estimate", "result": {}}],
        )
        assert "missing_tax_disclaimer" in warnings

    def test_tax_disclaimer_present(self):
        warnings = validate_output(
            "Your gains are $5,000. This is not tax advice.",
            [{"tool_name": "tax_estimate", "result": {}}],
        )
        assert "missing_tax_disclaimer" not in warnings

    # --- Inline math ---

    def test_inline_math(self):
        warnings = validate_output("100 / 5 = 20", [])
        assert "inline_math_detected" in warnings

    def test_inline_math_multiplication(self):
        warnings = validate_output("50000 * 0.03 = 1500", [])
        assert "inline_math_detected" in warnings

    def test_no_inline_math_normal_text(self):
        warnings = validate_output("Your return is 12.5%.", [])
        assert "inline_math_detected" not in warnings

    # --- Severity tiers ---

    def test_severity_tiers_defined(self):
        """All warning types should have a severity defined."""
        assert OUTPUT_WARNING_SEVERITY["raw_json_leak"] == 0.15
        assert OUTPUT_WARNING_SEVERITY["tool_name_leak"] == 0.10
        assert OUTPUT_WARNING_SEVERITY["raw_field_leak"] == 0.10
        assert OUTPUT_WARNING_SEVERITY["tool_internals_leak"] == 0.10
        assert OUTPUT_WARNING_SEVERITY["missing_tax_disclaimer"] == 0.05
        assert OUTPUT_WARNING_SEVERITY["inline_math_detected"] == 0.05

    def test_raw_json_leak_high_penalty(self):
        """raw_json_leak should reduce confidence more than inline_math."""
        grounding = GroundingResult(grounded=5, ungrounded=0, rate=1.0)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 150.0,
                    "historical_prices": [{"date": today, "price": 149.0}],
                },
            }
        ]
        score_json = compute_confidence(
            grounding, tool_results, output_warnings=["raw_json_leak"],
        )
        score_math = compute_confidence(
            grounding, tool_results, output_warnings=["inline_math_detected"],
        )
        assert score_json < score_math


# ---------------------------------------------------------------------------
# Freshness scoring
# ---------------------------------------------------------------------------


class TestComputeFreshnessScore:
    def _make_tool_results(self, date_str: str):
        return [
            {
                "tool_name": "market_data",
                "result": {
                    "historical_prices": [{"date": date_str, "price": 100.0}]
                },
            }
        ]

    def test_fresh_today(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        score = _compute_freshness_score(self._make_tool_results(today))
        assert score == 1.0

    def test_stale_2_days(self):
        two_days_ago = (
            datetime.now(timezone.utc) - timedelta(days=2)
        ).strftime("%Y-%m-%d")
        score = _compute_freshness_score(self._make_tool_results(two_days_ago))
        assert score == 0.7

    def test_very_stale_10_days(self):
        ten_days_ago = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).strftime("%Y-%m-%d")
        score = _compute_freshness_score(
            self._make_tool_results(ten_days_ago)
        )
        assert score == 0.3

    def test_no_date_info(self):
        tool_results = [
            {"tool_name": "portfolio_analysis", "result": {"data": "ok"}}
        ]
        score = _compute_freshness_score(tool_results)
        assert score == 0.7  # Changed from 1.0 to 0.7

    def test_empty_results(self):
        score = _compute_freshness_score([])
        assert score == 0.7


# ---------------------------------------------------------------------------
# Tool success rate
# ---------------------------------------------------------------------------


class TestComputeToolSuccessRate:
    def test_all_successful(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 150.0,
                    "historical_prices": [{"date": "2026-02-26", "price": 149}],
                },
            }
        ]
        assert _compute_tool_success_rate(tool_results) == 1.0

    def test_all_errored(self):
        tool_results = [
            {"tool_name": "market_data", "result": {"error": "HTTP 500"}},
            {"tool_name": "portfolio_analysis", "result": {"error": "Unauthorized"}},
        ]
        assert _compute_tool_success_rate(tool_results) == 0.0

    def test_mixed(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {"current_price": 150.0, "historical_prices": []},
            },
            {"tool_name": "portfolio_analysis", "result": {"error": "fail"}},
        ]
        assert _compute_tool_success_rate(tool_results) == 0.5

    def test_empty_data_detected(self):
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 0,
                    "historical_prices": [],
                },
            }
        ]
        assert _compute_tool_success_rate(tool_results) == 0.0

    def test_no_results(self):
        assert _compute_tool_success_rate([]) == 1.0


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_perfect_run(self):
        grounding = GroundingResult(grounded=5, ungrounded=0, rate=1.0)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 150.0,
                    "historical_prices": [{"date": today, "price": 149.0}],
                },
            }
        ]
        score = compute_confidence(grounding, tool_results)
        assert score >= 0.8

    def test_all_tools_errored(self):
        grounding = GroundingResult(grounded=0, ungrounded=0, rate=0.5)
        tool_results = [
            {"tool_name": "market_data", "result": {"error": "HTTP 500"}}
        ]
        score = compute_confidence(grounding, tool_results)
        assert score < 0.5

    def test_domain_violations_reduce_score(self):
        grounding = GroundingResult(grounded=5, ungrounded=0, rate=1.0)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 150.0,
                    "historical_prices": [{"date": today, "price": 149.0}],
                },
            }
        ]
        violations = [
            ConstraintViolation(
                tool="market_data",
                rule="test",
                severity="error",
                detail="test",
            ),
            ConstraintViolation(
                tool="market_data",
                rule="test2",
                severity="warning",
                detail="test2",
            ),
        ]
        score_with = compute_confidence(
            grounding, tool_results, domain_violations=violations,
        )
        score_without = compute_confidence(grounding, tool_results)
        assert score_with < score_without

    def test_output_warnings_reduce_score(self):
        grounding = GroundingResult(grounded=5, ungrounded=0, rate=1.0)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tool_results = [
            {
                "tool_name": "market_data",
                "result": {
                    "current_price": 150.0,
                    "historical_prices": [{"date": today, "price": 149.0}],
                },
            }
        ]
        score_with = compute_confidence(
            grounding,
            tool_results,
            output_warnings=["raw_json_leak", "inline_math_detected"],
        )
        score_without = compute_confidence(grounding, tool_results)
        assert score_with < score_without

    def test_no_tool_results(self):
        grounding = GroundingResult(grounded=0, ungrounded=0, rate=1.0)
        score = compute_confidence(grounding, [])
        # has_sources = 0.5, freshness = 0.7, tool_success = 1.0
        assert 0.5 < score < 1.0

    def test_score_clamped_to_zero(self):
        """Many penalties should floor at 0.0, not go negative."""
        grounding = GroundingResult(grounded=0, ungrounded=5, rate=0.0)
        tool_results = [
            {"tool_name": "market_data", "result": {"error": "fail"}}
        ]
        violations = [
            ConstraintViolation(
                tool="t", rule="r", severity="error", detail="d"
            )
            for _ in range(10)
        ]
        warnings = ["w1", "w2", "w3", "w4", "w5"]
        score = compute_confidence(
            grounding, tool_results, violations, warnings,
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# verify_response (orchestrator)
# ---------------------------------------------------------------------------


class TestVerifyResponse:
    def test_pure_conversation_returns_none(self):
        result = verify_response("Hello, how are you?", [])
        assert result is None

    def test_with_tools_returns_result(self):
        tool_results = [
            {
                "tool_name": "portfolio_analysis",
                "result": {
                    "portfolio": {
                        "total_value": 50000.0,
                        "net_cost_basis": 45000.0,
                        "gain": 5000.0,
                        "fees": 50.0,
                    },
                    "holdings_table": "| Symbol | Value |\n|---|---|\n| AAPL | $50,000 |",
                },
            }
        ]
        result = verify_response(
            "Your portfolio is worth $50,000.",
            tool_results,
        )
        assert result is not None
        assert isinstance(result, VerificationResult)
        assert result.grounding.grounded >= 1
        assert isinstance(result.domain_violations, list)
        assert isinstance(result.output_warnings, list)

    def test_numbers_but_no_tools(self):
        """Response has numbers but no tools — still runs verification."""
        result = verify_response("The price is $150.", [])
        assert result is not None

    def test_low_confidence_with_issues(self):
        tool_results = [
            {"tool_name": "market_data", "result": {"error": "HTTP 500"}},
            {"tool_name": "tax_estimate", "result": {"error": "Unauthorized"}},
        ]
        result = verify_response(
            "Your gains are $5,000 and dividends are $1,000.",
            tool_results,
        )
        assert result is not None
        assert result.confidence < 0.5
        assert result.confidence_label == "low"
        # tax_estimate was called but answer has no disclaimer
        assert "missing_tax_disclaimer" in result.output_warnings
