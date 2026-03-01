"""Calculator tool — deterministic financial arithmetic.

Provides named, chainable operations so the LLM never has to do math
in its head. Supports basic arithmetic, percentages, and aggregation.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ghostfolio_agent.tools.base import GhostfolioTool


class CalculateTool(GhostfolioTool):
    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return (
            "Performs deterministic financial calculations. Use this for any "
            "arithmetic the user needs: computing tax offsets, percentages, "
            "projections, share counts, or comparisons. Provide a list of named "
            "operations that execute sequentially. Each operation can reference "
            "results of previous operations by name via the 'ref' field."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "description": "List of calculation operations to perform sequentially.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": (
                                    "Name for this result (can be referenced "
                                    "in later operations via 'ref')."
                                ),
                            },
                            "op": {
                                "type": "string",
                                "enum": [
                                    "add", "subtract", "multiply", "divide",
                                    "percent", "sum", "min", "max", "abs",
                                    "negate", "round",
                                ],
                                "description": "Operation to perform.",
                            },
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": (
                                    "Numeric inputs. For 'ref' values, use the "
                                    "ref field instead."
                                ),
                            },
                            "ref": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Reference names of previous operation results "
                                    "to use as inputs (appended after 'values')."
                                ),
                            },
                            "label": {
                                "type": "string",
                                "description": (
                                    "Optional human-readable label for this result."
                                ),
                            },
                        },
                        "required": ["name", "op"],
                    },
                },
            },
            "required": ["operations"],
        }

    async def execute(self, jwt: str, **kwargs) -> Dict[str, Any]:
        operations = kwargs.get("operations", [])
        if not operations:
            return {
                "tool_name": self.name,
                "result": {"error": "No operations provided."},
            }

        results: Dict[str, float] = {}
        steps: List[Dict[str, Any]] = []

        for op_def in operations:
            name = op_def.get("name", "unnamed")
            op = op_def.get("op", "")
            label = op_def.get("label", "")

            # Resolve values: literal numbers + referenced results
            values = list(op_def.get("values") or [])
            for ref_name in (op_def.get("ref") or []):
                if ref_name in results:
                    values.append(results[ref_name])
                else:
                    steps.append({
                        "name": name,
                        "error": "Unknown reference '{r}'.".format(r=ref_name),
                    })
                    continue

            result = self._compute(op, values)

            if isinstance(result, str):
                # Error string
                steps.append({"name": name, "label": label, "error": result})
            else:
                results[name] = result
                step_entry: Dict[str, Any] = {
                    "name": name,
                    "value": round(result, 4),
                    "formatted": _fmt_number(result),
                }
                if label:
                    step_entry["label"] = label
                steps.append(step_entry)

        return {"tool_name": self.name, "result": {"calculations": steps}}

    @staticmethod
    def _compute(op: str, values: List[float]) -> float | str:
        """Execute a single operation. Returns float or error string."""
        if op == "add":
            if len(values) < 2:
                return "add requires at least 2 values."
            return values[0] + values[1]
        elif op == "subtract":
            if len(values) < 2:
                return "subtract requires at least 2 values."
            return values[0] - values[1]
        elif op == "multiply":
            if len(values) < 2:
                return "multiply requires at least 2 values."
            return values[0] * values[1]
        elif op == "divide":
            if len(values) < 2:
                return "divide requires at least 2 values."
            if values[1] == 0:
                return "Division by zero."
            return values[0] / values[1]
        elif op == "percent":
            if len(values) < 2:
                return "percent requires [amount, rate]."
            return values[0] * values[1] / 100
        elif op == "sum":
            return sum(values)
        elif op == "min":
            if not values:
                return "min requires at least 1 value."
            return min(values)
        elif op == "max":
            if not values:
                return "max requires at least 1 value."
            return max(values)
        elif op == "abs":
            if not values:
                return "abs requires 1 value."
            return abs(values[0])
        elif op == "negate":
            if not values:
                return "negate requires 1 value."
            return -values[0]
        elif op == "round":
            if not values:
                return "round requires at least 1 value."
            decimals = int(values[1]) if len(values) > 1 else 2
            return round(values[0], decimals)
        else:
            return "Unknown operation '{op}'.".format(op=op)


def _fmt_number(val: float) -> str:
    """Format number with commas. Use $ prefix for large values."""
    if abs(val) >= 1:
        return "{:,.2f}".format(val)
    return "{:.4f}".format(val)
