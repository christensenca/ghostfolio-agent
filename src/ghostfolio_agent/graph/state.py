"""LangGraph agent state definition."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Agent state with LangChain message reducer for proper message handling."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    jwt: str
    tool_results: List[Dict[str, Any]]
    pending_tool_calls: list
    final_answer: str
    confidence: float
    holdings_context: str
    planned_tools: List[str]
