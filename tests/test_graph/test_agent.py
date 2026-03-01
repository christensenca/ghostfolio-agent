"""Tests for LangGraph agent."""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage


def _make_mock_llm_response(content: str):
    """Create a mock AIMessage response from the LLM."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = []
    msg.additional_kwargs = {}
    return msg


@pytest.mark.asyncio
async def test_agent_returns_response_for_simple_query():
    """Agent should return a text response for a simple greeting."""
    from ghostfolio_agent.graph.agent import create_agent

    agent = create_agent()

    # Mock _get_llm so plan_node gets a mock LLM that returns empty plan
    mock_llm = AsyncMock()
    plan_response = _make_mock_llm_response("[]")

    with (
        patch("ghostfolio_agent.graph.nodes._get_llm", return_value=mock_llm),
        patch("ghostfolio_agent.graph.nodes.call_llm", new_callable=AsyncMock) as mock_call,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=plan_response)
        mock_call.return_value = {
            "content": "Hello! I can help you with your portfolio.",
        }

        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content="Hi there")],
                "jwt": "fake-jwt",
                "tool_results": [],
                "pending_tool_calls": [],
                "final_answer": "",
                "confidence": 1.0,
            }
        )

    assert result["final_answer"] != ""
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_agent_calls_tool_when_needed():
    """Agent should invoke a tool when the LLM requests one."""
    from ghostfolio_agent.graph.agent import create_agent

    agent = create_agent()

    call_count = 0

    async def mock_llm_side_effect(state):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "portfolio_analysis", "args": {}, "id": "call_1"}
                ],
            )
            return {
                "ai_message": ai_msg,
                "tool_calls": ai_msg.tool_calls,
            }
        return {
            "content": "Your portfolio has 5 holdings.",
        }

    mock_llm = AsyncMock()
    plan_response = _make_mock_llm_response('["portfolio_analysis"]')

    with (
        patch("ghostfolio_agent.graph.nodes._get_llm", return_value=mock_llm),
        patch("ghostfolio_agent.graph.nodes.call_llm", new_callable=AsyncMock) as mock_call,
        patch("ghostfolio_agent.graph.nodes.execute_tool", new_callable=AsyncMock) as mock_tool,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=plan_response)
        mock_call.side_effect = mock_llm_side_effect
        mock_tool.return_value = {
            "tool_name": "portfolio_analysis",
            "result": {"holdings": [], "total_value": 0},
        }

        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Show my portfolio holdings")
                ],
                "jwt": "fake-jwt",
                "tool_results": [],
                "pending_tool_calls": [],
                "final_answer": "",
                "confidence": 1.0,
            }
        )

    assert result["final_answer"] != ""
    assert len(result["tool_results"]) > 0


@pytest.mark.asyncio
async def test_agent_handles_empty_tool_result():
    """Agent should handle gracefully when a tool returns empty data."""
    from ghostfolio_agent.graph.agent import create_agent

    agent = create_agent()

    call_count = 0

    async def mock_llm_side_effect(state):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "portfolio_analysis", "args": {}, "id": "call_1"}
                ],
            )
            return {
                "ai_message": ai_msg,
                "tool_calls": ai_msg.tool_calls,
            }
        return {
            "content": "Your portfolio appears to be empty.",
        }

    mock_llm = AsyncMock()
    plan_response = _make_mock_llm_response('["portfolio_analysis"]')

    with (
        patch("ghostfolio_agent.graph.nodes._get_llm", return_value=mock_llm),
        patch("ghostfolio_agent.graph.nodes.call_llm", new_callable=AsyncMock) as mock_call,
        patch("ghostfolio_agent.graph.nodes.execute_tool", new_callable=AsyncMock) as mock_tool,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=plan_response)
        mock_call.side_effect = mock_llm_side_effect
        mock_tool.return_value = {
            "tool_name": "portfolio_analysis",
            "result": {"holdings": [], "total_value": 0},
        }

        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content="Show my portfolio")],
                "jwt": "fake-jwt",
                "tool_results": [],
                "pending_tool_calls": [],
                "final_answer": "",
                "confidence": 1.0,
            }
        )

    assert result["final_answer"] != ""
