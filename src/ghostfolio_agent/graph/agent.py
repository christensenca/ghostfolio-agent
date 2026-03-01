"""LangGraph ReAct agent definition."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from ghostfolio_agent.graph.state import AgentState
from ghostfolio_agent.graph.nodes import plan_node, reason_node, act_node, should_continue


def create_agent() -> StateGraph:
    """Create and compile the LangGraph ReAct agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)

    # Set entry point — plan first, then reason
    graph.set_entry_point("plan")
    graph.add_edge("plan", "reason")

    # Add conditional edges from reason node
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "act": "act",
            "finish": END,
        },
    )

    # After acting, go back to reason (ReAct loop)
    graph.add_edge("act", "reason")

    return graph.compile()
