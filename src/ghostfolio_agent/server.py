"""FastAPI server for the Ghostfolio AI Agent.

Optional mode: install with `pip install ghostfolio-agent[server]`
Run with: python -m ghostfolio_agent.server
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from ghostfolio_agent.config import settings
from ghostfolio_agent.db import close_db, init_db
from ghostfolio_agent.graph.agent import create_agent
from ghostfolio_agent.memory.store import conversation_store
from ghostfolio_agent.schemas.requests import ChatRequest
from ghostfolio_agent.schemas.responses import (
    ChatResponse,
    ConstraintViolationInfo,
    GroundingInfo,
    SourceInfo,
    ToolCallInfo,
    VerificationInfo,
)
from ghostfolio_agent.verification import verify_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LangSmith tracing before anything else so graph + LLM calls are captured
if settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)


async def _resolve_request_jwt(request_jwt: str | None) -> str:
    """Resolve JWT from request body, settings, or token exchange."""
    if request_jwt:
        return request_jwt
    if settings.ghostfolio_jwt:
        return settings.ghostfolio_jwt
    if settings.ghostfolio_access_token:
        from ghostfolio_agent.auth import exchange_token

        return await exchange_token(
            settings.ghostfolio_api_url,
            settings.ghostfolio_access_token,
        )
    raise HTTPException(
        status_code=401,
        detail=(
            "No JWT or access token configured. "
            "Set GHOSTFOLIO_ACCESS_TOKEN or pass jwt in request."
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(title="Ghostfolio AI Agent", version="0.1.0", lifespan=lifespan)

# Compile the LangGraph agent once at startup
agent = create_agent()


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return "event: {event}\ndata: {data}\n\n".format(
        event=event, data=json.dumps(data)
    )


async def run_agent(
    message: str, conversation_id: str, jwt: str, history: list
) -> dict:
    """Run the LangGraph ReAct agent (non-streaming)."""
    run_id = str(uuid.uuid4())
    initial_state = {
        "messages": history + [HumanMessage(content=message)],
        "jwt": jwt,
        "tool_results": [],
        "pending_tool_calls": [],
        "final_answer": "",
        "confidence": 1.0,
    }

    fallback_msg = "Sorry, I could not complete the request. Please try again."

    try:
        recursion_limit = settings.max_tool_steps * 2 + 2
        result = await asyncio.wait_for(
            agent.ainvoke(
                initial_state,
                config={"recursion_limit": recursion_limit, "run_id": run_id},
            ),
            timeout=settings.max_response_seconds,
        )

        tool_calls = []
        for tr in result.get("tool_results", []):
            tool_output = tr.get("result", "")
            if isinstance(tool_output, dict):
                tool_output = json.dumps(tool_output)
            tool_calls.append(
                {
                    "tool_name": tr.get("tool_name", "unknown"),
                    "tool_input": tr.get("tool_input", {}),
                    "tool_output": tool_output,
                }
            )

        answer = result.get("final_answer", "").strip()
        verification = verify_response(
            answer=answer or fallback_msg,
            tool_results=result.get("tool_results", []),
            user_message=message,
        )
        confidence = verification.confidence if verification else (0.7 if answer else 0.0)

        final_message = answer or fallback_msg
        if answer and verification and verification.confidence < 0.5:
            final_message += (
                "\n\n*Note: Some information in this response could not "
                "be fully verified against the source data.*"
            )

        return {
            "message": final_message,
            "conversation_id": conversation_id,
            "tool_calls": tool_calls,
            "confidence": confidence if answer else 0.0,
            "verification": verification,
            "run_id": run_id,
        }
    except asyncio.TimeoutError:
        logger.error(
            "Agent timed out after %ds for conversation %s",
            settings.max_response_seconds,
            conversation_id,
        )
        return {
            "message": fallback_msg,
            "conversation_id": conversation_id,
            "tool_calls": [],
            "confidence": 0.0,
            "run_id": run_id,
        }
    except Exception:
        logger.exception("Agent error for conversation %s", conversation_id)
        return {
            "message": fallback_msg,
            "conversation_id": conversation_id,
            "tool_calls": [],
            "confidence": 0.0,
            "run_id": run_id,
        }


async def stream_agent(
    message: str, conversation_id: str, jwt: str, history: list
) -> AsyncGenerator[str, None]:
    """Stream LangGraph agent events as SSE."""
    run_id = str(uuid.uuid4())
    initial_state = {
        "messages": history + [HumanMessage(content=message)],
        "jwt": jwt,
        "tool_results": [],
        "pending_tool_calls": [],
        "final_answer": "",
        "confidence": 1.0,
    }

    fallback_msg = "Sorry, I could not complete the request. Please try again."
    full_answer = ""
    graph_final_answer = ""
    tool_calls = []
    tool_results_seen = 0
    collected_tool_results = []
    confidence = 0.0

    try:
        recursion_limit = settings.max_tool_steps * 2 + 1
        deadline = asyncio.get_event_loop().time() + settings.max_response_seconds

        async for event in agent.astream_events(
            initial_state,
            version="v2",
            config={"recursion_limit": recursion_limit, "run_id": run_id},
        ):
            if asyncio.get_event_loop().time() > deadline:
                logger.error(
                    "Stream timed out after %ds for conversation %s",
                    settings.max_response_seconds,
                    conversation_id,
                )
                yield _sse_event("error", {"message": fallback_msg})
                return

            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node == "plan":
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_answer += chunk.content
                    yield _sse_event("token", {"content": chunk.content})

            elif kind == "on_chain_start" and name == "act":
                pending = event.get("data", {}).get("input", {}).get(
                    "pending_tool_calls", []
                )
                for tc in pending:
                    yield _sse_event(
                        "tool_start", {"tool_name": tc.get("name", "unknown")}
                    )

            elif kind == "on_chain_end" and name == "act":
                results = event.get("data", {}).get("output", {}).get(
                    "tool_results", []
                )
                new_results = results[tool_results_seen:]
                tool_results_seen = len(results)
                for tr in new_results:
                    collected_tool_results.append(tr)
                    tool_name = tr.get("tool_name", "unknown")
                    tool_output = tr.get("result", "")
                    if isinstance(tool_output, dict):
                        tool_output = json.dumps(tool_output)
                    tool_calls.append(
                        {
                            "tool_name": tool_name,
                            "tool_input": tr.get("tool_input", {}),
                            "tool_output": tool_output,
                        }
                    )
                    yield _sse_event("tool_end", {"tool_name": tool_name})

            elif kind == "on_chain_end" and name == "reason":
                output = event.get("data", {}).get("output", {})
                fa = output.get("final_answer", "")
                if fa:
                    graph_final_answer = fa

        answer = full_answer.strip()

        if not answer and graph_final_answer:
            answer = graph_final_answer.strip()
            if answer:
                yield _sse_event("token", {"content": answer})

        if not answer:
            yield _sse_event("token", {"content": fallback_msg})
            answer = fallback_msg

        verification = verify_response(
            answer=answer,
            tool_results=collected_tool_results,
            user_message=message,
        )
        confidence = verification.confidence if verification else 0.7

        if answer == fallback_msg:
            confidence = 0.0

        if (
            answer != fallback_msg
            and verification
            and verification.confidence < 0.5
        ):
            disclaimer = (
                "\n\n*Note: Some information in this response could not "
                "be fully verified against the source data.*"
            )
            answer += disclaimer
            yield _sse_event("token", {"content": disclaimer})

        await conversation_store.add_message(conversation_id, "assistant", answer)

        done_data = {
            "conversation_id": conversation_id,
            "tool_calls": tool_calls,
            "confidence": confidence,
            "run_id": run_id,
        }
        if verification:
            done_data["verification"] = verification.model_dump()

        yield _sse_event("done", done_data)
    except Exception:
        logger.exception("Stream error for conversation %s", conversation_id)
        yield _sse_event("error", {"message": fallback_msg})


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    jwt = await _resolve_request_jwt(request.jwt)

    conversation_id, history = await conversation_store.get_or_create(
        request.conversation_id
    )

    await conversation_store.add_message(conversation_id, "user", request.message)

    result = await run_agent(
        message=request.message,
        conversation_id=conversation_id,
        jwt=jwt,
        history=history,
    )

    await conversation_store.add_message(
        conversation_id, "assistant", result["message"]
    )

    verification = result.get("verification")
    verification_info = None
    if verification:
        verification_info = VerificationInfo(
            confidence=verification.confidence,
            confidence_label=verification.confidence_label,
            grounding=GroundingInfo(
                grounded=verification.grounding.grounded,
                ungrounded=verification.grounding.ungrounded,
                rate=verification.grounding.rate,
            ),
            sources=[
                SourceInfo(name=s.name, tool=s.tool)
                for s in verification.sources
            ],
            domain_violations=[
                ConstraintViolationInfo(
                    tool=v.tool, rule=v.rule,
                    severity=v.severity, detail=v.detail,
                )
                for v in verification.domain_violations
            ],
            output_warnings=verification.output_warnings,
        )

    return ChatResponse(
        message=result["message"],
        conversation_id=result["conversation_id"],
        tool_calls=[ToolCallInfo(**tc) for tc in result.get("tool_calls", [])],
        confidence=result.get("confidence", 1.0),
        verification=verification_info,
        run_id=result.get("run_id"),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    jwt = await _resolve_request_jwt(request.jwt)

    conversation_id, history = await conversation_store.get_or_create(
        request.conversation_id
    )

    await conversation_store.add_message(conversation_id, "user", request.message)

    return StreamingResponse(
        stream_agent(
            message=request.message,
            conversation_id=conversation_id,
            jwt=jwt,
            history=history,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
