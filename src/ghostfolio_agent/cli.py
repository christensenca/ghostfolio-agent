"""CLI chat interface for the Ghostfolio AI Agent."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid

from ghostfolio_agent.config import settings


def _setup_tracing():
    """Configure LangSmith tracing if credentials provided."""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        if settings.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project


async def _resolve_jwt() -> str:
    """Resolve JWT: direct JWT > exchange security token."""
    if settings.ghostfolio_jwt:
        return settings.ghostfolio_jwt

    from ghostfolio_agent.auth import AuthenticationError, exchange_token

    if not settings.ghostfolio_access_token:
        print(
            "Error: No Ghostfolio authentication configured.\n"
            "Provide via --token flag, GHOSTFOLIO_ACCESS_TOKEN env var, or .env file."
        )
        sys.exit(1)

    try:
        return await exchange_token(
            settings.ghostfolio_api_url,
            settings.ghostfolio_access_token,
        )
    except AuthenticationError as e:
        print("Authentication failed: {e}".format(e=e))
        sys.exit(1)


async def _chat_loop():
    """Main async chat loop."""
    from langchain_core.messages import HumanMessage

    from ghostfolio_agent.db import close_db, init_db
    from ghostfolio_agent.graph.agent import create_agent
    from ghostfolio_agent.memory.store import conversation_store
    from ghostfolio_agent.verification import verify_response

    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()

    # Initialize
    _setup_tracing()
    jwt = await _resolve_jwt()
    await init_db()
    agent = create_agent()

    console.print(
        "[dim]Connected to Ghostfolio at {url}[/dim]".format(
            url=settings.ghostfolio_api_url
        )
    )
    console.print('[dim]Type "quit" or "exit" to stop. Ctrl+C also works.\n[/dim]')

    conversation_id = None

    try:
        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            conversation_id, history = await conversation_store.get_or_create(
                conversation_id
            )
            await conversation_store.add_message(
                conversation_id, "user", user_input
            )

            run_id = str(uuid.uuid4())
            initial_state = {
                "messages": history + [HumanMessage(content=user_input)],
                "jwt": jwt,
                "tool_results": [],
                "pending_tool_calls": [],
                "final_answer": "",
                "confidence": 1.0,
            }

            with console.status("[bold cyan]Thinking...[/bold cyan]"):
                try:
                    result = await asyncio.wait_for(
                        agent.ainvoke(
                            initial_state,
                            config={
                                "recursion_limit": settings.max_tool_steps * 2 + 2,
                                "run_id": run_id,
                            },
                        ),
                        timeout=settings.max_response_seconds,
                    )
                except asyncio.TimeoutError:
                    console.print(
                        "[red]Request timed out after {s}s[/red]".format(
                            s=settings.max_response_seconds
                        )
                    )
                    continue
                except Exception as e:
                    console.print("[red]Error: {e}[/red]".format(e=e))
                    continue

            # Show tool calls
            tool_results = result.get("tool_results", [])
            if tool_results:
                tool_names = [tr.get("tool_name", "?") for tr in tool_results]
                console.print(
                    "[dim]  Tools used: {t}[/dim]".format(t=", ".join(tool_names))
                )

            # Get answer
            answer = result.get("final_answer", "").strip()
            if not answer:
                answer = "Sorry, I could not complete the request. Please try again."

            # Verify and add disclaimer
            verification = verify_response(
                answer=answer,
                tool_results=tool_results,
                user_message=user_input,
            )
            if verification and verification.confidence < 0.5:
                answer += (
                    "\n\n*Note: Some information in this response could not "
                    "be fully verified against the source data.*"
                )

            console.print()
            console.print(Markdown(answer))
            console.print()

            if verification:
                console.print(
                    "[dim]  Confidence: {c:.0%} ({l})[/dim]".format(
                        c=verification.confidence,
                        l=verification.confidence_label,
                    )
                )

            await conversation_store.add_message(
                conversation_id, "assistant", answer
            )

    except KeyboardInterrupt:
        console.print()
    finally:
        await close_db()
        console.print("[dim]Goodbye![/dim]")


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="AI-powered financial assistant for Ghostfolio"
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (or set OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--token",
        help="Ghostfolio security token (or set GHOSTFOLIO_ACCESS_TOKEN)",
    )
    parser.add_argument(
        "--url",
        help="Ghostfolio API URL (default: http://localhost:3333)",
    )
    parser.add_argument(
        "--model",
        help="LLM model ID (default: anthropic/claude-haiku-4.5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Max seconds per request (default: 180)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show debug logging (LLM calls, tool execution)",
    )
    args = parser.parse_args()

    # CLI flags override env/settings
    if args.api_key:
        settings.openrouter_api_key = args.api_key
    if args.token:
        settings.ghostfolio_access_token = args.token
    if args.url:
        settings.ghostfolio_api_url = args.url
    if args.model:
        settings.llm_model = args.model

    if args.timeout:
        settings.max_response_seconds = args.timeout
    elif settings.max_response_seconds < 180:
        settings.max_response_seconds = 180

    if not settings.openrouter_api_key:
        print(
            "Error: No OpenRouter API key configured.\n"
            "Provide via --api-key flag, OPENROUTER_API_KEY env var, or .env file."
        )
        sys.exit(1)

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    asyncio.run(_chat_loop())


if __name__ == "__main__":
    main()
