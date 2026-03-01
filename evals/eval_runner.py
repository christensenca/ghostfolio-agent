#!/usr/bin/env python3
"""Eval runner for the Ghostfolio AI Agent.

Loads eval cases from YAML, runs each question through the agent, and
scores with deterministic evaluators. By default runs locally (no
LangSmith traces). Use --langsmith to sync results to LangSmith.

Usage:
    # Auto-setup (create user, import data) + run evals:
    python -m evals.eval_runner --setup

    # With access token (auto-authenticates):
    python -m evals.eval_runner --access-token "YOUR_SECURITY_TOKEN"

    # With env var:
    export GHOSTFOLIO_ACCESS_TOKEN="YOUR_SECURITY_TOKEN"
    python -m evals.eval_runner

    # With JWT directly:
    python -m evals.eval_runner --jwt "eyJhbG..."

    # Filter by category:
    python -m evals.eval_runner --category happy_path

    # Filter by difficulty:
    python -m evals.eval_runner --difficulty easy

    # Dry run (just print cases, don't call agent):
    python -m evals.eval_runner --dry-run

    # Local only (no LangSmith traces — fast iteration):
    python -m evals.eval_runner --local --category multi_step

    # Custom experiment name (sends to LangSmith):
    python -m evals.eval_runner --experiment-prefix "v2-prompt"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import requests

from config import settings
from evals.dataset_sync import load_cases, sync_dataset


def _setup_langsmith():
    """Configure LangSmith env vars and return imports needed for aevaluate."""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    from langsmith import Client, aevaluate
    from evals.langsmith_evaluators import (
        assertion_evaluator,
        category_evaluator,
        difficulty_evaluator,
        latency_evaluator,
        no_error_evaluator,
        tool_match_evaluator,
    )
    return {
        "Client": Client,
        "aevaluate": aevaluate,
        "evaluators": [
            assertion_evaluator,
            tool_match_evaluator,
            latency_evaluator,
            no_error_evaluator,
            category_evaluator,
            difficulty_evaluator,
        ],
    }


def _disable_langsmith_tracing():
    """Disable LangSmith/LangChain tracing for local-only runs."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.pop("LANGCHAIN_API_KEY", None)

CREDENTIALS_FILE = Path(__file__).parent.parent / "scripts" / ".eval_credentials.json"
DEFAULT_API_URL = "http://localhost:3333"
DATASET_NAME = "ghostfolio-agent-evals"


def get_jwt(api_url: str, access_token: str) -> str:
    """Exchange a Ghostfolio security token for a JWT."""
    resp = requests.post(
        "{url}/api/v1/auth/anonymous".format(url=api_url),
        json={"accessToken": access_token},
        timeout=10,
    )
    if resp.status_code != 201:
        print("Auth failed (HTTP {code}): {body}".format(
            code=resp.status_code, body=resp.text[:200]
        ))
        sys.exit(1)
    jwt = resp.json().get("authToken")
    if not jwt:
        print("Auth response missing authToken: {body}".format(body=resp.text[:200]))
        sys.exit(1)
    return jwt


def make_target(jwt: str, collected_results: list):
    """Create the async target function that aevaluate() will call.

    The target receives ``inputs`` from the dataset example and must return
    a dict that becomes ``run.outputs`` for evaluators to inspect.

    Results are also appended to ``collected_results`` for local scoring.
    """

    async def target(inputs: dict) -> dict:
        from main import run_agent

        question = inputs["question"]
        case_id = inputs.get("case_id", "eval-unknown")

        try:
            result = await run_agent(
                message=question,
                conversation_id="eval-{cid}".format(cid=case_id),
                jwt=jwt,
                history=[],
            )
        except Exception as exc:
            result = {
                "message": "Agent error: {e}".format(e=exc),
                "conversation_id": "eval-{cid}".format(cid=case_id),
                "tool_calls": [],
                "confidence": 0.0,
            }

        output = {
            "message": result.get("message", ""),
            "tool_calls": result.get("tool_calls", []),
            "confidence": result.get("confidence", 0.0),
        }

        collected_results.append({"inputs": inputs, "outputs": output})
        return output

    return target


async def run_evals_local(
    jwt: str,
    category: str | None,
    difficulty: str | None,
) -> None:
    """Run evals locally — no LangSmith traces or dataset sync."""
    from main import run_agent

    # Disable AFTER importing main (main.py sets tracing=true at module level)
    _disable_langsmith_tracing()

    cases = load_cases(category=category, difficulty=difficulty)
    print("Loaded {n} eval cases from YAML".format(n=len(cases)))
    print("Running locally (no LangSmith traces)\n")

    collected_results = []
    for i, case in enumerate(cases, 1):
        cid = case["id"]
        question = case["question"]
        print("  [{i}/{n}] {cid}...".format(i=i, n=len(cases), cid=cid), end="", flush=True)

        try:
            result = await run_agent(
                message=question,
                conversation_id="eval-{cid}".format(cid=cid),
                jwt=jwt,
                history=[],
            )
        except Exception as exc:
            result = {
                "message": "Agent error: {e}".format(e=exc),
                "conversation_id": "eval-{cid}".format(cid=cid),
                "tool_calls": [],
                "confidence": 0.0,
            }

        output = {
            "message": result.get("message", ""),
            "tool_calls": result.get("tool_calls", []),
            "confidence": result.get("confidence", 0.0),
        }
        collected_results.append({"inputs": {"question": question, "case_id": cid}, "outputs": output})
        print(" done")

    _print_local_summary(cases, collected_results)


async def run_evals(
    jwt: str,
    category: str | None,
    difficulty: str | None,
    experiment_prefix: str | None = None,
) -> None:
    """Run evals via LangSmith aevaluate()."""
    ls = _setup_langsmith()
    Client = ls["Client"]
    aevaluate = ls["aevaluate"]
    evaluators = ls["evaluators"]

    # Load and filter cases from YAML
    cases = load_cases(category=category, difficulty=difficulty)
    print("Loaded {n} eval cases from YAML".format(n=len(cases)))

    # Always sync ALL cases to one dataset (LangSmith example IDs are global,
    # so we can't create per-filter datasets with the same deterministic IDs).
    all_cases = load_cases()
    print("Syncing all {n} cases to LangSmith dataset...".format(n=len(all_cases)))
    client = Client()
    dataset_name = sync_dataset(all_cases, dataset_name=DATASET_NAME, client=client)

    # Filter to just the requested examples for aevaluate
    from evals.dataset_sync import _deterministic_uuid
    filtered_ids = {_deterministic_uuid(c["id"]) for c in cases}
    ds = client.read_dataset(dataset_name=dataset_name)
    all_examples = list(client.list_examples(dataset_id=ds.id))
    examples = [ex for ex in all_examples if str(ex.id) in filtered_ids]
    print("  Dataset '{name}': running {n}/{total} examples".format(
        name=dataset_name, n=len(examples), total=len(all_examples)
    ))

    # Build experiment prefix
    if not experiment_prefix:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filters = []
        if category:
            filters.append("cat={c}".format(c=category))
        if difficulty:
            filters.append("diff={d}".format(d=difficulty))
        filter_str = "-{f}".format(f="-".join(filters)) if filters else ""
        experiment_prefix = "ghostfolio-eval{flt}-{ts}".format(flt=filter_str, ts=timestamp)

    # Run aevaluate
    print("Starting evaluation: {pfx}".format(pfx=experiment_prefix))
    print("Results will appear in LangSmith Experiments UI\n")

    collected_results = []
    target = make_target(jwt, collected_results)

    results = await aevaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata={
            "category_filter": category or "all",
            "difficulty_filter": difficulty or "all",
            "num_cases": len(cases),
            "agent_model": "anthropic/claude-sonnet-4",
        },
        max_concurrency=1,  # Sequential to avoid state conflicts
        client=client,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("Experiment: {pfx}".format(pfx=experiment_prefix))
    print("View results at: https://smith.langchain.com/")
    print("=" * 60)

    # Local scoring (independent of LangSmith upload)
    _print_local_summary(cases, collected_results)


def _print_local_summary(cases: list, collected_results: list) -> None:
    """Score all results locally and print a summary to stdout."""
    from evals.scorers import evaluate_assertions

    # Build a lookup from case_id to collected output
    output_map = {}
    for cr in collected_results:
        cid = cr["inputs"].get("case_id", "")
        output_map[cid] = cr["outputs"]

    total = 0
    passed = 0
    failed_cases = []

    for case in cases:
        cid = case["id"]
        output = output_map.get(cid)
        if output is None:
            continue

        total += 1
        assertions = case.get("assertions", [])
        expected_tools = set(case.get("expected_tools", []))
        actual_tools = {tc.get("tool_name") for tc in output.get("tool_calls", [])}

        # Score assertions
        scored = evaluate_assertions(output, assertions)
        assertion_pass = all(s["score"] == 1.0 for s in scored)

        # Score tool match
        tool_match = expected_tools == actual_tools if expected_tools else True

        case_pass = assertion_pass and tool_match
        if case_pass:
            passed += 1
        else:
            details = []
            for s in scored:
                label = "{t}:{v}".format(t=s["type"], v=s.get("value", "")) if s.get("value") else s["type"]
                status = "PASS" if s["score"] == 1.0 else "FAIL"
                details.append("{st} {lb}".format(st=status, lb=label))
            if not tool_match:
                details.append("FAIL tool_match (expected={e}, got={a})".format(
                    e=sorted(expected_tools), a=sorted(actual_tools)
                ))
            failed_cases.append({
                "case_id": cid,
                "question": case["question"][:80],
                "category": case.get("category", ""),
                "difficulty": case.get("difficulty", ""),
                "details": " | ".join(details),
                "message": output.get("message", "")[:200],
            })

    print("\n" + "=" * 70)
    print("LOCAL SCORING SUMMARY")
    print("=" * 70)
    print("  Total: {t} | Passed: {p} | Failed: {f} | Pass rate: {r:.1%}".format(
        t=total, p=passed, f=total - passed, r=passed / total if total else 0
    ))

    if failed_cases:
        print("\n  FAILING CASES:")
        for fc in failed_cases:
            print("\n  [{cid}] ({cat}/{diff}) {q}".format(
                cid=fc["case_id"], cat=fc["category"], diff=fc["difficulty"],
                q=fc["question"]
            ))
            print("    {d}".format(d=fc["details"]))
            print("    Response: {m}...".format(m=fc["message"]))
    else:
        print("\n  All cases passed!")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run Ghostfolio agent evals via LangSmith")
    parser.add_argument(
        "--access-token",
        type=str,
        help="Ghostfolio security token (auto-exchanges for JWT). Also reads GHOSTFOLIO_ACCESS_TOKEN env var.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Ghostfolio API base URL (default: {url})".format(url=DEFAULT_API_URL),
    )
    parser.add_argument("--jwt", type=str, help="JWT token directly (skip auto-auth)")
    parser.add_argument("--category", type=str, help="Filter by category (happy_path, edge_case, adversarial, multi_step)")
    parser.add_argument("--difficulty", type=str, help="Filter by difficulty (easy, medium, hard)")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Bootstrap eval environment (create user, import data) before running",
    )
    parser.add_argument("--dry-run", action="store_true", help="Just print cases")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally without LangSmith traces (faster, no trace costs)",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        help="LangSmith experiment prefix (auto-generated if omitted)",
    )
    args = parser.parse_args()

    # Run bootstrap if requested
    if args.setup:
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts.setup_eval", "--api-url", args.api_url],
        )
        if result.returncode != 0:
            print("Setup failed!")
            sys.exit(1)

    # Dry run — print cases and exit
    cases = load_cases(category=args.category, difficulty=args.difficulty)

    if args.dry_run:
        print("Found {n} eval cases:\n".format(n=len(cases)))
        for c in cases:
            tools = ", ".join(c.get("expected_tools", [])) or "(none)"
            n_assertions = len(c.get("assertions", []))
            print("  [{diff}] {cid}: {q}".format(
                diff=c["difficulty"], cid=c["id"], q=c["question"]
            ))
            print("    Tools: {t} | Assertions: {n} | Category: {cat}".format(
                t=tools, n=n_assertions, cat=c["category"]
            ))
        return

    # Resolve JWT: --jwt flag > --access-token flag > env var > credentials file
    jwt = args.jwt
    if not jwt:
        access_token = args.access_token or os.environ.get("GHOSTFOLIO_ACCESS_TOKEN")

        # Fall back to saved credentials from setup_eval
        if not access_token and CREDENTIALS_FILE.exists():
            try:
                with open(CREDENTIALS_FILE) as f:
                    creds = json.load(f)
                access_token = creds.get("accessToken")
                print("Using saved credentials from {path}".format(path=CREDENTIALS_FILE))
            except (json.JSONDecodeError, OSError):
                pass

        if not access_token:
            print(
                "Error: no credentials found. Provide one of:\n"
                "  --jwt TOKEN\n"
                "  --access-token TOKEN\n"
                "  GHOSTFOLIO_ACCESS_TOKEN env var\n"
                "  --setup (auto-creates user and imports data)"
            )
            sys.exit(1)
        print("Authenticating with Ghostfolio at {url}...".format(url=args.api_url))
        jwt = get_jwt(args.api_url, access_token)
        print("  OK (JWT obtained)")

    if args.local:
        asyncio.run(run_evals_local(jwt, args.category, args.difficulty))
    else:
        asyncio.run(run_evals(jwt, args.category, args.difficulty, args.experiment_prefix))


if __name__ == "__main__":
    main()
