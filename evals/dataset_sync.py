"""Sync eval_cases.yaml to a LangSmith dataset.

Provides deterministic UUID mapping so re-syncs update existing
examples rather than creating duplicates.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import yaml
from langsmith import Client

EVAL_CASES_FILE = Path(__file__).parent / "eval_cases.yaml"
DEFAULT_DATASET_NAME = "ghostfolio-agent-evals"


def _deterministic_uuid(case_id: str) -> str:
    """Generate a stable UUID from a case ID so re-syncs update, not duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"agentforge-eval-{case_id}"))


def load_cases(
    category: str | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    """Load and optionally filter eval cases from YAML."""
    with open(EVAL_CASES_FILE) as f:
        cases = yaml.safe_load(f)
    if category:
        cases = [c for c in cases if c.get("category") == category]
    if difficulty:
        cases = [c for c in cases if c.get("difficulty") == difficulty]
    return cases


def sync_dataset(
    cases: list[dict],
    dataset_name: str = DEFAULT_DATASET_NAME,
    client: Optional[Client] = None,
) -> str:
    """Upsert YAML cases into a LangSmith dataset. Returns dataset name."""
    if client is None:
        client = Client()

    # Create or get existing dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Ghostfolio agent eval suite. Source: eval_cases.yaml",
        )

    # Build examples with deterministic IDs
    inputs_list = []
    outputs_list = []
    metadata_list = []
    example_ids = []

    for case in cases:
        inputs_list.append({
            "question": case["question"],
            "case_id": case["id"],
        })
        outputs_list.append({
            "expected_tools": case.get("expected_tools", []),
            "assertions": case.get("assertions", []),
        })
        metadata_list.append({
            "case_id": case["id"],
            "category": case.get("category", ""),
            "difficulty": case.get("difficulty", ""),
        })
        example_ids.append(_deterministic_uuid(case["id"]))

    # Split into existing (update) vs new (create) by checking the dataset
    existing_ids = set()
    try:
        for ex in client.list_examples(dataset_id=dataset.id):
            existing_ids.add(str(ex.id))
    except Exception:
        pass

    create_idx = []
    update_idx = []
    for i, eid in enumerate(example_ids):
        if eid in existing_ids:
            update_idx.append(i)
        else:
            create_idx.append(i)

    # Create new examples
    if create_idx:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[inputs_list[i] for i in create_idx],
            outputs=[outputs_list[i] for i in create_idx],
            metadata=[metadata_list[i] for i in create_idx],
            ids=[example_ids[i] for i in create_idx],
        )

    # Update existing examples
    if update_idx:
        client.update_examples(
            example_ids=[example_ids[i] for i in update_idx],
            inputs=[inputs_list[i] for i in update_idx],
            outputs=[outputs_list[i] for i in update_idx],
            metadata=[metadata_list[i] for i in update_idx],
        )

    # Remove stale examples no longer in YAML
    target_ids = set(example_ids)
    stale_ids = existing_ids - target_ids
    if stale_ids:
        client.delete_examples(example_ids=list(stale_ids))

    return dataset_name
