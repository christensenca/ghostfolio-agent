#!/usr/bin/env python3
"""Import mock fixture data into Ghostfolio via the API.

Usage:
    # With access token (auto-authenticates):
    python import_mock_data.py --api-url http://localhost:3333 --access-token "YOUR_SECURITY_TOKEN"

    # With JWT directly:
    python import_mock_data.py --api-url http://localhost:3333 --jwt "eyJhbG..."

    # Dry run:
    python import_mock_data.py --api-url http://localhost:3333 --access-token "..." --dry-run

    # Specific file:
    python import_mock_data.py --api-url http://localhost:3333 --access-token "..." --file output/buy_and_hold.json

    # Using env var (GHOSTFOLIO_ACCESS_TOKEN):
    export GHOSTFOLIO_ACCESS_TOKEN="YOUR_SECURITY_TOKEN"
    python import_mock_data.py --api-url http://localhost:3333
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).parent / "output"


def get_jwt(api_url: str, access_token: str) -> str:
    """Exchange a Ghostfolio security token for a JWT."""
    resp = requests.post(
        f"{api_url}/api/v1/auth/anonymous",
        json={"accessToken": access_token},
        timeout=10,
    )
    if resp.status_code != 201:
        print(f"Auth failed (HTTP {resp.status_code}): {resp.text[:200]}")
        sys.exit(1)
    jwt = resp.json().get("authToken")
    if not jwt:
        print(f"Auth response missing authToken: {resp.text[:200]}")
        sys.exit(1)
    return jwt


def import_file(api_url: str, jwt: str, filepath: Path, dry_run: bool) -> bool:
    """Import a single JSON fixture file into Ghostfolio."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Importing {filepath.name}...")

    with open(filepath) as f:
        data = json.load(f)

    num_accounts = len(data.get("accounts", []))
    num_activities = len(data.get("activities", []))
    print(f"  Payload: {num_accounts} accounts, {num_activities} activities")

    url = f"{api_url}/api/v1/import"
    if dry_run:
        url += "?dryRun=true"

    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=data, headers=headers, timeout=60)

    if resp.status_code == 201:
        result = resp.json()
        activities = result.get("activities", [])
        duplicates = sum(1 for a in activities if a.get("error", {}).get("code") == "IS_DUPLICATE")
        imported = len(activities) - duplicates
        print(f"  OK: {imported} imported, {duplicates} duplicates skipped")
        return True
    else:
        print(f"  FAILED (HTTP {resp.status_code}): {resp.text[:500]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Import mock data into Ghostfolio")
    parser.add_argument("--api-url", default="http://localhost:3333", help="Ghostfolio API base URL (default: http://localhost:3333)")
    parser.add_argument("--access-token", type=str, help="Ghostfolio security token (auto-exchanges for JWT). Also reads GHOSTFOLIO_ACCESS_TOKEN env var.")
    parser.add_argument("--jwt", type=str, help="JWT token directly (skip auto-auth)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without importing")
    parser.add_argument("--file", type=str, help="Import a specific file (default: all_accounts.json)")
    args = parser.parse_args()

    # Resolve JWT: --jwt flag > --access-token flag > env var
    jwt = args.jwt
    if not jwt:
        access_token = args.access_token or os.environ.get("GHOSTFOLIO_ACCESS_TOKEN")
        if not access_token:
            print("Error: provide --jwt, --access-token, or set GHOSTFOLIO_ACCESS_TOKEN env var")
            sys.exit(1)
        print(f"Authenticating with Ghostfolio at {args.api_url}...")
        jwt = get_jwt(args.api_url, access_token)
        print("  OK (JWT obtained)")

    if args.file:
        filepath = Path(args.file)
        if not filepath.is_absolute():
            filepath = OUTPUT_DIR / filepath.name
    else:
        filepath = OUTPUT_DIR / "all_accounts.json"

    if not filepath.exists():
        print(f"File not found: {filepath}")
        print("Run generate_mock_data.py first to create fixture files.")
        sys.exit(1)

    success = import_file(args.api_url, jwt, filepath, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
