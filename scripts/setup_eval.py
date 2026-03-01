#!/usr/bin/env python3
"""Bootstrap the eval environment: create user + import mock data.

Usage:
    python -m scripts.setup_eval
    python -m scripts.setup_eval --api-url http://localhost:3333
    python -m scripts.setup_eval --force   # Re-create even if credentials exist
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

CREDENTIALS_FILE = Path(__file__).parent / ".eval_credentials.json"
FIXTURES_FILE = (
    Path(__file__).parent.parent / "fixtures" / "output" / "all_accounts.json"
)
DEFAULT_API_URL = "http://localhost:3333"
MAX_HEALTH_WAIT_SECONDS = 120


def wait_for_ghostfolio(api_url: str) -> None:
    """Poll /api/v1/health until OK, with exponential backoff."""
    delay = 1.0
    elapsed = 0.0
    print(f"Waiting for Ghostfolio at {api_url}...")

    while elapsed < MAX_HEALTH_WAIT_SECONDS:
        try:
            resp = requests.get(f"{api_url}/api/v1/health", timeout=5)
            if resp.status_code == 200:
                print("  Ghostfolio is healthy.")
                return
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass

        time.sleep(delay)
        elapsed += delay
        delay = min(delay * 2, 10.0)

    print(
        f"ERROR: Ghostfolio not reachable at {api_url} after {MAX_HEALTH_WAIT_SECONDS}s. "
        "Is it running?"
    )
    sys.exit(1)


def load_credentials() -> dict | None:
    """Load saved credentials from disk, or None."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        with open(CREDENTIALS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_credentials(creds: dict) -> None:
    """Persist credentials to disk."""
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(creds, f, indent=2)


def try_authenticate(api_url: str, access_token: str) -> str | None:
    """Try to exchange access token for JWT. Returns JWT or None."""
    try:
        resp = requests.post(
            f"{api_url}/api/v1/auth/anonymous",
            json={"accessToken": access_token},
            timeout=10,
        )
        if resp.status_code == 201:
            return resp.json().get("authToken")
    except requests.RequestException:
        pass
    return None


def create_user(api_url: str) -> dict:
    """Create a new anonymous user via the Ghostfolio API."""
    resp = requests.post(f"{api_url}/api/v1/user", timeout=10)

    if resp.status_code == 403:
        print(
            "ERROR: User signup is disabled. This usually means the DB "
            "already has users with signup disabled. Check the Property table "
            "or try with --force after resetting the DB."
        )
        sys.exit(1)

    if resp.status_code != 201:
        print(f"ERROR: Failed to create user (HTTP {resp.status_code}): {resp.text[:300]}")
        sys.exit(1)

    data = resp.json()
    return {
        "accessToken": data["accessToken"],
        "authToken": data["authToken"],
        "role": data.get("role", "USER"),
    }


def check_has_activities(api_url: str, jwt: str) -> bool:
    """Check if the user already has imported activities."""
    try:
        resp = requests.get(
            f"{api_url}/api/v1/order",
            headers={"Authorization": f"Bearer {jwt}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            activities = data.get("activities", data) if isinstance(data, dict) else data
            if isinstance(activities, list):
                return len(activities) > 0
    except requests.RequestException:
        pass
    return False


def import_mock_data(api_url: str, jwt: str) -> None:
    """Import all_accounts.json via the Ghostfolio API."""
    if not FIXTURES_FILE.exists():
        print(f"ERROR: Fixture file not found: {FIXTURES_FILE}")
        print("Run generate_mock_data.py first to create fixture files.")
        sys.exit(1)

    with open(FIXTURES_FILE) as f:
        payload = json.load(f)

    num_accounts = len(payload.get("accounts", []))
    num_activities = len(payload.get("activities", []))
    print(f"  Payload: {num_accounts} accounts, {num_activities} activities")

    resp = requests.post(
        f"{api_url}/api/v1/import",
        json=payload,
        headers={
            "Authorization": f"Bearer {jwt}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )

    if resp.status_code == 201:
        result = resp.json()
        activities = result.get("activities", [])
        duplicates = sum(
            1
            for a in activities
            if a.get("error", {}).get("code") == "IS_DUPLICATE"
        )
        imported = len(activities) - duplicates
        print(f"  OK: {imported} imported, {duplicates} duplicates skipped")
    else:
        print(f"  FAILED (HTTP {resp.status_code}): {resp.text[:500]}")
        sys.exit(1)


def main() -> tuple[str, str]:
    """Bootstrap eval environment. Returns (access_token, jwt)."""
    parser = argparse.ArgumentParser(
        description="Bootstrap eval environment (create user, import data)"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Ghostfolio API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-creation even if credentials exist",
    )
    args = parser.parse_args()

    # Step 1: Wait for Ghostfolio
    wait_for_ghostfolio(args.api_url)

    # Step 2: Get or create user credentials
    jwt = None
    creds = None if args.force else load_credentials()

    if creds:
        print("Found existing credentials, verifying...")
        jwt = try_authenticate(args.api_url, creds["accessToken"])
        if jwt:
            print("  Existing credentials valid.")
            creds["authToken"] = jwt
            save_credentials(creds)
        else:
            print("  Existing credentials invalid, creating new user...")
            creds = None

    if not creds:
        print("Creating new eval user...")
        creds = create_user(args.api_url)
        save_credentials(creds)
        jwt = creds["authToken"]
        print(f"  User created (role={creds['role']})")

    # Step 3: Import mock data (idempotent)
    if check_has_activities(args.api_url, jwt):
        print("Mock data already imported, skipping.")
    else:
        print("Importing mock data...")
        import_mock_data(args.api_url, jwt)

    # Step 4: Output for downstream use
    print(f"\n--- Eval Credentials ---")
    print(f"Access Token: {creds['accessToken'][:20]}...")
    print(f"JWT: {jwt[:40]}...")
    print(f"Saved to: {CREDENTIALS_FILE}")

    return creds["accessToken"], jwt


if __name__ == "__main__":
    main()
