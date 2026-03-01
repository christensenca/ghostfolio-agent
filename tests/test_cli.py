"""Tests for ghostfolio_agent.cli module."""
import pytest


def test_cli_import():
    from ghostfolio_agent.cli import main
    assert callable(main)
