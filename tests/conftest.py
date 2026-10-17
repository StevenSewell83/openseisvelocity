"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from osv import stages


@pytest.fixture
def restore_stage_registry() -> Iterator[None]:
    """Snapshot the stage registry and restore it after the test (even on failure)."""
    saved = dict(stages._REGISTRY)
    try:
        yield
    finally:
        stages._REGISTRY.clear()
        stages._REGISTRY.update(saved)
