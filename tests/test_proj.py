"""Tests for the stale PROJ_DATA guard and the doctor PROJ check."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from osv import _proj, doctor


def _mk(d: Path, minor: int | None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(d / "proj.db")
    con.execute("CREATE TABLE metadata(key TEXT, value TEXT)")
    if minor is not None:
        con.execute(
            "INSERT INTO metadata VALUES('DATABASE.LAYOUT.VERSION.MINOR', ?)", (str(minor),)
        )
    con.commit()
    con.close()
    return d


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("OSV_KEEP_PROJ_DATA", raising=False)
    monkeypatch.delenv("PROJ_LIB", raising=False)
    good = _mk(tmp_path / "good", 6)
    monkeypatch.setattr(_proj, "bundled_proj_data", lambda: good)
    return tmp_path, good


def test_stale_overridden(env, monkeypatch):
    tmp, good = env
    monkeypatch.setenv("PROJ_DATA", str(_mk(tmp / "old", 2)))
    assert _proj.ensure_proj_data() == str(good)
    assert os.environ["PROJ_DATA"] == str(good)
    assert os.environ["PROJ_LIB"] == str(good)


def test_missing_db_overridden(env, monkeypatch):
    tmp, good = env
    monkeypatch.setenv("PROJ_DATA", str(tmp / "nowhere"))
    assert _proj.ensure_proj_data() == str(good)


def test_compatible_kept(env, monkeypatch):
    tmp, _ = env
    new = _mk(tmp / "new", 7)
    monkeypatch.setenv("PROJ_DATA", str(new))
    assert _proj.ensure_proj_data() is None
    assert os.environ["PROJ_DATA"] == str(new)


def test_keep_flag(env, monkeypatch):
    tmp, _ = env
    old = str(_mk(tmp / "old", 2))
    monkeypatch.setenv("PROJ_DATA", old)
    monkeypatch.setenv("OSV_KEEP_PROJ_DATA", "1")
    assert _proj.ensure_proj_data() is None
    assert os.environ["PROJ_DATA"] == old


def test_unset_untouched(env, monkeypatch):
    monkeypatch.delenv("PROJ_DATA", raising=False)
    assert _proj.ensure_proj_data() is None
    assert "PROJ_DATA" not in os.environ


def test_doctor_proj_check_runs(monkeypatch):
    monkeypatch.setenv("OSV_KEEP_PROJ_DATA", "1")
    checks = doctor.check_proj()
    assert all(c.name.startswith("proj:") for c in checks)
    for c in checks:
        assert c.status in (doctor.OK, doctor.FAIL)
        assert "PROJ_DATA" in c.message
