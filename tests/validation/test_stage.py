"""ValidationStage with fake upstream directories and a monkeypatched relocation (fast)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv import stages
from osv.config import resolve_config
from osv.validation import stage as vstage
from osv.validation.relocate import RelocateOptions, ValidationResult, compare_relocations


@pytest.fixture(autouse=True)
def _reg(restore_stage_registry, monkeypatch):
    stages.register(vstage.ValidationStage)
    monkeypatch.setattr(vstage.ValidationStage, "tools", ())


def _complete(project: Path, name: str) -> Path:
    d = project / stages.stage_dirname(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"status": "complete", "run_id": name}))
    return d


def _project(tmp_path: Path, *, invert: bool) -> Path:
    prj = tmp_path / "w"
    prep = _complete(prj, "prep")
    n = 8
    pd.DataFrame(
        {"event_id": [f"e{i}" for i in range(n)], "split": "validation", "depth_km_bsl": 2.0}
    ).to_parquet(prep / "events.parquet")
    pd.DataFrame({"event_id": ["e0"], "phase": ["P"]}).to_parquet(prep / "picks.parquet")
    pd.DataFrame({"alias": ["A"]}).to_parquet(prep / "stations.parquet")
    an = _complete(prj, "velest_analyse")
    (an / "final_model.mod").write_text("model")
    pd.DataFrame({"sta": ["A"], "alias": ["A"], "ptcor": [0.0]}).to_csv(
        an / "station_delays.csv", index=False
    )
    b3 = _complete(prj, "build_model")
    (b3 / "fmtomo").mkdir()
    (b3 / "fmtomo" / "vgrids.in").write_text("start")
    if invert:
        inv = _complete(prj, "invert")
        (inv / "final_vgrids.in").write_text("final")
    return prj


def _fake_result(**_: object) -> ValidationResult:
    rng = np.random.default_rng(0)
    n = 8
    ev = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "lat0": -38.5 + rng.normal(0, 0.01, n),
            "lon0": 176.2 + rng.normal(0, 0.01, n),
            "depth0": 2.0,
            "lat_1d": -38.5 + rng.normal(0, 0.01, n),
            "lon_1d": 176.2 + rng.normal(0, 0.01, n),
            "depth_1d": 2.0 + rng.normal(0, 0.3, n),
            "rms_1d": 0.12 + rng.normal(0, 0.01, n),
            "lat_3d": -38.5 + rng.normal(0, 0.01, n),
            "lon_3d": 176.2 + rng.normal(0, 0.01, n),
            "depth_3d": 2.0 + rng.normal(0, 0.3, n),
            "rms_3d": 0.08 + rng.normal(0, 0.01, n),
            "n_picks": 10,
        }
    )
    stats, findings = compare_relocations(ev, RelocateOptions())
    ev.attrs["residuals_3d"] = pd.DataFrame({"x": [1]})
    return ValidationResult(ev, stats, findings)


def _run(tmp_path: Path, monkeypatch, *, invert: bool, extra: dict | None = None):
    prj = _project(tmp_path, invert=invert)
    calls: dict = {}

    def fake(events, picks, model_1d, delays, model_3d, grid, cfg, workdir, n_jobs, **kw):
        calls.update(model_3d=Path(model_3d), n_jobs=n_jobs, **kw)
        calls["model_1d"] = Path(model_1d)
        return _fake_result()

    monkeypatch.setattr(vstage, "relocate_validation", fake)
    monkeypatch.setattr(vstage, "grid_from_outputs", lambda d: object())
    cfg = resolve_config(
        {
            "project": {"workdir": str(prj), "n_jobs": 3},
            "plotting": {"format": ["png"], "dpi": 50},
            **(extra or {}),
        },
        tmp_path,
    )
    res = stages.run_pipeline(cfg, only="validation")
    return prj, calls, res


def test_stage_registered_and_cli_command():
    from osv.cli import app

    assert stages.stage_number("validation") == 9
    assert "validation" in {c.name for c in app.registered_commands}
    assert vstage.ValidationStage.requires == ("prep", "velest_analyse", "build_model")


def test_runs_with_inverted_model(tmp_path, monkeypatch):
    prj, calls, res = _run(tmp_path, monkeypatch, invert=True)
    d = prj / "09_validation"
    assert calls["model_3d"].name == "final_vgrids.in"
    assert calls["model_3d"].parent.name == "10_invert" or "invert" in str(calls["model_3d"])
    assert calls["n_jobs"] == 3
    assert calls["model_inputs_dir"].name == "fmtomo"
    assert len(pd.read_parquet(d / "validation_events.parquet")) == 8
    stats = json.loads((d / "stats.json").read_text())
    assert stats["model_source"] == "invert"
    assert stats["stats"]["rms"]["n"] == 8
    man = json.loads((d / "manifest.json").read_text())
    assert man["status"] == "complete"
    assert "validation_starting_model" not in {c["code"] for c in man["checks"]}
    assert man["summary"]["median_rms_improvement_ms"] > 0
    assert (d / "figures" / "validation_rms_scatter.png").exists()


def test_falls_back_to_starting_model_with_warning(tmp_path, monkeypatch):
    prj, calls, _ = _run(tmp_path, monkeypatch, invert=False)
    assert calls["model_3d"].name == "vgrids.in"
    man = json.loads((prj / "09_validation" / "manifest.json").read_text())
    warn = [c for c in man["checks"] if c["code"] == "validation_starting_model"]
    assert warn and warn[0]["severity"] == "warn"


def test_disabled_writes_empty_outputs(tmp_path, monkeypatch):
    prj, calls, _ = _run(
        tmp_path, monkeypatch, invert=True, extra={"validation": {"enabled": False}}
    )
    assert not calls
    d = prj / "09_validation"
    assert json.loads((d / "stats.json").read_text()) == {"enabled": False}
    assert pd.read_parquet(d / "validation_events.parquet").empty


def test_options_from_config(tmp_path):
    cfg = resolve_config(
        {"validation": {"max_iterations": 3, "use_s": True}, "fmtomo": {"phases": ["P"]}},
        tmp_path,
    )
    opt = vstage._options(cfg)
    assert opt.n_iterations == 3
    assert opt.phases == ("P", "S")
    assert vstage._options(resolve_config({}, tmp_path)).phases == ("P",)
