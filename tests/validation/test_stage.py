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


# ------------------------------------------------------------------ bootstrap wiring
def _fake_bootstrap(tmp_path: Path, n_ok: int, n_samples: int, std: float):
    import xarray as xr

    from osv.validation.bootstrap import BootstrapResult

    shape = (4, 4, 4)
    dws = np.zeros(shape)
    dws[1:3, 1:3, 1:3] = 100.0
    dims = ("depth", "lat", "lon")
    sd = np.full(shape, 9.0)  # large outside the coverage
    sd[1:3, 1:3, 1:3] = std
    ds = xr.Dataset(
        {
            "vp_std": (dims, sd / 10),
            "dvp_std_pct": (dims, sd),
            "dws": (dims, dws),
        },
        attrs={"n_cushion": 0},
    )
    table = pd.DataFrame(
        {
            "name": [f"s{i}" for i in range(n_samples)],
            "success": [i < n_ok for i in range(n_samples)],
        }
    )
    return BootstrapResult(
        tmp_path, table, ds, tmp_path / "bootstrap_std.nc", pd.DataFrame(), pd.DataFrame(),
        "bootstrap", 1.0, 2.0, 3, 8, 1, 1,
    )  # fmt: skip


def _run_bootstrap(tmp_path, monkeypatch, *, n_ok=10, n_samples=10, std=1.0, tradeoff=True):
    seen: dict = {}

    def fake_run(inputs, cfg, chosen, workdir, n_jobs=1, **kw):
        seen.update(inputs=inputs, chosen=chosen, workdir=Path(workdir), n_jobs=n_jobs, **kw)
        return _fake_bootstrap(tmp_path, n_ok, n_samples, std)

    figs: dict = {}

    def fake_figs(res, figures, section="Bootstrap", dws_threshold=None):
        figs.update(section=section, thr=dws_threshold)
        return []

    monkeypatch.setattr(vstage, "run_bootstrap", fake_run)
    monkeypatch.setattr(vstage, "make_bootstrap_figures", fake_figs)
    monkeypatch.setattr(
        vstage.BootstrapInputs, "from_stage3", classmethod(lambda cls, d, phase="P": (d, phase))
    )
    prj = _project(tmp_path, invert=True)
    if tradeoff:
        t = _complete(prj, "tradeoff")
        (t / "tradeoff.json").write_text(json.dumps({"chosen_damping": 3, "chosen_smoothing": 4}))
    xr = pytest.importorskip("xarray")
    dws = xr.DataArray(np.ones((4, 4, 4)), dims=("depth", "lat", "lon"))
    xr.Dataset({"dws": dws}).to_netcdf(prj / stages.stage_dirname("invert") / "final_model.nc")
    monkeypatch.setattr(vstage, "relocate_validation", lambda *a, **k: _fake_result())
    monkeypatch.setattr(vstage, "grid_from_outputs", lambda d: object())
    cfg = resolve_config(
        {
            "project": {"workdir": str(prj), "n_jobs": 3},
            "plotting": {"format": ["png"], "dpi": 50},
            "validation": {"bootstrap": {"enabled": True}},
        },
        tmp_path,
    )
    stages.run_pipeline(cfg, only="validation")
    man = json.loads((prj / "09_validation" / "manifest.json").read_text())
    return man, seen, figs


def test_bootstrap_not_run_by_default(tmp_path, monkeypatch):
    called: list = []
    monkeypatch.setattr(vstage, "run_bootstrap", lambda *a, **k: called.append(1))
    _run(tmp_path, monkeypatch, invert=True)
    assert not called


def test_bootstrap_wired(tmp_path, monkeypatch):
    man, seen, figs = _run_bootstrap(tmp_path, monkeypatch)
    assert seen["workdir"].name == "bootstrap"
    assert seen["chosen"].name == "tradeoff.json"
    assert seen["dws"] is not None
    assert seen["inputs"][1] == "P"
    assert figs["section"] == "Bootstrap" and figs["thr"] is not None
    s = man["summary"]
    assert s["bootstrap_samples_ok"] == 10
    assert s["bootstrap_median_std_pct"] == pytest.approx(1.0)  # covered nodes only
    assert s["bootstrap_max_std_pct"] == pytest.approx(1.0)
    codes = {c["code"] for c in man["checks"]}
    assert not codes & {"bootstrap_few_samples", "bootstrap_high_std"}
    assert "bootstrap_default_regularisation" not in codes


def test_bootstrap_checks(tmp_path, monkeypatch):
    man, seen, _ = _run_bootstrap(
        tmp_path, monkeypatch, n_ok=5, n_samples=10, std=5.0, tradeoff=False
    )
    codes = {c["code"] for c in man["checks"]}
    assert {
        "bootstrap_few_samples",
        "bootstrap_high_std",
        "bootstrap_default_regularisation",
    } <= codes
    assert seen["chosen"] is None


def test_bootstrap_failure_is_a_warning(tmp_path, monkeypatch):
    from osv.validation.bootstrap import BootstrapError

    def boom(*a, **k):
        raise BootstrapError("only 1 of 5 samples succeeded")

    prj = _project(tmp_path, invert=True)
    monkeypatch.setattr(vstage, "run_bootstrap", boom)
    monkeypatch.setattr(
        vstage.BootstrapInputs, "from_stage3", classmethod(lambda cls, d, phase="P": None)
    )
    monkeypatch.setattr(vstage, "relocate_validation", lambda *a, **k: _fake_result())
    monkeypatch.setattr(vstage, "grid_from_outputs", lambda d: object())
    cfg = resolve_config(
        {
            "project": {"workdir": str(prj)},
            "plotting": {"format": ["png"], "dpi": 50},
            "validation": {"bootstrap": {"enabled": True}},
        },
        tmp_path,
    )
    stages.run_pipeline(cfg, only="validation")
    man = json.loads((prj / "09_validation" / "manifest.json").read_text())
    assert man["status"] == "complete"
    assert "bootstrap_failed" in {c["code"] for c in man["checks"]}
