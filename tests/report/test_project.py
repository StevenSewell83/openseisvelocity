from __future__ import annotations

import json
from pathlib import Path

import pytest

from osv import stages
from osv.config import resolve_config
from osv.report.project import collect_project, headline_numbers
from osv.report.stage import ReportStage
from osv.stages import read_manifest

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478"
    "9c6360000002000001e221bc330000000049454e44ae426082"
)


def _stage(wd: Path, name: str, status="complete", checks=(), summary=None, figs=()):
    d = wd / stages.stage_dirname(name)
    (d / "figures").mkdir(parents=True)
    (d / "report.html").write_text("x")
    m = {
        "stage": name,
        "status": status,
        "wall_time_s": 12.5,
        "started": "2026-01-01T10:00:00+00:00",
        "finished": "2026-01-01T10:00:12+00:00",
        "osv_version": "0.1.0",
        "checks": list(checks),
        "summary": summary or {},
        "outputs": [],
        "tools": {"velest": {"path": "/bin/velest", "sha256": "ab" * 32}},
    }
    (d / "manifest.json").write_text(json.dumps(m))
    reg = []
    for f in figs:
        (d / "figures" / f"{f}.png").write_bytes(PNG)
        reg.append({"name": f, "caption": f"cap {f}", "files": [f"{f}.png"]})
    (d / "figures" / "figures.json").write_text(json.dumps({"figures": reg}))
    return d


@pytest.fixture
def cfg(tmp_path: Path):
    return resolve_config({"project": {"workdir": str(tmp_path / "w"), "name": "demo"}}, tmp_path)


@pytest.fixture
def wd(cfg) -> Path:
    w = Path(cfg.project.workdir)
    _stage(
        w,
        "prep",
        summary={"events_selected": 77, "vpvs": 1.7164, "picks_P_selected": 2171},
        checks=[{"code": "dups", "severity": "warn", "message": "5 duplicate picks", "count": 5}],
        figs=["other", "map_before_after"],
    )
    _stage(
        w,
        "validation",
        summary={"median_rms_1d_ms": 80.0, "median_rms_3d_ms": 60.5},
        checks=[{"name": "bad", "status": "fail", "message": "3-D worse"}],
    )
    _stage(w, "invert", status="failed", summary={"damping_P": 10})
    return w


@pytest.fixture(autouse=True)
def _reg(restore_stage_registry):
    stages.register(ReportStage)


def test_headlines_patterns_and_fallback():
    h = dict(headline_numbers("invert", {"damping_P": 5, "zzz": 1, "rms_final_ms_P": 3.14159}))
    assert h == {"damping_P": "5", "rms_final_ms_P": "3.142"}
    assert headline_numbers("unknown", {"a": 1, "b": {"c": 1}}) == [("a", "1")]


def test_report_stage(cfg, wd):
    res = stages.run_pipeline(cfg, only="report")
    assert res == {"report": "ran"}
    html = (wd / "10_report" / "index.html").read_text()
    root = (wd / "index.html").read_text()
    assert "demo" in html
    for s in ("complete", "failed", "not run", "5 duplicate picks", "3-D worse"):
        assert s in html
    assert 'href="../01_prep/report.html"' in html and 'href="01_prep/report.html"' in root
    assert 'src="../01_prep/figures/map_before_after.png"' in html
    assert "figures/other.png" not in html  # only the named key figure is embedded
    assert "77" in html and "80" in html
    md = (wd / "10_report" / "summary.md").read_text()
    assert "events_selected: 77" in md and "[FAIL]" in md
    # up to date until a manifest changes
    assert stages.run_pipeline(cfg, only="report") == {"report": "skipped"}
    mf = wd / "01_prep" / "manifest.json"
    m = json.loads(mf.read_text())
    m["finished"] = "2026-02-01T00:00:00+00:00"
    mf.write_text(json.dumps(m))
    assert stages.run_pipeline(cfg, only="report") == {"report": "ran"}
    assert read_manifest(wd / "10_report")["status"] == "complete"


def test_embed_mode(cfg, wd):
    cfg2 = resolve_config(
        {"project": {"workdir": str(wd), "name": "demo"}, "report": {"embed_images": True}},
        wd.parent,
    )
    stages.run_pipeline(cfg2, only="report")
    html = (wd / "index.html").read_text()
    assert "data:image/png;base64," in html
    assert "figures/map_before_after.png" not in html


def test_missing_stages(cfg, tmp_path):
    w = Path(cfg.project.workdir)
    w.mkdir()
    data = collect_project(w, cfg)
    assert data["n_run"] == 0 and all(s["status"] == "not run" for s in data["stages"])
    stages.run_pipeline(cfg, only="report")
    assert "not run" in (w / "index.html").read_text()
    # corrupt manifest tolerated
    d = w / "01_prep"
    d.mkdir()
    (d / "manifest.json").write_text("{nope")
    assert collect_project(w)["stages"][0]["status"] == "not run"
