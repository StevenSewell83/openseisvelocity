from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from osv import stages
from osv.cli import app
from osv.config import OSVConfig, resolve_config
from osv.stages import Stage, StageContext, StageError, StageFailed

CALLS: list[str] = []


class DPrep(Stage):
    name = "prep"
    config_sections = ("inputs",)
    outputs = ("events.txt",)

    def inputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project_dir / "input.txt"]

    def run(self, ctx: StageContext) -> None:
        CALLS.append("prep")
        ctx.logger.info("hello from prep")
        ctx.path("events.txt").write_text((ctx.project_dir / "input.txt").read_text())
        ctx.add_check({"name": "n", "status": "pass"})

    def plot(self, ctx: StageContext) -> None:
        CALLS.append("plot:prep")
        (ctx.figures_dir / "a.png").write_bytes(b"x")


class DMc(Stage):
    name = "velest_mc"
    requires = ("prep",)
    config_sections = ("velest",)
    outputs = ("out.txt",)

    def run(self, ctx: StageContext) -> None:
        CALLS.append("velest_mc")
        ctx.path("out.txt").write_text(ctx.upstream("prep").path("events.txt").read_text())


class DAnalyse(Stage):
    name = "velest_analyse"
    requires = ("velest_mc",)
    config_sections = ("plotting",)
    outputs = ("res.txt",)

    def run(self, ctx: StageContext) -> None:
        CALLS.append("velest_analyse")
        if getattr(DAnalyse, "fail", False):
            raise RuntimeError("boom")
        ctx.path("res.txt").write_text("ok")


@pytest.fixture(autouse=True)
def _registry() -> Iterator[None]:
    # import the real stage modules first, so the CLI's discover() is a no-op and cannot
    # replace the dummy stages registered below
    stages.discover()
    saved = dict(stages._REGISTRY)
    stages._REGISTRY.clear()
    for c in (DPrep, DMc, DAnalyse):
        stages.register(c)
    DAnalyse.fail = False  # type: ignore[attr-defined]
    CALLS.clear()
    yield
    stages._REGISTRY.clear()
    stages._REGISTRY.update(saved)


def make_cfg(tmp_path: Path, **velest: object) -> OSVConfig:
    (tmp_path / "proj").mkdir(exist_ok=True)
    (tmp_path / "proj" / "input.txt").write_text("data")
    return resolve_config(
        {"project": {"workdir": str(tmp_path / "proj")}, "velest": velest}, tmp_path
    )


def manifest(cfg: OSVConfig, name: str) -> dict:
    return json.loads(
        (cfg.project.workdir / stages.stage_dirname(name) / "manifest.json").read_text()
    )


def test_ordering_and_layout(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    res = stages.run_pipeline(cfg)
    assert CALLS == ["prep", "velest_mc", "velest_analyse"]
    assert res == dict.fromkeys(CALLS, "ran")
    d = cfg.project.workdir / "01_prep"
    assert (d / "figures").is_dir() and (d / "logs").is_dir()
    assert "hello from prep" in (d / "logs" / "stage.log").read_text()
    m = manifest(cfg, "prep")
    assert m["status"] == "complete" and m["number"] == 1
    assert m["outputs"][0]["path"] == "events.txt" and len(m["outputs"][0]["sha256"]) == 64
    assert m["checks"] == [{"name": "n", "status": "pass"}]
    assert m["config_hash"] and m["osv_version"] and m["wall_time_s"] is not None
    assert list(m["input_hashes"].values())[0] != "missing"
    assert manifest(cfg, "velest_mc")["upstream"]["prep"] == m["run_id"]
    assert stages.stage_dirname("velest_analyse") == "03_velest_analyse"


def test_skip_if_complete_and_force(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    stages.run_pipeline(cfg)
    CALLS.clear()
    assert set(stages.run_pipeline(cfg).values()) == {"skipped"}
    assert CALLS == []
    res = stages.run_pipeline(cfg, force=True)
    assert set(res.values()) == {"ran"}
    assert CALLS == ["prep", "velest_mc", "velest_analyse"]


def test_config_change_propagates_downstream_only(tmp_path: Path) -> None:
    stages.run_pipeline(make_cfg(tmp_path, n_models=10))
    CALLS.clear()
    res = stages.run_pipeline(make_cfg(tmp_path, n_models=11))
    assert res == {"prep": "skipped", "velest_mc": "ran", "velest_analyse": "ran"}


def test_input_change_makes_everything_stale(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    stages.run_pipeline(cfg)
    (tmp_path / "proj" / "input.txt").write_text("changed")
    res = stages.run_pipeline(cfg)
    assert set(res.values()) == {"ran"}
    assert (cfg.project.workdir / "02_velest_mc" / "out.txt").read_text() == "changed"


def test_missing_output_triggers_rerun(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    stages.run_pipeline(cfg)
    (cfg.project.workdir / "03_velest_analyse" / "res.txt").unlink()
    assert stages.run_pipeline(cfg)["velest_analyse"] == "ran"


def test_failure_manifest_and_stop(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    DAnalyse.fail = True  # type: ignore[attr-defined]
    with pytest.raises(StageFailed, match="boom"):
        stages.run_pipeline(cfg)
    m = manifest(cfg, "velest_analyse")
    assert m["status"] == "failed"
    assert "RuntimeError" in m["error"]["traceback"] and m["error"]["message"] == "boom"
    # failed stage re-runs next time while completed upstream stages are skipped
    DAnalyse.fail = False  # type: ignore[attr-defined]
    res = stages.run_pipeline(cfg)
    assert res == {"prep": "skipped", "velest_mc": "skipped", "velest_analyse": "ran"}


def test_failure_stops_downstream(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    DMc.run_orig = DMc.run  # type: ignore[attr-defined]

    def bad(self: Stage, ctx: StageContext) -> None:
        raise ValueError("x")

    DMc.run = bad  # type: ignore[method-assign]
    try:
        with pytest.raises(StageFailed):
            stages.run_pipeline(cfg)
    finally:
        DMc.run = DMc.run_orig  # type: ignore[attr-defined,method-assign]
    assert "velest_analyse" not in CALLS


def test_missing_declared_output_fails(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)

    class Bad(DPrep):
        def run(self, ctx: StageContext) -> None:
            pass

    stages._REGISTRY["prep"] = Bad
    with pytest.raises(StageFailed, match="events.txt"):
        stages.run_pipeline(cfg)


def test_from_to_only(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    assert list(stages.run_pipeline(cfg, to="velest_mc")) == ["prep", "velest_mc"]
    CALLS.clear()
    assert list(stages.run_pipeline(cfg, from_="velest-analyse")) == ["velest_analyse"]
    assert CALLS == ["velest_analyse"]
    assert list(stages.run_pipeline(cfg, from_="velest_mc", to="velest_mc")) == ["velest_mc"]
    assert stages.run_pipeline(cfg, only="prep", force=True) == {"prep": "ran"}
    with pytest.raises(StageError, match="after"):
        stages.run_pipeline(cfg, from_="velest_mc", to="prep")
    with pytest.raises(StageError, match="Unknown"):
        stages.run_pipeline(cfg, to="nope")


def test_missing_upstream(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    with pytest.raises(stages.MissingUpstreamError, match="prep"):
        stages.run_pipeline(cfg, from_="velest_mc")


def test_plot_only(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    with pytest.raises(stages.MissingUpstreamError):
        stages.plot_only(cfg, "prep")
    stages.run_pipeline(cfg)
    run_id = manifest(cfg, "prep")["run_id"]
    CALLS.clear()
    stages.plot_only(cfg, "prep")
    assert CALLS == ["plot:prep"]
    assert (cfg.project.workdir / "01_prep" / "figures" / "a.png").exists()
    m = manifest(cfg, "prep")
    assert m["run_id"] == run_id and "plotted" in m
    # plotting does not make downstream stale
    assert set(stages.run_pipeline(cfg).values()) == {"skipped"}


def test_cli_run_plot_and_stage_command(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"project": {"workdir": "proj"}}))
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "input.txt").write_text("data")
    runner = CliRunner()
    r = runner.invoke(app, ["run", "-c", str(cfg_path), "--to", "velest-mc"])
    assert r.exit_code == 0, r.output
    assert "prep: ran" in r.output
    r = runner.invoke(app, ["run", "-c", str(cfg_path), "--to", "velest-mc"])
    assert "prep: skipped" in r.output
    r = runner.invoke(app, ["prep", "-c", str(cfg_path), "--force"])
    assert "prep: ran" in r.output and "velest_mc" not in r.output
    r = runner.invoke(app, ["plot", "prep", "-c", str(cfg_path)])
    assert r.exit_code == 0 and "regenerated" in r.output
    r = runner.invoke(app, ["plot", "velest-analyse", "-c", str(cfg_path)])
    assert r.exit_code == 1 and "not completed" in r.output
    DAnalyse.fail = True  # type: ignore[attr-defined]
    r = runner.invoke(app, ["run", "-c", str(cfg_path)])
    assert r.exit_code == 1 and "failed" in r.output and "boom" in r.output
