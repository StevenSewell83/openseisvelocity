from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from osv import stages
from osv.config import resolve_config
from osv.plotting.style import add_scale_bar, new_figure, perturbation_norm, setup_map_axes
from osv.report import render_project_index
from osv.report.checks import QCFinding
from osv.stages import Stage, StageContext


class Dummy(Stage):
    name = "prep"
    config_sections = ("plotting",)
    outputs = ("x.txt",)

    def run(self, ctx: StageContext) -> None:
        ctx.path("x.txt").write_text("x")
        ctx.add_check(QCFinding("fine", "info", "all good"))
        ctx.add_check(QCFinding("dups", "warn", "5 duplicate picks", count=5))
        ctx.summary.update(n_events=12, mean_rms=0.123456789)
        for i, sec in enumerate(["Maps", "Maps"]):
            fig, ax = new_figure()
            ax.scatter(np.arange(5), np.arange(5) * (i + 1))
            setup_map_axes(ax, lat0=-38.5)
            add_scale_bar(ax)
            ctx.figures.save(fig, f"fig{i}", f"caption {i} <b>", section=sec)


class Boom(Dummy):
    def run(self, ctx: StageContext) -> None:
        ctx.logger.info("about to fail")
        raise RuntimeError("kaboom")


@pytest.fixture
def cfg(tmp_path: Path):
    return resolve_config({"project": {"workdir": str(tmp_path / "w")}}, tmp_path)


@pytest.fixture(autouse=True)
def _reg():
    yield
    stages.unregister("prep")


def test_headless_backend():
    assert matplotlib.get_backend().lower() == "agg"


def test_dummy_stage_report(cfg):
    stages.register(Dummy)
    stages.run_pipeline(cfg, only="prep")
    d = Path(cfg.project.workdir) / "01_prep"
    html = (d / "report.html").read_text()
    assert html.index("5 duplicate picks") < html.index("all good")
    assert "n_events" in html and "0.123457" in html
    assert "figures/fig0.png" in html and 'href="figures/fig0.pdf"' in html
    assert "caption 0 &lt;b&gt;" in html
    for fmt in cfg.plotting.format:
        for n in ("fig0", "fig1"):
            assert (d / "figures" / f"{n}.{fmt}").stat().st_size > 0
    meta = json.loads((d / "figures" / "figures.json").read_text())["figures"]
    assert [m["name"] for m in meta] == ["fig0", "fig1"]
    assert meta[0]["section"] == "Maps" and meta[0]["files"][0] == "fig0.png"


def test_plot_only_rerenders_and_is_idempotent(cfg):
    stages.register(Dummy)
    stages.run_pipeline(cfg, only="prep")
    d = Path(cfg.project.workdir) / "01_prep"
    (d / "report.html").unlink()
    stages.plot_only(cfg, "prep")
    assert (d / "report.html").exists()


def test_failed_stage_report_has_error_and_log(cfg):
    stages.register(Boom)
    with pytest.raises(stages.StageFailed):
        stages.run_pipeline(cfg, only="prep")
    html = (Path(cfg.project.workdir) / "01_prep" / "report.html").read_text()
    assert "kaboom" in html and "about to fail" in html and "failed" in html


def test_project_index(cfg):
    stages.register(Dummy)
    stages.run_pipeline(cfg, only="prep")
    out = render_project_index(Path(cfg.project.workdir))
    html = out.read_text()
    assert "01_prep/report.html" in html and "1 warn" in html


def test_perturbation_norm_centred():
    n = perturbation_norm(np.array([-2.0, 1.0, np.nan]))
    assert n.vcenter == 0 and n.vmin == -2 and n.vmax == 2
