"""Tests for osv.resolution.stage (Stage 6) with fake upstream stage directories."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.config.schema import Model3DConfig, OSVConfig
from osv.fmtomo.invert import InversionResult
from osv.fmtomo.stage import default_mapping
from osv.io.fmtomo import (
    PathSpec,
    Receivers,
    Source,
    write_interfaces,
    write_observed,
    write_propgrid,
    write_receivers,
    write_sources,
    write_vgrids,
)
from osv.model3d.grid import define_grid
from osv.model3d.stage import grid_summary
from osv.resolution import runner as rn
from osv.resolution import stage as st
from osv.resolution.models import checkerboard
from osv.stages import StageContext, StageError, stage_dirname
from osv.tools import ToolNotFoundError, which_tool

LAT0, LON0 = -38.5, 176.2


def _have(*names: str) -> bool:
    try:
        for n in names:
            which_tool(n)
    except ToolNotFoundError:
        return False
    return True


def _cfg(tmp_path, **fm) -> OSVConfig:
    return OSVConfig(
        project={"workdir": str(tmp_path)},
        fmtomo={
            "package": "base",
            "relocate": "linear",
            "damping": 10.0,
            "smoothing": 5.0,
            "iterations": 2,
            "subspace_dimension": 15,
            **fm,
        },
        resolution={
            "checkerboard": {"cell_km": [3.0], "amplitude_pct": 20.0, "offset_half_cell": False},
            "input_models": [],
            "noise": {"scale": 0.02},
        },
    )


def _complete(project, name: str) -> None:
    d = project / stage_dirname(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"status": "complete", "run_id": name}))


@pytest.fixture
def project(tmp_path):
    """Fake Stage 3 (build_model) and Stage 4a (tradeoff) outputs for a tiny geometry."""
    la, lo = np.meshgrid(np.linspace(0, 0.09, 3), np.linspace(0, 0.11, 3), indexing="ij")
    stations = pd.DataFrame(
        {"lat": LAT0 + la.ravel(), "lon": LON0 + lo.ravel(), "elevation_m": 0.0}
    )
    mcfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 1.0, "zmax_km": 6.0},
        propgrid_refine=2,
    )
    grid = define_grid(stations, None, mcfg)
    rng = np.random.default_rng(0)
    n_src = 8
    src = [
        Source(
            float(rng.uniform(2, 4)),
            float(LAT0 + rng.uniform(0.02, 0.07)),
            float(LON0 + rng.uniform(0.02, 0.09)),
            [PathSpec.direct()],
            uncertainty=(1.0, 0.01, 0.01),
        )
        for _ in range(n_src)
    ]
    pos = pd.DataFrame(
        {"depth_km": 0.0, "lat": stations["lat"].to_numpy(), "lon": stations["lon"].to_numpy()},
        index=pd.RangeIndex(1, len(stations) + 1, name="receiver"),
    )
    rays = pd.DataFrame(
        [(r, s, 1) for r in pos.index for s in range(1, n_src + 1) if (r + s) % 3 != 0],
        columns=["receiver", "source", "path"],
    )
    rr = rays.assign(ray=rays.groupby("receiver").cumcount() + 1)
    obs = pd.DataFrame(
        {
            "receiver": rr["receiver"],
            "source": rr["source"],
            "ray": rr["ray"],
            "subray": 0,
            "time": 1.0,
            "uncertainty": 0.04,
        }
    )
    obs = obs.sort_values(["source", "receiver", "ray"]).reset_index(drop=True)

    s3 = tmp_path / stage_dirname("build_model")
    (s3 / "fmtomo").mkdir(parents=True)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    bg.to_netcdf(s3 / "starting_model.nc")
    (s3 / "grid.json").write_text(json.dumps(grid_summary(grid)))
    write_propgrid(s3 / "fmtomo" / "propgrid.in", grid.propgrid)
    write_interfaces(s3 / "fmtomo" / "interfaces.in", grid.flat_interfaces())
    write_vgrids(s3 / "fmtomo" / "vgrids.in", grid.to_vgrids(bg["vp"]))
    write_sources(s3 / "fmtomo" / "sources.in", src)
    write_receivers(s3 / "fmtomo" / "receivers.in", Receivers(pos, rays))
    write_observed(s3 / "fmtomo" / "otimes_P.dat", obs)
    default_mapping(obs).to_csv(s3 / "fmtomo" / "otimes_P_mapping.csv", index=False)
    _complete(tmp_path, "build_model")
    td = tmp_path / stage_dirname("tradeoff")
    td.mkdir()
    (td / "tradeoff.json").write_text(
        json.dumps(
            {
                "chosen_damping": 3.5,
                "chosen_smoothing": 7.0,
                "chosen": {"P": {"damping": 3.5, "smoothing": 7.0}},
                "phases": ["P"],
                "results": {},
            }
        )
    )
    _complete(tmp_path, "tradeoff")
    return SimpleNamespace(dir=tmp_path, grid=grid)


def _context(cfg: OSVConfig, tmp_path) -> StageContext:
    ctx = StageContext(cfg, st.ResolutionStage(), tmp_path)
    ctx.prepare_dirs()
    return ctx


@pytest.fixture
def fakes(monkeypatch):
    calls: dict[str, list] = {"inv": []}

    def fake_forward(model, grid, base, workdir, **kw):
        return pd.DataFrame()

    def fake_synth(true, grid, base, workdir, cfg, seed=42, n_jobs=1, phase="P", **kw):
        workdir.mkdir(parents=True, exist_ok=True)
        _, _, src, _, obs = base.load()
        from osv.io.fmtomo import write_observed, write_sources

        write_sources(workdir / "sources.in", src)
        rng = np.random.default_rng(1)
        noisy = obs.assign(time=obs["time"] + rng.normal(0, 0.02, len(obs)))
        write_observed(workdir / f"otimes_{phase}.dat", noisy)
        write_observed(workdir / f"otimes_{phase}_noiseless.dat", obs)
        return SimpleNamespace(
            sources=workdir / "sources.in",
            otimes=workdir / f"otimes_{phase}.dat",
            summary=lambda: {"snr": 3.0},
        )

    def fake_inv(run, controls, n_iter, **kw):
        calls["inv"].append((run.parent.name, n_iter, controls.run.invert3d))
        true = xr.load_dataset(run.parent / "true_model.nc")
        grid = fake_inv.grid
        write_vgrids(run / "vgrids.in", grid.to_vgrids(true["vp"]))
        (run / "frechet.dat").write_text("x")
        return InversionResult(
            run_dir=run,
            iterations=pd.DataFrame(
                {"iter": [0, 1, 2], "rms_ms": [100.0, 60.0, 40.0], "wall_s": 0.1}
            ),
            hypocentres=pd.DataFrame(),
            final_model=run / "vgrids.in",
            final_sources=None,
            residuals_path=run / "residuals.dat",
        )

    monkeypatch.setattr(rn._synth, "forward_times", fake_forward)
    monkeypatch.setattr(rn._synth, "make_synthetic_data", fake_synth)
    monkeypatch.setattr(rn._inv, "run_inversion", fake_inv)
    monkeypatch.setattr(rn, "attach_metrics", lambda res: res.iterations)
    return SimpleNamespace(calls=calls, inv=fake_inv)


@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
def test_stage_run_with_fake_inversion(project, fakes, tmp_path) -> None:
    fakes.inv.grid = project.grid
    cfg = _cfg(tmp_path)
    ctx = _context(cfg, tmp_path)
    st.ResolutionStage().run(ctx)
    sd = ctx.stage_dir
    # stable outputs
    for f in ("resolution_runs.json", "resolvable_mask.nc", "resolved_fractions.csv"):
        assert (sd / f).exists(), f
    assert (sd / "metrics" / "summary.csv").exists() and (sd / "metrics" / "metrics.nc").exists()
    runs = json.loads((sd / "resolution_runs.json").read_text())
    assert runs["chosen"]["P"] == {"damping": 3.5, "smoothing": 7.0}
    assert runs["coverage_source"].startswith("straight-ray proxy")
    # the chosen regularisation and the iterations reached the inversion
    name, n_iter, inv = fakes.calls["inv"][0]
    assert name == "checker_3km_P" and n_iter == cfg.fmtomo.iterations
    assert inv.global_damping == 3.5 and inv.global_smoothing == 7.0
    # mask: perfect recovery -> every covered node resolvable at 3 km
    m = xr.load_dataset(sd / "resolvable_mask.nc")
    assert bool(m["resolvable"].any())
    scale = m["resolved_scale_km"].values
    assert np.nanmax(scale) == 3.0 and np.isfinite(scale).any()
    interior = ~project.grid.cushion_mask().values
    assert not m["resolvable"].values[~interior].any()
    frac = pd.read_csv(sd / "resolved_fractions.csv")
    assert set(frac["phase"]) == {"P"} and (frac["cell_km"] == 3.0).all()
    assert frac.loc[frac["depth_range_km"] == "all", "fraction_resolved"].iloc[0] > 0.5
    # figures, checks, summary
    names = {e["name"] for e in ctx.figures.entries()}
    for n in (
        "true_vs_recovered_slices_checker_3km_P",
        "true_vs_recovered_sections_checker_3km_P",
        "checkerboard_matrix_P",
        "recovery_corr_P",
        "recovery_amp_P",
        "resolved_scale_P",
        "noise_realisation_P",
        "summary_metrics",
    ):
        assert n in names, n
    assert ctx.summary["tests_succeeded"] == 1 and "resolved_fraction_P" in ctx.summary
    codes = {c["code"] for c in ctx.checks}
    assert "resolved_fraction" in codes and "tests_failed" not in codes
    # compacted
    assert not list(sd.rglob("frechet.dat"))


def test_nonlinear_src_dat_blocks_are_stations_ids_are_events(project, fakes, tmp_path) -> None:
    """With gridloc relocation each test run gets the src.dat that gridloc expects."""
    from osv.io.fmtomo import read_observed
    from osv.io.fmtomo.extras import read_src_dat

    fakes.inv.grid = project.grid
    cfg = _cfg(tmp_path, package="extras", relocate="nonlinear")
    ctx = _context(cfg, tmp_path)
    st.ResolutionStage().run(ctx)
    files = list(ctx.stage_dir.rglob("src.dat"))
    assert files
    obs = read_observed(project.dir / stage_dirname("build_model") / "fmtomo" / "otimes_P.dat")
    for f in files:
        sd = read_src_dat(f)
        assert sd.attrs["n_blocks"] == 9  # one block per station
        # block = station (old receiver), id = event (old source)
        assert set(zip(sd["block"], sd["id"], strict=True)) == set(
            zip(obs["receiver"], obs["source"], strict=True)
        )


@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
def test_stage_checks_failure_paths(project, fakes, tmp_path) -> None:
    fakes.inv.grid = project.grid
    # recovery of nothing: the final model is the background -> nothing resolvable
    grid = project.grid
    bg = xr.load_dataset(tmp_path / stage_dirname("build_model") / "starting_model.nc")

    def flat_inv(run, controls, n_iter, **kw):
        write_vgrids(run / "vgrids.in", grid.to_vgrids(bg["vp"]))
        return InversionResult(
            run_dir=run,
            iterations=pd.DataFrame({"iter": [0, 1], "rms_ms": [100.0, 90.0], "wall_s": 0.1}),
            hypocentres=pd.DataFrame(),
            final_model=run / "vgrids.in",
            final_sources=None,
            residuals_path=run / "residuals.dat",
        )

    rn._inv.run_inversion = flat_inv  # restored by the monkeypatch fixture of `fakes`
    cfg = _cfg(tmp_path)
    ctx = _context(cfg, tmp_path)
    st.ResolutionStage().run(ctx)
    sev = {c["code"]: c["severity"] for c in ctx.checks}
    assert sev["no_resolvable_node"] == "fail"


def test_stage_requires_disk(project, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(st, "MIN_FREE_GB", 1e9)
    ctx = _context(_cfg(tmp_path), tmp_path)
    with pytest.raises(StageError, match="free"):
        st.ResolutionStage().run(ctx)


def test_registered_and_cli() -> None:
    from typer.testing import CliRunner

    from osv import stages
    from osv.cli import app

    stages.discover()
    assert stages.get_stage("resolution") is st.ResolutionStage
    assert st.ResolutionStage.requires == ("build_model", "tradeoff")
    out = CliRunner().invoke(app, ["resolution", "--help"])
    assert out.exit_code == 0 and "--force" in out.output


def test_resolved_fractions_and_proxy(project) -> None:
    grid = project.grid
    scale = np.full(grid.shape_full, np.nan)
    inter = ~grid.cushion_mask().values
    scale[inter] = 3.0
    scale[0:3] = 1.5
    t = st.resolved_fractions(scale, None, grid, [1.5, 3.0], n_ranges=2)
    allrow = t[t["depth_range_km"] == "all"].set_index("cell_km")["fraction_resolved"]
    assert allrow[3.0] == pytest.approx(1.0) and 0 < allrow[1.5] < 1.0
    assert len(t) == 2 * 3
    from osv.io.fmtomo import read_receivers, read_sources

    d = project.dir / stage_dirname("build_model") / "fmtomo"
    cov = st.straight_ray_coverage(
        grid, read_sources(d / "sources.in"), read_receivers(d / "receivers.in"), 1
    )
    assert cov.shape == grid.shape_full and cov.any() and not cov.all()
    cov3 = st.straight_ray_coverage(
        grid, read_sources(d / "sources.in"), read_receivers(d / "receivers.in"), 10_000
    )
    assert not cov3.any()


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
@pytest.mark.skipif(
    not _have("fm3d", "frechgen", "invert3d", "residuals", "scorrect"), reason="binaries missing"
)
def test_stage_end_to_end_real_binaries(project, tmp_path) -> None:
    cfg = _cfg(tmp_path, damping=0.01, smoothing=0.1, invert_velocity=True)
    cfg = cfg.model_copy(update={"project": cfg.project.model_copy(update={"n_jobs": 2})})
    ctx = _context(cfg, tmp_path)
    st.ResolutionStage().run(ctx)
    runs = json.loads((ctx.stage_dir / "resolution_runs.json").read_text())
    assert runs["table"][0]["success"], runs["table"][0]["flags"]
    assert (ctx.stage_dir / "resolvable_mask.nc").exists()
    assert "checkerboard_matrix_P" in {e["name"] for e in ctx.figures.entries()}
