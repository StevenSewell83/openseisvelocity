"""Tests for osv.resolution.runner (resolution inversions with the chosen settings)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import Model3DConfig, OSVConfig
from osv.fmtomo.invert import InversionResult
from osv.io.fmtomo import PathSpec, Receivers, Source, read_vgrids, write_observed, write_sources
from osv.model3d.grid import define_grid
from osv.resolution import runner as rn
from osv.resolution.models import checkerboard
from osv.resolution.synth import BaseInputs
from osv.tools import ToolNotFoundError, which_tool

LAT0, LON0 = -38.5, 176.2


def _setup(n_src: int = 6):
    la, lo = np.meshgrid(np.linspace(0, 0.09, 3), np.linspace(0, 0.11, 3), indexing="ij")
    st = pd.DataFrame({"lat": LAT0 + la.ravel(), "lon": LON0 + lo.ravel(), "elevation_m": 0.0})
    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 1.0, "zmax_km": 6.0},
        propgrid_refine=2,
    )
    grid = define_grid(st, None, cfg)
    rng = np.random.default_rng(0)
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
        {"depth_km": 0.0, "lat": st["lat"].to_numpy(), "lon": st["lon"].to_numpy()},
        index=pd.RangeIndex(1, len(st) + 1, name="receiver"),
    )
    rays = pd.DataFrame(
        [(r, s, 1) for r in pos.index for s in range(1, n_src + 1) if (r + s) % 3 != 0],
        columns=["receiver", "source", "path"],
    )
    rec = Receivers(pos, rays)
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
    base = BaseInputs(grid.flat_interfaces(), grid.propgrid, src, rec, obs)
    return grid, base


def _cfg(**fm) -> OSVConfig:
    return OSVConfig(
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
            "noise": {"scale": 0.02},
        },
    )


def _bg(grid):
    return checkerboard(grid, 5.0, 2.0, 0.0)


# ---------------------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------------------
def test_build_tests_from_config() -> None:
    cfg = OSVConfig(
        fmtomo={"phases": ["P", "S"]},
        resolution={
            "checkerboard": {"cell_km": [1.5, 3.0], "offset_half_cell": True},
            "spikes": [{"x_km": 0, "y_km": 0, "z_km": 2, "radius_km": 1, "pct": -8}],
            "input_models": ["anomaly", "final_3d"],
            "anomalies": [{"kind": "block", "x_km": 0, "y_km": 0, "z_km": 2, "pct": 5}],
        },
    )
    tests = rn.build_tests(cfg)
    names = [t.name for t in tests]
    assert len(names) == len(set(names)) == (4 + 1 + 1) * 2  # final_3d skipped: no dataset
    assert {t.phase for t in tests} == {"P", "S"}
    assert "checker_1p5km_off_S" in names and "spikes_P" in names
    assert {t.kind for t in tests} == {"checkerboard", "spikes", "anomaly"}
    only_p = rn.build_tests(cfg, phases=["P"])
    assert len(only_p) == 6
    json.dumps([t.params for t in tests])


def test_concurrency_budget() -> None:
    cfg = _cfg()
    k, j = rn._concurrency(cfg, 24, 10, None, 0.1, 100.0)
    assert k * j <= 24 and k == 6 and j == 4
    k, j = rn._concurrency(cfg, 4, 10, None, 0.1, 100.0)
    assert (k, j) == (1, 4)
    k, j = rn._concurrency(cfg, 24, 3, None, 0.1, 100.0)
    assert k == 3 and j == 8
    k, j = rn._concurrency(cfg, 24, 10, None, 10.0, 60.0)  # memory limited
    assert k == 1  # fm3d itself caps the jobs of a single test


# ---------------------------------------------------------------------------------------
# orchestration with fake fm3d / inversion
# ---------------------------------------------------------------------------------------
@pytest.fixture
def fakes(monkeypatch):
    calls: dict[str, list] = {"synth": [], "inv": [], "fwd": []}

    def fake_forward(model, grid, base, workdir, **kw):
        calls["fwd"].append(kw.get("phase"))
        return pd.DataFrame()

    def fake_synth(true, grid, base, workdir, cfg, seed=42, n_jobs=1, phase="P", **kw):
        calls["synth"].append((workdir.parent.name, seed, n_jobs, phase, kw["perturb_hypocentres"]))
        workdir.mkdir(parents=True, exist_ok=True)
        _, _, src, _, obs = base.load()
        write_sources(workdir / "sources.in", src)
        write_observed(workdir / "otimes_P.dat", obs)
        (workdir / "fm3d_true").mkdir()
        (workdir / "fm3d_true" / "frechet.dat").write_text("x" * 100)
        return SimpleNamespace(
            sources=workdir / "sources.in",
            otimes=workdir / "otimes_P.dat",
            summary=lambda: {"snr": 3.0},
        )

    def fake_inv(run, controls, n_iter, **kw):
        calls["inv"].append((run.parent.name, n_iter, kw["phase"], controls.run.invert3d))
        if "boom" in run.parent.name:
            raise RuntimeError("fm3d crashed")
        vg = read_vgrids(run / "vgridsref.in")
        v = vg.get(1, 1)["velocity"].values
        # recovered = starting model; run dir keeps a heavy file
        (run / "frechet.dat").write_text("x" * 100)
        (run / "iterations" / "00").mkdir(parents=True, exist_ok=True)
        # write the final vgrids unchanged
        (run / "vgrids.in").write_text((run / "vgridsref.in").read_text())
        assert np.isfinite(v).all()
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
    return calls


@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
def test_orchestration_and_resume(tmp_path, fakes) -> None:
    grid, base = _setup()
    bg = _bg(grid)
    cfg = _cfg()
    tests = rn.build_tests(cfg)
    boom = rn.ResolutionTest("boom_P", "checkerboard", "P", {}, tests[0].build)
    tests = [*tests, boom, rn.ResolutionTest("c2_P", "checkerboard", "P", {}, tests[0].build)]
    res = rn.run_resolution_tests(
        tests, base, grid, bg, cfg, tmp_path, n_jobs=8, chosen=(3.5, 7.0), seed=5
    )
    assert (res.chosen_damping, res.chosen_smoothing) == (3.5, 7.0)
    assert res.max_concurrent * res.jobs_per_test <= 8
    t = res.table.set_index("name")
    assert list(t.index) == ["checker_3km_P", "boom_P", "c2_P"]
    assert t.loc["checker_3km_P", "success"] and t.loc["c2_P", "success"]
    assert not t.loc["boom_P", "success"] and "fm3d crashed" in t.loc["boom_P", "flags"]
    assert t.loc["checker_3km_P", "misfit_initial_s"] == pytest.approx(0.1)
    assert t.loc["checker_3km_P", "misfit_final_s"] == pytest.approx(0.04)
    assert res.failed == ["boom_P"]
    # settings reached the controls and the seed the synthetic data
    inv = [c for c in fakes["inv"] if c[0] == "checker_3km_P"][0]
    assert inv[1] == 2 and inv[2] == "P"
    assert inv[3].global_damping == 3.5 and inv[3].global_smoothing == 7.0
    assert all(c[1] == 5 for c in fakes["synth"])
    # outputs
    import xarray as xr

    rec = xr.load_dataset(res.path("checker_3km_P"))
    tru = xr.load_dataset(res.path("checker_3km_P", "true"))
    assert rec["vp"].shape == grid.shape_full == tru["vp"].shape
    np.testing.assert_allclose(rec["vp"], _bg(grid)["vp"], rtol=1e-5)  # unchanged fake model
    assert (tmp_path / "checker_3km_P" / "iterations.csv").exists()
    assert (tmp_path / "resolution_runs.json").exists()
    # compacted
    assert not list(tmp_path.rglob("frechet.dat"))
    # resume: successful tests are skipped, the failed one is retried
    n_inv = len(fakes["inv"])
    res2 = rn.run_resolution_tests(
        tests, base, grid, bg, cfg, tmp_path, n_jobs=8, chosen=(3.5, 7.0), seed=5
    )
    assert sorted(res2.skipped) == ["c2_P", "checker_3km_P"]
    assert len(fakes["inv"]) == n_inv + 1
    # different settings invalidate the marker
    res3 = rn.run_resolution_tests(
        tests[:1], base, grid, bg, cfg, tmp_path, n_jobs=2, chosen=(4.0, 7.0), seed=5
    )
    assert res3.skipped == [] and len(fakes["inv"]) == n_inv + 2


@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
def test_chosen_from_tradeoff_json_and_disk_guard(tmp_path, fakes) -> None:
    grid, base = _setup()
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "tradeoff.json").write_text(
        json.dumps({"chosen_damping": 2.5, "chosen_smoothing": 9.0})
    )
    cfg = _cfg()
    res = rn.run_resolution_tests(
        rn.build_tests(cfg), base, grid, _bg(grid), cfg, tmp_path / "w", 1, chosen=tmp_path / "t"
    )
    assert (res.chosen_damping, res.chosen_smoothing) == (2.5, 9.0)
    with pytest.raises(OSError, match="free"):
        rn.run_resolution_tests(
            rn.build_tests(cfg), base, grid, _bg(grid), cfg, tmp_path / "w2", 1,
            min_free_disk_gb=1e9,
        )  # fmt: skip


# ---------------------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------------------
def _have(*names: str) -> bool:
    try:
        for n in names:
            which_tool(n)
    except ToolNotFoundError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
@pytest.mark.skipif(
    not _have("fm3d", "frechgen", "invert3d", "residuals", "scorrect"), reason="binaries missing"
)
def test_checkerboard_end_to_end(tmp_path) -> None:
    import xarray as xr

    grid, base = _setup(n_src=8)
    bg = _bg(grid)
    cfg = _cfg(iterations=2, damping=0.01, smoothing=0.1)
    cfg = cfg.model_copy(update={"fmtomo": cfg.fmtomo.model_copy(update={"invert_velocity": True})})
    res = rn.run_resolution_tests(
        rn.build_tests(cfg), base, grid, bg, cfg, tmp_path, n_jobs=2, iterations=2
    )
    row = res.table.iloc[0]
    assert row["success"], row["flags"]
    assert row["misfit_final_s"] < row["misfit_initial_s"]
    true = xr.load_dataset(res.path(row["name"], "true"))
    rec = xr.load_dataset(res.path(row["name"]))
    interior = ~grid.cushion_mask().values
    dv = rec["dvp_pct"].values
    # where rays exist the recovered perturbation follows the true pattern
    corr = np.corrcoef(dv[interior], true["dvp_pct"].values[interior])[0, 1]
    assert 0 < np.abs(dv[interior]).max() and corr > 0.1, corr
    assert not list(tmp_path.rglob("frechet.dat")) and not list(tmp_path.rglob("rays.dat"))
