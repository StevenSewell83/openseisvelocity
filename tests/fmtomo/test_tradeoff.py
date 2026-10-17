"""Trade-off sweeps and L-curve corner picking (``osv.fmtomo.tradeoff``)."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import FmtomoConfig
from osv.fmtomo import tradeoff as to
from osv.fmtomo.controls import ModelInfo
from osv.fmtomo.invert import InversionResult
from osv.fmtomo.rundir import RunInputs
from osv.io.fmtomo import (
    read_interfaces,
    read_observed,
    read_propgrid,
    read_receivers,
    read_sources,
    read_vgrids,
)

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "tests" / "data" / "fmtomo" / "example1"
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))
WORK = Path(os.environ.get("OSV_TEST_WORKDIR", REPO / "build" / "test-runs")) / "osv_lg54"


def _inputs() -> RunInputs:
    return RunInputs(
        read_vgrids(E1 / "vgrids.in"),
        read_interfaces(E1 / "interfaces.in"),
        read_propgrid(E1 / "propgrid.in"),
        read_sources(E1 / "sourcesref.in"),
        read_receivers(E1 / "receivers.in"),
        read_observed(E1 / "otimes.dat"),
    )


def _cfg(**tc) -> FmtomoConfig:
    return FmtomoConfig(
        package="base",
        relocate="linear",
        damping=10.0,
        smoothing=5.0,
        iterations=2,
        subspace_dimension=15,
        tradeoff=tc,
    )


# ---------------------------------------------------------------------------------------
# corner picking
# ---------------------------------------------------------------------------------------
def _L(p, corner):
    """Symmetric L in log-log space: vertical branch for p << corner, horizontal for p >> corner."""
    return 1.0 + (corner / p) ** 2, 1.0 + (p / corner) ** 2


def test_menger_curvature_circle():
    th = np.array([0.0, 0.1, 0.2])
    pts = np.column_stack([np.cos(th), np.sin(th)]) * 2.0
    assert to.menger_curvature(*pts) == pytest.approx(0.5, rel=1e-6)  # left turn, r = 2
    assert to.menger_curvature(*pts[::-1]) == pytest.approx(-0.5, rel=1e-6)
    assert to.menger_curvature(pts[0], pts[0], pts[0]) == 0.0


@pytest.mark.parametrize("corner", [0.5, 5.0, 50.0])
def test_lcurve_corner_known_corner(corner):
    p = np.logspace(-2, 4, 41)
    # L in (x = model, y = data): data flat while model is large, rises as model -> 0
    model, data = _L(p, corner)
    cp = to.lcurve_corner(p, model, data)
    assert cp.index is not None
    assert corner / 3 <= cp.value <= corner * 3
    assert np.isnan(cp.curvature[0]) and np.isnan(cp.curvature[-1])


def test_lcurve_corner_order_independent():
    p = np.logspace(-1, 3, 17)
    model, data = _L(p, 3)
    a = to.lcurve_corner(p, model, data)
    perm = np.random.default_rng(1).permutation(len(p))
    b = to.lcurve_corner(p[perm], model[perm], data[perm])
    assert a.value == b.value


def test_lcurve_corner_scale_invariant():
    p = np.logspace(-1, 3, 17)
    model, data = _L(p, 3)
    assert (
        to.lcurve_corner(p, model, data).value
        == to.lcurve_corner(p, 1e4 * model, 1e-3 * data).value
    )


def test_lcurve_corner_three_points_picks_middle():
    cp = to.lcurve_corner([1, 10, 100], [1.0, 0.1, 0.09], [1.0, 1.1, 10.0])
    assert cp.value == 10


def test_lcurve_corner_few_points():
    for n in (0, 1, 2):
        cp = to.lcurve_corner(np.arange(1, n + 1), np.ones(n), np.ones(n))
        assert cp.index is None and cp.value is None and "need 3" in cp.reason


def test_lcurve_corner_straight_line_has_no_corner():
    p = np.array([1.0, 10.0, 100.0, 1000.0])
    cp = to.lcurve_corner(p, 10.0 ** -np.arange(4), 10.0 ** np.arange(4))
    assert cp.index is None and "no convex corner" in cp.reason


def test_lcurve_corner_drops_non_monotone_points():
    p = np.logspace(-2, 4, 41)
    model, data = _L(p, 3)
    clean = to.lcurve_corner(p, model, data)
    # a noisy run: both model norm and misfit worse than its neighbours' trade-off
    p2 = np.append(p, 7.0)
    cp = to.lcurve_corner(p2, np.append(model, 5.0), np.append(data, 500.0))
    assert not cp.used[-1]
    assert cp.value == clean.value


def test_lcurve_corner_ignores_invalid_values():
    p = np.logspace(-2, 4, 41)
    model, data = _L(p, 3)
    ref = to.lcurve_corner(p, model, data).value
    model2, data2 = model.copy(), data.copy()
    model2[4] = np.nan
    data2[9] = -1.0
    cp = to.lcurve_corner(p, model2, data2)
    assert not cp.used[4] and not cp.used[9]
    assert cp.value is not None and ref / 3 < cp.value < ref * 3
    with pytest.raises(ValueError):
        to.lcurve_corner([1, 2], [1], [1, 2])


# ---------------------------------------------------------------------------------------
# orchestration with a fake run_inversion
# ---------------------------------------------------------------------------------------
@pytest.fixture
def fake(monkeypatch):
    """Replace the inversion by an analytic L-curve (corner near damping 20, smoothing 50)."""
    calls: list[dict] = []

    def run_inversion(run_dir, controls, n_iter=None, n_jobs=1, keep_iterations=True, **kw):
        run = Path(run_dir)
        assert (run / "vgridsref.in").exists()  # a real assembled run directory
        calls.append(
            {"dir": run.name, "n_iter": n_iter, "n_jobs": n_jobs, "mem": kw.get("mem_per_job_gb")}
        )
        (run / "frechet.dat").write_text("x" * 100)
        (run / "iterations").mkdir(exist_ok=True)
        return InversionResult(
            run_dir=run,
            iterations=pd.DataFrame(
                {
                    "iter": [0, 1],
                    "rms_ms": [100.0, 50.0],
                    "variance_s2": [1e-2, 2.5e-3],
                    "chi2": [3.0, 1.0],
                    "n_valid": [10.0, 10.0],
                }
            ),
            hypocentres=pd.DataFrame(),
            final_model=None,
            final_sources=None,
            residuals_path=run / "residuals.dat",
            wall_s=0.1,
        )

    def attach(res):
        name = res.run_dir.name
        d = float(name.split("_")[0][1:])
        s = float(name.split("_")[1][1:])
        # damping: L in (model var, data var) with corner at d = 20; smoothing: corner at s = 50
        mv, dv_d = _L(d, 20.0)
        rg, dv_s = _L(s, 50.0)
        dv = dv_d * dv_s * 1e-3
        mv, rg = mv * 1e-3, rg * 1e-3
        res.iterations.loc[
            res.iterations.index[-1],
            [
                "variance_s2",
                "model_variance",
                "pert_roughness_phys",
                "pert_roughness_index",
                "roughness_phys",
                "roughness_index",
            ],
        ] = [dv, mv, rg, rg, rg, rg]
        return res.iterations

    monkeypatch.setattr(to._inv, "run_inversion", run_inversion)
    monkeypatch.setattr(to, "_attach", attach)
    return calls


_D = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
_S = [1, 5, 10, 20, 50, 100, 200, 500, 1000]


def test_sequential_sweep(tmp_path, fake):
    cfg = _cfg(damping=_D, smoothing=_S, iterations=2)
    cfg = cfg.model_copy(update={"smoothing": 5.0})
    res = to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    names = sorted(c["dir"] for c in fake)
    assert len(fake) == len(_D) + len(_S) - 1  # (d*, s=5) is in both sweeps
    assert all(c["n_iter"] == 2 for c in fake)
    assert res.mode == "sequential"
    assert res.chosen_damping in _D and res.chosen_smoothing in _S
    assert res.picks["damping"]["source"] == "max_curvature"
    assert 20 / 4 <= res.chosen_damping <= 20 * 4
    assert 50 / 4 <= res.chosen_smoothing <= 50 * 4
    assert (tmp_path / "tradeoff.csv").exists()
    js = json.loads((tmp_path / "tradeoff.json").read_text())
    assert (
        js["chosen_damping"] == res.chosen_damping
        and js["chosen_smoothing"] == res.chosen_smoothing
    )
    assert len(js["table"]) == len(res.table) == len(names)
    assert to.load_chosen(tmp_path) == (res.chosen_damping, res.chosen_smoothing)
    assert res.picks["damping"]["manual"] == 10.0
    # heavy files removed, summaries written
    run = tmp_path / names[0]
    assert not (run / "frechet.dat").exists() and (run / "summary.json").exists()
    assert set(to.TABLE_COLUMNS) <= set(res.table.columns)
    assert res.table["success"].all()


def test_resume_skips_completed(tmp_path, fake):
    cfg = _cfg(damping=_D, smoothing=_S, iterations=2)
    first = to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    n = len(fake)
    again = to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    assert len(fake) == n  # nothing re-run
    assert again.chosen_damping == first.chosen_damping
    # a different iteration count invalidates the summaries
    to.run_tradeoff(_inputs(), _cfg(damping=_D[:3], smoothing=_S[:3], iterations=3), tmp_path, 1)
    assert len(fake) > n
    # an interrupted run (no summary) is redone
    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    fake.clear()
    to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    victim = next(tmp_path.glob("d20_s*"))
    (victim / "summary.json").unlink()
    fake.clear()
    to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    assert [c["dir"] for c in fake] == [victim.name]


def test_full_grid_and_concurrency(tmp_path, fake):
    cfg = _cfg(damping=_D, smoothing=[5, 50, 500], mode="full_grid", iterations=1)
    res = to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=16, mem_per_job_gb=1.0)
    assert len(fake) == len(_D) * 3 == len(res.table)
    assert all(c["n_jobs"] == 16 // 2 for c in fake)  # n_jobs // 8 = 2 concurrent runs
    assert all(c["mem"] == 2.0 for c in fake)
    assert res.picks["smoothing"]["at_damping"] == res.chosen_damping or res.chosen_damping in _D
    assert res.picks["damping"]["at_smoothing"] == 5  # nearest grid value to fmtomo.smoothing


def test_manual_pick_reports_both(tmp_path, fake):
    cfg = _cfg(damping=_D, smoothing=_S, pick="manual")
    res = to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    assert res.chosen_damping == 10.0 and res.chosen_smoothing == 5.0
    assert res.picks["damping"]["source"] == "manual"
    assert res.picks["damping"]["max_curvature"] is not None  # corner still reported
    assert res.picks["smoothing"]["max_curvature"] is not None


def test_too_few_points_falls_back_to_manual(tmp_path, fake):
    cfg = _cfg(damping=[5, 50], smoothing=[1, 10], iterations=1)
    res = to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    assert res.chosen_damping == 10.0 and res.chosen_smoothing == 5.0
    assert res.picks["damping"]["source"] == "manual (fallback)"
    assert any("no L-curve corner" in f for f in res.flags)


def test_failed_run_is_recorded(tmp_path, fake, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("fm3d died")

    monkeypatch.setattr(to._inv, "run_inversion", boom)
    cfg = _cfg(damping=[1, 5, 10], smoothing=[1, 5, 10])
    res = to.run_tradeoff(_inputs(), cfg, tmp_path, n_jobs=1)
    assert not res.table["success"].any()
    assert any("failed" in f for f in res.flags)
    assert res.chosen_damping == 10.0  # fallback


# ---------------------------------------------------------------------------------------
# real programs
# ---------------------------------------------------------------------------------------
@pytest.fixture
def workdir():
    """Unique scratch directory under build/ (safe for concurrent pytest runs)."""
    import tempfile

    WORK.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="run-", dir=WORK))
    yield d
    shutil.rmtree(d, ignore_errors=True)
    for p in (WORK, WORK.parent):
        try:
            p.rmdir()
        except OSError:
            pass


@pytest.mark.integration
def test_example1_sweep(workdir):
    for e in ("fm3d", "frechgen", "invert3d", "residuals", "scorrect"):
        if not (BIN / e).exists():
            pytest.skip(f"{BIN / e} not built")
    WORK_ = workdir
    cfg = _cfg(damping=[0.1, 10.0, 1000.0], smoothing=[5.0], iterations=2)
    cfg = cfg.model_copy(update={"invert_velocity": True})
    res = to.run_tradeoff(
        _inputs(), cfg, WORK_, n_jobs=3, model_info=ModelInfo.from_model(_inputs().grids)
    )
    t = res.table.sort_values("damping")
    assert t["success"].all(), t["flags"].tolist()
    assert len(t) >= 3
    assert (t["n_iter"] == 2).all()
    # stronger damping -> smaller model perturbation
    assert t["model_variance"].iloc[-1] == t["model_variance"].min()
    assert (t["model_variance"] > 0).all()
    assert np.isfinite(t[["variance_s2", "chi2_reduced", "pert_roughness_phys"]]).all().all()
    assert math.isfinite(res.chosen_damping)
    assert (WORK_ / "tradeoff.json").exists()
    assert not list(WORK_.glob("*/frechet.dat"))
