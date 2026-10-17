"""Data misfit, model variance and roughness (``osv.fmtomo.metrics``) on analytic grids."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.fmtomo import metrics as M
from osv.io.fmtomo import read_arrivals, read_observed, read_residuals, read_vgrids, write_vgrids
from osv.io.fmtomo.grids import VelocityGrids, velocity_dataset

E1 = Path(__file__).resolve().parents[2] / "tests" / "data" / "fmtomo" / "example1"
N = (8, 9, 10)


def _grid(values: np.ndarray, unc: float | None = 0.3, nc: int = 1):
    """Dataset from a ``(depth, lat, lon)`` array (depth ascending)."""
    unc_arr = None if unc is None else np.full(N, unc)
    return velocity_dataset(
        values[::-1],  # file order: deepest first
        r0_km=6371.0 - 7.0 * (N[0] - 1),
        dr_km=1.0,
        lat0_rad=-0.7,
        dlat_rad=0.01,
        lon0_rad=2.4,
        dlon_rad=0.012,
        uncertainty=None if unc_arr is None else unc_arr[::-1],
        n_cushion=nc,
    ).pipe(lambda ds: ds)


def _idx():
    k, j, i = np.meshgrid(*(np.arange(n, dtype=float) for n in N), indexing="ij")
    return k, j, i


# ---- data misfit -----------------------------------------------------------------------


def test_misfit_example1_matches_residuals_dat():
    obs = read_observed(E1 / "otimes.dat")
    arr = read_arrivals(E1 / "arrivals.dat")
    got = M.data_misfit(obs, arr)
    last = read_residuals(E1 / "residuals.dat").iloc[-1]
    assert got["n"] == 478  # 22 lost rays
    assert round(got["rms_ms"], 2) == last["rms_ms"]
    assert round(got["variance_s2"], 5) == last["variance_s2"]
    assert round(got["chi2_reduced"], 5) == last["chi2"]
    assert got["chi2"] == pytest.approx(got["chi2_reduced"] * got["n"])
    assert got["rms_s"] == pytest.approx(got["rms_ms"] / 1000)


def test_misfit_small_cases():
    g = M.data_misfit([1.0, 2.0, -1.0, 3.0], [1.1, 1.8, 5.0, 0.0], [0.1, 0.2, 1.0, 1.0])
    assert g["n"] == 2  # pairs with a non-positive time are skipped (residuals.f90:139)
    assert g["rms_s"] == pytest.approx(math.sqrt((0.01 + 0.04) / 2))
    assert g["variance_s2"] == pytest.approx(0.05)  # divided by n-1
    assert g["chi2"] == pytest.approx(1.0 + 1.0)
    assert g["chi2_reduced"] == pytest.approx(1.0)
    one = M.data_misfit([1.0], [2.0])
    assert math.isnan(one["variance_s2"])
    assert math.isnan(M.data_misfit([-1.0], [1.0])["rms_s"])
    with pytest.raises(ValueError):
        M.data_misfit([1.0], [1.0, 2.0])


# ---- Laplacian -------------------------------------------------------------------------


def test_constant_and_linear_have_zero_laplacian():
    k, j, i = _idx()
    for f in (np.full(N, 5.0), 2 + 0.3 * k - 0.7 * j + 0.1 * i):
        ds = _grid(f)
        assert np.allclose(M.laplacian_index(f), 0.0)
        assert M.model_roughness(ds, kind="index") == pytest.approx(0.0, abs=1e-20)
    # physical operator: linear in depth/lat/lon coordinates is not linear in the node spacing
    # only along z; test z-linear exactly
    z = np.broadcast_to(np.arange(N[0], dtype=float)[:, None, None], N)
    assert M.model_roughness(_grid(z), kind="physical") == pytest.approx(0.0, abs=1e-18)


def test_quadratic_index_value_and_boundary_rule():
    k, j, i = _idx()
    f = k**2  # D f = 2 where the depth axis is interior, else 0
    lap = M.laplacian_index(f)
    assert np.allclose(lap[1:-1], 2.0)
    assert np.allclose(lap[0], 0.0) and np.allclose(lap[-1], 0.0)
    ds = _grid(f)
    assert M.model_roughness(ds, kind="index") == pytest.approx(4.0)  # interior mean of 2**2
    # sum over axes: k^2 + 3 j^2 - i^2 -> 2 + 6 - 2 = 6 where all axes are interior
    g = k**2 + 3 * j**2 - i**2
    assert np.allclose(M.laplacian_index(g)[1:-1, 1:-1, 1:-1], 6.0)
    # a node at an end of one axis still gets the other axes (face node)
    assert M.laplacian_index(g)[0, 4, 4] == pytest.approx(6.0 - 2.0)


def test_quadratic_physical_value():
    kk = 0.05  # km/s per km^2: v = 0.5 * kk * depth^2
    ds0 = _grid(np.zeros(N))
    depth = ds0["depth"].values
    v = np.broadcast_to((0.5 * kk * depth**2)[:, None, None], N)
    ds = _grid(v)
    assert M.model_roughness(ds, kind="physical") == pytest.approx(kk**2, rel=1e-9)
    # reference given: roughness of the difference (constant offset drops out)
    ref = _grid(np.full(N, 3.0))
    assert M.model_roughness(ds, reference=ref, kind="physical") == pytest.approx(kk**2, rel=1e-9)


def test_physical_laplacian_horizontal_uses_km_spacing():
    ds = _grid(np.zeros(N))
    lat = np.deg2rad(ds["lat"].values)
    dy = (6371.0 - ds["depth"].values[3]) * 0.01
    # f = y^2 / 2 with y = r*(lat-lat0): laplacian = 1 along lat axis
    y = (lat - lat[0]) * (6371.0 - ds["depth"].values[3])
    f = np.broadcast_to((0.5 * y**2)[None, :, None], N).copy()
    lap = M.laplacian_physical(f, ds)
    # the depth-dependent spacing makes this exact only on the depth layer used for y
    assert lap[3, 4, 5] == pytest.approx(1.0, rel=1e-9)
    assert dy > 0


# ---- model variance --------------------------------------------------------------------


def test_model_variance_known_and_cushions_excluded():
    ref = _grid(np.full(N, 6.0))
    v = np.full(N, 6.0)
    v[2:-2, 2:-2, 2:-2] += 0.5  # interior perturbation
    v[0, :, :] += 100.0  # cushion layer: must not count
    ds = _grid(v)
    n_pert = (N[0] - 4) * (N[1] - 4) * (N[2] - 4)
    n_int = (N[0] - 2) * (N[1] - 2) * (N[2] - 2)
    assert M.model_variance(ds, ref) == pytest.approx(0.25 * n_pert / n_int)
    # weighted by sigma = 0.3
    assert M.model_variance(ds, ref, "reference") == pytest.approx(0.25 / 0.09 * n_pert / n_int)
    assert M.model_variance(ds, ref, 0.5) == pytest.approx(0.25 / 0.25 * n_pert / n_int)
    # mask='all' includes the cushion layer
    n_all = int(np.prod(N))
    expect = (0.25 * n_pert + 100.0**2 * N[1] * N[2]) / n_all
    # nodes of the 100-shift layer in the perturbed block do not overlap (layer 0 vs 2:-2)
    assert M.model_variance(ds, ref, mask="all") == pytest.approx(expect)
    custom = np.zeros(N, bool)
    custom[3, 3, 3] = True
    assert M.model_variance(ds, ref, mask=custom) == pytest.approx(0.25)
    assert M.model_variance(ds, ds) == 0.0


def test_roughness_ignores_cushion_spike_only_through_neighbours():
    f = np.zeros(N)
    f[0, 4, 4] = 10.0  # spike in the cushion layer
    ds = _grid(f)
    # its second difference touches interior node (1,4,4): -> mean over interior nonzero
    r = M.model_roughness(ds, kind="index")
    n_int = (N[0] - 2) * (N[1] - 2) * (N[2] - 2)
    assert r == pytest.approx(10.0**2 / n_int)
    assert M.model_roughness(ds, mask="all", kind="index") > 0


def test_variance_requires_matching_shapes_and_uncertainty():
    a, b = _grid(np.zeros(N)), _grid(np.zeros(N), unc=None)
    with pytest.raises(ValueError):
        M.model_variance(a, b, "reference")
    c = velocity_dataset(
        np.zeros((4, 4, 4)),
        r0_km=6000,
        dr_km=1,
        lat0_rad=0,
        dlat_rad=0.01,
        lon0_rad=0,
        dlon_rad=0.01,
    )
    with pytest.raises(ValueError):
        M.model_variance(a, c)


# ---- example1 grids and run directory --------------------------------------------------


def test_model_metrics_example1_zero_against_itself():
    vg = read_vgrids(E1 / "vgrids.in")
    out = M.model_metrics(vg, vg, start=vg)
    assert out["model_variance"] == 0.0
    assert out["pert_roughness_phys"] == 0.0
    assert out["obj_damping"] == 0.0 and out["obj_smoothing"] == 0.0
    assert out["roughness_phys"] >= 0.0


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "iterations").mkdir(parents=True)
    ref = read_vgrids(E1 / "vgrids.in")
    write_vgrids(run / "vgridsref.in", ref)
    shutil.copyfile(E1 / "invert3d.in", run / "invert3d.in")
    res = read_residuals(E1 / "residuals.dat")
    for it in range(3):
        d = run / "iterations" / f"{it:02d}"
        d.mkdir()
        grids = [[ds.copy(deep=True) for ds in row] for row in ref.grids]
        for row in grids:
            for ds in row:
                ds["velocity"] = ds["velocity"] + 0.1 * it
        write_vgrids(d / "vgrids.in", VelocityGrids(grids))
        r = res.iloc[it]
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "iter": it,
                    "rms_ms": r["rms_ms"],
                    "variance_s2": r["variance_s2"],
                    "chi2": r["chi2"],
                    "n_valid": 478,
                }
            )
        )
    return run


def test_iteration_metrics_and_attach(tmp_path):
    run = _fake_run(tmp_path)
    df = M.iteration_metrics(run)
    assert list(df.index) == [0, 1, 2]
    assert df["model_variance"].tolist() == pytest.approx([0.0, 0.01, 0.04], abs=1e-9)
    # constant shift: no change in roughness of the perturbation of v
    assert df["model_variance_start"].tolist() == pytest.approx([0.0, 0.01, 0.04], abs=1e-9)
    assert df["rms_s"].iloc[1] == pytest.approx(1.05224, rel=1e-4)
    # constant shift has zero Laplacian
    assert df["pert_roughness_index"].tolist() == pytest.approx([0.0] * 3, abs=1e-12)
    # damping sum = n_nodes * dm^2 / sigma^2 * (epsilon*epsv = 1)
    n_nodes = sum(ds["velocity"].size for _, _, ds in read_vgrids(E1 / "vgrids.in"))
    assert df["obj_damping"].iloc[1] == pytest.approx(n_nodes * 0.01 / 0.09, rel=1e-6)

    class R:
        run_dir = run
        iterations = pd.DataFrame({"iter": [0, 1, 2], "rms_ms": [1.0, 2.0, 3.0]})

    out = M.attach_metrics(R)  # type: ignore[arg-type]
    assert "model_variance" in out.columns and out["rms_ms"].tolist() == [1.0, 2.0, 3.0]
    assert out["rms_s"].iloc[0] == pytest.approx(2.17477)
