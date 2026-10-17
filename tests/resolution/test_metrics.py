"""Tests for osv.resolution.metrics (recovery metrics, smearing, resolvable mask)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.config.schema import OSVConfig, ResolutionConfig, ResolutionMaskConfig
from osv.resolution import metrics as M

SP = (0.5, 0.5, 0.5)
N = 24


def pattern(cell=2.0, n=N):
    ax = np.arange(n) * 0.5
    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")
    k = np.pi / cell
    return 5.0 * np.sin(k * x) * np.sin(k * y) * np.sin(k * z), (z, y, x)


def test_perfect_zero_half():
    t, _ = pattern()
    loc = M.local_metrics(t, t, SP, 2.0)
    c = loc["corr"][4:-4, 4:-4, 4:-4]
    assert np.nanmin(c) == pytest.approx(1.0)
    assert np.nanmin(loc["amp"][4:-4, 4:-4, 4:-4]) == pytest.approx(1.0)
    assert np.nanmin(loc["rms_ratio"][4:-4, 4:-4, 4:-4]) == pytest.approx(1.0)
    assert np.nanmin(loc["sign_agree"][4:-4, 4:-4, 4:-4]) == pytest.approx(1.0)
    half = M.local_metrics(t, 0.5 * t, SP, 2.0)
    assert np.nanmax(np.abs(half["amp"] - 0.5)[4:-4, 4:-4, 4:-4]) < 1e-9
    assert np.nanmin(half["corr"][4:-4, 4:-4, 4:-4]) == pytest.approx(1.0)
    zero = M.local_metrics(t, np.zeros_like(t), SP, 2.0)
    assert np.nanmax(zero["corr"]) == 0.0
    assert np.nanmax(np.abs(zero["amp"])) < 1e-9
    inv = M.local_metrics(t, -t, SP, 2.0)
    assert np.nanmax(inv["corr"]) == pytest.approx(-1.0)


def test_global_metrics():
    t, _ = pattern()
    g = M.global_metrics(t, t)
    assert (g["corr"], g["semblance"], g["rms_ratio"]) == pytest.approx((1.0, 1.0, 1.0))
    g0 = M.global_metrics(t, 0 * t)
    assert g0["semblance"] == pytest.approx(0.5) and g0["rms_ratio"] == 0.0
    assert M.global_metrics(t, -t)["semblance"] == pytest.approx(0.0)
    m = np.zeros(t.shape, bool)
    m[:5] = True
    r = t.copy()
    r[5:] = 0
    assert M.global_metrics(t, r, m)["corr"] == pytest.approx(1.0)
    assert M.global_metrics(t, 0.5 * t)["amp"] == pytest.approx(0.5)


def test_window_weighting_and_invalid():
    # left half recovered perfectly, right half inverted: correlation changes sign across
    t, _ = pattern(n=32)
    r = t.copy()
    r[..., 16:] *= -1
    loc = M.local_metrics(t, r, SP, 1.5)
    assert loc["corr"][16, 16, 3] > 0.99
    assert loc["corr"][16, 16, 28] < -0.99
    # a wide window mixes both
    wide = M.local_metrics(t, r, SP, 12.0)
    assert abs(wide["corr"][16, 16, 16]) < 0.7
    # Gaussian weighting: nearby nodes dominate (nearer the good side -> higher corr)
    assert wide["corr"][16, 16, 8] > wide["corr"][16, 16, 16]
    # invalid nodes are excluded and NaN
    valid = np.ones(t.shape, bool)
    valid[:, :, 28:] = False
    loc2 = M.local_metrics(t, r, SP, 1.5, valid)
    assert np.isnan(loc2["corr"][16, 16, 30])
    assert loc2["corr"][16, 16, 3] > 0.99
    with pytest.raises(ValueError):
        M.local_metrics(t, r[:5], SP, 2.0)
    with pytest.raises(ValueError):
        M.local_metrics(t, r, SP, 0.0)


def _spike(sig, n=33, centre=(8.0, 8.0, 8.0)):
    ax = np.arange(n) * 0.5
    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")  # x east (axis 2), y north, z down
    c = centre
    q = sum(((p - ci) / s) ** 2 for p, ci, s in zip((x, y, z), c, sig, strict=True))
    return 8.0 * np.exp(-0.5 * q), (x, y, z)


def test_spike_smearing_principal_axis():
    true, (x, y, z) = _spike((0.6, 0.6, 0.6))
    # recovered: elongated along azimuth 45 deg (NE), shifted 0.5 km east
    ang = np.radians(45.0)
    u = (x - 8.5) * np.sin(ang) + (y - 8.0) * np.cos(ang)  # along strike
    v = (x - 8.5) * np.cos(ang) - (y - 8.0) * np.sin(ang)
    rec = 3.0 * np.exp(-0.5 * ((u / 2.0) ** 2 + (v / 0.7) ** 2 + ((z - 8.0) / 0.7) ** 2))
    res = M.spike_smearing(true, rec, x, y, z, centre=(8.0, 8.0, 8.0), search_radius_km=5.0)
    assert res["major_azimuth_deg"] == pytest.approx(45.0, abs=2.0)
    assert res["major_plunge_deg"] == pytest.approx(0.0, abs=2.0)
    assert res["rec_sigma_km"][0] == pytest.approx(2.0, rel=0.1)
    assert res["true_sigma_km"][0] == pytest.approx(0.6, rel=0.1)
    assert res["centroid_offset_km"][0] == pytest.approx(0.5, abs=0.1)
    assert res["centroid_offset_km"][1] == pytest.approx(0.0, abs=0.1)
    assert res["elongation"] > 2.5
    assert res["peak_ratio"] < 0.5
    # vertical smear: plunge 90
    rec2 = 3.0 * np.exp(-0.5 * (((x - 8) / 0.7) ** 2 + ((y - 8) / 0.7) ** 2 + ((z - 8) / 2.0) ** 2))
    r2 = M.spike_smearing(true, rec2, x, y, z, search_radius_km=5.0)
    assert r2["major_plunge_deg"] == pytest.approx(90.0, abs=2.0)
    # nothing recovered
    r0 = M.spike_smearing(true, 0 * rec2, x, y, z, search_radius_km=5.0)
    assert np.isnan(r0["offset_km"])


def test_pattern_anisotropy():
    t, _ = pattern(2.0)
    a = M.pattern_anisotropy(t, t, SP)
    assert a["retention_x"] == pytest.approx(1.0) and a["anisotropy"] == pytest.approx(1.0)
    # smooth along z only
    from scipy import ndimage

    r = ndimage.gaussian_filter1d(t, 1.5, axis=0)
    a = M.pattern_anisotropy(t, r, SP)
    assert a["weakest_axis"] == "z"
    assert a["retention_z"] < a["retention_x"]
    assert a["anisotropy"] > 1.0


def test_mask_logic():
    shape = (2, 2, 4)
    f = lambda c, a: {"corr": np.full(shape, c, float), "amp": np.full(shape, a, float)}  # noqa: E731
    big = {"corr": np.full(shape, 0.9), "amp": np.full(shape, 0.8)}
    small = {
        "corr": np.array([[[0.9, 0.9, 0.2, np.nan]] * 2] * 2),
        "amp": np.array([[[0.8, 0.1, 0.8, 0.8]] * 2] * 2),
    }
    res, scale = M.resolvable_from_checkerboards({3.0: [big], 1.5: [small]})
    assert res[0, 0].tolist() == [True, True, True, True]
    # smallest passing size: node0 passes 1.5; nodes 1-3 fail small (amp / corr / nan) -> 3.0
    assert scale[0, 0].tolist() == [1.5, 3.0, 3.0, 3.0]
    # only small cells given: failing nodes are False / NaN
    res, scale = M.resolvable_from_checkerboards({1.5: [small]})
    assert res[0, 0].tolist() == [True, False, False, False]
    assert np.isnan(scale[0, 0, 1])
    # zero recovery -> all False
    res, scale = M.resolvable_from_checkerboards({2.0: [f(0.0, 0.0)]})
    assert not res.any() and np.isnan(scale).all()
    # thresholds
    res, _ = M.resolvable_from_checkerboards({2.0: [f(0.6, 0.35)]}, min_corr=0.7)
    assert not res.any()
    res, _ = M.resolvable_from_checkerboards({2.0: [f(0.6, 0.35)]})
    assert res.all()
    # offset copies are averaged
    res, _ = M.resolvable_from_checkerboards({2.0: [f(0.9, 0.9), f(0.0, 0.0)]})
    assert not res.any()
    # combination with coverage / valid
    cov = np.zeros(shape, bool)
    cov[:, :, :2] = True
    res, scale = M.resolvable_from_checkerboards({3.0: [big]}, coverage=cov)
    assert res[..., :2].all() and not res[..., 2:].any() and np.isnan(scale[..., 2:]).all()


# --------------------------------------------------------------------- driver
class FakeGrid:
    dz_km = dy_km = dx_km = 0.5
    n_cushion = 1

    def __init__(self, n=14):
        self.depth_full = np.arange(n) * 0.5
        self.lat_full = -38.5 + np.arange(n) * 0.0045
        self.lon_full = 176.2 + np.arange(n) * 0.0057
        self.shape_full = (n, n, n)
        self.centre_lat, self.centre_lon = -38.5 + 0.03, 176.2 + 0.04

    def cushion_mask(self):
        m = np.zeros(self.shape_full, bool)
        m[0] = m[-1] = m[:, 0] = m[:, -1] = m[:, :, 0] = m[:, :, -1] = True
        return xr.DataArray(m, dims=M.DIMS)


def test_run_metrics(tmp_path):
    g = FakeGrid()
    coords = {"depth": g.depth_full, "lat": g.lat_full, "lon": g.lon_full}
    rows = []
    t, _ = pattern(1.5, n=14)
    for name, amp in (("checker_1p5km_P", 1.0), ("checker_3km_P", 0.0)):
        tp = tmp_path / f"{name}_true.nc"
        rp = tmp_path / f"{name}_rec.nc"
        tt = t if "1p5" in name else pattern(3.0, n=14)[0]
        xr.Dataset({"dvp_pct": (M.DIMS, tt)}, coords=coords).to_netcdf(tp)
        xr.Dataset({"dvp_pct": (M.DIMS, amp * tt)}, coords=coords).to_netcdf(rp)
        cell = 1.5 if "1p5" in name else 3.0
        rows.append(
            dict(
                name=name,
                kind="checkerboard",
                phase="P",
                params={"cell_km": cell},
                success=True,
                true_model=str(tp),
                recovered_model=str(rp),
            )
        )
    res = type("R", (), {"workdir": tmp_path, "table": pd.DataFrame(rows)})()
    cfg = OSVConfig.model_construct(resolution=ResolutionConfig())
    cov = xr.DataArray(np.ones(g.shape_full, bool), dims=M.DIMS)
    cov[:, :, 9:] = False
    out = M.run_metrics(res, g, cfg, coverage=cov)
    rm = out["resolvable"]
    assert rm["vp"].dtype == bool
    r = rm["vp"].values
    assert r[7, 7, 6] and not r[7, 7, 11]  # perfect 1.5 km recovery, but DWS-masked
    assert not r[0].any()
    assert float(rm["resolved_scale_km"].values[7, 7, 6]) == 1.5
    assert out["paths"]["resolvable"].exists() and out["paths"]["metrics"].exists()
    s = out["summary"].set_index("name")
    assert s.loc["checker_1p5km_P", "global_corr"] == pytest.approx(1.0)
    assert s.loc["checker_3km_P", "global_rms_ratio"] == 0.0
    back = xr.open_dataset(out["paths"]["resolvable"])
    assert back["vp"].values.dtype == bool
    # consumable by plotting
    from osv.plotting.slices import ModelMasks

    dws = xr.DataArray(np.ones(g.shape_full), dims=M.DIMS, coords=coords)
    mm = ModelMasks.from_coverage(dws, 0.5, back)
    assert mm.resolvable_applied
    # without combine_with_dws the mask ignores coverage
    cfg2 = ResolutionConfig(mask=ResolutionMaskConfig(combine_with_dws=False))
    out2 = M.run_metrics(res, g, cfg2, tmp_path / "m2", coverage=cov)
    assert out2["resolvable"]["vp"].values[7, 7, 11]


def test_schema_defaults():
    m = ResolutionConfig().mask
    assert (m.min_corr, m.min_amplitude, m.window_km, m.combine_with_dws) == (0.5, 0.3, None, True)
