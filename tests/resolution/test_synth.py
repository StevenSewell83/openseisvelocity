"""Tests for osv.resolution.synth (synthetic data with real geometry and pick-based noise)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import Model3DConfig, ResolutionConfig
from osv.io.fmtomo import PathSpec, Receivers, Source, read_observed, read_sources
from osv.model3d.grid import define_grid
from osv.resolution.models import checkerboard
from osv.resolution.synth import (
    BaseInputs,
    make_synthetic_data,
    perturb_sources,
    synthetic_observed,
)
from osv.tools import ToolNotFoundError, which_tool

KEY = ["receiver", "source", "ray", "subray"]


def _obs(n: int, unc: float | np.ndarray = 0.05, invalid: tuple[int, ...] = ()) -> pd.DataFrame:
    t = np.full(n, 3.0)
    t[list(invalid)] = -100.0
    return pd.DataFrame(
        {
            "receiver": np.arange(1, n + 1),
            "source": 1,
            "ray": 1,
            "subray": 0,
            "time": t,
            "uncertainty": unc,
        }
    )


def _calc(obs: pd.DataFrame, t: float = 2.0) -> pd.DataFrame:
    c = obs[KEY].copy()
    c["time"] = t
    c["diffracted"] = False
    c["headwave"] = False
    return c


def test_noise_statistics_from_pick_uncertainty() -> None:
    n = 40000
    unc = np.where(np.arange(n) % 2, 0.02, 0.10)
    obs = _obs(n, unc)
    rng = np.random.default_rng(7)
    noisy, clean = synthetic_observed(
        obs, _calc(obs), scale=0.5, from_pick_uncertainty=True, sigma_s=0.02, rng=rng
    )
    z = noisy["noise"].to_numpy() / (0.5 * unc)
    assert abs(z.mean()) < 0.02
    assert z.std() == pytest.approx(1.0, abs=0.02)
    # per class standard deviation follows the pick uncertainty
    assert noisy["noise"][unc == 0.10].std() == pytest.approx(0.05, rel=0.03)
    assert noisy["noise"][unc == 0.02].std() == pytest.approx(0.01, rel=0.03)
    np.testing.assert_allclose(noisy["time"], 2.0 + noisy["noise"])
    np.testing.assert_array_equal(noisy["uncertainty"], unc)
    assert (clean["time"] == 2.0).all()


def test_constant_sigma_and_seed_reproducible() -> None:
    obs = _obs(20000)
    kw = {"scale": 1.0, "from_pick_uncertainty": False, "sigma_s": 0.03}
    a, _ = synthetic_observed(obs, _calc(obs), rng=np.random.default_rng(1), **kw)
    b, _ = synthetic_observed(obs, _calc(obs), rng=np.random.default_rng(1), **kw)
    c, _ = synthetic_observed(obs, _calc(obs), rng=np.random.default_rng(2), **kw)
    assert a["noise"].std() == pytest.approx(0.03, rel=0.03)
    assert abs(a["noise"].mean()) < 0.001
    np.testing.assert_array_equal(a["noise"], b["noise"])
    assert not np.array_equal(a["noise"], c["noise"])
    assert (a["uncertainty"] == 0.05).all()  # uncertainty column untouched


def test_zero_scale_is_noiseless() -> None:
    obs = _obs(50)
    noisy, _ = synthetic_observed(
        obs, _calc(obs), scale=0.0, from_pick_uncertainty=True, sigma_s=0.02,
        rng=np.random.default_rng(0),
    )  # fmt: skip
    assert (noisy["time"] == 2.0).all()


def test_invalid_handling() -> None:
    obs = _obs(10, invalid=(2, 5))
    calc = _calc(obs)
    calc = calc.drop(index=7)  # fm3d row missing
    calc.loc[calc["receiver"] == 4, "time"] = -1.0  # fm3d invalid ray
    noisy, clean = synthetic_observed(
        obs, calc, scale=1.0, from_pick_uncertainty=True, sigma_s=0.02, rng=np.random.default_rng(3)
    )
    bad = {2, 5, 7, 3}  # zero based rows: real invalid 2,5; missing 7; fm3d -1 at receiver 4
    for i in range(10):
        if i in bad:
            assert not noisy["valid"][i]
            assert noisy["time"][i] == -100.0 and clean["time"][i] == -100.0
            assert noisy["noise"][i] == 0.0
        else:
            assert noisy["valid"][i] and noisy["time"][i] > 0
    # the stream of random numbers does not depend on validity
    other, _ = synthetic_observed(
        _obs(10), _calc(_obs(10)), scale=1.0, from_pick_uncertainty=True, sigma_s=0.02,
        rng=np.random.default_rng(3),
    )  # fmt: skip
    ok = [i for i in range(10) if i not in bad]
    np.testing.assert_array_equal(noisy["noise"][ok], other["noise"][ok])


def test_perturbation_statistics() -> None:
    n = 20000
    src = [Source(3.0, -38.5, 176.2, [PathSpec.direct()], uncertainty=(1.0, 0.01, 0.01))] * n
    new, sh = perturb_sources(src, 0.5, 0.2, np.random.default_rng(5))
    assert abs(sh[:, 0].mean()) < 0.02 and abs(sh[:, 2].mean()) < 0.01
    assert sh[:, 0].std() == pytest.approx(0.5, rel=0.03)
    assert sh[:, 1].std() == pytest.approx(0.5, rel=0.03)
    assert sh[:, 2].std() == pytest.approx(0.2, rel=0.03)
    # shifts consistent with the new coordinates; other attributes kept
    km = 6371 * np.pi / 180
    assert (new[10].lat - src[0].lat) * km == pytest.approx(sh[10, 1])
    assert new[10].depth_km - 3.0 == pytest.approx(sh[10, 2])
    assert new[0].uncertainty == (1.0, 0.01, 0.01) and new[0].paths == src[0].paths
    same, sh0 = perturb_sources(src[:5], 0.0, 0.0, np.random.default_rng(5))
    assert np.all(sh0 == 0) and [s.lat for s in same] == [src[0].lat] * 5


# ---------------------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------------------
def _have_fm3d() -> bool:
    try:
        which_tool("fm3d")
    except ToolNotFoundError:
        return False
    return True


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
    rec_rays = rays.assign(ray=rays.groupby("receiver").cumcount() + 1)
    obs = pd.DataFrame(
        {
            "receiver": rec_rays["receiver"],
            "source": rec_rays["source"],
            "ray": rec_rays["ray"],
            "subray": 0,
            "time": 1.0,
            "uncertainty": 0.04,
        }
    )
    obs = obs.sort_values(["source", "receiver", "ray"]).reset_index(drop=True)
    obs.loc[3, "time"] = -100.0
    base = BaseInputs(grid.flat_interfaces(), grid.propgrid, src, rec, obs)
    return grid, base


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
@pytest.mark.skipif(not _have_fm3d(), reason="fm3d binary not built")
def test_make_synthetic_data_checkerboard(tmp_path) -> None:
    grid, base = _setup()
    bg = checkerboard(grid, 5.0, 2.0, 0.0)  # zero amplitude = homogeneous-gradient background
    true = checkerboard(grid, 5.0, 2.0, 8.0)
    cfg = ResolutionConfig(noise={"scale": 1.0}, hypocentre_noise_km={"xy": 0.3, "z": 0.2})
    res = make_synthetic_data(
        true, grid, base, tmp_path / "s", cfg, seed=11, n_jobs=2, background_model=bg
    )
    st = res.summary()
    n = len(base.observed)
    assert st["n_obs"] == n and st["n_invalid"] == 1 and st["n_valid"] == n - 1
    assert st["noise_std_s"] == pytest.approx(0.04, rel=0.35)
    # 8 % anomalies over a few km paths: signal of order 10-100 ms, well above zero
    assert 0.002 < st["signal_rms_s"] < 0.5
    assert st["snr"] > 0
    assert st["hypocentres_perturbed"] and st["hypocentre_shift_rms_km"]["xy"] > 0
    # files
    out = read_observed(res.otimes)
    assert len(out) == n and out["uncertainty"].eq(0.04).all()
    assert out.loc[3, "time"] == -100.0
    clean = read_observed(res.otimes_noiseless)
    d = (out["time"] - clean["time"])[out["valid"]]
    assert d.abs().max() > 0
    # true hypocentres were used for the times; inversion start differs
    true_src, start = read_sources(res.true_sources), read_sources(res.sources)
    assert any(abs(a.lat - b.lat) > 0 for a, b in zip(true_src, start, strict=True))
    assert [s.lat for s in true_src] == pytest.approx([s.lat for s in base.sources], abs=1e-5)
    # reproducible with the same seed
    res2 = make_synthetic_data(
        true, grid, base, tmp_path / "s2", cfg, seed=11, n_jobs=1,
        background_arrivals=None,
    )  # fmt: skip
    assert res2.otimes.read_text() == res.otimes.read_text()
    assert np.isnan(res2.summary()["signal_rms_s"])


@pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")
@pytest.mark.parametrize(("phase", "vtype", "n_types"), [("P", 1, 1), ("S", 1, 1), ("S", 2, 2)])
def test_forward_times_velocity_type_matches_the_paths(
    tmp_path, monkeypatch, phase, vtype, n_types
) -> None:
    """S with type-1 paths (separate layout) must run on Vs, not on Vp."""
    from dataclasses import replace

    from osv.io.fmtomo import read_vgrids
    from osv.resolution import synth

    grid, base = _setup()
    model = checkerboard(grid, 5.0, 2.0, 0.0)
    model["vs"] = model["vp"] / 1.8
    src = [replace(s, paths=[PathSpec.direct(vtype)]) for s in base.sources]
    seen = {}

    def fake_fm3d(inputs, sources, receivers, n_jobs, workdir):
        vg = read_vgrids(inputs / "vgrids.in")
        seen["n"] = vg.n_types
        seen["v1"] = float(np.median(vg.get(1, 1)["velocity"].values))
        raise RuntimeError("stop")

    monkeypatch.setattr(synth, "run_fm3d_parallel", fake_fm3d)
    with pytest.raises(RuntimeError, match="stop"):
        synth.forward_times(model, grid, replace(base, sources=src), tmp_path, phase=phase)
    vp = float(np.median(grid.to_vgrids(model["vp"]).get(1, 1)["velocity"].values))
    assert seen["n"] == n_types
    want = vp / 1.8 if (phase, vtype) == ("S", 1) else vp
    assert seen["v1"] == pytest.approx(want, rel=1e-3)
