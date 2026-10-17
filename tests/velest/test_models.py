from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import VelestConfig
from osv.io.velest import read_model
from osv.velest.models import (
    InfeasibleBoundsError,
    generate_trial_models,
    models_to_frame,
    write_trial_models,
)

_OLD_LAYERS = [-0.6, 0.2, 0.7, 1.2, 1.7, 2.2, 3.0, 4.0]
_OLD_BOUNDS = [
    (1.8, 3.5), (2.5, 4.0), (3.0, 4.5), (3.5, 5.0), (4.0, 5.5), (4.5, 6.0), (5.0, 6.2), (5.5, 6.5),
]  # fmt: skip


def _cfg(**kw) -> VelestConfig:
    kw.setdefault("n_models", 64)
    kw.setdefault("layers_top_km", _OLD_LAYERS)  # the tests below assume 8 layers
    kw.setdefault("vp_bounds", _OLD_BOUNDS)
    return VelestConfig(**kw)


def _arr(models, attr):
    return np.array([getattr(m, attr) for m in models if not m.is_reference])


@pytest.mark.parametrize("sampler", ["latin_hypercube", "sobol", "uniform"])
@pytest.mark.parametrize("mode", ["fixed", "sampled", "independent_s_model"])
def test_reproducible_and_seed_dependent(sampler, mode):
    cfg = _cfg(sampler=sampler, vpvs={"mode": mode})
    a, b, c = (generate_trial_models(cfg, s) for s in (1, 1, 2))
    assert np.array_equal(_arr(a, "vp"), _arr(b, "vp"))
    assert np.array_equal(_arr(a, "vs"), _arr(b, "vs"))
    assert not np.array_equal(_arr(a, "vp"), _arr(c, "vp"))


@pytest.mark.parametrize("monotonic", [True, False])
@pytest.mark.parametrize("sampler", ["latin_hypercube", "sobol", "uniform"])
def test_bounds_and_ids(monotonic, sampler):
    cfg = _cfg(monotonic=monotonic, sampler=sampler, vpvs={"mode": "independent_s_model"})
    models = generate_trial_models(cfg, 3)
    assert [m.id for m in models][:2] == ["MC_0000", "MC_0001"]
    assert len(models) == 65 and models[0].is_reference
    b = np.array(cfg.vp_bounds)
    vp = np.array([m.vp for m in models])
    assert (vp >= b[:, 0] - 1e-12).all() and (vp <= b[:, 1] + 1e-12).all()
    vs = np.array([m.vs for m in models])
    assert (vs > 0).all()
    np.testing.assert_allclose(models[1].depths, cfg.layers_top_km)


def test_no_reference():
    models = generate_trial_models(_cfg(include_reference_model=False), 0)
    assert len(models) == 64 and models[0].id == "MC_0001"


def test_monotonic_and_marginals_cover_bounds():
    cfg = _cfg(monotonic=True, n_models=500)
    models = generate_trial_models(cfg, 5)
    vp = np.array([m.vp for m in models])
    assert (np.diff(vp, axis=1) >= -1e-12).all()
    b = np.array(cfg.vp_bounds)
    span = b[:, 1] - b[:, 0]
    # every layer's marginal reaches into both outer 15% of its (tightened) range
    for i in range(vp.shape[1]):
        lo_eff = b[: i + 1, 0].max()
        hi_eff = b[i:, 1].min()
        w = hi_eff - lo_eff
        assert vp[:, i].min() < lo_eff + 0.15 * w, i
        assert vp[:, i].max() > hi_eff - 0.15 * w, i
        assert span[i] > 0


def test_non_monotonic_allows_lvz_and_is_uniform_per_layer():
    cfg = _cfg(monotonic=False, n_models=400)
    vp = np.array([m.vp for m in generate_trial_models(cfg, 5) if not m.is_reference])
    assert (np.diff(vp, axis=1) < 0).any()
    b = np.array(cfg.vp_bounds)
    u = (vp - b[:, 0]) / (b[:, 1] - b[:, 0])
    np.testing.assert_allclose(u.mean(axis=0), 0.5, atol=0.05)


def test_lhs_stratification():
    cfg = _cfg(monotonic=False, n_models=50, sampler="latin_hypercube")
    vp = np.array([m.vp for m in generate_trial_models(cfg, 9) if not m.is_reference])
    b = np.array(cfg.vp_bounds)
    u = (vp - b[:, 0]) / (b[:, 1] - b[:, 0])
    strata = np.floor(u * 50).astype(int).clip(0, 49)
    for i in range(u.shape[1]):
        assert sorted(strata[:, i]) == list(range(50))


def test_vpvs_modes():
    fixed = generate_trial_models(_cfg(vpvs={"mode": "fixed", "value": 1.8}), 1)
    for m in fixed:
        np.testing.assert_allclose(m.vpvs, 1.8)

    samp = generate_trial_models(_cfg(vpvs={"mode": "sampled", "range": (1.6, 1.9)}), 1)
    ratios = []
    for m in samp:
        assert np.allclose(m.vpvs, m.vpvs[0])  # one ratio per model
        ratios.append(m.vpvs[0])
    r = np.array(ratios[1:])
    assert r.min() >= 1.6 - 1e-9 and r.max() <= 1.9 + 1e-9 and r.std() > 0.05
    assert ratios[0] == pytest.approx(1.75)

    sb = [(1.0, 2.0)] * 8
    ind = _cfg(vpvs={"mode": "independent_s_model", "s_bounds": sb}, monotonic=False)
    vs = np.array([m.vs for m in generate_trial_models(ind, 1)])
    assert vs.min() >= 1.0 and vs.max() <= 2.0
    assert vs.std() > 0.2


def test_independent_s_default_bounds_from_range():
    cfg = _cfg(vpvs={"mode": "independent_s_model", "range": (1.6, 1.9)})
    vs = np.array([m.vs for m in generate_trial_models(cfg, 1)])
    b = np.array(cfg.vp_bounds)
    assert (vs >= b[:, 0] / 1.9 - 1e-9).all() and (vs <= b[:, 1] / 1.6 + 1e-9).all()
    assert (np.diff(vs, axis=1) >= -1e-12).all()  # monotonic applies to Vs too


def test_infeasible_monotonic_bounds():
    # layer 2 upper bound is below layer 0 lower bound; schema only checks neighbours
    cfg = _cfg(
        layers_top_km=[0, 1, 2],
        vp_bounds=[(4.0, 5.0), (3.0, 6.0), (3.0, 3.5)],
        monotonic=True,
    )
    with pytest.raises(InfeasibleBoundsError, match="monotonic"):
        generate_trial_models(cfg, 0)
    cfg2 = cfg.model_copy(update={"monotonic": False})
    assert len(generate_trial_models(cfg2, 0)) == 65


def test_write_and_read_back(tmp_path):
    cfg = _cfg(n_models=5, vpvs={"mode": "sampled"})
    models = generate_trial_models(cfg, 4)
    paths = write_trial_models(models, tmp_path, vdamp=2.5)
    assert [p.name for p in paths] == [f"{m.id}.mod" for m in models]
    for m, p in zip(models, paths, strict=True):
        vm = read_model(p)
        assert vm.s is not None
        np.testing.assert_allclose(vm.p.layers["vel"], m.vp, atol=0.005 + 1e-9)
        np.testing.assert_allclose(vm.s.layers["vel"], m.vs, atol=0.005 + 1e-9)
        np.testing.assert_allclose(vm.p.layers["depth"], m.depths)
        assert (vm.p.layers["vdamp"] == 2.5).all()
        assert vm.p.layers["depth"].iloc[0] == -0.6
    df = pd.read_parquet(tmp_path / "trial_models.parquet")
    assert len(df) == len(models) * 8
    assert df.equals(models_to_frame(models))


def test_per_layer_vdamp(tmp_path):
    models = generate_trial_models(_cfg(n_models=1), 0)
    vd = np.arange(1, 9, dtype=float)
    p = write_trial_models(models[:1], tmp_path, vdamp=vd, parquet=False)[0]
    np.testing.assert_allclose(read_model(p).p.layers["vdamp"], vd)
    assert not (tmp_path / "trial_models.parquet").exists()
