"""Tests for osv.validation.bootstrap (event-resampled inversions) and its figures."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.config.schema import Model3DConfig, OSVConfig
from osv.fmtomo.invert import InversionResult
from osv.io.fmtomo import PathSpec, Receivers, Source, read_vgrids, write_vgrids
from osv.model3d.grid import define_grid
from osv.resolution.models import checkerboard
from osv.tools import ToolNotFoundError, which_tool
from osv.validation import bootstrap as bs

LAT0, LON0 = -38.5, 176.2

pytestmark = pytest.mark.filterwarnings("ignore::osv.resolution.models.UnderSamplingWarning")


def _geometry(n_src: int = 6):
    """Tiny geometry; observation times encode (source, receiver) so they can be tracked."""
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
    rr = rays.assign(ray=rays.groupby("receiver").cumcount() + 1)
    obs = pd.DataFrame(
        {
            "receiver": rr["receiver"],
            "source": rr["source"],
            "ray": rr["ray"],
            "subray": 0,
            "time": 1.0 + 0.1 * rr["source"] + 0.001 * rr["receiver"],
            "uncertainty": 0.04,
        }
    )
    obs = obs.sort_values(["source", "receiver", "ray"]).reset_index(drop=True)
    return grid, src, Receivers(pos, rays), obs


def _inputs(n_src: int = 6, mapping: bool = False):
    grid, src, rec, obs = _geometry(n_src)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    grids = grid.to_vgrids(bg["vp"], uncertainty=0.3)
    mp = None
    if mapping:  # two sources per event
        mp = pd.DataFrame(
            {
                "source_idx": range(1, n_src + 1),
                "event_id": [f"ev{i // 2}" for i in range(n_src)],
            }
        )
    inp = bs.BootstrapInputs(grids, grid.flat_interfaces(), grid.propgrid, src, rec, obs, mp, "P")
    return grid, inp


def _cfg(**bc) -> OSVConfig:
    return OSVConfig(
        fmtomo={
            "package": "base",
            "relocate": "linear",
            "damping": 10.0,
            "smoothing": 5.0,
            "iterations": 2,
            "subspace_dimension": 15,
        },
        validation={"bootstrap": {"enabled": True, **bc}},
    )


# ---------------------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------------------
def test_make_samples_bootstrap_and_jackknife() -> None:
    boot = bs.make_samples(10, "bootstrap", 5, seed=1)
    assert [s.name for s in boot] == [f"boot_{i:03d}" for i in range(1, 6)]
    assert all(len(s.events) == 10 and all(0 <= e < 10 for e in s.events) for s in boot)
    assert any(s.n_unique < 10 for s in boot)  # drawn with replacement
    assert boot == bs.make_samples(10, "bootstrap", 5, seed=1)  # deterministic
    assert boot != bs.make_samples(10, "bootstrap", 5, seed=2)

    loo = bs.make_samples(6, "jackknife", 20, seed=1, delete_fraction=0.1)  # d = 1
    assert len(loo) == 6
    assert sorted(set(range(6)) - set(s.events) for s in loo) == [{i} for i in range(6)]
    few = bs.make_samples(30, "jackknife", 4, seed=3, delete_fraction=0.1)  # d = 3
    assert all(len(s.events) == 27 and len(set(s.events)) == 27 for s in few)
    with pytest.raises(bs.BootstrapError, match="at least"):
        bs.make_samples(2, "bootstrap", 3, 1)
    with pytest.raises(ValueError, match="method"):
        bs.make_samples(5, "nope", 3, 1)


def test_build_sample_bookkeeping_with_duplicates() -> None:
    _, inp = _inputs(6)
    _, _, _, src, rec, obs, groups = inp.load()
    assert [g[0] for g in groups] == [f"src{i}" for i in range(1, 7)]
    sample = bs.Sample(1, "boot_001", (2, 2, 4, 0))  # event 3 twice, 5 and 1 once
    si = bs.build_sample(src, rec, obs, groups, sample)

    assert list(si.old_source) == [3, 3, 5, 1]
    assert list(si.event_of_source) == [2, 2, 4, 0]
    assert len(si.sources) == 4
    assert si.sources[0] == si.sources[1] == src[2]  # copies, distinct indices
    assert si.sources[0] is not si.sources[1]
    # otimes: source-major (source, receiver, ray), consecutive new source ids
    key = si.observed[["source", "receiver", "ray"]]
    assert key.equals(key.sort_values(["source", "receiver", "ray"]).reset_index(drop=True))
    assert set(si.observed["source"]) == {1, 2, 3, 4}
    # every observation is the one of its (old source, receiver)
    for _, o in si.observed.iterrows():
        old_src_id = si.old_source[int(o["source"]) - 1]
        old_rec = si.old_receiver[int(o["receiver"]) - 1]
        assert o["time"] == pytest.approx(1.0 + 0.1 * old_src_id + 0.001 * old_rec)
    n_old = obs.groupby("source").size()
    assert len(si.observed) == sum(n_old[s] for s in si.old_source)
    # rays of every receiver are numbered in file order and match the observations
    rays = si.receivers.rays
    rays = rays.assign(ray=rays.groupby("receiver").cumcount() + 1)
    assert rays["receiver"].is_monotonic_increasing
    m = si.observed.merge(rays, on=["receiver", "source", "ray"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all()
    # duplicate events give identical picks under distinct source ids
    a = si.observed[si.observed["source"] == 1].drop(columns=["source", "ray"])
    b = si.observed[si.observed["source"] == 2].drop(columns=["source", "ray"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    # receiver ids are contiguous
    assert list(si.receivers.positions.index) == list(range(1, len(si.receivers.positions) + 1))


def test_build_sample_drops_receivers_without_rays_and_groups_events() -> None:
    _, inp = _inputs(6, mapping=True)
    _, _, _, src, rec, obs, groups = inp.load()
    assert groups == [("ev0", [1, 2]), ("ev1", [3, 4]), ("ev2", [5, 6])]
    si = bs.build_sample(src, rec, obs, groups, bs.Sample(1, "j", (1,)))  # sources 3, 4
    assert list(si.old_source) == [3, 4] and list(si.event_of_source) == [1, 1]
    # a single source: receivers with (r + s) % 3 == 0 have no ray and are dropped
    _, inp = _inputs(6)
    _, _, _, src, rec, obs, groups = inp.load()
    si = bs.build_sample(src, rec, obs, groups, bs.Sample(1, "j", (0,)))
    assert list(si.old_source) == [1] and list(si.event_of_source) == [0]
    n_rec = len(si.receivers.positions)
    assert n_rec < len(rec.positions)  # receivers with (r + s) % 3 == 0 for both dropped
    assert set(si.receivers.rays["receiver"]) == set(range(1, n_rec + 1))
    assert set(si.observed["receiver"]) <= set(range(1, n_rec + 1))
    for _, o in si.observed.iterrows():
        old_rec = si.old_receiver[int(o["receiver"]) - 1]
        assert o["time"] == pytest.approx(
            1.0 + 0.1 * si.old_source[int(o["source"]) - 1] + 0.001 * old_rec
        )
    # positions follow the old receivers
    np.testing.assert_allclose(
        si.receivers.positions["lat"].to_numpy(),
        rec.positions.loc[si.old_receiver, "lat"].to_numpy(),
    )


# ---------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------
def _start(shape=(3, 4, 5)) -> xr.DataArray:
    return xr.DataArray(
        np.full(shape, 4.0),
        dims=("depth", "lat", "lon"),
        coords={
            "depth": np.arange(shape[0]) * 1.0,
            "lat": np.arange(shape[1]) * 0.01,
            "lon": np.arange(shape[2]) * 0.01,
        },
        attrs={"n_cushion": 1},
    )


def test_model_statistics_bootstrap() -> None:
    rng = np.random.default_rng(0)
    start = _start()
    m = 4.0 + rng.normal(0, 1, (400, *start.shape)) * np.linspace(0.0, 0.2, 5)  # std grows with lon
    st = bs.model_statistics(m, start, interval_percent=90.0)
    np.testing.assert_allclose(st["vp_mean"], m.mean(0), rtol=1e-5)
    np.testing.assert_allclose(st["vp_std"], m.std(0, ddof=1), rtol=1e-4)
    np.testing.assert_allclose(st["vp_lo"], np.percentile(m, 5, axis=0), rtol=1e-4)
    np.testing.assert_allclose(st["vp_hi"], np.percentile(m, 95, axis=0), rtol=1e-4)
    np.testing.assert_allclose(st["dvp_std_pct"], 100 * st["vp_std"] / 4.0, rtol=1e-4)
    np.testing.assert_allclose(st["dvp_mean_pct"], 100 * (m.mean(0) / 4 - 1), atol=1e-4)
    assert st["vp_std"].isel(lon=0).max() == pytest.approx(0.0, abs=1e-6)
    assert st["vp_std"].isel(lon=4).min() > 0.1
    assert st.attrs["n_samples"] == 400 and st.attrs["n_cushion"] == 1
    assert st["vp_std"].dims == ("depth", "lat", "lon")


def test_model_statistics_jackknife_leave_one_out_and_validation() -> None:
    # leave-one-out jackknife of the mean of n values: std must equal the standard error
    rng = np.random.default_rng(1)
    n = 8
    data = rng.normal(5.0, 0.5, n)
    start = _start((1, 1, 1))
    loo = np.array([np.delete(data, i).mean() for i in range(n)]).reshape(n, 1, 1, 1)
    st = bs.model_statistics(loo, start, method="jackknife", n_events=n, delete_d=1)
    se = data.std(ddof=1) / np.sqrt(n)
    assert st["vp_std"].values.item() == pytest.approx(se, rel=1e-5)
    assert st["vp_hi"].values.item() > st["vp_mean"].values.item() > st["vp_lo"].values.item()
    with pytest.raises(bs.BootstrapError):
        bs.model_statistics(loo[:1], start)
    with pytest.raises(ValueError, match="differ"):
        bs.model_statistics(np.ones((3, 2, 2, 2)), start)
    with pytest.raises(ValueError, match="n_events"):
        bs.model_statistics(loo, start, method="jackknife")


def test_hypocentre_variability() -> None:
    rng = np.random.default_rng(2)
    rows = []
    for s in range(6):
        rows.append(("a", f"s{s}", -38.5 + rng.normal(0, 0.001), 176.2, 3.0 + rng.normal(0, 0.5)))
        rows.append(("a", f"s{s}", -38.5, 176.2, 3.0))  # duplicate copy of a in the same sample
    rows.append(("b", "s0", -38.6, 176.3, 2.0))  # only in one sample
    h = pd.DataFrame(rows, columns=["event", "sample", "lat", "lon", "depth_km"])
    out = bs.hypocentre_variability(h)
    assert list(out["event"]) == ["a"] and int(out.loc[0, "n_samples"]) == 6
    assert out.loc[0, "std_depth_km"] > 0 and out.loc[0, "std_east_km"] == pytest.approx(
        0, abs=1e-9
    )
    assert out.loc[0, "std_3d_km"] >= out.loc[0, "std_h_km"]
    assert bs.hypocentre_variability(h.iloc[:0]).empty


# ---------------------------------------------------------------------------------------
# orchestration with a fake inversion
# ---------------------------------------------------------------------------------------
@pytest.fixture
def fake_inversion(monkeypatch):
    calls: list[dict] = []

    def fake_inv(run, controls, n_iter, **kw):
        n_src = len((run / "sources.in").read_text().split("\n"))  # any sample-dependent number
        name = run.parent.name
        calls.append({"name": name, "n_iter": n_iter, "jobs": kw["n_jobs"], "ctl": controls})
        if name == "boot_002":
            raise RuntimeError("fm3d crashed")
        vg = read_vgrids(run / "vgridsref.in")
        ds = vg.get(1, 1)
        ds["velocity"] = ds["velocity"] * (1.0 + 0.002 * (n_src % 7) + 0.001 * int(name[-3:]))
        write_vgrids(run / "vgrids.in", vg)
        (run / "frechet.dat").write_text("x" * 100)
        (run / "iterations" / "00").mkdir(parents=True, exist_ok=True)
        n = len(read_srcs(run))
        hypo = pd.DataFrame(
            {
                "iter": [2] * n,
                "source": range(1, n + 1),
                "lat": LAT0 + 0.001 * np.arange(n) + 1e-4 * int(name[-3:]),
                "lon": LON0,
                "depth_km": 3.0 + 0.1 * int(name[-3:]),
            }
        )
        return InversionResult(
            run_dir=run,
            iterations=pd.DataFrame(
                {"iter": [0, 1, 2], "rms_ms": [100.0, 60.0, 40.0], "wall_s": 0.1}
            ),
            hypocentres=hypo,
            final_model=run / "vgrids.in",
            final_sources=None,
            residuals_path=run / "residuals.dat",
        )

    def read_srcs(run):
        from osv.io.fmtomo import read_sources

        return read_sources(run / "sources.in")

    monkeypatch.setattr(bs._inv, "run_inversion", fake_inv)
    return calls


def test_run_bootstrap_orchestration_and_resume(tmp_path, fake_inversion) -> None:
    _, inp = _inputs(8)
    cfg = _cfg(n_samples=4, seed=7, method="bootstrap", max_concurrent=2)
    res = None
    with pytest.raises(bs.BootstrapError, match="succeeded"):  # force too few successes
        bs.run_bootstrap(
            inp, _cfg(n_samples=2, seed=7), (3.5, 7.0), tmp_path / "x", 2,
            samples=[bs.Sample(2, "boot_002", (0, 1, 2)), bs.Sample(3, "boot_003", (0, 1, 3))],
        )  # fmt: skip
    fake_inversion.clear()
    res = bs.run_bootstrap(inp, cfg, (3.5, 7.0), tmp_path / "w", n_jobs=4)
    assert (res.chosen_damping, res.chosen_smoothing) == (3.5, 7.0)
    assert res.max_concurrent == 2 and res.jobs_per_sample == 2
    t = res.table.set_index("name")
    assert list(t.index) == ["boot_001", "boot_002", "boot_003", "boot_004"]
    assert res.failed == ["boot_002"] and res.n_ok == 3
    assert "fm3d crashed" in t.loc["boot_002", "flags"]
    assert t.loc["boot_001", "misfit_initial_s"] == pytest.approx(0.1)
    assert t.loc["boot_001", "misfit_final_s"] == pytest.approx(0.04)
    assert t.loc["boot_001", "n_events"] == 8
    # regularisation and iterations reached the controls
    c = fake_inversion[0]["ctl"]
    assert c.run.invert3d.global_damping == 3.5 and c.run.invert3d.global_smoothing == 7.0
    assert all(x["n_iter"] == 2 for x in fake_inversion)
    # statistics from the three successful samples
    st = res.stats
    assert st.attrs["n_samples"] == 3 and st.attrs["method"] == "bootstrap"
    assert float(st["vp_std"].max()) > 0
    assert (tmp_path / "w" / "bootstrap_std.nc").exists()
    assert xr.load_dataset(res.stats_path)["dvp_std_pct"].shape == st["vp_std"].shape
    # hypocentres of events present in several samples
    assert {"event", "n_samples", "std_depth_km", "std_h_km"} <= set(res.hypocentre_stats.columns)
    assert (res.hypocentre_stats["n_samples"] >= 2).all()
    assert res.hypocentre_stats["std_depth_km"].max() > 0
    # compacted, outputs recorded
    assert not list(tmp_path.rglob("frechet.dat")) and not list(
        (tmp_path / "w").rglob("iterations")
    )
    meta = json.loads((tmp_path / "w" / "bootstrap.json").read_text())
    assert meta["n_ok"] == 3 and meta["failed"] == ["boot_002"]
    # resume: completed samples skipped, the failed one retried
    n = len(fake_inversion)
    res2 = bs.run_bootstrap(inp, cfg, (3.5, 7.0), tmp_path / "w", n_jobs=4)
    assert sorted(res2.skipped) == ["boot_001", "boot_003", "boot_004"]
    assert len(fake_inversion) == n + 1
    # different regularisation invalidates the markers
    res3 = bs.run_bootstrap(inp, cfg, (4.0, 7.0), tmp_path / "w", n_jobs=4)
    assert res3.skipped == []


def test_jackknife_and_disk_guard(tmp_path, fake_inversion) -> None:
    _, inp = _inputs(6)
    cfg = _cfg(method="jackknife", n_samples=10, delete_fraction=0.1, interval_percent=68.0)
    res = bs.run_bootstrap(inp, cfg, None, tmp_path / "j", n_jobs=1)  # chosen: config values
    assert (res.chosen_damping, res.chosen_smoothing) == (10.0, 5.0)
    assert len(res.table) == 6 and res.stats.attrs["delete_d"] == 1  # leave-one-out
    assert (res.table["n_unique_events"] == 5).all()
    assert fake_inversion[0]["ctl"].run.invert3d.global_damping == 10.0
    with pytest.raises(OSError, match="free"):
        bs.run_bootstrap(inp, _cfg(min_free_disk_gb=1e9), None, tmp_path / "j2", 1)


def test_chosen_from_tradeoff_json(tmp_path, fake_inversion) -> None:
    _, inp = _inputs(6)
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "tradeoff.json").write_text(
        json.dumps({"chosen_damping": 2.5, "chosen_smoothing": 9.0})
    )
    res = bs.run_bootstrap(inp, _cfg(n_samples=3), tmp_path / "t", tmp_path / "w", 1)
    assert (res.chosen_damping, res.chosen_smoothing) == (2.5, 9.0)


def test_nonlinear_geometry_hypocentre_mapping() -> None:
    """The tomo3di geometry keeps events consistent with the resampled sources."""
    pytest.importorskip("osv.fmtomo.stage")
    from osv.fmtomo.stage import to_nonlinear_geometry  # noqa: F401

    grid, inp = _inputs(6)
    g, itf, pg, src, rec, obs, groups = inp.load()
    si = bs.build_sample(src, rec, obs, groups, bs.Sample(1, "b", (1, 1, 3)))
    prepared = {"grids": g, "interfaces": itf, "propgrid": pg}
    (nsrc, nrec, nobs), src_dat, r2s = bs._nonlinear(prepared, si, "P")
    assert set(r2s.values()) == {1, 2, 3}  # new source ids (event copies)
    assert len(nsrc) == len(rec.positions[rec.positions.index.isin(si.old_receiver)])
    assert len(nobs) == len(si.observed)


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------
def test_figures(tmp_path, fake_inversion) -> None:
    from osv.validation import plots_bootstrap as pb

    _, inp = _inputs(8)
    cfg = _cfg(n_samples=4, seed=1)
    dws = xr.DataArray(
        np.random.default_rng(0).uniform(0, 50, inp.load()[0].get(1, 1)["velocity"].shape),
        dims=("depth", "lat", "lon"),
        coords={k: inp.load()[0].get(1, 1)["velocity"][k] for k in ("depth", "lat", "lon")},
    )
    res = bs.run_bootstrap(inp, cfg, (1.0, 1.0), tmp_path / "w", 1, dws=dws)
    assert res.n_ok == 3  # the fake inversion fails boot_002
    names = pb.make_figures(res, tmp_path / "figs")
    assert {
        "bootstrap_std_slices",
        "bootstrap_std_sections",
        "bootstrap_std_vs_dws",
        "bootstrap_sample_misfit",
    } <= set(names)
    assert (tmp_path / "figs" / "bootstrap_std_slices.png").stat().st_size > 1000


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
@pytest.mark.skipif(
    not _have("fm3d", "frechgen", "invert3d", "residuals", "scorrect"), reason="binaries missing"
)
def test_jackknife_end_to_end(tmp_path) -> None:
    from osv.resolution.synth import forward_times

    grid, src, rec, obs = _geometry(6)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    # synthetic travel times of a perturbed model; the inversion starts from the uniform one
    true = checkerboard(grid, 5.0, 2.0, 8.0)
    base = bs.BootstrapInputs(
        grid.to_vgrids(bg["vp"], uncertainty=0.3), grid.flat_interfaces(), grid.propgrid, src, rec,
        obs, None, "P",
    )  # fmt: skip
    from osv.resolution.synth import BaseInputs

    calc = forward_times(
        true,
        grid,
        BaseInputs(base.interfaces, base.propgrid, src, rec, obs),
        tmp_path / "fwd",
        n_jobs=2,
    )
    t = obs.merge(
        calc[["receiver", "source", "ray", "time"]],
        on=["receiver", "source", "ray"],
        suffixes=("_x", ""),
    )
    base.observed = t[obs.columns].sort_values(["source", "receiver", "ray"]).reset_index(drop=True)
    cfg = _cfg(method="jackknife", n_samples=3, iterations=1, seed=3)
    cfg = cfg.model_copy(
        update={
            "fmtomo": cfg.fmtomo.model_copy(
                update={"invert_velocity": True, "damping": 0.01, "smoothing": 0.1}
            )
        }
    )
    res = bs.run_bootstrap(base, cfg, None, tmp_path / "w", n_jobs=2)
    assert res.n_ok == 3, res.table["flags"].tolist()
    assert np.isfinite(res.stats["vp_std"].values).all() and res.stats["vp_std"].max() > 0
    assert not list(tmp_path.rglob("frechet.dat"))
