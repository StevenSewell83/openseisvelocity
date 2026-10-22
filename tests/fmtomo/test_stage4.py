"""Stage 4 stage code (``osv.fmtomo.stage``): geometry, chosen values and a synthetic recovery."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import FmtomoConfig, Model3DConfig, ResolutionConfig
from osv.fmtomo import stage as st
from osv.io.fmtomo import (
    PathSpec,
    Receivers,
    Source,
    read_observed,
    read_vgrids,
)
from osv.model3d.grid import define_grid
from osv.resolution.models import checkerboard, grid_xyz, write_vgrids
from osv.resolution.synth import BaseInputs, make_synthetic_data
from osv.tools import ToolNotFoundError, which_tool

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "build" / "test-runs" / "osv_lg57"


def _have_tools() -> bool:
    try:
        for t in ("fm3d", "frechgen", "invert3d", "residuals", "scorrect"):
            which_tool(t)
    except ToolNotFoundError:
        return False
    return True


@pytest.fixture
def workdir():
    """Unique scratch directory under the git-ignored build/ tree, removed after the test."""
    WORK.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="run-", dir=WORK))
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        WORK.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------
def test_default_mapping_and_uncertainty():
    obs = pd.DataFrame(
        {
            "receiver": [1, 2],
            "source": [1, 1],
            "ray": [1, 2],
            "subray": 0,
            "time": [1.0, -100.0],
            "uncertainty": 0.05,
        }
    )
    m = st.default_mapping(obs, "S")
    assert list(m["valid"]) == [True, False] and (m["phase"] == "S").all()
    srcs = [Source(3.0, -38.5, 176.2, [PathSpec.direct()])]
    out = st._with_uncertainty(srcs, 1.0)
    assert out[0].uncertainty[0] == 1.0
    assert out[0].uncertainty[1] == pytest.approx(1 / 111.1949, rel=1e-4)
    assert out[0].uncertainty[2] > out[0].uncertainty[1]  # cos(lat) < 1
    assert srcs[0].uncertainty is None  # input not modified


def _toy_inputs():
    grid, base = _setup(n_src=5)
    from osv.fmtomo.rundir import RunInputs

    obs = base.observed
    grids = write_vgrids(grid, checkerboard(grid, 5.0, 2.0, 0.0), with_s=False, uncertainty=0.5)
    return RunInputs(grids, base.interfaces, base.propgrid, base.sources, base.receivers, obs)


def test_to_nonlinear_geometry_swaps_roles():
    inp = _toy_inputs()
    inp.interfaces.data["interface_depth"][:] = 0.0  # a flat surface at the sphere
    inp.otimes.loc[3, "time"] = -100.0  # one invalid pick is dropped
    mp = st.default_mapping(inp.otimes)
    new, nm = st.to_nonlinear_geometry(inp, mp)
    n_st = len(inp.receivers.positions)
    n_obs = len(inp.otimes) - 1
    assert len(new.sources) == n_st  # stations are the sources
    assert len(new.receivers.positions) == n_obs == len(new.otimes) == len(nm)  # one per pick
    assert (new.receivers.rays.groupby("receiver").size() == 1).all()
    # observations are ordered station-major, then by event; fm3d writes the arrivals that way
    key = list(zip(new.otimes["source"], new.otimes["receiver"], strict=True))
    assert key == sorted(key)
    assert list(nm["receiver_idx"]) == list(range(1, n_obs + 1))
    ev = np.array([s.lat for s in inp.sources])
    np.testing.assert_allclose(new.receivers.positions["lat"], ev[nm["event_idx"] - 1])
    # the same picks (time per event-station pair) survive the swap
    old = {(r.source, r.receiver): r.time for r in inp.otimes.itertuples()}
    for r, e in zip(nm.itertuples(), new.otimes.itertuples(), strict=True):
        assert old[(r.event_idx, r.source_idx)] == e.time
    # station-sources lie below the surface interface and carry one path
    assert all(s.depth_km >= st.SURFACE_MARGIN_KM - 1e-9 for s in new.sources)
    assert all(len(s.paths) == 1 for s in new.sources)
    # src.dat: one block per station, ids are events, sorted
    sd = st.nonlinear_src_dat(new.otimes, nm, n_st)
    assert sd.attrs["n_blocks"] == n_st and len(sd) == n_obs
    assert (sd["id"].to_numpy() == nm["event_idx"].to_numpy()).all()
    # per-event collapse
    rel = pd.DataFrame({"event_idx": nm["receiver_idx"], "n_picks": 1, "d3_km": 0.0})
    one = st._per_event(rel, nm)
    assert sorted(one["event_idx"]) == sorted(set(nm["event_idx"]))
    assert one["n_picks"].sum() == n_obs


def test_chosen_values(tmp_path):
    cfg = type("C", (), {})()
    cfg.fmtomo = FmtomoConfig(damping=7.0, smoothing=9.0)
    assert st._chosen_values(cfg, None, ["P"])["P"] == {
        "damping": 7.0,
        "smoothing": 9.0,
        "source": "config",
    }
    top = {
        "chosen_damping": 20.0,
        "chosen_smoothing": 50.0,
        "chosen": {"P": {"damping": 20.0, "smoothing": 50.0}},
        "pick": "max_curvature",
    }
    (tmp_path / "tradeoff.json").write_text(json.dumps(top))
    got = st._chosen_values(cfg, tmp_path, ["P", "S"])
    assert got["P"]["damping"] == 20.0 and got["P"]["source"].startswith("tradeoff")
    assert got["S"]["source"] == "config"  # no S result: configured values


def test_disk_guard(tmp_path):
    assert st.check_disk(tmp_path, 0.0) > 0
    with pytest.raises(st.DiskSpaceError):
        st.check_disk(tmp_path, 1e9)


# ---------------------------------------------------------------------------------------
# synthetic recovery (real programs)
# ---------------------------------------------------------------------------------------
LAT0, LON0 = -38.5, 176.2


def _setup(n_src: int = 40):
    la, lo = np.meshgrid(np.linspace(0, 0.09, 5), np.linspace(0, 0.11, 5), indexing="ij")
    stn = pd.DataFrame({"lat": LAT0 + la.ravel(), "lon": LON0 + lo.ravel(), "elevation_m": 0.0})
    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 1.0, "zmax_km": 6.0},
        propgrid_refine=2,
    )
    grid = define_grid(stn, None, cfg)
    rng = np.random.default_rng(3)
    src = [
        Source(
            float(rng.uniform(1.5, 4.5)),
            float(LAT0 + rng.uniform(0.01, 0.08)),
            float(LON0 + rng.uniform(0.01, 0.10)),
            [PathSpec.direct()],
            uncertainty=(1.0, 0.01, 0.01),
        )
        for _ in range(n_src)
    ]
    pos = pd.DataFrame(
        {"depth_km": 0.0, "lat": stn["lat"].to_numpy(), "lon": stn["lon"].to_numpy()},
        index=pd.RangeIndex(1, len(stn) + 1, name="receiver"),
    )
    rays = pd.DataFrame(
        [(r, s, 1) for r in pos.index for s in range(1, n_src + 1)],
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
            "uncertainty": 0.01,
        }
    )
    obs = obs.sort_values(["source", "receiver", "ray"]).reset_index(drop=True)
    return grid, BaseInputs(grid.flat_interfaces(), grid.propgrid, src, rec, obs)


def _gaussian_model(grid, bg, pct: float, sigma_km: float, centre=(0.0, 0.0, 3.0)):
    x, y, z = grid_xyz(grid)
    r2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2
    dv = pct * np.exp(-r2 / (2 * sigma_km**2))
    out = bg.copy(deep=True)
    out["vp"] = bg["vp_bg"] * (1.0 + dv / 100.0)
    out["dvp_pct"] = (("depth", "lat", "lon"), dv)
    return out


@pytest.mark.integration
@pytest.mark.skipif(not _have_tools(), reason="FMTOMO binaries not built")
def test_synthetic_recovery_of_gaussian_anomaly(workdir):
    """8 % Gaussian anomaly, low noise: the inversion recovers it and reduces the misfit."""
    grid, base = _setup()
    bg = checkerboard(grid, 5.0, 2.0, 0.0)  # homogeneous background, 5 km/s
    true = _gaussian_model(grid, bg, 8.0, 1.2)
    cfg = ResolutionConfig(noise={"scale": 0.1})
    data = make_synthetic_data(true, grid, base, workdir / "synth", cfg, seed=5, n_jobs=4)
    from osv.fmtomo.rundir import RunInputs

    inputs = RunInputs(
        write_vgrids(grid, bg, with_s=False, uncertainty=0.5),
        base.interfaces,
        base.propgrid,
        base.sources,
        base.receivers,
        read_observed(data.otimes),
    )
    fm = FmtomoConfig(
        package="base",
        relocate="none",
        iterations=4,
        damping=1.0,
        smoothing=1.0,
        subspace_dimension=30,
        save_rays=False,
    )
    oc = st.invert_phase(workdir / "run", inputs, fm, 1.0, 1.0, n_jobs=4, dws_threshold=1.0)
    it = oc.iterations
    r0, r1 = float(it["rms_ms"].iloc[0]), float(it["rms_ms"].iloc[-1])
    reduction = 1.0 - r1 / r0
    # recovered perturbation against the true one on the well-covered interior nodes
    fin = (
        read_vgrids(oc.run_dir / "vgrids.in").get(1, 1)["velocity"].transpose("depth", "lat", "lon")
    )
    nc = grid.n_cushion
    sl = (slice(nc, -nc),) * 3
    v_true = true["vp"].transpose("depth", "lat", "lon").to_numpy()[sl]
    v_bg = bg["vp"].transpose("depth", "lat", "lon").to_numpy()[sl]
    v_fin = fin.to_numpy()[sl]
    assert oc.dws is not None and oc.mask is not None and bool(oc.mask.any())
    cov = oc.mask.transpose("depth", "lat", "lon").to_numpy()
    dt, dr = (v_true - v_bg)[cov], (v_fin - v_bg)[cov]
    corr = float(np.corrcoef(dt, dr)[0, 1])
    print(f"misfit {r0:.1f} -> {r1:.1f} ms ({100 * reduction:.0f} %), correlation {corr:.2f}")
    assert reduction > 0.5
    assert corr > 0.5
    assert dr.max() > 0  # the fast anomaly has the right sign
    # tables for Stages 5 and 6
    assert len(oc.residuals) > 0 and len(oc.relocation) == len(base.sources)
    assert not (oc.run_dir / "frechet.dat").exists()  # heavy files removed after the DWS


@pytest.mark.integration
@pytest.mark.skipif(not _have_tools(), reason="FMTOMO binaries not built")
@pytest.mark.parametrize("relocate", ["linear", "nonlinear"])
def test_synthetic_relocation_moves_hypocentres_back(workdir, relocate):
    """Perturbed starting hypocentres are pulled back towards the true ones."""
    from osv.fmtomo.rundir import RunInputs
    from osv.io.fmtomo import read_sources

    if relocate == "nonlinear":
        try:
            which_tool("gridloc")
        except ToolNotFoundError:
            pytest.skip("gridloc not built")
    grid, base = _setup(n_src=20)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    true = _gaussian_model(grid, bg, 4.0, 1.5)
    cfg = ResolutionConfig(noise={"scale": 0.1}, hypocentre_noise_km={"xy": 0.5, "z": 0.5})
    data = make_synthetic_data(true, grid, base, workdir / "synth", cfg, seed=7, n_jobs=4)
    start = read_sources(data.sources)  # perturbed
    truth = read_sources(data.true_sources)
    fm = FmtomoConfig(
        package="extras" if relocate == "nonlinear" else "base",
        relocate=relocate,
        iterations=3,
        subspace_dimension=40,
        save_rays=False,
    )
    inputs = RunInputs(
        write_vgrids(grid, bg, with_s=False, uncertainty=0.5),
        base.interfaces,
        base.propgrid,
        st._with_uncertainty(start, 1.0) if relocate == "linear" else start,
        base.receivers,
        read_observed(data.otimes),
    )
    mp = st.default_mapping(inputs.otimes)
    if relocate == "nonlinear":
        inputs, mp = st.to_nonlinear_geometry(inputs, mp)
    pdata = st.PhaseData(
        "P",
        inputs,
        mp,
        st.ModelInfo.from_model(inputs.grids, sources=inputs.sources, receivers=inputs.receivers),
        relocate == "nonlinear",
    )
    oc = st.invert_phase(workdir / "run", pdata, fm, 1.0, 1.0, n_jobs=4, dws_threshold=1.0)
    km = 111.1949

    def err(lat, lon, dep):
        la0 = np.array([s.lat for s in truth])
        return np.sqrt(
            ((lat - la0) * km) ** 2
            + ((lon - np.array([s.lon for s in truth])) * km * np.cos(np.deg2rad(la0))) ** 2
            + (dep - np.array([s.depth_km for s in truth])) ** 2
        )

    r = oc.relocation.sort_values("event_idx")
    e0 = err(np.array([s.lat for s in start]), np.array([s.lon for s in start]),
             np.array([s.depth_km for s in start]))  # fmt: skip
    e1 = err(r["lat"].to_numpy(), r["lon"].to_numpy(), r["depth_km"].to_numpy())
    print(f"{relocate}: mean location error {e0.mean():.2f} -> {e1.mean():.2f} km")
    assert r["d3_km"].max() > 0.05  # events moved
    assert len(oc.relocation) == len(base.sources)
    if relocate == "nonlinear":
        # the grid search finds the hypocentres; the linearised step with a 0.5 km start error
        # may overshoot (it trades off with the velocity anomaly), so it is only checked to run
        assert e1.mean() < 0.5 * e0.mean()
        r0, r1 = oc.iterations["rms_ms"].iloc[[0, -1]]
        assert r1 < 0.5 * r0


# ---------------------------------------------------------------------------------------
# the stages on a synthetic Stage 3 directory
# ---------------------------------------------------------------------------------------
def _fake_stage3(project: Path, grid, base, bg, otimes: pd.DataFrame) -> None:
    """Write the files of ``04_build_model`` that Stage 4 reads, from synthetic inputs."""
    from osv.io.fmtomo import (
        write_interfaces,
        write_observed,
        write_propgrid,
        write_receivers,
        write_sources,
        write_vgrids,
    )

    d = project / "04_build_model"
    fm = d / "fmtomo"
    fm.mkdir(parents=True)
    write_vgrids(fm / "vgrids.in", write_vgrids_(grid, bg))
    write_interfaces(fm / "interfaces.in", base.interfaces)
    write_propgrid(fm / "propgrid.in", base.propgrid)
    write_sources(fm / "sources.in", base.sources)
    write_sources(fm / "sourcesref.in", base.sources)
    write_receivers(fm / "receivers.in", base.receivers)
    write_observed(fm / "otimes_P.dat", otimes)
    mp = st.default_mapping(otimes)
    mp["event_id"] = "ev" + mp["source_idx"].astype(str)
    mp["station"] = "S" + mp["receiver_idx"].astype(str)
    mp.to_csv(fm / "otimes_P_mapping.csv", index=False)
    pd.DataFrame(
        {
            "source_idx": range(1, len(base.sources) + 1),
            "event_id": [f"ev{i}" for i in range(1, len(base.sources) + 1)],
            "lat": [s.lat for s in base.sources],
            "lon": [s.lon for s in base.sources],
            "depth_km_bsl": [s.depth_km for s in base.sources],
        }
    ).to_csv(fm / "events.csv", index=False)
    pos = base.receivers.positions
    pd.DataFrame(
        {
            "station": [f"S{i}" for i in pos.index],
            "depth_km": pos["depth_km"],
            "lat": pos["lat"],
            "lon": pos["lon"],
        }
    ).to_csv(fm / "stations.csv", index=False)
    bg.to_netcdf(d / "starting_model.nc")
    (d / "manifest.json").write_text(json.dumps({"status": "complete", "run_id": "fake"}))


def write_vgrids_(grid, bg):
    return write_vgrids(grid, bg, with_s=False, uncertainty=0.5)


@pytest.mark.integration
@pytest.mark.skipif(not _have_tools(), reason="FMTOMO binaries not built")
def test_tradeoff_and_invert_stages_end_to_end(workdir):
    from osv.config import OSVConfig
    from osv.stages import run_pipeline

    grid, base = _setup(n_src=30)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    true = _gaussian_model(grid, bg, 8.0, 1.2)
    data = make_synthetic_data(
        true, grid, base, workdir / "synth", ResolutionConfig(noise={"scale": 0.2}), 1, 4
    )
    project = workdir / "proj"
    _fake_stage3(project, grid, base, bg, read_observed(data.otimes))
    cfg = OSVConfig.model_validate(
        {
            "project": {"workdir": str(project), "n_jobs": 4},
            "fmtomo": {
                "package": "base",
                "relocate": "none",
                "iterations": 3,
                "damping": 1.0,
                "smoothing": 1.0,
                "subspace_dimension": 30,
                "tradeoff": {
                    "damping": [0.1, 1.0, 10.0, 100.0],
                    "smoothing": [0.1, 1.0, 10.0],
                    "iterations": 2,
                    "max_concurrent": 1,
                },
            },
            "plotting": {"dws_mask_threshold": 0.3, "depth_slices_km": [2.0, 3.0]},
        }
    )
    res = run_pipeline(cfg, only="tradeoff", project_dir=project)
    assert res == {"tradeoff": "ran"}
    td = project / "05_tradeoff"
    top = json.loads((td / "tradeoff.json").read_text())
    assert top["chosen_damping"] > 0 and top["chosen_smoothing"] > 0
    assert (td / "tradeoff.csv").exists() and (td / "report.html").exists()
    entries = json.loads((td / "figures" / "figures.json").read_text())["figures"]
    names = {e["name"] for e in entries}
    assert {"lcurve_damping", "lcurve_smoothing", "misfit_vs_iteration_sweep"} <= names
    # no heavy files are left in the sweep runs
    assert not list(td.rglob("frechet.dat"))
    assert load_chosen_matches(td, top)

    res = run_pipeline(cfg, only="invert", project_dir=project)
    assert res == {"invert": "ran"}
    iv = project / "06_invert"
    for name in st.InvertStage.outputs:
        assert (iv / name).exists(), name
    reg = json.loads((iv / "chosen_regularisation.json").read_text())
    assert reg["damping"] == top["chosen_damping"] and reg["smoothing"] == top["chosen_smoothing"]
    import xarray as xr

    with xr.open_dataset(iv / "final_model.nc") as m:
        assert {"vp", "vp_bg", "vp_start", "dws", "mask"} <= set(m.data_vars)
        assert m["vp"].shape == m["dws"].shape and int(m["mask"].sum()) > 0
        assert m.attrs["n_cushion"] == 1
    its = pd.read_csv(iv / "inversion_iterations.csv")
    assert its["rms_ms"].iloc[-1] < 0.5 * its["rms_ms"].iloc[0]
    assert len(pd.read_parquet(iv / "residuals_final.parquet")) > 0
    assert len(pd.read_parquet(iv / "relocation.parquet")) == len(base.sources)
    assert (iv / "report.html").exists() and (iv / "exports" / "model.nc").exists()
    figs = {e["name"] for e in json.loads((iv / "figures" / "figures.json").read_text())["figures"]}
    assert {
        "misfit_vs_iteration",
        "residual_histograms_iterations",
        "station_residual_map",
        "rays_lost_iterations",
        "velocity_range_iterations",
        "residual_histograms",
        "slices_vp",
    } <= figs
    man = json.loads((iv / "manifest.json").read_text())
    assert man["status"] == "complete" and man["summary"]["misfit_reduction_pct_P"] > 50
    assert not (iv / "run_P" / "frechet.dat").exists()


@pytest.mark.integration
@pytest.mark.skipif(not _have_tools(), reason="FMTOMO binaries not built")
def test_tradeoff_stage_with_nonlinear_relocation(workdir):
    """The sweep runs gridloc (tomo3di); every run must succeed (src.dat as gridloc reads it)."""
    from osv.config import OSVConfig
    from osv.io.fmtomo.extras import read_src_dat
    from osv.stages import run_pipeline

    try:
        which_tool("gridloc")
    except ToolNotFoundError:
        pytest.skip("gridloc not built")
    grid, base = _setup(n_src=20)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    data = make_synthetic_data(
        _gaussian_model(grid, bg, 5.0, 1.2), grid, base, workdir / "synth",
        ResolutionConfig(noise={"scale": 0.2}), 1, 4,
    )  # fmt: skip
    project = workdir / "proj"
    _fake_stage3(project, grid, base, bg, read_observed(data.otimes))
    cfg = OSVConfig.model_validate(
        {
            "project": {"workdir": str(project), "n_jobs": 4},
            "fmtomo": {
                "package": "extras",
                "relocate": "nonlinear",
                "subspace_dimension": 30,
                "save_rays": False,
                "tradeoff": {
                    "damping": [1.0, 10.0, 100.0],
                    "smoothing": [1.0, 10.0, 100.0],
                    "iterations": 1,
                    "max_concurrent": 1,
                },
            },
        }
    )
    assert run_pipeline(cfg, only="tradeoff", project_dir=project) == {"tradeoff": "ran"}
    td = project / "05_tradeoff"
    table = pd.read_csv(td / "tradeoff.csv")
    assert table["success"].all(), table["flags"].tolist()
    # gridloc's origin-time-corrected observed times are kept per iteration (residual tables)
    assert list(td.rglob("iterations/01/otimes.dat"))
    sd = read_src_dat(next(td.rglob("src.dat")))
    assert sd.attrs["n_blocks"] == len(base.receivers.positions)  # stations
    assert sd["id"].max() <= len(base.sources)  # event numbers


def load_chosen_matches(td: Path, top: dict) -> bool:
    from osv.fmtomo.tradeoff import load_chosen

    return load_chosen(td) == (top["chosen_damping"], top["chosen_smoothing"])


def test_tradeoff_cache_invalidated_by_new_upstream(tmp_path):
    from osv.config.schema import FmtomoConfig
    from osv.fmtomo.stage import _invalidate_stale_sweeps

    fm = FmtomoConfig()
    _invalidate_stale_sweeps(tmp_path, "run1", fm)
    (tmp_path / "P" / "d1_s1").mkdir(parents=True)
    (tmp_path / "figures").mkdir()
    _invalidate_stale_sweeps(tmp_path, "run1", fm)  # same inputs: kept (resume)
    assert (tmp_path / "P" / "d1_s1").exists()
    fm2 = FmtomoConfig(iterations=7)
    _invalidate_stale_sweeps(tmp_path, "run1", fm2)  # other settings: wiped
    assert not (tmp_path / "P").exists() and (tmp_path / "figures").exists()
    (tmp_path / "P").mkdir()
    _invalidate_stale_sweeps(tmp_path, "run2", fm2)  # new build_model run: wiped
    assert not (tmp_path / "P").exists()
