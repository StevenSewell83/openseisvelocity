"""Stage 5 (osv.analysis.stage) with fake upstream stage directories (no binaries)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv import stages
from osv.analysis import stage as st
from osv.config.schema import OSVConfig
from osv.stages import StageContext, stage_dirname


def _load_residual_fixture():
    spec = importlib.util.spec_from_file_location(
        "_residual_fixture", Path(__file__).with_name("test_residuals.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build


NZ, NY, NX = 7, 9, 9  # full grid including one cushion node per side
DEPTH = np.linspace(-0.5, 5.5, NZ)
LAT = -38.5 + 0.01 * np.arange(NY)
LON = 176.2 + 0.012 * np.arange(NX)
C = ("depth", "lat", "lon")


def _cfg(tmp_path: Path, **extra) -> OSVConfig:
    return OSVConfig(
        project={"workdir": str(tmp_path)},
        plotting={
            "dws_mask_threshold": 10.0,
            "format": ["png"],
            "dpi": 50,
            "depth_slices_km": [1.0, 3.0],
        },
        fmtomo={"relocate": "linear"},
        **extra,
    )


def _manifest(d: Path, name: str, run_id: str = "x") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps({"stage": name, "status": "complete", "run_id": run_id})
    )


def _model(final: bool) -> xr.Dataset:
    d, la, lo = np.meshgrid(DEPTH, LAT, LON, indexing="ij")
    bgp = 3.0 + 0.4 * d
    bgs = bgp / 1.75
    ds = xr.Dataset(
        {
            "vp": (C, bgp.copy()),
            "vs": (C, bgs.copy()),
            "vp_bg": (C, bgp),
            "vs_bg": (C, bgs),
        },
        coords={"depth": DEPTH, "lat": LAT, "lon": LON},
        attrs={"n_cushion": 1},
    )
    if final:
        core = np.zeros(d.shape, bool)
        core[2:5, 3:6, 3:6] = True
        pert = np.where(core, 0.05, 0.12)  # larger change outside the coverage
        ds["vp_start"], ds["vs_start"] = ds["vp"].copy(), ds["vs"].copy()
        ds["vp"] = ds["vp"] * (1 + pert * np.sin(d + la * 100))
        ds["vs"] = ds["vs"] * (1 + 0.5 * pert * np.sin(d + la * 100))
        dws = np.where(core, 50.0, 1.0)
        ds["dws"] = (C, dws)
        ds["dws_vs"] = (C, dws * 0.8)
        ds["mask"] = (C, (dws >= 10).astype("int8"))
        ds["mask_vs"] = (C, (dws * 0.8 >= 10).astype("int8"))
    return ds


def _mask_ds() -> xr.Dataset:
    res = np.zeros((NZ, NY, NX), bool)
    res[2:5, 3:5, 3:6] = True
    return xr.Dataset(
        {"vp": (C, res), "vs": (C, res), "resolvable": (C, res)},
        coords={"depth": DEPTH, "lat": LAT, "lon": LON},
    )


@pytest.fixture
def project(tmp_path):
    """Fake completed upstream stages; returns (cfg, run dir of the fake inversion)."""
    cfg = _cfg(tmp_path)
    for n in ("prep", "velest_analyse", "build_model", "tradeoff", "invert"):
        _manifest(tmp_path / stage_dirname(n), n)
    bld, inv = tmp_path / stage_dirname("build_model"), tmp_path / stage_dirname("invert")
    (bld / "fmtomo").mkdir()
    _model(False).to_netcdf(bld / "starting_model.nc")
    pd.DataFrame(
        {
            "name": ["S1", "S2", "S3"],
            "lat": LAT[[2, 4, 6]],
            "lon": LON[[2, 4, 6]],
            "depth_km": -0.3,
        }
    ).to_csv(bld / "fmtomo" / "stations.csv", index=False)
    pd.DataFrame({"layer": [1, 2], "depth_top_km": [0.0, 2.0], "vp": [3.0, 4.0]}).to_csv(
        tmp_path / stage_dirname("velest_analyse") / "final_model.csv", index=False
    )
    _model(True).to_netcdf(inv / "final_model.nc")
    n = 20
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "phase": "P",
            "event_id": [f"e{i}" for i in range(n)],
            "lat": LAT[4] + 0.01 * rng.normal(size=n),
            "lon": LON[4] + 0.01 * rng.normal(size=n),
            "depth_km": 2 + rng.normal(size=n),
            "dh_km": np.abs(rng.normal(0, 0.5, n)),
            "dz_km": rng.normal(0, 0.5, n),
            "d3_km": np.abs(rng.normal(0.6, 0.5, n)),
            "dt_s": rng.normal(0, 0.02, n),
        }
    ).to_parquet(inv / "relocation.parquet")
    pd.DataFrame(
        {
            "phase": "P",
            "iter": [0, 1, 2],
            "rms_ms": [200.0, 120.0, 100.0],
            "variance_s2": [0.04, 0.0144, 0.01],
            "chi2": [4.0, 2.0, 1.5],
        }
    ).to_csv(inv / "inversion_iterations.csv", index=False)
    pd.DataFrame({"phase": "P", "residual": rng.normal(0, 0.1, 50)}).to_parquet(
        inv / "residuals_final.parquet"
    )
    return cfg


def _run(cfg: OSVConfig) -> Path:
    stages.run_pipeline(cfg, only="analyse")
    return Path(cfg.project.workdir) / stage_dirname("analyse")


def _manifest_of(d: Path) -> dict:
    return json.loads((d / "manifest.json").read_text())


def _codes(d: Path) -> dict[str, str]:
    return {c["code"]: c["severity"] for c in _manifest_of(d)["checks"]}


def test_stage_registered_and_requires() -> None:
    cls = stages.get_stage("analyse")
    assert cls is st.AnalyseStage
    assert set(cls.requires) == {"invert", "build_model", "prep", "velest_analyse"}
    assert "resolution" not in cls.requires


def test_run_end_to_end(project) -> None:
    d = _run(project)
    m = _manifest_of(d)
    assert m["status"] == "complete"
    summ = json.loads((d / "summary.json").read_text())
    assert summ["variance_reduction_pct_P"] == pytest.approx(75.0, abs=0.1)
    assert summ["rms_final_ms_P"] == 100.0 and "final_rms_ms_P" in summ
    assert 0.0 < summ["covered_fraction_vp"] < 0.2
    assert summ["relocation_P"]["n_events"] == 20
    assert not summ["resolvable_mask_applied"]
    stats = pd.read_csv(d / "perturbation_stats.csv")
    assert {"covered", "uncovered", "all"} <= set(stats["region"])
    cov = stats.query("phase == 'P' and reference == 'start' and region == 'covered'").iloc[0]
    unc = stats.query("phase == 'P' and reference == 'start' and region == 'uncovered'").iloc[0]
    assert unc["max_abs_pct"] > cov["max_abs_pct"]
    codes = _codes(d)
    assert codes["change_outside_coverage"] == "warn"
    assert codes["resolution_not_applied"] == "info"
    assert "no_residual_figures" in codes  # no run directory in this fake project
    figs = {p.name for p in (d / "figures").glob("*.png")}
    assert "slices_vp.png" in figs
    assert not any("resolvable" in f for f in figs)
    for f in ("model.nc", "model.csv", "hypocentres.csv"):
        assert (d / "exports" / f).exists()
    with xr.open_dataset(d / "exports" / "model.nc") as ex:
        assert "covered_vp" in ex and "dws" in ex
    assert (d / "report.html").exists()


def test_checks_for_bad_inversion(project) -> None:
    inv = Path(project.project.workdir) / stage_dirname("invert")
    pd.DataFrame(
        {
            "phase": "P",
            "iter": [0, 1],
            "rms_ms": [100.0, 110.0],
            "variance_s2": [0.01, 0.0121],
            "chi2": [1.0, 1.1],
        }
    ).to_csv(inv / "inversion_iterations.csv", index=False)
    rel = pd.read_parquet(inv / "relocation.parquet")
    rel["d3_km"] = 3.0
    rel.to_parquet(inv / "relocation.parquet")
    codes = _codes(_run(project))
    assert codes["variance_increased"] == "fail"
    assert codes["many_events_moved"] == "warn"


def test_tolerates_missing_optional_files(project) -> None:
    inv = Path(project.project.workdir) / stage_dirname("invert")
    for f in ("relocation.parquet", "residuals_final.parquet", "inversion_iterations.csv"):
        (inv / f).unlink()
    ds = (
        xr.open_dataset(inv / "final_model.nc")
        .load()
        .drop_vars(["dws", "dws_vs", "mask", "mask_vs"])
    )
    ds.to_netcdf(inv / "final_model2.nc")
    (inv / "final_model2.nc").replace(inv / "final_model.nc")
    d = _run(project)
    codes = _codes(d)
    assert {"no_coverage", "no_relocation", "no_misfit"} <= set(codes)
    assert _manifest_of(d)["status"] == "complete"


def test_residual_figures_from_run_dir(project) -> None:
    inv = Path(project.project.workdir) / stage_dirname("invert")
    run = inv / "run_P"
    mapping = _load_residual_fixture()(run)
    mapping.to_csv(run / "mapping.csv", index=False)
    d = _run(project)
    figs = {p.name for p in (d / "figures").glob("*.png")}
    assert {"residual_histograms.png", "variance_reduction.png"} <= figs
    assert "no_residual_figures" not in _codes(d)


def test_resolvable_mask_added_by_plot(project) -> None:
    cfg = project
    d = _run(cfg)
    assert not any("resolvable" in p.name for p in (d / "figures").glob("*"))
    rd = Path(cfg.project.workdir) / stage_dirname("resolution")
    _manifest(rd, "resolution")
    _mask_ds().to_netcdf(rd / "resolvable_mask.nc")
    stages.plot_only(cfg, "analyse")
    figs = {p.name for p in (d / "figures").glob("*.png")}
    assert "slices_vp_resolvable.png" in figs
    with xr.open_dataset(d / "exports" / "model.nc") as ex:
        assert "resolvable_vp" in ex
    assert json.loads((d / "summary.json").read_text())["resolvable_mask_applied"] is True


def test_incomplete_resolution_is_ignored(project) -> None:
    rd = Path(project.project.workdir) / stage_dirname("resolution")
    rd.mkdir()
    (rd / "manifest.json").write_text(json.dumps({"status": "failed"}))
    _mask_ds().to_netcdf(rd / "resolvable_mask.nc")
    assert st.resolution_mask_path(Path(project.project.workdir)) is None


def test_user_model_comparison(tmp_path, project) -> None:
    cfg = _cfg(
        Path(project.project.workdir),
        model3d={"source": "user_grid", "user_model": {"grid_file": "dummy.nc"}},
    )
    d = _run(cfg)
    summ = json.loads((d / "summary.json").read_text())
    assert "user_comparison" in summ, _codes(d)
    assert any(p.name.startswith("user_diff_slices") for p in (d / "figures").glob("*.png"))


def test_helpers() -> None:
    m = xr.DataArray(
        np.zeros((2, 3, 2), bool),
        dims=C,
        coords={"depth": [0, 1], "lat": [0.0, 0.0, 0.0], "lon": [0, 1]},
    )
    m[:, 0] = True
    assert st.volume_fraction(m) == pytest.approx(1 / 3)
    rel = pd.DataFrame({"dh_km": [0.5, 2.0], "dz_km": [-1.0, 1.0], "d3_km": [1.1, 2.2]})
    out = st.relocation_summary(rel, 2.0)
    assert out["moved_count"] == 1 and out["median_abs_dz_km"] == 1.0
    assert st.relocation_summary(rel.iloc[0:0], 1.0) == {"n_events": 0}


def test_context_summary_keys(project) -> None:
    _run(project)
    ctx = StageContext(project, st.AnalyseStage())
    assert stages.read_manifest(ctx.stage_dir)["summary"]["variance_reduction_pct_P"] > 0
