"""Stage 6 (``resolution``): checkerboard, spike and input-model recovery tests.

``resolution`` (``<workdir>/08_resolution/``) needs ``build_model`` and ``tradeoff`` (the damping
and smoothing of the real inversion).  If ``invert`` has completed, its ``final_model.nc`` is the
``final_3d`` input model and its DWS mask the coverage; otherwise ``final_3d`` is skipped and the
coverage is a straight-ray proxy of the Stage 3 geometry (nodes crossed by at least
:data:`MIN_PROXY_RAYS` source-receiver segments).

Steps: build the tests from ``resolution.*``, run them per phase with the chosen regularisation,
the iterations and the relocation mode of the real inversion
(:func:`osv.resolution.runner.run_resolution_tests`), compute the recovery metrics and the
resolvable mask (:func:`osv.resolution.metrics.run_metrics`), then figures, checks and summary.

Outputs (stable names)
    ``resolvable_mask.nc``
        On the full grid: bool ``vp`` (``vs``), ``resolvable``, float ``resolved_scale_km_vp``
        (``_vs``) and ``resolved_scale_km``.  Stage 5 uses it to mask its figures.
    ``resolution_runs.json``
        Settings and one row per test (success, misfits, directories).
    ``metrics/``
        ``metrics.nc`` (local recovery fields), ``summary.csv`` / ``summary.json`` (global
        metrics, spike smearing, checkerboard anisotropy), ``resolvable_mask.nc``.
    ``resolved_fractions.csv``
        Fraction of the covered volume resolved at each cell size per depth range.
    ``runs/<test>/``
        ``true_model.nc``, ``recovered_model.nc``, ``iterations.csv``, ``summary.json``.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.fmtomo.stage import FMTOMO_DIR, load_phase_data
from osv.io.fmtomo import read_observed
from osv.model3d.stage import MODEL_FILE, grid_from_outputs
from osv.report.checks import QCFinding
from osv.resolution import plots as _plots
from osv.resolution.metrics import interior_mask, run_metrics
from osv.resolution.runner import ResolutionRunResult, build_tests, run_resolution_tests
from osv.resolution.synth import NETCDF_LOCK, BaseInputs
from osv.stages import (
    COMPLETE,
    Stage,
    StageContext,
    StageError,
    read_manifest,
    register,
    stage_dirname,
)

log = logging.getLogger(__name__)

__all__ = ["MIN_FREE_GB", "ResolutionStage", "resolved_fractions", "straight_ray_coverage"]

#: refuse to start with less free disk than this (GB)
MIN_FREE_GB = 3.0
#: a node is covered in the straight-ray proxy if this many segments cross its cell
MIN_PROXY_RAYS = 3
#: number of equal-thickness depth ranges of the resolved-fraction table
N_DEPTH_RANGES = 3
#: more than this fraction of the covered volume unresolved raises a warning
MAX_UNRESOLVED_FRACTION = 0.5
RUNS = "runs"
METRICS = "metrics"
MASK_FILE = "resolvable_mask.nc"
#: files of the Stage 3 geometry per phase
_OTIMES = {"P": "otimes_P.dat", "S": "otimes_S.dat"}


def _finding(code: str, severity: str, message: str, count: int = 0, **details: Any) -> QCFinding:
    return QCFinding(code, severity, message, count, details)  # type: ignore[arg-type]


def _phases(cfg: Any, s3_dir: Path) -> list[str]:
    ph = set(cfg.fmtomo.phases)
    want = ["P"] + (["S"] if {"S", "vpvs"} & ph else [])
    if "P" not in ph and "vpvs" not in ph:
        want = [p for p in want if p in ph]
    have = [p for p in want if (s3_dir / FMTOMO_DIR / _OTIMES[p]).exists()]
    if not have:
        raise StageError("fmtomo.phases selects no phase with data in Stage 3 (run build-model)")
    return have


def _invert_dir(ctx: StageContext) -> Path | None:
    d = ctx.project_dir / stage_dirname("invert")
    m = read_manifest(d)
    if m is not None and m.get("status") == COMPLETE and (d / "final_model.nc").exists():
        return d
    return None


def _chosen(td: Path, phase: str) -> tuple[float, float]:
    top = json.loads((td / "tradeoff.json").read_text())
    c = top.get("chosen", {}).get(phase)
    if c is None:
        c = {"damping": top["chosen_damping"], "smoothing": top["chosen_smoothing"]}
    return float(c["damping"]), float(c["smoothing"])


# ---------------------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------------------
def straight_ray_coverage(
    grid: Any, sources: Any, receivers: Any, min_rays: int = MIN_PROXY_RAYS
) -> np.ndarray:
    """Boolean full-grid mask of nodes crossed by at least ``min_rays`` straight segments.

    A cheap stand-in for the DWS coverage when no inversion has been run: every ray of
    ``receivers.rays`` is a straight line between its source (``Source`` list) and receiver;
    a node counts a ray once if the line passes through its (nearest-node) cell.
    """
    shape = grid.shape_full
    axes = (grid.depth_full, grid.lat_full, grid.lon_full)
    step = np.array([float(np.median(np.diff(a))) for a in axes])
    org = np.array([float(a[0]) for a in axes])
    pos = receivers.positions
    rays = receivers.rays
    counts = np.zeros(int(np.prod(shape)), dtype=np.int64)
    src = np.array([[s.depth_km, s.lat, s.lon] for s in sources], float)
    rec = pos[["depth_km", "lat", "lon"]].reindex(rays["receiver"].to_numpy()).to_numpy(float)
    sc = src[rays["source"].to_numpy(int) - 1]
    # about two samples per cell along the longest axis
    nstep = np.maximum(2, np.ceil(2.0 * np.abs(rec - sc).max(axis=1) / np.abs(step).min())).astype(
        int
    )
    for a, b, n in zip(sc, rec, nstep, strict=True):
        t = np.linspace(0.0, 1.0, int(n))[:, None]
        p = a + t * (b - a)
        idx = np.rint((p - org) / step).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        if not ok.any():
            continue
        flat = np.unique(np.ravel_multi_index(tuple(idx[ok].T), shape))
        counts[flat] += 1
    return (counts >= int(min_rays)).reshape(shape)


def _coverage(
    invert_dir: Path | None, grid: Any, phases: list[str], data: dict[str, Any]
) -> tuple[dict[str, xr.DataArray], str]:
    cov: dict[str, xr.DataArray] = {}
    da_tpl = xr.DataArray(
        np.zeros(grid.shape_full, bool),
        dims=("depth", "lat", "lon"),
        coords={"depth": grid.depth_full, "lat": grid.lat_full, "lon": grid.lon_full},
    )
    if invert_dir is not None:
        with NETCDF_LOCK, xr.open_dataset(invert_dir / "final_model.nc") as ds:
            ds = ds.load()
        for ph in phases:
            name = "mask" if ph == "P" else "mask_vs"
            if name in ds and ds[name].shape == grid.shape_full:
                cov[f"v{ph.lower()}"] = da_tpl.copy(data=ds[name].values.astype(bool))
        if len(cov) == len(phases):
            return cov, "invert DWS mask"
    for ph in phases:
        if f"v{ph.lower()}" in cov:
            continue
        pd_ = data[ph]
        cov[f"v{ph.lower()}"] = da_tpl.copy(
            data=straight_ray_coverage(grid, pd_.inputs.sources, pd_.inputs.receivers)
        )
    return (
        cov,
        f"straight-ray proxy (>= {MIN_PROXY_RAYS} rays per node)"
        if invert_dir is None
        else "mixed",
    )


def resolved_fractions(
    scale: np.ndarray,
    coverage: np.ndarray | None,
    grid: Any,
    sizes: list[float],
    n_ranges: int = N_DEPTH_RANGES,
) -> pd.DataFrame:
    """Fraction of the covered interior nodes resolved at each cell size, per depth range.

    ``scale`` is ``resolved_scale_km`` (NaN = unresolved); a node is resolved at size ``c`` when
    its scale is ``<= c``.  Depth ranges are ``n_ranges`` equal-thickness parts of the interior
    depth axis; the last row of each size (``range = "all"``) covers the whole volume.
    """
    inter = interior_mask(grid)
    cov = inter if coverage is None else inter & np.asarray(coverage, bool)
    z = np.broadcast_to(np.asarray(grid.depth_full)[:, None, None], scale.shape)
    zi = np.asarray(grid.depth, float)
    edges = np.linspace(zi[0] - 0.5 * grid.dz_km, zi[-1] + 0.5 * grid.dz_km, n_ranges + 1)
    rows = []
    for c in sorted(sizes):
        ok = np.isfinite(scale) & (scale <= c + 1e-9)
        for lo, hi, lab in [
            *(
                (edges[i], edges[i + 1], f"{edges[i]:.1f}-{edges[i + 1]:.1f}")
                for i in range(n_ranges)
            ),
            (-np.inf, np.inf, "all"),
        ]:
            sel = cov & (z >= lo) & (z < hi if np.isfinite(hi) else True)
            n = int(sel.sum())
            rows.append(
                {
                    "cell_km": float(c),
                    "depth_range_km": lab,
                    "n_covered": n,
                    "fraction_resolved": float((ok & sel).sum() / n) if n else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _smallest_anomaly_km(rc: Any) -> float | None:
    """Smallest dimension (km) of the configured spikes and anomaly bodies, or None."""
    sizes: list[float] = []
    for s in rc.spikes:
        sizes.append(2.0 * float(s.radius_km))
    for b in rc.anomalies:
        dims = [getattr(b, k, None) for k in ("length_km", "width_km", "height_km", "thickness_km")]
        dims = [float(d) for d in dims if d]
        if getattr(b, "radius_km", None) and getattr(b, "kind", "") == "cylinder":
            dims.append(2.0 * float(b.radius_km))
        sizes += dims
    return min(sizes) if sizes else None


# ---------------------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------------------
@register
class ResolutionStage(Stage):
    """Stage 6: resolution tests, recovery metrics, resolvable mask."""

    name = "resolution"
    requires = ("build_model", "tradeoff")
    config_sections = ("project.random_seed", "resolution", "fmtomo")
    tools = ("fm3d", "frechgen", "invert3d", "residuals")
    outputs = ("resolution_runs.json", MASK_FILE, f"{METRICS}/summary.csv")

    def inputs(self, ctx: StageContext) -> list[Path]:
        d = _invert_dir(ctx)
        return [d / "final_model.nc"] if d is not None else []

    # ------------------------------------------------------------------ run

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        rc = cfg.resolution
        s3, td = ctx.upstream("build_model"), ctx.upstream("tradeoff")
        free = shutil.disk_usage(ctx.stage_dir).free / 1e9
        if free < MIN_FREE_GB:
            raise StageError(
                f"only {free:.1f} GB free on the disk of {ctx.stage_dir}; resolution tests need "
                f"{MIN_FREE_GB:g} GB. Free space (delete old run directories) and re-run."
            )
        grid = grid_from_outputs(s3.dir)
        with NETCDF_LOCK, xr.open_dataset(s3.dir / MODEL_FILE) as raw:
            background = raw.load()
        phases = _phases(cfg, s3.dir)
        data = {ph: load_phase_data(s3.dir, ph, cfg.fmtomo) for ph in phases}
        bases = {
            ph: BaseInputs(
                d.inputs.interfaces,
                d.inputs.propgrid,
                d.inputs.sources,
                d.inputs.receivers,
                d.inputs.otimes,
            )
            for ph, d in data.items()
        }
        findings: list[QCFinding] = []
        inv = _invert_dir(ctx)
        kw: dict[str, dict[str, Any]] = {}
        if inv is not None:
            with NETCDF_LOCK, xr.open_dataset(inv / "final_model.nc") as raw:
                kw["final_3d"] = {"dataset": raw.load()}
        elif "final_3d" in rc.input_models:
            findings.append(
                _finding(
                    "final_3d_skipped",
                    "warn",
                    "input model 'final_3d' skipped: the invert stage has not completed "
                    "(run `osv invert`, then `osv resolution --force`)",
                )
            )
        for k in ("user_model", "starting"):
            if k in rc.input_models:
                findings.append(
                    _finding(
                        f"{k}_skipped",
                        "info",
                        f"input model {k!r} is not run by the stage (its perturbation against "
                        "the starting model would be zero): use 'file' or 'anomaly' instead",
                    )
                )
        cov, cov_src = _coverage(inv, grid, phases, data)
        ctx.logger.info("coverage source: %s", cov_src)
        free_nodes = {k: int(v.values.sum()) for k, v in cov.items()}

        # ---- the tests, per phase (each phase has its own chosen regularisation)
        runs_dir = ctx.stage_dir / RUNS
        tables = []
        picked: dict[str, tuple[float, float]] = {}
        n_iter = None
        for ph in phases:
            tests = build_tests(cfg, phases=[ph], input_model_kwargs=kw)
            if not tests:
                continue
            d, s = _chosen(td.dir, ph)
            picked[ph] = (d, s)
            ctx.logger.info(
                "resolution %s: %d tests, damping %g, smoothing %g", ph, len(tests), d, s
            )
            res = run_resolution_tests(
                tests,
                bases[ph],
                grid,
                background,
                cfg,
                runs_dir,
                n_jobs=ctx.n_jobs,
                chosen=(d, s),
                min_free_disk_gb=MIN_FREE_GB,
                on_test=lambda row: ctx.logger.info(
                    "test %s: success %s, misfit %.3g -> %.3g s",
                    row["name"],
                    row["success"],
                    row.get("misfit_initial_s", math.nan),
                    row.get("misfit_final_s", math.nan),
                ),
            )
            n_iter = res.n_iter
            tables.append(res.table)
        if not tables:
            raise StageError("no resolution tests to run: check resolution.* in the config")
        table = pd.concat(tables, ignore_index=True)
        result = ResolutionRunResult(
            runs_dir, table, picked[phases[0]][0], picked[phases[0]][1], int(n_iter or 0), 1, 1
        )
        (ctx.stage_dir / "resolution_runs.json").write_text(
            json.dumps(
                {
                    "chosen": {p: {"damping": d, "smoothing": s} for p, (d, s) in picked.items()},
                    "n_iter": n_iter,
                    "coverage_source": cov_src,
                    "table": json.loads(table.to_json(orient="records")),
                },
                indent=1,
            )
        )
        failed = result.failed
        if failed:
            findings.append(
                _finding(
                    "tests_failed",
                    "fail" if len(failed) == len(table) else "warn",
                    f"{len(failed)} of {len(table)} resolution tests failed: {', '.join(failed)}",
                    len(failed),
                )
            )
        if len(failed) == len(table):
            for f in findings:
                ctx.add_check(f)
            raise StageError("all resolution tests failed; see runs/<test>/summary.json")

        # ---- metrics and mask
        out = run_metrics(result, grid, cfg, ctx.stage_dir / METRICS, coverage=cov)
        res_ds: xr.Dataset = out["resolvable"]
        if "resolvable" not in res_ds:
            for f in findings:
                ctx.add_check(f)
            raise StageError(
                "no checkerboard test succeeded: there is no resolvable mask "
                "(add resolution.checkerboard.cell_km values and re-run)"
            )
        with NETCDF_LOCK:
            res_ds.to_netcdf(ctx.stage_dir / MASK_FILE)
        sizes = sorted(
            {
                float(_params(r)["cell_km"])
                for _, r in table.iterrows()
                if r["kind"] == "checkerboard" and bool(r["success"])
            }
        )
        fr = []
        for ph in phases:
            v = f"v{ph.lower()}"
            if f"resolved_scale_km_{v}" not in res_ds:
                continue
            t = resolved_fractions(
                res_ds[f"resolved_scale_km_{v}"].values, cov[v].values, grid, sizes
            )
            t.insert(0, "phase", ph)
            fr.append(t)
        frac = pd.concat(fr, ignore_index=True)
        frac.to_csv(ctx.stage_dir / "resolved_fractions.csv", index=False)

        findings += self._checks(cfg, res_ds, cov, grid, phases, frac, table)
        for f in findings:
            ctx.add_check(f)
        self._headline(ctx, table, picked, frac, res_ds, cov, cov_src, free_nodes, n_iter)
        self.plot(ctx)

    # ------------------------------------------------------------------ checks and summary

    @staticmethod
    def _checks(
        cfg: Any,
        res_ds: xr.Dataset,
        cov: dict[str, xr.DataArray],
        grid: Any,
        phases: list[str],
        frac: pd.DataFrame,
        table: pd.DataFrame,
    ) -> list[QCFinding]:
        out: list[QCFinding] = []
        inter = interior_mask(grid)
        small = _smallest_anomaly_km(cfg.resolution)
        for ph in phases:
            v = f"v{ph.lower()}"
            if v not in res_ds:
                continue
            ok = res_ds[v].values.astype(bool)
            covered = inter & cov[v].values.astype(bool)
            n_cov = int(covered.sum())
            if not ok.any():
                out.append(
                    _finding(
                        "no_resolvable_node",
                        "fail",
                        f"{ph}: no node is resolvable at any tested cell size "
                        "(correlation/amplitude thresholds never met): the model is not "
                        "constrained at this scale; interpret nothing from the inversion",
                    )
                )
                continue
            unres = 1.0 - float((ok & covered).sum() / n_cov) if n_cov else math.nan
            out.append(
                _finding(
                    "resolved_fraction",
                    "warn" if unres > MAX_UNRESOLVED_FRACTION else "info",
                    f"{ph}: {100 * (1 - unres):.0f}% of the covered volume is resolved at "
                    "some tested scale",
                    n_cov,
                )
            )
            sc = res_ds[f"resolved_scale_km_{v}"].values
            vals = sc[covered & np.isfinite(sc)]
            if small is not None and vals.size:
                med = float(np.median(vals))
                if med > small:
                    out.append(
                        _finding(
                            "resolution_coarser_than_anomaly",
                            "warn",
                            f"{ph}: median resolved scale {med:g} km (resolved nodes) is coarser "
                            f"than the smallest anomaly of interest ({small:g} km)",
                        )
                    )
        return out

    @staticmethod
    def _headline(
        ctx: StageContext,
        table: pd.DataFrame,
        picked: dict[str, tuple[float, float]],
        frac: pd.DataFrame,
        res_ds: xr.Dataset,
        cov: dict[str, xr.DataArray],
        cov_src: str,
        free_nodes: dict[str, int],
        n_iter: int | None,
    ) -> None:
        s = ctx.summary
        s["tests"] = int(len(table))
        s["tests_succeeded"] = int(table["success"].astype(bool).sum())
        s["iterations"] = n_iter
        s["coverage_source"] = cov_src
        for ph, (d, sm) in picked.items():
            s[f"damping_{ph}"] = d
            s[f"smoothing_{ph}"] = sm
        for ph in picked:
            t = frac[(frac["phase"] == ph) & (frac["depth_range_km"] == "all")]
            s[f"covered_nodes_{ph}"] = free_nodes.get(f"v{ph.lower()}")
            s[f"resolved_fraction_{ph}"] = {
                f"{r.cell_km:g} km": round(r.fraction_resolved, 3) for r in t.itertuples()
            }

    # ------------------------------------------------------------------ plot

    def plot(self, ctx: StageContext) -> None:
        cfg = ctx.config
        s3 = ctx.project_dir / stage_dirname("build_model")
        grid = grid_from_outputs(s3)
        runs = json.loads((ctx.stage_dir / "resolution_runs.json").read_text())
        table = pd.DataFrame(runs["table"])
        mdir = ctx.stage_dir / METRICS
        metrics = None
        if (mdir / "metrics.nc").exists():
            with NETCDF_LOCK, xr.open_dataset(mdir / "metrics.nc") as d:
                metrics = d.load()
        summary = (
            pd.DataFrame(json.loads((mdir / "summary.json").read_text()))
            if (mdir / "summary.json").exists()
            else None
        )
        resolvable = None
        if (ctx.stage_dir / MASK_FILE).exists():
            with NETCDF_LOCK, xr.open_dataset(ctx.stage_dir / MASK_FILE) as d:
                resolvable = d.load()
        noise = _noise(ctx.stage_dir / RUNS, table, s3)
        real, synth = _tradeoffs(ctx)
        _plots.make_all_figures(
            ctx.figures.save,
            grid=grid,
            table=table,
            metrics=metrics,
            summary=summary,
            resolvable=resolvable,
            depth_slices_km=cfg.plotting.depth_slices_km,
            profiles=cfg.plotting.profiles,
            noise=noise,
            real_tradeoff=real,
            synthetic_tradeoff=synth,
        )


def _params(row: Any) -> dict[str, Any]:
    p = row["params"]
    return p if isinstance(p, dict) else json.loads(p)


def _noise(runs: Path, table: pd.DataFrame, s3: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Noise realisation (first successful test per phase) and the real pick uncertainties."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ph in sorted({str(p) for p in table["phase"]}):
        sub = table[(table["phase"] == ph) & table["success"].astype(bool)]
        if sub.empty:
            continue
        d = runs / str(sub.iloc[0]["name"]) / "synth"
        try:
            a = read_observed(d / f"otimes_{ph}.dat")
            b = read_observed(d / f"otimes_{ph}_noiseless.dat")
            real = read_observed(s3 / FMTOMO_DIR / _OTIMES[ph])
        except (OSError, ValueError):
            continue
        ok = (a["time"].to_numpy(float) > -50) & (b["time"].to_numpy(float) > -50)
        noise = a["time"].to_numpy(float)[ok] - b["time"].to_numpy(float)[ok]
        sig = real.loc[real["time"] > -50, "uncertainty"].to_numpy(float)
        out[ph] = (noise, sig)
    return out


def _tradeoffs(ctx: StageContext) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Real L-curve picks and (optional) synthetic ones from ``synthetic_tradeoff.json``."""
    try:
        top = json.loads(
            (ctx.project_dir / stage_dirname("tradeoff") / "tradeoff.json").read_text()
        )
        ph = top["phases"][0]
        real = top["results"][ph]["picks"]
    except (OSError, KeyError, ValueError, IndexError):
        return None, None
    p = ctx.stage_dir / "synthetic_tradeoff.json"
    if not p.exists():
        return real, None
    try:
        return real, json.loads(p.read_text())
    except ValueError:
        return real, None
