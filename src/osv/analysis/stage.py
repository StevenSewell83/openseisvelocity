"""Stage 5 (``analyse``): analysis of the inversion outputs (``<workdir>/07_analyse/``).

Reads the hand-off files of ``invert`` (``final_model.nc``, ``relocation.parquet``,
``residuals_final.parquet``, ``inversion_iterations.csv``, run directories), the starting model of
``build_model`` and, when it exists and is complete, the ``resolvable_mask.nc`` of the
``resolution`` stage (Stage 6, which comes *after* this stage in the pipeline).  The resolution
stage is **not** a requirement: run ``osv plot analyse`` after it to add the ``*_resolvable``
figures and the resolvable masks in the exports.

Outputs (stable names)
    ``summary.json``
        Headline numbers, thresholds used and the checks.
    ``perturbation_stats.csv``
        Model change against the starting model and the 1-D background, per phase and region
        (``all``, ``covered``, ``uncovered``, ``resolvable``).
    ``exports/``
        NetCDF, VTK and CSV of the final model with masks, hypocentres and stations.
    ``figures/``
        Model slices and sections (:mod:`osv.analysis.plots_model`), residual and relocation
        figures (:mod:`osv.analysis.plots_residuals`) and, for a user model, the comparison
        figures (:mod:`osv.analysis.compare_user`).

Optional inputs are tolerated: every missing file or failing figure group becomes a check
instead of an error, so the stage completes with whatever the inversion produced.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.report.checks import QCFinding
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

__all__ = [
    "AnalyseStage",
    "Thresholds",
    "perturbation_stats",
    "relocation_summary",
    "volume_fraction",
]

FMTOMO_DIR = "fmtomo"
STARTING_MODEL = "starting_model.nc"
RESOLVABLE_MASK = "resolvable_mask.nc"
_PHASE_VAR = {"P": "vp", "S": "vs"}
_DIMS = ("depth", "lat", "lon")


@dataclass(frozen=True)
class Thresholds:
    """QC thresholds of the checks (module defaults; pass another instance to override)."""

    #: Variance reduction (%) below which the inversion is called poor.
    min_variance_reduction_pct: float = 10.0
    #: Hypocentre shift (km) that counts as "moved".
    moved_km: float = 1.0
    #: Fraction of events moved by more than ``moved_km`` that triggers a warning.
    max_moved_fraction: float = 0.25
    #: Model change (%) from the start that the 5th/95th percentile of the uncovered nodes must
    #: not exceed (i.e. more than 5% of them changed by more than this).
    outside_change_pct: float = 10.0
    #: Covered volume fraction below which the coverage is called small.
    min_covered_fraction: float = 0.05

    @classmethod
    def from_config(cls, cfg: Any) -> Thresholds:
        """Build from the ``analysis.qc`` config section (:class:`AnalysisQCConfig`)."""
        return cls(
            min_variance_reduction_pct=float(cfg.min_variance_reduction_pct),
            moved_km=float(cfg.moved_km),
            max_moved_fraction=float(cfg.max_moved_fraction),
            outside_change_pct=float(cfg.max_change_outside_coverage_pct),
            min_covered_fraction=float(cfg.min_covered_fraction),
        )


THRESHOLDS = Thresholds()


# ----------------------------------------------------------------------------------------
# numbers
# ----------------------------------------------------------------------------------------


def _interior(ds: xr.Dataset, n_cushion: int | None = None) -> xr.Dataset:
    nc = int(ds.attrs.get("n_cushion", 0)) if n_cushion is None else n_cushion
    if nc <= 0:
        return ds
    return ds.isel({d: slice(nc, ds.sizes[d] - nc) for d in _DIMS})


def volume_fraction(mask: xr.DataArray) -> float:
    """Fraction of the volume where ``mask`` is True (nodes weighted by ``cos(lat)``)."""
    m = mask.transpose(*_DIMS)
    w = np.cos(np.deg2rad(m["lat"].to_numpy()))[None, :, None] * np.ones(m.shape)
    tot = float(w.sum())
    return float((w * m.to_numpy().astype(float)).sum() / tot) if tot > 0 else float("nan")


def perturbation_stats(
    model: xr.Dataset,
    masks: dict[str, dict[str, xr.DataArray]],
) -> pd.DataFrame:
    """Statistics of the model change within regions.

    Parameters
    ----------
    model
        Interior final model with ``vp``/``vs`` and any of ``vp_start``/``vs_start`` and
        ``vp_bg``/``vs_bg``.
    masks
        ``{"vp": {"covered": da, "resolvable": da}, "vs": {...}}`` (bool, same grid); the
        regions ``all``, ``covered``, ``uncovered`` and ``resolvable`` are derived from them.

    Returns
    -------
    pandas.DataFrame
        One row per ``phase``, ``reference`` (``start``/``1d``), ``region``: ``n``, ``mean_pct``,
        ``std_pct``, ``p05_pct``, ``p95_pct``, ``max_abs_pct`` and ``rms_km_s``.
    """
    rows: list[dict[str, Any]] = []
    for ph, v in _PHASE_VAR.items():
        if v not in model:
            continue
        cur = model[v].transpose(*_DIMS).to_numpy().astype(float)
        for ref_name, ref_var in (("start", f"{v}_start"), ("1d", f"{v}_bg")):
            if ref_var not in model:
                continue
            ref = model[ref_var].transpose(*_DIMS).to_numpy().astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                pct = 100.0 * (cur - ref) / ref
            ok = np.isfinite(pct)
            regions: dict[str, np.ndarray] = {"all": ok}
            pm = masks.get(v, {})
            if "covered" in pm:
                cov = pm["covered"].transpose(*_DIMS).to_numpy().astype(bool)
                regions["covered"] = ok & cov
                regions["uncovered"] = ok & ~cov
                if "resolvable" in pm:
                    regions["resolvable"] = (
                        ok & cov & pm["resolvable"].transpose(*_DIMS).to_numpy().astype(bool)
                    )
            for region, sel in regions.items():
                x = pct[sel]
                d = (cur - ref)[sel]
                rows.append(
                    {
                        "phase": ph,
                        "reference": ref_name,
                        "region": region,
                        "n": int(x.size),
                        "mean_pct": float(x.mean()) if x.size else np.nan,
                        "std_pct": float(x.std()) if x.size else np.nan,
                        "p05_pct": float(np.percentile(x, 5)) if x.size else np.nan,
                        "p95_pct": float(np.percentile(x, 95)) if x.size else np.nan,
                        "max_abs_pct": float(np.abs(x).max()) if x.size else np.nan,
                        "rms_km_s": float(np.sqrt(np.mean(d**2))) if x.size else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def relocation_summary(reloc: pd.DataFrame, moved_km: float) -> dict[str, Any]:
    """Medians of the hypocentre shifts and the fraction of events moved by more than ``moved_km``.

    ``reloc`` needs ``dh_km``, ``dz_km``, ``d3_km`` (see ``osv.analysis.residuals``) and may
    have ``dt_s``; uses one phase's table.
    """
    if not len(reloc):
        return {"n_events": 0}
    out: dict[str, Any] = {
        "n_events": int(len(reloc)),
        "median_dh_km": float(reloc["dh_km"].median()),
        "median_abs_dz_km": float(reloc["dz_km"].abs().median()),
        "median_d3_km": float(reloc["d3_km"].median()),
        "p90_d3_km": float(reloc["d3_km"].quantile(0.9)),
        "moved_fraction": float((reloc["d3_km"] > moved_km).mean()),
        "moved_count": int((reloc["d3_km"] > moved_km).sum()),
    }
    if "dt_s" in reloc and reloc["dt_s"].notna().any():
        out["median_abs_dt_s"] = float(reloc["dt_s"].abs().median())
    return out


# ----------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------


@dataclass
class _Inputs:
    """Everything the stage reads."""

    model: xr.Dataset  # full grid, final_model.nc
    starting: xr.Dataset | None
    model_1d: pd.DataFrame | None
    events: pd.DataFrame | None  # P (or first phase) relocation: lat, lon, depth_km_bsl
    reloc: pd.DataFrame | None  # all phases
    stations: pd.DataFrame | None
    iterations: pd.DataFrame | None
    residuals: pd.DataFrame | None
    coverage: dict[str, xr.DataArray]  # interior dws per phase key ("dws", "dws_vs")
    resolvable: xr.Dataset | None  # interior
    resolution_dir: Path | None
    invert_dir: Path
    build_dir: Path


def _load_nc(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as raw:
        return raw.load()


def _optional(ctx: StageContext, what: str, fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - optional inputs must not stop the stage
        ctx.logger.warning("%s not available: %s", what, exc)
        return None


def resolution_mask_path(project_dir: Path) -> Path | None:
    """``resolvable_mask.nc`` of a completed resolution stage, else None."""
    d = project_dir / stage_dirname("resolution")
    m = read_manifest(d)
    p = d / RESOLVABLE_MASK
    return p if m is not None and m.get("status") == COMPLETE and p.exists() else None


def _load_inputs(ctx: StageContext) -> _Inputs:
    inv = ctx.upstream("invert").dir
    bld = ctx.upstream("build_model").dir
    an = ctx.upstream("velest_analyse").dir
    model = _load_nc(inv / "final_model.nc")
    nc = int(model.attrs.get("n_cushion", 1))
    starting = _optional(ctx, "starting model", lambda: _load_nc(bld / STARTING_MODEL))
    model_1d = _optional(ctx, "1-D model", lambda: pd.read_csv(an / "final_model.csv"))
    reloc = _optional(ctx, "relocation table", lambda: pd.read_parquet(inv / "relocation.parquet"))
    events = None
    if reloc is not None and len(reloc):
        first = reloc[reloc["phase"] == reloc["phase"].iloc[0]] if "phase" in reloc else reloc
        events = first[["lat", "lon", "depth_km"]].rename(columns={"depth_km": "depth_km_bsl"})
    stations = _optional(ctx, "stations", lambda: pd.read_csv(bld / FMTOMO_DIR / "stations.csv"))
    its = _optional(ctx, "iteration table", lambda: pd.read_csv(inv / "inversion_iterations.csv"))
    res = _optional(
        ctx, "final residuals", lambda: pd.read_parquet(inv / "residuals_final.parquet")
    )
    inner = _interior(model, nc)
    cov = {k: inner[k] for k in inner.data_vars if str(k).startswith("dws")}
    rpath = resolution_mask_path(ctx.project_dir)
    resolvable = None
    if rpath is not None:
        resolvable = _optional(
            ctx, "resolvable mask", lambda: _interior(_load_nc(rpath), nc).load()
        )
    return _Inputs(
        model, starting, model_1d, events, reloc, stations, its, res, cov, resolvable,
        rpath.parent if rpath is not None and resolvable is not None else None, inv, bld,
    )  # fmt: skip


def _reference_1d(model: xr.Dataset, model_1d: pd.DataFrame | None) -> Any:
    """1-D reference for ``ModelFields``: the model's own ``*_bg`` or the VELEST table."""
    if "vp_bg" in model or model_1d is None:
        return None
    df = model_1d.rename(columns={"depth_top_km": "depth"})
    return df if {"depth", "vp"} <= set(df.columns) else None


def _phase_masks(
    inp: _Inputs, threshold: float
) -> tuple[dict[str, dict[str, xr.DataArray]], dict[str, xr.DataArray]]:
    """Covered/resolvable masks per velocity variable and named bool masks for export."""
    from osv.analysis.coverage import coverage_mask

    per: dict[str, dict[str, xr.DataArray]] = {}
    export: dict[str, xr.DataArray] = {}
    for ph, v in _PHASE_VAR.items():
        key = "dws" if ph == "P" else "dws_vs"
        if key not in inp.coverage:
            continue
        cov = coverage_mask(inp.coverage[key], threshold).astype(bool)
        per[v] = {"covered": cov}
        export[f"covered_{v}"] = cov
        export[key] = inp.coverage[key]
        if inp.resolvable is not None:
            res = inp.resolvable
            for cand in (v, "resolvable"):
                if cand in res:
                    r = res[cand].astype(bool)
                    r = r.reindex_like(cov, method="nearest")
                    per[v]["resolvable"] = r
                    export[f"resolvable_{v}"] = r
                    break
    return per, export


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------


class _Prefixed:
    """Registry proxy that prefixes figure names (for the second phase's residual figures)."""

    def __init__(self, reg: Any, prefix: str) -> None:
        self._reg, self._prefix = reg, prefix

    def save(self, fig: Any, name: str, caption: str, section: str | None = None) -> list[str]:
        return self._reg.save(
            fig, f"{self._prefix}{name}", f"{self._prefix[:-1]}: {caption}", section
        )


def _model_figures(ctx: StageContext, inp: _Inputs) -> None:
    from osv.analysis.plots_model import make_model_figures

    cov = dict(inp.coverage) or None
    inner = _interior(inp.model)
    names = make_model_figures(
        ctx.figures.save,
        inner,
        _reference_1d(inner, inp.model_1d),
        None if inp.starting is None else _interior(inp.starting),
        cov,
        inp.events,
        inp.stations,
        ctx.config,
        resolvable_mask=inp.resolvable,
    )
    ctx.logger.info("%d model figures", len(names))


def _residual_figures(ctx: StageContext, inp: _Inputs) -> list[str]:
    """Residual and relocation figures from the run directories; returns the phases done."""
    from osv.analysis.plots_residuals import make_residual_figures

    done: list[str] = []
    phases = list(inp.iterations["phase"].unique()) if inp.iterations is not None else ["P"]
    nonlinear = ctx.config.fmtomo.relocate == "nonlinear"
    for i, ph in enumerate(phases):
        run = inp.invert_dir / f"run_{ph}"
        mp = run / "mapping.csv"
        if not (run / "iterations").is_dir() or not mp.exists():
            ctx.logger.warning("no run directory with iterations/mapping for phase %s", ph)
            continue
        reg: Any = ctx.figures if i == 0 else _Prefixed(ctx.figures, f"{ph.lower()}_")
        try:
            make_residual_figures(
                run, pd.read_csv(mp), reg, nonlinear=nonlinear, section=f"residuals_{ph}"
            )
            done.append(ph)
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("residual figures for %s failed: %s", ph, exc, exc_info=True)
    return done


@dataclass
class _UserComparison:
    summary: dict[str, Any]
    figures: list[str]


def _user_comparison(ctx: StageContext, inp: _Inputs, threshold: float) -> _UserComparison | None:
    """Final model against the user model (the Stage 3 starting model and the wells)."""
    cfg = ctx.config
    if cfg.model3d.source == "velest_1d" or inp.starting is None:
        return None
    from osv.analysis.compare_user import compare_with_user_model, make_comparison_figures

    wells = None
    if cfg.model3d.source in ("wells", "geology+wells"):
        from osv.model3d.stage import grid_from_outputs
        from osv.model3d.wells import build_wells_model

        wells = build_wells_model(
            cfg.model3d.user_model.wells, grid_from_outputs(inp.build_dir), background=None
        )
    inner = _interior(inp.model)
    res = compare_with_user_model(
        inner,
        _interior(inp.starting),
        wells,
        starting=_interior(inp.starting),
        coverage=dict(inp.coverage) or None,
        dws_threshold=threshold,
        reference_1d=_reference_1d(inner, inp.model_1d),
    )
    names = make_comparison_figures(ctx.figures.save, res, cfg)
    return _UserComparison(dict(res.summary), names)


# ----------------------------------------------------------------------------------------
# exports
# ----------------------------------------------------------------------------------------


def _exports(ctx: StageContext, inp: _Inputs, export_masks: dict[str, xr.DataArray]) -> None:
    from osv.analysis.exports import export_model

    inner = _interior(inp.model)
    keep = [v for v in ("vp", "vs", "vp_bg", "vs_bg") if v in inner]
    core = inner[keep]
    hyp = None
    if inp.reloc is not None and len(inp.reloc):
        first = inp.reloc[inp.reloc["phase"] == inp.reloc["phase"].iloc[0]]
        cols = [c for c in ("event_id", "lat", "lon", "depth_km", "d3_km") if c in first]
        hyp = first[cols]
    sta = None
    if inp.stations is not None and {"lat", "lon", "depth_km"} <= set(inp.stations.columns):
        sta = inp.stations
    try:
        export_model(
            core, ctx.stage_dir / "exports", masks=export_masks, hypocentres=hyp, stations=sta
        )
    except Exception as exc:  # noqa: BLE001 - exports are convenience files
        ctx.logger.warning("export failed: %s", exc, exc_info=True)
        ctx.add_check(QCFinding("export_failed", "warn", f"model export failed: {exc}"))


# ----------------------------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------------------------


@register
class AnalyseStage(Stage):
    """Stage 5: figures, exports, summary numbers and checks for the inversion result."""

    name = "analyse"
    requires = ("invert", "build_model", "prep", "velest_analyse")
    config_sections = (
        "analysis",
        "plotting",
        "model3d.source",
        "model3d.user_model",
        "fmtomo.relocate",
    )
    outputs = ("summary.json", "perturbation_stats.csv")

    # ------------------------------------------------------------------ run
    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        thr = float(cfg.plotting.dws_mask_threshold)
        qc = Thresholds.from_config(cfg.analysis.qc)
        inp = _load_inputs(ctx)
        if inp.starting is None:
            ctx.add_check(
                QCFinding(
                    "no_starting_model",
                    "warn",
                    "starting_model.nc of build_model is missing: no change-from-start figures",
                )
            )
        findings: list[QCFinding] = []
        masks, export_masks = _phase_masks(inp, thr)
        summary: dict[str, Any] = {"dws_mask_threshold": thr, "thresholds": asdict(qc)}

        # ---- coverage
        if not masks:
            findings.append(
                QCFinding(
                    "no_coverage",
                    "warn",
                    "final_model.nc has no dws variable: figures and statistics are not masked "
                    "by ray coverage",
                )
            )
        for v, m in masks.items():
            f = volume_fraction(m["covered"])
            summary[f"covered_fraction_{v}"] = round(f, 4)
            if f < qc.min_covered_fraction:
                findings.append(
                    QCFinding(
                        "small_coverage",
                        "warn",
                        f"only {100 * f:.1f}% of the {v.upper()} volume has DWS >= {thr:g}; "
                        "lower plotting.dws_mask_threshold or check the ray geometry",
                    )
                )
            if "resolvable" in m:
                summary[f"resolvable_fraction_{v}"] = round(
                    volume_fraction(m["covered"] & m["resolvable"]), 4
                )
        summary["resolvable_mask_applied"] = inp.resolvable is not None
        if inp.resolvable is None:
            findings.append(
                QCFinding(
                    "resolution_not_applied",
                    "info",
                    "no completed resolution stage: figures are masked by DWS only; run "
                    "`osv resolution` then `osv plot analyse` to add the resolvable masks",
                )
            )

        # ---- model perturbation
        stats = perturbation_stats(_interior(inp.model), masks)
        stats.to_csv(ctx.stage_dir / "perturbation_stats.csv", index=False)
        self._perturbation_summary(summary, findings, stats, masks, qc)

        # ---- misfit
        self._misfit_summary(summary, findings, inp, qc)

        # ---- relocation
        if inp.reloc is not None and len(inp.reloc):
            for ph in inp.reloc["phase"].unique() if "phase" in inp.reloc else [None]:
                sub = inp.reloc if ph is None else inp.reloc[inp.reloc["phase"] == ph]
                rs = relocation_summary(sub, qc.moved_km)
                sfx = "" if ph is None else f"_{ph}"
                summary[f"relocation{sfx}"] = rs
                for k in ("median_dh_km", "median_abs_dz_km", "moved_fraction"):
                    if k in rs:
                        ctx.summary[f"{k}{sfx}"] = round(rs[k], 3)
                tag = f" ({ph})" if ph is not None else ""
                if rs.get("moved_fraction", 0.0) > qc.max_moved_fraction:
                    findings.append(
                        QCFinding(
                            "many_events_moved",
                            "warn",
                            f"{100 * rs['moved_fraction']:.0f}% of the events{tag}"
                            f" moved by more than {qc.moved_km:g} km: check the "
                            "hypocentre damping (fmtomo.source_damping_*) and the velocity model",
                            rs["moved_count"],
                        )
                    )
        else:
            findings.append(QCFinding("no_relocation", "info", "no relocation table available"))

        # ---- exports, figures
        _exports(ctx, inp, export_masks)
        self._figures(ctx, inp, summary, findings, thr)

        for f in findings:
            ctx.add_check(f)
        summary["checks"] = [f.to_dict() for f in findings]
        (ctx.stage_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
        for k, val in summary.items():
            if isinstance(val, int | float | str | bool) and k not in ctx.summary:
                ctx.summary[k] = val

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _perturbation_summary(
        summary: dict[str, Any],
        findings: list[QCFinding],
        stats: pd.DataFrame,
        masks: dict[str, dict[str, xr.DataArray]],
        qc: Thresholds,
    ) -> None:
        out: dict[str, Any] = {}
        for r in stats.itertuples():
            region = "covered" if "covered" in masks.get(_PHASE_VAR[r.phase], {}) else "all"
            if r.region != region:
                continue
            out[f"{r.phase}_vs_{r.reference}"] = {
                "region": r.region,
                "n": r.n,
                "mean_pct": round(r.mean_pct, 3),
                "std_pct": round(r.std_pct, 3),
                "p05_pct": round(r.p05_pct, 3),
                "p95_pct": round(r.p95_pct, 3),
                "max_abs_pct": round(r.max_abs_pct, 3),
            }
        summary["perturbation"] = out
        for key, val in out.items():
            if key.endswith("_vs_start"):
                summary[f"d{key[0].lower()}_std_pct_covered"] = val["std_pct"]
        AnalyseStage._outside_check(summary, findings, stats, qc)

    @staticmethod
    def _outside_check(
        summary: dict[str, Any], findings: list[QCFinding], stats: pd.DataFrame, qc: Thresholds
    ) -> None:
        for ph in _PHASE_VAR:
            sel = stats[
                (stats["phase"] == ph)
                & (stats["reference"] == "start")
                & (stats["region"] == "uncovered")
            ]
            if not len(sel) or not sel.iloc[0]["n"]:
                continue
            row = sel.iloc[0]
            lim = qc.outside_change_pct
            if max(abs(row["p05_pct"]), abs(row["p95_pct"])) > lim:
                findings.append(
                    QCFinding(
                        "change_outside_coverage",
                        "warn",
                        f"{ph}: more than 5% of the nodes without ray coverage changed by over "
                        f"{lim:g}% from the starting model (max {row['max_abs_pct']:.1f}%): "
                        "smoothing or damping lets the model drift where no data constrain it",
                        int(row["n"]),
                    )
                )

    @staticmethod
    def _misfit_summary(
        summary: dict[str, Any], findings: list[QCFinding], inp: _Inputs, qc: Thresholds
    ) -> None:
        its = inp.iterations
        if its is not None and len(its) and "phase" in its:
            for ph, g in its.groupby("phase"):
                g = g.sort_values("iter") if "iter" in g else g
                r0, r1 = float(g["rms_ms"].iloc[0]), float(g["rms_ms"].iloc[-1])
                v0, v1 = float(g["variance_s2"].iloc[0]), float(g["variance_s2"].iloc[-1])
                vr = 100.0 * (1.0 - v1 / v0) if v0 > 0 else float("nan")
                summary[f"rms_start_ms_{ph}"] = round(r0, 2)
                summary[f"rms_final_ms_{ph}"] = round(r1, 2)
                summary[f"variance_reduction_pct_{ph}"] = round(vr, 1)
                if not np.isfinite(vr):
                    continue
                if vr < 0:
                    findings.append(
                        QCFinding(
                            "variance_increased",
                            "fail",
                            f"{ph}: the variance increased by {-vr:.1f}% in the inversion; the "
                            "final model fits worse than the starting one (check regularisation)",
                        )
                    )
                elif vr < qc.min_variance_reduction_pct:
                    findings.append(
                        QCFinding(
                            "poor_variance_reduction",
                            "warn",
                            f"{ph}: variance reduction only {vr:.1f}% (below "
                            f"{qc.min_variance_reduction_pct:g}%): the data may be fit "
                            "already by the starting model, or regularisation is too strong",
                        )
                    )
        res = inp.residuals
        if res is not None and len(res) and "residual" in res:
            by = res.groupby("phase") if "phase" in res else [("all", res)]
            for ph, g in by:
                r = g["residual"].to_numpy(float)
                r = r[np.isfinite(r)]
                if r.size:
                    summary[f"final_rms_ms_{ph}"] = round(1000.0 * float(np.sqrt(np.mean(r**2))), 2)
                    summary[f"final_mean_residual_ms_{ph}"] = round(1000.0 * float(r.mean()), 2)
                    summary[f"n_picks_{ph}"] = int(r.size)
        elif its is None:
            findings.append(
                QCFinding(
                    "no_misfit", "info", "no iteration or residual tables: misfit not summarised"
                )
            )

    def _figures(
        self,
        ctx: StageContext,
        inp: _Inputs,
        summary: dict[str, Any],
        findings: list[QCFinding],
        thr: float,
    ) -> None:
        for what, fn in (("model figures", lambda: _model_figures(ctx, inp)),):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                ctx.logger.warning("%s failed: %s", what, exc, exc_info=True)
                findings.append(QCFinding("figures_failed", "warn", f"{what} failed: {exc}"))
        done = _residual_figures(ctx, inp)
        summary["residual_figures_phases"] = done
        if not done:
            findings.append(
                QCFinding(
                    "no_residual_figures",
                    "info",
                    "residual figures skipped: the invert run directories (iterations/NN and "
                    "mapping.csv) are missing or incomplete",
                )
            )
        try:
            uc = _user_comparison(ctx, inp, thr)
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("user-model comparison failed: %s", exc, exc_info=True)
            findings.append(
                QCFinding("user_comparison_failed", "warn", f"user-model comparison failed: {exc}")
            )
            uc = None
        if uc is not None:
            summary["user_comparison"] = uc.summary
            summary["user_comparison_figures"] = uc.figures

    # ------------------------------------------------------------------ plot
    def plot(self, ctx: StageContext) -> None:
        """Regenerate figures and exports (adds the resolvable masks once Stage 6 has run)."""
        thr = float(ctx.config.plotting.dws_mask_threshold)
        try:
            inp = _load_inputs(ctx)
        except StageError:
            raise
        masks, export_masks = _phase_masks(inp, thr)
        findings: list[QCFinding] = []
        summary: dict[str, Any] = {}
        _exports(ctx, inp, export_masks)
        self._figures(ctx, inp, summary, findings, thr)
        for f in findings:
            ctx.logger.warning("%s", f)
        # record whether the resolvable mask is now applied
        sp = ctx.stage_dir / "summary.json"
        try:
            data = json.loads(sp.read_text())
        except (OSError, ValueError):
            data = {}
        data["resolvable_mask_applied"] = inp.resolvable is not None
        for v, m in masks.items():
            if "resolvable" in m:
                data[f"resolvable_fraction_{v}"] = round(
                    volume_fraction(m["covered"] & m["resolvable"]), 4
                )
        if inp.resolvable is not None and isinstance(data.get("checks"), list):
            data["checks"] = [
                c for c in data["checks"] if c.get("code") != "resolution_not_applied"
            ]
        sp.write_text(json.dumps(_clean(data), indent=1))


def _clean(obj: Any) -> Any:
    """JSON-safe copy (numpy scalars to Python, NaN to None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
