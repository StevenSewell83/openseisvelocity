"""Stage 7 (``validation``): relocate the held-out events with the 1-D and the final 3-D model.

``validation`` (``<workdir>/09_validation/``) needs ``prep``, ``velest_analyse`` and
``build_model``.  If ``invert`` has completed, its ``final_vgrids.in`` is the 3-D model;
otherwise the Stage 3 starting model (``fmtomo/vgrids.in``) is used and a ``warn`` check says
so.  ``interfaces.in`` and ``propgrid.in`` always come from Stage 3 (``fmtomo/``).

The events of the Stage 1 ``validation`` split are relocated with VELEST (final 1-D model and
station delays) and with FMTOMO (fixed 3-D velocities), then compared
(:func:`osv.validation.relocate.relocate_validation`).

Outputs (stable names)
    ``validation_events.parquet``
        One row per held-out event: 1-D and 3-D hypocentres and RMS, ``rms_change`` (positive:
        3-D better), ``dh_km``, ``dz_km``.
    ``stats.json``
        Paired statistics, phases, findings and the model used (``model_source``).
    ``work/``
        The relocation run directories and tables.

When ``validation.bootstrap.enabled`` (default false) the event-resampled bootstrap/jackknife
inversions (:func:`osv.validation.bootstrap.run_bootstrap`) also run, from the Stage 3 FMTOMO
inputs, with the regularisation chosen by ``tradeoff`` (``tradeoff.json``; ``fmtomo.damping`` and
``fmtomo.smoothing`` when that stage has not completed) and the DWS of ``invert``'s
``final_model.nc`` when available.  Output: ``bootstrap/`` (``bootstrap_std.nc``,
``bootstrap_samples.csv``, ...) and the ``Bootstrap`` figures.  ``tradeoff`` is not required.

Settings are in the ``validation`` config section.  When ``validation.enabled`` is false the
stage completes with empty outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from osv.model3d.stage import FMTOMO_DIR, grid_from_outputs
from osv.plotting.slices import crop_interior
from osv.report.checks import QCFinding
from osv.stages import (
    COMPLETE,
    Stage,
    StageContext,
    read_manifest,
    register,
    stage_dirname,
)
from osv.validation.bootstrap import BootstrapError, BootstrapInputs, BootstrapResult, run_bootstrap
from osv.validation.plots import make_figures
from osv.validation.plots_bootstrap import make_figures as make_bootstrap_figures
from osv.validation.relocate import (
    MAX_FM3D_JOBS,
    RelocateOptions,
    ValidationResult,
    relocate_validation,
)

EVENTS_FILE = "validation_events.parquet"
STATS_FILE = "stats.json"
WORK_DIR = "work"
FINAL_VGRIDS = "final_vgrids.in"
BOOTSTRAP_DIR = "bootstrap"


def _invert_vgrids(ctx: StageContext) -> Path | None:
    """``final_vgrids.in`` of a completed ``invert`` stage, else ``None``."""
    d = ctx.project_dir / stage_dirname("invert")
    m = read_manifest(d)
    if m is not None and m.get("status") == COMPLETE and (d / FINAL_VGRIDS).is_file():
        return d / FINAL_VGRIDS
    return None


def _invert_station_terms(ctx: StageContext) -> Path | None:
    """``station_terms.csv`` of a completed ``invert`` stage (iterated terms), else ``None``."""
    from osv.fmtomo.stage import STATION_TERMS_FILE

    if _invert_vgrids(ctx) is None:
        return None
    p = ctx.project_dir / stage_dirname("invert") / STATION_TERMS_FILE
    return p if p.is_file() else None


def _tradeoff_json(ctx: StageContext) -> Path | None:
    """``tradeoff.json`` of a completed ``tradeoff`` stage, else ``None``."""
    d = ctx.project_dir / stage_dirname("tradeoff")
    m = read_manifest(d)
    if m is not None and m.get("status") == COMPLETE and (d / "tradeoff.json").is_file():
        return d / "tradeoff.json"
    return None


def _invert_dws(ctx: StageContext, var: str) -> xr.DataArray | None:
    """Derivative weight sum of the completed ``invert`` stage (``None`` if unavailable)."""
    d = ctx.project_dir / stage_dirname("invert")
    m = read_manifest(d)
    p = d / "final_model.nc"
    if m is None or m.get("status") != COMPLETE or not p.is_file():
        return None
    try:
        with xr.open_dataset(p) as ds:
            for name in ("dws" if var == "vp" else "dws_vs", "dws"):
                if name in ds:
                    return ds[name].load()
    except (OSError, ValueError):
        return None
    return None


def _bootstrap_summary(
    res: BootstrapResult, var: str, threshold: float
) -> tuple[dict[str, float], bool]:
    """Median and maximum velocity std (km/s, percent) inside the coverage; ``covered`` flag."""
    import numpy as np

    ds = crop_interior(res.stats)
    std = ds[f"{var}_std"].values
    pct = ds[f"d{var}_std_pct"].values
    covered = False
    if "dws" in ds:
        mask = ds["dws"].values >= threshold
        if mask.any():
            std, pct, covered = std[mask], pct[mask], True
    out: dict[str, float] = {}
    for key, a in (("std", std), ("std_pct", pct)):
        a = a[np.isfinite(a)]
        if a.size:
            out[f"median_{key}"] = float(np.median(a))
            out[f"max_{key}"] = float(a.max())
    return out, covered


def _options(cfg: Any) -> RelocateOptions:
    v = cfg.validation
    use_s = v.use_s if v.use_s is not None else bool({"S", "vpvs"} & set(cfg.fmtomo.phases))
    return RelocateOptions(
        n_iterations=v.max_iterations,
        source_sigma_km=v.source_sigma_km,
        velest_ittmax=v.velest_ittmax,
        phases=("P", "S") if use_s else ("P",),
        worse_fraction_warn=v.worse_fraction_warn,
        min_events_stat=v.min_events_stat,
        improve_tol_ms=v.improve_tol_ms,
        max_hypo_shift_warn_km=v.max_hypo_shift_warn_km,
    )


def _clean(obj: Any) -> Any:
    """Strictly JSON-serialisable copy (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, str | bytes):
        try:
            obj = obj.item()
        except (ValueError, TypeError):
            return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


@register
class ValidationStage(Stage):
    """Stage 7: held-out events relocated with the 1-D and the 3-D model."""

    name = "validation"
    requires = ("prep", "velest_analyse", "build_model")
    config_sections = (
        "project.random_seed",
        "validation",
        "velest",
        "fmtomo",
        "model3d.apply_velest_station_delays",
        "plotting.dws_mask_threshold",
    )
    tools = ("velest", "fm3d", "frechgen", "invert3d", "residuals")
    outputs = (EVENTS_FILE, STATS_FILE)

    def inputs(self, ctx: StageContext) -> list[Path]:
        paths = [p for p in (_invert_vgrids(ctx), _invert_station_terms(ctx)) if p is not None]
        if ctx.config.validation.bootstrap.enabled:
            t = _tradeoff_json(ctx)
            if t is not None:
                paths.append(t)
        return paths

    # ------------------------------------------------------------------ run

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        log = ctx.logger
        out = ctx.stage_dir
        if not cfg.validation.enabled:
            pd.DataFrame({"event_id": []}).to_parquet(out / EVENTS_FILE)
            (out / STATS_FILE).write_text(json.dumps({"enabled": False}, indent=2))
            ctx.add_check(
                QCFinding("validation_disabled", "info", "validation.enabled is false: skipped")
            )
            ctx.summary["enabled"] = False
            return

        prep = ctx.upstream("prep")
        an = ctx.upstream("velest_analyse")
        s3 = ctx.upstream("build_model")
        events = pd.read_parquet(prep.path("events.parquet"))
        picks = pd.read_parquet(prep.path("picks.parquet"))
        stations = pd.read_parquet(prep.path("stations.parquet"))
        delays = pd.read_csv(an.path("station_delays.csv"))
        grid = grid_from_outputs(s3.dir)
        fm_dir = s3.dir / FMTOMO_DIR

        findings: list[QCFinding] = []
        vgrids = _invert_vgrids(ctx)
        if vgrids is None:
            vgrids = s3.path(f"{FMTOMO_DIR}/vgrids.in")
            source = "build_model"
            findings.append(
                QCFinding(
                    "validation_starting_model",
                    "warn",
                    "the invert stage has not completed: the held-out events were relocated in "
                    "the Stage 3 starting model, not the inverted model (run `osv invert`, then "
                    "`osv validation --force`)",
                )
            )
        else:
            source = "invert"
        log.info("validation: 3-D model %s (%s)", vgrids, source)

        n_jobs = cfg.validation.n_jobs or ctx.n_jobs
        res = relocate_validation(
            events,
            picks,
            an.path("final_model.mod"),
            delays,
            vgrids,
            grid,
            cfg,
            out / WORK_DIR,
            min(int(n_jobs), MAX_FM3D_JOBS),
            stations=stations,
            model_inputs_dir=fm_dir,
            options=_options(cfg),
            station_terms_3d=(
                pd.read_csv(t) if (t := _invert_station_terms(ctx)) is not None else None
            ),
        )
        findings = findings + list(res.findings)

        table = res.events.copy()
        table.attrs = {}
        table.to_parquet(out / EVENTS_FILE)
        stats = _clean(res.summary())
        stats["model_source"] = source
        stats["model_file"] = str(vgrids)
        (out / STATS_FILE).write_text(json.dumps(stats, indent=2))
        for f in findings:
            ctx.add_check(f)
        self._headline(ctx, res, source)
        self.plot(ctx)
        if cfg.validation.bootstrap.enabled:
            self._bootstrap(ctx, fm_dir, int(n_jobs))

    @staticmethod
    def _bootstrap(ctx: StageContext, fm_dir: Path, n_jobs: int) -> None:
        """Run the bootstrap/jackknife inversions; figures, summary and checks."""
        cfg = ctx.config
        log = ctx.logger
        bc = cfg.validation.bootstrap
        phases = set(cfg.fmtomo.phases)
        phase = "P" if phases & {"P", "vpvs"} else "S"
        var = "vp" if phase == "P" else "vs"
        tj = _tradeoff_json(ctx)
        if tj is None:
            ctx.add_check(
                QCFinding(
                    "bootstrap_default_regularisation",
                    "info",
                    "the tradeoff stage has not completed: the bootstrap uses fmtomo.damping and "
                    "fmtomo.smoothing",
                )
            )
        dws = _invert_dws(ctx, var)
        threshold = float(cfg.plotting.dws_mask_threshold)
        ctx.invalidate_cache(BOOTSTRAP_DIR, extra_stages=("tradeoff", "invert"))
        wd = ctx.stage_dir / BOOTSTRAP_DIR
        s = ctx.summary
        try:
            res = run_bootstrap(
                BootstrapInputs.from_stage3(fm_dir, phase),
                cfg,
                tj,
                wd,
                n_jobs,
                dws=dws,
            )
        except BootstrapError as exc:
            log.warning("bootstrap failed: %s", exc)
            ctx.add_check(QCFinding("bootstrap_failed", "warn", str(exc)))
            s["bootstrap_samples_ok"] = 0
            return
        n_total = len(res.table)
        s["bootstrap_method"] = res.method
        s["bootstrap_samples_ok"] = res.n_ok
        s["bootstrap_samples"] = n_total
        if n_total and res.n_ok < bc.min_ok_fraction * n_total:
            ctx.add_check(
                QCFinding(
                    "bootstrap_few_samples",
                    "warn",
                    f"only {res.n_ok} of {n_total} {res.method} inversions succeeded; the "
                    f"uncertainty estimate is poorly determined (failed: {', '.join(res.failed)})",
                    count=n_total - res.n_ok,
                )
            )
        stats, covered = _bootstrap_summary(res, var, threshold)
        s["bootstrap_std_within_coverage"] = covered
        for key, label, nd in (
            ("median_std", f"bootstrap_median_{var}_std_kms", 4),
            ("max_std", f"bootstrap_max_{var}_std_kms", 4),
            ("median_std_pct", "bootstrap_median_std_pct", 3),
            ("max_std_pct", "bootstrap_max_std_pct", 3),
        ):
            if key in stats:
                s[label] = round(stats[key], nd)
        med = stats.get("median_std_pct")
        if med is not None and med > bc.std_warn_pct:
            where = "inside the DWS coverage" if covered else "over the model interior"
            ctx.add_check(
                QCFinding(
                    "bootstrap_high_std",
                    "warn",
                    f"median {res.method} velocity-perturbation std is {med:.2f}% {where} "
                    f"(threshold {bc.std_warn_pct:g}%): the model is not stable against "
                    "resampling of the events",
                    details={"median_std_pct": med, "threshold_pct": float(bc.std_warn_pct)},
                )
            )
        make_bootstrap_figures(
            res, ctx.figures, section="Bootstrap", dws_threshold=threshold if covered else None
        )

    @staticmethod
    def _headline(ctx: StageContext, res: ValidationResult, source: str) -> None:
        s = ctx.summary
        s["model_3d"] = source
        s["events"] = len(res.events)
        s["phases"] = "+".join(res.phases)
        r = res.stats.get("rms", {})
        for key, label, scale in (
            ("median_1d", "median_rms_1d_ms", 1000.0),
            ("median_3d", "median_rms_3d_ms", 1000.0),
            ("median_improvement_s", "median_rms_improvement_ms", 1000.0),
            ("frac_3d_better", "fraction_3d_better", 1.0),
            ("frac_3d_worse", "fraction_3d_worse", 1.0),
            ("p_two_sided", "wilcoxon_p", 1.0),
        ):
            v = r.get(key)
            if isinstance(v, int | float) and math.isfinite(v):
                s[label] = round(float(v) * scale, 4)

    # ------------------------------------------------------------------ plot

    def plot(self, ctx: StageContext) -> None:
        out = ctx.stage_dir
        events = pd.read_parquet(out / EVENTS_FILE)
        if events.empty:
            return
        stats = json.loads((out / STATS_FILE).read_text())
        res = ValidationResult(
            events=events,
            stats=stats.get("stats", {}),
            phases=tuple(stats.get("phases", ("P",))),
        )
        make_figures(res, ctx.figures)


__all__ = ["ValidationStage"]
