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

Settings are in the ``validation`` config section.  When ``validation.enabled`` is false the
stage completes with empty outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from osv.model3d.stage import FMTOMO_DIR, grid_from_outputs
from osv.report.checks import QCFinding
from osv.stages import (
    COMPLETE,
    Stage,
    StageContext,
    read_manifest,
    register,
    stage_dirname,
)
from osv.validation.plots import make_figures
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


def _invert_vgrids(ctx: StageContext) -> Path | None:
    """``final_vgrids.in`` of a completed ``invert`` stage, else ``None``."""
    d = ctx.project_dir / stage_dirname("invert")
    m = read_manifest(d)
    if m is not None and m.get("status") == COMPLETE and (d / FINAL_VGRIDS).is_file():
        return d / FINAL_VGRIDS
    return None


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
    )
    tools = ("velest", "fm3d", "frechgen", "invert3d", "residuals")
    outputs = (EVENTS_FILE, STATS_FILE)

    def inputs(self, ctx: StageContext) -> list[Path]:
        p = _invert_vgrids(ctx)
        return [p] if p is not None else []

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
