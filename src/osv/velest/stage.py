"""Stage 2: Monte Carlo VELEST (``velest_mc``) and ensemble analysis (``velest_analyse``).

``velest_mc`` (``<workdir>/02_velest_mc/``)
    Generates the trial 1-D models, runs the damping schedule (and optional reseeded rounds)
    for each of them with VELEST in parallel, and stores the run table.  Files:
    ``trial_models.parquet``, ``runs.parquet`` (one row per model and round, paths relative to
    the stage directory), ``workflow_history.csv``, ``mc_summary.json`` and ``runs/`` (one
    directory per VELEST run; compacted unless ``velest.compact_runs`` is false).

``velest_analyse`` (``<workdir>/03_velest_analyse/``)
    Parses the runs, selects the best models (``velest.select``), derives the final 1-D model
    and its uncertainty, re-runs VELEST with that model fixed for consistent station delays and
    hypocentres, repeats VELEST from its own outputs until stable (a drift check), runs the
    stability tests and an optional Vp/Vs grid search.

    Hand-off files (stable names, read by Stage 3):

    ``final_model.mod``, ``final_model.csv``
        The final model (per-layer median/mean/best of the selected runs) in VELEST format and
        as a table (``layer, depth_top_km, vp, vs, vpvs`` with ``*_std``, ``*_p05``, ``*_p95``).
    ``station_delays.csv``
        ``sta, alias, lat, lon, elevation, ptcor, stcor`` (+ ``nobs``) from the fixed-model run.
    ``final_hypocentres.parquet``
        Relocated hypocentres of the fixed-model run with the Stage 1 ``event_id``; ``time`` is the
        relocated origin time (``time0``: catalogue time).
    ``summary.json``
        Headline numbers and QC findings.

    Other files: the ensemble tables (``runs.parquet``, ``final_models.parquet``,
    ``station_delays_all.parquet``, ``hypocentres.parquet``, ``station_delays_ensemble.csv``),
    ``stability/`` (tables) and ``stability_runs/`` (run directories), ``final/iterate/``
    (``convergence.csv``), ``vpvs_grid.csv`` and ``vpvs_grid/``.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from osv.data.stage import load_velest_control
from osv.io.velest import ModelBlock, VelestModel, read_stations
from osv.report.checks import QCFinding
from osv.stages import Stage, StageContext, register
from osv.velest.ensemble import (
    EnsembleResult,
    Finding,
    analyse_ensemble,
    with_final_model,
    write_ensemble,
)
from osv.velest.models import (
    TrialModel,
    generate_trial_models,
    models_to_frame,
)
from osv.velest.plots import make_all_figures, to_qc_findings
from osv.velest.runner import STATUS_OK
from osv.velest.stability import (
    FixedModelResult,
    StabilityResult,
    fixed_model_run,
    run_stability,
    write_stability,
)
from osv.velest.vpvs import vpvs_grid_search
from osv.velest.workflow import (
    iterate_final_model,
    resolve_vdamp,
    resolve_vpvs,
    run_mc_workflow,
)

#: Directory (in the ``velest_mc`` stage) holding one directory per VELEST run.
RUNS_DIR = "runs"
#: Run-table columns holding paths (stored relative to the stage directory).
_PATH_COLUMNS = ("run_dir", "out_path", "cnv_path", "sta_path", "mod_path")
_MC_SECTION = "Monte Carlo VELEST"


# ----------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------


def _inputs(prep_dir: Path, control: Any) -> SimpleNamespace:
    return SimpleNamespace(
        cnv_path=prep_dir / "velest" / "events.cnv",
        sta_path=prep_dir / "velest" / "stations.sta",
        control=control,
    )


def _control_with_ratio(cfg: Any, prep_dir: Path) -> tuple[Any, float | None]:
    """Prefilled control from Stage 1 with the resolved Vp/Vs, and that ratio.

    The ratio is ``None`` for ``nsp = 2`` (independent S model); in ``sampled`` mode the
    control keeps the middle of the range and the returned ratio is the same value.
    """
    ctrl = load_velest_control(prep_dir)
    if ctrl.nsp != 3:
        return ctrl, None
    if cfg.velest.vpvs.mode == "fixed":
        ctrl.vpvs = round(resolve_vpvs(cfg, prep_dir), 3)
    return ctrl, float(ctrl.vpvs)


def _stage_vdamp(cfg: Any, prep_dir: Path) -> np.ndarray:
    """Per-layer ``vdamp`` for the config (``"auto"`` uses the station and event depths)."""
    v = cfg.velest
    if not isinstance(v.vdamp, str):
        return resolve_vdamp(v)
    sta = read_stations(prep_dir / "velest" / "stations.sta")
    em = pd.read_csv(prep_dir / "velest" / "event_map.csv")
    return resolve_vdamp(
        v,
        station_z_km=(-sta["elev_m"].to_numpy(float) / 1000.0),
        event_depth_km=em["depth_km"].to_numpy(float),
    )


def _runs_to_parquet(runs: pd.DataFrame, stage_dir: Path) -> pd.DataFrame:
    df = runs.drop(columns=["parsed"], errors="ignore").copy()
    prefix = str(stage_dir) + "/"
    for c in _PATH_COLUMNS:
        if c in df.columns:
            df[c] = [
                v[len(prefix) :] if isinstance(v, str) and v.startswith(prefix) else v
                for v in df[c]
            ]
    if "detected_errors" in df.columns:
        df["detected_errors"] = [
            "; ".join(map(str, v)) if isinstance(v, list | tuple | np.ndarray) else v
            for v in df["detected_errors"]
        ]
    return df


def load_run_table(mc_dir: str | Path, *, parse: bool = True) -> pd.DataFrame:
    """Load ``runs.parquet`` of a ``velest_mc`` stage directory.

    Paths are made absolute again and, with ``parse``, the ``parsed`` column
    (:func:`osv.io.velest_out.read_run_dir`) and ``final_rms`` are rebuilt from the run
    directories.
    """
    from osv.io.velest_out import read_run_dir

    mc = Path(mc_dir)
    df = pd.read_parquet(mc / "runs.parquet")
    for c in _PATH_COLUMNS:
        if c in df.columns:
            df[c] = [
                str(mc / v) if isinstance(v, str) and not Path(v).is_absolute() else v
                for v in df[c]
            ]
    if parse:
        parsed = []
        for r in df.itertuples():
            obj = None
            if r.status == STATUS_OK:
                try:
                    obj = read_run_dir(r.run_dir)
                except Exception:  # keep the row, the ensemble marks it failed
                    obj = None
            parsed.append(obj)
        df["parsed"] = parsed
        df["final_rms"] = [
            float(p.final_rms) if p is not None and np.isfinite(p.final_rms) else math.nan
            for p in parsed
        ]
    return df


def _trial_models_frame(mc_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(mc_dir / "trial_models.parquet")


def _velest_model(fm: pd.DataFrame, col: str = "vp", vdamp: Any = 1.0) -> VelestModel:
    lay = pd.DataFrame({"vel": fm[col].round(2), "depth": fm["depth_top"], "vdamp": vdamp})
    return VelestModel("osv final", ModelBlock(lay, label="P final"))


def _station_table(prep_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(prep_dir / "stations.parquet")


def _final_station_delays(delays: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    st = stations.drop_duplicates("alias", keep="last")
    st = st[["alias", "lat", "lon", "elevation_m"]].rename(columns={"elevation_m": "elevation"})
    out = delays.rename(columns={"sta": "alias"}).merge(st, on="alias", how="left")
    out.insert(0, "sta", out["alias"])
    cols = ["sta", "alias", "lat", "lon", "elevation", "ptcor", "stcor"]
    cols += [c for c in out.columns if c not in cols]
    return out[cols]


def _input_hypocentres(event_map: pd.DataFrame) -> pd.DataFrame:
    """Starting hypocentres indexed like the parsed CNV events (0-based, ``cnv_id - 1``)."""
    out = event_map[["lat", "lon", "depth_km"]].copy()
    out.insert(0, "event", event_map["cnv_id"].astype(int) - 1)
    return out


def _final_hypocentres(fixed: FixedModelResult, event_map: pd.DataFrame) -> pd.DataFrame:
    h = fixed.hypocentres.copy()
    h.insert(0, "cnv_id", h["event"].astype(int) + 1)  # parsed events are 0-based
    h = h.drop(columns="event")
    if "time" not in h.columns:
        raise ValueError(
            "the fixed-model run has no relocated origin times (velout.cnv missing); Stage 3 "
            "needs them (osv-jac.13)"
        )
    # ``time`` is VELEST's relocated origin time; the catalogue time is kept as ``time0``.
    em = event_map.rename(
        columns={"lat": "lat0", "lon": "lon0", "depth_km": "depth0_km", "time": "time0"}
    )
    keep = [c for c in em.columns if c not in h.columns or c == "cnv_id"]
    out = h.merge(em[keep], on="cnv_id", how="left")
    cos = np.cos(np.radians(out["lat0"]))
    out["dlat_km"] = (out["lat"] - out["lat0"]) * 111.195
    out["dlon_km"] = (out["lon"] - out["lon0"]) * 111.195 * cos
    out["ddepth_km"] = out["depth_km"] - out["depth0_km"]
    first = ["event_id", "cnv_id", "time", "lat", "lon", "depth_km", "rms"]
    return out[
        [*[c for c in first if c in out.columns], *[c for c in out.columns if c not in first]]
    ]


def _model_table(ens: EnsembleResult) -> pd.DataFrame:
    fm = ens.final_model.rename(columns={"depth_top": "depth_top_km"})
    return fm


def load_stability(stab_dir: str | Path, fixed_run_dir: str | Path) -> StabilityResult:
    """Rebuild a :class:`StabilityResult` from :func:`write_stability` files.

    The ``fixed`` run is re-parsed from ``fixed_run_dir`` (needed by the residual figure); the
    findings are not restored.
    """
    from osv.io.velest_out import read_run_dir

    d = Path(stab_dir)
    kw: dict[str, pd.DataFrame] = {}
    for name in ("runs", "events", "models", "layers", "delays"):
        p = d / f"stability_{name}.parquet"
        if p.exists():
            kw[name] = pd.read_parquet(p)
    if (d / "stability_summary.csv").exists():
        kw["summary"] = pd.read_csv(d / "stability_summary.csv")
    fixed = None
    meta = json.loads((d / "stability.json").read_text()) if (d / "stability.json").exists() else {}
    fx = meta.get("fixed_model")
    if fx is not None and (d / "fixed_model.csv").exists():
        try:
            parsed = read_run_dir(fixed_run_dir)
        except Exception:
            parsed = None
        fixed = FixedModelResult(
            run_id=fx["run_id"],
            run_dir=Path(fixed_run_dir),
            status=fx["status"],
            parsed=parsed,
            rms=float(fx["rms"]) if fx["rms"] is not None else math.nan,
            model=pd.read_csv(d / "fixed_model.csv"),
            max_velocity_change=(
                float(fx["max_velocity_change"])
                if fx["max_velocity_change"] is not None
                else math.nan
            ),
            station_delays=pd.read_csv(d / "fixed_station_delays.csv"),
            hypocentres=pd.read_csv(d / "fixed_hypocentres.csv"),
        )
    return StabilityResult(fixed=fixed, **kw)


def _fmt_model(fm: pd.DataFrame) -> str:
    return "; ".join(
        f"{r.depth_top:.2f} km: {r.vp:.2f} ({r.vp_p05:.2f}-{r.vp_p95:.2f})" for r in fm.itertuples()
    )


# ----------------------------------------------------------------------------------------
# Stage: velest_mc
# ----------------------------------------------------------------------------------------


@register
class VelestMCStage(Stage):
    """Stage 2a: Monte Carlo VELEST runs."""

    name = "velest_mc"
    requires = ("prep",)
    outputs = ("trial_models.parquet", "runs.parquet", "workflow_history.csv", "mc_summary.json")
    tools = ("velest",)

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        t0 = time.monotonic()
        prep = ctx.upstream("prep")
        ctrl, ratio = _control_with_ratio(cfg, prep.dir)
        vcfg = cfg.velest.model_copy(deep=True)
        if ratio is not None and vcfg.vpvs.mode == "fixed":
            vcfg.vpvs.value = ratio
        seed = cfg.project.random_seed
        models = generate_trial_models(vcfg, seed)
        ctx.logger.info("running %d trial models (n_jobs=%d)", len(models), ctx.n_jobs)
        models_to_frame(models).to_parquet(ctx.path("trial_models.parquet"), index=False)

        vdamp = _stage_vdamp(cfg, prep.dir)
        ctx.logger.info("layer vdamp: %s", [round(float(x), 3) for x in vdamp])
        root = ctx.stage_dir / RUNS_DIR
        shutil.rmtree(root, ignore_errors=True)  # a stage re-run starts from scratch
        res = run_mc_workflow(
            models,
            _inputs(prep.dir, ctrl),
            cfg,
            root,
            n_jobs=ctx.n_jobs,
            seed=seed,
            compact=cfg.velest.compact_runs,
            vdamp=vdamp,
        )
        runs = _runs_to_parquet(res.runs, ctx.stage_dir)
        runs.to_parquet(ctx.path("runs.parquet"), index=False)
        res.history.to_csv(ctx.path("workflow_history.csv"), index=False)

        wall = time.monotonic() - t0
        n_ok = int((runs["status"] == STATUS_OK).sum())
        rms = runs.loc[runs["status"] == STATUS_OK, "final_rms"].astype(float)
        rms = rms[np.isfinite(rms)]
        info = {
            "n_trial_models": len(models),
            "n_runs": len(runs),
            "n_ok": n_ok,
            "n_failed": len(runs) - n_ok,
            "rms_min": float(rms.min()) if len(rms) else None,
            "rms_max": float(rms.max()) if len(rms) else None,
            "vpvs": ratio,
            "vdamp": [float(x) for x in vdamp],
            "stages": res.stages,
            "wall_time_s": round(wall, 1),
        }
        ctx.path("mc_summary.json").write_text(json.dumps(info, indent=2))
        if n_ok == 0:
            raise RuntimeError(
                f"all {len(runs)} VELEST runs failed; see {root} (status.json and *.log in each "
                "run directory)"
            )
        n_bad = len(runs) - n_ok
        if n_bad:
            ctx.add_check(
                QCFinding(
                    "mc_runs_failed",
                    "warn" if n_bad / len(runs) > 0.1 else "info",
                    f"{n_bad} of {len(runs)} VELEST runs failed or could not be parsed.",
                    n_bad,
                    {"ids": [str(i) for i in runs.loc[runs["status"] != STATUS_OK, "id"][:50]]},
                )
            )
        ctx.summary.update(
            {
                "trial_models": len(models),
                "runs_ok": n_ok,
                "runs_failed": n_bad,
                "rms_range_s": (
                    [round(info["rms_min"], 4), round(info["rms_max"], 4)] if len(rms) else None
                ),
                "damping_stages": res.stages,
                "vdamp": [round(float(x), 3) for x in vdamp],
                "vpvs": ratio,
                "wall_time_s": info["wall_time_s"],
            }
        )


# ----------------------------------------------------------------------------------------
# Stage: velest_analyse
# ----------------------------------------------------------------------------------------


def _analyse(
    ctx: StageContext, runs: pd.DataFrame, trial: pd.DataFrame, prep_dir: Path
) -> EnsembleResult:
    cfg = ctx.config.velest
    em = pd.read_csv(prep_dir / "velest" / "event_map.csv")
    inp = _input_hypocentres(em)
    summary = json.loads((prep_dir / "velest" / "velest_inputs_summary.json").read_text())
    return analyse_ensemble(
        runs,
        trial,
        cfg.select,
        vp_bounds=cfg.vp_bounds,
        input_hypocentres=inp,
        reference_station=summary.get("reference_station"),
    )


def _best_percent(cfg: Any) -> float:
    s = cfg.velest.select
    return float(s.value) if s.method == "best_percent" else 10.0


@register
class VelestAnalyseStage(Stage):
    """Stage 2b: ensemble analysis, final model, stability tests."""

    name = "velest_analyse"
    requires = ("velest_mc", "prep")
    outputs = (
        "final_model.mod",
        "final_model.csv",
        "station_delays.csv",
        "final_hypocentres.parquet",
        "summary.json",
        "runs.parquet",
    )
    tools = ("velest",)

    # ------------------------------------------------------------------ run

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        vcfg = cfg.velest
        prep = ctx.upstream("prep")
        mc = ctx.upstream("velest_mc")
        seed = cfg.project.random_seed
        ctrl, ratio = _control_with_ratio(cfg, prep.dir)
        runs = load_run_table(mc.dir)
        trial = _trial_models_frame(mc.dir)
        history = pd.read_csv(mc.dir / "workflow_history.csv")
        vd = _stage_vdamp(cfg, prep.dir)
        ens = _analyse(ctx, runs, trial, prep.dir)
        ctx.logger.info(
            "ensemble: %d runs, %d eligible, %d selected, closest %s",
            len(ens.runs),
            len(ens.eligible_ids),
            len(ens.selected_ids),
            ens.closest_run_id,
        )

        # the final model's own Vp/Vs: the fixed ratio, or the layer median for sampled ratios
        fctrl = copy.deepcopy(ctrl)
        fratio = ratio if fctrl.nsp == 3 else None
        if fratio is not None and vcfg.vpvs.mode == "sampled":
            fratio = round(float(np.nanmedian(ens.final_model["vpvs"])), 3)
            fctrl.vpvs = fratio
        inputs = _inputs(prep.dir, fctrl)

        out = ctx.stage_dir
        for sub in ("final0", "final", "stability_runs", "vpvs_grid"):
            shutil.rmtree(out / sub, ignore_errors=True)
        timeout = 3600.0
        policy_info: dict[str, Any] = {"policy": vcfg.select.final_model}
        if vcfg.select.final_model == "iterated":
            ens, policy_info = self._iterated_final_model(
                ctx, ens, inputs, fctrl, fratio, vd, out / "final0", timeout
            )
        write_ensemble(ens, out, vdamp=vd)
        (out / "station_delays.csv").replace(out / "station_delays_ensemble.csv")

        # ---- fixed-model run (canonical delays and hypocentres) and stability tests
        if vcfg.stability.enabled:
            stab = run_stability(
                ens, inputs, cfg, out / "stability_runs", seed=seed, control=fctrl,
                n_jobs=ctx.n_jobs, timeout=timeout, vdamp=vd,
            )  # fmt: skip
        else:
            fixed0 = fixed_model_run(
                ens, inputs, fctrl, out / "stability_runs" / "fixed_model",
                cfg=vcfg.stability, timeout=timeout, vdamp=vd,
            )  # fmt: skip
            if not fixed0.ok:
                raise RuntimeError(
                    f"fixed-model run failed ({fixed0.status}); see {fixed0.run_dir}"
                )
            stab = StabilityResult(fixed=fixed0)
        write_stability(stab, out / "stability")
        fixed = stab.fixed
        assert fixed is not None and fixed.ok

        # ---- repeat VELEST from its own outputs until stable (drift check)
        extra: list[Finding] = []
        iter_hist = pd.DataFrame()
        iter_info: dict[str, Any] | None = None
        if vcfg.iterate_until.max_cycles > 0:
            fm = ens.final_model
            start = TrialModel(
                "final",
                fm["depth_top"].to_numpy(float),
                fm["vp"].to_numpy(float),
                fm["vs"].to_numpy(float),
                "ensemble",
                seed,
            )
            it = iterate_final_model(
                start, inputs, cfg, out / "final", n_jobs=1,
                cnv_path=fixed.cnv_path,
                corrections=fixed.station_delays if len(fixed.station_delays) else None,
                vpvs=fratio, timeout=timeout, vdamp=vd,
            )  # fmt: skip
            iter_hist = it.history
            iter_info = {"converged": bool(it.converged), "n_cycles": len(iter_hist)}
            if it.best_cycle is not None:
                best = it[it.best_cycle - 1].final_model.p.layers["vel"].to_numpy(float)
                drift = float(np.max(np.abs(best - fm["vp"].to_numpy(float))))
                iter_info.update(best_cycle=it.best_cycle, max_vp_drift_km_s=round(drift, 3))
                if drift > vcfg.iterate_until.drift_warn_km_s:
                    extra.append(
                        Finding(
                            "iterate_model_drift",
                            "warn",
                            f"the final model is not a fixed point of VELEST: repeating the "
                            f"inversion (delays held) moved a layer's Vp by {drift:.2f} km/s "
                            f"(limit {vcfg.iterate_until.drift_warn_km_s:g}); use "
                            "velest.select.final_model: iterated, damp the poorly resolved "
                            "layers (velest.vdamp) or reduce the number of layers.",
                            data={"max_vp_drift_km_s": drift},
                        )
                    )
            if not it.converged:
                extra.append(
                    Finding(
                        "iterate_not_converged",
                        "info",
                        f"the repeat-until-stable loop did not converge in {len(iter_hist)} "
                        "cycles (model and hypocentre changes above the limits).",
                        data=iter_info,
                    )
                )

        # ---- optional Vp/Vs grid search
        grid: pd.DataFrame | None = None
        if vcfg.vpvs_grid.enabled and fctrl.nsp == 3:
            grid = vpvs_grid_search(
                _velest_model(ens.final_model, vdamp=vd), inputs, out / "vpvs_grid",
                grid=vcfg.vpvs_grid.values, n_jobs=ctx.n_jobs, timeout=timeout,
            )  # fmt: skip
            grid.to_csv(out / "vpvs_grid.csv", index=False)
            (out / "vpvs_grid.json").write_text(json.dumps(dict(grid.attrs), indent=2))
            extra.append(
                Finding(
                    "vpvs_grid_best",
                    "info",
                    f"Vp/Vs grid search minimum {grid.attrs['best_vpvs']:.3f} "
                    f"(Wadati/used ratio {fratio:.3f}).",
                    data=dict(grid.attrs),
                )
            )

        # ---- hand-off files
        stations = _station_table(prep.dir)
        fmod = _model_table(ens)
        fmod.to_csv(out / "final_model.csv", index=False)
        _final_station_delays(fixed.station_delays, stations).to_csv(
            out / "station_delays.csv", index=False
        )
        em = pd.read_csv(prep.dir / "velest" / "event_map.csv")
        hyp = _final_hypocentres(fixed, em)
        hyp.to_parquet(out / "final_hypocentres.parquet", index=False)

        findings = [*ens.findings, *stab.findings, *extra]
        summary = json.loads((out / "summary.json").read_text())
        summary.pop("next_step", None)
        summary.update(
            vpvs=fratio,
            final_model_rms=_f(fixed.rms),
            final_model_policy=policy_info,
            vdamp=[float(x) for x in vd],
            n_hypocentres=len(hyp),
            iterate=iter_info,
            stability=stab.to_json(),
            vpvs_grid=None if grid is None else dict(grid.attrs),
            findings=[
                {"code": f.code, "severity": f.severity, "message": f.message, "layer": f.layer}
                for f in findings
            ],
        )
        (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
        if len(iter_hist):
            iter_hist.to_csv(out / "iterate_convergence.csv", index=False)

        for q in to_qc_findings(findings):
            ctx.add_check(q)
        self._headline(ctx, ens, stab, fixed, fratio, iter_info, grid)
        ctx._velest_state = SimpleNamespace(  # type: ignore[attr-defined]
            ens=ens, runs=runs, history=history, stab=stab, grid=grid
        )
        self.plot(ctx)

    @staticmethod
    def _iterated_final_model(
        ctx: StageContext,
        ens: EnsembleResult,
        inputs: SimpleNamespace,
        fctrl: Any,
        fratio: float | None,
        vd: np.ndarray,
        root: Path,
        timeout: float,
    ) -> tuple[EnsembleResult, dict[str, Any]]:
        """``select.final_model = iterated``: iterate the statistic model to a fixed point.

        A fixed-model run gives delays and hypocentres consistent with the statistic model
        (:func:`~osv.velest.stability.fixed_model_run`), then VELEST is repeated from its own
        outputs with the delays held (``iterate_until``).  The last cycle's model is used when
        the loop converged, else the lowest-RMS cycle's.  The ensemble's p05-p95 uncertainty is
        kept.
        """
        cfg = ctx.config
        fm = ens.final_model
        fixed0 = fixed_model_run(
            ens, inputs, fctrl, root / "fixed", cfg=cfg.velest.stability, timeout=timeout,
            vdamp=vd,
        )  # fmt: skip
        if not fixed0.ok:
            raise RuntimeError(f"fixed-model run failed ({fixed0.status}); see {fixed0.run_dir}")
        start = TrialModel(
            "median", fm["depth_top"].to_numpy(float), fm["vp"].to_numpy(float),
            fm["vs"].to_numpy(float), "ensemble", cfg.project.random_seed,
        )  # fmt: skip
        it = iterate_final_model(
            start, inputs, cfg, root, n_jobs=1, cnv_path=fixed0.cnv_path,
            corrections=fixed0.station_delays if len(fixed0.station_delays) else None,
            vpvs=fratio, timeout=timeout, vdamp=vd,
        )  # fmt: skip
        info: dict[str, Any] = {
            "policy": "iterated",
            "converged": bool(it.converged),
            "n_cycles": len(it.history),
        }
        if not len(it):
            ctx.logger.warning("iterate to a fixed point failed; keeping the statistic model")
            return ens, {**info, "used": "median"}
        k = len(it) if it.converged else (it.best_cycle or len(it))
        model = it[k - 1].final_model
        vp = model.p.layers["vel"].to_numpy(float)
        vs = model.s.layers["vel"].to_numpy(float) if model.s is not None else None
        shift = float(np.max(np.abs(vp - fm["vp"].to_numpy(float))))
        info.update(used_cycle=k, max_vp_shift_from_median_km_s=round(shift, 3))
        ctx.logger.info("iterated final model: cycle %d, moved %.2f km/s from the median", k, shift)
        return with_final_model(ens, vp, vs), info

    @staticmethod
    def _headline(
        ctx: StageContext,
        ens: EnsembleResult,
        stab: StabilityResult,
        fixed: FixedModelResult,
        fratio: float | None,
        iter_info: dict[str, Any] | None,
        grid: pd.DataFrame | None,
    ) -> None:
        s = ctx.summary
        s["final_model_vp"] = _fmt_model(ens.final_model)
        s["final_model_policy"] = ens.final_model_policy
        s["top_layer_vp_km_s"] = round(float(ens.final_model["vp"].iloc[0]), 3)
        s["runs_diverged"] = f"{int(ens.runs['diverged'].sum())} of {len(ens.runs)} runs"
        s["vpvs"] = fratio
        s["final_model_rms_s"] = round(float(fixed.rms), 4)
        s["ensemble_rms_range_s"] = [round(float(v), 4) for v in ens.rms_range]
        s["convergence_rate"] = round(float(ens.convergence_rate), 3)
        s["runs_selected"] = f"{len(ens.selected_ids)} of {len(ens.eligible_ids)} eligible"
        s["closest_run"] = f"{ens.closest_run_id} (rms {ens.closest_rms:.4f})"
        s["hypocentres"] = int(len(fixed.hypocentres))
        s["station_delays"] = int(len(fixed.station_delays))
        for label, test, variant, metric in (
            ("mislocation_fixed_h_km", "hypocentre_perturbation", "fixed", "dh_mean_km"),
            ("mislocation_free_h_km", "hypocentre_perturbation", "free", "dh_mean_km"),
            ("mislocation_fixed_z_km", "hypocentre_perturbation", "fixed", "dz_mean_abs_km"),
            ("vp_spread_free_km_s", "hypocentre_perturbation", "free", "vp_spread_max"),
            ("reference_swap_vp_dev_km_s", "reference_station", "all", "vp_dev_max"),
        ):
            v = stab.metric(test, variant, metric)
            if np.isfinite(v):
                s[label] = round(v, 3)
        if iter_info is not None:
            s["iterate"] = iter_info
        if grid is not None:
            s["vpvs_grid_best"] = grid.attrs.get("best_vpvs")

    # ------------------------------------------------------------------ plot

    def plot(self, ctx: StageContext) -> None:
        prep = ctx.upstream("prep")
        state = getattr(ctx, "_velest_state", None)
        if state is None:
            mc = ctx.upstream("velest_mc")
            runs = load_run_table(mc.dir)
            history = pd.read_csv(mc.dir / "workflow_history.csv")
            ens = _analyse(ctx, runs, _trial_models_frame(mc.dir), prep.dir)
            fm_csv = ctx.stage_dir / "final_model.csv"
            if ens.final_model_policy == "iterated" and fm_csv.exists():
                saved = pd.read_csv(fm_csv)  # the iterated model written by run()
                ens = with_final_model(ens, saved["vp"].to_numpy(), saved["vs"].to_numpy())
            stab = load_stability(
                ctx.stage_dir / "stability",
                ctx.stage_dir / "stability_runs" / "fixed_model" / "fixed",
            )
            grid_p = ctx.stage_dir / "vpvs_grid.csv"
            grid = pd.read_csv(grid_p) if grid_p.exists() else None
        else:
            ens, runs, history, stab, grid = (
                state.ens, state.runs, state.history, state.stab, state.grid,
            )  # fmt: skip
        wadati = json.loads((prep.dir / "wadati.json").read_text())
        em = pd.read_csv(prep.dir / "velest" / "event_map.csv")
        make_all_figures(
            ctx.figures.save,
            ensemble=ens,
            workflow=SimpleNamespace(runs=runs, history=history),
            stability=stab,
            vpvs_grid=grid,
            wadati=wadati or None,
            stations=_station_table(prep.dir),
            input_hypocentres=_input_hypocentres(em),
            vp_bounds=ctx.config.velest.vp_bounds,
            best_percent=_best_percent(ctx.config),
            section=_MC_SECTION,
        )


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean(obj: Any) -> Any:
    """Make ``obj`` strictly JSON-serialisable (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | np.ndarray):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if math.isfinite(obj) else None
    return obj
