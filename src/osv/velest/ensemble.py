"""Ensemble analysis of the Monte Carlo VELEST runs and selection of the final 1-D model.

:func:`analyse_ensemble` turns the table returned by :func:`osv.velest.runner.run_jobs`
into tidy tables, selects a subset of the runs (``velest.select``), derives the final model
and its per-layer uncertainty and runs the Stage 2 QC checks.  :func:`write_ensemble` writes
the results (the interface read by Stage 3).

Selection
---------
Only runs that finished, did not fail and did not diverge (finite RMS) are eligible.
``best_percent`` keeps the best ``value`` percent (at least one run), ``best_n`` the best
``value`` runs and ``rms_below`` all runs with final RMS below ``value``.  The final model
is the per-layer ``median`` or ``mean`` of the selected final models (``best``: the single
lowest-RMS model).  The uncertainty columns are the standard deviation and the 5th/95th
percentiles over the selected set.

``select.final_model`` chooses what the final model is: ``median`` (the statistic above),
``best`` (the lowest-RMS run), ``closest`` (the selected run nearest the statistic model: a
model VELEST actually produced) or ``iterated`` (the statistic model, replaced by the stage
with its fixed point, see :func:`with_final_model`).  The uncertainty columns always describe
the selected ensemble.

A per-layer median is not itself a tested model.  ``closest_run_id`` names the selected run
whose final Vp model is nearest (L2) to the final model, with its RMS; **re-running VELEST with
the final model fixed** (task 3.5) gives station delays and hypocentres consistent with it.
Until then the station delays written are those of the closest run.

Findings
--------
Each QC check appends a :class:`Finding` (``severity`` ``info``/``warn``/``error``, a stable
``code``).  Thresholds live in :class:`QCThresholds`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.config.schema import VelestSelect
from osv.io.velest import ModelBlock, VelestModel, write_model
from osv.velest.models import TrialModel, models_to_frame

__all__ = [
    "EnsembleError",
    "EnsembleResult",
    "Finding",
    "QCThresholds",
    "analyse_ensemble",
    "final_model_checks",
    "with_final_model",
    "write_ensemble",
]

logger = logging.getLogger(__name__)


class EnsembleError(ValueError):
    """The ensemble cannot be analysed (no usable runs, inconsistent layers, ...)."""


@dataclass
class QCThresholds:
    """Thresholds of the QC checks (defaults suit the 8-layer example)."""

    min_convergence_rate: float = 0.8
    bound_fraction: float = 0.02  # of the search range, distance from lo/hi
    bound_cluster_share: float = 0.5  # share of selected runs at a bound to warn
    unresolved_dv: float = 0.02  # km/s, |median dv|
    unresolved_spread_ratio: float = 0.8  # std(final) / std(initial)
    min_hits: int = 100  # ray hits per layer
    lvz_warn_dv: float = 0.2  # km/s decrease with depth that is a warning
    delay_spread_warn: float = 0.2  # s, std of a station delay over the selected runs
    reference_delay_tol: float = 0.005  # s (files hold two decimals)
    max_rms_ratio: float = 0.9  # median final/initial RMS above this warns
    vp_floor_km_s: float = 1.5  # physical floor of a rock Vp (km/s)
    out_of_bounds_fraction: float = 0.1  # of the sampled range, beyond lo/hi
    max_divergence_rate: float = 0.2  # share of diverged runs above this warns


@dataclass
class Finding:
    """One QC result."""

    code: str
    severity: str  # "info" | "warn" | "error"
    message: str
    layer: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class EnsembleResult:
    """Result of :func:`analyse_ensemble`.

    Attributes
    ----------
    runs, final_models, station_delays, hypocentres
        Tidy tables of all runs (see :func:`analyse_ensemble`).
    final_model
        One row per layer: ``layer, depth_top, vp, vs, vpvs`` and ``<v>_std, <v>_p05,
        <v>_p95`` for ``vp`` and ``vs``, ``vp_initial_std`` and ``hits``.
    selected_ids, eligible_ids
        Run ids of the selected set and of all eligible runs (best first).
    closest_run_id, closest_rms
        Selected run closest to the final model (Vp, L2) and its RMS.
    best_run_id, best_rms
        Lowest-RMS run.
    station_summary
        Per station: median/mean/std of ``ptcor`` and ``stcor`` over the selected runs, and
        the values of the closest run.
    findings
        QC findings.
    """

    runs: pd.DataFrame
    final_models: pd.DataFrame
    station_delays: pd.DataFrame
    hypocentres: pd.DataFrame
    final_model: pd.DataFrame
    selected_ids: list[str]
    eligible_ids: list[str]
    closest_run_id: str
    closest_rms: float
    best_run_id: str
    best_rms: float
    station_summary: pd.DataFrame
    select: VelestSelect
    findings: list[Finding]
    convergence_rate: float
    rms_range: tuple[float, float]
    reference_station: str | None = None
    final_model_policy: str = "median"
    vp_bounds: np.ndarray | None = None

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "info"]

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary (``summary.json``)."""
        return _jsonable(
            {
                "n_runs": len(self.runs),
                "n_eligible": len(self.eligible_ids),
                "n_selected": len(self.selected_ids),
                "select": self.select.model_dump(),
                "final_model_policy": self.final_model_policy,
                "convergence_rate": self.convergence_rate,
                "n_failed": int(self.runs["failed"].sum() + (self.runs["status"] != "ok").sum()),
                "rms_range": list(self.rms_range),
                "best_run_id": self.best_run_id,
                "best_rms": self.best_rms,
                "closest_run_id": self.closest_run_id,
                "closest_rms": self.closest_rms,
                "reference_station": self.reference_station,
                "selected_ids": self.selected_ids,
                "final_model": self.final_model.to_dict(orient="list"),
                "findings": [
                    {"code": f.code, "severity": f.severity, "message": f.message, "layer": f.layer}
                    for f in self.findings
                ],
                "next_step": (
                    "re-run VELEST with the final model fixed (task 3.5) for station delays and "
                    "hypocentres consistent with it; until then the closest run's are used"
                ),
            }
        )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ----------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------


def _parsed_of(row: Any) -> Any:
    obj = getattr(row, "parsed", None)
    if obj is not None:
        return obj
    if getattr(row, "status", None) != "ok":
        return None
    from osv.io.velest_out import read_run_dir

    try:
        return read_run_dir(row.run_dir)
    except Exception as exc:
        logger.warning("could not parse %s: %s", row.run_dir, exc)
        return None


def _model_rows(rid: str, parsed: Any, trial: pd.DataFrame | None) -> list[pd.DataFrame]:
    out = []
    fm, im = parsed.final_model, parsed.initial_model
    hits = None
    rs = parsed.ray_statistics
    if rs is not None and len(rs):
        hits = pd.Series(rs["nhit"].to_numpy(dtype=float), index=rs["nlay"].astype(int) - 1)
    ctrl = getattr(parsed, "control", None) or {}
    ratio = float(ctrl.get("vpvs", math.nan)) if ctrl.get("nsp") == 3 else math.nan
    for phase in ("P", "S"):
        fb = fm.p if phase == "P" else fm.s
        derived = False
        if fb is None and phase == "S" and np.isfinite(ratio) and ratio > 0:
            # nsp = 3 (fixed/sampled Vp/Vs): no S block was inverted, Vs = Vp / vpvs
            # (velest.f:4357) with this run's ratio.
            fb = fm.p
            derived = True
        if fb is None:
            continue
        ib = im.p if (im is not None and phase == "P") else (im.s if im is not None else None)
        if derived and im is not None:
            ib = im.p
        lay = fb.layers
        n = len(lay)
        if ib is not None and len(ib.layers) == n:
            v0 = ib.layers["vel"].to_numpy(dtype=float)
            if derived:
                v0 = v0 / ratio
        elif trial is not None and phase in ("P", "S") and rid in trial.index.get_level_values(0):
            t = trial.loc[rid]
            v0 = t["vp" if phase == "P" else "vs"].to_numpy(dtype=float)
        else:
            v0 = np.full(n, np.nan)
        v1 = lay["vel"].to_numpy(dtype=float)
        if derived:
            v1 = v1 / ratio
        out.append(
            pd.DataFrame(
                {
                    "id": rid,
                    "phase": phase,
                    "layer": np.arange(n),
                    "depth_top": lay["depth"].to_numpy(dtype=float),
                    "v_initial": v0,
                    "v_final": v1,
                    "dv": v1 - v0,
                    "hits": (
                        hits.reindex(range(n)).to_numpy()
                        if hits is not None
                        else np.full(n, np.nan)
                    ),
                }
            )
        )
    return out


def _delay_rows(rid: str, parsed: Any) -> pd.DataFrame | None:
    sc = parsed.station_corrections
    if sc is None or not len(sc):
        return None
    df = sc[["sta", "ptcor", "stcor", "nobs"]].copy()
    df.insert(0, "id", rid)
    return df


def _hypo_rows(rid: str, parsed: Any, inp: pd.DataFrame | None) -> pd.DataFrame | None:
    if parsed.cnv is not None and len(parsed.cnv.events):
        ev = parsed.cnv.events
        df = pd.DataFrame(
            {
                "event": ev["event"].to_numpy(),
                "lat": ev["lat"].to_numpy(dtype=float),
                "lon": ev["lon"].to_numpy(dtype=float),
                "depth": ev["depth_km"].to_numpy(dtype=float),
                "rms": ev["rms"].to_numpy(dtype=float) if "rms" in ev else np.nan,
            }
        )
    elif len(parsed.final_hypocentres):
        fh = parsed.final_hypocentres
        df = pd.DataFrame(
            {
                "event": fh["eq"].to_numpy() - 1,  # 0-based like the CNV event index
                "lat": fh["lat"].to_numpy(dtype=float),
                "lon": fh["lon"].to_numpy(dtype=float),
                "depth": fh["depth_km"].to_numpy(dtype=float),
                "rms": fh["rms"].to_numpy(dtype=float),
            }
        )
    else:
        return None
    df.insert(0, "id", rid)
    if inp is not None:
        m = df.merge(
            inp.rename(columns={"lat": "lat0", "lon": "lon0", "depth_km": "depth0"})[
                ["event", "lat0", "lon0", "depth0"]
            ],
            on="event",
            how="left",
        )
        coslat = np.cos(np.radians(m["lat0"]))
        df["dlat_km"] = ((m["lat"] - m["lat0"]) * 111.195).to_numpy()
        df["dlon_km"] = ((m["lon"] - m["lon0"]) * 111.195 * coslat).to_numpy()
        df["ddepth_km"] = (m["depth"] - m["depth0"]).to_numpy()
        df["dh_km"] = np.hypot(df["dlat_km"], df["dlon_km"])
    return df


def _trial_frame(trial_models: Any) -> pd.DataFrame | None:
    if trial_models is None:
        return None
    if isinstance(trial_models, pd.DataFrame):
        frame = trial_models
    else:
        tm = list(trial_models)
        if tm and not isinstance(tm[0], TrialModel):
            raise EnsembleError(
                "trial_models must be TrialModel objects or a models_to_frame table"
            )
        frame = models_to_frame(tm)
    if frame.empty:
        return None
    return frame.set_index(["id", "layer"]).sort_index().rename_axis(index=["id", "layer"])


def _parse_all(
    run_table: pd.DataFrame, trial: pd.DataFrame | None, inp: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    fms: list[pd.DataFrame] = []
    dls: list[pd.DataFrame] = []
    hys: list[pd.DataFrame] = []
    for row in run_table.itertuples():
        rid = str(row.id)
        parsed = _parsed_of(row)
        status = str(getattr(row, "status", "ok"))
        rec: dict[str, Any] = {
            "id": rid,
            "status": status,
            "converged": False,
            "diverged": False,
            "failed": parsed is None,
            "finished": False,
            "n_iter": 0,
            "n_backups": 0,
            "initial_rms": math.nan,
            "final_rms": math.nan,
            "stop_reason": None,
            "wall_time": getattr(row, "wall_time", math.nan),
            "has_model": False,
        }
        if parsed is not None:
            rec.update(
                converged=bool(parsed.converged),
                diverged=bool(parsed.diverged),
                failed=bool(parsed.failed),
                finished=bool(parsed.finished),
                n_iter=int(parsed.n_iterations),
                n_backups=int(parsed.n_backups),
                initial_rms=float(parsed.initial_rms),
                final_rms=float(parsed.final_rms),
                stop_reason=parsed.stop_reason,
            )
            if parsed.final_model is not None:
                rec["has_model"] = True
                fms.extend(_model_rows(rid, parsed, trial))
            for lst, part in (
                (dls, _delay_rows(rid, parsed)),
                (hys, _hypo_rows(rid, parsed, inp)),
            ):
                if part is not None:
                    lst.append(part)
        runs.append(rec)
    cat = lambda parts: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()  # noqa: E731
    return pd.DataFrame(runs), cat(fms), cat(dls), cat(hys)


# ----------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------


def _select_ids(eligible: pd.DataFrame, cfg: VelestSelect) -> list[str]:
    """Selected ids, best (lowest RMS) first; ``eligible`` is sorted by ``final_rms``."""
    n = len(eligible)
    if cfg.method == "best_percent":
        k = max(1, math.ceil(cfg.value / 100.0 * n - 1e-9))
        sel = eligible.head(k)
    elif cfg.method == "best_n":
        sel = eligible.head(max(1, int(cfg.value)))
    elif cfg.method == "rms_below":
        sel = eligible[eligible["final_rms"] < cfg.value]
        if sel.empty:
            raise EnsembleError(
                f"no eligible run has final RMS below {cfg.value} s (lowest is "
                f"{eligible['final_rms'].iloc[0]:.4f} s); raise select.value"
            )
    else:  # pragma: no cover - validated by the schema
        raise EnsembleError(f"unknown selection method {cfg.method!r}")
    return [str(i) for i in sel["id"]]


def _matrix(fm: pd.DataFrame, ids: list[str], phase: str, col: str) -> np.ndarray:
    sub = fm[(fm["phase"] == phase) & (fm["id"].isin(ids))]
    wide = sub.pivot(index="id", columns="layer", values=col).reindex(ids)
    return wide.to_numpy(dtype=float)


def _stat(arr: np.ndarray, how: str, best_row: int = 0) -> np.ndarray:
    if how == "median":
        return np.nanmedian(arr, axis=0)
    if how == "mean":
        return np.nanmean(arr, axis=0)
    return arr[best_row]


# ----------------------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------------------


def analyse_ensemble(
    run_table: pd.DataFrame,
    trial_models: Sequence[TrialModel] | pd.DataFrame | None,
    select_cfg: VelestSelect,
    *,
    vp_bounds: Sequence[Sequence[float]] | None = None,
    input_hypocentres: pd.DataFrame | None = None,
    reference_station: str | None = None,
    thresholds: QCThresholds | None = None,
) -> EnsembleResult:
    """Analyse a batch of VELEST runs and pick the final 1-D model.

    Parameters
    ----------
    run_table
        Output of :func:`osv.velest.runner.run_jobs` (``id, status, run_dir, wall_time`` and
        optionally ``parsed``, a :class:`~osv.io.velest_out.VelestRunResult`; runs without
        ``parsed`` are read from ``run_dir``).
    trial_models
        The trial models (list of :class:`TrialModel` or a :func:`models_to_frame` table);
        used for initial velocities missing from the output and for the search range.
    select_cfg
        ``velest.select`` of the config.
    vp_bounds
        ``[lo, hi]`` per layer of the Vp search range for the bound check (default:
        min/max of the trial models).
    input_hypocentres
        Optional starting hypocentres (``event, lat, lon, depth_km``, event index as in the CNV
        events) to compute shifts (``dlat_km, dlon_km, ddepth_km, dh_km``).
    reference_station
        Name of the reference station (its delay must stay 0); inferred as a station with a
        zero delay in all selected runs when omitted.
    thresholds
        QC thresholds.

    Returns
    -------
    EnsembleResult
        Tables ``runs`` (``id, status, converged, diverged, failed, finished, n_iter,
        n_backups, initial_rms, final_rms, stop_reason, wall_time, eligible, selected,
        rank``), ``final_models`` (long: ``id, phase, layer, depth_top, v_initial, v_final,
        dv, hits``), ``station_delays`` (``id, sta, ptcor, stcor, nobs``), ``hypocentres``
        (``id, event, lat, lon, depth, rms`` [+ shifts]) and the selection, final model and
        findings.

    Raises
    ------
    EnsembleError
        With no eligible run, no run under an ``rms_below`` threshold, or runs with different
        layer counts.
    """
    th = thresholds or QCThresholds()
    trial = _trial_frame(trial_models)
    runs, fm, dl, hy = _parse_all(run_table, trial, input_hypocentres)
    if runs.empty:
        raise EnsembleError("run table is empty")

    ok = (
        (runs["status"] == "ok")
        & runs["finished"]
        & ~runs["failed"]
        & ~runs["diverged"]
        & runs["has_model"]
        & np.isfinite(runs["final_rms"])
    )
    runs["eligible"] = ok
    elig = runs[ok].sort_values(["final_rms", "id"], kind="stable")
    if elig.empty:
        raise EnsembleError(
            f"none of the {len(runs)} runs finished without failing or diverging; "
            "check the run logs (status, failure reasons)"
        )
    eligible_ids = [str(i) for i in elig["id"]]
    selected_ids = _select_ids(elig, select_cfg)
    runs["selected"] = runs["id"].isin(selected_ids)
    runs["rank"] = runs["id"].map({i: k + 1 for k, i in enumerate(eligible_ids)})

    nlay = fm[fm["id"].isin(eligible_ids)].groupby("id")["layer"].nunique()
    if nlay.nunique() != 1:
        raise EnsembleError(f"runs have different layer counts: {nlay.value_counts().to_dict()}")

    vp = _matrix(fm, selected_ids, "P", "v_final")
    vs = _matrix(fm, selected_ids, "S", "v_final")
    has_s = bool(np.isfinite(vs).any())
    how = select_cfg.statistic
    fvp = _stat(vp, how)
    fvs = _stat(vs, how) if has_s else np.full(vp.shape[1], np.nan)
    stat_vp = fvp
    policy = select_cfg.final_model
    if policy in ("best", "closest"):
        row = 0 if policy == "best" else int(np.argmin(np.sqrt(np.nansum((vp - fvp) ** 2, axis=1))))
        fvp = vp[row]
        fvs = vs[row] if has_s else fvs

    def spread(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not np.isfinite(a).any():
            nan = np.full(vp.shape[1], np.nan)  # also when there is no S block at all
            return nan, nan, nan
        sd = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])
        return sd, np.nanpercentile(a, 5, axis=0), np.nanpercentile(a, 95, axis=0)

    sd_p, p05_p, p95_p = spread(vp)
    sd_s, p05_s, p95_s = spread(vs)
    p0 = fm[(fm["phase"] == "P") & (fm["id"] == selected_ids[0])].sort_values("layer")
    all_p0 = _matrix(fm, eligible_ids, "P", "v_initial")
    hits = (
        fm[(fm["phase"] == "P") & fm["id"].isin(eligible_ids)]
        .groupby("layer")["hits"]
        .median()
        .reindex(range(vp.shape[1]))
        .to_numpy()
    )
    final_model = pd.DataFrame(
        {
            "layer": np.arange(vp.shape[1]),
            "depth_top": p0["depth_top"].to_numpy(),
            "vp": fvp,
            "vs": fvs,
            "vpvs": fvp / fvs,
            "vp_std": sd_p,
            "vp_p05": p05_p,
            "vp_p95": p95_p,
            "vs_std": sd_s,
            "vs_p05": p05_s,
            "vs_p95": p95_s,
            "vp_initial_std": np.nanstd(all_p0, axis=0, ddof=1) if len(all_p0) > 1 else np.nan,
            "hits": hits,
        }
    )

    # closest run to the statistic model (L2 in Vp); the best run is selected_ids[0]
    d = np.sqrt(np.nansum((vp - stat_vp) ** 2, axis=1))
    closest = selected_ids[int(np.argmin(d))]
    rms_of = runs.set_index("id")["final_rms"]

    sel_delays = dl[dl["id"].isin(selected_ids)] if len(dl) else dl
    station_summary = _station_summary(sel_delays, dl, closest)

    res = EnsembleResult(
        runs=runs,
        final_models=fm,
        station_delays=dl,
        hypocentres=hy,
        final_model=final_model,
        selected_ids=selected_ids,
        eligible_ids=eligible_ids,
        closest_run_id=closest,
        closest_rms=float(rms_of[closest]),
        best_run_id=eligible_ids[0],
        best_rms=float(rms_of[eligible_ids[0]]),
        station_summary=station_summary,
        select=select_cfg,
        findings=[],
        convergence_rate=float(runs["converged"].mean()),
        rms_range=(float(elig["final_rms"].min()), float(elig["final_rms"].max())),
        reference_station=reference_station,
        final_model_policy=policy,
    )
    bounds = _bounds(vp_bounds, trial, vp.shape[1])
    res.vp_bounds = bounds
    res.findings = _qc(res, fm, vp, bounds, sel_delays, runs, th)
    res.findings += final_model_checks(res, th)
    for f in res.findings:
        if f.severity != "info":
            logger.warning("ensemble QC %s: %s", f.code, f.message)
    return res


def _bounds(
    vp_bounds: Sequence[Sequence[float]] | None, trial: pd.DataFrame | None, n: int
) -> np.ndarray | None:
    if vp_bounds is not None:
        b = np.asarray(vp_bounds, dtype=float)
        if b.shape != (n, 2):
            raise EnsembleError(f"vp_bounds has shape {b.shape}, expected ({n}, 2)")
        return b
    if trial is None:
        return None
    g = trial["vp"].groupby(level="layer").agg(["min", "max"]).reindex(range(n))
    return g.to_numpy(dtype=float) if g.notna().all().all() else None


def _station_summary(sel: pd.DataFrame, allr: pd.DataFrame, closest: str) -> pd.DataFrame:
    if sel is None or sel.empty:
        return pd.DataFrame(
            columns=["sta", "ptcor_median", "ptcor_mean", "ptcor_std", "stcor_median", "nobs"]
        )
    g = sel.groupby("sta")
    out = pd.DataFrame(
        {
            "ptcor_median": g["ptcor"].median(),
            "ptcor_mean": g["ptcor"].mean(),
            "ptcor_std": g["ptcor"].std(ddof=0),
            "ptcor_p05": g["ptcor"].quantile(0.05),
            "ptcor_p95": g["ptcor"].quantile(0.95),
            "stcor_median": g["stcor"].median(),
            "stcor_std": g["stcor"].std(ddof=0),
            "nobs": g["nobs"].median(),
        }
    )
    c = allr[allr["id"] == closest].set_index("sta")
    out["ptcor_closest"] = c["ptcor"].reindex(out.index)
    out["stcor_closest"] = c["stcor"].reindex(out.index)
    return out.reset_index()


# ----------------------------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------------------------


def _lvz_findings(v: np.ndarray, th: QCThresholds) -> list[Finding]:
    out = []
    for i in range(1, len(v)):
        drop = v[i - 1] - v[i]
        if drop > 1e-9:
            out.append(
                Finding(
                    "lvz",
                    "warn" if drop > th.lvz_warn_dv else "info",
                    f"low-velocity zone: layer {i} ({v[i]:.2f} km/s) is {drop:.2f} km/s slower "
                    f"than layer {i - 1}",
                    layer=i,
                    data={"drop": float(drop)},
                )
            )
    return out


def final_model_checks(res: EnsembleResult, th: QCThresholds | None = None) -> list[Finding]:
    """QC findings about the *final model itself* (not the selected runs).

    * ``vp_below_floor`` (warn): a layer's Vp is below ``th.vp_floor_km_s`` (1.5 km/s): not a
      physical rock velocity, usually a near-surface layer trading off with the station delays.
    * ``final_out_of_bounds`` (warn): a layer's Vp is outside the sampled ``vp_bounds`` by more
      than ``th.out_of_bounds_fraction`` of the layer's range, i.e. the inversion left the
      region the trial models covered.
    * ``divergence_rate`` (warn): more than ``th.max_divergence_rate`` of the runs diverged
      (final RMS above the RMS of the run's own start).
    """
    th = th or QCThresholds()
    out: list[Finding] = []
    vp = res.final_model["vp"].to_numpy(float)
    for i in np.flatnonzero(vp < th.vp_floor_km_s):
        out.append(
            Finding(
                "vp_below_floor",
                "warn",
                f"layer {int(i)}: final Vp {vp[i]:.2f} km/s is below the physical floor "
                f"{th.vp_floor_km_s:g} km/s; the layer is not resolved against the station "
                "delays (damp it with velest.vdamp or merge it)",
                layer=int(i),
                data={"vp": float(vp[i]), "floor": th.vp_floor_km_s},
            )
        )
    b = res.vp_bounds
    if b is not None and len(b) == len(vp):
        for i, (lo, hi) in enumerate(b):
            tol = th.out_of_bounds_fraction * (hi - lo)
            if vp[i] < lo - tol or vp[i] > hi + tol:
                side, edge = ("lower", lo) if vp[i] < lo else ("upper", hi)
                out.append(
                    Finding(
                        "final_out_of_bounds",
                        "warn",
                        f"layer {i}: final Vp {vp[i]:.2f} km/s is more than "
                        f"{th.out_of_bounds_fraction:.0%} of the sampled range beyond the "
                        f"{side} bound ({edge:.2f} km/s); the inversion left the region the "
                        "trial models covered",
                        layer=i,
                        data={"vp": float(vp[i]), "side": side, "bound": float(edge)},
                    )
                )
    n = len(res.runs)
    ndiv = int(res.runs["diverged"].sum())
    rate = ndiv / n if n else 0.0
    if rate > th.max_divergence_rate:
        out.append(
            Finding(
                "divergence_rate",
                "warn",
                f"{ndiv} of {n} runs ({rate:.0%}) diverged (> {th.max_divergence_rate:.0%}): "
                "the final RMS of a run is higher than that of its start; check the delay "
                "handling (velest.stability.delay_swtfac), the damping and the near-surface "
                "layer",
                data={"rate": rate, "diverged": ndiv, "n": n},
            )
        )
    return out


def with_final_model(
    res: EnsembleResult,
    vp: Sequence[float],
    vs: Sequence[float] | None = None,
    *,
    policy: str = "iterated",
    thresholds: QCThresholds | None = None,
) -> EnsembleResult:
    """Copy of ``res`` whose final model has the velocities ``vp`` (and ``vs``).

    Used for ``select.final_model = "iterated"``: the stage iterates the statistic model with
    VELEST and substitutes the fixed point.  The uncertainty columns (std, p05, p95) stay those
    of the selected ensemble; ``vs`` defaults to ``vp / vpvs`` with the model's own ratio.  The
    low-velocity-zone and final-model findings are recomputed for the new model.
    """
    th = thresholds or QCThresholds()
    fm = res.final_model.copy()
    new_vp = np.asarray(vp, dtype=float)
    if len(new_vp) != len(fm):
        raise EnsembleError(f"final model has {len(fm)} layers but {len(new_vp)} velocities given")
    ratio = fm["vpvs"].to_numpy(float)
    new_vs = np.asarray(vs, dtype=float) if vs is not None else new_vp / ratio
    fm["vp"], fm["vs"], fm["vpvs"] = new_vp, new_vs, new_vp / new_vs
    keep = [f for f in res.findings if f.code not in {"lvz", *_FINAL_CODES}]
    out = replace(res, final_model=fm, final_model_policy=policy)
    out.findings = [*keep, *_lvz_findings(new_vp, th), *final_model_checks(out, th)]
    return out


_FINAL_CODES = ("vp_below_floor", "final_out_of_bounds", "divergence_rate")


def _qc(
    res: EnsembleResult,
    fm: pd.DataFrame,
    vp_sel: np.ndarray,
    bounds: np.ndarray | None,
    sel_delays: pd.DataFrame,
    runs: pd.DataFrame,
    th: QCThresholds,
) -> list[Finding]:
    out: list[Finding] = []
    add = lambda *a, **k: out.append(Finding(*a, **k))  # noqa: E731
    nl = vp_sel.shape[1]

    # convergence and failures
    n = len(runs)
    rate = res.convergence_rate
    add(
        "convergence_rate",
        "warn" if rate < th.min_convergence_rate else "info",
        f"{rate:.0%} of {n} runs converged"
        + (
            f" (< {th.min_convergence_rate:.0%}: raise ittmax or relax damping)"
            if rate < th.min_convergence_rate
            else ""
        ),
        data={"rate": rate},
    )
    nbad = int(((runs["status"] != "ok") | runs["failed"]).sum())
    ndiv = int(runs["diverged"].sum())
    add(
        "failures",
        "warn" if nbad or ndiv else "info",
        f"{nbad} failed and {ndiv} diverged of {n} runs",
        data={"failed": nbad, "diverged": ndiv},
    )

    # bounds
    if bounds is not None:
        for i in range(nl):
            lo, hi = bounds[i]
            tol = th.bound_fraction * (hi - lo)
            v = vp_sel[:, i]
            at_lo = float(np.mean(v <= lo + tol))
            at_hi = float(np.mean(v >= hi - tol))
            for share, side, edge in ((at_lo, "lower", lo), (at_hi, "upper", hi)):
                if share >= th.bound_cluster_share:
                    add(
                        "at_bound",
                        "warn",
                        f"layer {i}: {share:.0%} of the selected runs end within "
                        f"{th.bound_fraction:.0%} of the {side} bound ({edge:.2f} km/s); "
                        "the search range may be too narrow",
                        layer=i,
                        data={"side": side, "share": share},
                    )

    # unresolved layers / hits (all eligible runs)
    p = fm[(fm["phase"] == "P") & fm["id"].isin(res.eligible_ids)]
    for i, g in p.groupby("layer"):
        i = int(i)
        med_dv = float(np.nanmedian(g["dv"]))
        sd_f = float(np.nanstd(g["v_final"]))
        sd_i = float(np.nanstd(g["v_initial"]))
        if abs(med_dv) < th.unresolved_dv and (
            sd_i == 0 or (sd_f / sd_i if sd_i > 0 else 1.0) >= th.unresolved_spread_ratio
        ):
            add(
                "unresolved_layer",
                "warn",
                f"layer {i}: median |dv| = {abs(med_dv):.3f} km/s and the spread of the final "
                f"velocities ({sd_f:.2f}) is like that of the start ({sd_i:.2f}); the layer "
                "is probably unresolved (few or no hits)",
                layer=i,
                data={"median_dv": med_dv, "std_final": sd_f, "std_initial": sd_i},
            )
        h = float(np.nanmedian(g["hits"])) if g["hits"].notna().any() else math.nan
        if np.isfinite(h) and h < th.min_hits:
            add(
                "low_hits",
                "warn",
                f"layer {i}: median {h:.0f} ray hits (< {th.min_hits})",
                layer=i,
                data={"hits": h},
            )

    out += _lvz_findings(res.final_model["vp"].to_numpy(), th)

    # RMS improvement
    sel = runs[runs["selected"]]
    ratio = float(np.nanmedian(sel["final_rms"] / sel["initial_rms"]))
    add(
        "rms_improvement",
        "warn" if np.isfinite(ratio) and ratio > th.max_rms_ratio else "info",
        f"final/initial RMS of the selected runs: {ratio:.2f} (median); "
        f"final RMS {res.rms_range[0]:.3f}-{res.rms_range[1]:.3f} s",
        data={"ratio": ratio},
    )

    # station delays
    if len(sel_delays):
        ss = res.station_summary
        worst = ss.loc[ss["ptcor_std"].idxmax()]
        big = ss[ss["ptcor_std"] > th.delay_spread_warn]
        add(
            "delay_spread",
            "warn" if len(big) else "info",
            f"largest station-delay spread {worst['ptcor_std']:.2f} s ({worst['sta']}); "
            f"{len(big)} station(s) above {th.delay_spread_warn} s",
            data={"stations": list(big["sta"])},
        )
        ref = res.reference_station
        if ref is None:
            zero = ss[(ss["ptcor_median"].abs() < 1e-9) & (ss["ptcor_std"] < 1e-9)]
            if len(zero) == 1:
                ref = res.reference_station = str(zero["sta"].iloc[0])
        if ref is None:
            add("reference_station", "info", "reference station not given and not identifiable")
        elif ref not in set(ss["sta"]):
            add("reference_station", "warn", f"reference station {ref} not in the station table")
        else:
            r = ss[ss["sta"] == ref].iloc[0]
            bad = (
                abs(r["ptcor_median"]) > th.reference_delay_tol
                or r["ptcor_std"] > th.reference_delay_tol
            )
            add(
                "reference_station",
                "warn" if bad else "info",
                f"reference station {ref}: delay {r['ptcor_median']:+.3f} s "
                f"(std {r['ptcor_std']:.3f}); " + ("it should stay 0" if bad else "stays 0"),
                data={"station": ref},
            )
    return out


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------


def write_ensemble(
    result: EnsembleResult,
    outdir: str | Path,
    vdamp: float | Sequence[float] = 1.0,
) -> dict[str, Path]:
    """Write the ensemble results into ``outdir``.

    Files: ``final_model.mod`` (VELEST format, P and S blocks, ``vdamp`` per layer),
    ``final_model.csv`` (depth, vp, vs, vpvs and uncertainties), ``station_delays.csv``
    (closest run plus median/spread over the selected runs), ``runs.parquet``,
    ``final_models.parquet``, ``station_delays_all.parquet``, ``hypocentres.parquet`` and
    ``summary.json``.  Returns ``{name: path}``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fmod = result.final_model
    vd = np.broadcast_to(np.asarray(vdamp, dtype=float), (len(fmod),)).copy()
    blocks = []
    for ph, col in (("P", "vp"), ("S", "vs")):
        if not np.isfinite(fmod[col]).all():
            blocks.append(None)
            continue
        lay = pd.DataFrame({"vel": fmod[col].round(2), "depth": fmod["depth_top"], "vdamp": vd})
        blocks.append(ModelBlock(lay, label=f"{ph} final {result.final_model_policy}"[:40]))
    if blocks[0] is None:
        raise EnsembleError("final model has no Vp")
    paths: dict[str, Path] = {}
    paths["final_model.mod"] = write_model(
        out / "final_model.mod",
        VelestModel(f"osv ensemble {result.final_model_policy}"[:40], blocks[0], blocks[1]),
    )
    csv = fmod.rename(columns={"depth_top": "depth"})
    csv.to_csv(out / "final_model.csv", index=False)
    paths["final_model.csv"] = out / "final_model.csv"
    result.station_summary.to_csv(out / "station_delays.csv", index=False)
    paths["station_delays.csv"] = out / "station_delays.csv"
    for name, df in (
        ("runs", result.runs),
        ("final_models", result.final_models),
        ("station_delays_all", result.station_delays),
        ("hypocentres", result.hypocentres),
    ):
        df.to_parquet(out / f"{name}.parquet", index=False)
        paths[f"{name}.parquet"] = out / f"{name}.parquet"
    (out / "summary.json").write_text(json.dumps(result.summary(), indent=2))
    paths["summary.json"] = out / "summary.json"
    return paths
