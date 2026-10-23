"""Damping/smoothing trade-off sweeps and L-curve corner picking (Stage 4).

:func:`run_tradeoff` runs short FMTOMO inversions for a set of ``(damping, smoothing)`` pairs
(``fmtomo.tradeoff``), records the final data misfit and the model norms of each, and picks the
preferred values at the corner (point of maximum curvature) of the L-curve in log-log space.

* Damping L-curve: data variance (y) against the mean squared model perturbation (x).
* Smoothing L-curve: data variance (y) against the roughness of the perturbation (x).

Modes (``fmtomo.tradeoff.mode``): ``sequential`` sweeps the damping at the configured
smoothing, picks the damping, then sweeps the smoothing at the picked damping; ``full_grid``
runs every combination and takes the two L-curves as slices of the grid (damping at the grid
smoothing nearest the configured one, smoothing at the grid damping nearest the picked one).

Every run lives in ``workdir/d<damping>_s<smoothing>/`` with a ``summary.json`` that marks it
complete, so an interrupted sweep resumes where it stopped.  Heavy outputs (``frechet.dat``,
``rays.dat``, travel-time fields) are deleted once the metrics are extracted.  Results are
written to ``workdir/tradeoff.json`` (table, picks, ``chosen_damping``, ``chosen_smoothing``).
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.fmtomo import forward as _fwd
from osv.fmtomo import invert as _inv
from osv.fmtomo.controls import ModelInfo, build_controls
from osv.fmtomo.metrics import attach_metrics
from osv.fmtomo.rundir import RunInputs, assemble_run_dir

__all__ = [
    "CornerPick",
    "TradeoffResult",
    "lcurve_corner",
    "load_chosen",
    "menger_curvature",
    "run_tradeoff",
]

log = logging.getLogger(__name__)

#: files deleted from each run directory after the metrics have been extracted
HEAVY_FILES = (
    "frechet.dat",
    "rays.dat",
    "arrtimes.dat",
    "arrivals.dat",
    "rtimes.dat",
    "mtimes.dat",
    "frechet.in",
)
#: smallest curvature (normalised axes) that counts as a corner
MIN_CURVATURE = 1e-6
#: floor applied before taking logarithms
_TINY = 1e-300
#: default fm3d jobs per concurrent inversion
MIN_JOBS_PER_RUN = 8
SUMMARY = "summary.json"
TABLE_COLUMNS = (
    "damping",
    "smoothing",
    "success",
    "n_iter",
    "rms_s",
    "variance_s2",
    "chi2_reduced",
    "model_variance",
    "roughness_phys",
    "roughness_index",
    "pert_roughness_phys",
    "pert_roughness_index",
    "flags",
    "converged",
    "wall_s",
    "dir",
)


# ---------------------------------------------------------------------------------------
# corner picking
# ---------------------------------------------------------------------------------------
@dataclass
class CornerPick:
    """Outcome of :func:`lcurve_corner`.

    ``index`` is the position (in the input order) of the corner, None if there is none;
    ``value`` the regularisation parameter there.  ``curvature`` has one entry per input point
    (NaN where undefined: end points, dominated points).  ``used`` marks the points that took
    part (not dominated); ``reason`` explains a failed pick.
    """

    index: int | None
    value: float | None
    curvature: np.ndarray
    used: np.ndarray
    reason: str = ""


def menger_curvature(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Signed Menger curvature of three 2-D points (positive for a left turn).

    ``kappa = 2 * cross(p1-p0, p2-p1) / (|p1-p0| |p2-p1| |p2-p0|)``; 0 for coincident points.
    """
    a, b = np.asarray(p1, float) - p0, np.asarray(p2, float) - p1
    c = np.asarray(p2, float) - p0
    den = float(np.hypot(*a) * np.hypot(*b) * np.hypot(*c))
    if den == 0.0:
        return 0.0
    return 2.0 * float(a[0] * b[1] - a[1] * b[0]) / den


def _pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True for points not dominated in (x, y): no other point has x and y both <= (one <)."""
    n = len(x)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if dom.any():
            keep[i] = False
    return keep


def lcurve_corner(
    param: Any,
    model_norm: Any,
    data_misfit: Any,
) -> CornerPick:
    """Corner of an L-curve: the point of maximum (convex) curvature in log-log space.

    Both axes are log10-transformed and rescaled to [0, 1] (so the curvature does not depend on
    units).  Points that are dominated (another point has both a smaller data misfit and a
    smaller model norm, i.e. violates the monotone trade-off, usually noise from too few
    iterations) are ignored.  The curvature is the Menger curvature of consecutive triplets of
    the remaining points ordered along the curve; end points have none, so at least 3 usable
    points are needed.  Only convex corners (the curve bends towards the origin) are accepted.

    Parameters
    ----------
    param
        Regularisation parameter of each point (damping or smoothing).
    model_norm
        x-axis: model variance or roughness (> 0).
    data_misfit
        y-axis: data variance (> 0).

    Returns
    -------
    CornerPick
        ``index`` None (with a ``reason``) if fewer than 3 usable points, non-finite values, or
        no convex corner.
    """
    p = np.asarray(param, float)
    x = np.asarray(model_norm, float)
    y = np.asarray(data_misfit, float)
    n = len(p)
    if not (len(x) == len(y) == n):
        raise ValueError("param, model_norm and data_misfit must have the same length")
    curv = np.full(n, np.nan)
    used = np.zeros(n, bool)
    ok = np.isfinite(p) & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return CornerPick(None, None, curv, used, f"only {int(ok.sum())} valid points (need 3)")
    lx, ly = np.log10(np.maximum(x, _TINY)), np.log10(np.maximum(y, _TINY))
    cand = np.where(ok)[0]
    keep = _pareto(lx[cand], ly[cand])
    idx = cand[keep]
    used[idx] = True
    if len(idx) < 3:
        return CornerPick(
            None, None, curv, used, f"only {len(idx)} points on the monotone trade-off (need 3)"
        )
    # Hansen's orientation: x = data misfit, y = model norm, ordered by increasing misfit, so
    # that the corner of the L bends towards the origin (positive Menger curvature).
    order = idx[np.argsort(ly[idx], kind="stable")]
    ux, uy = ly[order], lx[order]
    sx = np.ptp(ux) or 1.0
    sy = np.ptp(uy) or 1.0
    pts = np.column_stack([(ux - ux.min()) / sx, (uy - uy.min()) / sy])
    for k in range(1, len(order) - 1):
        curv[order[k]] = menger_curvature(pts[k - 1], pts[k], pts[k + 1])
    inner = order[1:-1]
    best = inner[int(np.nanargmax(curv[inner]))]
    if not curv[best] > MIN_CURVATURE:
        return CornerPick(None, None, curv, used, "no convex corner (the curve does not bend)")
    return CornerPick(int(best), float(p[best]), curv, used)


# ---------------------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------------------
@dataclass
class TradeoffResult:
    """Outcome of :func:`run_tradeoff`.

    ``table`` has one row per run (columns :data:`TABLE_COLUMNS`).  ``picks`` holds, for
    ``"damping"`` and ``"smoothing"``, a dict with the ``curve`` rows used, the corner
    (``max_curvature``: value or None, ``reason``), the ``manual`` value, the ``chosen`` value
    and ``source`` (``"max_curvature"``/``"manual"``/``"manual (fallback)"``).
    """

    workdir: Path
    table: pd.DataFrame
    chosen_damping: float
    chosen_smoothing: float
    picks: dict[str, dict[str, Any]] = field(default_factory=dict)
    mode: str = "sequential"
    pick: str = "max_curvature"
    flags: list[str] = field(default_factory=list)
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pick": self.pick,
            "chosen_damping": self.chosen_damping,
            "chosen_smoothing": self.chosen_smoothing,
            "picks": self.picks,
            "flags": self.flags,
            "table": json.loads(self.table.to_json(orient="records")),
        }


def load_chosen(path: str | Path) -> tuple[float, float]:
    """``(chosen_damping, chosen_smoothing)`` from a ``tradeoff.json`` (or its directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / "tradeoff.json"
    d = json.loads(p.read_text())
    return float(d["chosen_damping"]), float(d["chosen_smoothing"])


# ---------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{float(v):g}".replace("+", "")


def _combo_dir(workdir: Path, d: float, s: float) -> Path:
    return workdir / f"d{_fmt(d)}_s{_fmt(s)}"


def _f(x: Any) -> float:
    return math.nan if x is None else float(x)


def _read_summary(run: Path, d: float, s: float, n_iter: int) -> dict[str, Any] | None:
    try:
        sm = json.loads((run / SUMMARY).read_text())
    except (OSError, ValueError):
        return None
    if (
        sm.get("success")
        and sm.get("n_iter") == n_iter
        and math.isclose(_f(sm.get("damping")), d)
        and math.isclose(_f(sm.get("smoothing")), s)
    ):
        return sm
    return None


def _clean(run: Path) -> None:
    for name in HEAVY_FILES:
        (run / name).unlink(missing_ok=True)
    for p in [run / "parallel", *run.glob("fm3d_*"), *run.glob("group*")]:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for p in (run / "iterations").glob("*/*"):
        if p.name in HEAVY_FILES:
            p.unlink(missing_ok=True)


def _row_from_result(
    res: _inv.InversionResult, d: float, s: float, n_iter: int, run: Path
) -> dict[str, Any]:
    tab = res.iterations
    flags = list(res.flags)
    try:
        tab = _attach(res)
    except Exception as exc:  # noqa: BLE001 - metrics are best effort, keep the data misfit
        flags.append(f"metrics failed: {exc}")
    last = tab.iloc[-1]
    # the exact statistics (invert._ray_stats) replace the rounded numbers of residuals.dat
    var = _f(last.get("variance_exact_s2"))
    if not math.isfinite(var):
        var = _f(last.get("variance_s2"))
    chi2 = _f(last.get("chi2"))
    rms = (
        _f(
            last.get("rms_exact_ms")
            if math.isfinite(_f(last.get("rms_exact_ms")))
            else last.get("rms_ms")
        )
        / 1000.0
    )
    prev = tab.iloc[-2] if len(tab) > 1 else tab.iloc[-1]
    rms_prev = (
        _f(
            prev.get("rms_exact_ms")
            if math.isfinite(_f(prev.get("rms_exact_ms")))
            else prev.get("rms_ms")
        )
        / 1000.0
    )
    # converged: the last iteration changed the misfit by less than 2 %
    converged = bool(
        len(tab) > 1 and rms_prev > 0 and abs(rms - rms_prev) / rms_prev < 0.02 and res.success
    )
    return {
        "damping": d,
        "smoothing": s,
        "success": bool(res.success),
        "n_iter": n_iter,
        "rms_s": rms,
        "variance_s2": var,
        "chi2_reduced": chi2,
        "model_variance": _f(last.get("model_variance")),
        "roughness_phys": _f(last.get("roughness_phys")),
        "roughness_index": _f(last.get("roughness_index")),
        "pert_roughness_phys": _f(last.get("pert_roughness_phys")),
        "pert_roughness_index": _f(last.get("pert_roughness_index")),
        "flags": "; ".join(flags),
        "converged": converged,
        "wall_s": float(res.wall_s),
        "dir": str(run),
    }


def _attach(res: _inv.InversionResult) -> pd.DataFrame:
    return attach_metrics(res)


def _run_one(
    run: Path,
    inputs: RunInputs,
    cfg: Any,
    d: float,
    s: float,
    n_iter: int,
    n_jobs: int,
    model_info: ModelInfo | None,
    mem_per_job_gb: float | None,
    assemble_kw: dict[str, Any],
    timeout: float | None,
) -> dict[str, Any]:
    done = _read_summary(run, d, s, n_iter)
    if done is not None:
        log.info("trade-off run %s already complete, skipping", run.name)
        return done
    shutil.rmtree(run, ignore_errors=True)
    fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
    fm_run = fm.model_copy(update={"damping": float(d), "smoothing": float(s)})
    ctl = build_controls(fm_run, model_info)
    terms = assemble_kw.get("run_station_terms")
    assemble_run_dir(
        run,
        inputs.grids,
        inputs.interfaces,
        inputs.propgrid,
        inputs.sources,
        inputs.receivers,
        inputs.otimes,
        ctl,
        **{k: v for k, v in assemble_kw.items() if k != "run_station_terms"},
    )
    try:
        res = _inv.run_inversion(
            run,
            ctl,
            n_iter,
            n_jobs=n_jobs,
            keep_iterations=True,
            timeout=timeout,
            mem_per_job_gb=mem_per_job_gb,
            stop_on_error=False,
            phase=assemble_kw.get("phase"),
            station_terms=terms,
        )
        row = _row_from_result(res, d, s, n_iter, run)
    except Exception as exc:  # noqa: BLE001 - one failed combination must not kill the sweep
        log.error("trade-off run %s failed: %s", run.name, exc)
        row = {c: math.nan for c in TABLE_COLUMNS}
        row.update(
            damping=d,
            smoothing=s,
            success=False,
            n_iter=n_iter,
            flags=f"error: {exc}",
            converged=False,
            wall_s=math.nan,
            dir=str(run),
            traceback=traceback.format_exc(limit=3),
        )
    _clean(run)
    (run / SUMMARY).write_text(json.dumps(row, indent=1, default=float))
    return row


def _concurrency(cfg: Any, n_jobs: int, n_runs: int, override: int | None) -> int:
    tc = (cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg).tradeoff
    k = override or tc.max_concurrent or max(1, n_jobs // MIN_JOBS_PER_RUN)
    return max(1, min(int(k), n_runs, max(1, n_jobs)))


def _pick_one(
    name: str,
    curve: pd.DataFrame,
    xcol: str,
    manual: float,
    pick: str,
) -> dict[str, Any]:
    ok = curve[curve["success"].astype(bool)].sort_values(name)
    out: dict[str, Any] = {
        "x_axis": xcol,
        "y_axis": "variance_s2",
        "curve": json.loads(ok[[name, xcol, "variance_s2", "rms_s"]].to_json(orient="records")),
        "manual": float(manual),
    }
    cp = lcurve_corner(ok[name].to_numpy(), ok[xcol].to_numpy(), ok["variance_s2"].to_numpy())
    out["max_curvature"] = cp.value
    out["curvature"] = [None if not math.isfinite(c) else float(c) for c in cp.curvature]
    out["reason"] = cp.reason
    if pick == "manual":
        out["chosen"], out["source"] = float(manual), "manual"
    elif cp.value is not None:
        out["chosen"], out["source"] = float(cp.value), "max_curvature"
    else:
        out["chosen"], out["source"] = float(manual), "manual (fallback)"
    return out


def _nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda v: abs(math.log(max(v, _TINY)) - math.log(max(target, _TINY))))


def run_tradeoff(
    base_run_inputs: RunInputs,
    cfg: Any,
    workdir: str | Path,
    n_jobs: int = 1,
    *,
    model_info: ModelInfo | None = None,
    assemble_kwargs: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    mem_per_job_gb: float | None = None,
    timeout: float | None = None,
    on_run: Callable[[dict[str, Any]], None] | None = None,
) -> TradeoffResult:
    """Sweep damping and smoothing, pick the L-curve corners and write ``tradeoff.json``.

    Parameters
    ----------
    base_run_inputs
        Model, geometry and data (:class:`osv.fmtomo.rundir.RunInputs`) every run starts from.
    cfg
        :class:`osv.config.schema.OSVConfig` or its ``fmtomo`` section; ``tradeoff.damping``,
        ``smoothing``, ``mode``, ``pick``, ``iterations`` and ``max_concurrent`` apply, and
        ``damping``/``smoothing`` are the manual values (and the fixed smoothing of the first
        sequential sweep).
    workdir
        Output directory; each run goes in ``d<damping>_s<smoothing>/``.  Completed runs are
        reused.
    n_jobs
        Total ``fm3d`` processes; split between the concurrent inversions.
    model_info, assemble_kwargs
        Passed to :func:`build_controls` / :func:`assemble_run_dir` (``phase``, ``src_dat``,
        ...); the key ``run_station_terms`` (a :class:`osv.fmtomo.invert.StationTermSpec`) is
        not an assembler argument but goes to :func:`osv.fmtomo.invert.run_inversion`.
    max_concurrent
        Concurrent inversions (default ``tradeoff.max_concurrent`` or ``n_jobs // 8``).
    mem_per_job_gb
        Memory per ``fm3d`` job for the guard; each inversion is given ``k`` times the estimate
        so that ``k`` concurrent inversions fit together.
    on_run
        Optional callback with each finished row.

    Returns
    -------
    TradeoffResult
    """
    fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
    tc = fm.tradeoff
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    dampings = sorted({float(v) for v in tc.damping})
    smoothings = sorted({float(v) for v in tc.smoothing})
    n_iter = int(tc.iterations)
    akw = dict(assemble_kwargs or {})
    rows: dict[tuple[float, float], dict[str, Any]] = {}
    flags: list[str] = []

    def estimate() -> float | None:
        if mem_per_job_gb is not None:
            return mem_per_job_gb
        probe = wd / "_probe"
        shutil.rmtree(probe, ignore_errors=True)
        probe.mkdir()
        try:
            from osv.io.fmtomo import write_propgrid

            write_propgrid(probe / "propgrid.in", base_run_inputs.propgrid)
            return float(_fwd.estimate_job_memory_gb(probe / "propgrid.in"))
        except Exception:  # noqa: BLE001
            return None
        finally:
            shutil.rmtree(probe, ignore_errors=True)

    est_mem = estimate()

    def sweep(pairs: list[tuple[float, float]]) -> None:
        todo = [p for p in dict.fromkeys(pairs) if p not in rows]
        if not todo:
            return
        k = _concurrency(cfg, n_jobs, len(todo), max_concurrent)
        per = max(1, n_jobs // k)
        mem = None if est_mem is None else est_mem * k
        log.info("trade-off: %d runs, %d at a time x %d fm3d jobs", len(todo), k, per)

        def one(p: tuple[float, float]) -> dict[str, Any]:
            row = _run_one(
                _combo_dir(wd, *p),
                base_run_inputs,
                cfg,
                p[0],
                p[1],
                n_iter,
                per,
                model_info,
                mem,
                akw,
                timeout,
            )
            if on_run is not None:
                on_run(row)
            return row

        if k == 1:
            results = [one(p) for p in todo]
        else:
            with ThreadPoolExecutor(max_workers=k) as ex:
                results = list(ex.map(one, todo))
        for p, r in zip(todo, results, strict=True):
            rows[p] = r

    def curve(name: str, fixed: float) -> pd.DataFrame:
        col = "smoothing" if name == "damping" else "damping"
        tab = pd.DataFrame(list(rows.values()))
        return tab[np.isclose(tab[col].astype(float), fixed)]

    d0, s0 = float(fm.damping), float(fm.smoothing)
    pick = tc.pick
    if tc.mode == "full_grid":
        sweep([(d, s) for d in dampings for s in smoothings])
        s_fixed = _nearest(smoothings, s0)
        pd_ = _pick_one("damping", curve("damping", s_fixed), "model_variance", d0, pick)
        d_fixed = _nearest(dampings, pd_["chosen"])
    else:
        sweep([(d, s0) for d in dampings])
        pd_ = _pick_one("damping", curve("damping", s0), "model_variance", d0, pick)
        d_fixed = pd_["chosen"]
        sweep([(d_fixed, s) for s in smoothings])
    ps_ = _pick_one("smoothing", curve("smoothing", d_fixed), "pert_roughness_phys", s0, pick)
    pd_["at_smoothing"] = s_fixed if tc.mode == "full_grid" else s0
    ps_["at_damping"] = d_fixed
    for nm, pk in (("damping", pd_), ("smoothing", ps_)):
        if pk["source"] == "manual (fallback)":
            flags.append(f"{nm}: no L-curve corner ({pk['reason']}); using the manual value")
    table = pd.DataFrame(list(rows.values()))
    for c in TABLE_COLUMNS:
        if c not in table.columns:
            table[c] = math.nan
    table = (
        table[[*TABLE_COLUMNS, *[c for c in table.columns if c not in TABLE_COLUMNS]]]
        .sort_values(["damping", "smoothing"])
        .reset_index(drop=True)
    )
    n_bad = int((~table["success"].astype(bool)).sum())
    if n_bad:
        flags.append(f"{n_bad} trade-off run(s) failed")
    res = TradeoffResult(
        workdir=wd,
        table=table,
        chosen_damping=float(pd_["chosen"]),
        chosen_smoothing=float(ps_["chosen"]),
        picks={"damping": pd_, "smoothing": ps_},
        mode=tc.mode,
        pick=pick,
        flags=flags,
        path=wd / "tradeoff.json",
    )
    res.path.write_text(json.dumps(res.to_dict(), indent=1, default=float))
    table.to_csv(wd / "tradeoff.csv", index=False)
    log.info(
        "trade-off: chosen damping %g, smoothing %g (%s)",
        res.chosen_damping,
        res.chosen_smoothing,
        pick,
    )
    return res
