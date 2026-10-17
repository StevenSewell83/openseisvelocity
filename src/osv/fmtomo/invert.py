"""FMTOMO inversion loop with parallel forward steps and per-iteration capture.

:func:`run_inversion` replaces the ``tomo3d``/``tomo3di`` ksh drivers
(``fortran/fmtomo/bin/tomo3d``, ``fortran/fmtomo_extras/source/tomo3di``)
by a Python loop that executes the *same programs in the same order* with the same file
shuffling, except that every ``fm3d`` call goes through
:func:`osv.fmtomo.forward.run_fm3d_parallel`.

Linear / no relocation (driver ``tomo3d``)::

    start (begin_from_current = 0):  vgridsref/interfacesref/sourcesref -> current files, frechgen,
                                     inviter.in = 1, fm3d, arrivals -> mtimes + rtimes, residuals
    each iteration:                  invert3d, scorrect, fm3d, arrivals -> mtimes, residuals,
                                     inviter.in = iteration + 1

Nonlinear relocation (driver ``tomo3di``; stations are the FMTOMO sources and events the
receivers, so the ``fm3d`` runs also save the time fields ``arrtimes.dat`` that ``gridloc``
searches)::

    each iteration:                  gridloc, fm3d, residuals, invert3d, fm3d, residuals

Deviations from the scripts (all deliberate):

* ``fm3d`` runs in parallel (identical merged output, see :mod:`osv.fmtomo.forward`); for the
  nonlinear geometry the time-field grids ``arrtimes.dat`` of the groups are concatenated in
  global source order (``gridsave.in`` is renumbered per group).
* ``tomo3di`` writes ``inviter.in`` with the shell variable ``COUNT`` *before* ``invert3d``; in the
  first iteration ``COUNT`` is still 0 (ksh) or empty (the bash port) which ``invert3d`` cannot
  use as an iteration number.  We write the number of the iteration that is running.
* ``scorrect`` is run exactly when the controls contain it (``tomo3d`` geometry); the
  nonlinear driver never ran it.
* The residual line is the first line printed by ``residuals`` with blanks collapsed (the
  ``read ttres; print -u5 $ttres`` of the scripts); the file is bit-identical.

Per iteration ``run_dir/iterations/NN/`` receives ``summary.json`` and the logs; with
``keep_iterations`` also ``vgrids.in``, ``sources.in`` (or ``receivers.in`` for the nonlinear
geometry, which holds the events) and ``stimes.dat``.  ``NN = 00`` is the starting state.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from osv.fmtomo import forward as _fwd
from osv.fmtomo.controls import FmtomoControls, RunControls
from osv.fmtomo.forward import Fm3dError, Fm3dResult, run_fm3d_parallel
from osv.io.fmtomo import (
    FmtomoFormatError,
    GridSave,
    ModeSet,
    Receivers,
    Source,
    read_arrivals,
    read_gridsave,
    read_observed,
    read_receivers,
    read_sources,
    read_vgrids,
    write_gridsave,
)
from osv.io.fmtomo.times import INVALID_OBSERVED_BELOW
from osv.tools import ToolError, run_tool

log = logging.getLogger(__name__)

__all__ = ["InversionError", "InversionResult", "run_inversion"]

KM_PER_DEG = 111.194926645
#: relative tolerance for "velocity sits at min_velocity"
MIN_VEL_TOL = 1e-4
#: a residual (rms) increase larger than this fraction of the previous value is flagged
RESIDUAL_INCREASE_TOL = 0.0
#: fraction of previously valid arrivals that may be lost in one step before it is flagged
LOST_RAY_FRACTION = 0.05


class InversionError(RuntimeError):
    """A program of the loop failed or produced unusable output.

    ``result`` holds everything captured up to the failure (``result.success`` is False).
    """

    def __init__(self, message: str, result: InversionResult | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class InversionResult:
    """Outcome of :func:`run_inversion`.

    ``iterations`` has one row per iteration (0 = starting model) with columns ``iter``,
    ``rms_ms``, ``variance_s2`` (data variance), ``chi2``, ``rms_reloc_ms`` (nonlinear: after
    ``gridloc``, before ``invert3d``; NaN otherwise), ``n_valid``, ``n_lost`` (observed picks
    whose model ray was lost), ``n_obs_invalid``, ``mean_resid_s``, ``model_change_rms``/
    ``model_change_max`` (km/s against the previous iteration), ``model_rms_from_ref``,
    ``vel_min``, ``vel_max``, ``frac_at_min_velocity``, ``subspace_dim`` (as reduced by
    ``invert3d``), ``mean_hypo_shift_km`` (against the previous iteration), ``wall_s``,
    ``flags``.  ``hypocentres`` is long format (``iter``, ``source`` = FMTOMO id of the sources
    (linear) or receivers (nonlinear) file, ``lat``, ``lon``, ``depth_km``).
    """

    run_dir: Path
    iterations: pd.DataFrame
    hypocentres: pd.DataFrame
    final_model: Path | None
    final_sources: Path | None
    residuals_path: Path
    iteration_dirs: dict[int, Path] = field(default_factory=dict)
    log_paths: dict[str, list[Path]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    success: bool = True
    wall_s: float = 0.0
    forward_wall_s: float = 0.0
    n_forward: int = 0

    @property
    def initial_rms_ms(self) -> float:
        return float(self.iterations["rms_ms"].iloc[0])

    @property
    def final_rms_ms(self) -> float:
        return float(self.iterations["rms_ms"].iloc[-1])


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _cp(run: Path, src: str, dst: str) -> None:
    shutil.copyfile(run / src, run / dst)


def _read_ints(path: Path, n: int) -> list[int]:
    toks = path.read_text().split()
    return [int(t) for t in toks[:n]]


def _tool(name: str, run: Path, timeout: float | None, prefix: str | None = None):  # noqa: ANN202
    try:
        return run_tool(name, run, timeout=timeout, log_prefix=prefix)
    except ToolError as exc:
        raise InversionError(f"{name} failed: {exc}") from exc


def _first_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip():
            return ln
    return ""


def _parse_residual_line(line: str) -> tuple[float, float, float]:
    toks = line.split()
    try:
        vals = tuple(float(t) for t in toks)
    except ValueError:
        vals = ()
    if len(vals) != 3:
        raise InversionError(f"cannot parse the output of 'residuals': {line!r}")
    return vals  # type: ignore[return-value]


def _hypo_frame(it: int, run: Path, nonlinear: bool) -> pd.DataFrame:
    if nonlinear:
        pos = read_receivers(run / "receivers.in").positions
        df = pos[["lat", "lon", "depth_km"]].copy()
        df.index.name = "source"
    else:
        srcs = read_sources(run / "sources.in")
        df = pd.DataFrame(
            {
                "lat": [s.lat for s in srcs],
                "lon": [s.lon for s in srcs],
                "depth_km": [s.depth_km for s in srcs],
            },
            index=pd.RangeIndex(1, len(srcs) + 1, name="source"),
        )
    df = df.reset_index()
    df.insert(0, "iter", it)
    return df


def _hypo_shift_km(a: pd.DataFrame, b: pd.DataFrame) -> float:
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    lat = a["lat"].to_numpy(float)
    dlat = (b["lat"].to_numpy(float) - lat) * KM_PER_DEG
    dlon = (
        (b["lon"].to_numpy(float) - a["lon"].to_numpy(float)) * KM_PER_DEG * np.cos(np.radians(lat))
    )
    dz = b["depth_km"].to_numpy(float) - a["depth_km"].to_numpy(float)
    return float(np.mean(np.sqrt(dlat**2 + dlon**2 + dz**2)))


def _velocities(run: Path) -> list[np.ndarray]:
    vg = read_vgrids(run / "vgrids.in")
    return [ds["velocity"].values for _, _, ds in vg]


def _model_stats(
    cur: list[np.ndarray], prev: list[np.ndarray], ref: list[np.ndarray], vmin: float
) -> dict[str, float]:
    allc = np.concatenate([v.ravel() for v in cur])
    out: dict[str, float] = {
        "vel_min": float(np.nanmin(allc)) if allc.size else math.nan,
        "vel_max": float(np.nanmax(allc)) if allc.size else math.nan,
        "n_nan": float(np.count_nonzero(~np.isfinite(allc))),
        "frac_at_min_velocity": float(np.mean(allc <= vmin * (1 + MIN_VEL_TOL)))
        if allc.size
        else 0.0,
    }

    def rms(a: list[np.ndarray], b: list[np.ndarray]) -> tuple[float, float]:
        d = np.concatenate([(x - y).ravel() for x, y in zip(a, b, strict=True)])
        return float(np.sqrt(np.mean(d**2))), float(np.max(np.abs(d)))

    out["model_change_rms"], out["model_change_max"] = rms(cur, prev)
    out["model_rms_from_ref"], _ = rms(cur, ref)
    return out


def _ray_stats(run: Path) -> dict[str, float]:
    """Valid/lost counts of the current ``arrivals.dat`` against ``otimes.dat``."""
    arr = read_arrivals(run / "arrivals.dat")
    obs = read_observed(run / "otimes.dat")
    out: dict[str, float] = {"n_arrivals": float(len(arr))}
    if len(arr) != len(obs):
        out.update(n_valid=float(arr["valid"].sum()), n_lost=math.nan, n_obs_invalid=math.nan)
        out["mean_resid_s"] = math.nan
        return out
    ov = (obs["time"].to_numpy() >= INVALID_OBSERVED_BELOW) & (obs["time"].to_numpy() > 0)
    ov = obs["time"].to_numpy() >= INVALID_OBSERVED_BELOW
    mv = arr["valid"].to_numpy()
    both = ov & mv
    out["n_valid"] = float(both.sum())
    out["n_lost"] = float((ov & ~mv).sum())
    out["n_obs_invalid"] = float((~ov).sum())
    out["mean_resid_s"] = (
        float(np.mean(arr["time"].to_numpy()[both] - obs["time"].to_numpy()[both]))
        if both.any()
        else math.nan
    )
    # residuals.dat has 5 decimals for the variance (s^2): useless below ~10 ms rms.  The same
    # statistics from the times themselves (residuals.f90:152-154) without the rounding.
    r = arr["time"].to_numpy()[both] - obs["time"].to_numpy()[both]
    n = int(both.sum())
    out["rms_exact_ms"] = 1000.0 * math.sqrt(float(np.sum(r**2)) / n) if n else math.nan
    out["variance_exact_s2"] = float(np.sum(r**2)) / (n - 1) if n > 1 else math.nan
    return out


_RE_SUBSPACE = re.compile(r"reduced from\s+(\d+)\s+to\s+(\d+)")


def _parse_invert_log(text: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    m = _RE_SUBSPACE.search(text)
    if m:
        out["subspace_dim_requested"] = float(m.group(1))
        out["subspace_dim"] = float(m.group(2))
    if "Singular matrix" in text:
        out["note"] = "invert3d reported a singular matrix (falling back to C.G.)"
    out["completed"] = float("successfully completed" in text)
    return out


# ---------------------------------------------------------------------------------------
# forward step
# ---------------------------------------------------------------------------------------
def _merge_arrtimes(groups: list[_fwd._Group], out: Path) -> None:
    """Concatenate the group ``arrtimes.dat`` (fields in global source order)."""
    header: list[str] | None = None
    blocks: list[tuple[int, int, list[str]]] = []  # (global src, path, lines)
    for grp in groups:
        lines = (grp.dir / "arrtimes.dat").read_text().splitlines()
        h = lines[:4]
        if header is None:
            header = h
        elif h[:3] != header[:3]:
            raise Fm3dError(f"arrtimes.dat grids of the groups differ ({grp.dir})")
        nr, nlat, nlon = (int(t) for t in h[0].split()[:3])
        nf, n = int(h[3].split()[0]), nr * nlat * nlon
        pos = 4
        for _ in range(nf):
            src, path, first = (int(t) for t in lines[pos].split()[:3])
            blocks.append(
                (grp.sources[src - 1], path, [f"{grp.sources[src - 1]:6d}{path:6d}{first:6d}"])
            )
            blocks[-1][2].extend(lines[pos + 1 : pos + 1 + n])
            pos += 1 + n
    assert header is not None
    blocks.sort(key=lambda b: (b[0], b[1]))
    header[3] = f"{len(blocks)}"
    with out.open("w") as fh:
        fh.write("\n".join(header) + "\n")
        for _, _, ls in blocks:
            fh.write("\n".join(ls) + "\n")


def _forward_timefields(
    inputs: Path,
    sources: list[Source],
    receivers: Receivers,
    n_jobs: int,
    workdir: Path,
    timeout: float | None,
) -> Fm3dResult:
    """Parallel ``fm3d`` with ``save_timefields_mode`` (merges ``arrtimes.dat`` as well)."""
    from osv.io.fmtomo import FrechetSpec

    t0 = time.perf_counter()
    ms = ModeSet.read(inputs / "mode_set.in")
    if not ms.no_pp_mode or ms.parallel_mode:
        raise Fm3dError("parallel fm3d needs no_pp_mode = T and parallel_mode = F")
    spec = FrechetSpec.read(inputs / "frechet.in") if (inputs / "frechet.in").exists() else None
    if spec is not None and not spec.compute:
        spec = None
    gs_all = read_gridsave(inputs / "gridsave.in")
    mem = _fwd.estimate_job_memory_gb(inputs / "propgrid.in")
    avail = _fwd.available_memory_gb()
    cap = max(1, int(_fwd.MEM_SAFETY * avail // mem)) if np.isfinite(avail) else n_jobs
    k = max(1, min(int(n_jobs), len(sources), cap))
    nrays = _fwd._ray_table(receivers).groupby("source").size()
    weights = [float(nrays.get(i, 0)) + _fwd.SOURCE_COST_RAYS for i in range(1, len(sources) + 1)]
    part = _fwd.partition_sources(weights, k)
    workdir.mkdir(parents=True, exist_ok=True)
    groups = _fwd._make_groups(sources, receivers, part, workdir)
    for g in groups:
        _fwd._write_group_inputs(g, inputs, sources, receivers, spec)
        lsrc = {s: n for n, s in enumerate(g.sources, start=1)}
        write_gridsave(
            g.dir / "gridsave.in",
            [
                GridSave(lsrc[e.source], e.paths, e.first_timefield)
                for e in gs_all
                if e.source in lsrc
            ],
        )
    walls = [0.0] * len(groups)

    def job(g: _fwd._Group) -> None:
        t1 = time.perf_counter()
        for stale in ("arrivals.dat", "frechet.dat", "rays.dat", "arrtimes.dat"):
            (g.dir / stale).unlink(missing_ok=True)
        run_tool("fm3d", g.dir, timeout=timeout, env={"OMP_NUM_THREADS": "1"})
        walls[g.index] = time.perf_counter() - t1

    if len(groups) == 1:
        job(groups[0])
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            for fut in [ex.submit(job, g) for g in groups]:
                fut.result()
    arr = workdir / "arrivals.dat"
    _fwd._merge_arrivals(groups, arr)
    _merge_arrtimes(groups, workdir / "arrtimes.dat")
    fre = None
    if spec is not None and all((g.dir / "frechet.dat").exists() for g in groups):
        fre = workdir / "frechet.dat"
        _fwd._merge_frechet(groups, fre, inputs, spec)
    rays = None
    if ms.save_rays_mode and all((g.dir / "rays.dat").exists() for g in groups):
        rays = workdir / "rays.dat"
        _fwd._merge_rays(groups, rays)
    for g in groups:
        shutil.rmtree(g.dir, ignore_errors=True)
    return Fm3dResult(
        out_dir=workdir,
        arrivals_path=arr,
        frechet_path=fre,
        rays_path=rays,
        n_jobs=len(groups),
        groups=[g.sources for g in groups],
        wall_s=time.perf_counter() - t0,
        job_wall_s=walls,
        job_mem_gb=mem,
    )


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
class _Loop:
    def __init__(
        self,
        run: Path,
        ctl: RunControls,
        n_jobs: int,
        keep: bool,
        timeout: float | None,
        mem_per_job_gb: float | None,
    ) -> None:
        self.run, self.ctl, self.n_jobs, self.keep = run, ctl, n_jobs, keep
        self.timeout, self.mem = timeout, mem_per_job_gb
        self.nonlinear = ctl.nonlinear
        self.fwd_wall = 0.0
        self.n_fwd = 0
        self.logs: dict[str, list[Path]] = {}
        self.vmin = float(ctl.invert3d.min_velocity)
        self.timefields = ModeSet.read(run / "mode_set.in").save_timefields_mode

    # ---- program steps -------------------------------------------------------------
    def tool(self, name: str, prefix: str | None = None) -> Path:
        res = _tool(name, self.run, self.timeout, prefix)
        self.logs.setdefault(name, []).append(res.stdout_log)
        return res.stdout_log

    def forward(self) -> None:
        run = self.run
        t0 = time.perf_counter()
        sources = read_sources(run / "sources.in")
        receivers = read_receivers(run / "receivers.in")
        work = run / "parallel"
        try:
            if self.timefields:
                res = _forward_timefields(run, sources, receivers, self.n_jobs, work, self.timeout)
            else:
                res = run_fm3d_parallel(
                    run,
                    sources,
                    receivers,
                    self.n_jobs,
                    self.mem,
                    work,
                    timeout=self.timeout,
                )
        except (Fm3dError, ToolError) as exc:
            raise InversionError(f"fm3d failed: {exc}") from exc
        shutil.copyfile(res.arrivals_path, run / "arrivals.dat")
        for name, p in (("frechet.dat", res.frechet_path), ("rays.dat", res.rays_path)):
            if p is not None:
                shutil.copyfile(p, run / name)
        if self.timefields:
            shutil.copyfile(work / "arrtimes.dat", run / "arrtimes.dat")
        (run / "fm3dlog.out").write_text(
            f"parallel fm3d: {res.n_jobs} jobs, wall {res.wall_s:.1f} s, "
            f"job walls {[round(w, 1) for w in res.job_wall_s]}\n"
        )
        self.fwd_wall += time.perf_counter() - t0
        self.n_fwd += 1

    def residuals(self, mode: str) -> tuple[float, float, float]:
        log_path = self.tool("residuals")
        line = _first_line(log_path.read_text(errors="replace"))
        vals = _parse_residual_line(line)
        with (self.run / "residuals.dat").open(mode) as fh:
            fh.write(" ".join(line.split()) + "\n")
        return vals

    def set_iter(self, n: int) -> None:
        (self.run / "inviter.in").write_text(f"{n}\n")


def _check_residual(rms: float) -> None:
    if not math.isfinite(rms):
        raise InversionError(f"residuals returned a non-finite misfit ({rms})")


def run_inversion(
    run_dir: str | Path,
    controls: FmtomoControls,
    n_iter: int | None = None,
    n_jobs: int = 1,
    keep_iterations: bool = True,
    *,
    phase: str | None = None,
    timeout: float | None = None,
    mem_per_job_gb: float | None = None,
    stop_on_error: bool = True,
    on_iteration: Callable[[int, dict[str, object]], None] | None = None,
) -> InversionResult:
    """Run the FMTOMO iteration loop in an assembled run directory.

    Parameters
    ----------
    run_dir
        Directory written by :func:`osv.fmtomo.rundir.assemble_run_dir` (with ``ref=True`` for a
        fresh start).
    controls
        The controls the directory was assembled with (selects the linear or the nonlinear loop).
    n_iter
        Number of inversion iterations (default: from the controls / ``tomo3d.in``).
    n_jobs
        Parallel ``fm3d`` processes (memory-capped by :func:`run_fm3d_parallel`).
    keep_iterations
        Also keep model and hypocentre snapshots per iteration (``summary.json`` and logs are
        always kept).
    phase
        Phase to run when the controls hold several.
    stop_on_error
        Raise :class:`InversionError` (with the partial result attached) when a program fails.
        If False, return the partial result with ``success=False`` instead.
    on_iteration
        Optional callback ``(iter, summary_dict)``.

    Returns
    -------
    InversionResult

    Raises
    ------
    InversionError
        A program failed, the model became non-finite or a misfit was not finite.
    NotImplementedError
        For joint P+S or Vp/Vs controls (use the dedicated drivers).
    """
    if controls.joint is not None or controls.vpvs is not None:
        raise NotImplementedError("joint P+S / Vp-Vs inversions are not driven by run_inversion")
    ctl = controls.runs[phase] if phase else controls.run
    run = Path(run_dir)
    for need in ("vgridsref.in", "interfacesref.in", "sourcesref.in", "mode_set.in"):
        if not (run / need).exists():
            raise InversionError(f"{run / need} is missing: assemble the run directory first")
    tfile = run / ctl.tomo_run_name
    ni, bgiter, binv = (
        _read_ints(tfile, 3)
        if tfile.exists()
        else (
            ctl.tomo_run.n_iterations,
            ctl.tomo_run.begin_from_current,
            ctl.tomo_run.start_with_forward,
        )
    )
    n_iter = int(ni if n_iter is None else n_iter)
    lp = _Loop(run, ctl, n_jobs, keep_iterations, timeout, mem_per_job_gb)
    t_start = time.perf_counter()
    idir = run / "iterations"
    rows: list[dict[str, object]] = []
    hypos: list[pd.DataFrame] = []
    dirs: dict[int, Path] = {}
    flags: list[str] = []
    result = InversionResult(
        run_dir=run,
        iterations=pd.DataFrame(),
        hypocentres=pd.DataFrame(),
        final_model=None,
        final_sources=None,
        residuals_path=run / "residuals.dat",
    )

    def finish(success: bool) -> InversionResult:
        result.iterations = pd.DataFrame(rows)
        result.hypocentres = pd.concat(hypos, ignore_index=True) if hypos else pd.DataFrame()
        result.iteration_dirs = dirs
        result.log_paths = lp.logs
        result.flags = flags
        result.success = success
        result.wall_s = time.perf_counter() - t_start
        result.forward_wall_s = lp.fwd_wall
        result.n_forward = lp.n_fwd
        result.final_model = run / "vgrids.in" if (run / "vgrids.in").exists() else None
        result.final_sources = run / ("receivers.in" if lp.nonlinear else "sources.in")
        return result

    state: dict[str, object] = {"prev_model": None, "ref_model": None, "prev_hypo": None}
    first_iter = 0

    def snapshot(
        it: int,
        rms: tuple[float, float, float],
        t0: float,
        inv_info: dict[str, float | str],
        reloc_rms: float,
        inv_log: Path | None,
    ) -> dict[str, object]:
        d = idir / f"{it:02d}"
        d.mkdir(parents=True, exist_ok=True)
        dirs[it] = d
        row: dict[str, object] = {
            "iter": it,
            "rms_ms": rms[0],
            "variance_s2": rms[1],
            "chi2": rms[2],
            "rms_reloc_ms": reloc_rms,
        }
        row.update(_ray_stats(run))
        cur = _velocities(run)
        if state["ref_model"] is None:
            state["ref_model"] = cur
            state["prev_model"] = cur
        ms = _model_stats(cur, state["prev_model"], state["ref_model"], lp.vmin)  # type: ignore[arg-type]
        state["prev_model"] = cur
        hy = _hypo_frame(it, run, lp.nonlinear)
        row["mean_hypo_shift_km"] = (
            0.0 if state["prev_hypo"] is None else _hypo_shift_km(state["prev_hypo"], hy)  # type: ignore[arg-type]
        )
        state["prev_hypo"] = hy
        hypos.append(hy)
        row.update({k: v for k, v in ms.items() if k != "n_nan"})
        row["subspace_dim"] = inv_info.get("subspace_dim", math.nan)
        # failure / quality flags
        fl: list[str] = []
        if ms["n_nan"] > 0:
            fl.append("nan_model")
        if ms["frac_at_min_velocity"] > 0:
            fl.append("min_velocity_hit")
        if rows:
            prev = rows[-1]
            if rms[0] > float(prev["rms_ms"]) * (1 + RESIDUAL_INCREASE_TOL):  # type: ignore[arg-type]
                fl.append("residual_increase")
            pv = float(prev["n_valid"])  # type: ignore[arg-type]
            if pv > 0 and (pv - float(row["n_valid"])) / pv > LOST_RAY_FRACTION:  # type: ignore[arg-type]
                fl.append("rays_lost")
        if "note" in inv_info:
            fl.append("invert3d_note")
        row["flags"] = ",".join(fl)
        row["wall_s"] = time.perf_counter() - t0
        row["dir"] = str(d)
        flags.extend(f"iteration {it}: {f}" for f in fl)
        if lp.keep:
            for name in ("vgrids.in", "sources.in", "stimes.dat", "mtimes.dat"):
                if (run / name).exists():
                    shutil.copyfile(run / name, d / name)
            if lp.nonlinear and (run / "receivers.in").exists():
                shutil.copyfile(run / "receivers.in", d / "receivers.in")
        if inv_log is not None and inv_log.exists():
            shutil.copyfile(inv_log, d / "invert3d.log")
        (d / "residuals.txt").write_text(f"{rms[0]:.2f} {rms[1]:.5f} {rms[2]:.5f}\n")
        summary = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in row.items()
        }
        summary["invert3d"] = {k: v for k, v in inv_info.items()}
        (d / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        rows.append(row)
        if on_iteration is not None:
            on_iteration(it, summary)
        if "nan_model" in fl:
            raise InversionError(f"iteration {it}: the velocity model contains NaN/Inf values")
        return row

    try:
        # ---- start (driver preamble)
        if bgiter == 0:
            _cp(run, "vgridsref.in", "vgrids.in")
            _cp(run, "interfacesref.in", "interfaces.in")
            _cp(run, "sourcesref.in", "sources.in")
            if lp.nonlinear:
                _cp(run, "otimesref.dat", "otimes.dat")
                _cp(run, "receiversref.in", "receivers.in")
            lp.tool("frechgen")
            lp.set_iter(1)
        else:
            try:
                first_iter = max(0, len((run / "residuals.dat").read_text().split("\n")) - 2)
            except OSError:
                first_iter = 0
        t0 = time.perf_counter()
        if bgiter == 0 or (bgiter == 1 and binv == 1):
            lp.forward()
            _cp(run, "arrivals.dat", "mtimes.dat")
            _cp(run, "arrivals.dat", "rtimes.dat")
            rms = lp.residuals("w")
            _check_residual(rms[0])
            snapshot(first_iter, rms, t0, {}, math.nan, None)
        elif not (run / "residuals.dat").exists():
            raise InversionError("continuing a run needs the residuals.dat of the previous run")

        # ---- iterations
        for k in range(1, n_iter + 1):
            it = first_iter + k
            t0 = time.perf_counter()
            reloc_rms = math.nan
            if lp.nonlinear:
                lp.tool("gridloc")
                lp.forward()
                _cp(run, "arrivals.dat", "mtimes.dat")
                mid = lp.residuals("a")
                reloc_rms = mid[0]
                lp.set_iter(k)
                inv_log = lp.tool("invert3d")
            else:
                inv_log = lp.tool("invert3d")
                if ctl.scorrect is not None and (run / "scorrect.in").exists():
                    lp.tool("scorrect")
            lp.forward()
            _cp(run, "arrivals.dat", "mtimes.dat")
            rms = lp.residuals("a")
            _check_residual(rms[0])
            lp.set_iter(k + 1)
            info = _parse_invert_log(inv_log.read_text(errors="replace"))
            if not info.get("completed"):
                raise InversionError(
                    f"iteration {it}: invert3d did not report successful completion; see {inv_log}"
                )
            snapshot(it, rms, t0, info, reloc_rms, inv_log)
    except FmtomoFormatError as exc:
        err = InversionError(f"unreadable program output: {exc}")
        flags.append(f"error: {err}")
        finish(False)
        if stop_on_error:
            err.result = result
            raise err from exc
        return result
    except InversionError as exc:
        log.error("inversion stopped: %s", exc)
        flags.append(f"error: {exc}")
        finish(False)
        if stop_on_error:
            exc.result = result
            raise
        return result
    return finish(True)
