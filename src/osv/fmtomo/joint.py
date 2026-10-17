"""P+S and joint Vp/Vs inversions with the FMTOMO Extras programs (experimental).

Three drivers, each replacing a ksh script of ``fortran/fmtomo_extras/bin`` by a Python loop
that executes the same programs in the same order (only ``fm3d`` is run in parallel):

:func:`run_ps_inversions`
    Independent P and S inversions (two :func:`osv.fmtomo.invert.run_inversion` runs).  Separate
    runs relocate the events independently, so the P and the S hypocentres of the same event end
    up in different places.  Three relocation modes are offered: ``"independent"`` (each phase
    relocates on its own), ``"p_fixed_for_s"`` (invert and relocate with P only, transfer the
    P hypocentres and origin times to the S run and keep them fixed) and ``"joint"`` (delegates
    to :func:`run_joint_inversion`).

:func:`run_joint_inversion`
    Driver ``tomo3dj``: coupled non-linear relocation with ``gridlocj`` (uses the P and the S
    time fields and the P/S association file ``src.dat`` built by ``jointimes``) followed by the
    P and S velocity inversions of every iteration::

        start:        frechgen, fm3d, residuals (P then S)
        iteration:    gridlocj, [fm3d, residuals] P, S, [invert3d, fm3d, residuals] P, S

:func:`run_vpvs_inversion`
    Driver ``cp_script`` + ``tomo3dps``: linear direct inversion for the Vp/Vs ratio from S-P
    differential times (``mkvpvs``, ``frvpvs``, ``vpvsinv``, ``vpvsresids``) on top of finished
    P and S runs::

        start:        mkvpvs, frvpvs, vpvsresids
        iteration:    vpvsinv, frvpvs, vpvsresids

Deviations from the scripts (deliberate, as in :mod:`osv.fmtomo.invert`):

* ``tomo3dj`` writes ``inviter.in`` from the shell variable ``COUNT`` *before* the first
  ``invert3d`` while it is still unset (crash of the bash port); we write the number of the
  running iteration.
* ``tomo3dj`` does not call ``jointimes``; the user is expected to run it by hand.  We run it at
  the start when ``src.dat`` is missing (or ``run_jointimes=True``).
* ``tomo3dps`` calls ``$frech`` which is never set (a no-op); omitted.
* ``fm3d`` runs through :func:`osv.fmtomo.forward.run_fm3d_parallel` (or the time-field variant
  of :mod:`osv.fmtomo.invert`), output is identical.

Everything here is **experimental**: it is verified against the shipped drivers on the Extras
``observed`` example for one iteration (``tests/fmtomo/test_joint.py``), but not on other data.
Known limits of the Extras programs (see ``fortran/fmtomo_extras/docs/Instructions.pdf``): local
earthquakes only, no internal interfaces, P and S propagation grids identical, at most 300
stations and 5000 events (``maxr``, ``maxs`` of ``jointimes``/``frvpvs``), and at most 30
characters in the file names of ``jointimes.in``/``gridlocj.in``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from osv.fmtomo import invert as _inv
from osv.fmtomo.controls import FmtomoControls
from osv.fmtomo.invert import InversionError, InversionResult, run_inversion
from osv.fmtomo.rundir import DIR_P, DIR_S
from osv.io.fmtomo import FmtomoFormatError
from osv.tools import ToolError, run_tool

log = logging.getLogger(__name__)

__all__ = [
    "JointInversionResult",
    "PSInversionResult",
    "VpVsInversionResult",
    "prepare_vpvs_dir",
    "run_joint_inversion",
    "run_ps_inversions",
    "run_vpvs_inversion",
    "transfer_hypocentres",
]

#: relocation modes of :func:`run_ps_inversions`
PS_RELOCATION_MODES = ("independent", "p_fixed_for_s", "joint")
#: ``gridloc`` ignores events with fewer picks than this (used to keep S hypocentres fixed)
NO_RELOCATION_MIN_PICKS = 10**8
#: file names are read with ``a30`` by ``jointimes``/``gridlocj``
MAX_NAME_LEN = 30
VPVS_DIR = "vpvs"


# ---------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------
@dataclass
class PSInversionResult:
    """Outcome of :func:`run_ps_inversions` (``p`` and ``s`` are the per-phase results)."""

    mode: str
    p: InversionResult
    s: InversionResult
    transferred_events: int = 0
    notes: list[str] = field(default_factory=list)
    joint: JointInversionResult | None = None

    @property
    def success(self) -> bool:
        return self.p.success and self.s.success


@dataclass
class JointInversionResult:
    """Outcome of :func:`run_joint_inversion`.

    ``p`` and ``s`` are :class:`InversionResult` tables of the two run directories (same
    columns as for :func:`run_inversion`; ``rms_reloc_ms`` is the misfit after ``gridlocj``).
    ``iterations`` combines them: ``iter``, ``rms_p_ms``, ``rms_s_ms``, ``rms_reloc_p_ms``,
    ``rms_reloc_s_ms``, ``variance_p_s2``, ``variance_s_s2``, ``mean_hypo_shift_p_km``,
    ``mean_hypo_shift_s_km``, ``n_valid_p``, ``n_valid_s``, ``flags``, ``wall_s``.
    """

    root: Path
    iterations: pd.DataFrame
    p: InversionResult
    s: InversionResult
    src_dat: Path
    log_paths: dict[str, list[Path]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    success: bool = True
    wall_s: float = 0.0
    forward_wall_s: float = 0.0
    n_forward: int = 0

    @property
    def initial_rms_ms(self) -> tuple[float, float]:
        return float(self.iterations["rms_p_ms"].iloc[0]), float(
            self.iterations["rms_s_ms"].iloc[0]
        )

    @property
    def final_rms_ms(self) -> tuple[float, float]:
        return float(self.iterations["rms_p_ms"].iloc[-1]), float(
            self.iterations["rms_s_ms"].iloc[-1]
        )


@dataclass
class VpVsInversionResult:
    """Outcome of :func:`run_vpvs_inversion`.

    ``iterations`` columns: ``iter`` (0 = starting constant Vp/Vs), ``rms_s`` and ``variance_s2``
    (S-P time misfit in seconds, from ``vpvsresids``), ``n_data``, ``vpvs_min``/``vpvs_max``/
    ``vpvs_mean`` of the current Vp/Vs grid, ``model_change_rms``/``model_change_max`` (against
    the previous iteration), ``model_rms_from_ref``, ``flags``, ``wall_s``.
    """

    vpvs_dir: Path
    iterations: pd.DataFrame
    final_model: Path | None
    ratio_of_vp_vs_model: Path | None
    residuals_path: Path
    iteration_dirs: dict[int, Path] = field(default_factory=dict)
    log_paths: dict[str, list[Path]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    success: bool = True
    wall_s: float = 0.0

    @property
    def initial_rms_s(self) -> float:
        return float(self.iterations["rms_s"].iloc[0])

    @property
    def final_rms_s(self) -> float:
        return float(self.iterations["rms_s"].iloc[-1])


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _tool(name: str, cwd: Path, timeout: float | None, prefix: str | None = None) -> Path:
    try:
        return run_tool(name, cwd, timeout=timeout, log_prefix=prefix).stdout_log
    except ToolError as exc:
        raise InversionError(f"{name} failed: {exc}") from exc


def _cp(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def _need(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise InversionError(f"{p} is missing: assemble the run directories first")


def _check_name(path: str, what: str) -> None:
    if len(path) > MAX_NAME_LEN:
        raise InversionError(
            f"{what} '{path}' has more than {MAX_NAME_LEN} characters; the Extras programs "
            "read file names with a30. Use shorter directory names"
        )


def _read_tomo_ints(path: Path, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not path.exists():
        return default
    return tuple(_inv._read_ints(path, 3))  # type: ignore[return-value]


def _phase_ctl(controls: FmtomoControls, phase: str):  # noqa: ANN202
    try:
        return controls.runs[phase]
    except KeyError:
        raise InversionError(
            f"the controls hold no {phase} run (phases={controls.phases})"
        ) from None


class _Phase:
    """One phase directory of the joint run: loop helpers plus per-iteration capture."""

    def __init__(
        self,
        label: str,
        run: Path,
        controls: FmtomoControls,
        n_jobs: int,
        keep: bool,
        timeout: float | None,
        mem: float | None,
        *,
        timefields: bool | None = None,
    ) -> None:
        self.label, self.run, self.keep = label, run, keep
        self.lp = _inv._Loop(run, _phase_ctl(controls, label), n_jobs, keep, timeout, mem)
        if timefields is not None:
            self.lp.timefields = timefields
        self.rows: list[dict[str, object]] = []
        self.hypos: list[pd.DataFrame] = []
        self.dirs: dict[int, Path] = {}
        self.flags: list[str] = []
        self.state: dict[str, object] = {"prev": None, "ref": None, "hypo": None}
        self.first_iter = 0

    # -- capture (same quantities as run_inversion) --
    def record(
        self,
        it: int,
        rms: tuple[float, float, float],
        t0: float,
        info: dict[str, float | str],
        reloc_rms: float,
        inv_log: Path | None,
        on_iteration: Callable[[int, dict[str, object]], None] | None = None,
    ) -> None:
        run, lp, st = self.run, self.lp, self.state
        d = run / "iterations" / f"{it:02d}"
        d.mkdir(parents=True, exist_ok=True)
        self.dirs[it] = d
        row: dict[str, object] = {
            "iter": it,
            "rms_ms": rms[0],
            "variance_s2": rms[1],
            "chi2": rms[2],
            "rms_reloc_ms": reloc_rms,
        }
        row.update(_inv._ray_stats(run))
        cur = _inv._velocities(run)
        if st["ref"] is None:
            st["ref"] = st["prev"] = cur
        ms = _inv._model_stats(cur, st["prev"], st["ref"], lp.vmin)  # type: ignore[arg-type]
        st["prev"] = cur
        hy = _inv._hypo_frame(it, run, True)
        row["mean_hypo_shift_km"] = (
            0.0 if st["hypo"] is None else _inv._hypo_shift_km(st["hypo"], hy)  # type: ignore[arg-type]
        )
        st["hypo"] = hy
        self.hypos.append(hy)
        row.update({k: v for k, v in ms.items() if k != "n_nan"})
        row["subspace_dim"] = info.get("subspace_dim", math.nan)
        fl: list[str] = []
        if ms["n_nan"] > 0:
            fl.append("nan_model")
        if ms["frac_at_min_velocity"] > 0:
            fl.append("min_velocity_hit")
        if self.rows:
            prev = self.rows[-1]
            if rms[0] > float(prev["rms_ms"]) * (1 + _inv.RESIDUAL_INCREASE_TOL):  # type: ignore[arg-type]
                fl.append("residual_increase")
            pv = float(prev["n_valid"])  # type: ignore[arg-type]
            lost = (pv - float(row["n_valid"])) / pv if pv > 0 else 0.0  # type: ignore[arg-type]
            if lost > _inv.LOST_RAY_FRACTION:
                fl.append("rays_lost")
        if "note" in info:
            fl.append("invert3d_note")
        row["flags"] = ",".join(fl)
        row["wall_s"] = time.perf_counter() - t0
        row["dir"] = str(d)
        self.flags.extend(f"{self.label} iteration {it}: {f}" for f in fl)
        if self.keep:
            for name in ("vgrids.in", "sources.in", "receivers.in", "stimes.dat", "mtimes.dat"):
                if (run / name).exists():
                    _cp(run / name, d / name)
        if inv_log is not None and inv_log.exists():
            _cp(inv_log, d / "invert3d.log")
        (d / "residuals.txt").write_text(f"{rms[0]:.2f} {rms[1]:.5f} {rms[2]:.5f}\n")
        summary = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in row.items()
        }
        summary["invert3d"] = dict(info)
        (d / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        self.rows.append(row)
        if on_iteration is not None:
            on_iteration(it, {"phase": self.label, **summary})
        if "nan_model" in fl:
            raise InversionError(
                f"{self.label} iteration {it}: the velocity model contains NaN/Inf values"
            )

    def result(self, success: bool, wall: float) -> InversionResult:
        lp, run = self.lp, self.run
        return InversionResult(
            run_dir=run,
            iterations=pd.DataFrame(self.rows),
            hypocentres=pd.concat(self.hypos, ignore_index=True) if self.hypos else pd.DataFrame(),
            final_model=run / "vgrids.in" if (run / "vgrids.in").exists() else None,
            final_sources=run / "receivers.in",
            residuals_path=run / "residuals.dat",
            iteration_dirs=self.dirs,
            log_paths=lp.logs,
            flags=self.flags,
            success=success,
            wall_s=wall,
            forward_wall_s=lp.fwd_wall,
            n_forward=lp.n_fwd,
        )


# ---------------------------------------------------------------------------------------
# P/S association (jointimes) and hypocentre transfer
# ---------------------------------------------------------------------------------------
def _run_jointimes(root: Path, controls: FmtomoControls | None, timeout: float | None) -> Path:
    """Run ``jointimes`` in ``root`` (writes ``jointimes.in`` from the controls if missing)."""
    jt = root / "jointimes.in"
    if not jt.exists():
        if controls is None or controls.joint is None:
            raise InversionError(f"{jt} is missing and the controls hold no joint setup")
        jt.write_text(controls.joint.jointimes.format())
    _tool("jointimes", root, timeout)
    out = root / "src.dat"
    if not out.exists():
        raise InversionError("jointimes did not write src.dat")
    return out


def _read_src_dat(path: Path) -> list[list[tuple[int, float, float]]]:
    """Source-receiver association file: per record the (event id, time, error) of its picks."""
    toks = path.read_text().split()
    pos = 0
    n = int(toks[pos])
    pos += 1
    out: list[list[tuple[int, float, float]]] = []
    for _ in range(n):
        ns = int(toks[pos + 1])
        pos += 2
        rec = []
        for _ in range(ns):
            rec.append((int(toks[pos]), float(toks[pos + 1]), float(toks[pos + 2])))
            pos += 3
        out.append(rec)
    return out


def _read_receiver_lines(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines()
    return lines[:1], lines[1:]


def _read_otimes_lines(path: Path) -> tuple[str, list[list[str]]]:
    lines = path.read_text().splitlines()
    return lines[0], [ln.split() for ln in lines[1 : 1 + int(lines[0].split()[0])]]


def transfer_hypocentres(
    root: str | Path,
    p_dir: str | Path,
    s_dir: str | Path,
    src_dat: str | Path | None = None,
    *,
    backup_suffix: str = ".orig",
) -> int:
    """Copy the P-relocated hypocentres and origin times of ``p_dir`` into the S reference files.

    ``src_dat`` is the joint association file made by ``jointimes`` (records of P stations
    first, then S stations; the same event id marks the same event).  For every S pick whose
    event also has a P pick, ``receiversref.in`` and ``otimesref.dat`` of ``s_dir`` get the
    current event location of the P run (``receivers.in``) and the S time shifted by the origin
    time change of the P run (``otimes.dat`` - ``otimesref.dat``).  S-only events keep their
    position.  The originals are saved with ``backup_suffix``.

    Returns
    -------
    int
        Number of S picks that received a P hypocentre.
    """
    root, p_dir, s_dir = Path(root), Path(p_dir), Path(s_dir)
    assoc = _read_src_dat(Path(src_dat) if src_dat else root / "src.dat")
    n_p = len(_read_src_dat(p_dir / "src.dat"))
    sid_p = [sid for rec in assoc[:n_p] for sid, _, _ in rec]
    sid_s = [sid for rec in assoc[n_p:] for sid, _, _ in rec]
    _, rec_p = _read_receiver_lines(p_dir / "receivers.in")
    _, rec_s = _read_receiver_lines(s_dir / "receiversref.in")
    hdr_p, ot_p = _read_otimes_lines(p_dir / "otimes.dat")
    _, ot_pref = _read_otimes_lines(p_dir / "otimesref.dat")
    hdr_s, ot_sref = _read_otimes_lines(s_dir / "otimesref.dat")
    if not (len(sid_p) == len(ot_p) == len(ot_pref) and len(sid_s) == len(ot_sref)):
        raise InversionError("src.dat does not match the P/S observed-time files")
    # event id -> (position line, origin time shift) of the P run
    ev: dict[int, tuple[str, float]] = {}
    for k, sid in enumerate(sid_p):
        if sid not in ev:
            ev[sid] = (rec_p[4 * k], float(ot_p[k][4]) - float(ot_pref[k][4]))
    n = 0
    new_rec = list(rec_s)
    new_ot = [list(t) for t in ot_sref]
    for k, sid in enumerate(sid_s):
        if sid in ev:
            pos, shift = ev[sid]
            new_rec[4 * k] = pos
            if float(new_ot[k][4]) > 0:
                new_ot[k][4] = f"{float(new_ot[k][4]) + shift:.9g}"
            n += 1
    for name in ("receiversref.in", "otimesref.dat"):
        shutil.copyfile(s_dir / name, s_dir / (name + backup_suffix))
    (s_dir / "receiversref.in").write_text(f"{len(sid_s)}\n" + "\n".join(new_rec) + "\n")
    (s_dir / "otimesref.dat").write_text(
        f"{hdr_s}\n" + "\n".join(" ".join(t) for t in new_ot) + "\n"
    )
    return n


# ---------------------------------------------------------------------------------------
# 1. independent P and S inversions
# ---------------------------------------------------------------------------------------
def run_ps_inversions(
    p_dir: str | Path,
    s_dir: str | Path,
    controls: FmtomoControls,
    n_iter: int | None = None,
    n_jobs: int = 1,
    keep_iterations: bool = True,
    *,
    relocation: str = "independent",
    timeout: float | None = None,
    mem_per_job_gb: float | None = None,
    stop_on_error: bool = True,
    on_iteration: Callable[[int, dict[str, object]], None] | None = None,
) -> PSInversionResult:
    """Run the P and the S inversion in two assembled run directories (experimental).

    Parameters
    ----------
    p_dir, s_dir
        Run directories (e.g. ``root/invert_p`` and ``root/invert_s`` from
        :func:`osv.fmtomo.rundir.assemble_joint_run_dirs`), each for the phase named in
        ``controls.runs``.
    controls
        Controls with ``runs["P"]`` and ``runs["S"]`` (``phases=["P", "S"]``).
    relocation
        ``"independent"``: both runs relocate (or not, per the controls) on their own.  The P
        and S hypocentres of one event therefore end up in different places: a
        limitation of separate runs.  ``"p_fixed_for_s"``: run P first with its relocation,
        transfer the relocated hypocentres and origin times to the S run
        (:func:`transfer_hypocentres`, needs ``jointimes`` and ``src.dat`` in the common parent
        directory) and keep them fixed in the S run (``gridloc`` is told to ignore every
        event).  ``"joint"``: coupled relocation, see :func:`run_joint_inversion` (directories
        must be ``<root>/invert_p`` and ``<root>/invert_s``).

    Returns
    -------
    PSInversionResult
    """
    if relocation not in PS_RELOCATION_MODES:
        raise ValueError(f"relocation must be one of {PS_RELOCATION_MODES}, got {relocation!r}")
    p_dir, s_dir = Path(p_dir), Path(s_dir)
    if relocation == "joint":
        if p_dir.name != DIR_P or s_dir.name != DIR_S or p_dir.parent != s_dir.parent:
            raise InversionError(f"joint relocation needs <root>/{DIR_P} and <root>/{DIR_S}")
        jr = run_joint_inversion(
            p_dir.parent,
            controls,
            n_iter,
            n_jobs,
            keep_iterations,
            timeout=timeout,
            mem_per_job_gb=mem_per_job_gb,
            stop_on_error=stop_on_error,
            on_iteration=on_iteration,
        )
        return PSInversionResult("joint", jr.p, jr.s, joint=jr, notes=list(jr.flags))
    kw = dict(
        n_iter=n_iter,
        n_jobs=n_jobs,
        keep_iterations=keep_iterations,
        timeout=timeout,
        mem_per_job_gb=mem_per_job_gb,
        stop_on_error=stop_on_error,
        on_iteration=on_iteration,
    )
    notes: list[str] = []
    _phase_ctl(controls, "P")
    ctl_s = _phase_ctl(controls, "S")
    pc = dataclasses.replace(controls, joint=None, vpvs=None)
    res_p = run_inversion(p_dir, pc, phase="P", **kw)  # type: ignore[arg-type]
    n_tr = 0
    if relocation == "p_fixed_for_s":
        if not res_p.success:
            raise InversionError("the P inversion failed; not starting S", res_p)
        root = p_dir.parent
        if s_dir.parent != root:
            raise InversionError("p_fixed_for_s needs p_dir and s_dir in the same parent directory")
        _check_name(f"{p_dir.name}/receiversref.in", "jointimes file name")
        _check_name(f"{s_dir.name}/receiversref.in", "jointimes file name")
        src = root / "src.dat"
        if not src.exists():
            src = _run_jointimes(root, controls, timeout)
        n_tr = transfer_hypocentres(root, p_dir, s_dir, src)
        notes.append(f"{n_tr} S picks received the P hypocentres; S hypocentres are fixed")
        if ctl_s.gridloc is not None:
            frozen = dataclasses.replace(ctl_s.gridloc, min_stations=NO_RELOCATION_MIN_PICKS)
            (s_dir / "gridloc.in").write_text(frozen.format())
    else:
        notes.append(
            "P and S were relocated independently: hypocentres of the same event differ "
            "between the two runs"
        )
    res_s = run_inversion(s_dir, pc, phase="S", **kw)  # type: ignore[arg-type]
    return PSInversionResult(relocation, res_p, res_s, n_tr, notes)


# ---------------------------------------------------------------------------------------
# 2. joint P+S inversion with joint relocation (tomo3dj)
# ---------------------------------------------------------------------------------------
def run_joint_inversion(
    root: str | Path,
    controls: FmtomoControls,
    n_iter: int | None = None,
    n_jobs: int = 1,
    keep_iterations: bool = True,
    *,
    run_jointimes: bool | None = None,
    timeout: float | None = None,
    mem_per_job_gb: float | None = None,
    stop_on_error: bool = True,
    on_iteration: Callable[[int, dict[str, object]], None] | None = None,
) -> JointInversionResult:
    """Joint P+S inversion with coupled relocation, the loop of ``tomo3dj`` (experimental).

    Parameters
    ----------
    root
        Directory with ``invert_p/``, ``invert_s/``, ``tomo3dj.in``, ``gridlocj.in`` and
        ``jointimes.in`` (:func:`osv.fmtomo.rundir.assemble_joint_run_dirs`).  Both phase
        directories need ``save_timefields_mode = T`` in ``mode_set.in`` and one ``gridsave.in``
        entry per station (written by the assembler).
    controls
        Controls built for ``phases=['P', 'S']`` with nonlinear relocation (``controls.joint``).
    n_iter
        Iterations (default: ``tomo3dj.in``).
    run_jointimes
        Run ``jointimes`` first.  Default: only if ``src.dat`` is missing in ``root`` (the shipped
        script never calls it).
    on_iteration
        Callback ``(iter, summary)``; the summary has a ``phase`` key (``"P"``/``"S"``).

    Returns
    -------
    JointInversionResult

    Raises
    ------
    InversionError
        A program failed or produced unusable output (``result`` attribute holds the partial
        result).
    """
    if controls.joint is None:
        raise InversionError("controls hold no joint P+S setup (phases=['P','S'], nonlinear)")
    root = Path(root)
    pd_, sd_ = root / DIR_P, root / DIR_S
    for need in (
        root / "gridlocj.in",
        pd_ / "vgridsref.in",
        pd_ / "mode_set.in",
        sd_ / "vgridsref.in",
        sd_ / "mode_set.in",
    ):
        _need(need)
    ni, bgiter, binv = _read_tomo_ints(
        root / "tomo3dj.in",
        (controls.joint.tomo_run.n_iterations, 0, 0),
    )
    n_iter = int(ni if n_iter is None else n_iter)
    ph = {
        lab: _Phase(lab, d, controls, n_jobs, keep_iterations, timeout, mem_per_job_gb)
        for lab, d in (("P", pd_), ("S", sd_))
    }
    for p in ph.values():
        if not p.lp.timefields:
            raise InversionError(
                f"{p.run / 'mode_set.in'}: save_timefields_mode must be T (gridlocj searches the "
                "model time fields)"
            )
    t_start = time.perf_counter()
    logs: dict[str, list[Path]] = {}
    flags: list[str] = []
    out: dict[str, JointInversionResult] = {}

    def tool_root(name: str) -> Path:
        res = _tool(name, root, timeout)
        logs.setdefault(name, []).append(res)
        return res

    def build(success: bool) -> JointInversionResult:
        wall = time.perf_counter() - t_start
        rp, rs = ph["P"].result(success, wall), ph["S"].result(success, wall)
        by_iter = {
            lab: p.result(success, wall).iterations.set_index("iter") for lab, p in ph.items()
        }
        rows = []
        for it in sorted(set(by_iter["P"].index) & set(by_iter["S"].index)):
            a, b = by_iter["P"].loc[it], by_iter["S"].loc[it]
            rows.append(
                {
                    "iter": int(it),
                    "rms_p_ms": a["rms_ms"],
                    "rms_s_ms": b["rms_ms"],
                    "rms_reloc_p_ms": a["rms_reloc_ms"],
                    "rms_reloc_s_ms": b["rms_reloc_ms"],
                    "variance_p_s2": a["variance_s2"],
                    "variance_s_s2": b["variance_s2"],
                    "mean_hypo_shift_p_km": a["mean_hypo_shift_km"],
                    "mean_hypo_shift_s_km": b["mean_hypo_shift_km"],
                    "n_valid_p": a["n_valid"],
                    "n_valid_s": b["n_valid"],
                    "flags": ",".join(f for f in (a["flags"], b["flags"]) if f),
                    "wall_s": a["wall_s"] + b["wall_s"],
                }
            )
        res = JointInversionResult(
            root=root,
            iterations=pd.DataFrame(rows),
            p=rp,
            s=rs,
            src_dat=root / "src.dat",
            log_paths={
                **logs,
                **{f"{k}_P": v for k, v in ph["P"].lp.logs.items()},
                **{f"{k}_S": v for k, v in ph["S"].lp.logs.items()},
            },
            flags=flags + ph["P"].flags + ph["S"].flags,
            success=success,
            wall_s=wall,
            forward_wall_s=ph["P"].lp.fwd_wall + ph["S"].lp.fwd_wall,
            n_forward=ph["P"].lp.n_fwd + ph["S"].lp.n_fwd,
        )
        out["r"] = res
        return res

    def step_forward_resid(p: _Phase, mode: str) -> tuple[float, float, float]:
        p.lp.forward()
        _cp(p.run / "arrivals.dat", p.run / "mtimes.dat")
        return p.lp.residuals(mode)

    try:
        if run_jointimes or (run_jointimes is None and not (root / "src.dat").exists()):
            for d in (DIR_P, DIR_S):
                _check_name(f"{d}/receiversref.in", "jointimes file name")
            _run_jointimes(root, controls, timeout)
        _need(root / "src.dat")
        t0 = time.perf_counter()
        if bgiter == 0:
            for p in ph.values():
                for a, b in (
                    ("vgridsref.in", "vgrids.in"),
                    ("interfacesref.in", "interfaces.in"),
                    ("sourcesref.in", "sources.in"),
                    ("otimesref.dat", "otimes.dat"),
                    ("receiversref.in", "receivers.in"),
                ):
                    _cp(p.run / a, p.run / b)
                p.lp.tool("frechgen")
                p.lp.set_iter(1)
        else:
            for p in ph.values():
                try:
                    txt = (p.run / "residuals.dat").read_text()
                except OSError:
                    raise InversionError(
                        f"continuing a run needs {p.run / 'residuals.dat'}"
                    ) from None
                p.first_iter = max(0, len(txt.split("\n")) - 2)
        first = ph["P"].first_iter
        if bgiter == 0 or (bgiter == 1 and binv == 1):
            for p in ph.values():
                p.lp.forward()
                _cp(p.run / "arrivals.dat", p.run / "mtimes.dat")
                _cp(p.run / "arrivals.dat", p.run / "rtimes.dat")
                rms = p.lp.residuals("w")
                _inv._check_residual(rms[0])
                p.record(first, rms, t0, {}, math.nan, None, on_iteration)

        for k in range(1, n_iter + 1):
            it = first + k
            t0 = time.perf_counter()
            tool_root("gridlocj")
            if "successfully completed" not in logs["gridlocj"][-1].read_text(errors="replace"):
                raise InversionError(
                    f"iteration {it}: gridlocj did not complete; see {logs['gridlocj'][-1]}"
                )
            reloc = {}
            for lab, p in ph.items():
                reloc[lab] = step_forward_resid(p, "a")[0]
                p.lp.set_iter(k)
            rec: dict[str, tuple[tuple[float, float, float], dict, Path]] = {}
            for lab, p in ph.items():
                inv_log = p.lp.tool("invert3d")
                rms = step_forward_resid(p, "a")
                _inv._check_residual(rms[0])
                p.lp.set_iter(k + 1)
                info = _inv._parse_invert_log(inv_log.read_text(errors="replace"))
                if not info.get("completed"):
                    raise InversionError(
                        f"{lab} iteration {it}: invert3d did not report successful completion; "
                        f"see {inv_log}"
                    )
                rec[lab] = (rms, info, inv_log)
            for lab, p in ph.items():
                rms, info, inv_log = rec[lab]
                p.record(it, rms, t0, info, reloc[lab], inv_log, on_iteration)
    except FmtomoFormatError as exc:
        err = InversionError(f"unreadable program output: {exc}")
        flags.append(f"error: {err}")
        build(False)
        if stop_on_error:
            err.result = out["r"]  # type: ignore[assignment]
            raise err from exc
        return out["r"]
    except InversionError as exc:
        log.error("joint inversion stopped: %s", exc)
        flags.append(f"error: {exc}")
        build(False)
        if stop_on_error:
            exc.result = out["r"]  # type: ignore[assignment]
            raise
        return out["r"]
    return build(True)


# ---------------------------------------------------------------------------------------
# 3. Vp/Vs inversion (cp_script + tomo3dps)
# ---------------------------------------------------------------------------------------
#: (source dir, file) -> file in the vpvs directory (``cp_script``)
_CP_SCRIPT = (
    ("p", "vgrids.in", "vgridsp.in"),
    ("p", "vgridsref.in", "vgridspr.in"),
    ("s", "vgrids.in", "vgridss.in"),
    ("s", "vgridsref.in", "vgridssr.in"),
    ("p", "interfacesref.in", "interfacesref.in"),
    ("p", "interfaces.in", "interfaces.in"),
    ("p", "otimes.dat", "otimesp.dat"),
    ("s", "otimes.dat", "otimess.dat"),
    ("s", "rays.dat", "rays.dat"),
    ("p", "sources.in", "sourcesp.in"),
    ("s", "sources.in", "sourcess.in"),
)


def prepare_vpvs_dir(
    root: str | Path,
    vpvs_dir: str | Path | None = None,
    controls: FmtomoControls | None = None,
) -> Path:
    """Copy the P/S results into the Vp/Vs directory (what ``cp_script`` does).

    ``root`` holds ``invert_p/``, ``invert_s/`` and ``src.dat``; the Vp/Vs directory defaults to
    ``root/vpvs``.  The control files (``mkvpvs.in``, ``frvpvs.in``, ``vpvsinv.in``,
    ``vpvsresids.in``, ``tomo3dps.in``) are written from ``controls.vpvs`` when missing.
    ``invert_s/rays.dat`` must exist (save_rays_mode, written by the last ``fm3d`` run).
    """
    root = Path(root)
    vd = Path(vpvs_dir) if vpvs_dir else root / VPVS_DIR
    vd.mkdir(parents=True, exist_ok=True)
    srcs = {"p": root / DIR_P, "s": root / DIR_S}
    for which, a, b in _CP_SCRIPT:
        f = srcs[which] / a
        if not f.exists():
            hint = " (run the S forward step, e.g. forward=True)" if a == "rays.dat" else ""
            raise InversionError(f"{f} is missing{hint}")
        _cp(f, vd / b)
    if not (root / "src.dat").exists():
        raise InversionError(f"{root / 'src.dat'} is missing: run jointimes first")
    _cp(root / "src.dat", vd / "src.dat")
    if controls is not None and controls.vpvs is not None:
        for name, text in controls.vpvs.texts().items():
            if not (vd / name).exists():
                (vd / name).write_text(text)
    return vd


def _read_grid_values(path: Path) -> np.ndarray:
    """Values of a single-parameter grid file (4 header lines, one value per line)."""
    try:
        return np.loadtxt(path, skiprows=4, ndmin=1)
    except ValueError as exc:
        raise FmtomoFormatError(f"{path}: not a Vp/Vs grid file ({exc})") from exc


def _count_data(mtimes: Path) -> int:
    n = 0
    with mtimes.open() as fh:
        for ln in fh:
            t = ln.split()
            if t and t[0] == "1":
                n += 1
    return n


def run_vpvs_inversion(
    root: str | Path,
    controls: FmtomoControls | None = None,
    n_iter: int | None = None,
    keep_iterations: bool = True,
    *,
    vpvs_dir: str | Path | None = None,
    forward: bool = False,
    prepare: bool = True,
    n_jobs: int = 1,
    timeout: float | None = None,
    mem_per_job_gb: float | None = None,
    stop_on_error: bool = True,
    on_iteration: Callable[[int, dict[str, object]], None] | None = None,
) -> VpVsInversionResult:
    """Direct linear Vp/Vs inversion from S-P times, the loop of ``tomo3dps`` (experimental).

    Parameters
    ----------
    root
        Directory with the finished P and S run directories ``invert_p/``/``invert_s/`` and the
        joint association file ``src.dat`` (``jointimes``).
    controls
        Optional; supplies the control files when the Vp/Vs directory does not have them
        (``phases=['vpvs']``) and the iteration count.
    n_iter
        Iterations (default: ``tomo3dps.in``).
    vpvs_dir
        Directory of the run (default ``root/vpvs``).
    forward
        Run ``fm3d`` (in parallel) in the S directory first to regenerate ``rays.dat`` for the
        current S model (``invert_s/vgrids.in``, or ``vgridsref.in`` if there is none yet).
    prepare
        Copy the P/S files into the Vp/Vs directory first (:func:`prepare_vpvs_dir`, i.e.
        ``cp_script``).  False: the directory already holds them (e.g. the shipped example).

    Returns
    -------
    VpVsInversionResult

    Notes
    -----
    The starting Vp/Vs is the constant mean ratio over the model below the surface layers
    (``mkvpvs``).  ``vgridvpvs.in`` is the node-wise Vp/Vs from simply dividing the Vp and Vs
    solutions of the P and S runs, for comparison.
    """
    root = Path(root)
    vd = Path(vpvs_dir) if vpvs_dir else root / VPVS_DIR
    t_start = time.perf_counter()
    logs: dict[str, list[Path]] = {}
    rows: list[dict[str, object]] = []
    dirs: dict[int, Path] = {}
    flags: list[str] = []
    state: dict[str, np.ndarray | None] = {"prev": None, "ref": None}

    def tool(name: str) -> Path:
        res = _tool(name, vd, timeout)
        logs.setdefault(name, []).append(res)
        return res

    def residuals(mode: str) -> tuple[float, float]:
        line = _inv._first_line(tool("vpvsresids").read_text(errors="replace"))
        toks = line.split()
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            vals = []
        if len(vals) != 2 or not all(math.isfinite(v) for v in vals):
            raise InversionError(f"cannot use the output of 'vpvsresids': {line!r}")
        with (vd / "vpvsresids.out").open(mode) as fh:
            fh.write(" ".join(toks) + "\n")
        return vals[0], vals[1]

    def record(it: int, rms: tuple[float, float], t0: float) -> None:
        d = vd / "iterations" / f"{it:02d}"
        d.mkdir(parents=True, exist_ok=True)
        dirs[it] = d
        cur = _read_grid_values(vd / "vgrids.in")
        if state["ref"] is None:
            state["ref"] = state["prev"] = cur
        dd = cur - state["prev"]  # type: ignore[operator]
        dr = cur - state["ref"]  # type: ignore[operator]
        fl = []
        if not np.all(np.isfinite(cur)):
            fl.append("nan_model")
        if rows and rms[0] > float(rows[-1]["rms_s"]):  # type: ignore[arg-type]
            fl.append("residual_increase")
        row: dict[str, object] = {
            "iter": it,
            "rms_s": rms[0],
            "variance_s2": rms[1],
            "n_data": _count_data(vd / "mtimes.dat"),
            "vpvs_min": float(np.nanmin(cur)),
            "vpvs_max": float(np.nanmax(cur)),
            "vpvs_mean": float(np.nanmean(cur)),
            "model_change_rms": float(np.sqrt(np.mean(dd**2))),
            "model_change_max": float(np.max(np.abs(dd))),
            "model_rms_from_ref": float(np.sqrt(np.mean(dr**2))),
            "flags": ",".join(fl),
            "wall_s": time.perf_counter() - t0,
            "dir": str(d),
        }
        state["prev"] = cur
        flags.extend(f"iteration {it}: {f}" for f in fl)
        if keep_iterations:
            for name in ("vgrids.in", "mtimes.dat"):
                _cp(vd / name, d / name)
        (d / "residuals.txt").write_text(f"{rms[0]:.9g} {rms[1]:.9g}\n")
        (d / "summary.json").write_text(json.dumps(row, indent=1, default=float))
        rows.append(row)
        if on_iteration is not None:
            on_iteration(it, row)
        if "nan_model" in fl:
            raise InversionError(f"iteration {it}: the Vp/Vs model contains NaN/Inf values")

    def finish(success: bool) -> VpVsInversionResult:
        return VpVsInversionResult(
            vpvs_dir=vd,
            iterations=pd.DataFrame(rows),
            final_model=vd / "vgrids.in" if (vd / "vgrids.in").exists() else None,
            ratio_of_vp_vs_model=vd / "vgridvpvs.in" if (vd / "vgridvpvs.in").exists() else None,
            residuals_path=vd / "vpvsresids.out",
            iteration_dirs=dirs,
            log_paths=logs,
            flags=flags,
            success=success,
            wall_s=time.perf_counter() - t_start,
        )

    try:
        if forward:
            if controls is None:
                raise InversionError("forward=True needs the controls (run settings of S)")
            sdir = root / DIR_S
            _need(sdir / "vgridsref.in", sdir / "mode_set.in")
            if not (sdir / "vgrids.in").exists():
                _cp(sdir / "vgridsref.in", sdir / "vgrids.in")
            fw = _Phase(
                "S", sdir, controls, n_jobs, False, timeout, mem_per_job_gb, timefields=False
            )
            fw.lp.forward()
            if not (sdir / "rays.dat").exists():
                raise InversionError(
                    "fm3d wrote no rays.dat: set save_rays_mode = T in mode_set.in"
                )
        if prepare:
            prepare_vpvs_dir(root, vd, controls)
        tfile = vd / "tomo3dps.in"
        default_n = controls.vpvs.tomo_run.n_iterations if controls and controls.vpvs else 1
        ni, bgiter, binv = _read_tomo_ints(tfile, (default_n, 0, 0))
        n_iter = int(ni if n_iter is None else n_iter)
        for need in ("mkvpvs.in", "frvpvs.in", "vpvsinv.in", "vpvsresids.in"):
            _need(vd / need)
        t0 = time.perf_counter()
        first = 0
        tool("mkvpvs")
        if bgiter == 0:
            _cp(vd / "vgridsref.in", vd / "vgrids.in")  # (the script's `$frech` is unset: no-op)
            (vd / "inviter.in").write_text("1\n")
        else:
            try:
                first = max(0, len((vd / "vpvsresids.out").read_text().split("\n")) - 2)
            except OSError:
                raise InversionError(f"continuing needs {vd / 'vpvsresids.out'}") from None
        if bgiter == 0 or (bgiter == 1 and binv == 1):
            tool("frvpvs")
            record(first, residuals("w"), t0)
        for k in range(1, n_iter + 1):
            t0 = time.perf_counter()
            tool("vpvsinv")
            tool("frvpvs")
            rms = residuals("a")
            (vd / "inviter.in").write_text(f"{k + 1}\n")
            record(first + k, rms, t0)
    except FmtomoFormatError as exc:
        err = InversionError(f"unreadable program output: {exc}")
        flags.append(f"error: {err}")
        if stop_on_error:
            raise err from exc
        return finish(False)
    except InversionError as exc:
        log.error("Vp/Vs inversion stopped: %s", exc)
        flags.append(f"error: {exc}")
        if stop_on_error:
            raise
        return finish(False)
    return finish(True)
