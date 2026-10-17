"""Parallel VELEST runner with one isolated run directory per trial (Stage 2).

VELEST has no command line: it opens the fixed name ``velest.cmn`` in its working
directory (``velest.f:994-999``; it stops with "control file `velest.cmn` not found"
otherwise) and takes every other file name from that control file.  Each trial therefore
gets its own directory ``<root>/<job.id>/`` holding

* ``velest.cmn`` (written by :func:`osv.io.velest.write_control`, relative file names),
* ``model.mod`` (the trial model),
* symlinks (fallback: copies) to the shared ``.cnv`` / ``.sta`` inputs,
* the outputs ``velest.OUT``, ``velout.cnv``, ``velout.sta`` (and ``velout.mod`` etc. when
  VELEST writes them), the tool logs and ``status.json`` (the completion marker).

``run_jobs`` runs the directories in parallel (a thread pool is enough: every job is a
subprocess), never aborts the batch on a failure and can resume.  This replaces the
legacy ``velestMC.sh``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from osv.io.velest import VelestControl, read_stations, write_control, write_stations
from osv.tools import ToolError, ToolNotFoundError, run_tool
from osv.velest.models import TrialModel, write_trial_models

logger = logging.getLogger(__name__)

CONTROL_NAME = "velest.cmn"
MODEL_NAME = "model.mod"
OUT_NAME = "velest.OUT"
CNV_OUT_NAME = "velout.cnv"
STA_OUT_NAME = "velout.sta"
MOD_OUT_NAME = "velout.mod"
STATUS_NAME = "status.json"

#: Files kept by ``compact=True`` (plus ``*.log`` and the VELEST outputs listed here).
KEEP_FILES = frozenset(
    {CONTROL_NAME, MODEL_NAME, OUT_NAME, CNV_OUT_NAME, STA_OUT_NAME, MOD_OUT_NAME, STATUS_NAME}
)

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class RunnerError(ValueError):
    """Invalid job definition."""


@dataclass
class VelestJob:
    """One VELEST run.

    Parameters
    ----------
    id
        Directory name under the batch root (letters, digits, ``_ . -``).
    model
        A ``.mod`` file path or a :class:`osv.velest.models.TrialModel`.
    control
        Control parameters; deep-copied per job, so the caller's object is never modified.
        File-name fields are overwritten by :func:`prepare_run_dir`.
    cnv_path, sta_path
        Shared input files (symlinked into the run directory).
    station_corrections
        Optional initial station delays ``{sta: ptcor}``.  The job then gets its own
        copy of the station file with ``ptcor`` replaced (not a symlink).
    vdamp
        Layer damping written into the model file of a ``TrialModel``.
    timeout
        Per-job timeout in seconds (overrides ``run_jobs(timeout=...)``).
    """

    id: str
    model: Path | str | TrialModel
    control: VelestControl
    cnv_path: Path | str
    sta_path: Path | str
    station_corrections: dict[str, float] | None = None
    vdamp: float | Sequence[float] = 1.0
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not _SAFE_ID.match(self.id):
            raise RunnerError(
                f"job id {self.id!r} is not a valid directory name; "
                "use letters, digits, '_', '.', '-'"
            )


def jobs_from_models(
    models: Sequence[TrialModel],
    control: VelestControl,
    cnv_path: Path | str,
    sta_path: Path | str,
    **kwargs: Any,
) -> list[VelestJob]:
    """One :class:`VelestJob` per trial model, sharing ``control`` and the inputs."""
    return [VelestJob(m.id, m, control, cnv_path, sta_path, **kwargs) for m in models]


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        logger.debug("symlink failed for %s, copying", dst)
        shutil.copy2(src, dst)


def prepare_run_dir(job: VelestJob, root: str | Path) -> Path:
    """Create ``<root>/<job.id>/`` with ``velest.cmn``, the model and linked inputs.

    Returns the run directory.  Existing files of the same names are replaced; outputs of an
    earlier run are left untouched (``run_jobs`` clears them when it re-runs a job).
    """
    cnv, sta = Path(job.cnv_path).resolve(), Path(job.sta_path).resolve()
    for p in (cnv, sta):
        if not p.is_file():
            raise FileNotFoundError(f"VELEST input file not found: {p}")
    run_dir = Path(root) / job.id
    run_dir.mkdir(parents=True, exist_ok=True)

    # model
    dst_mod = run_dir / MODEL_NAME
    if isinstance(job.model, TrialModel):
        written = write_trial_models([job.model], run_dir, job.vdamp, parquet=False)[0]
        os.replace(written, dst_mod)
    else:
        src = Path(job.model).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"VELEST model file not found: {src}")
        shutil.copyfile(src, dst_mod)

    # inputs
    ctrl = copy.deepcopy(job.control)
    cnv_name = ctrl.phasefile or cnv.name
    sta_name = ctrl.stationfile or sta.name
    _link_or_copy(cnv, run_dir / cnv_name)
    if job.station_corrections:
        table = read_stations(sta)
        if "ptcor" not in table.columns:
            table["ptcor"] = 0.0
        known = set(table["sta"])
        unknown = sorted(set(job.station_corrections) - known)
        if unknown:
            raise RunnerError(
                f"job {job.id}: station_corrections for unknown stations {unknown[:5]}; "
                f"they must appear in {sta}"
            )
        table["ptcor"] = [
            float(job.station_corrections.get(s, p))
            for s, p in zip(table["sta"], table["ptcor"], strict=True)
        ]
        dst_sta = run_dir / sta_name
        if dst_sta.is_symlink():
            dst_sta.unlink()
        write_stations(dst_sta, table)
    else:
        _link_or_copy(sta, run_dir / sta_name)

    ctrl.phasefile, ctrl.stationfile, ctrl.modelfile = cnv_name, sta_name, MODEL_NAME
    ctrl.outfile, ctrl.cnvfile, ctrl.stafile = OUT_NAME, CNV_OUT_NAME, STA_OUT_NAME
    write_control(run_dir / CONTROL_NAME, ctrl)
    return run_dir


def _read_status(run_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((run_dir / STATUS_NAME).read_text())
    except (OSError, ValueError):
        return None


def _record_paths(run_dir: Path) -> dict[str, str | None]:
    def p(name: str) -> str | None:
        f = run_dir / name
        return str(f) if f.exists() else None

    return {
        "run_dir": str(run_dir),
        "out_path": p(OUT_NAME),
        "cnv_path": p(CNV_OUT_NAME),
        "sta_path": p(STA_OUT_NAME),
        "mod_path": p(MOD_OUT_NAME),
    }


def _clear_outputs(run_dir: Path) -> None:
    for f in run_dir.glob("velout.*"):
        f.unlink()
    for name in (OUT_NAME, STATUS_NAME):
        (run_dir / name).unlink(missing_ok=True)


def compact_run_dir(run_dir: str | Path) -> None:
    """Delete everything but the control file, model, main outputs, logs and status."""
    d = Path(run_dir)
    for f in d.iterdir():
        if f.name in KEEP_FILES or f.suffix == ".log":
            continue
        if f.is_dir() and not f.is_symlink():
            shutil.rmtree(f, ignore_errors=True)
        else:
            f.unlink(missing_ok=True)


def _run_one(job: VelestJob, root: Path, timeout: float | None, force: bool, compact: bool) -> dict:
    run_dir = root / job.id
    if not force:
        prev = _read_status(run_dir)
        if prev is not None and prev.get("status") == STATUS_OK:
            prev["skipped"] = True
            return prev
    rec: dict[str, Any] = {"id": job.id, "skipped": False}
    t0 = time.monotonic()
    try:
        run_dir = prepare_run_dir(job, root)
        _clear_outputs(run_dir)
        res = run_tool(
            "velest",
            run_dir,
            timeout=job.timeout if job.timeout is not None else timeout,
            check=False,
        )
        errors = list(res.detected_errors)
        if res.timed_out:
            status = STATUS_TIMEOUT
        elif res.ok and (run_dir / OUT_NAME).is_file():
            status = STATUS_OK
        else:
            status = STATUS_FAILED
            if res.ok:
                errors.append(f"no {OUT_NAME} produced")
        rec.update(
            status=status,
            returncode=res.returncode,
            wall_time=res.wall_time,
            detected_errors=errors,
            error=None if status == STATUS_OK else "; ".join(errors) or f"rc={res.returncode}",
        )
    except ToolNotFoundError:
        raise  # a missing executable fails every job: abort loudly instead of recording it
    except (ToolError, OSError, ValueError) as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        rec.update(
            status=STATUS_FAILED,
            returncode=None,
            wall_time=time.monotonic() - t0,
            detected_errors=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    rec.update(_record_paths(run_dir))
    (run_dir / STATUS_NAME).write_text(json.dumps(rec, indent=2))
    if compact and rec["status"] == STATUS_OK:
        compact_run_dir(run_dir)
        rec.update(_record_paths(run_dir))
        (run_dir / STATUS_NAME).write_text(json.dumps(rec, indent=2))
    return rec


def _attach_parsed(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``parsed`` / ``final_rms`` using ``osv.io.velest_out.read_run_dir`` if available."""
    try:
        from osv.io.velest_out import read_run_dir  # type: ignore[import-not-found]
    except ImportError:
        df = df.copy()
        df["parsed"] = None
        df["final_rms"] = np.nan
        return df
    parsed: list[Any] = []
    rms: list[float] = []
    for r in df.itertuples():
        obj = None
        if r.status == STATUS_OK:
            try:
                obj = read_run_dir(r.run_dir)
            except Exception as exc:  # parser problems must not lose the batch
                logger.warning("could not parse %s: %s", r.run_dir, exc)
        parsed.append(obj)
        val = np.nan
        for attr in ("final_rms", "rms_final", "rms"):
            v = getattr(obj, attr, None)
            if isinstance(v, int | float | np.floating):
                val = float(v)
                break
        rms.append(val)
    df = df.copy()
    df["parsed"] = parsed
    df["final_rms"] = rms
    return df


_COLUMNS = [
    "id",
    "status",
    "returncode",
    "wall_time",
    "detected_errors",
    "error",
    "skipped",
    "run_dir",
    "out_path",
    "cnv_path",
    "sta_path",
    "mod_path",
]


def run_jobs(
    jobs: Sequence[VelestJob],
    root: str | Path,
    n_jobs: int = -1,
    timeout: float | None = 3600.0,
    progress: bool = True,
    *,
    force: bool = False,
    compact: bool = False,
    parse: bool = True,
) -> pd.DataFrame:
    """Run VELEST for every job in its own directory, in parallel.

    Parameters
    ----------
    jobs
        Jobs with unique ids.
    root
        Batch directory; job ``X`` runs in ``root/X``.
    n_jobs
        Parallel workers (``-1`` = all cores, as in ``project.n_jobs``).
    timeout
        Default per-job timeout in seconds (``None`` = no limit).
    progress
        Log progress roughly every 10 % of the jobs.
    force
        Re-run jobs even when ``status.json`` says they completed OK.  Without it
        completed jobs are skipped (resume); failed or timed-out jobs are re-run.
    compact
        After a successful run delete all but ``velest.cmn``, the model, ``velest.OUT``,
        ``velout.cnv/.sta/.mod``, logs and ``status.json``.
    parse
        Attach ``parsed`` and ``final_rms`` columns when :mod:`osv.io.velest_out` is
        importable.

    Returns
    -------
    pandas.DataFrame
        One row per job (input order): ``id, status`` (``ok``/``failed``/``timeout``),
        ``returncode, wall_time, detected_errors, error, skipped`` and the paths
        ``run_dir, out_path, cnv_path, sta_path, mod_path``.

    Raises
    ------
    ToolNotFoundError
        When the ``velest`` executable cannot be found (nothing is run).
    """
    ids = [j.id for j in jobs]
    if len(set(ids)) != len(ids):
        raise RunnerError("job ids must be unique")
    if n_jobs == 0 or n_jobs < -1:
        raise RunnerError("n_jobs must be -1 (all cores) or >= 1")
    from osv.tools import which_tool

    which_tool("velest")  # fail early, once
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    done = 0
    lock = threading.Lock()
    step = max(1, total // 10)
    t0 = time.monotonic()

    def work(job: VelestJob) -> dict:
        nonlocal done
        rec = _run_one(job, root_p, timeout, force, compact)
        with lock:
            done += 1
            if progress and (done % step == 0 or done == total):
                logger.info("VELEST runs: %d/%d done (%.0fs)", done, total, time.monotonic() - t0)
        return rec

    records = Parallel(n_jobs=n_jobs, backend="threading")(delayed(work)(j) for j in jobs)
    df = pd.DataFrame(records)
    for c in _COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[_COLUMNS]
    counts = df["status"].value_counts().to_dict()
    logger.info("VELEST batch finished: %s", counts)
    if parse:
        df = _attach_parsed(df)
    return df
