"""Stage 4 (``tradeoff`` and ``invert``): regularisation sweeps and the final FMTOMO inversion.

``tradeoff`` (``<workdir>/05_tradeoff/``, needs ``build_model``)
    Runs :func:`osv.fmtomo.tradeoff.run_tradeoff` on the Stage 3 inputs for every phase of
    ``fmtomo.phases`` (``P``; ``S`` as a separate run when Stage 3 wrote S data), picks the damping
    and smoothing at the L-curve corners (or ``tradeoff.pick: manual``) and writes the figures
    (:mod:`osv.fmtomo.plots`), the QC checks and the summary.  Files: ``tradeoff.json`` (chosen
    values, per-phase results), ``tradeoff.csv`` (all runs), ``<phase>/tradeoff.json`` and
    ``<phase>/d<damping>_s<smoothing>/`` (run directories, heavy files deleted).

``invert`` (``<workdir>/06_invert/``, needs ``tradeoff`` and ``build_model``)
    Final inversion with the chosen damping/smoothing for ``fmtomo.iterations`` iterations,
    hypocentre relocation as configured (``fmtomo.relocate``: ``none``, ``linear`` or
    ``nonlinear``), then metrics, residual and relocation tables, coverage (derivative weight sum
    of the final model, computed from the ``frechet.dat`` of the last forward run before it is
    deleted), exports, model and residual figures and the report.

Hand-off files for Stages 5 and 6 (stable names)
    ``final_model.nc``
        xarray dataset on the **full** FMTOMO grid (cushion nodes included, ``attrs["n_cushion"]``):
        ``vp`` (and ``vs`` if S was inverted), ``vp_start`` (``vs_start``), ``vp_bg`` (``vs_bg``),
        ``dws`` (``dws_vs``; derivative weight sum, 0 in the cushion) and ``mask`` (``mask_vs``;
        int8, 1 where ``dws >= plotting.dws_mask_threshold``).
    ``final_vgrids.in`` (``final_vgrids_S.in``), ``final_sources.in`` (``final_sources_S.in``)
        The final FMTOMO model and hypocentres.  For ``relocate: nonlinear`` the hypocentres are
        the FMTOMO *receivers* (``receivers.in``).
    ``residuals_final.parquet``, ``relocation.parquet``
        Per-pick residuals of the final model and the hypocentre shifts against the Stage 3 start
        (``residual_table`` and ``relocation_table`` of :mod:`osv.analysis.residuals`),
        both with a ``phase`` column.
    ``inversion_iterations.csv``
        One row per iteration and phase (misfit, model and data metrics, ray counts, velocity
        range, hypocentre shift).
    ``chosen_regularisation.json``
        The damping and smoothing used and where they came from.

Further: ``exports/`` (NetCDF, VTK, CSV), ``hypocentres_iterations.parquet`` and the run
directories ``run_<phase>/`` (heavy files removed; ``iterations/NN/`` snapshots kept).

Geometry: ``none``/``linear`` use the Stage 3 files as they are (events are FMTOMO sources,
stations receivers; linear relocation adds the a priori hypocentre uncertainty
``fmtomo.source_uncertainty_km`` to ``sources.in``).  ``nonlinear`` swaps the roles (stations
become sources, events receivers, :func:`to_nonlinear_geometry`) as the Extras ``tomo3di`` needs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.analysis import coverage as _cov
from osv.analysis.exports import export_model
from osv.analysis.residuals import relocation_table, residual_table, residual_tables
from osv.fmtomo import invert as _inv
from osv.fmtomo.controls import ModelInfo, build_controls, phase_path
from osv.fmtomo.joint import (
    JointInversionResult,
    VpVsInversionResult,
    run_joint_inversion,
    run_ps_inversions,
    run_vpvs_inversion,
)
from osv.fmtomo.metrics import attach_metrics
from osv.fmtomo.plots import (
    inversion_findings,
    make_inversion_figures,
    make_tradeoff_figures,
    tradeoff_findings,
)
from osv.fmtomo.rundir import (
    RunInputs,
    assemble_joint_run_dirs,
    assemble_run_dir,
    derive_src_dat,
)
from osv.fmtomo.tradeoff import TradeoffResult, run_tradeoff
from osv.io.fmtomo import (
    PathSpec,
    Receivers,
    Source,
    read_interfaces,
    read_observed,
    read_propgrid,
    read_receivers,
    read_sources,
    read_vgrids,
)
from osv.io.fmtomo.controls import FrechetSpec
from osv.io.fmtomo.extras import format_src_dat
from osv.io.fmtomo.frechet import read_frechet
from osv.report.checks import QCFinding
from osv.stages import Stage, StageContext, StageError, register
from osv.tools import DEFAULT_ERROR_PATTERNS, TOOL_ERROR_PATTERNS

log = logging.getLogger(__name__)

# gridloc/gridlocj stop with "ERROR!!" messages but exit code 0
for _tool_name in ("gridloc", "gridlocj"):
    TOOL_ERROR_PATTERNS.setdefault(_tool_name, (*DEFAULT_ERROR_PATTERNS, r"ERROR!!"))

__all__ = [
    "InvertStage",
    "PhaseData",
    "PhaseOutcome",
    "CoupledOutcome",
    "TradeoffStage",
    "common_event_numbers",
    "station_ids",
    "coupled_mode",
    "default_mapping",
    "invert_coupled",
    "invert_phase",
    "load_phase_data",
    "nonlinear_src_dat",
    "to_nonlinear_geometry",
]

FMTOMO_DIR = "fmtomo"
STARTING_MODEL = "starting_model.nc"
#: Stage 3 file names per phase
_FILES = {
    "P": {
        "vgrids": "vgrids.in",
        "sources": "sources.in",
        "receivers": "receivers.in",
        "otimes": "otimes_P.dat",
        "mapping": "otimes_P_mapping.csv",
        "events": "events.csv",
    },
    "S": {
        "vgrids": "vgrids_S.in",
        "sources": "sources_S.in",
        "receivers": "receivers_S.in",
        "otimes": "otimes_S.dat",
        "mapping": "otimes_S_mapping.csv",
        "events": "events_S.csv",
    },
}
#: velocity name of each phase in the model datasets
_VNAME = {"P": "vp", "S": "vs"}
#: free disk space (GB) below which a run is not started
MIN_FREE_GB = 3.0
#: station-sources of the nonlinear geometry are put this far below the surface interface (km)
SURFACE_MARGIN_KM = 0.01
KM_PER_DEG = 111.194926645
#: files deleted from a finished run directory (velocity derivatives, travel-time fields)
HEAVY_FILES = (
    "frechet.dat",
    "arrtimes.dat",
    "arrivals.dat",
    "frechet.in",
    "stimes.dat",
)
#: rays.dat is kept (Stage 5 ray plots) unless larger than this (bytes)
MAX_KEEP_RAYS_BYTES = 500_000_000


class DiskSpaceError(StageError):
    """Not enough free disk space for the next inversion run."""


def check_disk(path: Path, min_free_gb: float = MIN_FREE_GB) -> float:
    """Free space (GB) on the filesystem of ``path``; raises :class:`DiskSpaceError` if low."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free / 1e9
    if free < min_free_gb:
        raise DiskSpaceError(
            f"only {free:.1f} GB free on the disk of {path} (need {min_free_gb:g} GB): free "
            "space (delete old run directories) and re-run; finished runs are kept"
        )
    return free


# ---------------------------------------------------------------------------------------
# Stage 3 inputs
# ---------------------------------------------------------------------------------------
@dataclass
class PhaseData:
    """Run inputs of one phase built from the Stage 3 files.

    ``mapping`` has one row per row of ``inputs.otimes`` (``source_idx`` / ``receiver_idx`` index
    the FMTOMO files of this geometry).  ``events`` is the Stage 3 event table.
    """

    phase: str
    inputs: RunInputs
    mapping: pd.DataFrame
    model_info: ModelInfo
    nonlinear: bool
    events: pd.DataFrame | None = None
    stations: pd.DataFrame | None = None


def default_mapping(otimes: pd.DataFrame, phase: str = "P") -> pd.DataFrame:
    """A pick mapping for data without one (synthetic tests): ids are the FMTOMO indices."""
    o = otimes.reset_index(drop=True)
    return pd.DataFrame(
        {
            "source_idx": o["source"].astype(int),
            "receiver_idx": o["receiver"].astype(int),
            "path_idx": 1,
            "ray_idx": o["ray"].astype(int),
            "event_id": o["source"].astype(int).astype(str),
            "station": o["receiver"].astype(int).astype(str),
            "phase": phase,
            "pick_id": np.arange(len(o)).astype(str),
            "time": o["time"].astype(float),
            "uncertainty": o["uncertainty"].astype(float),
            "valid": o["time"].to_numpy(float) >= -50.0,
        }
    )


def _surface_depth(interfaces: Any, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    da = interfaces.data["interface_depth"].sel(interface=1)
    pts = xr.Dataset({"lat": ("p", lat), "lon": ("p", lon)})
    return da.interp(lat=pts["lat"], lon=pts["lon"], kwargs={"fill_value": None}).to_numpy()


def to_nonlinear_geometry(
    inputs: RunInputs, mapping: pd.DataFrame, phase: str = "P"
) -> tuple[RunInputs, pd.DataFrame]:
    """Swap sources and receivers for the Extras ``tomo3di`` driver (nonlinear relocation).

    In ``tomo3di`` the stations are the FMTOMO sources (one direct path each, placed at least
    :data:`SURFACE_MARGIN_KM` below the top interface) and ``gridloc`` moves the events.  The
    Extras file layout has **one receiver record per observation**, each at the position of its
    event with a single ray from the station (``gridloc.f90`` rewrites every record of an
    event); the observations are ordered station-major, then by event, which is also the order
    of ``src.dat`` and of the ``fm3d`` arrivals.  Invalid picks (time <= 0) are dropped.

    Returns
    -------
    (RunInputs, mapping)
        ``mapping`` follows the new ``otimes`` rows (``source_idx`` = station, ``receiver_idx`` =
        observation record) and has an extra ``event_idx`` column (old source id).
    """
    st_pos = inputs.receivers.positions  # station positions, index = old receiver id
    ev = inputs.sources  # events, 1-based
    surf = _surface_depth(
        inputs.interfaces, st_pos["lat"].to_numpy(float), st_pos["lon"].to_numpy(float)
    )
    depth = np.maximum(st_pos["depth_km"].to_numpy(float), np.nan_to_num(surf) + SURFACE_MARGIN_KM)
    path = phase_path(phase)
    new_sources = [
        Source(float(d), float(la), float(lo), [PathSpec(path.steps, path.vtypes)])
        for d, la, lo in zip(depth, st_pos["lat"], st_pos["lon"], strict=True)
    ]
    m = mapping.reset_index(drop=True)
    o = inputs.otimes.reset_index(drop=True)
    if len(m) != len(o):
        raise ValueError(f"mapping ({len(m)} rows) and otimes ({len(o)} rows) differ")
    keep = o["time"].to_numpy(float) > 0
    nm = m[keep].copy()
    nm["event_idx"] = m.loc[keep, "source_idx"].to_numpy()
    nm["source_idx"] = m.loc[keep, "receiver_idx"].to_numpy()  # station
    nm = nm.sort_values(["source_idx", "event_idx"], kind="stable").reset_index(drop=True)
    n = len(nm)
    nm["receiver_idx"] = np.arange(1, n + 1)
    nm["ray_idx"] = 1
    nm["path_idx"] = 1
    pos = pd.DataFrame(
        {
            "depth_km": [ev[i - 1].depth_km for i in nm["event_idx"]],
            "lat": [ev[i - 1].lat for i in nm["event_idx"]],
            "lon": [ev[i - 1].lon for i in nm["event_idx"]],
        },
        index=pd.RangeIndex(1, n + 1, name="receiver"),
    )
    rays = pd.DataFrame(
        {
            "receiver": nm["receiver_idx"].to_numpy(),
            "source": nm["source_idx"].to_numpy(),
            "path": 1,
        },
        dtype="int64",
    )
    no = pd.DataFrame(
        {
            "receiver": nm["receiver_idx"].to_numpy(),
            "source": nm["source_idx"].to_numpy(),
            "ray": 1,
            "subray": 0,
            "time": nm["time"].to_numpy(float),
            "uncertainty": nm["uncertainty"].to_numpy(float),
        }
    )
    new = RunInputs(
        inputs.grids,
        inputs.interfaces,
        inputs.propgrid,
        new_sources,
        Receivers(pos, rays),
        no,
    )
    return new, nm


def nonlinear_src_dat(otimes: pd.DataFrame, mapping: pd.DataFrame, n_sources: int) -> pd.DataFrame:
    """``src.dat`` of the ``tomo3di`` geometry: one block per FMTOMO source (station).

    Each block lists the valid picks as ``id`` = event number (``mapping["event_idx"]`` of
    :func:`to_nonlinear_geometry`), time and uncertainty, as ``gridloc`` reads them
    (``READ(20,*) idtto, tto(i, idtto)``); a thin wrapper of
    :func:`osv.fmtomo.rundir.derive_src_dat`.
    """
    return derive_src_dat(otimes, n_sources, mapping["event_idx"].to_numpy())


def _with_uncertainty(sources: list[Source], km: float) -> list[Source]:
    out = []
    for s in sources:
        dlat = km / KM_PER_DEG
        dlon = km / (KM_PER_DEG * max(math.cos(math.radians(s.lat)), 1e-3))
        out.append(dataclasses.replace(s, uncertainty=(km, dlat, dlon)))
    return out


def load_phase_data(stage3_dir: str | Path, phase: str, fm: Any) -> PhaseData:
    """Read the Stage 3 hand-off files of ``phase`` and apply the relocation geometry of ``fm``.

    ``fm`` is the config's ``fmtomo`` section (``relocate``, ``source_uncertainty_km``).
    """
    d = Path(stage3_dir) / FMTOMO_DIR
    names = _FILES[phase]
    for key in ("vgrids", "sources", "receivers", "otimes", "mapping"):
        if not (d / names[key]).exists():
            raise StageError(f"Stage 3 has no {d / names[key]}: run `osv build-model` first")
    grids = read_vgrids(d / names["vgrids"])
    sources = read_sources(d / names["sources"])
    receivers = read_receivers(d / names["receivers"])
    inputs = RunInputs(
        grids,
        read_interfaces(d / "interfaces.in"),
        read_propgrid(d / "propgrid.in"),
        sources,
        receivers,
        read_observed(d / names["otimes"]),
    )
    mapping = pd.read_csv(d / names["mapping"])
    events = pd.read_csv(d / names["events"]) if (d / names["events"]).exists() else None
    stations = pd.read_csv(d / "stations.csv") if (d / "stations.csv").exists() else None
    relocate = fm.relocate
    if relocate == "nonlinear":
        inputs, mapping = to_nonlinear_geometry(inputs, mapping, phase)
    elif relocate == "linear":
        inputs = dataclasses.replace(
            inputs, sources=_with_uncertainty(sources, float(fm.source_uncertainty_km))
        )
    info = ModelInfo.from_model(grids, sources=inputs.sources, receivers=inputs.receivers)
    return PhaseData(phase, inputs, mapping, info, relocate == "nonlinear", events, stations)


def _phases(cfg: Any, stage3_dir: Path) -> list[str]:
    ph = set(cfg.fmtomo.phases)
    want = ["P"] + (["S"] if {"S", "vpvs"} & ph else [])
    if "P" not in ph and "vpvs" not in ph:
        want = [p for p in want if p in ph]
    have = [p for p in want if (stage3_dir / FMTOMO_DIR / _FILES[p]["otimes"]).exists()]
    if not have:
        raise StageError("fmtomo.phases selects no phase with data in Stage 3")
    return have


def _phase_cfg(fm: Any, phase: str, **update: Any) -> Any:
    """Copy of the ``fmtomo`` config restricted to one phase (separate P and S runs)."""
    return fm.model_copy(update={"phases": [phase], **update})


# ---------------------------------------------------------------------------------------
# one inversion
# ---------------------------------------------------------------------------------------
@dataclass
class PhaseOutcome:
    """Everything :func:`invert_phase` extracted from a finished inversion."""

    phase: str
    run_dir: Path
    result: _inv.InversionResult
    iterations: pd.DataFrame
    mapping: pd.DataFrame
    nonlinear: bool
    dws: xr.DataArray | None = None
    mask: xr.DataArray | None = None
    residuals: pd.DataFrame | None = None
    relocation: pd.DataFrame | None = None
    hypocentres: pd.DataFrame | None = None
    damping: float = math.nan
    smoothing: float = math.nan
    notes: list[str] = field(default_factory=list)


def _dws(run: Path, threshold: float) -> tuple[xr.DataArray, xr.DataArray]:
    """Derivative weight sum of the velocity grid from the run's ``frechet.dat``."""
    spec = FrechetSpec.read(run / "frechet.in")
    grids = read_vgrids(run / "vgridsref.in")
    layout = _cov.velocity_layout(spec, grids)
    fr = read_frechet(run / "frechet.dat")
    dws = _cov.derivative_weight_sum(fr, layout, grids, layer=1, vtype=1)
    return dws, _cov.coverage_mask(dws, threshold)


def _per_event(
    df: pd.DataFrame, mapping: pd.DataFrame, col: str = "event_idx", *by: str
) -> pd.DataFrame:
    """Collapse the per-observation receiver records of the nonlinear geometry to one row per event.

    ``col`` holds the receiver record number; it is replaced by the event number
    (``mapping["event_idx"]``), the first record of each event is kept and ``n_picks`` summed.
    """
    ev = mapping.drop_duplicates("receiver_idx").set_index("receiver_idx")["event_idx"]
    out = df.copy()
    out[col] = out[col].map(ev)
    out = out[out[col].notna()]
    out[col] = out[col].astype(int)
    if "n_picks" in out:
        out["n_picks"] = out.groupby([*by, col])["n_picks"].transform("sum")
    return out.drop_duplicates([*by, col]).reset_index(drop=True)


def _clean_run(run: Path, keep_rays: bool) -> None:
    for name in HEAVY_FILES:
        (run / name).unlink(missing_ok=True)
    rays = run / "rays.dat"
    if rays.exists() and not (keep_rays and rays.stat().st_size <= MAX_KEEP_RAYS_BYTES):
        rays.unlink()
    for p in [run / "parallel", *run.glob("fm3d_*"), *run.glob("group*")]:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for p in (run / "iterations").glob("*/*"):
        if p.name in ("frechet.dat", "rays.dat", "arrtimes.dat", "arrivals.dat"):
            p.unlink(missing_ok=True)


def invert_phase(
    run_dir: str | Path,
    data: PhaseData | RunInputs,
    fm: Any,
    damping: float,
    smoothing: float,
    n_jobs: int = 1,
    *,
    phase: str = "P",
    mapping: pd.DataFrame | None = None,
    n_iter: int | None = None,
    dws_threshold: float = 10.0,
    keep_rays: bool | None = None,
    timeout: float | None = None,
    mem_per_job_gb: float | None = None,
    clean: bool = True,
) -> PhaseOutcome:
    """Assemble and run one phase's final inversion and extract its results.

    The steps are: assemble the run directory, run the iteration loop, attach the model metrics,
    compute the derivative weight sum from the final ``frechet.dat`` (the velocity derivatives of
    the last forward run, i.e. of the final model), build the residual and relocation tables of
    the final iteration and delete the heavy files (``clean``).

    Parameters
    ----------
    run_dir
        Directory to create (an existing one is replaced).
    data
        :class:`PhaseData` (from :func:`load_phase_data`) or :class:`~osv.fmtomo.rundir.RunInputs`
        (``mapping`` is then optional and synthesised).
    fm
        ``cfg.fmtomo`` (relocation, subspace dimension, iterations, ...).
    damping, smoothing
        Regularisation of the inversion.
    n_iter
        Iterations (default ``fm.iterations``).
    dws_threshold
        Absolute DWS threshold of the coverage mask (``plotting.dws_mask_threshold``).
    keep_rays
        Keep ``rays.dat`` (default ``fm.save_rays``).
    """
    run = Path(run_dir)
    if isinstance(data, RunInputs):
        mp = mapping if mapping is not None else default_mapping(data.otimes, phase)
        pd_ = PhaseData(
            phase,
            data,
            mp,
            ModelInfo.from_model(data.grids, sources=data.sources, receivers=data.receivers),
            fm.relocate == "nonlinear",
        )
    else:
        pd_ = data
    check_disk(run.parent)
    shutil.rmtree(run, ignore_errors=True)
    fm_run = _phase_cfg(fm, phase, damping=float(damping), smoothing=float(smoothing))
    ctl = build_controls(fm_run, pd_.model_info)
    i = pd_.inputs
    assemble_run_dir(
        run,
        i.grids,
        i.interfaces,
        i.propgrid,
        i.sources,
        i.receivers,
        i.otimes,
        ctl,
        phase=phase,
        src_dat=(
            nonlinear_src_dat(i.otimes, pd_.mapping, len(i.sources)) if pd_.nonlinear else None
        ),
    )
    pd_.mapping.to_csv(run / "mapping.csv", index=False)
    n_iter = int(fm.iterations if n_iter is None else n_iter)
    res = _inv.run_inversion(
        run,
        ctl,
        n_iter,
        n_jobs=n_jobs,
        keep_iterations=True,
        phase=phase,
        timeout=timeout,
        mem_per_job_gb=mem_per_job_gb,
        stop_on_error=True,
    )
    return _finish_phase(
        run,
        res,
        pd_,
        float(damping),
        float(smoothing),
        dws_threshold,
        bool(fm.save_rays if keep_rays is None else keep_rays),
        clean,
    )


def _finish_phase(
    run: Path,
    res: _inv.InversionResult,
    pd_: PhaseData,
    damping: float,
    smoothing: float,
    dws_threshold: float,
    keep_rays: bool,
    clean: bool,
) -> PhaseOutcome:
    """Metrics, coverage, residual and relocation tables of a finished run directory."""
    phase = pd_.phase
    out = PhaseOutcome(
        phase,
        run,
        res,
        res.iterations,
        pd_.mapping,
        pd_.nonlinear,
        damping=damping,
        smoothing=smoothing,
    )
    try:
        out.iterations = attach_metrics(res)
    except Exception as exc:  # noqa: BLE001 - metrics are diagnostics, keep the inversion
        log.warning("model metrics failed: %s", exc)
        out.notes.append(f"metrics failed: {exc}")
    if (run / "frechet.dat").exists() and (run / "frechet.in").exists():
        try:
            out.dws, out.mask = _dws(run, dws_threshold)
        except Exception as exc:  # noqa: BLE001
            log.warning("DWS failed: %s", exc)
            out.notes.append(f"coverage failed: {exc}")
    else:
        out.notes.append("no frechet.dat in the final run: no coverage")
    nl = pd_.nonlinear
    out.residuals = residual_table(run, pd_.mapping, "final", nonlinear=nl)
    out.relocation = relocation_table(run, pd_.mapping, "final", nonlinear=nl)
    out.hypocentres = res.hypocentres
    if nl:
        out.relocation = _per_event(out.relocation, pd_.mapping)
        out.hypocentres = _per_event(res.hypocentres, pd_.mapping, "source", "iter")
    if clean:
        _clean_run(run, keep_rays)
    return out


# ---------------------------------------------------------------------------------------
# coupled P+S and Vp/Vs inversions (Extras tomo3dj / tomo3dps)
# ---------------------------------------------------------------------------------------
#: directory of the coupled run (``invert_p/``, ``invert_s/``, ``src.dat``, ``vpvs/``) in the stage
#: directory; ``invert_p``/``invert_s`` are moved to ``run_P``/``run_S`` when the run is done
COUPLED_DIR = "run_joint"


def coupled_mode(fm: Any, phases: list[str]) -> str | None:
    """How P and S are coupled for ``fm`` and the phases that have data.

    ``None``: separate runs (``[P]``, ``[S]`` or ``[P, S]`` with ``joint.relocation`` independent).
    Otherwise the relocation mode of the coupled run: ``joint`` (``tomo3dj``), ``p_fixed_for_s``,
    or, for ``phases: [vpvs]``, the mode followed by the Vp/Vs inversion (``tomo3dps``), see
    :func:`coupled_vpvs`.

    Raises
    ------
    StageError
        If the coupled run needs data or settings that are missing.
    """
    ph = list(fm.phases)
    vpvs = "vpvs" in ph
    mode = fm.joint.relocation or ("joint" if vpvs else "independent")
    if not vpvs and (set(ph) != {"P", "S"} or mode == "independent"):
        return None
    if set(phases) != {"P", "S"}:
        raise StageError(
            "P+S coupled / Vp/Vs inversion needs Stage 3 data of both phases (S picks missing); "
            "run `osv build-model` with fmtomo.phases containing S"
        )
    if fm.relocate != "nonlinear":
        raise StageError(
            "joint P+S and Vp/Vs inversions need fmtomo.relocate: nonlinear (stations as sources)"
        )
    if fm.package != "extras":
        raise StageError("joint P+S and Vp/Vs inversions need fmtomo.package: extras")
    return mode


def coupled_vpvs(fm: Any) -> bool:
    """True when the Vp/Vs inversion follows the P and S inversions (``phases: [vpvs]``)."""
    return "vpvs" in list(fm.phases)


def common_event_numbers(mappings: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Event numbers shared by the P and S data (per row of each phase's mapping).

    Stage 3 numbers the events of each phase separately (events without picks of the phase are
    dropped), but the P/S association file of the Extras programs (``jointimes``) identifies the
    same event by the same number.  P events keep their numbers, S events are matched by
    ``event_id``; S-only events get new numbers after the P ones.
    """
    num: dict[str, int] = {}
    for ph in ("P", "S"):
        m = mappings[ph].drop_duplicates("event_idx").sort_values("event_idx")
        for eid in m["event_id"].astype(str):
            num.setdefault(eid, len(num) + 1)
    return {
        ph: m["event_id"].astype(str).map(num).to_numpy(dtype="int64") for ph, m in mappings.items()
    }


def station_ids(datas: dict[str, PhaseData]) -> dict[str, list[int]]:
    """Receiver ids of the Extras ``sources.in`` (fourth column) of the P and S station-sources.

    ``gridlocj`` forms S-P differences between the P and S pick of the same station: the id of a
    P station is its own number, the id of an S station is the number of the P station of the
    same name, or a number above the count of P stations (a station with S picks only).  Stations
    are matched by the ``station`` column of the pick mappings (:func:`to_nonlinear_geometry`).
    """

    def names(d: PhaseData) -> dict[int, str]:
        m = d.mapping.drop_duplicates("source_idx")
        return dict(zip(m["source_idx"].astype(int), m["station"].astype(str), strict=True))

    n_p = len(datas["P"].inputs.sources)
    p_index = {name: i for i, name in names(datas["P"]).items()}
    s_names = names(datas["S"])
    return {
        "P": list(range(1, n_p + 1)),
        "S": [
            p_index.get(s_names.get(j, ""), n_p + j)
            for j in range(1, len(datas["S"].inputs.sources) + 1)
        ],
    }


@dataclass
class CoupledOutcome:
    """Results of :func:`invert_coupled`: per-phase outcomes and the coupled-run results."""

    mode: str
    root: Path
    outcomes: dict[str, PhaseOutcome]
    joint: JointInversionResult | None = None
    vpvs: VpVsInversionResult | None = None
    vpvs_model: xr.DataArray | None = None
    notes: list[str] = field(default_factory=list)


def invert_coupled(
    out_dir: str | Path,
    datas: dict[str, PhaseData],
    fm: Any,
    reg: dict[str, dict[str, Any]],
    n_jobs: int = 1,
    *,
    mode: str,
    vpvs: bool = False,
    n_iter: int | None = None,
    dws_threshold: float = 10.0,
    clean: bool = True,
    timeout: float | None = None,
    mem_per_job_gb: float | None = None,
) -> CoupledOutcome:
    """Run the P and S inversions coupled through the Extras programs, optionally plus Vp/Vs.

    ``mode`` is ``joint`` (:func:`osv.fmtomo.joint.run_joint_inversion`: coupled ``gridlocj``
    relocation, P and S velocity inversions each iteration) or ``p_fixed_for_s`` (P first with its
    relocation, its hypocentres fixed in the S run; :func:`~osv.fmtomo.joint.run_ps_inversions`).
    With ``vpvs`` the direct Vp/Vs inversion (:func:`~osv.fmtomo.joint.run_vpvs_inversion`,
    ``tomo3dps``) follows on the final P and S models.  Both geometries are the nonlinear
    (stations as sources) ones from :func:`load_phase_data`; ``fm.relocate`` must be ``nonlinear``.

    The run directory is ``<out_dir>/run_joint`` (``src.dat``, ``vpvs/``); the phase
    directories are moved to ``<out_dir>/run_P`` and ``run_S`` (the names Stages 4-6 use) and
    post-processed as in :func:`invert_phase`.  ``reg`` holds the damping and smoothing per phase;
    the Vp/Vs inversion uses ``fm.damping`` and ``fm.smoothing`` (it is not part of the trade-off).
    """
    from osv.fmtomo.controls import DIR_P, DIR_S

    if mode not in ("joint", "p_fixed_for_s"):
        raise StageError(f"unknown coupled relocation mode {mode!r}")
    out_dir = Path(out_dir)
    root = out_dir / COUPLED_DIR
    check_disk(out_dir)
    for d in (root, out_dir / "run_P", out_dir / "run_S"):
        shutil.rmtree(d, ignore_errors=True)
    n_iter = int(fm.iterations if n_iter is None else n_iter)
    update: dict[str, Any] = {"phases": ["P", "S"]}
    if vpvs:
        update["save_rays"] = True  # the Vp/Vs inversion reads the S rays
    ctls = {
        ph: build_controls(
            fm.model_copy(
                update={
                    **update,
                    "damping": float(reg[ph]["damping"]),
                    "smoothing": float(reg[ph]["smoothing"]),
                }
            ),
            datas[ph].model_info,
        )
        for ph in ("P", "S")
    }
    ctl = dataclasses.replace(ctls["P"], runs={"P": ctls["P"].runs["P"], "S": ctls["S"].runs["S"]})
    sid = station_ids(datas)
    inputs = {
        ph: dataclasses.replace(
            datas[ph].inputs,
            sources=[
                dataclasses.replace(src, extra_id=k)
                for src, k in zip(datas[ph].inputs.sources, sid[ph], strict=True)
            ],
        )
        for ph in ("P", "S")
    }
    assemble_joint_run_dirs(root, inputs, ctl)
    ids = common_event_numbers({ph: datas[ph].mapping for ph in ("P", "S")})
    for ph, sub in (("P", DIR_P), ("S", DIR_S)):
        d, i = datas[ph], datas[ph].inputs
        sd = nonlinear_src_dat(i.otimes, d.mapping.assign(event_idx=ids[ph]), len(i.sources))
        (root / sub / "src.dat").write_text(format_src_dat(sd))
        d.mapping.to_csv(root / sub / "mapping.csv", index=False)
    kw: dict[str, Any] = {"timeout": timeout, "mem_per_job_gb": mem_per_job_gb}
    jres: JointInversionResult | None = None
    if mode == "joint":
        jres = run_joint_inversion(root, ctl, n_iter, n_jobs, True, **kw)
        res = {"P": jres.p, "S": jres.s}
        notes = list(jres.flags)
    else:
        pres = run_ps_inversions(
            root / DIR_P, root / DIR_S, ctl, n_iter, n_jobs, True, relocation=mode, **kw
        )
        res = {"P": pres.p, "S": pres.s}
        notes = list(pres.notes)
    vres: VpVsInversionResult | None = None
    vmodel: xr.DataArray | None = None
    if vpvs:
        vfm = fm.model_copy(update={"phases": ["vpvs"], "relocate": "none", "save_rays": True})
        vctl = build_controls(vfm, datas["P"].model_info)
        vres = run_vpvs_inversion(root, vctl, n_iter, True, forward=True, n_jobs=n_jobs, **kw)
        if vres.final_model is not None:
            vmodel = (
                read_vgrids(vres.final_model)
                .get(1, 1)["velocity"]
                .transpose("depth", "lat", "lon")
                .rename("vpvs")
            )
        notes.extend(vres.flags)
    outcomes: dict[str, PhaseOutcome] = {}
    for ph, sub in (("P", DIR_P), ("S", DIR_S)):
        oc = _finish_phase(
            root / sub,
            res[ph],
            datas[ph],
            float(reg[ph]["damping"]),
            float(reg[ph]["smoothing"]),
            dws_threshold,
            bool(fm.save_rays),
            clean,
        )
        outcomes[ph] = oc
    for ph, sub in (("P", DIR_P), ("S", DIR_S)):  # the names the later stages use
        run = out_dir / f"run_{ph}"
        shutil.move(root / sub, run)
        outcomes[ph].run_dir = run
        outcomes[ph].result.run_dir = run
    if clean and (root / "vpvs").is_dir():
        _clean_run(root / "vpvs", False)
    return CoupledOutcome(mode, root, outcomes, jres, vres, vmodel, notes)


# ---------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------
def _clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | np.ndarray):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _median_event_depth(stage3: Path) -> float | None:
    p = stage3 / FMTOMO_DIR / "events.csv"
    try:
        return float(pd.read_csv(p)["depth_km_bsl"].median())
    except (OSError, KeyError, ValueError):
        return None


def _invalidate_stale_sweeps(stage_dir: Path, upstream_run_id: str, fm: Any) -> None:
    """Delete cached sweep runs made from another Stage 3 run or other inversion settings.

    The per-run ``summary.json`` only records damping, smoothing and iterations, so without this
    a re-run of ``build_model`` (new data, new 1-D model) would silently reuse old sweeps.
    Interrupted sweeps of the same inputs still resume.
    """
    import hashlib

    settings = fm.model_dump(mode="json", exclude={"tradeoff"})
    stamp = hashlib.sha256(
        json.dumps([upstream_run_id, settings], sort_keys=True, default=str).encode()
    ).hexdigest()
    f = stage_dir / ".sweep_inputs"
    if f.exists() and f.read_text().strip() == stamp:
        return
    for d in stage_dir.iterdir() if stage_dir.exists() else []:
        if d.is_dir() and d.name not in ("figures", "logs"):
            shutil.rmtree(d, ignore_errors=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(stamp)


@register
class TradeoffStage(Stage):
    """Stage 4a: damping and smoothing sweeps and L-curve corner picking."""

    name = "tradeoff"
    requires = ("build_model",)
    config_sections = ("project.random_seed", "fmtomo")
    tools = ("fm3d", "frechgen", "invert3d", "residuals")
    outputs = ("tradeoff.json", "tradeoff.csv")

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        fm = cfg.fmtomo
        s3 = ctx.upstream("build_model")
        findings: list[QCFinding] = []
        chosen: dict[str, dict[str, float]] = {}
        results: dict[str, TradeoffResult] = {}
        tables = []
        _invalidate_stale_sweeps(ctx.stage_dir, s3.manifest.get("run_id", ""), fm)
        for ph in _phases(cfg, s3.dir):
            data = load_phase_data(s3.dir, ph, fm)
            # rays are not needed in the sweeps; saving them costs time and disk
            fm_ph = _phase_cfg(fm, ph, save_rays=False)
            wd = ctx.stage_dir / ph
            ctx.logger.info(
                "trade-off %s: damping %s, smoothing %s, %d iterations each",
                ph,
                list(fm.tradeoff.damping),
                list(fm.tradeoff.smoothing),
                fm.tradeoff.iterations,
            )
            check_disk(ctx.stage_dir)

            def after_run(row: dict[str, Any], wd: Path = wd) -> None:
                ctx.logger.info(
                    "run d=%g s=%g: rms %.1f ms, success %s",
                    row["damping"],
                    row["smoothing"],
                    1000 * _f(row.get("rms_s")),
                    row.get("success"),
                )
                check_disk(wd)

            res = run_tradeoff(
                data.inputs,
                fm_ph,
                wd,
                ctx.n_jobs,
                model_info=data.model_info,
                assemble_kwargs={
                    "phase": ph,
                    "src_dat": (
                        nonlinear_src_dat(
                            data.inputs.otimes, data.mapping, len(data.inputs.sources)
                        )
                        if data.nonlinear
                        else None
                    ),
                },
                on_run=after_run,
            )
            results[ph] = res
            chosen[ph] = {"damping": res.chosen_damping, "smoothing": res.chosen_smoothing}
            t = res.table.copy()
            t.insert(0, "phase", ph)
            tables.append(t)
            findings.extend(tradeoff_findings(res.to_dict(), ph if len(results) > 1 else ""))
            for fl in res.flags:
                ctx.logger.warning("%s: %s", ph, fl)
        first = next(iter(chosen))
        top = {
            "chosen_damping": chosen[first]["damping"],
            "chosen_smoothing": chosen[first]["smoothing"],
            "chosen": chosen,
            "phases": list(results),
            "mode": fm.tradeoff.mode,
            "pick": fm.tradeoff.pick,
            "results": {ph: r.to_dict() for ph, r in results.items()},
        }
        (ctx.stage_dir / "tradeoff.json").write_text(json.dumps(_clean_json(top), indent=1))
        pd.concat(tables, ignore_index=True).to_csv(ctx.stage_dir / "tradeoff.csv", index=False)
        for f in findings:
            ctx.add_check(f)
        s = ctx.summary
        for ph, r in results.items():
            s[f"chosen_damping_{ph}"] = r.chosen_damping
            s[f"chosen_smoothing_{ph}"] = r.chosen_smoothing
            s[f"damping_source_{ph}"] = r.picks["damping"]["source"]
            s[f"smoothing_source_{ph}"] = r.picks["smoothing"]["source"]
            s[f"runs_{ph}"] = int(len(r.table))
        s["mode"] = fm.tradeoff.mode
        s["iterations_per_run"] = int(fm.tradeoff.iterations)
        self.plot(ctx)

    def plot(self, ctx: StageContext) -> None:
        s3 = ctx.project_dir / "04_build_model"
        top = json.loads((ctx.stage_dir / "tradeoff.json").read_text())
        depth = _median_event_depth(s3)
        multi = len(top["phases"]) > 1
        for ph in top["phases"]:
            res = top["results"][ph]
            make_tradeoff_figures(
                ctx.figures.save,
                res,
                ctx.stage_dir / ph,
                depth_km=depth,
                suffix=f"_{ph}" if multi else "",
                label=f"{ph}: " if multi else "",
            )


def _chosen_values(
    cfg: Any, stage_dir: Path | None, phases: list[str]
) -> dict[str, dict[str, Any]]:
    """Damping and smoothing per phase: from ``tradeoff.json`` or the config values."""
    fm = cfg.fmtomo
    out: dict[str, dict[str, Any]] = {
        p: {"damping": float(fm.damping), "smoothing": float(fm.smoothing), "source": "config"}
        for p in phases
    }
    tj = stage_dir / "tradeoff.json" if stage_dir is not None else None
    if tj is not None and tj.exists():
        top = json.loads(tj.read_text())
        for p in phases:
            c = top.get("chosen", {}).get(p)
            if c is None and p == phases[0]:
                c = {"damping": top["chosen_damping"], "smoothing": top["chosen_smoothing"]}
            if c is not None:
                src = top.get("results", {}).get(p, {}).get("picks", {})
                out[p] = {
                    "damping": float(c["damping"]),
                    "smoothing": float(c["smoothing"]),
                    "source": "tradeoff "
                    f"({src.get('damping', {}).get('source', top.get('pick', ''))}"
                    f"/{src.get('smoothing', {}).get('source', top.get('pick', ''))})",
                }
    return out


def _vpvs_findings(res: VpVsInversionResult) -> list[QCFinding]:
    """QC of the Vp/Vs inversion: failure, no misfit reduction, flagged iterations."""
    out: list[QCFinding] = []
    if not res.success:
        out.append(QCFinding("vpvs_failed", "fail", "the Vp/Vs inversion stopped early"))
    if len(res.iterations) > 1 and res.final_rms_s >= res.initial_rms_s:
        out.append(
            QCFinding(
                "vpvs_misfit_not_decreasing",
                "warn",
                f"S-P misfit {res.initial_rms_s:.4f} s -> {res.final_rms_s:.4f} s: the Vp/Vs "
                "inversion did not reduce it (damping too strong, or too few S-P pairs)",
            )
        )
    for f in res.flags:
        if not f.startswith("error"):
            out.append(QCFinding("vpvs_note", "warn", f"Vp/Vs {f}"))
    return out


def _full_grid(da: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Pad an interior-grid array with zeros onto the full grid of ``like`` (by position)."""
    arr = da.transpose("depth", "lat", "lon").to_numpy()
    nc = [(a - b) // 2 for a, b in zip(like.shape, arr.shape, strict=True)]
    if any(2 * c != a - b for c, a, b in zip(nc, like.shape, arr.shape, strict=True)):
        raise StageError(f"coverage grid {arr.shape} does not fit the model grid {like.shape}")
    full = np.pad(arr, [(c, c) for c in nc])
    return xr.DataArray(full, dims=("depth", "lat", "lon"), coords=like.coords)


def _write_model(
    path: Path,
    start: xr.Dataset,
    outcomes: dict[str, PhaseOutcome],
    threshold: float,
    vpvs: xr.DataArray | None = None,
) -> xr.Dataset:
    """``final_model.nc`` on the full grid (see the module docstring).

    ``vpvs`` (Vp/Vs inversion) is stored as ``vpvs``; without it Stage 5 derives ``vp / vs``.
    """
    ds = start.copy()
    names = ["vp", "vp_bg"] + (["vs", "vs_bg"] if "S" in outcomes else [])
    keep = [v for v in names if v in ds]
    out = ds[keep].copy()
    out.attrs = {
        k: ("\n".join(map(str, v)) if isinstance(v, list | tuple) else v)
        for k, v in start.attrs.items()
    }
    for v in ("vp", "vs"):
        if v in out:
            out[f"{v}_start"] = start[v]
    for ph, oc in outcomes.items():
        v = _VNAME[ph]
        fin = read_vgrids(oc.run_dir / "vgrids.in").get(1, 1)["velocity"]
        shape = tuple(out[v].shape)
        arr = fin.transpose("depth", "lat", "lon").to_numpy()
        if arr.shape != shape:
            raise StageError(f"final {ph} grid {arr.shape} differs from the starting grid {shape}")
        out[v] = (("depth", "lat", "lon"), arr, {"units": "km/s", "long_name": f"{v} (final)"})
        sfx = "" if ph == "P" else "_vs"
        if oc.dws is not None:
            out[f"dws{sfx}"] = _full_grid(oc.dws, out[v]).assign_attrs(
                units="s/(km/s)", long_name=f"derivative weight sum of {v}"
            )
            out[f"mask{sfx}"] = (
                (out[f"dws{sfx}"] >= threshold)
                .astype("int8")
                .assign_attrs(
                    long_name=f"{v} coverage mask (dws >= {threshold:g})", flag_values="0 1"
                )
            )
    if vpvs is not None:
        arr = vpvs.to_numpy()
        if arr.shape != tuple(out["vp"].shape):
            raise StageError(f"Vp/Vs grid {arr.shape} differs from the model grid")
        out["vpvs"] = (
            ("depth", "lat", "lon"),
            arr,
            {"units": "1", "long_name": "Vp/Vs ratio (direct inversion of S-P times)"},
        )
    out.attrs.update(dws_threshold=float(threshold), n_cushion=int(start.attrs.get("n_cushion", 1)))
    out.to_netcdf(path)
    return out


@register
class InvertStage(Stage):
    """Stage 4b: final FMTOMO inversion with the chosen regularisation."""

    name = "invert"
    requires = ("tradeoff", "build_model")
    config_sections = ("project.random_seed", "fmtomo", "plotting.dws_mask_threshold")
    tools = ("fm3d", "frechgen", "invert3d", "residuals")
    outputs = (
        "final_model.nc",
        "final_vgrids.in",
        "final_sources.in",
        "residuals_final.parquet",
        "relocation.parquet",
        "inversion_iterations.csv",
        "chosen_regularisation.json",
    )

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        fm = cfg.fmtomo
        s3 = ctx.upstream("build_model")
        try:
            tdir: Path | None = ctx.upstream("tradeoff").dir
        except StageError:
            tdir = None
        phases = _phases(cfg, s3.dir)
        reg = _chosen_values(cfg, tdir, phases)
        out = ctx.stage_dir
        outcomes: dict[str, PhaseOutcome] = {}
        findings: list[QCFinding] = []
        thr = float(cfg.plotting.dws_mask_threshold)
        mode = coupled_mode(fm, phases)
        co: CoupledOutcome | None = None
        if mode is None:
            self._run_separate(ctx, phases, reg, outcomes, findings, thr)
        else:
            datas = {ph: load_phase_data(s3.dir, ph, fm) for ph in phases}
            ctx.logger.info(
                "coupled P+S inversion (%s%s): %d iterations, damping P %g / S %g, smoothing "
                "P %g / S %g",
                mode,
                " + Vp/Vs" if coupled_vpvs(fm) else "",
                fm.iterations,
                reg["P"]["damping"],
                reg["S"]["damping"],
                reg["P"]["smoothing"],
                reg["S"]["smoothing"],
            )
            co = invert_coupled(
                out,
                datas,
                fm,
                reg,
                ctx.n_jobs,
                mode=mode,
                vpvs=coupled_vpvs(fm),
                dws_threshold=thr,
            )
            outcomes = co.outcomes
            for ph, oc in outcomes.items():
                self._collect(ctx, ph, oc, len(phases), findings, fm)
            for n in co.notes:
                ctx.logger.warning("coupled run: %s", n)
            if co.vpvs is not None:
                findings.extend(_vpvs_findings(co.vpvs))
        start = _load_start(s3.dir)
        model = _write_model(
            out / "final_model.nc", start, outcomes, thr, None if co is None else co.vpvs_model
        )
        self._write_tables(ctx, outcomes, reg)
        self._exports(ctx, model, outcomes)
        if co is not None:
            self._write_coupled(ctx, co)
        for f in findings:
            ctx.add_check(f)
        self._summary(ctx, outcomes, reg)
        if co is not None:
            self._summary_coupled(ctx, co)
        self.plot(ctx)

    # ------------------------------------------------------------------ runs
    def _run_separate(
        self,
        ctx: StageContext,
        phases: list[str],
        reg: dict[str, dict[str, Any]],
        outcomes: dict[str, PhaseOutcome],
        findings: list[QCFinding],
        thr: float,
    ) -> None:
        fm = ctx.config.fmtomo
        s3 = ctx.upstream("build_model")
        for ph in phases:
            data = load_phase_data(s3.dir, ph, fm)
            ctx.logger.info(
                "final inversion %s: damping %g, smoothing %g, %d iterations, relocate=%s",
                ph,
                reg[ph]["damping"],
                reg[ph]["smoothing"],
                fm.iterations,
                fm.relocate,
            )
            oc = invert_phase(
                ctx.stage_dir / f"run_{ph}",
                data,
                fm,
                reg[ph]["damping"],
                reg[ph]["smoothing"],
                ctx.n_jobs,
                phase=ph,
                dws_threshold=thr,
            )
            outcomes[ph] = oc
            self._collect(ctx, ph, oc, len(phases), findings, fm)

    @staticmethod
    def _collect(
        ctx: StageContext,
        ph: str,
        oc: PhaseOutcome,
        n_phases: int,
        findings: list[QCFinding],
        fm: Any,
    ) -> None:
        for n in oc.notes:
            ctx.logger.warning("%s: %s", ph, n)
            findings.append(QCFinding("inversion_note", "warn", f"{ph}: {n}"))
        findings.extend(
            inversion_findings(
                oc.iterations, float(fm.min_velocity_km_s), ph if n_phases > 1 else ""
            )
        )

    @staticmethod
    def _write_coupled(ctx: StageContext, co: CoupledOutcome) -> None:
        out = ctx.stage_dir
        cj = out / "chosen_regularisation.json"
        info = json.loads(cj.read_text())
        info["coupled_mode"] = co.mode
        if co.vpvs is not None:
            fm = ctx.config.fmtomo
            info["vpvs"] = {"damping": float(fm.damping), "smoothing": float(fm.smoothing)}
        cj.write_text(json.dumps(info, indent=1))
        if co.joint is not None:
            co.joint.iterations.to_csv(out / "joint_iterations.csv", index=False)
        if co.vpvs is not None:
            co.vpvs.iterations.drop(columns=["dir"], errors="ignore").to_csv(
                out / "vpvs_iterations.csv", index=False
            )
            if co.vpvs.ratio_of_vp_vs_model is not None and co.vpvs.ratio_of_vp_vs_model.exists():
                shutil.copyfile(co.vpvs.ratio_of_vp_vs_model, out / "final_vgridvpvs.in")
            if co.vpvs.final_model is not None:
                shutil.copyfile(co.vpvs.final_model, out / "final_vgrids_vpvs.in")

    @staticmethod
    def _summary_coupled(ctx: StageContext, co: CoupledOutcome) -> None:
        s = ctx.summary
        s["coupled_mode"] = co.mode
        if co.joint is not None:
            s["joint_rms_start_ms"] = [round(float(x), 2) for x in co.joint.initial_rms_ms]
            s["joint_rms_final_ms"] = [round(float(x), 2) for x in co.joint.final_rms_ms]
        if co.vpvs is not None and len(co.vpvs.iterations):
            s["vpvs_rms_start_s"] = round(co.vpvs.initial_rms_s, 4)
            s["vpvs_rms_final_s"] = round(co.vpvs.final_rms_s, 4)
            it = co.vpvs.iterations.iloc[-1]
            s["vpvs_range"] = [round(float(it["vpvs_min"]), 3), round(float(it["vpvs_max"]), 3)]
            s["vpvs_mean"] = round(float(it["vpvs_mean"]), 3)

    # ------------------------------------------------------------------ outputs
    @staticmethod
    def _write_tables(
        ctx: StageContext, outcomes: dict[str, PhaseOutcome], reg: dict[str, dict[str, Any]]
    ) -> None:
        out = ctx.stage_dir
        its, res, rel, hyp = [], [], [], []
        for ph, oc in outcomes.items():
            sfx = "" if ph == "P" else "_S"
            shutil.copyfile(oc.run_dir / "vgrids.in", out / f"final_vgrids{sfx}.in")
            src = "receivers.in" if oc.nonlinear else "sources.in"
            shutil.copyfile(oc.run_dir / src, out / f"final_sources{sfx}.in")
            t = oc.iterations.copy()
            t.insert(0, "phase", ph)
            its.append(t)
            r = oc.residuals.copy()
            r["phase"] = ph
            res.append(r)
            rl = oc.relocation.copy()
            rl["phase"] = ph
            rel.append(rl)
            h = (oc.hypocentres if oc.hypocentres is not None else oc.result.hypocentres).copy()
            if len(h):
                h.insert(0, "phase", ph)
                hyp.append(h)
        pd.concat(its, ignore_index=True).to_csv(out / "inversion_iterations.csv", index=False)
        pd.concat(res, ignore_index=True).to_parquet(out / "residuals_final.parquet")
        pd.concat(rel, ignore_index=True).to_parquet(out / "relocation.parquet")
        if hyp:
            pd.concat(hyp, ignore_index=True).to_parquet(out / "hypocentres_iterations.parquet")
        first = next(iter(reg.values()))
        (out / "chosen_regularisation.json").write_text(
            json.dumps(
                {
                    "damping": first["damping"],
                    "smoothing": first["smoothing"],
                    "phases": reg,
                    "relocate": ctx.config.fmtomo.relocate,
                    "iterations": int(ctx.config.fmtomo.iterations),
                },
                indent=1,
            )
        )

    @staticmethod
    def _exports(ctx: StageContext, model: xr.Dataset, outcomes: dict[str, PhaseOutcome]) -> None:
        from osv.plotting.slices import crop_interior

        inner = crop_interior(model)
        masks = {k: inner[k] for k in inner.data_vars if k.startswith(("dws", "mask"))}
        core = inner.drop_vars(list(masks))
        first = next(iter(outcomes.values()))
        hyp = first.relocation[["event_id", "lat", "lon", "depth_km", "d3_km"]]
        try:
            export_model(core, ctx.stage_dir / "exports", masks=masks, hypocentres=hyp)
        except Exception as exc:  # noqa: BLE001 - exports are convenience files
            ctx.logger.warning("export failed: %s", exc, exc_info=True)
            ctx.add_check(QCFinding("export_failed", "warn", f"model export failed: {exc}"))

    @staticmethod
    def _summary(
        ctx: StageContext, outcomes: dict[str, PhaseOutcome], reg: dict[str, dict[str, Any]]
    ) -> None:
        s = ctx.summary
        s["relocate"] = ctx.config.fmtomo.relocate
        for ph, oc in outcomes.items():
            it = oc.iterations
            r0, r1 = float(it["rms_ms"].iloc[0]), float(it["rms_ms"].iloc[-1])
            s[f"damping_{ph}"] = reg[ph]["damping"]
            s[f"smoothing_{ph}"] = reg[ph]["smoothing"]
            s[f"regularisation_source_{ph}"] = reg[ph]["source"]
            s[f"iterations_{ph}"] = int(it["iter"].iloc[-1])
            s[f"rms_start_ms_{ph}"] = round(r0, 2)
            s[f"rms_final_ms_{ph}"] = round(r1, 2)
            s[f"misfit_reduction_pct_{ph}"] = round(100 * (1 - r1 / r0), 1) if r0 > 0 else None
            vcol = "variance_exact_s2" if "variance_exact_s2" in it else "variance_s2"
            vr = 100 * (1 - float(it[vcol].iloc[-1]) / float(it[vcol].iloc[0]))
            s[f"variance_reduction_pct_{ph}"] = round(vr, 1)
            s[f"chi2_final_{ph}"] = round(float(it["chi2"].iloc[-1]), 3)
            if "model_rms_from_ref" in it:
                s[f"model_change_rms_km_s_{ph}"] = round(
                    float(it["model_rms_from_ref"].iloc[-1]), 4
                )
            s[f"vel_range_km_s_{ph}"] = [
                round(float(it["vel_min"].iloc[-1]), 3),
                round(float(it["vel_max"].iloc[-1]), 3),
            ]
            if oc.mask is not None:
                s[f"covered_fraction_{ph}"] = round(float(oc.mask.mean()), 3)
            if oc.relocation is not None and len(oc.relocation):
                s[f"hypocentre_shift_median_km_{ph}"] = round(
                    float(oc.relocation["d3_km"].median()), 3
                )
            s[f"wall_s_{ph}"] = round(float(oc.result.wall_s), 1)

    # ------------------------------------------------------------------ plots
    def plot(self, ctx: StageContext) -> None:
        cfg = ctx.config
        out = ctx.stage_dir
        s3 = ctx.project_dir / "04_build_model"
        its_all = pd.read_csv(out / "inversion_iterations.csv")
        hyp_path = out / "hypocentres_iterations.parquet"
        hyp_all = pd.read_parquet(hyp_path) if hyp_path.exists() else None
        phases = list(its_all["phase"].unique())
        multi = len(phases) > 1
        for ph in phases:
            run = out / f"run_{ph}"
            mapping = pd.read_csv(run / "mapping.csv")
            nl = cfg.fmtomo.relocate == "nonlinear"
            try:
                tables = residual_tables(run, mapping, nonlinear=nl)
            except Exception as exc:  # noqa: BLE001
                ctx.logger.warning("residual tables failed: %s", exc)
                tables = {}
            make_inversion_figures(
                ctx.figures.save,
                its_all[its_all["phase"] == ph].reset_index(drop=True),
                tables=tables,
                hypocentres=None if hyp_all is None else hyp_all[hyp_all["phase"] == ph],
                min_velocity=float(cfg.fmtomo.min_velocity_km_s),
                suffix=f"_{ph}" if multi else "",
                label=f"{ph}: " if multi else "",
            )
            if tables:
                from osv.analysis.plots_residuals import make_residual_figures

                try:
                    make_residual_figures(
                        run, mapping, ctx.figures, nonlinear=nl, section=f"residuals_{ph}"
                    )
                except Exception as exc:  # noqa: BLE001
                    ctx.logger.warning("residual figures failed: %s", exc, exc_info=True)
            if multi:
                break  # the standard residual set once (P); S has the diagnostics above
        self._model_figures(ctx, s3)

    @staticmethod
    def _model_figures(ctx: StageContext, s3: Path) -> None:
        from osv.analysis.plots_model import make_model_figures
        from osv.plotting.slices import crop_interior

        out = ctx.stage_dir
        with xr.open_dataset(out / "final_model.nc") as raw:
            model = raw.load()
        with xr.open_dataset(s3 / STARTING_MODEL) as raw:
            start = raw.load()
        inner = crop_interior(model)
        cov = {k: inner[k] for k in inner.data_vars if k.startswith("dws")}
        rel = pd.read_parquet(out / "relocation.parquet")
        rel = rel[rel["phase"] == "P"]
        sta = pd.read_csv(s3 / FMTOMO_DIR / "stations.csv")
        make_model_figures(
            ctx.figures.save,
            model,
            None,
            start,
            cov or None,
            rel[["lat", "lon", "depth_km"]].rename(columns={"depth_km": "depth_km_bsl"}),
            sta,
            ctx.config,
        )


def _load_start(stage3_dir: Path) -> xr.Dataset:
    with xr.open_dataset(stage3_dir / STARTING_MODEL) as raw:
        return raw.load()
