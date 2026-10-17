"""Bootstrap / jackknife model uncertainty: inversions of event-resampled data (Stage 7).

The data set is resampled **by event** (an event brings all its picks):

``bootstrap``
    ``n`` events drawn with replacement.  A duplicated event becomes several FMTOMO sources with
    distinct indices (and identical picks); an event that is not drawn is absent.
``jackknife``
    delete-d: ``d = round(delete_fraction * n)`` events removed per sample (``d = 1`` and
    ``n_samples >= n`` gives the classic leave-one-out set, one sample per event).

For each sample :func:`build_sample` rebuilds ``sources.in``, ``receivers.in`` and ``otimes.dat``
consistently: the sources are the chosen events' sources in sample order; the rays of a receiver
are numbered in file order of ``receivers.in`` (receiver-major, then by source); ``otimes`` stays
**source-major** (sorted by source, receiver, ray), the order of ``arrivals.dat``.  Receivers
that lose all their rays are dropped and the remaining ones renumbered.

:func:`run_bootstrap` runs one inversion per sample from the same starting model with the chosen
damping and smoothing (:func:`osv.fmtomo.invert.run_inversion`), concurrently within an ``fm3d``
process budget.  Each sample writes ``model.nc`` (final velocity), ``hypocentres.csv``,
``iterations.csv`` and ``summary.json`` into ``<workdir>/<sample>/`` and loses its heavy files.
Finished samples are skipped on a re-run with identical settings.  Afterwards
:func:`model_statistics` gives per-node mean, standard deviation and percentile interval of the
velocity (and of the perturbation relative to the starting model) and
:func:`hypocentre_variability` the scatter of the relocated events (those present in at least
two samples).

Uncertainty estimates
---------------------
Bootstrap: sample standard deviation (``ddof = 1``) and the percentile interval of the sample
values.  Delete-d jackknife: ``var = (n - d) / d * mean((x_i - mean)^2)`` over the samples
and, because pseudo-values are not an empirical distribution, a normal interval
``mean +- z * std``.

``<workdir>/bootstrap_std.nc`` (full FMTOMO grid, cushions included) holds ``vp_mean``,
``vp_std``, ``vp_lo``, ``vp_hi``, ``vp_start``, ``dvp_mean_pct``, ``dvp_std_pct``,
``dvp_lo_pct``, ``dvp_hi_pct`` (``vs_*`` for S) and ``dws`` when given.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import shutil
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.fmtomo import invert as _inv
from osv.fmtomo.controls import ModelInfo, build_controls
from osv.fmtomo.rundir import assemble_run_dir
from osv.fmtomo.tradeoff import HEAVY_FILES, load_chosen
from osv.io.fmtomo import (
    InterfaceGrids,
    PropGrid,
    Receivers,
    Source,
    VelocityGrids,
    read_interfaces,
    read_observed,
    read_propgrid,
    read_receivers,
    read_sources,
    read_vgrids,
)
from osv.resolution.synth import NETCDF_LOCK

log = logging.getLogger(__name__)

__all__ = [
    "BootstrapError",
    "BootstrapInputs",
    "BootstrapResult",
    "Sample",
    "SampleInputs",
    "build_sample",
    "hypocentre_variability",
    "make_samples",
    "model_statistics",
    "run_bootstrap",
]

SUMMARY = "summary.json"
STATS_FILE = "bootstrap_std.nc"
KM_PER_DEG = 111.194926645
#: samples needed for a standard deviation
MIN_SAMPLES = 2
#: minimum events for resampling
MIN_EVENTS = 3
TABLE_COLUMNS = (
    "name",
    "index",
    "n_events",
    "n_unique_events",
    "n_sources",
    "success",
    "misfit_initial_s",
    "misfit_final_s",
    "n_iter",
    "runtime_s",
    "flags",
    "model",
    "dir",
)
#: Stage 3 file names per phase (``<stage3>/fmtomo/``)
_FILES = {
    "P": ("vgrids.in", "sources.in", "receivers.in", "otimes_P.dat", "otimes_P_mapping.csv"),
    "S": ("vgrids_S.in", "sources_S.in", "receivers_S.in", "otimes_S.dat", "otimes_S_mapping.csv"),
}


class BootstrapError(RuntimeError):
    """The bootstrap/jackknife cannot run or too few samples succeeded (message says why)."""


# ---------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------
@dataclass
class BootstrapInputs:
    """Stage 3 FMTOMO inputs of one phase (objects or paths) in the linear geometry.

    Events are the FMTOMO sources, stations the receivers.  ``mapping`` (optional, the Stage 3
    ``otimes_*_mapping.csv``) provides ``source_idx`` and ``event_id``; sources sharing an
    ``event_id`` are resampled together, without a mapping every source is its own event.
    ``observed`` is the ``otimes.dat`` table in ``arrivals.dat`` order.
    """

    grids: VelocityGrids | str | Path
    interfaces: InterfaceGrids | str | Path
    propgrid: PropGrid | str | Path
    sources: list[Source] | str | Path
    receivers: Receivers | str | Path
    observed: pd.DataFrame | str | Path
    mapping: pd.DataFrame | str | Path | None = None
    phase: str = "P"

    @classmethod
    def from_stage3(cls, stage3_dir: str | Path, phase: str = "P") -> BootstrapInputs:
        """Read the hand-off files of ``<stage3_dir>/fmtomo`` (or ``stage3_dir`` itself)."""
        d = Path(stage3_dir)
        if (d / "fmtomo").is_dir():
            d = d / "fmtomo"
        vg, so, re_, ot, mp = (d / n for n in _FILES[phase])
        for p in (vg, so, re_, ot, d / "interfaces.in", d / "propgrid.in"):
            if not p.exists():
                raise BootstrapError(f"{p} not found: run `osv build-model` first")
        return cls(
            vg, d / "interfaces.in", d / "propgrid.in", so, re_, ot, mp if mp.exists() else None,
            phase,
        )  # fmt: skip

    def load(
        self,
    ) -> tuple[
        VelocityGrids, InterfaceGrids, PropGrid, list[Source], Receivers, pd.DataFrame, list[Any]
    ]:
        """``(grids, interfaces, propgrid, sources, receivers, observed, event_groups)``.

        ``event_groups`` is a list of ``(event_id, [1-based source ids])``.
        """
        g = read_vgrids(self.grids) if isinstance(self.grids, (str, Path)) else self.grids
        itf = (
            read_interfaces(self.interfaces)
            if isinstance(self.interfaces, (str, Path))
            else self.interfaces
        )
        pg = (
            read_propgrid(self.propgrid)
            if isinstance(self.propgrid, (str, Path))
            else self.propgrid
        )
        src = read_sources(self.sources) if isinstance(self.sources, (str, Path)) else self.sources
        rec = (
            read_receivers(self.receivers)
            if isinstance(self.receivers, (str, Path))
            else self.receivers
        )
        obs = (
            read_observed(self.observed)
            if isinstance(self.observed, (str, Path))
            else self.observed
        )
        mp = self.mapping
        if isinstance(mp, (str, Path)):
            mp = pd.read_csv(mp)
        return g, itf, pg, list(src), rec, obs.reset_index(drop=True), _event_groups(len(src), mp)


def _event_groups(n_sources: int, mapping: pd.DataFrame | None) -> list[tuple[str, list[int]]]:
    """Group source ids by event id (first appearance order); unmapped sources are alone."""
    ev_of: dict[int, str] = {}
    if mapping is not None and {"source_idx", "event_id"} <= set(mapping.columns):
        m = mapping[["source_idx", "event_id"]].drop_duplicates("source_idx")
        ev_of = {int(s): str(e) for s, e in zip(m["source_idx"], m["event_id"], strict=True)}
    groups: dict[str, list[int]] = {}
    for sid in range(1, n_sources + 1):
        groups.setdefault(ev_of.get(sid, f"src{sid}"), []).append(sid)
    return list(groups.items())


# ---------------------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    """One resample: ``events`` are positions into the event list, repeats allowed."""

    index: int
    name: str
    events: tuple[int, ...]

    @property
    def n_unique(self) -> int:
        return len(set(self.events))


def make_samples(
    n_events: int,
    method: str = "bootstrap",
    n_samples: int = 20,
    seed: int = 42,
    delete_fraction: float = 0.1,
) -> list[Sample]:
    """Draw the event resamples (deterministic for a given ``seed``).

    Bootstrap samples have ``n_events`` draws with replacement (at least two distinct events).
    Jackknife samples delete ``d = max(1, round(delete_fraction * n_events))`` events; for
    ``d == 1`` the deleted events are distinct (all of them when ``n_samples >= n_events``).
    """
    if n_events < MIN_EVENTS:
        raise BootstrapError(f"{n_events} event(s): need at least {MIN_EVENTS} to resample")
    if method not in ("bootstrap", "jackknife"):
        raise ValueError(f"method must be 'bootstrap' or 'jackknife', got {method!r}")
    rng = np.random.default_rng(seed)
    out: list[Sample] = []
    if method == "bootstrap":
        for i in range(1, n_samples + 1):
            while True:
                ev = rng.integers(0, n_events, n_events)
                if len(set(ev.tolist())) >= 2:
                    break
            out.append(Sample(i, f"boot_{i:03d}", tuple(int(e) for e in ev)))
        return out
    d = max(1, min(int(round(delete_fraction * n_events)), n_events - 2))
    if d == 1:
        victims = [{int(v)} for v in rng.permutation(n_events)[: min(n_samples, n_events)]]
        victims.sort(key=lambda s: next(iter(s)))
    else:
        victims = [set(rng.choice(n_events, d, replace=False).tolist()) for _ in range(n_samples)]
    for i, dele in enumerate(victims, start=1):
        keep = tuple(e for e in range(n_events) if e not in dele)
        out.append(Sample(i, f"jack_{i:03d}", keep))
    return out


def jackknife_d(n_events: int, delete_fraction: float) -> int:
    """Number of events deleted per jackknife sample (see :func:`make_samples`)."""
    return max(1, min(int(round(delete_fraction * n_events)), n_events - 2))


@dataclass
class SampleInputs:
    """Inputs of one resampled inversion.

    ``event_of_source[j]`` is the event position (into the event list) of new source ``j + 1``;
    ``old_source[j]`` its source id in the full data set.
    """

    sources: list[Source]
    receivers: Receivers
    observed: pd.DataFrame
    event_of_source: np.ndarray
    old_source: np.ndarray
    #: old receiver id of each new receiver id - 1
    old_receiver: np.ndarray


def build_sample(
    sources: Sequence[Source],
    receivers: Receivers,
    observed: pd.DataFrame,
    groups: Sequence[tuple[str, Sequence[int]]],
    sample: Sample,
) -> SampleInputs:
    """Rebuild ``sources``, ``receivers`` and ``otimes`` for the events of ``sample``.

    Parameters
    ----------
    sources, receivers, observed
        Full data set (linear geometry); ``observed`` has ``receiver, source, ray, subray, time,
        uncertainty`` with ``ray`` the 1-based position of the ray among the rays of its receiver
        in ``receivers.rays`` (file order).
    groups
        ``(event_id, [source ids])`` of every event.
    sample
        The chosen events.

    Raises
    ------
    BootstrapError
        If an observation has no matching ray in ``receivers.in``.
    """
    old_src: list[int] = []
    ev_pos: list[int] = []
    for e in sample.events:
        for sid in groups[e][1]:
            old_src.append(int(sid))
            ev_pos.append(int(e))
    new_of_pos = np.arange(1, len(old_src) + 1)

    rays = receivers.rays.reset_index(drop=True).copy()
    rays["old_ray"] = rays.groupby("receiver").cumcount() + 1
    obs = observed.reset_index(drop=True)
    # every observation must match a ray of the full data set
    key = obs[["receiver", "source", "ray"]].rename(columns={"ray": "old_ray"})
    chk = key.merge(rays[["receiver", "source", "old_ray"]], how="left", indicator=True)
    if (chk["_merge"] != "both").any():
        raise BootstrapError(
            f"{int((chk['_merge'] != 'both').sum())} observation(s) have no matching ray in "
            "receivers.in (receiver, source, ray); cannot resample"
        )
    # new ray table: sources in sample order, then numbered per receiver
    sel = pd.DataFrame({"old_source": old_src, "new_source": new_of_pos})
    nr = sel.merge(rays.rename(columns={"source": "old_source"}), on="old_source", how="inner")
    nr = nr.sort_values(["receiver", "new_source", "old_ray"], kind="stable").reset_index(drop=True)
    nr["ray"] = nr.groupby("receiver").cumcount() + 1
    # drop receivers without rays, renumber
    used = np.sort(nr["receiver"].unique())
    rmap = {int(o): i + 1 for i, o in enumerate(used)}
    pos = receivers.positions.loc[used].copy()
    pos.index = pd.RangeIndex(1, len(used) + 1, name="receiver")
    nr["receiver_new"] = nr["receiver"].map(rmap)
    new_rays = pd.DataFrame(
        {
            "receiver": nr["receiver_new"].astype("int64"),
            "source": nr["new_source"].astype("int64"),
            "path": nr["path"].astype("int64"),
        }
    )
    # observations of the chosen sources, repeated per copy, with the new indices
    o = obs.rename(columns={"source": "old_source", "ray": "old_ray", "receiver": "old_receiver"})
    o = sel.merge(o, on="old_source", how="inner")
    o = o.merge(
        nr[["receiver", "new_source", "old_ray", "ray", "receiver_new"]].rename(
            columns={"receiver": "old_receiver"}
        ),
        on=["old_receiver", "new_source", "old_ray"],
        how="left",
    )
    if o["ray"].isna().any():
        raise BootstrapError("internal error: observations without a rebuilt ray")
    new_obs = pd.DataFrame(
        {
            "receiver": o["receiver_new"].astype("int64"),
            "source": o["new_source"].astype("int64"),
            "ray": o["ray"].astype("int64"),
            "subray": o["subray"].astype("int64") if "subray" in o else 0,
            "time": o["time"].astype(float),
            "uncertainty": o["uncertainty"].astype(float),
        }
    )
    new_obs = new_obs.sort_values(["source", "receiver", "ray"], kind="stable").reset_index(
        drop=True
    )
    new_sources = [dataclasses.replace(sources[s - 1]) for s in old_src]
    return SampleInputs(
        new_sources,
        Receivers(pos, new_rays),
        new_obs,
        np.asarray(ev_pos, dtype=int),
        np.asarray(old_src, dtype=int),
        used.astype(int),
    )


# ---------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------
def model_statistics(
    models: np.ndarray,
    start: xr.DataArray,
    *,
    method: str = "bootstrap",
    n_events: int | None = None,
    delete_d: int = 1,
    interval_percent: float = 90.0,
    var: str = "vp",
    dws: xr.DataArray | None = None,
) -> xr.Dataset:
    """Per-node mean, standard deviation and interval of the sample models.

    Parameters
    ----------
    models
        Stack of final velocities, shape ``(n_samples, depth, lat, lon)``.
    start
        Starting model on the same grid (DataArray with dims ``depth, lat, lon``); the reference
        of the perturbations in percent.
    method, n_events, delete_d
        ``"jackknife"`` scales the variance by ``(n_events - delete_d) / delete_d`` and uses a
        normal interval; ``"bootstrap"`` uses ``ddof = 1`` and sample percentiles.
    interval_percent
        Central interval, e.g. 90 gives the 5th to 95th percentile.
    """
    from scipy.stats import norm

    m = np.asarray(models, dtype=float)
    if m.ndim != 4 or m.shape[0] < MIN_SAMPLES:
        raise BootstrapError(f"need at least {MIN_SAMPLES} sample models, got shape {m.shape}")
    if m.shape[1:] != tuple(start.shape):
        raise ValueError(f"models {m.shape[1:]} and starting model {start.shape} differ")
    n = m.shape[0]
    mean = m.mean(axis=0)
    lo_p, hi_p = 50.0 - interval_percent / 2, 50.0 + interval_percent / 2
    if method == "jackknife":
        if n_events is None:
            raise ValueError("jackknife statistics need n_events")
        std = np.sqrt((n_events - delete_d) / delete_d * ((m - mean) ** 2).mean(axis=0))
        z = float(norm.ppf(0.5 + interval_percent / 200.0))
        lo, hi = mean - z * std, mean + z * std
    else:
        std = m.std(axis=0, ddof=1)
        lo, hi = np.percentile(m, [lo_p, hi_p], axis=0)
    s0 = start.values.astype(float)
    pct = 100.0 * (m / s0 - 1.0)
    pmean = pct.mean(axis=0)
    pstd = std / s0 * 100.0
    if method == "jackknife":
        plo, phi = 100.0 * (lo / s0 - 1.0), 100.0 * (hi / s0 - 1.0)
    else:
        plo, phi = np.percentile(pct, [lo_p, hi_p], axis=0)
    dims = ("depth", "lat", "lon")
    coords = {d: start[d] for d in dims}

    def v(a: np.ndarray, units: str, name: str) -> tuple[Any, ...]:
        return dims, a.astype(np.float32), {"units": units, "long_name": name}

    ds = xr.Dataset(
        {
            f"{var}_mean": v(mean, "km/s", f"{var} mean of samples"),
            f"{var}_std": v(std, "km/s", f"{var} standard deviation ({method})"),
            f"{var}_lo": v(lo, "km/s", f"{var} lower interval bound ({interval_percent:g}%)"),
            f"{var}_hi": v(hi, "km/s", f"{var} upper interval bound ({interval_percent:g}%)"),
            f"{var}_start": v(s0, "km/s", f"{var} starting model"),
            f"d{var}_mean_pct": v(pmean, "percent", f"d{var} mean vs starting model"),
            f"d{var}_std_pct": v(pstd, "percent", f"d{var} standard deviation"),
            f"d{var}_lo_pct": v(plo, "percent", f"d{var} lower interval bound"),
            f"d{var}_hi_pct": v(phi, "percent", f"d{var} upper interval bound"),
        },
        coords=coords,
        attrs={
            "method": method,
            "n_samples": n,
            "interval_percent": float(interval_percent),
            "n_cushion": int(start.attrs.get("n_cushion", 1)),
        },
    )
    if n_events is not None:
        ds.attrs["n_events"] = int(n_events)
    if method == "jackknife":
        ds.attrs["delete_d"] = int(delete_d)
    if dws is not None:
        ds["dws"] = dws.transpose(*dims).astype(np.float32)
    return ds


def hypocentre_variability(hypos: pd.DataFrame, min_samples: int = 2) -> pd.DataFrame:
    """Scatter of each event's relocated hypocentre over the samples.

    Parameters
    ----------
    hypos
        Columns ``sample, event, lat, lon, depth_km`` (one row per sample and source; copies of
        a duplicated event in one sample are averaged first).
    min_samples
        Events found in fewer samples are left out.

    Returns
    -------
    pandas.DataFrame
        Per event: ``n_samples``, mean ``lat``/``lon``/``depth_km`` and ``std_east_km``,
        ``std_north_km``, ``std_depth_km``, ``std_h_km`` (horizontal) and ``std_3d_km``.
    """
    cols = ["event", "n_samples", "lat", "lon", "depth_km", "std_east_km", "std_north_km",
            "std_depth_km", "std_h_km", "std_3d_km"]  # fmt: skip
    if hypos.empty:
        return pd.DataFrame(columns=cols)
    h = hypos.groupby(["event", "sample"], as_index=False)[["lat", "lon", "depth_km"]].mean()
    rows = []
    for ev, g in h.groupby("event"):
        if len(g) < min_samples:
            continue
        la0 = float(g["lat"].mean())
        n_e = float(
            (g["lon"] - g["lon"].mean()).std(ddof=1) * KM_PER_DEG * math.cos(math.radians(la0))
        )
        n_n = float(g["lat"].std(ddof=1) * KM_PER_DEG)
        n_z = float(g["depth_km"].std(ddof=1))
        rows.append(
            (ev, len(g), la0, float(g["lon"].mean()), float(g["depth_km"].mean()),
             n_e, n_n, n_z, math.hypot(n_e, n_n), math.sqrt(n_e**2 + n_n**2 + n_z**2))
        )  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """Outcome of :func:`run_bootstrap`.

    ``table`` has one row per sample (:data:`TABLE_COLUMNS`); ``stats`` is the statistics
    Dataset (also ``stats_path``); ``hypocentres`` the relocated hypocentres of every sample
    (``sample, event, lat, lon, depth_km``) and ``hypocentre_stats`` their per-event scatter.
    """

    workdir: Path
    table: pd.DataFrame
    stats: xr.Dataset
    stats_path: Path
    hypocentres: pd.DataFrame
    hypocentre_stats: pd.DataFrame
    method: str
    chosen_damping: float
    chosen_smoothing: float
    n_iter: int
    n_events: int
    max_concurrent: int
    jobs_per_sample: int
    skipped: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return [
            str(n)
            for n, ok in zip(self.table["name"], self.table["success"], strict=True)
            if not ok
        ]

    @property
    def n_ok(self) -> int:
        return int(self.table["success"].sum())


def _f(x: Any) -> float:
    try:
        return math.nan if x is None else float(x)
    except (TypeError, ValueError):
        return math.nan


def _compact(root: Path) -> None:
    if not root.exists():
        return
    for name in HEAVY_FILES:
        for p in root.rglob(name):
            p.unlink(missing_ok=True)
    for pat in ("parallel", "fm3d_*", "group*", "iterations"):
        for p in list(root.rglob(pat)):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    for p in root.rglob("*.dat"):
        if p.name.startswith(("timefield", "tfield", "gridsave", "rays")) or p.stat().st_size > (
            50 << 20
        ):
            p.unlink(missing_ok=True)


def _signature(s: Sample, d: float, sm: float, n_iter: int, phase: str, method: str) -> dict:
    return {
        "name": s.name,
        "method": method,
        "events": list(s.events),
        "damping": float(d),
        "smoothing": float(sm),
        "n_iter": int(n_iter),
        "phase": phase,
    }


def _read_summary(path: Path, sig: dict[str, Any]) -> dict[str, Any] | None:
    try:
        sm = json.loads((path / SUMMARY).read_text())
    except (OSError, ValueError):
        return None
    if not sm.get("success") or not (path / "model.nc").exists():
        return None
    for k, v in sig.items():
        if isinstance(v, float):
            if not math.isclose(_f(sm.get(k)), v):
                return None
        elif sm.get(k) != v:
            return None
    return sm


def _final_hypocentres(
    res: Any, si: SampleInputs, record_to_source: dict[int, int] | None
) -> pd.DataFrame:
    """Last-iteration hypocentres per new source (``source`` 1-based)."""
    h = res.hypocentres
    if h is None or len(h) == 0:
        return pd.DataFrame(columns=["source", "lat", "lon", "depth_km"])
    h = h[h["iter"] == h["iter"].max()]
    h = h[["source", "lat", "lon", "depth_km"]].copy()
    if record_to_source is not None:  # nonlinear: one record per observation
        h["source"] = h["source"].map(record_to_source)
        h = h.dropna(subset=["source"]).groupby("source", as_index=False).mean()
        h["source"] = h["source"].astype(int)
    return h.reset_index(drop=True)


def _run_sample(
    sample: Sample,
    prepared: dict[str, Any],
    cfg: Any,
    d: float,
    sm: float,
    n_iter: int,
    workdir: Path,
    n_jobs: int,
    var: str,
    min_free_gb: float,
    mem_per_job_gb: float | None,
    timeout: float | None,
) -> dict[str, Any]:
    tdir = workdir / sample.name
    phase = prepared["phase"]
    sig = _signature(sample, d, sm, n_iter, phase, prepared["method"])
    done = _read_summary(tdir, sig)
    if done is not None:
        log.info("sample %s already complete, skipping", sample.name)
        done["skipped"] = True
        return done
    shutil.rmtree(tdir, ignore_errors=True)
    tdir.mkdir(parents=True)
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        **sig,
        "index": sample.index,
        "n_events": len(sample.events),
        "n_unique_events": sample.n_unique,
        "success": False,
        "dir": str(tdir),
        "model": str(tdir / "model.nc"),
        "flags": "",
        "misfit_initial_s": math.nan,
        "misfit_final_s": math.nan,
    }
    try:
        free = shutil.disk_usage(tdir).free / 1e9
        if free < min_free_gb:
            raise OSError(f"only {free:.1f} GB free on the work disk (need {min_free_gb:g} GB)")
        si = build_sample(
            prepared["sources"], prepared["receivers"], prepared["observed"], prepared["groups"],
            sample,
        )  # fmt: skip
        row["n_sources"] = len(si.sources)
        fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
        fm_run = fm.model_copy(update={"damping": float(d), "smoothing": float(sm)})
        run_inputs = (si.sources, si.receivers, si.observed)
        src_dat = None
        record_to_source = None
        if fm.relocate == "nonlinear":
            run_inputs, src_dat, record_to_source = _nonlinear(prepared, si, phase)
        else:
            if fm.relocate == "linear":
                run_inputs = (_with_uncertainty(si.sources, float(fm.source_uncertainty_km)),
                              si.receivers, si.observed)  # fmt: skip
        sources, receivers, otimes = run_inputs
        grids = prepared["grids"]
        info = ModelInfo.from_model(grids, sources=sources, receivers=receivers)
        ctl = build_controls(fm_run, info, phases=[phase])
        run = tdir / "run"
        assemble_run_dir(
            run, grids, prepared["interfaces"], prepared["propgrid"], sources, receivers, otimes,
            ctl, phase=phase, src_dat=src_dat,
        )  # fmt: skip
        res = _inv.run_inversion(
            run, ctl, n_iter, n_jobs=n_jobs, keep_iterations=False, timeout=timeout,
            mem_per_job_gb=mem_per_job_gb, stop_on_error=False, phase=phase,
        )  # fmt: skip
        tab = res.iterations
        tab.to_csv(tdir / "iterations.csv", index=False)
        flags = list(res.flags)
        if res.success:
            vel = (
                read_vgrids(run / "vgrids.in")
                .get(1, 1)["velocity"]
                .transpose("depth", "lat", "lon")
            )
            if vel.shape != prepared["start"].shape:
                raise ValueError(
                    f"final model {vel.shape} differs from the start {prepared['start'].shape}"
                )
            with NETCDF_LOCK:
                vel.rename(var).astype(np.float32).to_dataset().to_netcdf(tdir / "model.nc")
            hyp = _final_hypocentres(res, si, record_to_source)
            if len(hyp):
                hyp["event"] = si.event_of_source[hyp["source"].to_numpy(int) - 1]
                hyp["sample"] = sample.name
            hyp.to_csv(tdir / "hypocentres.csv", index=False)
        row.update(
            success=bool(res.success),
            misfit_initial_s=_f(tab["rms_ms"].iloc[0]) / 1000.0,
            misfit_final_s=_f(tab["rms_ms"].iloc[-1]) / 1000.0,
            flags="; ".join(flags),
        )
    except Exception as exc:  # noqa: BLE001 - one failed sample must not kill the others
        log.error("sample %s failed: %s", sample.name, exc)
        row.update(success=False, flags=f"error: {exc}", traceback=traceback.format_exc(limit=4))
    _compact(tdir)
    row["runtime_s"] = time.perf_counter() - t0
    (tdir / SUMMARY).write_text(json.dumps(row, indent=1, default=float))
    return row


def _with_uncertainty(sources: list[Source], km: float) -> list[Source]:
    out = []
    for s in sources:
        if s.uncertainty is not None:
            out.append(s)
            continue
        dlat = km / KM_PER_DEG
        dlon = km / (KM_PER_DEG * max(math.cos(math.radians(s.lat)), 1e-3))
        out.append(dataclasses.replace(s, uncertainty=(km, dlat, dlon)))
    return out


def _nonlinear(prepared: dict[str, Any], si: SampleInputs, phase: str):  # noqa: ANN202
    """Swap to the ``tomo3di`` geometry (stations are sources); needs Stage 4 helpers."""
    from osv.fmtomo.rundir import RunInputs
    from osv.fmtomo.stage import nonlinear_src_dat, to_nonlinear_geometry

    inp = RunInputs(
        prepared["grids"], prepared["interfaces"], prepared["propgrid"], si.sources,
        si.receivers, si.observed,
    )  # fmt: skip
    mapping = pd.DataFrame(
        {
            "source_idx": si.observed["source"].to_numpy(),
            "receiver_idx": si.observed["receiver"].to_numpy(),
            "time": si.observed["time"].to_numpy(),
            "uncertainty": si.observed["uncertainty"].to_numpy(),
        }
    )
    new, nm = to_nonlinear_geometry(inp, mapping, phase)
    src_dat = nonlinear_src_dat(new.otimes, nm, len(new.sources))
    rec_to_src = {int(r): int(e) for r, e in zip(nm["receiver_idx"], nm["event_idx"], strict=True)}
    return (new.sources, new.receivers, new.otimes), src_dat, rec_to_src


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def _resolve_chosen(
    chosen: tuple[float, float] | str | Path | None, cfg: Any
) -> tuple[float, float]:
    fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
    if chosen is None:
        return float(fm.damping), float(fm.smoothing)
    if isinstance(chosen, (str, Path)):
        return load_chosen(chosen)
    return float(chosen[0]), float(chosen[1])


def run_bootstrap(
    base_inputs: BootstrapInputs,
    cfg: Any,
    chosen: tuple[float, float] | str | Path | None,
    workdir: str | Path,
    n_jobs: int = 1,
    *,
    dws: xr.DataArray | None = None,
    samples: Sequence[Sample] | None = None,
    mem_per_job_gb: float | None = None,
    timeout: float | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> BootstrapResult:
    """Invert event-resampled data sets and summarise the spread of the final models.

    Parameters
    ----------
    base_inputs
        Stage 3 inputs (:meth:`BootstrapInputs.from_stage3`); the starting model of every sample
        is its ``grids``.
    cfg
        :class:`~osv.config.schema.OSVConfig` (``validation.bootstrap``, ``fmtomo``,
        ``project.random_seed``) or an object with those sections.
    chosen
        ``(damping, smoothing)``, a ``tradeoff.json`` (or its directory), or ``None`` for
        ``fmtomo.damping`` / ``fmtomo.smoothing``.
    workdir
        Output directory; one subdirectory per sample, plus ``bootstrap_std.nc``,
        ``bootstrap_samples.csv``, ``bootstrap_hypocentres.csv`` and ``bootstrap.json``.
    n_jobs
        Total ``fm3d`` processes (``max_concurrent * jobs_per_sample <= n_jobs``).
    dws
        Optional derivative weight sum of the real inversion (full grid) stored in the output
        for the std-versus-DWS figure.
    samples
        Explicit resamples (default: :func:`make_samples` with the configuration).

    Returns
    -------
    BootstrapResult

    Raises
    ------
    BootstrapError
        Fewer than two samples succeeded (finished samples are kept; fix and re-run).
    OSError
        Less free disk than ``validation.bootstrap.min_free_disk_gb`` before starting.
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    bc = cfg.validation.bootstrap if hasattr(cfg, "validation") else cfg
    fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
    d, sm = _resolve_chosen(chosen, cfg)
    n_iter = int(bc.iterations if bc.iterations is not None else fm.iterations)
    seed = (
        bc.seed
        if bc.seed is not None
        else int(getattr(getattr(cfg, "project", None), "random_seed", 42))
    )
    free = shutil.disk_usage(wd).free / 1e9
    if free < bc.min_free_disk_gb:
        raise OSError(
            f"only {free:.1f} GB free in {wd}; need {bc.min_free_disk_gb:g} GB (free disk space "
            "or lower validation.bootstrap.min_free_disk_gb)"
        )
    grids, itf, pg, sources, receivers, observed, groups = base_inputs.load()
    phase = base_inputs.phase
    var = "vp" if phase == "P" else "vs"
    n_events = len(groups)
    if samples is None:
        samples = make_samples(n_events, bc.method, bc.n_samples, seed, bc.delete_fraction)
    samples = list(samples)
    start = grids.get(1, 1)["velocity"].transpose("depth", "lat", "lon")
    start.attrs = {**start.attrs, **{"n_cushion": grids.get(1, 1).attrs.get("n_cushion", 1)}}
    prepared = {
        "grids": grids, "interfaces": itf, "propgrid": pg, "sources": sources,
        "receivers": receivers, "observed": observed, "groups": groups, "phase": phase,
        "method": bc.method, "start": start,
    }  # fmt: skip
    n_jobs = max(1, int(n_jobs))
    k = max(1, min(int(bc.max_concurrent or max(1, n_jobs // 4)), len(samples), n_jobs))
    jobs = max(1, n_jobs // k)
    log.info(
        "%s: %d samples of %d events, damping %g smoothing %g, %d iterations, %d x %d fm3d jobs",
        bc.method, len(samples), n_events, d, sm, n_iter, k, jobs,
    )  # fmt: skip

    def work(s: Sample) -> dict[str, Any]:
        row = _run_sample(
            s, prepared, cfg, d, sm, n_iter, wd, jobs, var, float(bc.min_free_disk_gb),
            mem_per_job_gb, timeout,
        )  # fmt: skip
        if on_sample is not None:
            on_sample(row)
        return row

    if k == 1:
        rows = [work(s) for s in samples]
    else:
        with ThreadPoolExecutor(max_workers=k) as ex:
            rows = list(ex.map(work, samples))
    skipped = [str(r["name"]) for r in rows if r.pop("skipped", False)]
    table = pd.DataFrame(
        [{c: r.get(c, math.nan) for c in TABLE_COLUMNS} for r in rows], columns=list(TABLE_COLUMNS)
    )
    table["success"] = table["success"].astype(bool)
    ok = [r for r in rows if r.get("success")]
    table.to_csv(wd / "bootstrap_samples.csv", index=False)
    if len(ok) < MIN_SAMPLES:
        raise BootstrapError(
            f"only {len(ok)} of {len(samples)} {bc.method} samples succeeded (need "
            f"{MIN_SAMPLES}); see {wd / 'bootstrap_samples.csv'} (column flags); finished "
            "samples are kept, re-run to retry the others"
        )
    models = np.stack(
        [xr.load_dataset(r["model"])[var].transpose("depth", "lat", "lon").values for r in ok]
    )
    stats = model_statistics(
        models, start, method=bc.method, n_events=n_events,
        delete_d=jackknife_d(n_events, bc.delete_fraction), interval_percent=bc.interval_percent,
        var=var, dws=dws,
    )  # fmt: skip
    stats.attrs.update(damping=d, smoothing=sm, n_iter=n_iter, phase=phase)
    stats_path = wd / STATS_FILE
    with NETCDF_LOCK:
        stats.to_netcdf(stats_path)
    parts = []
    for r in ok:
        p = Path(r["dir"]) / "hypocentres.csv"
        try:
            h = pd.read_csv(p)
        except (OSError, ValueError):
            continue
        if len(h):
            parts.append(h)
    hyp = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["source", "lat", "lon", "depth_km", "event", "sample"])
    )
    if len(hyp):
        hyp["event_id"] = [groups[int(e)][0] for e in hyp["event"]]
    hyp.to_csv(wd / "bootstrap_hypocentres.csv", index=False)
    hstats = hypocentre_variability(hyp.assign(event=hyp.get("event_id", hyp["event"])))
    res = BootstrapResult(
        wd, table, stats, stats_path, hyp, hstats, bc.method, d, sm, n_iter, n_events, k, jobs,
        skipped,
    )  # fmt: skip
    (wd / "bootstrap.json").write_text(
        json.dumps(
            {
                "method": bc.method,
                "n_samples": len(samples),
                "n_ok": len(ok),
                "n_events": n_events,
                "chosen_damping": d,
                "chosen_smoothing": sm,
                "n_iter": n_iter,
                "seed": seed,
                "max_concurrent": k,
                "jobs_per_sample": jobs,
                "skipped": skipped,
                "failed": res.failed,
                "median_vp_std": float(np.nanmedian(stats[f"{var}_std"].values)),
            },
            indent=1,
        )  # fmt: skip
    )
    return res
