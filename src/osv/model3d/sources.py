"""FMTOMO sources, receivers and observed travel times from the Stage 1/2 tables (task 4.8).

:func:`build_fmtomo_data` turns the prep tables into the in-memory content of ``sources.in``,
``receivers.in`` and ``otimes.dat`` plus a mapping table; :func:`write_fmtomo_data` writes them.

Design (verified against ``3dfm_main.f90`` and ``osv.io.fmtomo``):

* **Sources** are local earthquakes (``is_teleseismic = 0``) at the relocated hypocentres
  (``depth_km_bsl`` below the 6371 km sphere = sea level).  The model has its surface as
  interface 1, so a source in the top layer reaches the surface receivers with the direct
  path ``0 1`` (one step, from the source to interface 1); velocity type 1 is P and type 2 is S
  (``layout="joint"``, a ``vgrids.in`` holding both).  With ``layout="separate"`` (one velocity
  set per run directory) both phases use type 1.  One path per requested phase, in the order of
  ``phases``.
* **Receivers**: one per station alias that has at least one retained pick.  ``depth =
  -sensor_elevation_m / 1000`` (negative above sea level; a borehole sensor lies below the
  surface, hence deeper).  Each receiver lists the ``(source, path)`` pairs it recorded; the
  ``ray`` column of ``arrivals.dat`` is the position in that list.
* **otimes.dat** rows are ordered like fm3d's ``arrivals.dat``: source-major (source, then
  receiver, then ray), because ``residuals``/``invert3d`` pair ``otimes.dat`` with the model
  times row by row.  ``receivers.in`` lists each receiver's rays sorted by source, then path;
  the ``ray`` column holds the position in that list (not the path id, see
  ``osv.io.fmtomo.times``), ``subray`` is 0.
* **Observed travel time** = absolute pick time minus the origin time of the *relocated*
  hypocentre (Stage 2 ``final_hypocentres`` carries a ``time`` column, the CNV origin time,
  resolution 0.01 s).  A relocation shifts origin time and position together, so the Stage 1
  ``travel_time_s`` (catalogue origin) must not be reused.  The origin-time resolution of
  VELEST is of the order of the smallest pick uncertainties.
* **Station delays** (optional, ``model3d.apply_velest_station_delays``): VELEST models
  ``t_obs = t_calc + delay + residual`` so ``t_obs - delay`` is the time that a 1-D model alone
  would predict.  Applying them removes the near-surface (and mislocation) signal that the 1-D
  model could not explain; leaving them in lets the 3-D model absorb it.  Stations without a
  delay for a phase are left uncorrected (reported).
* **Invalid** observations (non-finite or non-positive travel time) are dropped by default
  (``keep_invalid=True`` keeps them with time ``-100``, which ``invert3d`` ignores).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.io.fmtomo.sources import (
    PathSpec,
    Receivers,
    Source,
    validate_receivers,
    validate_sources,
    write_receivers,
    write_sources,
)
from osv.io.fmtomo.times import INVALID_OBSERVED_BELOW, OBSERVED_COLUMNS, write_observed

__all__ = [
    "FmtomoData",
    "FmtomoDataError",
    "build_fmtomo_data",
    "load_velest_tables",
    "write_fmtomo_data",
]

log = logging.getLogger(__name__)

#: Observed time written for an invalid pick (programs treat anything below -50 as invalid).
INVALID_TIME_S = -100.0
MAPPING_COLUMNS = [
    "source_idx",
    "receiver_idx",
    "path_idx",
    "ray_idx",
    "event_id",
    "station",
    "phase",
    "pick_id",
    "time",
    "uncertainty",
    "valid",
]


class FmtomoDataError(ValueError):
    """The prep tables cannot be turned into consistent FMTOMO data (message says why)."""


@dataclass
class FmtomoData:
    """FMTOMO inputs derived from the prep tables.

    ``sources``/``receivers`` are the contents of ``sources.in``/``receivers.in``;
    ``observed`` is ``otimes.dat`` (columns ``receiver, source, ray, subray, time,
    uncertainty``) in ``arrivals.dat`` order; ``mapping`` has one row per observed row with
    1-based ``source_idx``, ``receiver_idx``, ``path_idx``, ``ray_idx`` and ``event_id``,
    ``station``, ``phase``, ``pick_id``, the written ``time``/``uncertainty`` and ``valid``.
    """

    sources: list[Source]
    receivers: Receivers
    observed: pd.DataFrame
    mapping: pd.DataFrame
    events: pd.DataFrame
    stations: pd.DataFrame
    phases: tuple[str, ...]
    findings: list[str] = field(default_factory=list)

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_receivers(self) -> int:
        return len(self.receivers.positions)

    @property
    def n_observations(self) -> int:
        return len(self.observed)


def load_velest_tables(stage_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read ``final_hypocentres.parquet`` and ``station_delays.csv`` of the Stage 2 analysis dir."""
    d = Path(stage_dir)
    out = {"hypocentre_table": pd.read_parquet(d / "final_hypocentres.parquet")}
    sd = d / "station_delays.csv"
    if sd.exists():
        out["station_delays"] = pd.read_csv(sd)
    return out


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def _vtypes(phases: tuple[str, ...], layout: str) -> list[int]:
    if layout not in ("joint", "separate"):
        raise FmtomoDataError(f"layout must be 'joint' or 'separate', got {layout!r}")
    return [2 if (ph == "S" and layout == "joint") else 1 for ph in phases]


def build_fmtomo_data(
    picks: pd.DataFrame,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    cfg: Any,
    grid: Any | None = None,
    hypocentres: str = "velest",
    subset: str = "calibration",
    phases: tuple[str, ...] = ("P",),
    *,
    hypocentre_table: pd.DataFrame | None = None,
    station_delays: pd.DataFrame | None = None,
    surface_depth_km: Callable[[np.ndarray, np.ndarray], np.ndarray] | float | None = None,
    layout: str = "joint",
    keep_invalid: bool = False,
) -> FmtomoData:
    """Build FMTOMO sources, receivers and observed times.

    Parameters
    ----------
    picks
        Stage 1 ``picks.parquet`` (``event_id, pick_id, phase, time, alias, sensor_elevation_m,
        sigma_s``; ``sta`` if no ``alias``).
    events
        Stage 1 ``events.parquet`` (``event_id, time, lat, lon, depth_km_bsl, split``).
    stations
        Stage 1 ``stations.parquet`` (``alias, lat, lon, sensor_elevation_m``).
    cfg
        ``Model3DConfig`` or anything with ``.model3d`` (reads ``apply_velest_station_delays``).
    grid
        Optional :class:`~osv.model3d.grid.ModelGrid`; sources and receivers are checked
        against its propagation grid (receiver depth only laterally, because receivers sit on
        the free surface which the top of the propagation grid need not exceed).
    hypocentres
        ``"velest"`` (Stage 2 relocations, ``hypocentre_table`` with ``event_id, time, lat, lon,
        depth_km``; events without one are dropped) or ``"catalog"`` (Stage 1 origins).
    subset
        ``"calibration"``, ``"validation"`` (via ``events.split``) or ``"all"``.
    phases
        ``("P",)``, ``("S",)`` or ``("P", "S")``; one path per phase in this order.
    station_delays
        VELEST delay table (``alias`` or ``sta``, ``ptcor``, ``stcor``); used only if
        ``model3d.apply_velest_station_delays`` is true.
    surface_depth_km
        Optional surface depth (km, negative above sea level) as a constant or a function
        ``(lat, lon) -> depth``; sources must lie below it.
    layout
        ``"joint"`` (S is velocity type 2) or ``"separate"`` (every phase type 1).
    keep_invalid
        Keep invalid picks with time ``-100`` instead of dropping them.
    """
    m3 = getattr(cfg, "model3d", cfg)
    apply_delays = bool(getattr(m3, "apply_velest_station_delays", False))
    phases = tuple(phases)
    if not phases or any(p not in ("P", "S") for p in phases) or len(set(phases)) != len(phases):
        raise FmtomoDataError(f"phases must be a non-empty subset of ('P', 'S'), got {phases!r}")
    if hypocentres not in ("velest", "catalog"):
        raise FmtomoDataError(f"hypocentres must be 'velest' or 'catalog', got {hypocentres!r}")
    if subset not in ("calibration", "validation", "all"):
        raise FmtomoDataError(f"subset must be calibration, validation or all, got {subset!r}")
    vt = _vtypes(phases, layout)
    findings: list[str] = []

    # ---- events -----------------------------------------------------------------------
    ev = events.copy()
    if subset != "all":
        if "split" not in ev:
            raise FmtomoDataError("events has no 'split' column; use subset='all'")
        ev = ev[ev["split"] == subset]
    if hypocentres == "velest":
        if hypocentre_table is None:
            raise FmtomoDataError(
                "hypocentres='velest' needs hypocentre_table (Stage 2 final_hypocentres; see "
                "load_velest_tables)"
            )
        h = hypocentre_table.drop_duplicates("event_id").set_index("event_id")
        have = ev["event_id"].isin(h.index)
        if (~have).any():
            findings.append(
                f"{int((~have).sum())} {subset} event(s) have no VELEST relocation and are "
                "dropped (use hypocentres='catalog' for them)"
            )
        ev = ev[have].copy()
        hh = h.loc[ev["event_id"]]
        ev["lat"] = hh["lat"].to_numpy()
        ev["lon"] = hh["lon"].to_numpy()
        ev["depth_km_bsl"] = hh["depth_km"].to_numpy()  # VELEST run with zshift = 0 (sea level)
        ev["origin_time"] = _utc(hh["time"]).to_numpy()
    else:
        if "depth_km_bsl" not in ev:
            raise FmtomoDataError("events lacks depth_km_bsl (run Stage 1 depth-datum step)")
        ev["origin_time"] = _utc(ev["time"]).to_numpy()
    if ev.empty:
        raise FmtomoDataError(
            f"no events left for subset={subset!r}, hypocentres={hypocentres!r}"
            + (
                "; VELEST relocates only the calibration events"
                if hypocentres == "velest" and subset == "validation"
                else ""
            )
        )
    ev = ev.sort_values(["origin_time", "event_id"]).reset_index(drop=True)
    ev["source_idx"] = np.arange(1, len(ev) + 1)

    # ---- picks ------------------------------------------------------------------------
    pk = picks[picks["phase"].isin(phases) & picks["event_id"].isin(ev["event_id"])].copy()
    sta_col = "alias" if "alias" in pk else "sta"
    pk["station"] = pk[sta_col]
    n_before = len(pk)
    pk = pk.sort_values(["sigma_s", "pick_id"]).drop_duplicates(
        ["event_id", "station", "phase"], keep="first"
    )
    if len(pk) < n_before:
        findings.append(
            f"{n_before - len(pk)} duplicate (event, station, phase) pick(s) dropped "
            "(smallest sigma_s kept)"
        )
    pk = pk.merge(ev[["event_id", "source_idx", "origin_time"]], on="event_id", how="left")
    pk["time"] = _utc(pk["time"])
    tt = (pk["time"] - pk["origin_time"]).dt.total_seconds().to_numpy()

    if apply_delays and station_delays is None:
        findings.append(
            "model3d.apply_velest_station_delays is on but no station delays were given; "
            "observed times are uncorrected"
        )
        apply_delays = False
    if apply_delays:
        sd = station_delays.copy()
        key = "alias" if "alias" in sd else "sta"
        sd = sd.drop_duplicates(key).set_index(key)
        corr = np.zeros(len(pk))
        missing = 0
        for ph, col in (("P", "ptcor"), ("S", "stcor")):
            sel = (pk["phase"] == ph).to_numpy()
            d = (
                pk.loc[sel, "station"].map(sd[col])
                if col in sd
                else pd.Series(np.nan, pk.index[sel])
            )
            dv = np.array(d, dtype=float)
            bad = ~np.isfinite(dv)
            missing += int(bad.sum())
            dv[bad] = 0.0
            corr[sel] = dv
        tt = tt - corr
        if missing:
            findings.append(f"{missing} pick(s) lack a VELEST station delay and are uncorrected")
        findings.append("VELEST station delays (t_obs - delay) applied to observed times")

    sigma = pk["sigma_s"].to_numpy(float)
    valid = np.isfinite(tt) & (tt > 0) & np.isfinite(sigma) & (sigma > 0)
    if (~valid).any():
        findings.append(
            f"{int((~valid).sum())} pick(s) with non-positive/non-finite travel time or "
            f"uncertainty " + ("marked invalid" if keep_invalid else "dropped")
        )
    pk["time_s"] = np.where(valid, tt, INVALID_TIME_S)
    pk["unc"] = np.where(valid & np.isfinite(sigma), sigma, 1.0)
    pk["valid"] = valid
    if not keep_invalid:
        pk = pk[pk["valid"]]
    if pk.empty:
        raise FmtomoDataError("no valid picks remain for the requested events and phases")

    # ---- receivers --------------------------------------------------------------------
    st = stations.copy()
    skey = "alias" if "alias" in st else "sta"
    st = st.drop_duplicates(skey).set_index(skey)
    names = sorted(pk["station"].unique())
    unknown = [n for n in names if n not in st.index]
    if unknown:
        raise FmtomoDataError(f"picks refer to stations absent from the station table: {unknown}")
    sel = st.loc[names]
    elev = _sensor_elevation(sel)
    rpos = pd.DataFrame(
        {
            "depth_km": -elev / 1000.0,
            "lat": sel["lat"].to_numpy(float),
            "lon": sel["lon"].to_numpy(float),
        },
        index=pd.RangeIndex(1, len(names) + 1, name="receiver"),
    )
    rec_idx = {n: i + 1 for i, n in enumerate(names)}
    pk["receiver_idx"] = pk["station"].map(rec_idx).astype(int)
    pk["path_idx"] = pk["phase"].map({p: i + 1 for i, p in enumerate(phases)}).astype(int)

    # rays in file order: receiver, then source, then path
    pk = pk.sort_values(["receiver_idx", "source_idx", "path_idx"]).reset_index(drop=True)
    pk["ray_idx"] = pk.groupby("receiver_idx").cumcount() + 1
    rays = pk[["receiver_idx", "source_idx", "path_idx"]].rename(
        columns={"receiver_idx": "receiver", "source_idx": "source", "path_idx": "path"}
    )
    receivers = Receivers(rpos, rays.astype("int64").reset_index(drop=True))

    # ---- sources (only events that kept picks) ----------------------------------------
    used = sorted(pk["source_idx"].unique())
    if len(used) < len(ev):
        findings.append(f"{len(ev) - len(used)} event(s) without retained picks dropped")
        remap = {old: new for new, old in enumerate(used, start=1)}
        ev = ev[ev["source_idx"].isin(used)].copy()
        ev["source_idx"] = ev["source_idx"].map(remap)
        pk["source_idx"] = pk["source_idx"].map(remap)
        receivers.rays["source"] = receivers.rays["source"].map(remap).astype("int64")
        ev = ev.sort_values("source_idx").reset_index(drop=True)
    paths = [PathSpec.direct(v) for v in vt]
    sources = [
        Source(float(r.depth_km_bsl), float(r.lat), float(r.lon), list(paths))
        for r in ev.itertuples(index=False)
    ]

    # otimes/mapping rows in fm3d arrivals.dat order: source-major (source, receiver, ray)
    pk = pk.sort_values(["source_idx", "receiver_idx", "ray_idx"]).reset_index(drop=True)
    observed = pd.DataFrame(
        {
            "receiver": pk["receiver_idx"].to_numpy(),
            "source": pk["source_idx"].to_numpy(),
            "ray": pk["ray_idx"].to_numpy(),
            "subray": 0,
            "time": pk["time_s"].to_numpy(),
            "uncertainty": pk["unc"].to_numpy(),
        }
    )[OBSERVED_COLUMNS]
    observed["valid"] = observed["time"] >= INVALID_OBSERVED_BELOW
    mapping = pd.DataFrame(
        {
            "source_idx": pk["source_idx"],
            "receiver_idx": pk["receiver_idx"],
            "path_idx": pk["path_idx"],
            "ray_idx": pk["ray_idx"],
            "event_id": pk["event_id"],
            "station": pk["station"],
            "phase": pk["phase"],
            "pick_id": pk["pick_id"],
            "time": pk["time_s"],
            "uncertainty": pk["unc"],
            "valid": pk["valid"],
        }
    )[MAPPING_COLUMNS]

    ev_out = ev[["source_idx", "event_id", "origin_time", "lat", "lon", "depth_km_bsl"]].copy()
    st_out = rpos.copy()
    st_out.insert(0, "station", names)
    data = FmtomoData(sources, receivers, observed, mapping, ev_out, st_out, phases, findings)
    findings.extend(_validate(data, grid, surface_depth_km))
    for f in findings:
        log.info("fmtomo data: %s", f)
    return data


def _sensor_elevation(sel: pd.DataFrame) -> np.ndarray:
    if "sensor_elevation_m" in sel:
        e = sel["sensor_elevation_m"].to_numpy(float)
    elif "elevation_m" in sel:
        e = sel["elevation_m"].to_numpy(float)
    else:
        raise FmtomoDataError("station table needs sensor_elevation_m (or elevation_m)")
    if not np.isfinite(e).all():
        raise FmtomoDataError("station table has non-finite elevations")
    return e


def _validate(
    data: FmtomoData,
    grid: Any | None,
    surface: Callable[[np.ndarray, np.ndarray], np.ndarray] | float | None,
) -> list[str]:
    """Consistency checks; raises on structural problems, returns warnings as findings."""
    out: list[str] = []
    try:
        validate_sources(data.sources, n_vtypes=None)
        validate_receivers(data.receivers, data.sources)
    except Exception as exc:  # FmtomoFormatError
        raise FmtomoDataError(str(exc)) from exc
    m, obs, rays = data.mapping, data.observed, data.receivers.rays
    if not (len(m) == len(obs) == len(rays)):
        raise FmtomoDataError(
            f"count mismatch: {len(m)} mapping rows, {len(obs)} observations, {len(rays)} rays"
        )
    dup = m.duplicated(["source_idx", "receiver_idx", "path_idx"])
    if dup.any():
        raise FmtomoDataError(f"{int(dup.sum())} duplicate (source, receiver, path) rows")
    expect = rays.assign(ray=rays.groupby("receiver").cumcount() + 1).sort_values(
        ["source", "receiver", "ray"], kind="stable"
    )
    got = obs[["source", "receiver", "ray"]].to_numpy()
    if not (expect[["source", "receiver", "ray"]].to_numpy() == got).all():
        raise FmtomoDataError(
            "otimes rows are not in fm3d arrivals order (source, receiver, ray) "
            "consistent with receivers.in"
        )

    src = data.events
    if surface is not None:
        sd = (
            np.full(len(src), float(surface))
            if np.isscalar(surface)
            else np.asarray(surface(src["lat"].to_numpy(float), src["lon"].to_numpy(float)))
        )
        above = src["depth_km_bsl"].to_numpy(float) <= sd
        if above.any():
            raise FmtomoDataError(
                f"{int(above.sum())} source(s) lie at or above the surface, e.g. event "
                f"{src['event_id'][above].iloc[0]}"
            )
    if grid is not None:
        ok = grid.contains(src["lat"], src["lon"], src["depth_km_bsl"])
        if not ok.all():
            raise FmtomoDataError(
                f"{int((~ok).sum())} source(s) outside the propagation grid, e.g. event "
                f"{src['event_id'][~ok].iloc[0]}; enlarge grid.pad_km/zmax_km"
            )
        pos = data.receivers.positions
        top = float(grid.propgrid.depth_km[0])
        ok = grid.contains(pos["lat"], pos["lon"], np.full(len(pos), top))
        if not ok.all():
            raise FmtomoDataError(
                f"{int((~ok).sum())} receiver(s) outside the propagation grid laterally, e.g. "
                f"{data.stations['station'][~ok.astype(bool)].iloc[0]}"
            )
        deep = pos["depth_km"].to_numpy(float) > float(grid.propgrid.depth_km[-1])
        if deep.any():
            raise FmtomoDataError(f"{int(deep.sum())} receiver(s) below the propagation grid")
        if (pos["depth_km"] < top).any():
            out.append(
                f"{int((pos['depth_km'] < top).sum())} receiver(s) above the propagation-grid "
                "top (fine if the surface interface follows the topography to the receivers)"
            )
    if data.n_observations and not data.mapping["valid"].all():
        out.append(f"{int((~data.mapping['valid']).sum())} invalid observation(s) kept")
    return out


def write_fmtomo_data(
    data: FmtomoData,
    outdir: str | Path,
    *,
    source_uncertainty: tuple[float, float, float] | None = None,
) -> dict[str, Path]:
    """Write ``sources.in``, ``sourcesref.in``, ``receivers.in``, ``otimes.dat`` and the mapping.

    ``sourcesref.in`` repeats the sources, with the a priori uncertainties
    ``(depth km, lat deg, lon deg)`` if ``source_uncertainty`` is given (needed when
    ``invert3d`` relocates sources).  The mapping table is ``otimes_mapping.csv``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ref = [
        Source(s.depth_km, s.lat, s.lon, list(s.paths), uncertainty=source_uncertainty)
        for s in data.sources
    ]
    files = {
        "sources.in": write_sources(out / "sources.in", data.sources),
        "sourcesref.in": write_sources(out / "sourcesref.in", ref),
        "receivers.in": write_receivers(out / "receivers.in", data.receivers),
        "otimes.dat": write_observed(out / "otimes.dat", data.observed),
    }
    mp = out / "otimes_mapping.csv"
    data.mapping.to_csv(mp, index=False)
    files["otimes_mapping.csv"] = mp
    return files
