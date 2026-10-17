"""Write the VELEST event (CNV) and station files from the selected tables (task 2.6).

:func:`write_velest_inputs` turns a :class:`~osv.data.select.Selection` and the
:class:`~osv.data.stations.StationTables` into ``events.cnv`` and ``stations.sta`` plus a
pre-filled :class:`~osv.io.velest.VelestControl` (VELEST is *not* run here).

Datum decisions (``docs/conventions.md``, "Depth datum")
-------------------------------------------------------
* CNV depth = ``depth_km_bsl`` (km below sea level; ``depth_km`` is used, assumed to be
  below sea level, only when the column is missing, with a finding).
* ``zshift = 0`` and ``iuseelev = 1``: the 1-D model depths are measured from sea level and
  the station z is ``-elevation/1000`` km (``velest.f:1551-1563``).  Borehole sensors are
  therefore *below* the surface (sensor elevation = surface elevation - sensor depth, from
  :func:`osv.coords.borehole_sensor_elevation_m`, already stored as ``sensor_elevation_m``).
  With ``coordinates.borehole_sensors = "ignore_depth"`` the surface elevation is written.
* Station-file elevation is an integer in metres (``i4``, -999..9999 m).
* ``zmin`` of the control is lowered to 0.1 km above the shallowest event when events lie
  above sea level, because VELEST moves every hypocentre at or above ``zmin`` to
  ``zmin + 0.1`` (``velest.f:4385-4387``).

Reference station
-----------------
In the station file ``icc`` is the *station-correction index*: ``0`` holds the station delay
fixed, and the station with the **highest** ``icc`` has its P delay held fixed while its S
delay floats (comment ``velest.f:1580-1585``; ``ml = max(icc)`` at ``velest.f:1622``; the
solution has ``ksta = ml - 1`` station-correction unknowns, ``velest.f:1625-1629``).
VELEST does not use the first station of the file.  This module numbers the stations
``icc = 1..n`` in file order and puts the reference station **last**, so it gets the
highest ``icc``.  Default choice: among the stations that recorded at least half as many
usable picks as the best-recorded station, the one nearest the centroid of all stations.

Limits (``fortran/velest/velest_common.inc``): ``ieq = 5000`` events, ``ist = 300``
stations, ``maxobsperevent = 180`` observations per event (all weights count, also class 4);
exceeding one raises :class:`VelestLimitError` rather than letting VELEST fail obscurely.
All three are arguments of :func:`write_velest_inputs` in case the build is patched.

Magnitudes: invalid catalogue magnitudes (``mag_valid`` False, NaN) and negative magnitudes
are written as ``0.0`` (VELEST only copies the value; ``f5.2`` cannot hold values below
-9.99 and the legacy files never carry negative magnitudes).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from osv.config.schema import OSVConfig
from osv.data.select import REJECT_WEIGHT, Selection
from osv.data.stations import StationTables
from osv.io.velest import (
    MAX_OBS_PER_EVENT,
    PHASE_CHARS,
    VelestControl,
    write_cnv,
    write_stations,
)
from osv.report.checks import QCFinding
from osv.velest.vpvs import nsp_for_mode

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_STATIONS",
    "MAX_VELEST_EVENTS",
    "TT_MAX_S",
    "VelestInputError",
    "VelestInputs",
    "VelestLimitError",
    "choose_reference_station",
    "write_velest_inputs",
]

#: ``ieq`` of ``velest_common.inc`` (simultaneous mode up to ~5555).
MAX_VELEST_EVENTS = 5000
#: ``ist`` of ``velest_common.inc``.
MAX_STATIONS = 300
#: Travel times must be below this (``f6.2`` relative to the origin time).
TT_MAX_S = 100.0
#: Positions of one alias differing by more than this (degrees, ~11 m) count as moved.
_POS_DECIMALS = 4

Subset = Literal["calibration", "validation", "all"]


class VelestInputError(ValueError):
    """The selected data cannot be written as VELEST input."""


class VelestLimitError(VelestInputError):
    """A compiled VELEST array limit (events, stations, observations per event) is exceeded."""


@dataclass
class VelestInputs:
    """Result of :func:`write_velest_inputs`."""

    cnv_path: Path
    sta_path: Path
    summary_path: Path
    event_map_path: Path
    control: VelestControl
    subset: str
    n_events: int
    n_picks: int
    n_stations: int
    n_picks_by_weight: dict[int, int]
    n_picks_dropped: int
    reference_station: str
    events: pd.DataFrame  # written events: cnv_id, event_id, time, lat, lon, depth_km, mag, ...
    stations: pd.DataFrame  # written station table (VELEST columns, in file order)
    zshift: float = 0.0
    iuseelev: int = 1
    findings: list[QCFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def n_picks_unused(self) -> int:
        """Picks of weight class 4 (written, ignored by VELEST)."""
        return self.n_picks_by_weight.get(REJECT_WEIGHT, 0)


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def _subset_events(sel: Selection, subset: str) -> pd.DataFrame:
    if subset not in ("calibration", "validation", "all"):
        raise ValueError(f"subset must be 'calibration', 'validation' or 'all', got {subset!r}")
    ev = sel.events
    if subset == "all":
        return ev.copy()
    split = (
        ev["split"] if "split" in ev.columns else ev["event_id"].map(sel.split)  # type: ignore[arg-type]
    )
    return ev[split.to_numpy() == subset].copy()


def _station_positions(
    picks: pd.DataFrame, stations: StationTables, borehole: str, findings: list[QCFinding]
) -> pd.DataFrame:
    """One ``alias, lat, lon, elev_m`` row per alias used by ``picks``."""
    use_sensor = borehole == "use_depth_from_stationxml"
    elev_col = "sensor_elevation_m" if use_sensor else "station_elevation_m"
    needed = {"station_lat", "station_lon", elev_col}
    if needed <= set(picks.columns):
        pos = picks[["alias", "station_lat", "station_lon", elev_col]].rename(
            columns={"station_lat": "lat", "station_lon": "lon", elev_col: "elev_m"}
        )
    else:  # fall back to the station table (latest epoch of each alias)
        st = stations.stations
        col = "sensor_elevation_m" if use_sensor else "elevation_m"
        st = st.sort_values("start", na_position="first") if "start" in st else st
        last = st.drop_duplicates("alias", keep="last").set_index("alias")
        missing = sorted(set(picks["alias"]) - set(last.index))
        if missing:
            raise VelestInputError(
                f"picks refer to station aliases missing from the station table: {missing[:5]}"
            )
        pos = pd.DataFrame(
            {
                "alias": picks["alias"].to_numpy(),
                "lat": last.loc[picks["alias"], "lat"].to_numpy(),
                "lon": last.loc[picks["alias"], "lon"].to_numpy(),
                "elev_m": last.loc[picks["alias"], col].to_numpy(),
            }
        )
    pos = pos.astype({"lat": float, "lon": float, "elev_m": float})
    if pos[["lat", "lon", "elev_m"]].isna().any(axis=None):
        bad = sorted(pos.loc[pos.isna().any(axis=1), "alias"].unique())
        raise VelestInputError(f"stations without coordinates/elevation: {bad[:5]}")
    key = pos.assign(
        _la=pos["lat"].round(_POS_DECIMALS),
        _lo=pos["lon"].round(_POS_DECIMALS),
        _el=pos["elev_m"].round(0),
    )
    rows = []
    moved: dict[str, int] = {}
    for alias, grp in key.groupby("alias", sort=True):
        counts = (
            grp.groupby(["_la", "_lo", "_el"]).size().sort_values(ascending=False, kind="stable")
        )
        if len(counts) > 1:
            moved[str(alias)] = len(counts)
        la, lo, el = counts.index[0]
        rows.append({"alias": alias, "lat": la, "lon": lo, "elev_m": int(round(el))})
    if moved:
        findings.append(
            QCFinding(
                "station_epochs_collapsed",
                "warn",
                f"{len(moved)} stations have more than one position/elevation among the picks "
                "(station moved or re-installed); the most frequently used one is written "
                "because the VELEST station file holds one position per station.",
                len(moved),
                {"epochs_per_alias": moved},
            )
        )
    return pd.DataFrame(rows)


def choose_reference_station(
    stations: pd.DataFrame, n_usable_picks: pd.Series, min_fraction: float = 0.5
) -> str:
    """Pick the reference station: well recorded and central.

    Candidates are the stations with at least ``min_fraction`` of the largest usable-pick
    count; the one closest to the centroid of *all* stations (small-area flat approximation)
    is returned.  Ties go to the larger pick count, then the alias.
    """
    st = stations.set_index("alias")
    n = n_usable_picks.reindex(st.index).fillna(0)
    cands = n[n >= min_fraction * n.max()].index
    lat0 = float(st["lat"].mean())
    lon0 = float(st["lon"].mean())
    cl = np.cos(np.radians(lat0))
    d2 = (st["lat"] - lat0) ** 2 + ((st["lon"] - lon0) * cl) ** 2
    order = sorted(cands, key=lambda a: (round(float(d2[a]), 12), -float(n[a]), str(a)))
    return str(order[0])


def _clean_picks(
    picks: pd.DataFrame, id_map: dict[Any, int], findings: list[QCFinding]
) -> tuple[pd.DataFrame, dict[str, int]]:
    pk = picks[picks["event_id"].isin(id_map)].copy()
    drops: dict[str, int] = {}
    pk["tt"] = pk["travel_time_s"].astype(float)

    n_neg = int((pk["tt"] < 0.0).sum())
    bad_phase = ~pk["phase"].isin(list(PHASE_CHARS))
    n = int(bad_phase.sum())
    if n:
        drops["phase"] = n
        findings.append(
            QCFinding(
                "picks_dropped_phase",
                "warn",
                f"{n} picks dropped: phase not one of {list(PHASE_CHARS)}.",
                n,
            )
        )
        pk = pk[~bad_phase]
    bad_tt = ~(np.isfinite(pk["tt"]) & (pk["tt"] > 0.0) & (pk["tt"].round(2) < TT_MAX_S))
    n = int(bad_tt.sum())
    if n:
        drops["travel_time"] = n
        findings.append(
            QCFinding(
                "picks_dropped_travel_time",
                "warn",
                f"{n} picks dropped: travel time outside (0, {TT_MAX_S:g}) s ({n_neg} negative); "
                "VELEST reads f6.2 seconds after the origin time.",
                n,
            )
        )
        pk = pk[~bad_tt]

    # one pick per (event, station, phase): keep the best weight class, then earliest
    pk = pk.sort_values(["event_id", "alias", "phase", "velest_weight", "tt"], kind="stable")
    dup = pk.duplicated(["event_id", "alias", "phase"], keep="first")
    if dup.any():
        drops["duplicate"] = int(dup.sum())
        findings.append(
            QCFinding(
                "picks_dropped_duplicate",
                "warn",
                f"{int(dup.sum())} picks dropped: more than one pick of a phase at one station "
                "of an event (VELEST accepts one each); the best weight was kept.",
                int(dup.sum()),
            )
        )
        pk = pk[~dup]
    return pk, drops


# ----------------------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------------------


def write_velest_inputs(
    selection: Selection,
    stations: StationTables,
    cfg: OSVConfig,
    outdir: str | Path,
    subset: Subset = "calibration",
    *,
    max_events: int = MAX_VELEST_EVENTS,
    max_stations: int = MAX_STATIONS,
    max_obs_per_event: int = MAX_OBS_PER_EVENT,
    reference_station: str | None = None,
    cnv_name: str = "events.cnv",
    sta_name: str = "stations.sta",
) -> VelestInputs:
    """Write ``events.cnv`` and ``stations.sta`` for ``subset`` and return the details.

    Parameters
    ----------
    selection
        Output of :func:`osv.data.select.select_data` (``events`` with ``depth_km_bsl``,
        ``split``; ``picks`` with ``alias``, ``phase``, ``travel_time_s``, ``velest_weight``).
    stations
        Station tables (used for positions when the picks do not carry them).
    cfg
        Configuration (``coordinates``, ``data_selection``, ``velest`` are used).
    outdir
        Output directory (created).
    subset
        ``'calibration'`` (default), ``'validation'`` or ``'all'``.
    max_events, max_stations, max_obs_per_event
        Compiled VELEST limits (``ieq``, ``ist``, ``maxobsperevent``); exceeding raises
        :class:`VelestLimitError`.
    reference_station
        Alias of the reference station; default is chosen by
        :func:`choose_reference_station`.

    Returns
    -------
    VelestInputs
        Paths, counts, the reference station, a pre-filled ``VelestControl`` (no model file;
        ``zshift = 0``, ``iuseelev = 1``) and QC findings.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    findings: list[QCFinding] = []

    ev = _subset_events(selection, subset)
    if ev.empty:
        raise VelestInputError(f"no events in subset {subset!r}; nothing to write")
    ev = ev.assign(_t=pd.to_datetime(ev["time"], utc=True)).sort_values(
        ["_t", "event_id"], kind="stable"
    )
    id_map = {eid: i + 1 for i, eid in enumerate(ev["event_id"])}

    pk, drops = _clean_picks(selection.picks, id_map, findings)

    # events without a usable (weight < 4) pick cannot be located: drop them
    usable = pk[pk["velest_weight"] < REJECT_WEIGHT]
    keep = ev["event_id"].isin(usable["event_id"])
    if not keep.all():
        n = int((~keep).sum())
        findings.append(
            QCFinding(
                "events_dropped_no_usable_picks",
                "warn",
                f"{n} events dropped: no pick left with weight < {REJECT_WEIGHT}.",
                n,
                {"event_id": [str(x) for x in ev.loc[~keep, "event_id"]][:20]},
            )
        )
        ev = ev[keep]
        pk = pk[pk["event_id"].isin(ev["event_id"])]
    if ev.empty:
        raise VelestInputError("no events with usable picks; check the selection")
    id_map = {eid: i + 1 for i, eid in enumerate(ev["event_id"])}
    pk = pk.assign(cnv_id=pk["event_id"].map(id_map))

    # limits
    if len(ev) > max_events:
        raise VelestLimitError(
            f"{len(ev)} events exceed the VELEST limit ieq={max_events}; select fewer events "
            "(data_selection filters / validation split) or rebuild VELEST with a larger ieq"
        )
    per_event = pk.groupby("cnv_id").size()
    if int(per_event.max()) > max_obs_per_event:
        worst = int(per_event.idxmax())
        raise VelestLimitError(
            f"event {ev.iloc[worst - 1]['event_id']} has {int(per_event.max())} picks, above "
            f"VELEST's maxobsperevent={max_obs_per_event} "
            f"({int((per_event > max_obs_per_event).sum())} events affected); "
            "lower data_selection.max_epi_dist_km or thin the picks"
        )

    # depth and magnitude
    if "depth_km_bsl" in ev.columns:
        depth = ev["depth_km_bsl"].astype(float)
        depth_col = "depth_km_bsl"
    else:
        depth = ev["depth_km"].astype(float)
        depth_col = "depth_km"
        findings.append(
            QCFinding(
                "depth_datum_assumed",
                "warn",
                "events have no depth_km_bsl column (apply_depth_datum was not run); depth_km "
                "is written and assumed to be below sea level.",
                len(ev),
            )
        )
    if depth.isna().any():
        raise VelestInputError("some events have no depth; cannot write the CNV file")
    mag = ev["mag"].astype(float) if "mag" in ev.columns else pd.Series(np.nan, index=ev.index)
    valid = (
        ev["mag_valid"].astype(bool)
        if "mag_valid" in ev.columns
        else pd.Series(True, index=ev.index)
    )
    bad = ~valid | ~np.isfinite(mag)
    neg = ~bad & (mag < 0.0)
    mag_out = mag.where(~(bad | neg), 0.0)
    if bad.any():
        findings.append(
            QCFinding(
                "magnitude_written_zero",
                "info",
                f"{int(bad.sum())} events have an invalid/missing catalogue magnitude; 0.0 is "
                "written in the CNV (VELEST does not use it).",
                int(bad.sum()),
            )
        )
    if neg.any():
        findings.append(
            QCFinding(
                "magnitude_negative_clamped",
                "info",
                f"{int(neg.sum())} negative magnitudes written as 0.0.",
                int(neg.sum()),
            )
        )

    t_naive = pd.to_datetime(ev["_t"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    events_out = pd.DataFrame(
        {
            "cnv_id": np.arange(1, len(ev) + 1),
            "event_id": ev["event_id"].to_numpy(),
            "time": t_naive.astype("datetime64[ns]").to_numpy(),
            "lat": ev["lat"].astype(float).to_numpy(),
            "lon": ev["lon"].astype(float).to_numpy(),
            "depth_km": depth.to_numpy(),
            "mag": mag_out.to_numpy(),
        }
    )
    events_out["split"] = (
        ev["split"].to_numpy()
        if "split" in ev.columns
        else ev["event_id"].map(selection.split).to_numpy()
    )

    # stations
    borehole = cfg.coordinates.borehole_sensors
    pos = _station_positions(pk, stations, borehole, findings)
    if len(pos) > max_stations:
        raise VelestLimitError(
            f"{len(pos)} stations exceed the VELEST limit ist={max_stations}; restrict the "
            "stations/region or rebuild VELEST with a larger ist"
        )
    n_use = usable.groupby("alias").size()
    if reference_station is None:
        reference_station = choose_reference_station(pos, n_use)
    elif reference_station not in set(pos["alias"]):
        raise VelestInputError(
            f"reference station {reference_station!r} has no picks in subset {subset!r}; "
            f"available: {sorted(pos['alias'])[:10]}..."
        )
    order = [*sorted(a for a in pos["alias"] if a != reference_station), reference_station]
    pos = pos.set_index("alias").loc[order].reset_index()
    sta_out = pd.DataFrame(
        {
            "sta": pos["alias"],
            "lat": pos["lat"],
            "lon": pos["lon"],
            "elev_m": pos["elev_m"],
            "model": 1,
            "icc": np.arange(1, len(pos) + 1),
            "ptcor": 0.0,
            "stcor": 0.0,
            "n_picks": pos["alias"].map(pk.groupby("alias").size()).astype(int),
            "n_usable": pos["alias"].map(n_use).fillna(0).astype(int),
        }
    )
    findings.append(
        QCFinding(
            "reference_station",
            "info",
            f"Reference station {reference_station} (highest icc={len(pos)}: its P delay is held "
            f"fixed by VELEST, velest.f:1580-1629); {int(n_use.get(reference_station, 0))} "
            "usable picks, nearest of the well-recorded stations to the array centroid.",
            1,
            {"alias": reference_station, "icc": len(pos)},
        )
    )
    el = sta_out["elev_m"]
    if (el < -999).any() or (el > 9999).any():
        raise VelestInputError("station elevations outside -999..9999 m cannot be written (i4)")

    # write
    cnv_picks = pd.DataFrame(
        {
            "event": pk["cnv_id"].astype(int).to_numpy(),
            "sta": pk["alias"].to_numpy(),
            "phase": pk["phase"].to_numpy(),
            "weight": pk["velest_weight"].astype(int).to_numpy(),
            "tt": pk["tt"].to_numpy(),
        }
    ).sort_values(["event", "tt"], kind="stable")
    cnv_events = events_out.rename(columns={"cnv_id": "event"})[
        ["event", "time", "lat", "lon", "depth_km", "mag"]
    ]
    cnv_path = write_cnv(
        out / cnv_name, cnv_events, cnv_picks, zshift_km=0.0, max_obs=max_obs_per_event
    )
    sta_path = write_stations(out / sta_name, sta_out.drop(columns=["n_picks", "n_usable"]))

    by_w = cnv_picks["weight"].value_counts().reindex(range(5), fill_value=0)
    n_by_weight = {int(k): int(v) for k, v in by_w.items()}
    if n_by_weight[REJECT_WEIGHT]:
        findings.append(
            QCFinding(
                "weight_4_picks_written",
                "info",
                f"{n_by_weight[REJECT_WEIGHT]} weight-4 picks are written (VELEST ignores them) "
                "and count against maxobsperevent.",
                n_by_weight[REJECT_WEIGHT],
            )
        )

    # control
    origin = cfg.coordinates.origin
    if origin is not None:
        olat, olon = float(origin.lat), float(origin.lon)
    else:
        olat = float(sta_out["lat"].mean())
        lon_rel = (sta_out["lon"] - sta_out["lon"].iloc[0] + 180.0) % 360.0 - 180.0
        olon = float(sta_out["lon"].iloc[0] + lon_rel.mean())
        olat, olon = round(olat, 4), round(((olon + 180.0) % 360.0) - 180.0, 4)
        findings.append(
            QCFinding(
                "origin_from_station_centroid",
                "info",
                f"coordinates.origin is not set: the station centroid ({olat}, {olon}) is used "
                "as the VELEST origin.",
            )
        )
    vcfg = cfg.velest
    zmin = 0.0
    dmin = float(depth.min())
    if dmin <= 0.0:
        zmin = float(np.floor((dmin - 0.1) * 10.0) / 10.0)
        findings.append(
            QCFinding(
                "zmin_lowered",
                "info",
                f"Shallowest event is at {dmin:.2f} km (above sea level); zmin set to {zmin} km "
                "so VELEST does not push events down to zmin + 0.1 km (velest.f:4385). The "
                "model top must be at or above zmin.",
            )
        )
    control = VelestControl.from_origin(
        olat,
        olon,
        title=f"osv {cfg.project.name} {subset}"[:80],
        neqs=len(events_out),
        zshift=0.0,
        iuseelev=1,
        zmin=zmin,
        dmax=float(cfg.data_selection.max_epi_dist_km),
        nsp=nsp_for_mode(vcfg.vpvs.mode),
        vpvs=(
            float(sum(vcfg.vpvs.range) / 2.0)
            if vcfg.vpvs.mode == "sampled"
            else float(vcfg.vpvs.value)
        ),
        othet=vcfg.damping.othet,
        xythet=vcfg.damping.xythet,
        zthet=vcfg.damping.zthet,
        vthet=vcfg.damping.vthet,
        stathet=vcfg.damping.stathet,
        ittmax=vcfg.ittmax,
        invertratio=vcfg.invertratio,
        stationfile=sta_name,
        phasefile=cnv_name,
    )

    summary: dict[str, Any] = {
        "subset": subset,
        "files": {"cnv": cnv_path.name, "sta": sta_path.name},
        "n_events": len(events_out),
        "n_picks": int(len(cnv_picks)),
        "n_picks_by_weight": n_by_weight,
        "n_picks_dropped": drops,
        "n_stations": len(sta_out),
        "reference_station": reference_station,
        "reference_icc": len(sta_out),
        "depth_column": depth_col,
        "depth_km_bsl_range": [float(depth.min()), float(depth.max())],
        "time_range": [str(events_out["time"].min()), str(events_out["time"].max())],
        "zshift": 0.0,
        "iuseelev": 1,
        "borehole_sensors": borehole,
        "station_elevation_m_range": [int(el.min()), int(el.max())],
        "origin_lat_lon_east": [olat, olon],
        "max_picks_per_event": int(per_event.max()),
        "limits": {
            "max_events": max_events,
            "max_stations": max_stations,
            "max_obs_per_event": max_obs_per_event,
        },
        "picks_per_station": {
            str(r.sta): {"all": int(r.n_picks), "usable": int(r.n_usable)}
            for r in sta_out.itertuples()
        },
        "findings": [f.to_dict() for f in findings],
    }
    summary_path = out / "velest_inputs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    event_map_path = out / "event_map.csv"
    events_out.to_csv(event_map_path, index=False)
    for f in findings:
        logger.info("%s", f)
    return VelestInputs(
        cnv_path=cnv_path,
        sta_path=sta_path,
        summary_path=summary_path,
        event_map_path=event_map_path,
        control=control,
        subset=subset,
        n_events=len(events_out),
        n_picks=int(len(cnv_picks)),
        n_stations=len(sta_out),
        n_picks_by_weight=n_by_weight,
        n_picks_dropped=int(sum(drops.values())),
        reference_station=reference_station,
        events=events_out,
        stations=sta_out,
        findings=findings,
        summary=summary,
    )
