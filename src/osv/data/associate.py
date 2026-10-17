"""Pick-station association, de-duplication, pick weighting and geometry (Stage 1, task 2.3).

:func:`associate_picks` takes the catalogue tables (:func:`osv.data.quakeml.read_catalog`)
and the station tables (:func:`osv.data.stations.read_stations`) and returns clean pick and
event tables plus QC findings.  Steps, in order:

1. keep only ``P`` and ``S`` picks (the phase label alone decides; the channel is recorded
   in ``cha`` but never used to filter, so S on EHT, EH1, EHE, ... is accepted);
2. drop picks whose event is not in the event table;
3. join each pick to the station epoch active at the pick time; unknown stations and picks
   outside every epoch are dropped (findings carry counts per station);
4. de-duplicate per ``(event_id, net, sta, phase)``: smallest ``unc_s`` first (missing last),
   then highest ``comment_weight``, then earliest pick time;
5. compute ``sigma_s`` according to ``inputs.weight_source`` (see below);
6. add geometry columns and event-level statistics.

Depth datum
-----------
Distances use the event depth in km positive down and the sensor depth
``-sensor_elevation_m / 1000`` (km below sea level).  The event depth column used is
``depth_km_bsl`` when the events table has it (see :mod:`osv.data.datum`), otherwise
``depth_km``, which is then *assumed to be referenced to sea level*.  The column used is
recorded in ``AssociatedPicks.events.attrs["depth_column"]``.

Pick weighting (``sigma_s``, seconds; larger is worse)
------------------------------------------------------
* ``uncertainty``: ``sigma = unc_s``.
* ``comment``: ``sigma = base_sigma(phase) / w`` where ``w`` is ``comment_weight`` (in
  (0, 1]; a missing weight counts as 1) and ``base_sigma(phase)`` is the median ``unc_s`` of
  that phase over the retained picks.  ``w <= 0`` gives ``sigma = inf`` (pick to be
  rejected downstream).  If a phase has no uncertainties at all, ``sigma`` stays NaN.
* ``both``: ``sigma = unc_s / w``.

Missing (NaN) sigma values are filled with the per-phase median of the finite values and
reported; ``sigma_filled`` marks those picks.  Mapping ``sigma_s`` to VELEST weight classes is
done in task 2.4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from osv.config.schema import InputsConfig
from osv.coords import (
    azimuth_deg,
    backazimuth_deg,
    epicentral_distance_km,
    hypocentral_distance_km,
)
from osv.data.quakeml import CatalogTables
from osv.data.stations import StationTables, station_epoch_index
from osv.report.checks import QCFinding

logger = logging.getLogger(__name__)

__all__ = ["ASSOCIATED_STATION_COLUMNS", "KEEP_PHASES", "AssociatedPicks", "associate_picks"]

#: Phases kept by the association step.
KEEP_PHASES: tuple[str, ...] = ("P", "S")
#: Station columns copied onto each pick.
ASSOCIATED_STATION_COLUMNS = [
    "alias",
    "station_lat",
    "station_lon",
    "station_elevation_m",
    "sensor_elevation_m",
    "sensor_depth_m",
]


@dataclass
class AssociatedPicks:
    """Result of :func:`associate_picks`."""

    picks: pd.DataFrame
    events: pd.DataFrame
    findings: list[QCFinding] = field(default_factory=list)


def _per_station(df: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict[str, int]:
    sub = df[mask]
    key = sub["net"].fillna("").astype(str) + "." + sub["sta"].fillna("").astype(str)
    return {str(k): int(v) for k, v in key.value_counts().sort_index().items()}


def _azimuthal_gap(az: np.ndarray) -> float:
    """Largest azimuth interval (degrees) free of stations; 360 for fewer than 2 azimuths."""
    a = np.sort(np.asarray(az, dtype=float) % 360.0)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return 360.0
    d = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(d.max())


def _sigma(picks: pd.DataFrame, source: str) -> tuple[pd.Series, pd.Series, list[QCFinding]]:
    findings: list[QCFinding] = []
    unc = picks["unc_s"].astype(float)
    w = picks["comment_weight"].astype(float)
    if source == "uncertainty":
        sigma = unc.copy()
    else:
        w_eff = w.fillna(1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(w_eff > 0, 1.0 / np.where(w_eff > 0, w_eff, 1.0), np.inf)
        if source == "comment":
            base = unc.groupby(picks["phase"]).transform("median")
            sigma = base * inv
        else:  # both
            sigma = unc * inv
        n_zero = int((w_eff <= 0).sum())
        if n_zero:
            findings.append(
                QCFinding(
                    "zero_weight_picks",
                    "warn",
                    f"{n_zero} picks have comment weight <= 0; sigma_s set to inf.",
                    n_zero,
                )
            )
        n_nw = int(w.isna().sum())
        if n_nw:
            findings.append(
                QCFinding(
                    "missing_comment_weight",
                    "info",
                    f"{n_nw} picks have no comment weight; weight 1 assumed.",
                    n_nw,
                )
            )
    finite = sigma.where(np.isfinite(sigma))
    med = finite.groupby(picks["phase"]).transform("median")
    missing = sigma.isna()
    filled = missing & med.notna()
    sigma = sigma.where(~missing, med)
    if missing.any():
        meds = {
            str(k): float(v)
            for k, v in finite.groupby(picks["phase"]).median().items()
            if np.isfinite(v)
        }
        findings.append(
            QCFinding(
                "missing_sigma",
                "warn",
                f"{int(missing.sum())} picks had no usable uncertainty for weight_source="
                f"'{source}'; {int(filled.sum())} filled with the per-phase median.",
                int(missing.sum()),
                {"phase_median_s": meds, "unfilled": int((missing & ~filled).sum())},
            )
        )
    return sigma, filled, findings


def associate_picks(
    catalog: CatalogTables,
    stations: StationTables,
    inputs_cfg: InputsConfig | None = None,
) -> AssociatedPicks:
    """Associate catalogue picks with stations and add derived pick/event quantities.

    Parameters
    ----------
    catalog
        Output of :func:`osv.data.quakeml.read_catalog`.
    stations
        Output of :func:`osv.data.stations.read_stations`.
    inputs_cfg
        ``inputs`` configuration section; only ``weight_source`` is used.  Defaults apply
        when omitted.

    Returns
    -------
    AssociatedPicks
        ``picks``: retained P/S picks with columns of the catalogue pick table plus
        ``alias``, ``station_lat``, ``station_lon``, ``station_elevation_m``,
        ``sensor_elevation_m``, ``sensor_depth_m``, ``sigma_s``, ``sigma_filled``,
        ``epicentral_km``, ``hypocentral_km``, ``azimuth_deg`` (event to station),
        ``backazimuth_deg`` (station to event), ``apparent_velocity_kms``
        (hypocentral distance / travel time) and ``ts_minus_tp_s`` (S picks with a P at the
        same event and station, else NaN).  ``events``: copy of the event table with
        ``n_P``, ``n_S``, ``n_stations``, ``azgap_deg`` (primary azimuthal gap from the
        associated stations; 360 for fewer than two stations) and
        ``nearest_station_km`` (epicentral).  ``findings``: QC findings (inherits none from
        the readers).
    """
    cfg = inputs_cfg if inputs_cfg is not None else InputsConfig()
    findings: list[QCFinding] = []
    picks = catalog.picks.copy().reset_index(drop=True)
    events = catalog.events.copy().reset_index(drop=True)
    n_in = len(picks)

    # 1. phases
    bad_phase = ~picks["phase"].isin(KEEP_PHASES)
    if bad_phase.any():
        counts = picks.loc[bad_phase, "phase"].fillna("<none>").value_counts().to_dict()
        findings.append(
            QCFinding(
                "phase_dropped",
                "warn",
                f"{int(bad_phase.sum())} picks dropped: phase is not P or S.",
                int(bad_phase.sum()),
                {"phases": {str(k): int(v) for k, v in counts.items()}},
            )
        )
        picks = picks[~bad_phase].reset_index(drop=True)

    # 2. events
    orphan = ~picks["event_id"].isin(events["event_id"])
    if orphan.any():
        findings.append(
            QCFinding(
                "pick_without_event",
                "warn",
                f"{int(orphan.sum())} picks dropped: event_id not in the event table.",
                int(orphan.sum()),
            )
        )
        picks = picks[~orphan].reset_index(drop=True)

    # 3. station epochs
    idx = station_epoch_index(
        stations.stations, picks["net"].tolist(), picks["sta"].tolist(), picks["time"]
    )
    found = idx >= 0
    if not found.all():
        st = stations.stations
        known = set(zip(st["net"], st["sta"], strict=True))
        is_known = np.array(
            [(n, s) in known for n, s in zip(picks["net"], picks["sta"], strict=True)], dtype=bool
        )
        no_sta = ~found & ~is_known
        outside = ~found & is_known
        if no_sta.any():
            findings.append(
                QCFinding(
                    "pick_no_station",
                    "warn",
                    f"{int(no_sta.sum())} picks dropped: station not in the station table.",
                    int(no_sta.sum()),
                    {"per_station": _per_station(picks, no_sta)},
                )
            )
        if outside.any():
            findings.append(
                QCFinding(
                    "pick_outside_epoch",
                    "warn",
                    f"{int(outside.sum())} picks dropped: pick time outside every epoch of "
                    "its station.",
                    int(outside.sum()),
                    {"per_station": _per_station(picks, outside)},
                )
            )
    picks = picks[found].reset_index(drop=True)
    st_rows = stations.stations.reset_index(drop=True).iloc[idx[found]].reset_index(drop=True)
    picks["alias"] = st_rows["alias"].to_numpy()
    picks["station_lat"] = st_rows["lat"].to_numpy(dtype=float)
    picks["station_lon"] = st_rows["lon"].to_numpy(dtype=float)
    picks["station_elevation_m"] = st_rows["elevation_m"].to_numpy(dtype=float)
    picks["sensor_elevation_m"] = st_rows["sensor_elevation_m"].to_numpy(dtype=float)
    picks["sensor_depth_m"] = st_rows["sensor_depth_m"].to_numpy(dtype=float)

    # 4. de-duplication
    picks["_w"] = picks["comment_weight"].astype(float).fillna(-np.inf)
    picks = picks.sort_values(
        ["event_id", "net", "sta", "phase", "unc_s", "_w", "time"],
        ascending=[True, True, True, True, True, False, True],
        na_position="last",
        kind="mergesort",
    )
    dup = picks.duplicated(["event_id", "net", "sta", "phase"], keep="first")
    if dup.any():
        findings.append(
            QCFinding(
                "duplicate_picks",
                "info",
                f"{int(dup.sum())} duplicate picks removed per (event, station, phase); kept "
                "the smallest uncertainty, then highest comment weight, then earliest.",
                int(dup.sum()),
                {"per_station": _per_station(picks, dup)},
            )
        )
    picks = picks[~dup].drop(columns="_w").reset_index(drop=True)

    # 5. weights
    sigma, filled, wf = _sigma(picks, cfg.weight_source)
    picks["sigma_s"] = sigma
    picks["sigma_filled"] = filled
    findings.extend(wf)

    # 6. geometry
    depth_col = "depth_km_bsl" if "depth_km_bsl" in events.columns else "depth_km"
    ev = events.set_index("event_id")
    elat = ev["lat"].reindex(picks["event_id"]).to_numpy(dtype=float)
    elon = ev["lon"].reindex(picks["event_id"]).to_numpy(dtype=float)
    edep = ev[depth_col].reindex(picks["event_id"]).to_numpy(dtype=float)
    slat, slon = picks["station_lat"].to_numpy(), picks["station_lon"].to_numpy()
    sdep = -picks["sensor_elevation_m"].to_numpy() / 1000.0
    picks["epicentral_km"] = epicentral_distance_km(elat, elon, slat, slon)
    picks["hypocentral_km"] = hypocentral_distance_km(elat, elon, edep, slat, slon, sdep)
    picks["azimuth_deg"] = azimuth_deg(elat, elon, slat, slon)
    picks["backazimuth_deg"] = backazimuth_deg(elat, elon, slat, slon)
    tt = picks["travel_time_s"].astype(float)
    bad_tt = ~(tt > 0)
    if bad_tt.any():
        findings.append(
            QCFinding(
                "nonpositive_travel_time",
                "warn",
                f"{int(bad_tt.sum())} picks have missing or non-positive travel time "
                "(picks before the origin?); apparent velocity set to NaN.",
                int(bad_tt.sum()),
            )
        )
    picks["apparent_velocity_kms"] = np.where(bad_tt, np.nan, picks["hypocentral_km"] / tt)

    key = ["event_id", "net", "sta"]
    p_tt = (
        picks.loc[picks["phase"] == "P", [*key, "travel_time_s"]]
        .rename(columns={"travel_time_s": "_tp"})
        .drop_duplicates(key)
    )
    picks = picks.merge(p_tt, on=key, how="left")
    picks["ts_minus_tp_s"] = np.where(
        picks["phase"] == "S", picks["travel_time_s"] - picks["_tp"], np.nan
    )
    picks = picks.drop(columns="_tp")

    # event statistics
    events["n_P"] = events["event_id"].map(picks[picks.phase == "P"].groupby("event_id").size())
    events["n_S"] = events["event_id"].map(picks[picks.phase == "S"].groupby("event_id").size())
    events[["n_P", "n_S"]] = events[["n_P", "n_S"]].fillna(0).astype(int)
    sta_pick = picks.drop_duplicates(["event_id", "net", "sta"])
    g = sta_pick.groupby("event_id")
    events["n_stations"] = events["event_id"].map(g.size()).fillna(0).astype(int)
    gaps: dict[Any, float] = {k: _azimuthal_gap(v["azimuth_deg"].to_numpy()) for k, v in g}
    events["azgap_deg"] = events["event_id"].map(gaps).fillna(360.0)
    events["nearest_station_km"] = events["event_id"].map(g["epicentral_km"].min())
    events.attrs["depth_column"] = depth_col

    n_drop = n_in - len(picks)
    findings.append(
        QCFinding(
            "association_summary",
            "info",
            f"{len(picks)} of {n_in} picks retained "
            f"({int((picks.phase == 'P').sum())} P, {int((picks.phase == 'S').sum())} S); "
            f"{n_drop} dropped; event depth column '{depth_col}'.",
            len(picks),
            {"dropped": n_drop, "depth_column": depth_col},
        )
    )
    for f in findings:
        logger.info("%s", f)
    return AssociatedPicks(picks=picks, events=events, findings=findings)
