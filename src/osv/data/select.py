"""Event and pick selection, rejection log, weight classes and validation split (task 2.4).

:func:`select_data` applies the ``data_selection`` configuration to the output of
:func:`osv.data.associate.associate_picks`:

1. *Pick-level filters* (every failed reason is logged for each pick): epicentral distance
   above ``max_epi_dist_km``, zero weight (``sigma_s`` infinite) and, optionally, caller
   supplied outlier flags (Wadati / travel-time analysis).
2. Per-event ``n_P``, ``n_S``, ``n_stations``, ``azgap_deg`` and ``nearest_station_km`` are
   recomputed from the surviving picks.
3. *Event-level filters* (all failed reasons are logged): region, depth, ``min_p``,
   ``min_s``, ``max_azgap``, ``max_rms`` and minimum magnitude.  A missing (NaN) RMS is not
   treated as a failure.  The magnitude filter is skipped (with a finding) when all
   magnitudes are flagged invalid and ``magnitude.ignore_if_invalid`` is set.  Picks of
   rejected events are dropped.
4. ``velest_weight`` (0..4) is assigned from ``sigma_s`` with
   ``pick_uncertainty_to_weight``; class 4 means "unused by VELEST" and is only counted.
   ``sigma_s`` itself is kept for FMTOMO.
5. A seeded, depth-stratified calibration/validation split of the selected events.

The region comes from ``region_geojson`` (Polygon/MultiPolygon in lon/lat, holes allowed)
when set, otherwise from ``region_polygon`` (``[lat, lon]`` vertices).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from osv.config.schema import OSVConfig
from osv.data.associate import AssociatedPicks, _azimuthal_gap
from osv.report.checks import QCFinding

logger = logging.getLogger(__name__)

__all__ = [
    "REJECTION_COLUMNS",
    "Selection",
    "assign_velest_weight",
    "load_geojson_region",
    "points_in_region",
    "recompute_event_stats",
    "select_data",
    "split_events",
]

REJECTION_COLUMNS = ["level", "id", "reason", "value", "threshold"]
#: Weight class meaning "not used by VELEST".
REJECT_WEIGHT = 4
_SIGMA_TOL = 1e-9

#: A region is a list of polygons; each polygon is a list of rings (exterior first, then
#: holes); each ring is an (n, 2) array of ``(lon, lat)``.
Region = list[list[np.ndarray]]


@dataclass
class Selection:
    """Result of :func:`select_data`."""

    events: pd.DataFrame
    picks: pd.DataFrame
    rejections: pd.DataFrame
    split: pd.Series
    findings: list[QCFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# --- region -----------------------------------------------------------------------------


def _polygon_from_geometry(geom: dict[str, Any]) -> Region:
    kind = geom.get("type")
    coords = geom.get("coordinates")
    if kind == "Polygon":
        polys = [coords]
    elif kind == "MultiPolygon":
        polys = coords
    elif kind == "GeometryCollection":
        out: Region = []
        for g in geom.get("geometries", []):
            out.extend(_polygon_from_geometry(g))
        return out
    else:
        raise ValueError(f"GeoJSON geometry type {kind!r} is not a Polygon or MultiPolygon")
    return [[np.asarray(r, dtype=float)[:, :2] for r in poly] for poly in polys]


def load_geojson_region(path: str | Path) -> Region:
    """Read Polygon/MultiPolygon geometries (lon/lat) from a GeoJSON file.

    Accepts a bare geometry, a ``Feature`` or a ``FeatureCollection``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON region file not found: {p}")
    obj = json.loads(p.read_text())
    if obj.get("type") == "FeatureCollection":
        geoms = [f["geometry"] for f in obj.get("features", []) if f.get("geometry")]
    elif obj.get("type") == "Feature":
        geoms = [obj["geometry"]]
    else:
        geoms = [obj]
    region: Region = []
    for g in geoms:
        region.extend(_polygon_from_geometry(g))
    if not region:
        raise ValueError(f"No polygon found in GeoJSON file {p}")
    return region


def points_in_region(lon: np.ndarray, lat: np.ndarray, region: Region) -> np.ndarray:
    """Boolean mask of points inside the region (holes excluded, polygons united)."""
    pts = np.column_stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)])
    ok = np.isfinite(pts).all(axis=1)
    inside = np.zeros(len(pts), dtype=bool)
    for poly in region:
        m = MplPath(poly[0]).contains_points(pts)
        for hole in poly[1:]:
            m &= ~MplPath(hole).contains_points(pts)
        inside |= m
    return inside & ok


def _region_from_config(cfg: OSVConfig) -> Region | None:
    ds = cfg.data_selection
    if ds.region_geojson is not None:
        return load_geojson_region(ds.region_geojson)
    if ds.region_polygon is not None:
        return [[np.array([(lon, lat) for lat, lon in ds.region_polygon], dtype=float)]]
    return None


# --- weights and statistics -------------------------------------------------------------


def assign_velest_weight(sigma_s: pd.Series | np.ndarray, mapping: dict[float, int]) -> np.ndarray:
    """Map pick sigma (s) to VELEST weight classes.

    A sigma up to and including a threshold gets that threshold's class; larger (or
    non-finite, or missing) sigma gets class 4.
    """
    thr = np.array(sorted(mapping), dtype=float)
    cls = np.array([mapping[k] for k in sorted(mapping)], dtype=int)
    s = np.asarray(sigma_s, dtype=float)
    idx = np.searchsorted(thr + _SIGMA_TOL, s, side="left")
    out = np.where(idx < len(thr), cls[np.minimum(idx, len(thr) - 1)], REJECT_WEIGHT)
    return np.where(np.isfinite(s), out, REJECT_WEIGHT).astype(int)


def recompute_event_stats(events: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` with n_P, n_S, n_stations, azgap_deg, nearest_station_km from picks."""
    ev = events.copy()
    eid = ev["event_id"]
    ev["n_P"] = eid.map(picks[picks.phase == "P"].groupby("event_id").size()).fillna(0)
    ev["n_S"] = eid.map(picks[picks.phase == "S"].groupby("event_id").size()).fillna(0)
    sta = picks.drop_duplicates(["event_id", "net", "sta"])
    g = sta.groupby("event_id")
    ev["n_stations"] = eid.map(g.size()).fillna(0)
    ev[["n_P", "n_S", "n_stations"]] = ev[["n_P", "n_S", "n_stations"]].astype(int)
    gaps = {k: _azimuthal_gap(v["azimuth_deg"].to_numpy()) for k, v in g}
    ev["azgap_deg"] = eid.map(gaps).fillna(360.0)
    ev["nearest_station_km"] = eid.map(g["epicentral_km"].min())
    return ev


def split_events(
    events: pd.DataFrame, depth: pd.Series | np.ndarray, fraction: float, seed: int
) -> pd.Series:
    """Seeded depth-stratified split; returns event_id -> 'calibration' | 'validation'.

    Events are sorted by depth and cut into ``n_val`` contiguous depth groups of (nearly)
    equal size; one event per group is drawn at random for validation, so the validation
    set spans the depth range.  ``fraction == 0`` (or fewer than two events) gives an
    all-calibration split.
    """
    ids = events["event_id"].to_numpy()
    out = pd.Series("calibration", index=pd.Index(ids, name="event_id"), name="split")
    n = len(ids)
    if fraction <= 0 or n < 2:
        return out
    n_val = min(n - 1, max(1, round(fraction * n)))
    d = np.asarray(depth, dtype=float)
    order = np.lexsort((ids.astype(str), np.nan_to_num(d, nan=np.inf)))
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(grp) for grp in np.array_split(order, n_val)]
    out.iloc[np.array(chosen)] = "validation"
    return out


# --- main -------------------------------------------------------------------------------


def _rej(level: str, ids: Any, reason: str, value: Any, threshold: Any) -> pd.DataFrame:
    ids = np.asarray(ids)
    return pd.DataFrame(
        {
            "level": level,
            "id": ids,
            "reason": reason,
            "value": np.asarray(value, dtype=float),
            "threshold": threshold,
        },
        columns=REJECTION_COLUMNS,
    )


def select_data(
    assoc: AssociatedPicks, cfg: OSVConfig, outlier_flags: pd.Series | None = None
) -> Selection:
    """Apply the ``data_selection`` filters and split the surviving events.

    Parameters
    ----------
    assoc
        Output of :func:`osv.data.associate.associate_picks`.
    cfg
        Full configuration (``data_selection`` and ``project.random_seed`` are used).
    outlier_flags
        Optional boolean Series indexed like ``assoc.picks`` (True = outlier); such picks
        are rejected with reason ``'outlier'``.

    Returns
    -------
    Selection
        ``events`` (selected, with recomputed counts and a ``split`` column), ``picks``
        (selected, with ``velest_weight``), ``rejections`` (columns ``level``, ``id``,
        ``reason``, ``value``, ``threshold``; one row per failed reason; ``id`` is the
        ``event_id`` or ``pick_id``), ``split``, ``findings`` and ``summary`` (counts).
    """
    ds = cfg.data_selection
    picks = assoc.picks.copy()
    events_in = assoc.events.copy().reset_index(drop=True)
    findings: list[QCFinding] = []
    rej: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        "events_in": len(events_in),
        "picks_in": len(picks),
        "pick_filters": {},
        "event_filters": {},
    }
    pick_ids = (
        picks["pick_id"]
        if "pick_id" in picks.columns
        else pd.Series(picks.index, index=picks.index)
    )

    # 1. pick-level filters
    sigma = picks["sigma_s"].astype(float)
    masks = {
        "epicentral_distance": picks["epicentral_km"].astype(float) > ds.max_epi_dist_km,
        "zero_weight": np.isinf(sigma),
    }
    thresholds: dict[str, Any] = {
        "epicentral_distance": ds.max_epi_dist_km,
        "zero_weight": np.nan,
    }
    values = {"epicentral_distance": picks["epicentral_km"], "zero_weight": sigma}
    if outlier_flags is not None:
        flags = outlier_flags.reindex(picks.index).fillna(False).astype(bool)
        masks["outlier"] = flags
        thresholds["outlier"] = np.nan
        values["outlier"] = pd.Series(np.nan, index=picks.index)
    drop = pd.Series(False, index=picks.index)
    for reason, m in masks.items():
        m = m.fillna(False).astype(bool)
        summary["pick_filters"][reason] = int(m.sum())
        if m.any():
            rej.append(_rej("pick", pick_ids[m], reason, values[reason][m], thresholds[reason]))
            findings.append(
                QCFinding(
                    f"pick_{reason}",
                    "info",
                    f"{int(m.sum())} picks rejected: {reason.replace('_', ' ')}.",
                    int(m.sum()),
                )
            )
        drop |= m
    summary["picks_rejected_pick_level"] = int(drop.sum())
    picks = picks[~drop]

    # 2. recompute event statistics
    events = recompute_event_stats(events_in, picks)

    # 3. event-level filters
    depth_col = "depth_km_bsl" if "depth_km_bsl" in events.columns else "depth_km"
    depth = events[depth_col].astype(float)
    fails: dict[str, tuple[pd.Series, pd.Series, Any]] = {}
    region = _region_from_config(cfg)
    if region is not None:
        inside = points_in_region(events["lon"].to_numpy(), events["lat"].to_numpy(), region)
        fails["region"] = (pd.Series(~inside, index=events.index), pd.Series(np.nan), np.nan)
    dmin, dmax = ds.depth_km
    fails["depth"] = (~((depth >= dmin) & (depth <= dmax)), depth, f"[{dmin}, {dmax}]")
    fails["min_p"] = (events["n_P"] < ds.min_p, events["n_P"], ds.min_p)
    fails["min_s"] = (events["n_S"] < ds.min_s, events["n_S"], ds.min_s)
    fails["max_azgap"] = (events["azgap_deg"] > ds.max_azgap, events["azgap_deg"], ds.max_azgap)
    if "rms" in events.columns:
        rms = events["rms"].astype(float)
        fails["max_rms"] = (rms > ds.max_rms, rms, ds.max_rms)
    mag_cfg = ds.magnitude
    if mag_cfg.min is not None:
        valid = (
            events["mag_valid"].astype(bool)
            if "mag_valid" in events.columns
            else pd.Series(True, index=events.index)
        )
        if mag_cfg.ignore_if_invalid and not valid.any():
            findings.append(
                QCFinding(
                    "magnitude_filter_skipped",
                    "info",
                    "Minimum-magnitude filter skipped: all catalogue magnitudes are invalid.",
                    len(events),
                )
            )
            summary["magnitude_filter"] = "skipped"
        else:
            mag = events["mag"].astype(float)
            fails["magnitude"] = ((mag < mag_cfg.min) | ~valid, mag, mag_cfg.min)
            summary["magnitude_filter"] = "applied"
    reject = pd.Series(False, index=events.index)
    for reason, (m, val, thr) in fails.items():
        m = m.fillna(False).astype(bool)
        m.index = events.index
        summary["event_filters"][reason] = int(m.sum())
        if m.any():
            v = val.reindex(events.index) if isinstance(val, pd.Series) else val
            rej.append(_rej("event", events.loc[m, "event_id"], reason, v[m], thr))
        reject |= m
    summary["events_rejected"] = int(reject.sum())
    if reject.any():
        findings.append(
            QCFinding(
                "events_rejected",
                "info",
                f"{int(reject.sum())} of {len(events)} events rejected "
                f"({', '.join(f'{k}: {v}' for k, v in summary['event_filters'].items() if v)}).",
                int(reject.sum()),
                {"per_filter": summary["event_filters"]},
            )
        )
    sel_events = events[~reject].reset_index(drop=True)
    sel_picks = picks[picks["event_id"].isin(sel_events["event_id"])].copy()

    # 4. weight classes
    sel_picks["velest_weight"] = assign_velest_weight(
        sel_picks["sigma_s"], ds.pick_uncertainty_to_weight
    )
    wc = sel_picks["velest_weight"].value_counts().reindex(range(5), fill_value=0)
    summary["weight_classes"] = {int(k): int(v) for k, v in wc.items()}
    n4 = int(wc[REJECT_WEIGHT])
    if n4:
        findings.append(
            QCFinding(
                "weight_class_4_picks",
                "warn",
                f"{n4} selected picks fall in weight class {REJECT_WEIGHT} "
                "(sigma above the largest threshold) and will not be used by VELEST.",
                n4,
            )
        )

    # 5. split
    sel_depth = sel_events[depth_col]
    split = split_events(sel_events, sel_depth, ds.validation_fraction, cfg.project.random_seed)
    sel_events["split"] = sel_events["event_id"].map(split)
    sel_events.attrs.update(assoc.events.attrs)
    summary["n_calibration"] = int((split == "calibration").sum())
    summary["n_validation"] = int((split == "validation").sum())
    summary["events_selected"] = len(sel_events)
    summary["picks_selected"] = len(sel_picks)
    summary["picks_rejected_with_events"] = len(picks) - len(sel_picks)
    if len(sel_events) == 0:
        findings.append(
            QCFinding("no_events_selected", "fail", "No events survive the selection filters.")
        )

    rejections = (
        pd.concat(rej, ignore_index=True) if rej else pd.DataFrame(columns=REJECTION_COLUMNS)
    )
    for f in findings:
        logger.info("%s", f)
    return Selection(
        events=sel_events,
        picks=sel_picks.reset_index(drop=True),
        rejections=rejections,
        split=split,
        findings=findings,
        summary=summary,
    )
