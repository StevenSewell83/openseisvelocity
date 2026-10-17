"""Stage 1 (``prep``): StationXML + QuakeML to clean, selected event and pick tables.

Pipeline (see PLAN.md section 5): read catalogue and stations, apply the depth datum,
associate picks with stations, analyse Wadati / travel-time outliers, apply the selection
filters, write the tables, QC findings, figures and ``report.html``.

Outputs (in ``<workdir>/01_prep/``)
-----------------------------------
``events.parquet``, ``picks.parquet``
    Selected events (with ``split``) and picks (with ``velest_weight``).
``all_events.parquet``, ``all_picks.parquet``
    Every event / associated pick with ``selected``, ``reject_reasons``, outlier columns and
    event statistics computed after the pick-level filters (the basis of the event cut-offs).
``stations.parquet``, ``channels.parquet``, ``aliases.csv``
    Station epochs (plus ``used``, ``n_P``, ``n_S``), channels and VELEST station aliases.
``rejections.csv``, ``split.csv``
    One row per failed filter reason; calibration/validation assignment.
``wadati.json``, ``wadati_pairs.parquet``, ``wadati_events.csv``, ``tt_fits.json``,
``outliers.csv``, ``station_residuals.csv``
    Outlier analysis products.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.data.associate import associate_picks
from osv.data.datum import apply_depth_datum
from osv.data.outliers import analyse_outliers
from osv.data.quakeml import read_catalog
from osv.data.select import assign_velest_weight, recompute_event_stats, select_data
from osv.data.stations import read_stations
from osv.data.velest_inputs import write_velest_inputs
from osv.io.velest import VelestControl
from osv.report.checks import QCFinding
from osv.stages import Stage, StageContext, register

#: Stations with fewer associated picks than this are reported by the ``stations_few_picks`` check.
FEW_PICKS_PER_STATION = 5

#: Prefilled VELEST control (serialised ``VelestControl``) saved for Stage 2.
VELEST_CONTROL_JSON = "velest_control.json"


def load_velest_control(stage_dir: Path) -> VelestControl:
    """Load the prefilled control saved by Stage 1 in ``<stage_dir>/velest/``."""
    data = json.loads((Path(stage_dir) / "velest" / VELEST_CONTROL_JSON).read_text())
    return VelestControl(**data)


SECTIONS = ("Maps", "Distributions", "Travel times", "Wadati", "Stations", "Selection")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "profile"


def _surface_elevation_fn(stations: pd.DataFrame):
    """Ground elevation (m) at lat/lon interpolated linearly between stations (nearest outside)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    st = stations.drop_duplicates(["net", "sta"], keep="last")
    pts = np.column_stack([st["lon"], st["lat"]])
    z = st["elevation_m"].to_numpy(float)
    lin = LinearNDInterpolator(pts, z)
    near = NearestNDInterpolator(pts, z)

    def f(lat, lon):
        v = lin(lon, lat)
        return np.where(np.isnan(v), near(lon, lat), v)

    return f


def _join_reasons(rej: pd.DataFrame, level: str, prefix: str = "") -> pd.Series:
    r = rej[rej["level"] == level]
    if r.empty:
        return pd.Series(dtype=str)
    return r.groupby("id")["reason"].agg(lambda s: ";".join(prefix + x for x in s))


def _origin(cfg_origin: Any, stations: pd.DataFrame) -> tuple[float, float]:
    if cfg_origin is not None:
        return float(cfg_origin.lat), float(cfg_origin.lon)
    return float(stations["lat"].mean()), float(stations["lon"].mean())


@register
class PrepStage(Stage):
    """Stage 1: data preparation."""

    name = "prep"
    requires = ()
    outputs = (
        "events.parquet",
        "picks.parquet",
        "all_events.parquet",
        "all_picks.parquet",
        "stations.parquet",
        "channels.parquet",
        "aliases.csv",
        "rejections.csv",
        "split.csv",
        "wadati.json",
        "tt_fits.json",
        "velest/events.cnv",
        "velest/stations.sta",
        "velest/velest_inputs_summary.json",
        "velest/event_map.csv",
        f"velest/{VELEST_CONTROL_JSON}",
    )

    def inputs(self, ctx: StageContext) -> list[Path]:
        inp = ctx.config.inputs
        extra = [inp.station_aliases] if isinstance(inp.station_aliases, Path) else []
        return [Path(p) for p in (*inp.quakeml, *inp.stationxml, *extra)]

    # ------------------------------------------------------------------ run

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        log = ctx.logger
        cat = read_catalog(cfg.inputs.quakeml, cfg.inputs, cfg.data_selection)
        sta = read_stations(
            cfg.inputs.stationxml,
            cfg.inputs,
            cfg.coordinates,
            aliases=cfg.inputs.station_aliases,  # 'auto', mapping or resolved CSV path
        )
        log.info(
            "Read %d events, %d picks, %d station epochs",
            len(cat.events),
            len(cat.picks),
            len(sta.stations),
        )

        surface = (
            _surface_elevation_fn(sta.stations)
            if cfg.coordinates.depth_datum == "surface"
            else None
        )
        cat.events = apply_depth_datum(
            cat.events,
            cfg.coordinates.depth_datum,
            cfg.coordinates.reference_elevation_m,
            surface_elevation_m=surface,
        )
        assoc = associate_picks(cat, sta, cfg.inputs)
        oa = analyse_outliers(
            assoc, cfg.data_selection.wadati_outlier_sigma, seed=cfg.project.random_seed
        )
        sel = select_data(assoc, cfg, outlier_flags=oa.flags)

        findings: list[QCFinding] = [
            *cat.findings,
            *sta.findings,
            *assoc.findings,
            *oa.findings,
            *sel.findings,
        ]

        # ---- pick table with outlier and rejection information
        picks = assoc.picks.copy()
        if not picks["pick_id"].is_unique:
            raise ValueError(
                "pick_id values are not unique after association; cannot log rejections"
            )
        pt = oa.pick_table.drop(columns=["phase"]).rename(columns={"reason": "outlier_reason"})
        picks = picks.join(pt)
        picks["velest_weight"] = assign_velest_weight(
            picks["sigma_s"], cfg.data_selection.pick_uncertainty_to_weight
        )
        rej = sel.rejections
        pick_reasons = _join_reasons(rej, "pick")
        ev_reasons = _join_reasons(rej, "event")
        picks["reject_reasons"] = picks["pick_id"].map(pick_reasons).fillna("")
        ev_rej_of_pick = picks["event_id"].map(ev_reasons).fillna("")
        picks["reject_reasons"] = [
            ";".join(x for x in (a, "event:" + b.replace(";", ";event:") if b else "") if x)
            for a, b in zip(picks["reject_reasons"], ev_rej_of_pick, strict=True)
        ]
        picks["selected"] = picks["pick_id"].isin(sel.picks["pick_id"])

        # ---- event table for all events (statistics after the pick-level filters)
        pick_level_ok = picks[~picks["pick_id"].isin(pick_reasons.index)]
        all_events = recompute_event_stats(assoc.events, pick_level_ok)
        all_events["selected"] = all_events["event_id"].isin(sel.events["event_id"])
        all_events["reject_reasons"] = all_events["event_id"].map(ev_reasons).fillna("")
        all_events["split"] = (
            all_events["event_id"].map(sel.split).where(all_events["selected"], "")
        )
        all_events.attrs.clear()

        # ---- stations
        st = sta.stations.copy()
        keys = list(zip(st["net"], st["sta"], strict=True))
        counts = {
            ph: picks[picks["selected"] & (picks["phase"] == ph)].groupby(["net", "sta"]).size()
            for ph in ("P", "S")
        }
        for ph in ("P", "S"):
            st[f"n_{ph}"] = [int(counts[ph].get(k, 0)) for k in keys]
        st["used"] = (st["n_P"] + st["n_S"]) > 0
        sta.stations = st

        # ---- write
        sta.write(ctx.stage_dir)
        sel.events.to_parquet(ctx.path("events.parquet"))
        sel.picks.to_parquet(ctx.path("picks.parquet"))
        all_events.to_parquet(ctx.path("all_events.parquet"))
        picks.to_parquet(ctx.path("all_picks.parquet"))
        rej.to_csv(ctx.path("rejections.csv"), index=False)
        sel.split.rename("split").to_csv(ctx.path("split.csv"))
        picks[picks["outlier"]][
            [
                "event_id",
                "pick_id",
                "alias",
                "phase",
                "hypocentral_km",
                "travel_time_s",
                "tt_residual_s",
                "tt_z",
                "wadati_residual_s",
                "wadati_z",
                "outlier_reason",
            ]
        ].to_csv(ctx.path("outliers.csv"), index=False)
        oa.stations.to_csv(ctx.path("station_residuals.csv"), index=False)

        wad = oa.wadati
        if wad is not None:
            wad_json: dict[str, Any] = {
                "vpvs": wad.vpvs,
                "vpvs_ci": list(wad.vpvs_ci),
                "vpvs_std": wad.vpvs_std,
                "n_pairs": wad.n_pairs,
                "n_events": wad.n_events,
                "sigma_s": wad.sigma_s,
                "event_vpvs_median": wad.event_vpvs_median,
                "event_vpvs_spread": wad.event_vpvs_spread,
                "outlier_sigma": oa.sigma,
            }
            wad.pairs.reset_index(names="pick_index").to_parquet(ctx.path("wadati_pairs.parquet"))
            wad.events.to_csv(ctx.path("wadati_events.csv"), index=False)
        else:
            wad_json = {}
        ctx.path("wadati.json").write_text(json.dumps(wad_json, indent=2))
        tt_json = {
            ph: {"coef": f.coef.tolist(), "sigma_s": f.sigma_s, "degree": f.degree, "n": f.n}
            for ph, f in oa.tt.items()
        }
        ctx.path("tt_fits.json").write_text(json.dumps(tt_json, indent=2))

        # ---- VELEST inputs (calibration events; validation events separately)
        cfg_v = cfg
        if isinstance(cfg.velest.vpvs.value, str):  # 'wadati': resolve from the fit just made
            if wad is None:
                raise ValueError(
                    "velest.vpvs.value is 'wadati' but there are too few S-P pairs for a "
                    "Wadati fit; set velest.vpvs.value to a number"
                )
            cfg_v = cfg.model_copy(deep=True)
            cfg_v.velest.vpvs.value = round(float(wad.vpvs), 3)
        vcal = write_velest_inputs(sel, sta, cfg_v, ctx.stage_dir / "velest", subset="calibration")
        ctrl_path = ctx.stage_dir / "velest" / VELEST_CONTROL_JSON
        ctrl_path.write_text(json.dumps(dataclasses.asdict(vcal.control), indent=2))
        findings.extend(vcal.findings)
        vsummary: dict[str, Any] = {
            "velest_events": vcal.n_events,
            "velest_picks": vcal.n_picks,
            "velest_stations": vcal.n_stations,
            "velest_reference_station": vcal.reference_station,
        }
        if (sel.split == "validation").any():
            vval = write_velest_inputs(
                sel,
                sta,
                cfg_v,
                ctx.stage_dir / "velest_validation",
                subset="validation",
                reference_station=vcal.reference_station,
            )
            findings.extend(vval.findings)
            vsummary["velest_validation_events"] = vval.n_events
            vsummary["velest_validation_picks"] = vval.n_picks

        # ---- checks and summary
        n_by_sta = picks.groupby("alias").size()
        few = n_by_sta[n_by_sta < FEW_PICKS_PER_STATION]
        if len(few):
            findings.append(
                QCFinding(
                    "stations_few_picks",
                    "info",
                    f"{len(few)} stations have fewer than {FEW_PICKS_PER_STATION} picks.",
                    len(few),
                    {"picks": {str(k): int(v) for k, v in few.items()}},
                )
            )
        for f in findings:
            ctx.add_check(f)

        se = sel.events
        sp = sel.picks
        ctx.summary.update(
            {
                "events_in": len(assoc.events),
                "events_selected": len(se),
                "events_calibration": sel.summary.get("n_calibration", 0),
                "events_validation": sel.summary.get("n_validation", 0),
                "picks_P_in": int((picks["phase"] == "P").sum()),
                "picks_S_in": int((picks["phase"] == "S").sum()),
                "picks_P_selected": int((sp["phase"] == "P").sum()),
                "picks_S_selected": int((sp["phase"] == "S").sum()),
                "stations_with_picks": int(picks["alias"].nunique()),
                "stations_used": int(sp["alias"].nunique()),
                "vpvs": round(wad.vpvs, 4) if wad else None,
                "vpvs_ci_95": [round(v, 4) for v in wad.vpvs_ci] if wad else None,
                "median_azgap_deg": round(float(se["azgap_deg"].median()), 1) if len(se) else None,
                "weight_classes": sel.summary.get("weight_classes"),
            }
        )
        ctx.summary.update(vsummary)
        self.plot(ctx)

    # ------------------------------------------------------------------ plot

    def plot(self, ctx: StageContext) -> None:
        from osv.data import plots

        cfg = ctx.config
        ds = cfg.data_selection
        figs = ctx.figures
        events = pd.read_parquet(ctx.stage_dir / "all_events.parquet")
        picks = pd.read_parquet(ctx.stage_dir / "all_picks.parquet")
        stations = pd.read_parquet(ctx.stage_dir / "stations.parquet")
        rej = pd.read_csv(ctx.stage_dir / "rejections.csv")
        wadati = json.loads((ctx.stage_dir / "wadati.json").read_text())
        tt_fits = json.loads((ctx.stage_dir / "tt_fits.json").read_text())
        resid = pd.read_csv(ctx.stage_dir / "station_residuals.csv")
        origin = _origin(cfg.coordinates.origin, stations)
        cutoffs = {
            "min_p": ds.min_p,
            "min_s": ds.min_s,
            "max_azgap": ds.max_azgap,
            "max_rms": ds.max_rms,
            "depth_km": list(ds.depth_km),
            "max_epi_dist_km": ds.max_epi_dist_km,
        }
        n_sel = int(events["selected"].sum())

        figs.save(
            plots.plot_station_event_map(events, stations, picks),
            "map_before_after",
            "Stations and events before and after selection. Events are coloured by depth below "
            "sea level; rejected events are grey crosses, validation events diamonds, borehole "
            "sensors grey inverted triangles with a thick dark outline and stations without picks "
            "grey dots.",
            "Maps",
        )
        figs.save(
            plots.plot_depth_section(events, stations, origin, "ew", picks=picks),
            "section_ew",
            "East-west depth section: all events projected, stations at sensor elevation "
            "(depth below sea level, negative above).",
            "Maps",
        )
        figs.save(
            plots.plot_depth_section(events, stations, origin, "ns", picks=picks),
            "section_ns",
            "North-south depth section: all events projected, stations at sensor elevation.",
            "Maps",
        )
        for prof in cfg.plotting.profiles:
            p = {"name": prof.name, "start": list(prof.start), "end": list(prof.end)}
            figs.save(
                plots.plot_depth_section(events, stations, origin, "profile", p, picks),
                f"section_profile_{_slug(prof.name)}",
                f"Depth section along profile {prof.name} (events inside the swath).",
                "Maps",
            )
        figs.save(
            plots.plot_ray_map(picks, events, stations),
            "ray_map",
            "Straight ray paths from each selected event to the stations that picked it "
            "(alpha-blended; darker means more rays).",
            "Maps",
        )
        figs.save(
            plots.plot_ray_section(picks, events, stations, origin, "ew"),
            "ray_section_ew",
            "Ray paths on an east-west depth section (straight lines, selected picks).",
            "Maps",
        )
        figs.save(
            plots.plot_ray_section(picks, events, stations, origin, "ns"),
            "ray_section_ns",
            "Ray paths on a north-south depth section (straight lines, selected picks).",
            "Maps",
        )
        figs.save(
            plots.plot_histograms(events, cutoffs),
            "histograms",
            f"Event parameter histograms ({n_sel} of {len(events)} events selected); red dashed "
            "lines are the configured cut-offs. The magnitude panel is omitted when the "
            "catalogue magnitudes are invalid.",
            "Distributions",
        )
        figs.save(
            plots.plot_event_timeline(events),
            "event_timeline",
            "Event depth against origin time, with the cumulative event count.",
            "Distributions",
        )
        figs.save(
            plots.plot_uncertainty(picks, ds.pick_uncertainty_to_weight),
            "pick_uncertainty",
            "Pick uncertainty distributions per phase (dashed lines: weight-class limits) and "
            "the number of picks in each VELEST weight class.",
            "Distributions",
        )
        figs.save(
            plots.plot_travel_times(picks, tt_fits, ds.wadati_outlier_sigma),
            "travel_times",
            "Travel time against hypocentral distance for P and S, coloured by VELEST weight "
            "class. Red crosses are outliers; the line is the robust polynomial fit with its "
            "outlier band.",
            "Travel times",
        )
        pairs_path = ctx.stage_dir / "wadati_pairs.parquet"
        ev_path = ctx.stage_dir / "wadati_events.csv"
        figs.save(
            plots.plot_wadati(
                pd.read_parquet(pairs_path) if pairs_path.exists() else pd.DataFrame(),
                wadati,
                pd.read_csv(ev_path) if ev_path.exists() else None,
            ),
            "wadati",
            "Wadati diagram (Ts - Tp against Tp) with the robust pooled Vp/Vs fit, its 95% "
            "bootstrap interval and outliers. Inset: per-event Vp/Vs.",
            "Wadati",
        )
        figs.save(
            plots.plot_picks_per_station(picks, FEW_PICKS_PER_STATION),
            "picks_per_station",
            "Picks per station, P and S stacked and sorted; hatched parts were rejected. The "
            "dashed line marks the few-picks check threshold.",
            "Stations",
        )
        figs.save(
            plots.plot_station_residuals(resid),
            "station_residuals",
            "Median travel-time residual per station against the 1-D polynomial fit, with "
            "standard-error bars; systematic stations are red.",
            "Stations",
        )
        figs.save(
            plots.plot_station_residual_map(resid, stations),
            "station_residual_map",
            "Map of the median travel-time residual per station (red = late arrivals). "
            "Systematic stations are outlined and labelled.",
            "Stations",
        )
        n_pick_ev_rej = int(
            (~picks["selected"] & picks["reject_reasons"].str.startswith("event:")).sum()
        )
        figs.save(
            plots.plot_rejections(
                rej, len(events), len(picks), picks_with_rejected_events=n_pick_ev_rej
            ),
            "rejections",
            "Why events and picks were rejected (an item failing several filters counts once per "
            "reason). Full list: rejections.csv.",
            "Selection",
        )
