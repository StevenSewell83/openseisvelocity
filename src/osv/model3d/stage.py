"""Stage 3 (``build_model``): the FMTOMO starting model, its inputs and a forward check.

``build_model`` (``<workdir>/04_build_model/``) needs ``prep`` and ``velest_analyse``.

Steps: define the grid, build the starting model (``model3d.source``: the VELEST 1-D model alone
or blended with a user model), build the surface and base interfaces, move sources below the
surface, write the FMTOMO data (P always, S when ``fmtomo.phases`` has ``S`` or ``vpvs``), trace
the observed rays through the model with ``fm3d`` and compare the residuals with the 1-D
background and with VELEST, then write the outputs, figures, checks and report.

Hand-off files for Stage 4 (stable names, ``fmtomo/``)
------------------------------------------------------
``vgrids.in``, ``interfaces.in``, ``propgrid.in``
    Vp model (velocity type 1, with the a-priori uncertainty column), the surface and base
    interfaces and the propagation grid.
``sources.in``, ``sourcesref.in``, ``receivers.in``, ``otimes_P.dat``, ``otimes_P_mapping.csv``
    P sources (relocated VELEST hypocentres of the calibration events, moved below the surface
    where needed), stations with picks, observed travel times and the table mapping every
    ``otimes`` row to (event, station, phase, pick).  ``events.csv`` and ``stations.csv`` give
    the source and receiver numbering.
``vgrids_S.in``, ``sources_S.in``, ``sourcesref_S.in``, ``receivers_S.in``, ``otimes_S.dat``,
``otimes_S_mapping.csv``, ``events_S.csv``
    The same for S (written when S is requested): the Vs model is velocity type 1 of its own
    file (``layout='separate'``).

Other files: ``starting_model.nc`` (xarray: ``vp``, ``vs``, ``vp_unc``, ``vs_unc``, ``weight``,
``vp_bg``, ``vs_bg``, optional ``unit_id``, on the full grid with cushions), ``grid.json``
(:func:`grid_summary`), ``forward_check.json`` and ``forward_check/`` (``residuals.parquet``,
``by_station.csv``, ``by_event.csv``, ``arrivals.dat`` of each run).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.io.fmtomo import (
    read_interfaces,
    read_propgrid,
    write_interfaces,
    write_propgrid,
    write_vgrids,
)
from osv.model3d.builder import StartingInputs, StartingModel, build_starting_model
from osv.model3d.forward_check import ForwardCheck, VelestReference, forward_check, phase_vgrids
from osv.model3d.grid import ModelGrid, define_grid
from osv.model3d.interfaces import build_interfaces, interface_depth_at, place_sources
from osv.model3d.plots import make_all_figures
from osv.model3d.sources import (
    FmtomoData,
    FmtomoDataError,
    build_fmtomo_data,
    load_velest_tables,
    write_fmtomo_data,
)
from osv.report.checks import QCFinding
from osv.stages import Stage, StageContext, register

#: Hypocentres (``velest`` relocations) and event subset written for Stage 4.
HYPOCENTRES = "velest"
SUBSET = "calibration"
FMTOMO_DIR = "fmtomo"
FORWARD_DIR = "forward_check"
#: runs of ``model3d.estimate_station_terms`` (heavy files removed)
TERM_ESTIMATION_DIR = "station_term_estimation"
MODEL_FILE = "starting_model.nc"
GRID_FILE = "grid.json"
FORWARD_FILE = "forward_check.json"

_FILES = {
    "P": {
        "vgrids": "vgrids.in",
        "sources.in": "sources.in",
        "sourcesref.in": "sourcesref.in",
        "receivers.in": "receivers.in",
        "otimes.dat": "otimes_P.dat",
        "otimes_mapping.csv": "otimes_P_mapping.csv",
        "events": "events.csv",
    },
    "S": {
        "vgrids": "vgrids_S.in",
        "sources.in": "sources_S.in",
        "sourcesref.in": "sourcesref_S.in",
        "receivers.in": "receivers_S.in",
        "otimes.dat": "otimes_S.dat",
        "otimes_mapping.csv": "otimes_S_mapping.csv",
        "events": "events_S.csv",
    },
}
_LEVEL = {"info": "info", "warning": "warn", "error": "fail"}


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def grid_summary(grid: ModelGrid) -> dict[str, Any]:
    """JSON-able description of a :class:`~osv.model3d.grid.ModelGrid` (``grid.json``)."""
    pg = grid.propgrid
    return {
        "shape_interior": list(grid.shape),
        "shape_full": list(grid.shape_full),
        "n_nodes": grid.n_nodes,
        "n_prop_nodes": grid.n_prop_nodes,
        "fm3d_memory_gb_estimate": round(grid.fm3d_memory_gb, 3),
        "spacing_km": {"x": grid.dx_km, "y": grid.dy_km, "z": grid.dz_km},
        "spacing_deg": {"lat": grid.dlat_deg, "lon": grid.dlon_deg},
        "lat_range": [float(grid.lat[0]), float(grid.lat[-1])],
        "lon_range": [float(grid.lon[0]), float(grid.lon[-1])],
        "depth_range_km": [float(grid.depth[0]), float(grid.depth[-1])],
        "centre": {"lat": grid.centre_lat, "lon": grid.centre_lon},
        "n_cushion": grid.n_cushion,
        "propgrid": dataclasses.asdict(pg),
        "summary": grid.summary(),
        "warnings": list(grid.warnings),
    }


def grid_from_outputs(stage_dir: Path) -> ModelGrid:
    """Rebuild the :class:`ModelGrid` from ``starting_model.nc``, ``grid.json``, ``propgrid.in``."""
    info = json.loads((stage_dir / GRID_FILE).read_text())
    with xr.open_dataset(stage_dir / MODEL_FILE) as ds:
        lat_f, lon_f, dep_f = (ds[k].values.astype(float) for k in ("lat", "lon", "depth"))
    c = int(info["n_cushion"])
    return ModelGrid(
        lat=lat_f[c:-c],
        lon=lon_f[c:-c],
        depth=dep_f[c:-c],
        lat_full=lat_f,
        lon_full=lon_f,
        depth_full=dep_f,
        dlat_deg=info["spacing_deg"]["lat"],
        dlon_deg=info["spacing_deg"]["lon"],
        dz_km=info["spacing_km"]["z"],
        dx_km=info["spacing_km"]["x"],
        dy_km=info["spacing_km"]["y"],
        propgrid=read_propgrid(stage_dir / FMTOMO_DIR / "propgrid.in"),
        centre_lat=info["centre"]["lat"],
        centre_lon=info["centre"]["lon"],
        n_cushion=c,
        warnings=list(info.get("warnings", [])),
    )


def _save_model(ds: xr.Dataset, path: Path) -> None:
    out = ds.copy()
    out.attrs = {
        k: ("\n".join(map(str, v)) if isinstance(v, list | tuple) else v)
        for k, v in ds.attrs.items()
    }
    out.to_netcdf(path)


def _load_model(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as raw:
        ds = raw.load()
    if "unit_names" in ds.attrs:
        ds.attrs["unit_names"] = str(ds.attrs["unit_names"]).split("\n")
    return ds


def _events_for_grid(events: pd.DataFrame, hyp: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 events with the VELEST relocated position where there is one."""
    ev = events.copy()
    h = hyp.drop_duplicates("event_id").set_index("event_id")
    have = ev["event_id"].isin(h.index)
    for col, src in (("lat", "lat"), ("lon", "lon"), ("depth_km_bsl", "depth_km")):
        ev.loc[have, col] = h.loc[ev.loc[have, "event_id"], src].to_numpy()
    return ev


def _phases(cfg: Any) -> tuple[str, ...]:
    ph = set(cfg.fmtomo.phases)
    return ("P", "S") if ({"S", "vpvs"} & ph) else ("P",)


def _finding(code: str, severity: str, message: str, count: int = 0) -> QCFinding:
    return QCFinding(code, severity, message, count)  # type: ignore[arg-type]


def _user_inputs(cfg: Any) -> list[Path]:
    um = cfg.model3d.user_model
    paths: list[Path] = []
    if cfg.model3d.surface == "dem" and cfg.inputs.dem is not None:
        paths.append(Path(cfg.inputs.dem))
    if um.grid_file is not None:
        paths.append(Path(um.grid_file))
    paths += [Path(h.surface) for h in um.horizons]
    paths += [Path(p) for p in (um.wells.las_dir, um.wells.table) if p is not None]
    return paths


# ----------------------------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------------------------


@register
class BuildModelStage(Stage):
    """Stage 3: grid, starting model, interfaces, FMTOMO data, forward check."""

    name = "build_model"
    requires = ("prep", "velest_analyse")
    config_sections = ("model3d", "coordinates", "fmtomo.phases", "inputs.dem")
    tools = ("fm3d",)
    outputs = (
        f"{FMTOMO_DIR}/vgrids.in",
        f"{FMTOMO_DIR}/interfaces.in",
        f"{FMTOMO_DIR}/propgrid.in",
        f"{FMTOMO_DIR}/sources.in",
        f"{FMTOMO_DIR}/receivers.in",
        f"{FMTOMO_DIR}/otimes_P.dat",
        f"{FMTOMO_DIR}/otimes_P_mapping.csv",
        MODEL_FILE,
        GRID_FILE,
        FORWARD_FILE,
    )

    def extra_sections(self, cfg: Any) -> tuple[str, ...]:
        # the term estimation relocates as the inversion will
        if cfg.model3d.estimate_station_terms:
            return ("fmtomo.relocate", "fmtomo.package", "fmtomo.gridloc", "fmtomo.joint")
        return ()

    def inputs(self, ctx: StageContext) -> list[Path]:
        return _user_inputs(ctx.config)

    # ------------------------------------------------------------------ run

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        log = ctx.logger
        prep, an = ctx.upstream("prep"), ctx.upstream("velest_analyse")
        stations = pd.read_parquet(prep.path("stations.parquet"))
        events = pd.read_parquet(prep.path("events.parquet"))
        picks = pd.read_parquet(prep.path("picks.parquet"))
        model_1d = pd.read_csv(an.path("final_model.csv"))
        tabs = load_velest_tables(an.dir)
        hyp = tabs["hypocentre_table"]
        findings: list[QCFinding] = []

        # ---- grid
        grid = define_grid(stations, _events_for_grid(events, hyp), cfg)
        log.info("grid: %s", grid.summary())
        for w in grid.warnings:
            findings.append(_finding("grid_warning", "warn", w))
        mem_limit = float(cfg.model3d.grid.max_memory_gb)
        findings.append(
            _finding(
                "grid_size",
                "warn" if grid.fm3d_memory_gb > mem_limit else "info",
                f"{grid.summary()}; estimated fm3d memory {grid.fm3d_memory_gb:.1f} GB per process"
                f" (limit {mem_limit:g} GB)",
                grid.n_nodes,
            )
        )

        # ---- starting model
        inp = StartingInputs(stations=stations)
        if cfg.model3d.source in ("wells", "geology+wells"):
            from osv.model3d.wells import load_wells

            inp.wells, wf = load_wells(cfg.model3d.user_model.wells, grid)
            for f in wf:
                findings.append(_finding(f.code, _LEVEL.get(f.level, "info"), f.message, f.n_nodes))
        model = build_starting_model(grid, cfg, model_1d, inp)
        for f in model.findings:
            findings.append(_finding(f.code, _LEVEL.get(f.level, "info"), f.message, f.n_nodes))

        # ---- interfaces and sources
        fnd: list[str] = []
        interfaces = build_interfaces(grid, stations, cfg, cfg.inputs.dem, findings=fnd)
        hyp_placed = place_sources(interfaces, grid, hyp, findings=fnd)
        for m in fnd:
            bad = any(k in m for k in ("remain above", "clipped", "outside", "below the base"))
            findings.append(_finding("interfaces", "warn" if bad else "info", m))

        def surface(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
            return interface_depth_at(interfaces, grid, lat, lon, 0)

        # ---- FMTOMO data
        def build_data(cfg_: Any, delays: pd.DataFrame | None, report: bool) -> dict:
            out: dict[str, FmtomoData] = {}
            for ph in _phases(cfg):
                try:
                    out[ph] = build_fmtomo_data(
                        picks,
                        events,
                        stations,
                        cfg_,
                        grid=grid,
                        hypocentres=HYPOCENTRES,
                        subset=SUBSET,
                        phases=(ph,),
                        hypocentre_table=hyp_placed,
                        station_delays=delays,
                        surface_depth_km=surface,
                        layout="separate",
                    )
                except FmtomoDataError as exc:
                    if ph == "P":
                        raise
                    if report:
                        findings.append(_finding("no_s_data", "warn", f"S data not written: {exc}"))
                    continue
                if report:
                    for m in out[ph].findings:
                        findings.append(_finding(f"data_{ph}", "info", f"{ph}: {m}"))
            return out

        datas = build_data(cfg, tabs.get("station_delays"), not cfg.model3d.estimate_station_terms)
        if cfg.model3d.estimate_station_terms:
            datas = self._estimate_terms(ctx, interfaces, grid, model, datas, build_data, findings)

        # ---- forward check
        ref = VelestReference(
            rms_s=_velest_rms(an.dir),
            events=hyp_placed[["event_id", "rms"]] if "rms" in hyp_placed else None,
            delays=tabs.get("station_delays"),
        )
        fc = forward_check(
            model,
            grid,
            interfaces,
            list(datas.values()),
            ctx.stage_dir / FORWARD_DIR,
            ctx.n_jobs,
            velest=ref,
        )
        findings.extend(fc.findings)

        # ---- outputs
        out = ctx.stage_dir
        self._write_fmtomo(out, interfaces, grid, model, datas)
        _save_model(model.dataset, out / MODEL_FILE)
        (out / GRID_FILE).write_text(json.dumps(grid_summary(grid), indent=2))
        fcd = out / FORWARD_DIR
        fc.residuals.to_parquet(fcd / "residuals.parquet")
        fc.by_station.to_csv(fcd / "by_station.csv", index=False)
        fc.by_event.to_csv(fcd / "by_event.csv", index=False)
        (out / FORWARD_FILE).write_text(json.dumps(_clean(fc.to_json()), indent=2))

        for f in findings:
            ctx.add_check(f)
        self._headline(ctx, grid, model, datas, fc)
        ctx._build_state = SimpleNamespace(wells=inp.wells)  # type: ignore[attr-defined]
        self.plot(ctx)

    @staticmethod
    def _write_fmtomo(
        out: Path,
        interfaces: Any,
        grid: ModelGrid,
        model: StartingModel,
        datas: dict[str, FmtomoData],
    ) -> None:
        """FMTOMO hand-off files of all phases in ``out/fmtomo/``."""
        fm = out / FMTOMO_DIR
        fm.mkdir(exist_ok=True)
        write_interfaces(fm / "interfaces.in", interfaces)
        write_propgrid(fm / "propgrid.in", grid.propgrid)
        for ph, data in datas.items():
            names = _FILES[ph]
            write_vgrids(fm / names["vgrids"], phase_vgrids(model.dataset, grid, data))
            tmp = fm / f"_tmp_{ph}"
            files = write_fmtomo_data(data, tmp)
            for src, dst in ((k, v) for k, v in names.items() if k in files):
                shutil.move(str(files[src]), fm / dst)
            shutil.rmtree(tmp, ignore_errors=True)
            data.events.to_csv(fm / names["events"], index=False)
        datas["P"].stations.to_csv(fm / "stations.csv", index=False)

    def _estimate_terms(
        self,
        ctx: StageContext,
        interfaces: Any,
        grid: ModelGrid,
        model: StartingModel,
        datas: dict[str, FmtomoData],
        build_data: Any,
        findings: list[QCFinding],
    ) -> dict[str, FmtomoData]:
        """``model3d.estimate_station_terms``: terms with fixed velocities, then corrected data."""
        from osv.fmtomo.station_terms import (
            HISTORY_FILE,
            TERMS_FILE,
            delay_table,
            estimate_station_terms,
        )

        cfg = ctx.config
        out = ctx.stage_dir
        _save_model(model.dataset, out / MODEL_FILE)
        (out / GRID_FILE).write_text(json.dumps(grid_summary(grid), indent=2))
        self._write_fmtomo(out, interfaces, grid, model, datas)
        ctx.logger.info("estimating P/S station terms with fixed velocities")
        est = estimate_station_terms(out, cfg, out / TERM_ESTIMATION_DIR, ctx.n_jobs)
        cfg_apply = cfg.model_copy(deep=True)
        cfg_apply.model3d.apply_velest_station_delays = True
        cfg_apply.model3d.velest_station_delays_exclude = []
        corrected = build_data(cfg_apply, delay_table(est.terms), False)
        fm = out / FMTOMO_DIR
        est.terms.to_csv(fm / TERMS_FILE, index=False)
        est.history.to_csv(fm / HISTORY_FILE, index=False)
        est.iterations.to_csv(out / TERM_ESTIMATION_DIR / "iterations.csv", index=False)
        findings.extend(est.findings)
        for ph, t in est.terms.groupby("phase"):
            w = t["n"].clip(lower=1).to_numpy(float)
            v = t["term_s"].to_numpy(float)
            rms = 1000 * float(np.sqrt(np.average(v**2, weights=w)))
            findings.append(
                _finding(
                    f"data_{ph}",
                    "info",
                    f"{ph}: 3-D station terms (fixed velocities) subtracted from the observed "
                    f"times: rms {rms:.1f} ms, range {1000 * v.min():.0f}..{1000 * v.max():.0f} ms "
                    f"({int((t['n'] >= cfg.model3d.station_term_estimation.min_picks).sum())} "
                    f"stations), drift {est.last_step_ms.get(ph, 0.0):.2f} ms per update",
                )
            )
            ctx.summary[f"station_terms_rms_ms_{ph}"] = round(rms, 1)
            ctx.summary[f"station_terms_drift_ms_{ph}"] = round(est.last_step_ms.get(ph, 0), 2)
        return corrected

    @staticmethod
    def _headline(
        ctx: StageContext,
        grid: ModelGrid,
        model: StartingModel,
        datas: dict[str, FmtomoData],
        fc: ForwardCheck,
    ) -> None:
        s = ctx.summary
        s["source"] = ctx.config.model3d.source
        s["grid_interior"] = " x ".join(map(str, grid.shape))
        s["grid_nodes"] = grid.n_nodes
        s["spacing_km"] = f"{grid.dx_km:.2f} x {grid.dy_km:.2f} x {grid.dz_km:.2f}"
        s["propagation_grid"] = (
            f"{grid.propgrid.nr} x {grid.propgrid.nlat} x {grid.propgrid.nlon} "
            f"({grid.n_prop_nodes / 1e6:.2f}M nodes, {grid.propgrid.dr_km:.3f} km)"
        )
        s["sources"] = datas["P"].n_sources
        s["receivers"] = datas["P"].n_receivers
        for ph, d in datas.items():
            s[f"observations_{ph}"] = d.n_observations
        s["vp_range_km_s"] = [round(model.summary["vp_min"], 3), round(model.summary["vp_max"], 3)]
        s["user_model_influence"] = round(model.summary["influenced_fraction"], 4)
        s["max_gradient_km_s_per_km"] = round(model.summary["max_gradient_km_s_per_km"], 3)
        s["fm3d_runtime_s"] = round(fc.runtime_s, 1)
        s["fm3d_jobs"] = fc.n_jobs
        for ph, e in fc.summary["phases"].items():
            s[f"rms_{ph}_background_s"] = round(e["background"]["rms_s"], 4)
            s[f"rms_{ph}_starting_s"] = round(e["start"]["rms_s"], 4)
            s[f"rms_{ph}_demeaned_starting_s"] = round(e["start"]["rms_demeaned_s"], 4)
            s[f"verdict_{ph}"] = e["verdict"]
        vc = fc.summary.get("velest", {})
        if "velest_rms_s" in vc:
            s["velest_rms_s"] = round(vc["velest_rms_s"], 4)

    # ------------------------------------------------------------------ plot

    def plot(self, ctx: StageContext) -> None:
        out = ctx.stage_dir
        cfg = ctx.config
        ds = _load_model(out / MODEL_FILE)
        grid = grid_from_outputs(out)
        interfaces = read_interfaces(out / FMTOMO_DIR / "interfaces.in")
        st = pd.read_csv(out / FMTOMO_DIR / "stations.csv")
        st["sensor_elevation_m"] = -st["depth_km"] * 1000.0
        ev = pd.read_csv(out / FMTOMO_DIR / _FILES["P"]["events"])
        res_path = out / FORWARD_DIR / "residuals.parquet"
        residuals = pd.read_parquet(res_path) if res_path.exists() else None
        wells = getattr(getattr(ctx, "_build_state", None), "wells", None)
        if wells is None and cfg.model3d.source in ("wells", "geology+wells"):
            try:
                from osv.model3d.wells import load_wells

                wells, _ = load_wells(cfg.model3d.user_model.wells, grid)
            except Exception:
                ctx.logger.warning("could not reload the wells for plotting", exc_info=True)
        make_all_figures(
            ctx.figures.save,
            ds=ds,
            grid=grid,
            interfaces=interfaces,
            stations=st,
            events=ev,
            residuals=residuals,
            wells=wells,
            profiles=cfg.plotting.profiles,
            depth_slices_km=cfg.plotting.depth_slices_km,
            checks=cfg.model3d.checks,
        )


def _velest_rms(an_dir: Path) -> float | None:
    p = an_dir / "summary.json"
    try:
        v = json.loads(p.read_text()).get("final_model_rms")
    except (OSError, ValueError):
        return None
    return float(v) if isinstance(v, int | float) else None


def _clean(obj: Any) -> Any:
    """Strictly JSON-serialisable copy (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | np.ndarray):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if np.isfinite(obj) else None
    return obj
