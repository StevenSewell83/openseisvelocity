"""Forward check of the starting model: ``fm3d`` travel times against the observed picks (task 4.9).

:func:`forward_check` writes a run directory for the starting model (and, for a user model, for
the 1-D background it was blended into), traces the rays with
:func:`osv.fmtomo.forward.run_fm3d_parallel` and compares the calculated times with the observed
travel times that Stage 3 writes to ``otimes_*.dat``.

Residual convention
-------------------
``residual = observed - calculated`` (positive when the model is too fast).  The observed time is
the pick time minus the origin time of the *relocated* VELEST hypocentre, so a model that differs
from the VELEST model also leaves an origin-time offset per event.  The summary therefore reports
two RMS values per model: ``rms_s`` (as is) and ``rms_demeaned_s`` (event mean removed, what a
relocation of the origin times alone would leave).  ``mad_s`` is the median absolute deviation
about the median (unscaled).

Comparisons
-----------
* **1-D background**: for a user model (any model that differs from its background) the same
  picks are traced through ``vp_bg``/``vs_bg``.  The verdict per phase is ``better`` or
  ``worse`` if the RMS changes by more than :data:`VERDICT_TOLERANCE`, else ``same``; a
  starting model that fits *worse* than the background it replaces is reported as a warning.
* **VELEST**: the Stage 2 RMS (all phases, with station delays and its own layered 1-D model) and
  the per-event RMS and station delays (:class:`VelestReference`).  The station delays
  ``ptcor`` should resemble the mean P residual of each station in the 1-D background, because
  VELEST absorbed the near-surface effects into them; the correlation is reported.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.coords import spherical_distance_km
from osv.fmtomo.forward import run_fm3d_parallel
from osv.io.fmtomo import (
    FrechetSpec,
    InterfaceGrids,
    ModeSet,
    write_interfaces,
    write_propgrid,
    write_vgrids,
)
from osv.io.fmtomo.grids import VelocityGrids
from osv.model3d.builder import StartingModel
from osv.model3d.extrude import extrusion_to_vgrids
from osv.model3d.grid import ModelGrid
from osv.model3d.sources import FmtomoData
from osv.report.checks import QCFinding

log = logging.getLogger(__name__)

__all__ = [
    "VERDICT_TOLERANCE",
    "ForwardCheck",
    "VelestReference",
    "forward_check",
    "phase_vgrids",
    "residual_stats",
]

#: Relative RMS change below which the starting model and the background fit equally well.
VERDICT_TOLERANCE = 0.02
#: A residual beyond this many robust standard deviations (1.4826 MAD) counts as an outlier.
OUTLIER_SIGMA = 5.0
#: Warn when the mean residual exceeds this fraction of the RMS (a bias, e.g. unmodelled delays).
BIAS_FRACTION = 0.5
#: Mapping columns copied to the residual table.
_KEEP = ["source_idx", "receiver_idx", "ray_idx", "event_id", "station", "phase", "pick_id"]


@dataclass
class VelestReference:
    """VELEST (Stage 2) numbers to compare the 3-D forward check with (all optional).

    Attributes
    ----------
    rms_s
        RMS residual (all phases) of the Stage 2 fixed-model run.
    events
        Table with ``event_id`` and ``rms`` (VELEST per-event RMS).
    delays
        Table with ``alias`` (or ``sta``), ``ptcor`` and ``stcor`` (station delays, s).
    """

    rms_s: float | None = None
    events: pd.DataFrame | None = None
    delays: pd.DataFrame | None = None


@dataclass
class ForwardCheck:
    """Result of :func:`forward_check`.

    Attributes
    ----------
    residuals
        One row per observed pick: mapping columns, ``dist_km`` (hypocentral), ``t_obs``,
        ``t_start``, ``t_bg``, ``res_start``, ``res_bg`` (NaN where fm3d returned no valid time
        or the background run was skipped) and the source/receiver positions.
    summary
        Nested dict (see :meth:`to_json`): per phase ``start``/``background`` statistics,
        ``verdict``, run times and the VELEST comparison.
    by_station, by_event
        Statistics per (station, phase) and per (event, phase).
    findings
        QC findings (:class:`~osv.report.checks.QCFinding`).
    """

    residuals: pd.DataFrame
    summary: dict[str, Any]
    by_station: pd.DataFrame
    by_event: pd.DataFrame
    runtime_s: float
    n_jobs: int
    findings: list[QCFinding] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """JSON-able summary (statistics, run times, findings; not the residual tables)."""
        return {
            **self.summary,
            "runtime_s": round(self.runtime_s, 3),
            "n_jobs": self.n_jobs,
            "findings": [f.to_dict() for f in self.findings],
        }


def residual_stats(res: np.ndarray, event: np.ndarray | None = None) -> dict[str, float]:
    """``n``, ``mean_s``, ``median_s``, ``rms_s``, ``mad_s`` and ``rms_demeaned_s`` of residuals.

    NaN residuals are ignored.  ``event`` (labels of the same length) enables the RMS after
    removing each event's mean residual.
    """
    r = np.asarray(res, dtype=float)
    ok = np.isfinite(r)
    n = int(ok.sum())
    if n == 0:
        return dict.fromkeys(("mean_s", "median_s", "rms_s", "mad_s", "rms_demeaned_s"), np.nan) | {
            "n": 0
        }
    x = r[ok]
    med = float(np.median(x))
    out = {
        "n": n,
        "mean_s": float(x.mean()),
        "median_s": med,
        "rms_s": float(np.sqrt(np.mean(x**2))),
        "mad_s": float(np.median(np.abs(x - med))),
        "rms_demeaned_s": np.nan,
    }
    if event is not None:
        d = x - pd.Series(x).groupby(np.asarray(event)[ok]).transform("mean").to_numpy()
        out["rms_demeaned_s"] = float(np.sqrt(np.mean(d**2)))
    return out


def phase_vgrids(
    ds: Any, grid: ModelGrid, data: FmtomoData, *, background: bool = False
) -> VelocityGrids:
    """Velocity grids for the paths of ``data``.

    Sources whose paths all use velocity type 1 get one type: Vs if ``data.phases == ('S',)``
    (the ``layout='separate'`` S run), else Vp.  Otherwise (joint P/S layout) Vp and Vs.
    ``background`` selects ``vp_bg``/``vs_bg`` of the dataset.
    """
    vts = {v for s in data.sources for p in s.paths for v in p.vtypes}
    d = ds
    if background:
        d = ds.copy()
        d["vp"], d["vs"] = ds["vp_bg"], ds["vs_bg"]
    if vts <= {1}:
        field_ = "vs" if tuple(data.phases) == ("S",) else "vp"
        sub = d.copy()
        sub["vp"] = d[field_]
        if f"{field_}_unc" in d:
            sub["vp_unc"] = d[f"{field_}_unc"]
        return extrusion_to_vgrids(grid, sub, with_s=False)
    return extrusion_to_vgrids(grid, d, with_s=True)


def _write_run_dir(
    run: Path, vg: VelocityGrids, grid: ModelGrid, interfaces: InterfaceGrids
) -> None:
    run.mkdir(parents=True, exist_ok=True)
    write_vgrids(run / "vgrids.in", vg)
    write_interfaces(run / "interfaces.in", interfaces)
    write_propgrid(run / "propgrid.in", grid.propgrid)
    ModeSet().write(run / "mode_set.in")
    FrechetSpec(False, [], [], []).write(run / "frechet.in")


def _trace(
    ds: Any,
    grid: ModelGrid,
    interfaces: InterfaceGrids,
    data: FmtomoData,
    run: Path,
    n_jobs: int,
    *,
    background: bool,
) -> tuple[pd.DataFrame, float, int]:
    """Run fm3d for one data set and model; returns ``(arrivals, wall_s, n_jobs_used)``."""
    _write_run_dir(run, phase_vgrids(ds, grid, data, background=background), grid, interfaces)
    res = run_fm3d_parallel(run, data.sources, data.receivers, n_jobs, workdir=run / "out")
    arr = res.arrivals()
    keep = run / "arrivals.dat"
    shutil.copyfile(res.arrivals_path, keep)
    for name in ("vgrids.in", "interfaces.in", "propgrid.in", "mode_set.in", "frechet.in"):
        (run / name).unlink(missing_ok=True)
    shutil.rmtree(run / "out", ignore_errors=True)
    return arr, res.wall_s, res.n_jobs


def _residual_table(
    data: FmtomoData, arr_start: pd.DataFrame, arr_bg: pd.DataFrame | None
) -> pd.DataFrame:
    m = data.mapping
    m = m[m["valid"]][[*_KEEP, "time"]].rename(columns={"time": "t_obs"}).copy()

    def _times(arr: pd.DataFrame, name: str) -> pd.DataFrame:
        a = arr[arr["valid"]] if "valid" in arr else arr
        a = a.rename(columns={"receiver": "receiver_idx", "source": "source_idx", "ray": "ray_idx"})
        return a[["receiver_idx", "source_idx", "ray_idx", "time"]].rename(columns={"time": name})

    key = ["receiver_idx", "source_idx", "ray_idx"]
    df = m.merge(_times(arr_start, "t_start"), on=key, how="left")
    if arr_bg is not None:
        df = df.merge(_times(arr_bg, "t_bg"), on=key, how="left")
    else:
        df["t_bg"] = df["t_start"]
    df["res_start"] = df["t_obs"] - df["t_start"]
    df["res_bg"] = df["t_obs"] - df["t_bg"]
    src = data.events.set_index("source_idx")
    pos = data.receivers.positions
    s = src.loc[df["source_idx"]]
    p = pos.loc[df["receiver_idx"]]
    df["src_lat"], df["src_lon"] = s["lat"].to_numpy(), s["lon"].to_numpy()
    df["src_depth_km"] = s["depth_km_bsl"].to_numpy()
    df["sta_lat"], df["sta_lon"] = p["lat"].to_numpy(), p["lon"].to_numpy()
    df["sta_depth_km"] = p["depth_km"].to_numpy()
    epi = spherical_distance_km(df["src_lat"], df["src_lon"], df["sta_lat"], df["sta_lon"])
    df["dist_km"] = np.hypot(np.asarray(epi, dtype=float), df["src_depth_km"] - df["sta_depth_km"])
    return df


def _group_stats(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for (key, ph), g in df.groupby([by, "phase"], sort=True):
        row: dict[str, Any] = {by: key, "phase": ph}
        for tag, col in (("start", "res_start"), ("bg", "res_bg")):
            st = residual_stats(g[col].to_numpy())
            row.update(
                {f"n_{tag}": st["n"], f"mean_{tag}_s": st["mean_s"], f"rms_{tag}_s": st["rms_s"]}
            )
        row["n"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _velest_comparison(
    df: pd.DataFrame, by_event: pd.DataFrame, by_station: pd.DataFrame, ref: VelestReference
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if ref.rms_s is not None and np.isfinite(ref.rms_s):
        out["velest_rms_s"] = float(ref.rms_s)
        out["background_rms_all_phases_s"] = residual_stats(df["res_bg"].to_numpy())["rms_s"]
        out["start_rms_all_phases_s"] = residual_stats(df["res_start"].to_numpy())["rms_s"]
    if ref.events is not None and len(ref.events):
        e = by_event.groupby("event_id").apply(
            lambda g: float(np.sqrt(np.average(g["rms_bg_s"] ** 2, weights=g["n"]))),
            include_groups=False,
        )
        j = ref.events.drop_duplicates("event_id").set_index("event_id")["rms"].reindex(e.index)
        out["event_rms_corr_with_velest"] = _corr(e.to_numpy(), j.to_numpy(float))
    if ref.delays is not None and len(ref.delays):
        d = ref.delays.copy()
        key = "alias" if "alias" in d else "sta"
        d = d.drop_duplicates(key).set_index(key)
        for ph, col in (("P", "ptcor"), ("S", "stcor")):
            if col not in d:
                continue
            s = by_station[by_station["phase"] == ph].set_index("station")
            if s.empty:
                continue
            v = d[col].reindex(s.index).to_numpy(float)
            # VELEST: t_obs = t_calc + delay + residual  =>  delay ~ mean(obs - calc)
            out[f"station_{ph}_delay_corr_with_velest"] = _corr(s["mean_bg_s"].to_numpy(), v)
    return out


def forward_check(
    model: StartingModel,
    grid: ModelGrid,
    interfaces: InterfaceGrids,
    fmtomo_data: FmtomoData | Sequence[FmtomoData],
    workdir: str | Path,
    n_jobs: int,
    *,
    velest: VelestReference | None = None,
    compare_background: bool = True,
) -> ForwardCheck:
    """Trace the observed rays through the starting model and summarise the residuals.

    Parameters
    ----------
    model
        :class:`~osv.model3d.builder.StartingModel` (``vp``, ``vs``, ``vp_bg``, ``vs_bg``).
    grid, interfaces
        The model grid and its ``interfaces.in`` content.
    fmtomo_data
        One :class:`~osv.model3d.sources.FmtomoData` or a sequence (e.g. P and an S set built
        with ``layout='separate'``); each is traced separately.
    workdir
        Directory for the run directories (``<workdir>/start_<phases>/``, ``bg_<phases>/``);
        only ``arrivals.dat`` is kept in each.
    n_jobs
        Maximum parallel ``fm3d`` processes (see :func:`~osv.fmtomo.forward.run_fm3d_parallel`).
    velest
        Optional Stage 2 numbers for the VELEST comparison.
    compare_background
        Also trace through the 1-D background when the model differs from it.

    Returns
    -------
    ForwardCheck
    """
    t0 = time.perf_counter()
    datas = [fmtomo_data] if isinstance(fmtomo_data, FmtomoData) else list(fmtomo_data)
    work = Path(workdir)
    ds = model.dataset
    differs = not (
        np.allclose(ds["vp"].values, ds["vp_bg"].values, equal_nan=True)
        and np.allclose(ds["vs"].values, ds["vs_bg"].values, equal_nan=True)
    )
    run_bg = differs and compare_background
    findings: list[QCFinding] = []
    tables: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}
    n_used = 1
    for data in datas:
        tag = "".join(data.phases)
        arr_s, wall, nj = _trace(
            ds, grid, interfaces, data, work / f"start_{tag}", n_jobs, background=False
        )
        runtimes[f"start_{tag}"] = round(wall, 2)
        n_used = max(n_used, nj)
        arr_b = None
        if run_bg:
            arr_b, wall, nj = _trace(
                ds, grid, interfaces, data, work / f"bg_{tag}", n_jobs, background=True
            )
            runtimes[f"bg_{tag}"] = round(wall, 2)
        n_inv = int((~arr_s["valid"]).sum()) if "valid" in arr_s else 0
        if n_inv:
            findings.append(
                QCFinding(
                    "fm3d_invalid_arrivals",
                    "warn",
                    f"fm3d returned no valid {tag} time for {n_inv} of {len(arr_s)} rays "
                    "(source or receiver outside the model, or a ray that failed); they are "
                    "left out of the residuals",
                    n_inv,
                )
            )
        tables.append(_residual_table(data, arr_s, arr_b))
    df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    by_station = _group_stats(df, "station")
    by_event = _group_stats(df, "event_id")

    summary: dict[str, Any] = {
        "model_differs_from_background": bool(differs),
        "background_run": bool(run_bg),
        "run_seconds": runtimes,
        "n_picks": int(len(df)),
        "phases": {},
    }
    for ph in sorted(df["phase"].unique()) if len(df) else []:
        g = df[df["phase"] == ph]
        st = residual_stats(g["res_start"].to_numpy(), g["event_id"].to_numpy())
        bg = residual_stats(g["res_bg"].to_numpy(), g["event_id"].to_numpy())
        entry: dict[str, Any] = {"start": st, "background": bg}
        if differs and np.isfinite(st["rms_s"]) and np.isfinite(bg["rms_s"]) and bg["rms_s"] > 0:
            ratio = st["rms_s"] / bg["rms_s"]
            entry["rms_ratio"] = ratio
            entry["verdict"] = (
                "better"
                if ratio < 1 - VERDICT_TOLERANCE
                else "worse"
                if ratio > 1 + VERDICT_TOLERANCE
                else "same"
            )
        else:
            entry["verdict"] = "not_applicable"
        summary["phases"][ph] = entry
        n_out = _count_outliers(g["res_start"].to_numpy())
        if n_out:
            findings.append(
                QCFinding(
                    "residual_outliers",
                    "info",
                    f"{n_out} {ph} residual(s) beyond {OUTLIER_SIGMA:g} robust standard "
                    f"deviations of the starting model (MAD {st['mad_s']:.3f} s); check the "
                    "picks or the stations involved",
                    n_out,
                )
            )
        if np.isfinite(st["rms_s"]) and abs(st["mean_s"]) > BIAS_FRACTION * st["rms_s"]:
            findings.append(
                QCFinding(
                    "residual_bias",
                    "warn",
                    f"{ph} residuals have mean {st['mean_s']:+.3f} s (RMS {st['rms_s']:.3f} s): "
                    "the model is systematically "
                    + ("fast" if st["mean_s"] > 0 else "slow")
                    + " for these picks (or station delays are not applied)",
                    st["n"],
                )
            )
        if entry["verdict"] == "worse":
            findings.append(
                QCFinding(
                    "user_model_fits_worse",
                    "warn",
                    f"the starting model fits the {ph} picks worse than its 1-D background: "
                    f"RMS {st['rms_s']:.3f} s against {bg['rms_s']:.3f} s "
                    f"({100 * (entry['rms_ratio'] - 1):+.1f}%)",
                    st["n"],
                    {"phase": ph, "rms_start_s": st["rms_s"], "rms_background_s": bg["rms_s"]},
                )
            )
        elif entry["verdict"] == "better":
            findings.append(
                QCFinding(
                    "user_model_fits_better",
                    "info",
                    f"the starting model fits the {ph} picks better than its 1-D background: "
                    f"RMS {st['rms_s']:.3f} s against {bg['rms_s']:.3f} s "
                    f"({100 * (entry['rms_ratio'] - 1):+.1f}%)",
                    st["n"],
                )
            )
    if velest is not None and len(df):
        vc = _velest_comparison(df, by_event, by_station, velest)
        summary["velest"] = vc
        c = vc.get("station_P_delay_corr_with_velest")
        if c is not None and np.isfinite(c):
            findings.append(
                QCFinding(
                    "velest_delay_agreement",
                    "info" if c > 0.5 else "warn",
                    f"mean P residuals of the 1-D background correlate with the VELEST station "
                    f"delays with r = {c:.2f}"
                    + ("" if c > 0.5 else " (weak: the 3-D forward model sees different delays)"),
                )
            )
    log.info(
        "forward check: %s",
        {k: (v["start"]["rms_s"], v["background"]["rms_s"]) for k, v in summary["phases"].items()},
    )
    return ForwardCheck(
        df, summary, by_station, by_event, time.perf_counter() - t0, n_used, findings
    )


def _count_outliers(res: np.ndarray) -> int:
    r = res[np.isfinite(res)]
    if r.size < 5:
        return 0
    med = np.median(r)
    sig = 1.4826 * np.median(np.abs(r - med))
    return int((np.abs(r - med) > OUTLIER_SIGMA * sig).sum()) if sig > 0 else 0
