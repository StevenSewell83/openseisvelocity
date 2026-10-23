"""Stage 7: relocate held-out (validation) events with the 1-D and the 3-D model and compare.

:func:`relocate_validation` locates the validation events twice with the *same picks*:

* **1-D benchmark**: VELEST in single-event mode (``isingle = 1``, ``iturbo = 0``,
  ``invertratio = 0``, ``nsinv = 0``) with the final 1-D model and the Stage 2 station delays
  fixed.  The hypocentres and VELEST's own RMS per event are read from the print file.
* **3-D relocation**: FMTOMO with the velocity model *fixed* (``invert_velocity`` off) and the
  events as free sources (``relocate = linear``: hypocentre and origin time in the subspace
  inversion, Gauss-Newton iterations, :func:`osv.fmtomo.invert.run_inversion`).  Linear
  relocation was chosen over Extras ``gridloc`` because the latter needs the swapped
  station-as-source geometry *and* an ``invert3d`` velocity update in every ``tomo3di``
  iteration (the driver cannot run without one), whereas the linear driver can run with source
  parameters only and reuses the Stage 3 data builder unchanged.  The start is the catalogue
  hypocentre (the same start as VELEST).

RMS definition (identical for both, ``VELEST`` formula ``velest.f:4979-4992``): with
pick weights ``w = 1/4**class`` (S: times ``swtfac``) and residual ``r = t_obs - t_calc``,
``rms = sqrt((sum (w r)^2 - (sum w r)^2 / n) / (n - 1))``.  VELEST obtains it with its
fitted origin time; for the 3-D run the origin time is removed per event as the weighted
least-squares shift ``sum(w^2 r) / sum(w^2)`` because FMTOMO's linear relocation does not carry the
origin time forward (``scorrect`` only clamps positions).  Station delays are applied to the 1-D
run always (they belong to the 1-D model) and to the 3-D observed times exactly when
``model3d.apply_velest_station_delays`` is on (the Stage 3 convention).  If 3-D times are not
delay corrected the comparison favours the 1-D model; the result records this
(``ValidationResult.delays_applied_3d``).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.coords import FMTOMO_EARTH_RADIUS_KM
from osv.report.checks import QCFinding

log = logging.getLogger(__name__)

__all__ = [
    "RelocateOptions",
    "ValidationError",
    "ValidationResult",
    "compare_relocations",
    "event_rms",
    "hypocentre_offsets_km",
    "nearest_neighbour_km",
    "paired_rms_stats",
    "relocate_1d",
    "relocate_3d",
    "relocate_validation",
]

#: km per degree of latitude on the FMTOMO sphere.
KM_PER_DEG = FMTOMO_EARTH_RADIUS_KM * math.pi / 180.0
#: ``max_fm3d_jobs``: other stages may be running; never exceed this many ``fm3d`` processes.
MAX_FM3D_JOBS = 8
#: Weight of S relative to P in VELEST (the control default Stage 1 writes).
SWTFAC = 0.75
#: VELEST weight class that is not used.
REJECT_WEIGHT = 4


class ValidationError(RuntimeError):
    """The validation relocation cannot be run (message says what to fix)."""


@dataclass
class RelocateOptions:
    """Tunables of the validation relocation (defaults are sensible for local networks).

    ``n_iterations``: Gauss-Newton iterations of the 3-D relocation.  ``source_sigma_km``:
    a priori standard deviation of the hypocentre (damping of the source updates).
    ``velest_ittmax``: VELEST iterations per event.  ``phases``: ``None`` takes
    ``cfg.fmtomo.phases`` restricted to P/S.  ``vpvs``: Vp/Vs for S in VELEST (``None``:
    mean ``vp/vs`` of the S block of the 1-D model, else ``cfg.velest.vpvs.value``).
    ``worse_fraction_warn``: warn when 3-D is worse than 1-D for more than this fraction
    of the events.  ``min_events_stat``: fewer paired events than this makes the test
    result informational only.  ``improve_tol_ms``: |change| below this counts as a tie.
    """

    n_iterations: int = 5
    source_sigma_km: float = 2.0
    velest_ittmax: int = 20
    phases: tuple[str, ...] | None = None
    vpvs: float | None = None
    worse_fraction_warn: float = 0.5
    min_events_stat: int = 6
    improve_tol_ms: float = 0.5
    max_hypo_shift_warn_km: float = 5.0


@dataclass
class ValidationResult:
    """Outcome of :func:`relocate_validation`.

    ``events`` has one row per validation event (``event_id``, catalogue ``lat0/lon0/depth0``,
    ``lat_1d/lon_1d/depth_1d/rms_1d``, ``lat_3d/lon_3d/depth_3d/rms_3d``, ``n_picks``,
    ``rms_change`` (1d - 3d, s; positive = 3-D better), ``dh_km`` (horizontal 1-D to 3-D
    distance), ``dz_km`` (3-D depth minus 1-D depth), ``dx_km``, ``dy_km``).  ``stats`` is
    the comparison from :func:`compare_relocations`.
    """

    events: pd.DataFrame
    stats: dict[str, Any]
    findings: list[QCFinding] = field(default_factory=list)
    workdir: Path | None = None
    phases: tuple[str, ...] = ("P",)
    delays_applied_3d: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """JSON-able summary (statistics, flags, finding codes)."""
        return {
            "n_events": len(self.events),
            "phases": list(self.phases),
            "delays_applied_3d": self.delays_applied_3d,
            "stats": _jsonable(self.stats),
            "findings": [f.code for f in self.findings],
            "notes": self.notes,
        }

    def write(self, outdir: str | Path | None = None) -> Path:
        """Write ``events.csv`` and ``summary.json`` (default: the workdir)."""
        d = Path(outdir if outdir is not None else (self.workdir or "."))
        d.mkdir(parents=True, exist_ok=True)
        self.events.to_csv(d / "validation_events.csv", index=False)
        (d / "validation_summary.json").write_text(json.dumps(self.summary(), indent=2))
        return d


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------------------------------
# statistics (pure functions)
# ---------------------------------------------------------------------------------------
def event_rms(res: np.ndarray, w: np.ndarray) -> float:
    """VELEST RMS of one event from residuals ``res`` (s) and weights ``w`` (``velest.f:4979``).

    ``sqrt((sum (w r)^2 - (sum w r)^2 / n) / (n - 1))`` over the picks with ``w > 0``;
    ``nan`` for fewer than 2 picks.
    """
    res = np.asarray(res, float)
    w = np.asarray(w, float)
    m = (w > 0) & np.isfinite(res)
    n = int(m.sum())
    if n < 2:
        return float("nan")
    wr = w[m] * res[m]
    ss = float(np.sum(wr * wr) - np.sum(wr) ** 2 / n)
    return math.sqrt(max(ss, 0.0) / (n - 1))


def hypocentre_offsets_km(
    lat_a: Any, lon_a: Any, depth_a: Any, lat_b: Any, lon_b: Any, depth_b: Any
) -> pd.DataFrame:
    """Offsets ``b - a`` in km on the FMTOMO sphere: columns dx_km (east), dy_km, dz_km, dh_km."""
    la, lb = np.asarray(lat_a, float), np.asarray(lat_b, float)
    dy = (lb - la) * KM_PER_DEG
    dx = (np.asarray(lon_b, float) - np.asarray(lon_a, float)) * KM_PER_DEG
    dx = dx * np.cos(np.deg2rad(0.5 * (la + lb)))
    dz = np.asarray(depth_b, float) - np.asarray(depth_a, float)
    return pd.DataFrame({"dx_km": dx, "dy_km": dy, "dz_km": dz, "dh_km": np.hypot(dx, dy)})


def nearest_neighbour_km(lat: Any, lon: Any, depth: Any) -> np.ndarray:
    """Distance (km) from each event to its nearest neighbour (3-D); ``nan`` for one event."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    z = np.asarray(depth, float)
    n = len(lat)
    if n < 2:
        return np.full(n, np.nan)
    lat0 = float(np.nanmean(lat))
    x = (lon - np.nanmean(lon)) * KM_PER_DEG * math.cos(math.radians(lat0))
    y = (lat - lat0) * KM_PER_DEG
    p = np.column_stack([x, y, z])
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def paired_rms_stats(
    rms_1d: Any, rms_3d: Any, *, improve_tol_ms: float = 0.5
) -> dict[str, float | int | bool]:
    """Paired comparison of per-event RMS (seconds) of the 1-D and the 3-D relocation.

    Improvement is ``rms_1d - rms_3d`` (positive: 3-D fits better).  Returns the paired
    count, medians and means of both, median/mean improvement (s and percent of the 1-D
    value), fractions better/worse/tied (tolerance ``improve_tol_ms``), the two-sided
    Wilcoxon signed-rank p-value (``p_two_sided``) and the one-sided p-value for "3-D better"
    (``p_3d_better``).  p-values are ``nan`` with fewer than 2 informative pairs.
    """
    a = np.asarray(rms_1d, float)
    b = np.asarray(rms_3d, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(ok.sum())
    out: dict[str, float | int | bool] = {"n": n}
    nan = float("nan")
    if n == 0:
        out.update(
            median_1d=nan,
            median_3d=nan,
            mean_1d=nan,
            mean_3d=nan,
            median_improvement_s=nan,
            mean_improvement_s=nan,
            median_improvement_pct=nan,
            mean_improvement_pct=nan,
            frac_3d_better=nan,
            frac_3d_worse=nan,
            frac_tied=nan,
            p_two_sided=nan,
            p_3d_better=nan,
        )
        return out
    d = a - b
    tol = improve_tol_ms / 1000.0
    out.update(
        median_1d=float(np.median(a)),
        median_3d=float(np.median(b)),
        mean_1d=float(np.mean(a)),
        mean_3d=float(np.mean(b)),
        median_improvement_s=float(np.median(d)),
        mean_improvement_s=float(np.mean(d)),
        median_improvement_pct=float(100.0 * np.median(d) / np.median(a)) if np.median(a) else nan,
        mean_improvement_pct=float(100.0 * np.mean(d) / np.mean(a)) if np.mean(a) else nan,
        frac_3d_better=float(np.mean(d > tol)),
        frac_3d_worse=float(np.mean(d < -tol)),
        frac_tied=float(np.mean(np.abs(d) <= tol)),
    )
    p2 = p1 = nan
    if np.count_nonzero(d) >= 2:
        from scipy import stats

        try:
            p2 = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
            p1 = float(stats.wilcoxon(d, alternative="greater").pvalue)
        except ValueError:  # all differences zero
            pass
    out["p_two_sided"] = p2
    out["p_3d_better"] = p1
    return out


def compare_relocations(
    events: pd.DataFrame, options: RelocateOptions | None = None
) -> tuple[dict[str, Any], list[QCFinding]]:
    """Statistics and QC findings from a per-event table (see :class:`ValidationResult`).

    Needs ``rms_1d``, ``rms_3d``, ``lat_1d/lon_1d/depth_1d``, ``lat_3d/lon_3d/depth_3d``
    and, for the clustering metric, the catalogue ``lat0/lon0/depth0`` (optional).
    Adds ``rms_change``, ``dx_km``, ``dy_km``, ``dz_km`` and ``dh_km`` to ``events`` in place.
    """
    opt = options or RelocateOptions()
    ev = events
    ev["rms_change"] = ev["rms_1d"] - ev["rms_3d"]
    off = hypocentre_offsets_km(
        ev["lat_1d"], ev["lon_1d"], ev["depth_1d"], ev["lat_3d"], ev["lon_3d"], ev["depth_3d"]
    )
    for c in off:
        ev[c] = off[c].to_numpy()
    stats = {"rms": paired_rms_stats(ev["rms_1d"], ev["rms_3d"], improve_tol_ms=opt.improve_tol_ms)}
    ok = np.isfinite(ev["dh_km"]) & np.isfinite(ev["dz_km"])
    stats["hypocentre_difference"] = {
        "n": int(ok.sum()),
        "dh_median_km": float(np.nanmedian(ev["dh_km"])) if ok.any() else float("nan"),
        "dh_mean_km": float(np.nanmean(ev["dh_km"])) if ok.any() else float("nan"),
        "dh_max_km": float(np.nanmax(ev["dh_km"])) if ok.any() else float("nan"),
        "dz_median_km": float(np.nanmedian(ev["dz_km"])) if ok.any() else float("nan"),
        "dz_mean_km": float(np.nanmean(ev["dz_km"])) if ok.any() else float("nan"),
        "dz_abs_median_km": float(np.nanmedian(np.abs(ev["dz_km"]))) if ok.any() else float("nan"),
    }
    clus: dict[str, float] = {}
    for tag, cols in (
        ("catalogue", ("lat0", "lon0", "depth0")),
        ("1d", ("lat_1d", "lon_1d", "depth_1d")),
        ("3d", ("lat_3d", "lon_3d", "depth_3d")),
    ):
        if all(c in ev for c in cols):
            m = np.isfinite(ev[list(cols)].to_numpy(float)).all(axis=1)
            nn = nearest_neighbour_km(*(ev.loc[m, c] for c in cols))
            clus[f"nn_median_{tag}_km"] = (
                float(np.nanmedian(nn)) if np.isfinite(nn).any() else float("nan")
            )
    stats["clustering"] = clus

    findings: list[QCFinding] = []
    r = stats["rms"]
    n = int(r["n"])
    if n == 0:
        findings.append(
            QCFinding(
                "validation_no_pairs",
                "fail",
                "No validation event has an RMS from both the 1-D and the 3-D relocation.",
            )
        )
        return stats, findings
    if n < opt.min_events_stat:
        findings.append(
            QCFinding(
                "validation_few_events",
                "warn",
                f"Only {n} validation events: the paired statistics are indicative only.",
                n,
            )
        )
    worse = float(r["frac_3d_worse"])
    if worse > opt.worse_fraction_warn:
        findings.append(
            QCFinding(
                "validation_3d_worse",
                "warn",
                f"The 3-D model fits {100 * worse:.0f}% of the held-out events worse than the 1-D "
                f"model (median RMS {1000 * r['median_3d']:.0f} ms vs {1000 * r['median_1d']:.0f} "
                "ms): the 3-D model is over-fitted to the calibration events or the 3-D picks "
                "are not delay corrected; check damping/smoothing and "
                "model3d.apply_velest_station_delays.",
                int(round(worse * n)),
                {"frac_3d_worse": worse},
            )
        )
    elif r["median_improvement_s"] < 0:
        findings.append(
            QCFinding(
                "validation_no_improvement",
                "warn",
                f"The median held-out RMS is not lower for the 3-D model "
                f"({1000 * r['median_3d']:.0f} ms vs {1000 * r['median_1d']:.0f} ms).",
                n,
            )
        )
    big = ev["dh_km"] > opt.max_hypo_shift_warn_km
    if bool(big.any()):
        findings.append(
            QCFinding(
                "validation_large_hypocentre_shift",
                "warn",
                f"{int(big.sum())} event(s) move more than {opt.max_hypo_shift_warn_km:g} km "
                "horizontally between the 1-D and 3-D relocation; these are poorly constrained.",
                int(big.sum()),
                {"event_id": [str(x) for x in ev.loc[big, "event_id"]][:20]}
                if "event_id" in ev
                else {},
            )
        )
    if "nn_median_1d_km" in clus and "nn_median_3d_km" in clus:
        findings.append(
            QCFinding(
                "validation_clustering",
                "info",
                f"Median nearest-neighbour distance: catalogue "
                f"{clus.get('nn_median_catalogue_km', float('nan')):.2f} km, 1-D "
                f"{clus['nn_median_1d_km']:.2f} km, 3-D {clus['nn_median_3d_km']:.2f} km "
                "(smaller = tighter clustering; interpret with care for few events).",
            )
        )
    return stats, findings


# ---------------------------------------------------------------------------------------
# helpers shared by both relocations
# ---------------------------------------------------------------------------------------
def _cfg_section(cfg: Any, name: str) -> Any:
    return getattr(cfg, name, cfg)


def _phases(cfg: Any, options: RelocateOptions) -> tuple[str, ...]:
    if options.phases is not None:
        ph = tuple(options.phases)
    else:
        fm = getattr(cfg, "fmtomo", None)
        raw = tuple(getattr(fm, "phases", ("P",))) if fm is not None else ("P",)
        ph = tuple(p for p in raw if p in ("P", "S"))
        if "vpvs" in raw and "S" not in ph:
            ph = ("P", "S")
    if not ph:
        ph = ("P",)
    return ph


def _validation_events(events: pd.DataFrame, subset: str = "validation") -> pd.DataFrame:
    ev = events
    if subset != "all":
        if "split" not in ev:
            raise ValidationError("events has no 'split' column (Stage 1 events.parquet)")
        ev = ev[ev["split"] == subset]
    if ev.empty:
        raise ValidationError(
            f"no events with split == {subset!r}; set data_selection.validation_fraction > 0"
        )
    if "depth_km_bsl" not in ev:
        raise ValidationError("events lacks depth_km_bsl (run the Stage 1 depth-datum step)")
    return ev.drop_duplicates("event_id").reset_index(drop=True)


def _usable_picks(
    picks: pd.DataFrame, event_ids: Sequence[Any], phases: tuple[str, ...]
) -> pd.DataFrame:
    pk = picks[picks["event_id"].isin(set(event_ids)) & picks["phase"].isin(phases)].copy()
    if "alias" not in pk:
        pk["alias"] = pk["sta"]
    if "velest_weight" in pk:
        pk = pk[pk["velest_weight"] < REJECT_WEIGHT]
    else:
        pk["velest_weight"] = 0
    if "sigma_s" in pk:
        pk = pk.sort_values(["sigma_s", "pick_id"])
    return pk.drop_duplicates(["event_id", "alias", "phase"], keep="first").reset_index(drop=True)


def _stations_from_picks(picks: pd.DataFrame) -> pd.DataFrame:
    need = ["alias", "station_lat", "station_lon", "sensor_elevation_m"]
    miss = [c for c in need if c not in picks]
    if miss:
        raise ValidationError(
            f"picks lack {miss}; pass the Stage 1 stations table via stations=... instead"
        )
    st = picks.drop_duplicates("alias")[need].rename(
        columns={"station_lat": "lat", "station_lon": "lon"}
    )
    return st.reset_index(drop=True)


def _delay_key(delays: pd.DataFrame | None) -> pd.DataFrame | None:
    if delays is None:
        return None
    d = delays.copy()
    if "alias" not in d:
        d["alias"] = d["sta"]
    return d


def pick_weights(picks: pd.DataFrame, swtfac: float) -> np.ndarray:
    """VELEST weight of each pick: ``1/4**class`` (``velest.f:2077``), S times ``swtfac``."""
    w = 1.0 / (4.0 ** picks["velest_weight"].to_numpy(float))
    return np.where(picks["phase"].to_numpy() == "S", swtfac * w, w)


# ---------------------------------------------------------------------------------------
# 1-D: VELEST single-event mode
# ---------------------------------------------------------------------------------------
def _model_for_velest(model_1d: Any, nsp: int) -> tuple[Any, float | None]:
    """VELEST model object (P block only for ``nsp != 2``) and the model's own Vp/Vs."""
    from osv.io.velest import ModelBlock, VelestModel, read_model
    from osv.velest.models import TrialModel

    if isinstance(model_1d, TrialModel):
        vm = VelestModel(
            title=str(getattr(model_1d, "id", "osv"))[:40],
            p=ModelBlock(
                layers=pd.DataFrame(
                    {
                        "vel": np.asarray(model_1d.vp, float),
                        "depth": np.asarray(model_1d.depth_top, float),
                        "vdamp": 1.0,
                        "reflector": " ",
                    }
                ),
                label="P",
            ),
        )
        vpvs = float(np.mean(model_1d.vpvs))
        return vm, vpvs
    vm = model_1d if isinstance(model_1d, VelestModel) else read_model(model_1d, nmod=None)
    vpvs = None
    if vm.s is not None:
        vp = vm.p.layers["vel"].to_numpy(float)
        vs = vm.s.layers["vel"].to_numpy(float)
        if len(vp) == len(vs) and (vs > 0).all():
            # the top (air/dummy) layer ratio is meaningless: use the median over layers
            vpvs = float(np.median(vp / vs))
    if nsp != 2 and vm.s is not None:
        vm = VelestModel(title=vm.title, p=vm.p, s=None)
    return vm, vpvs


def relocate_1d(
    events: pd.DataFrame,
    picks: pd.DataFrame,
    model_1d: Any,
    station_delays: pd.DataFrame | None,
    cfg: Any,
    workdir: str | Path,
    *,
    stations: pd.DataFrame | None = None,
    options: RelocateOptions | None = None,
    timeout: float | None = 1800.0,
) -> pd.DataFrame:
    """Relocate ``events`` with VELEST single-event mode, model and delays fixed.

    ``events`` are the events to locate (any ``split``) with ``event_id``, ``time``, ``lat``,
    ``lon`` and ``depth_km_bsl`` (catalogue start).  Returns one row per located event:
    ``event_id, time, lat, lon, depth_km, rms, n_picks, n_used``.
    """
    from osv.io.velest import (
        VelestControl,
        write_cnv,
        write_control,
        write_model,
        write_stations,
    )
    from osv.io.velest_out import read_run_dir
    from osv.tools import ToolError, run_tool

    opt = options or RelocateOptions()
    phases = _phases(cfg, opt)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    ev = events.drop_duplicates("event_id").copy()
    ev["_t"] = pd.to_datetime(ev["time"], utc=True)
    ev = ev.sort_values(["_t", "event_id"]).reset_index(drop=True)
    pk = _usable_picks(picks, ev["event_id"], phases)
    ev = ev[ev["event_id"].isin(pk["event_id"])].reset_index(drop=True)
    if ev.empty:
        raise ValidationError("no validation event has usable picks for the 1-D relocation")
    ev["event"] = np.arange(len(ev))
    emap = dict(zip(ev["event_id"], ev["event"], strict=True))
    pk["event"] = pk["event_id"].map(emap)
    t0 = pk["event_id"].map(dict(zip(ev["event_id"], ev["_t"], strict=True)))
    pk["tt"] = (pd.to_datetime(pk["time"], utc=True) - t0).dt.total_seconds()
    bad = (pk["tt"] <= 0) | (pk["tt"] >= 100.0)
    pk = pk[~bad]

    # ---- control / model / stations
    vcfg = getattr(cfg, "velest", None)
    nsp = 3 if "S" in phases else 1
    vm, model_vpvs = _model_for_velest(model_1d, nsp)
    vpvs = opt.vpvs
    if vpvs is None:
        vpvs = model_vpvs
    if vpvs is None:
        try:
            vpvs = float(vcfg.vpvs.value)
        except (AttributeError, TypeError, ValueError):
            vpvs = 1.73
    vpvs = round(float(vpvs), 3)

    st = (stations if stations is not None else _stations_from_picks(pk)).copy()
    if "alias" not in st:
        st["alias"] = st["sta"]
    if "sensor_elevation_m" not in st:
        st["sensor_elevation_m"] = st["elevation_m"]
    st = st[st["alias"].isin(set(pk["alias"]))].drop_duplicates("alias")
    dl = _delay_key(station_delays)
    ptcor = np.zeros(len(st))
    if dl is not None and "ptcor" in dl:
        dmap = dl.drop_duplicates("alias").set_index("alias")["ptcor"]
        ptcor = st["alias"].map(dmap).astype(float).fillna(0.0).to_numpy()
    sta_tab = pd.DataFrame(
        {
            "sta": st["alias"].to_numpy(),
            "lat": st["lat"].astype(float).to_numpy(),
            "lon": st["lon"].astype(float).to_numpy(),
            "elev_m": np.rint(st["sensor_elevation_m"].astype(float)).astype(int).to_numpy(),
            "model": 1,
            "icc": np.arange(1, len(st) + 1),
            "ptcor": np.round(ptcor, 3),
            "stcor": 0.0,
        }
    )
    pk = pk[pk["alias"].isin(set(sta_tab["sta"]))]
    cnv_ev = pd.DataFrame(
        {
            "event": ev["event"],
            "time": ev["_t"].dt.tz_convert("UTC").dt.tz_localize(None),
            "lat": ev["lat"].astype(float),
            "lon": ev["lon"].astype(float),
            "depth_km": ev["depth_km_bsl"].astype(float),
            "mag": 0.0,
        }
    )
    cnv_pk = pd.DataFrame(
        {
            "event": pk["event"].astype(int),
            "sta": pk["alias"],
            "phase": pk["phase"],
            "weight": pk["velest_weight"].astype(int),
            "tt": pk["tt"].round(2),
        }
    )
    org = getattr(getattr(cfg, "coordinates", None), "origin", None)
    if org is not None:
        olat, olon = float(org.lat), float(org.lon)
    else:
        olat, olon = float(sta_tab["lat"].mean()), float(sta_tab["lon"].mean())
    zmin = 0.0
    dmin = float(cnv_ev["depth_km"].min())
    if dmin <= 0.0:
        zmin = float(np.floor((dmin - 0.1) * 10.0) / 10.0)
    try:
        dmax = float(cfg.data_selection.max_epi_dist_km)
    except AttributeError:
        dmax = 150.0
    swt = SWTFAC
    ctrl = VelestControl.from_origin(
        olat,
        olon,
        title=f"osv validation 1D {len(ev)} events"[:80],
        neqs=len(ev),
        zshift=0.0,
        iuseelev=1,
        zmin=zmin,
        dmax=dmax,
        nsp=nsp,
        swtfac=swt,
        vpvs=vpvs,
        nmod=1 if nsp != 2 else 2,
        nsinv=0,
        iusestacorr=int(dl is not None),  # 0 would zero the delays of the station file
        isingle=1,
        iturbo=0,
        invertratio=0,
        ittmax=int(opt.velest_ittmax),
        icnvout=1,
        istaout=0,
        modelfile="model.mod",
        stationfile="stations.sta",
        phasefile="events.cnv",
        outfile="velest.out",
        cnvfile="velout.cnv",
        stafile="velout.sta",
    )
    rd = work / "velest_1d"
    rd.mkdir(parents=True, exist_ok=True)
    for f in rd.glob("velout.*"):
        f.unlink()
    (rd / "velest.out").unlink(missing_ok=True)
    write_cnv(rd / "events.cnv", cnv_ev, cnv_pk)
    write_stations(rd / "stations.sta", sta_tab)
    write_model(rd / "model.mod", vm)
    write_control(rd / "velest.cmn", ctrl)
    try:
        res = run_tool("velest", rd, timeout=timeout, check=False)
    except ToolError as exc:
        raise ValidationError(f"VELEST failed: {exc}") from exc
    if res.timed_out or not (rd / "velest.out").is_file():
        raise ValidationError(f"VELEST single-event run failed (see {rd}): {res.detected_errors}")
    parsed = read_run_dir(rd)
    fh = parsed.final_hypocentres
    if len(fh) != len(ev):
        raise ValidationError(
            f"VELEST single-event run returned {len(fh)} hypocentres for {len(ev)} events "
            f"(see {rd / 'velest.out'}); events may have been rejected (airquakes, too few picks)"
        )
    n_by = pk.groupby("event_id").size()
    rms = fh["rms"].to_numpy(float)
    acc = parsed.accepted
    if len(acc) and acc["run"].nunique() == len(ev):
        # the iteration log carries the RMS with 6 decimals (the hypocentre table only 2)
        last = acc.sort_values(["run", "iteration"]).groupby("run").tail(1)
        hi = last.sort_values("run")["rms"].to_numpy(float)
        if np.isfinite(hi).all() and np.all(np.abs(hi - rms) < 0.02):
            rms = hi
    out = pd.DataFrame(
        {
            "event_id": ev["event_id"].to_numpy(),
            "time": pd.to_datetime(fh["time"], utc=True).to_numpy(),
            "lat": fh["lat"].to_numpy(float),
            "lon": fh["lon"].to_numpy(float),
            "depth_km": fh["depth_km"].to_numpy(float),
            "rms": rms,
            "n_picks": ev["event_id"].map(n_by).fillna(0).astype(int).to_numpy(),
        }
    )
    out.to_csv(work / "velest_1d_hypocentres.csv", index=False)
    return out


# ---------------------------------------------------------------------------------------
# 3-D: FMTOMO linear relocation with fixed velocities
# ---------------------------------------------------------------------------------------
def _load_model_3d(
    model_3d: Any, model_inputs_dir: str | Path | None
) -> tuple[Any, Any, Any, Path | None]:
    """``(VelocityGrids, InterfaceGrids, PropGrid, dir)`` from a run dir/vgrids file/object."""
    from osv.io.fmtomo import read_interfaces, read_propgrid, read_vgrids

    base: Path | None = None
    grids: Any
    if isinstance(model_3d, (str, Path)):
        p = Path(model_3d)
        if p.is_dir():
            base = p
            p = p / "vgrids.in"
        else:
            base = p.parent
        if not p.is_file():
            raise ValidationError(f"3-D velocity model {p} not found")
        grids = read_vgrids(p)
    else:
        grids = model_3d
    d = Path(model_inputs_dir) if model_inputs_dir is not None else base
    if d is None:
        raise ValidationError(
            "model_inputs_dir (directory with interfaces.in and propgrid.in) is required when "
            "model_3d is a VelocityGrids object"
        )
    for n in ("interfaces.in", "propgrid.in"):
        if not (d / n).is_file():
            raise ValidationError(
                f"{d / n} not found: pass model_inputs_dir with the Stage 3/4 FMTOMO files"
            )
    return grids, read_interfaces(d / "interfaces.in"), read_propgrid(d / "propgrid.in"), d


def _surface_depth_km(itf: Any, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Depth (km, positive down) of the top interface at ``lat/lon`` (bilinear, edge-clamped)."""
    from scipy.interpolate import RegularGridInterpolator

    d = itf.data
    a = d.attrs
    nlat, nlon = d.sizes["lat"], d.sizes["lon"]
    lat_ax = np.rad2deg(a["fmtomo_lat0_rad"] + a["fmtomo_dlat_rad"] * np.arange(nlat))
    lon_ax = np.rad2deg(a["fmtomo_lon0_rad"] + a["fmtomo_dlon_rad"] * np.arange(nlon))
    f = RegularGridInterpolator(
        (lat_ax, lon_ax),
        d["interface_depth"].isel(interface=0).to_numpy(),
        bounds_error=False,
        fill_value=None,
    )
    pts = np.column_stack(
        [np.clip(lat, lat_ax[0], lat_ax[-1]), np.clip(lon, lon_ax[0], lon_ax[-1])]
    )
    return np.asarray(f(pts))


def _clamp_sources(
    sources: list[Any], itf: Any, pg: Any, margin_km: float
) -> tuple[list[Any], int]:
    """Keep sources inside the propagation grid and below the surface interface.

    ``scorrect`` only checks the corner node of the surface interface, so a source moved up by
    a Gauss-Newton step can end above the local topography, where ``fm3d`` stops with
    ``source above surface``.  Returns the corrected list and the number of sources changed.
    """
    from dataclasses import replace

    lat = np.array([s.lat for s in sources])
    lon = np.array([s.lon for s in sources])
    z = np.array([s.depth_km for s in sources])
    top = np.maximum(_surface_depth_km(itf, lat, lon), float(pg.depth_km[0])) + margin_km
    bot = float(pg.depth_km[-1]) - margin_km
    lat_lo, lat_hi = pg.lat0_deg + margin_km / KM_PER_DEG, pg.lat0_deg + (pg.nlat - 1) * pg.dlat_deg
    lon_lo, lon_hi = pg.lon0_deg, pg.lon0_deg + (pg.nlon - 1) * pg.dlon_deg
    dlat = margin_km / KM_PER_DEG
    z2 = np.clip(z, top, bot)
    lat2 = np.clip(lat, lat_lo, lat_hi - dlat)
    lon2 = np.clip(lon, lon_lo + dlat, lon_hi - dlat)
    changed = (z2 != z) | (lat2 != lat) | (lon2 != lon)
    out = [
        replace(s, depth_km=float(z2[i]), lat=float(lat2[i]), lon=float(lon2[i]))
        if changed[i]
        else s
        for i, s in enumerate(sources)
    ]
    return out, int(changed.sum())


def _relocation_loop(
    run: Path, itf: Any, pg: Any, n_iter: int, n_jobs: int, margin_km: float = 0.05
) -> tuple[pd.DataFrame, list[str]]:
    """The ``tomo3d`` loop with source parameters only and a source clamp after each step.

    Same programs in the same order as :func:`osv.fmtomo.invert.run_inversion` (``frechgen``,
    ``fm3d``, ``residuals``; then per iteration ``invert3d``, ``scorrect``, ``fm3d``,
    ``residuals``) with :func:`_clamp_sources` between ``scorrect`` and ``fm3d``.  Returns the
    iteration table (``iter``, ``rms_ms``: plain RMS of ``residuals``, which still contains the
    origin-time error, ``n_clamped``) and flags.
    """
    import shutil

    from osv.fmtomo.forward import Fm3dError, run_fm3d_parallel
    from osv.io.fmtomo import read_receivers, read_sources, write_sources
    from osv.tools import ToolError, run_tool

    flags: list[str] = []
    rows: list[dict[str, Any]] = []

    def tool(name: str) -> Path:
        try:
            return run_tool(name, run).stdout_log
        except ToolError as exc:
            raise ValidationError(f"{name} failed in {run}: {exc}") from exc

    def forward() -> None:
        try:
            res = run_fm3d_parallel(
                run,
                read_sources(run / "sources.in"),
                read_receivers(run / "receivers.in"),
                n_jobs,
                None,
                run / "parallel",
            )
        except (Fm3dError, ToolError) as exc:
            raise ValidationError(f"fm3d failed in {run}: {exc}") from exc
        shutil.copyfile(res.arrivals_path, run / "arrivals.dat")
        if res.frechet_path is not None:
            shutil.copyfile(res.frechet_path, run / "frechet.dat")
        shutil.copyfile(run / "arrivals.dat", run / "mtimes.dat")

    def residual_ms() -> float:
        txt = tool("residuals").read_text(errors="replace").split()
        try:
            return float(txt[0])
        except (IndexError, ValueError) as exc:
            raise ValidationError(f"cannot parse the output of residuals in {run}") from exc

    for src, dst in (
        ("vgridsref.in", "vgrids.in"),
        ("interfacesref.in", "interfaces.in"),
        ("sourcesref.in", "sources.in"),
    ):
        shutil.copyfile(run / src, run / dst)
    tool("frechgen")
    (run / "inviter.in").write_text("1\n")
    forward()
    shutil.copyfile(run / "arrivals.dat", run / "rtimes.dat")
    rows.append({"iter": 0, "rms_ms": residual_ms(), "n_clamped": 0})
    for k in range(1, n_iter + 1):
        log_path = tool("invert3d")
        if "successfully completed" not in log_path.read_text(errors="replace"):
            raise ValidationError(f"invert3d did not complete in iteration {k}; see {log_path}")
        tool("scorrect")
        srcs, n_cl = _clamp_sources(read_sources(run / "sources.in"), itf, pg, margin_km)
        if n_cl:
            write_sources(run / "sources.in", srcs)
            flags.append(f"iteration {k}: {n_cl} source(s) kept inside the model (surface/edges)")
        forward()
        rows.append({"iter": k, "rms_ms": residual_ms(), "n_clamped": n_cl})
        (run / "inviter.in").write_text(f"{k + 1}\n")
    return pd.DataFrame(rows), flags


def relocate_3d(
    events: pd.DataFrame,
    picks: pd.DataFrame,
    model_3d: Any,
    station_delays: pd.DataFrame | None,
    grid: Any,
    cfg: Any,
    workdir: str | Path,
    n_jobs: int = 1,
    *,
    stations: pd.DataFrame | None = None,
    model_inputs_dir: str | Path | None = None,
    options: RelocateOptions | None = None,
    subset: str = "validation",
    station_terms_3d: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Relocate events with FMTOMO, velocities fixed (linear source-only inversion).

    ``station_terms_3d`` (``phase``, ``station``, ``term_s``: the iterated terms of the
    inversion) are subtracted from the observed P times, as in the inversion.

    Returns ``(table, notes)``; ``table`` has ``event_id, lat, lon, depth_km, rms, n_picks`` and
    the per-pick residuals are kept in ``table.attrs['residuals']`` (columns ``event_id,
    station, phase, residual_s, weight``).
    """
    from osv.fmtomo.controls import ModelInfo, build_controls
    from osv.fmtomo.rundir import assemble_run_dir
    from osv.io.fmtomo import read_arrivals, read_sources
    from osv.model3d.sources import build_fmtomo_data

    opt = options or RelocateOptions()
    phases = _phases(cfg, opt)
    if len(phases) != 1:
        # one velocity type per tomo3d run; use P (the Stage 3/4 default) and say so
        log.warning("3-D validation relocation uses P only (requested %s)", phases)
        phases = ("P",)
    work = Path(workdir)
    run = work / "fmtomo_3d"
    grids, itf, pg, mdir = _load_model_3d(model_3d, model_inputs_dir)
    ev = _validation_events(events, subset)
    pk = _usable_picks(picks, ev["event_id"], phases)
    st = stations if stations is not None else _stations_from_picks(pk)
    if "alias" not in st:
        st = st.assign(alias=st["sta"])
    if "sensor_elevation_m" not in st:
        st = st.assign(sensor_elevation_m=st["elevation_m"])
    dl = _delay_key(station_delays)
    data = build_fmtomo_data(
        pk,
        ev,
        st,
        cfg,
        grid,
        hypocentres="catalog",
        subset=subset,
        phases=phases,
        station_delays=dl,
        layout="separate",
    )
    notes = list(data.findings)
    # fm3d writes arrivals.dat source by source (then receiver, then ray) and ``residuals`` /
    # ``invert3d`` pair otimes.dat and mtimes.dat row by row, so otimes.dat must have that
    # order.  build_fmtomo_data orders rows receiver-first (its docstring says arrivals.dat
    # does; fortran/fmtomo/example1/otimes.dat and merged fm3d output show it does not).
    order = np.lexsort(
        (
            data.observed["ray"].to_numpy(),
            data.observed["receiver"].to_numpy(),
            data.observed["source"].to_numpy(),
        )
    )
    data.observed = data.observed.iloc[order].reset_index(drop=True)
    data.mapping = data.mapping.iloc[order].reset_index(drop=True)
    if station_terms_3d is not None and len(station_terms_3d):
        tp = station_terms_3d[station_terms_3d["phase"] == "P"]
        term = (
            data.mapping["station"]
            .astype(str)
            .map(tp.set_index(tp["station"].astype(str))["term_s"])
        )
        t = data.observed["time"].to_numpy(float)
        ok = t > 0
        data.observed.loc[ok, "time"] = t[ok] - term.fillna(0.0).to_numpy(float)[ok]
        notes.append(
            f"3-D station terms of the inversion subtracted ({int(term.notna().sum())} picks)"
        )
    fm = cfg.fmtomo.model_copy(
        update=dict(
            relocate="linear",
            invert_velocity=False,
            invert_interfaces=[],
            smoothing=0.0,
            iterations=int(opt.n_iterations),
            save_rays=False,
            phases=["P"],
        )
    )
    controls = build_controls(fm, ModelInfo.from_model(grids), ["P"])
    # invert3d uses these as km for all three coordinates (invert3d.f90:575-596 compares them
    # with displacements in km), although osv.io.fmtomo documents lat/lon in degrees
    sig = float(opt.source_sigma_km)
    unc = (sig, sig, sig)
    sources_unc = [
        type(s)(s.depth_km, s.lat, s.lon, list(s.paths), uncertainty=unc) for s in data.sources
    ]
    if run.exists():
        import shutil

        shutil.rmtree(run)
    assemble_run_dir(
        run, grids, itf, pg, sources_unc, data.receivers, data.observed, controls, phase="P"
    )
    hist, flags = _relocation_loop(
        run, itf, pg, int(opt.n_iterations), max(1, min(int(n_jobs), MAX_FM3D_JOBS))
    )
    notes.extend(flags)
    fsrc = read_sources(run / "sources.in")
    arr = read_arrivals(run / "arrivals.dat")
    obs = data.observed.reset_index(drop=True)
    mp = data.mapping.reset_index(drop=True)
    if len(arr) != len(obs):
        raise ValidationError(f"arrivals.dat has {len(arr)} rows, otimes {len(obs)}")
    calc = arr["time"].to_numpy(float)
    ok = arr["valid"].to_numpy(bool) & (calc > 0) & mp["valid"].to_numpy(bool)
    r = obs["time"].to_numpy(float) - calc
    pkw = pk.drop_duplicates("pick_id").set_index("pick_id") if "pick_id" in pk else None
    wcls = (
        mp["pick_id"].map(pkw["velest_weight"]).fillna(0).to_numpy(float)
        if pkw is not None
        else np.zeros(len(mp))
    )
    swt = SWTFAC
    wgt = np.where(mp["phase"].to_numpy() == "S", swt, 1.0) / (4.0**wcls)
    res = pd.DataFrame(
        {
            "event_id": mp["event_id"].to_numpy(),
            "source_idx": mp["source_idx"].to_numpy(),
            "station": mp["station"].to_numpy(),
            "phase": mp["phase"].to_numpy(),
            "residual_s": r,
            "weight": np.where(ok, wgt, 0.0),
        }
    )
    rows = []
    for sidx, g in res.groupby("source_idx"):
        w = g["weight"].to_numpy()
        rr = g["residual_s"].to_numpy()
        m = w > 0
        # origin time: weighted least squares (FMTOMO's relocation does not carry it forward)
        dt = float(np.sum(w[m] ** 2 * rr[m]) / np.sum(w[m] ** 2)) if m.any() else 0.0
        res.loc[g.index, "residual_s"] = rr - dt
        s = fsrc[int(sidx) - 1]
        rows.append(
            {
                "event_id": g["event_id"].iloc[0],
                "lat": s.lat,
                "lon": s.lon,
                "depth_km": s.depth_km,
                "rms": event_rms(rr - dt, w),
                "n_picks": int(m.sum()),
                "origin_shift_s": dt,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["residuals"] = res
    out.attrs["iterations"] = hist
    if mdir is not None:
        notes.append(f"3-D model: {mdir}")
    out.to_csv(work / "fmtomo_3d_hypocentres.csv", index=False)
    return out, notes


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def relocate_validation(
    events: pd.DataFrame,
    picks: pd.DataFrame,
    model_1d: Any,
    station_delays: pd.DataFrame | None,
    model_3d: Any,
    grid: Any,
    cfg: Any,
    workdir: str | Path,
    n_jobs: int = 1,
    *,
    stations: pd.DataFrame | None = None,
    model_inputs_dir: str | Path | None = None,
    options: RelocateOptions | None = None,
    subset: str = "validation",
    station_terms_3d: pd.DataFrame | None = None,
) -> ValidationResult:
    """Relocate the held-out events with the 1-D and the 3-D model and compare.

    Parameters
    ----------
    events, picks
        Stage 1 ``events.parquet`` (``split``, ``depth_km_bsl``) and ``picks.parquet``
        (``velest_weight``, ``sigma_s``, ``alias``, station position columns).
    model_1d
        Final 1-D model: ``.mod`` path (``03_velest_analyse/final_model.mod``),
        ``VelestModel`` or ``TrialModel``.
    station_delays
        Stage 2 ``station_delays.csv`` table (``alias``/``sta``, ``ptcor``).
    model_3d
        FMTOMO run directory or ``vgrids.in`` path (``interfaces.in``/``propgrid.in`` are taken
        from the same directory unless ``model_inputs_dir`` is given) or a ``VelocityGrids``.
    grid
        :class:`~osv.model3d.grid.ModelGrid` (checks events/receivers against the propagation
        grid) or ``None``.
    cfg
        Full config (``fmtomo``, ``model3d.apply_velest_station_delays``, ``velest``,
        ``coordinates``, ``data_selection``).
    workdir
        Output directory (``velest_1d/``, ``fmtomo_3d/``, CSV tables).
    n_jobs
        Parallel ``fm3d`` processes (capped at 8).
    stations
        Stage 1 stations table (default: derived from the picks).
    subset
        ``events.split`` value to relocate (default ``"validation"``).
    station_terms_3d
        Iterated station terms of the 3-D inversion, applied to the 3-D relocation only.

    Returns
    -------
    ValidationResult
        Per-event table, paired statistics and QC findings.
    """
    opt = options or RelocateOptions()
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    ev = _validation_events(events, subset)
    phases3 = ("P",)
    phases = _phases(cfg, opt)
    # the 1-D run must use the same picks as the 3-D run
    opt1 = RelocateOptions(**{**opt.__dict__, "phases": phases3})
    h1 = relocate_1d(
        ev, picks, model_1d, station_delays, cfg, work, stations=stations, options=opt1
    )
    h3, notes = relocate_3d(
        ev,
        picks,
        model_3d,
        station_delays,
        grid,
        cfg,
        work,
        n_jobs,
        stations=stations,
        model_inputs_dir=model_inputs_dir,
        options=opt1,
        subset=subset,
        station_terms_3d=station_terms_3d,
    )
    if phases != phases3:
        notes.append(f"phases {phases} requested; both relocations use {phases3} (P) for parity")
    res = _assemble(ev, h1, h3, opt)
    res.workdir = work
    res.phases = phases3
    res.delays_applied_3d = bool(
        getattr(_cfg_section(cfg, "model3d"), "apply_velest_station_delays", False)
    )
    res.notes = notes
    if not res.delays_applied_3d and station_terms_3d is None:
        res.findings.append(
            QCFinding(
                "validation_3d_without_delays",
                "info",
                "3-D times were not corrected with the VELEST station delays "
                "(model3d.apply_velest_station_delays is off) while the 1-D benchmark uses "
                "them: the comparison favours the 1-D model.",
            )
        )
    res.write(work)
    return res


def _assemble(
    ev: pd.DataFrame, h1: pd.DataFrame, h3: pd.DataFrame, opt: RelocateOptions
) -> ValidationResult:
    base = ev[["event_id", "lat", "lon", "depth_km_bsl"]].rename(
        columns={"lat": "lat0", "lon": "lon0", "depth_km_bsl": "depth0"}
    )
    a = h1.rename(
        columns={
            "lat": "lat_1d",
            "lon": "lon_1d",
            "depth_km": "depth_1d",
            "rms": "rms_1d",
            "n_picks": "n_picks_1d",
        }
    )[["event_id", "lat_1d", "lon_1d", "depth_1d", "rms_1d", "n_picks_1d"]]
    b = h3.rename(
        columns={
            "lat": "lat_3d",
            "lon": "lon_3d",
            "depth_km": "depth_3d",
            "rms": "rms_3d",
            "n_picks": "n_picks_3d",
        }
    )[["event_id", "lat_3d", "lon_3d", "depth_3d", "rms_3d", "n_picks_3d"]]
    table = base.merge(a, on="event_id", how="inner").merge(b, on="event_id", how="inner")
    table["n_picks"] = table[["n_picks_1d", "n_picks_3d"]].min(axis=1)
    if table.empty:
        raise ValidationError("no event was located by both the 1-D and the 3-D relocation")
    stats, findings = compare_relocations(table, opt)
    dropped = len(ev) - len(table)
    if dropped:
        findings.append(
            QCFinding(
                "validation_events_dropped",
                "warn",
                f"{dropped} validation event(s) could not be located by both methods and are "
                "left out of the comparison.",
                dropped,
            )
        )
    res = ValidationResult(table, stats, findings)
    res.events.attrs["residuals_3d"] = h3.attrs.get("residuals")
    return res
