"""Per-pick residual and hypocentre-relocation tables of an FMTOMO inversion (task 6.3).

Run directory layout (see :mod:`osv.fmtomo.invert`)::

    run_dir/otimes.dat, sourcesref.in, receivers.in, rtimes.dat, mtimes.dat, sources.in
    run_dir/iterations/NN/{summary.json, vgrids.in, sources.in, stimes.dat, mtimes.dat}

``NN = 00`` is the starting model.  Calculated times of an iteration are read from
``iterations/NN/mtimes.dat`` (or ``arrivals.dat``); iteration 0 falls back to ``rtimes.dat`` and
the last iteration to the ``mtimes.dat`` of the run directory.  Positions of the hypocentres
come from ``iterations/NN/sources.in`` (``receivers.in`` for the nonlinear geometry, where the
events are the FMTOMO receivers), the initial ones from ``sourcesref.in`` (``receiversref.in``)
when ``iterations/00`` has none.

Residual convention: ``residual = observed - calculated`` (positive when the model is too fast),
the same as :mod:`osv.model3d.forward_check`; the observed times are the ``time`` of the mapping
(``otimes.dat``).  Only picks with a valid observed and a valid calculated time are kept.
Distances are hypocentral, azimuths go from the event to the station (degrees from north).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.coords import azimuth_deg, hypocentral_distance_km
from osv.io.fmtomo import read_arrivals, read_receivers, read_sources, read_stimes
from osv.model3d.forward_check import residual_stats
from osv.model3d.grid import KM_PER_DEG

log = logging.getLogger(__name__)

__all__ = [
    "available_iterations",
    "relocation_table",
    "residual_stats_by",
    "residual_table",
    "residual_tables",
    "variance_reduction",
]

_KEEP = ["source_idx", "receiver_idx", "ray_idx", "event_id", "station", "phase", "pick_id"]


def available_iterations(run_dir: str | Path) -> list[int]:
    """Iteration numbers with a ``iterations/NN`` directory (sorted)."""
    d = Path(run_dir) / "iterations"
    if not d.is_dir():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit())


def _resolve(run_dir: Path, iteration: int | str) -> int:
    its = available_iterations(run_dir)
    if iteration in ("final", "last", -1):
        if not its:
            raise FileNotFoundError(f"{run_dir / 'iterations'} has no iteration directories")
        return its[-1]
    if iteration in ("initial", "start"):
        return 0
    return int(iteration)


def _calc_path(run: Path, it: int) -> Path:
    its = available_iterations(run)
    d = run / "iterations" / f"{it:02d}"
    cands = [d / "mtimes.dat", d / "arrivals.dat"]
    if it == 0:
        cands.append(run / "rtimes.dat")
    if its and it == its[-1]:
        cands += [run / "mtimes.dat", run / "arrivals.dat"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no calculated travel times for iteration {it}: looked for "
        f"{', '.join(str(c) for c in cands)} (inversions run before mtimes.dat was kept per "
        "iteration only have the initial and the final times)"
    )


def _event_positions(run: Path, it: int, nonlinear: bool, initial: bool = False) -> pd.DataFrame:
    """Hypocentres indexed 1..n with ``lat``, ``lon``, ``depth_km`` for an iteration."""
    name = "receivers.in" if nonlinear else "sources.in"
    ref = "receiversref.in" if nonlinear else "sourcesref.in"
    d = run / "iterations" / f"{it:02d}"
    its = available_iterations(run)
    cands = [d / name]
    if initial or it == 0:
        cands.append(run / ref)
    if its and it == its[-1] and not initial:
        cands.append(run / name)
    for c in cands:
        if c.exists():
            return _read_positions(c, nonlinear)
    raise FileNotFoundError(f"no hypocentre file for iteration {it}: looked for {cands}")


def _read_positions(path: Path, nonlinear: bool) -> pd.DataFrame:
    if nonlinear:
        pos = read_receivers(path).positions
        return pos[["lat", "lon", "depth_km"]].copy()
    srcs = read_sources(path)
    return pd.DataFrame(
        {
            "lat": [s.lat for s in srcs],
            "lon": [s.lon for s in srcs],
            "depth_km": [s.depth_km for s in srcs],
        },
        index=pd.RangeIndex(1, len(srcs) + 1),
    )


def _station_positions(run: Path, nonlinear: bool) -> pd.DataFrame:
    if not nonlinear:
        return read_receivers(run / "receivers.in").positions[["lat", "lon", "depth_km"]]
    f = run / "sourcesref.in" if (run / "sourcesref.in").exists() else run / "sources.in"
    return _read_positions(f, False)


def residual_table(
    run_dir: str | Path,
    mapping: pd.DataFrame,
    iteration: int | str = "final",
    *,
    nonlinear: bool = False,
) -> pd.DataFrame:
    """Per-pick residuals (observed - calculated) of one iteration.

    Parameters
    ----------
    run_dir
        Inversion run directory.
    mapping
        Pick mapping (``otimes_mapping.csv`` of :func:`osv.model3d.sources.write_fmtomo_data`):
        ``source_idx``, ``receiver_idx``, ``ray_idx``, ``event_id``, ``station``, ``phase``,
        ``pick_id``, ``time`` in the row order of ``otimes.dat``/``arrivals.dat``.
    iteration
        Iteration number, ``"initial"`` (0) or ``"final"`` (the last ``iterations/NN``).
    nonlinear
        Nonlinear geometry (events are the FMTOMO receivers, stations the sources); the
        mapping's ``source_idx``/``receiver_idx`` then still index the *files*.

    Returns
    -------
    pandas.DataFrame
        Mapping columns plus ``iteration``, ``t_obs``, ``t_calc``, ``residual``, ``dist_km``,
        ``azimuth_deg`` (event to station), ``ev_lat``, ``ev_lon``, ``ev_depth_km``,
        ``sta_lat``, ``sta_lon``, ``sta_depth_km``.  Invalid picks are dropped.
    """
    run = Path(run_dir)
    it = _resolve(run, iteration)
    arr = read_arrivals(_calc_path(run, it))
    if len(arr) != len(mapping):
        raise ValueError(
            f"arrivals ({len(arr)} rows) and mapping ({len(mapping)} rows) differ: the mapping "
            "must have one row per row of otimes.dat"
        )
    m = mapping.reset_index(drop=True)
    same = (
        (arr["receiver"].to_numpy() == m["receiver_idx"].to_numpy())
        & (arr["source"].to_numpy() == m["source_idx"].to_numpy())
        & (arr["ray"].to_numpy() == m["ray_idx"].to_numpy())
    )
    if not same.all():
        raise ValueError(
            f"arrivals and mapping disagree on receiver/source/ray in {int((~same).sum())} rows"
        )
    keep = [c for c in _KEEP if c in m]
    df = m[keep].copy()
    df["iteration"] = it
    df["t_obs"] = m["time"].to_numpy(float)
    df["t_calc"] = arr["time"].to_numpy(float)
    ok = arr["valid"].to_numpy() & (m["time"].to_numpy(float) >= -50.0)
    if "valid" in m:
        ok &= m["valid"].to_numpy(bool)
    df = df[ok].copy()
    df["residual"] = df["t_obs"] - df["t_calc"]
    ev_col, st_col = ("receiver_idx", "source_idx") if nonlinear else ("source_idx", "receiver_idx")
    ev = _event_positions(run, it, nonlinear).loc[df[ev_col]]
    st = _station_positions(run, nonlinear).loc[df[st_col]]
    for k, v in (("lat", "lat"), ("lon", "lon"), ("depth_km", "depth_km")):
        df[f"ev_{k}"] = ev[v].to_numpy(float)
        df[f"sta_{k}"] = st[v].to_numpy(float)
    df["dist_km"] = hypocentral_distance_km(
        df["ev_lat"],
        df["ev_lon"],
        df["ev_depth_km"],
        df["sta_lat"],
        df["sta_lon"],
        df["sta_depth_km"],
    )
    df["azimuth_deg"] = azimuth_deg(df["ev_lat"], df["ev_lon"], df["sta_lat"], df["sta_lon"])
    return df.reset_index(drop=True)


def residual_tables(
    run_dir: str | Path,
    mapping: pd.DataFrame,
    iterations: Iterable[int | str] | None = None,
    *,
    nonlinear: bool = False,
) -> dict[int, pd.DataFrame]:
    """:func:`residual_table` for several iterations (default: all that have times).

    Iterations without calculated times are skipped (logged) when ``iterations`` is None.
    """
    run = Path(run_dir)
    explicit = iterations is not None
    its = [_resolve(run, i) for i in iterations] if explicit else available_iterations(run)
    out: dict[int, pd.DataFrame] = {}
    for it in its:
        try:
            out[it] = residual_table(run, mapping, it, nonlinear=nonlinear)
        except FileNotFoundError as exc:
            if explicit:
                raise
            log.info("iteration %d skipped: %s", it, exc)
    return out


def residual_stats_by(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Residual statistics (``n``, mean, median, rms, mad in seconds) per group.

    ``by`` is a column or list, e.g. ``"station"``, ``"event_id"``, ``"phase"`` or
    ``["station", "phase"]``.
    """
    keys = [by] if isinstance(by, str) else list(by)
    rows = []
    for k, g in df.groupby(keys, sort=True):
        st = residual_stats(g["residual"].to_numpy())
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k, strict=True)), **st})
    return pd.DataFrame(rows)


def _summary_series(run: Path) -> pd.DataFrame:
    rows = []
    for it in available_iterations(run):
        p = run / "iterations" / f"{it:02d}" / "summary.json"
        try:
            s = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        rows.append(
            {
                "iteration": it,
                "fmtomo_rms_ms": s.get("rms_ms"),
                "fmtomo_variance_s2": s.get("variance_s2"),
                "fmtomo_chi2": s.get("chi2"),
            }
        )
    return pd.DataFrame(rows)


def variance_reduction(
    run_dir: str | Path,
    mapping: pd.DataFrame,
    tables: dict[int, pd.DataFrame] | None = None,
    *,
    nonlinear: bool = False,
) -> pd.DataFrame:
    """Misfit per iteration and variance reduction against iteration 0.

    Columns: ``iteration``, ``n``, ``rms_s``, ``variance_s2`` (mean squared residual of the
    picks), ``vr_pct`` (``100 * (1 - variance / variance_0)``), and the numbers the FMTOMO
    ``residuals`` program wrote (``fmtomo_rms_ms``, ``fmtomo_variance_s2``, ``fmtomo_chi2``,
    ``fmtomo_vr_pct``) which exist for every iteration even when no per-pick times were kept.
    """
    run = Path(run_dir)
    tables = tables if tables is not None else residual_tables(run, mapping, nonlinear=nonlinear)
    rows = []
    for it, df in sorted(tables.items()):
        r = df["residual"].to_numpy(float)
        r = r[np.isfinite(r)]
        rows.append(
            {
                "iteration": it,
                "n": len(r),
                "rms_s": float(np.sqrt(np.mean(r**2))) if len(r) else np.nan,
                "variance_s2": float(np.mean(r**2)) if len(r) else np.nan,
            }
        )
    out = pd.DataFrame(rows, columns=["iteration", "n", "rms_s", "variance_s2"])
    summ = _summary_series(run)
    if len(summ):
        out = out.merge(summ, on="iteration", how="outer").sort_values("iteration")
    else:
        for c in ("fmtomo_rms_ms", "fmtomo_variance_s2", "fmtomo_chi2"):
            out[c] = np.nan
    out = out.reset_index(drop=True)

    def vr(col: str) -> pd.Series:
        v = out[col].astype(float)
        base = v.iloc[0] if len(v) else np.nan
        return 100.0 * (1.0 - v / base) if base and np.isfinite(base) else v * np.nan

    out["vr_pct"] = vr("variance_s2")
    out["fmtomo_vr_pct"] = vr("fmtomo_variance_s2")
    return out


def relocation_table(
    run_dir: str | Path,
    mapping: pd.DataFrame,
    iteration: int | str = "final",
    *,
    nonlinear: bool = False,
) -> pd.DataFrame:
    """Hypocentre of one iteration against the initial one.

    Columns: ``iteration``, ``event_idx`` (FMTOMO source id; receiver id when nonlinear),
    ``event_id``, ``n_picks``, initial ``lat0``/``lon0``/``depth0_km`` and current ``lat``/
    ``lon``/``depth_km``, shifts ``dx_km`` (east), ``dy_km`` (north), ``dz_km`` (down,
    positive = deeper), ``dh_km`` (horizontal), ``d3_km`` (3-D) and ``dt_s`` (origin-time
    correction ``stimes.dat`` of the iteration, NaN if the file is absent).
    """
    run = Path(run_dir)
    it = _resolve(run, iteration)
    idx_col = "receiver_idx" if nonlinear else "source_idx"
    ev = (
        mapping.groupby(idx_col)
        .agg(event_id=("event_id", "first"), n_picks=("pick_id", "size"))
        .reset_index()
        .rename(columns={idx_col: "event_idx"})
    )
    p0 = _event_positions(run, 0, nonlinear, initial=True)
    p1 = _event_positions(run, it, nonlinear)
    ev = ev[ev["event_idx"].isin(p0.index) & ev["event_idx"].isin(p1.index)].copy()
    a, b = p0.loc[ev["event_idx"]], p1.loc[ev["event_idx"]]
    out = pd.DataFrame(
        {
            "iteration": it,
            "event_idx": ev["event_idx"].to_numpy(),
            "event_id": ev["event_id"].to_numpy(),
            "n_picks": ev["n_picks"].to_numpy(),
            "lat0": a["lat"].to_numpy(),
            "lon0": a["lon"].to_numpy(),
            "depth0_km": a["depth_km"].to_numpy(),
            "lat": b["lat"].to_numpy(),
            "lon": b["lon"].to_numpy(),
            "depth_km": b["depth_km"].to_numpy(),
        }
    )
    out["dy_km"] = (out["lat"] - out["lat0"]) * KM_PER_DEG
    out["dx_km"] = (out["lon"] - out["lon0"]) * KM_PER_DEG * np.cos(np.deg2rad(out["lat0"]))
    out["dz_km"] = out["depth_km"] - out["depth0_km"]
    out["dh_km"] = np.hypot(out["dx_km"], out["dy_km"])
    out["d3_km"] = np.hypot(out["dh_km"], out["dz_km"])
    out["dt_s"] = _origin_time_shift(run, it, out["event_idx"].to_numpy())
    return out


def _origin_time_shift(run: Path, it: int, idx: np.ndarray) -> np.ndarray:
    p = run / "iterations" / f"{it:02d}" / "stimes.dat"
    if not p.exists():
        return np.full(len(idx), np.nan)
    st: Any = read_stimes(p)
    ok = (idx >= 1) & (idx <= len(st))
    out = np.full(len(idx), np.nan)
    out[ok] = st[idx[ok] - 1]
    return out
