"""Wadati-diagram Vp/Vs and travel-time outlier detection (Stage 1, task 2.5).

Single module covering both analyses; everything uses numpy/scipy only.

Wadati analysis
---------------
For every S pick with a P pick at the same event and station (``ts_minus_tp_s`` finite):

* **Pooled, through the origin.**  ``Ts - Tp = (Vp/Vs - 1) * Tp`` with ``Tp`` the P travel
  time.  The slope is estimated by Huber iteratively-reweighted least squares (starting from
  the median of ``(Ts-Tp)/Tp``).  The pick residual is ``(Ts-Tp) - k*Tp``; the robust z-score is
  ``(r - median(r)) / (1.4826 * MAD(r))`` and S picks with ``|z| > wadati_outlier_sigma`` are
  Wadati outliers.  The uncertainty is a seeded *event-cluster bootstrap* (events are
  resampled, so correlated picks of one event do not make the interval too narrow); the
  interval is the 2.5/97.5 percentiles.
* **Per event, free intercept** (origin time unknown): ``Ts`` vs ``Tp`` with a Theil-Sen
  line, slope = Vp/Vs, for events with at least ``min_pairs`` pairs.  Events whose Vp/Vs
  differs from the median by more than ``sigma`` robust standard deviations (with a floor of
  ``min_event_sigma`` on the spread) are reported as inconsistent.

Travel-time analysis
--------------------
Per phase, travel time vs hypocentral distance is fitted with a Huber-IRLS polynomial
(default degree 2).  A low-order polynomial follows the mild concavity of ``t(d)`` in a 1-D
medium whose velocity increases with depth, while being insensitive to the outliers.  The
residual scale is the MAD within distance bins (equal-count, at least ``min_bin`` picks per
bin, floored at half the global MAD), so scatter that grows with distance does not make
far picks outliers.  Picks with non-positive travel time are flagged with reason ``tt_invalid``.

Combined output
---------------
:func:`analyse_outliers` returns an :class:`OutlierAnalysis`; :func:`outlier_flags` returns just
the boolean Series (aligned with the pick index; union of both criteria) for
``select_data(outlier_flags=...)``.  ``OutlierAnalysis.pick_table`` has the per-pick residuals,
z-scores, flags and a ``reason`` column (``""``, ``"tt"``, ``"wadati"``, ``"tt+wadati"`` or
``"tt_invalid"``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from osv.report.checks import QCFinding

logger = logging.getLogger(__name__)

__all__ = [
    "OutlierAnalysis",
    "TTFit",
    "WadatiFit",
    "analyse_outliers",
    "fit_travel_time",
    "fit_wadati",
    "outlier_flags",
    "robust_sigma",
]

_MAD_TO_SIGMA = 1.4826


def robust_sigma(x: np.ndarray) -> float:
    """MAD-based standard deviation (``1.4826 * median(|x - median(x)|)``); NaN if empty."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(_MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def _huber_irls(
    design: np.ndarray, y: np.ndarray, beta0: np.ndarray | None = None, n_iter: int = 50
) -> np.ndarray:
    """Huber (c=1.345) IRLS linear regression; scale is the MAD of the residuals."""
    if beta0 is None:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    else:
        beta = np.asarray(beta0, dtype=float)
    for _ in range(n_iter):
        r = y - design @ beta
        s = robust_sigma(r)
        if not np.isfinite(s) or s <= 0:
            s = max(float(np.std(r)), 1e-12)
        u = np.abs(r) / (1.345 * s)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        sw = np.sqrt(w)
        new, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        if np.allclose(new, beta, rtol=1e-9, atol=1e-12):
            beta = new
            break
        beta = new
    return beta


def _robust_z(r: np.ndarray, scale: np.ndarray | float | None = None) -> np.ndarray:
    med = np.median(r)
    s = robust_sigma(r) if scale is None else scale
    s = np.where(np.asarray(s) > 0, s, np.nan)
    return (r - med) / s


# ----------------------------------------------------------------------------- Wadati


@dataclass
class WadatiFit:
    """Result of :func:`fit_wadati`.

    ``pairs`` has one row per S pick with a P (index = original pick index): ``event_id``,
    ``alias``, ``tp_s``, ``ts_s``, ``sp_s`` (Ts-Tp), ``residual_s``, ``z``, ``outlier``.
    ``events`` has per-event free-intercept fits: ``event_id``, ``n_pairs``, ``vpvs``,
    ``intercept_s`` (origin-time shift), ``vpvs_z`` and ``inconsistent``.
    """

    vpvs: float
    vpvs_ci: tuple[float, float]
    vpvs_std: float
    n_pairs: int
    n_events: int
    sigma_s: float
    event_vpvs_median: float
    event_vpvs_spread: float
    pairs: pd.DataFrame
    events: pd.DataFrame
    tp_line: np.ndarray = field(default_factory=lambda: np.empty(0))
    sp_line: np.ndarray = field(default_factory=lambda: np.empty(0))


def _fit_k(tp: np.ndarray, sp: np.ndarray) -> float:
    k0 = float(np.median(sp / tp))
    return float(_huber_irls(tp[:, None], sp, np.array([k0]))[0])


def fit_wadati(
    picks: pd.DataFrame,
    sigma: float = 3.0,
    *,
    n_boot: int = 500,
    seed: int = 0,
    min_pairs: int = 3,
    min_event_sigma: float = 0.05,
) -> WadatiFit:
    """Fit Vp/Vs from the Wadati diagram (see module docstring).

    Parameters
    ----------
    picks
        ``AssociatedPicks.picks`` (needs ``phase``, ``travel_time_s``, ``ts_minus_tp_s``,
        ``event_id``; ``alias`` optional).
    sigma
        Robust z threshold for the outlier flag (``data_selection.wadati_outlier_sigma``).
    n_boot, seed
        Bootstrap replicates (events resampled) and RNG seed.
    min_pairs
        Minimum S-P pairs for a per-event fit.
    min_event_sigma
        Floor on the spread of per-event Vp/Vs used for the consistency test.

    Raises
    ------
    ValueError
        If fewer than 3 usable S-P pairs exist.
    """
    s = picks[(picks["phase"] == "S") & picks["ts_minus_tp_s"].notna()]
    sp = s["ts_minus_tp_s"].to_numpy(float)
    tp = s["travel_time_s"].to_numpy(float) - sp
    ok = np.isfinite(sp) & np.isfinite(tp) & (tp > 0)
    s, sp, tp = s[ok], sp[ok], tp[ok]
    if len(s) < 3:
        raise ValueError(
            f"Wadati fit needs at least 3 S picks with a matching P pick; found {len(s)}. "
            "Check that P and S picks share event and station."
        )
    k = _fit_k(tp, sp)
    res = sp - k * tp
    z = _robust_z(res)
    ev = s["event_id"].to_numpy()
    alias = s["alias"].to_numpy() if "alias" in s else s["sta"].to_numpy()
    pairs = pd.DataFrame(
        {
            "event_id": ev,
            "alias": alias,
            "tp_s": tp,
            "ts_s": tp + sp,
            "sp_s": sp,
            "residual_s": res,
            "z": z,
            "outlier": np.abs(z) > sigma,
        },
        index=s.index,
    )

    # event-cluster bootstrap
    rng = np.random.default_rng(seed)
    uev, inv = np.unique(ev.astype(str), return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uev))]
    boots: list[float] = []
    if n_boot > 0:
        for _ in range(n_boot):
            if len(uev) >= 3:
                pick = rng.integers(0, len(uev), len(uev))
                idx = np.concatenate([groups[i] for i in pick])
            else:
                idx = rng.integers(0, len(tp), len(tp))
            boots.append(1.0 + _fit_k(tp[idx], sp[idx]))
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        std = float(np.std(boots, ddof=1))
    else:
        lo = hi = std = float("nan")

    # per-event free-intercept fits
    rows = []
    for i, g in enumerate(groups):
        if len(g) < min_pairs or np.ptp(tp[g]) <= 0:
            continue
        ts = tp[g] + sp[g]
        slope, icpt, _, _ = stats.theilslopes(ts, tp[g])
        rows.append((uev[i], len(g), float(slope), float(icpt)))
    events = pd.DataFrame(rows, columns=["event_id", "n_pairs", "vpvs", "intercept_s"])
    if len(events):
        med = float(events["vpvs"].median())
        spread = max(robust_sigma(events["vpvs"].to_numpy()), min_event_sigma)
        events["vpvs_z"] = (events["vpvs"] - med) / spread
        events["inconsistent"] = events["vpvs_z"].abs() > sigma
    else:
        med = spread = float("nan")
        events["vpvs_z"] = []
        events["inconsistent"] = []
    tl = np.array([0.0, float(tp.max())])
    return WadatiFit(
        vpvs=1.0 + k,
        vpvs_ci=(float(lo), float(hi)),
        vpvs_std=std,
        n_pairs=len(s),
        n_events=len(uev),
        sigma_s=robust_sigma(res),
        event_vpvs_median=med,
        event_vpvs_spread=float(spread),
        pairs=pairs,
        events=events,
        tp_line=tl,
        sp_line=k * tl,
    )


# ------------------------------------------------------------------------ travel times


@dataclass
class TTFit:
    """Result of :func:`fit_travel_time` for one phase.

    ``table`` (index = original pick index) has ``distance_km``, ``travel_time_s``,
    ``residual_s``, ``scale_s`` (local robust sigma), ``z``, ``outlier``, ``invalid``.
    ``coef`` are polynomial coefficients, highest power first (``numpy.polyval``);
    ``curve_d``/``curve_t`` are a plotting line.
    """

    phase: str
    degree: int
    coef: np.ndarray
    n: int
    sigma_s: float
    table: pd.DataFrame
    curve_d: np.ndarray = field(default_factory=lambda: np.empty(0))
    curve_t: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def n_outliers(self) -> int:
        """Number of flagged picks (including invalid travel times)."""
        return int(self.table["outlier"].sum())


def _local_scale(d: np.ndarray, r: np.ndarray, min_bin: int, global_sigma: float) -> np.ndarray:
    n = len(d)
    n_bins = max(1, min(8, n // max(min_bin, 1)))
    scale = np.full(n, global_sigma)
    if n_bins == 1:
        return scale
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)
    floor = 0.5 * global_sigma
    for b in range(n_bins):
        m = which == b
        if m.sum() >= 3:
            scale[m] = max(robust_sigma(r[m]), floor)
    return scale


def fit_travel_time(
    picks: pd.DataFrame,
    phase: str,
    sigma: float = 3.0,
    *,
    degree: int = 2,
    min_bin: int = 40,
) -> TTFit:
    """Robust polynomial fit of travel time vs hypocentral distance for one phase.

    Raises
    ------
    ValueError
        If there are not enough valid picks (``degree + 2``) for the phase.
    """
    p = picks[picks["phase"] == phase]
    d = p["hypocentral_km"].to_numpy(float)
    t = p["travel_time_s"].to_numpy(float)
    invalid = ~(np.isfinite(d) & np.isfinite(t) & (t > 0))
    v = ~invalid
    if v.sum() < degree + 2:
        raise ValueError(f"Too few valid {phase} picks ({int(v.sum())}) for a degree-{degree} fit.")
    deg = degree
    design = np.vander(d[v], deg + 1)
    beta = _huber_irls(design, t[v])
    res = np.full(len(p), np.nan)
    res[v] = t[v] - design @ beta
    gs = robust_sigma(res[v])
    scale = np.full(len(p), np.nan)
    scale[v] = _local_scale(d[v], res[v], min_bin, gs)
    z = np.full(len(p), np.nan)
    z[v] = (res[v] - np.median(res[v])) / np.where(scale[v] > 0, scale[v], np.nan)
    flag = invalid | (np.abs(np.nan_to_num(z)) > sigma)
    table = pd.DataFrame(
        {
            "distance_km": d,
            "travel_time_s": t,
            "residual_s": res,
            "scale_s": scale,
            "z": z,
            "outlier": flag,
            "invalid": invalid,
        },
        index=p.index,
    )
    cd = np.linspace(0.0, float(np.nanmax(d[v])), 100)
    return TTFit(phase, deg, beta, int(v.sum()), gs, table, cd, np.polyval(beta, cd))


# --------------------------------------------------------------------------- combined


@dataclass
class OutlierAnalysis:
    """Everything produced by :func:`analyse_outliers`."""

    wadati: WadatiFit | None
    tt: dict[str, TTFit]
    pick_table: pd.DataFrame
    stations: pd.DataFrame
    findings: list[QCFinding]
    sigma: float

    @property
    def flags(self) -> pd.Series:
        """Boolean outlier flag aligned with the picks index."""
        return self.pick_table["outlier"]

    @property
    def vpvs(self) -> float:
        """Pooled robust Vp/Vs (NaN if no Wadati fit was possible)."""
        return self.wadati.vpvs if self.wadati else float("nan")

    def counts(self) -> dict[str, dict[str, int]]:
        """Outlier counts per phase and criterion."""
        return _counts(self.pick_table)


def _station_stats(
    picks: pd.DataFrame,
    tt: dict[str, TTFit],
    min_picks: int,
    min_abs_s: dict[str, float],
    k: float,
) -> pd.DataFrame:
    """Per (station, phase) median travel-time residual, its standard error and a flag.

    ``se = 1.253 * 1.4826 * MAD / sqrt(n)``; a station is flagged when ``n >= min_picks`` and
    ``|median| > max(min_abs_s[phase], k * se)``.  Outliers are deliberately not excluded:
    the median is robust and a large station delay is what this should reveal.
    """
    key = "alias" if "alias" in picks else "sta"
    rows = []
    for ph, fit in tt.items():
        r = fit.table["residual_s"]
        sub = pd.DataFrame({"station": picks.loc[r.index, key], "r": r})
        sub = sub[sub["r"].notna()]
        thr_abs = min_abs_s.get(ph, 0.0)
        for sta, g in sub.groupby("station"):
            x = g["r"].to_numpy()
            n = len(x)
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            se = 1.253 * _MAD_TO_SIGMA * mad / np.sqrt(n)
            thr = max(thr_abs, k * se)
            rows.append((sta, ph, n, med, mad, se, thr, n >= min_picks and abs(med) > thr))
    return pd.DataFrame(
        rows,
        columns=[
            "station",
            "phase",
            "n",
            "median_residual_s",
            "mad_s",
            "se_s",
            "threshold_s",
            "systematic",
        ],
    )


def analyse_outliers(
    picks: pd.DataFrame | Any,
    sigma: float = 3.0,
    *,
    tt_degree: int = 2,
    n_boot: int = 500,
    seed: int = 0,
    max_outlier_fraction: float = 0.05,
    min_station_picks: int = 5,
    station_min_abs_s: dict[str, float] | None = None,
    station_k: float = 3.0,
) -> OutlierAnalysis:
    """Run the Wadati and travel-time analyses and combine the outlier flags.

    Parameters
    ----------
    picks
        ``AssociatedPicks`` or its ``picks`` DataFrame.
    sigma
        Robust z threshold (``data_selection.wadati_outlier_sigma``), used for both analyses.
    tt_degree
        Polynomial degree for the travel-time fits.
    n_boot, seed
        Wadati bootstrap settings.
    max_outlier_fraction
        Per-phase outlier fraction above which a ``warn`` finding is emitted.
    min_station_picks
        Minimum picks of a (station, phase) for the systematic-residual test.
    station_min_abs_s
        Minimum absolute median residual (s) to flag a station, per phase; default
        ``{"P": 0.15, "S": 0.25}``.
    station_k
        A station is flagged when ``|median| > max(min_abs, station_k * se)``.

    Returns
    -------
    OutlierAnalysis
    """
    df = getattr(picks, "picks", picks)
    findings: list[QCFinding] = []
    wad: WadatiFit | None = None
    try:
        wad = fit_wadati(df, sigma, n_boot=n_boot, seed=seed)
    except ValueError as exc:
        findings.append(QCFinding("wadati_unavailable", "warn", str(exc)))
    tt: dict[str, TTFit] = {}
    for ph in ("P", "S"):
        if (df["phase"] == ph).any():
            try:
                tt[ph] = fit_travel_time(df, ph, sigma, degree=tt_degree)
            except ValueError as exc:
                findings.append(QCFinding(f"tt_fit_unavailable_{ph}", "warn", str(exc)))

    t = pd.DataFrame(index=df.index)
    t["phase"] = df["phase"]
    t["tt_residual_s"] = np.nan
    t["tt_z"] = np.nan
    t["is_tt_outlier"] = False
    t["tt_invalid"] = False
    for fit in tt.values():
        ix = fit.table.index
        t.loc[ix, "tt_residual_s"] = fit.table["residual_s"]
        t.loc[ix, "tt_z"] = fit.table["z"]
        t.loc[ix, "is_tt_outlier"] = fit.table["outlier"]
        t.loc[ix, "tt_invalid"] = fit.table["invalid"]
    t["wadati_residual_s"] = np.nan
    t["wadati_z"] = np.nan
    t["is_wadati_outlier"] = False
    if wad is not None:
        t.loc[wad.pairs.index, "wadati_residual_s"] = wad.pairs["residual_s"]
        t.loc[wad.pairs.index, "wadati_z"] = wad.pairs["z"]
        t.loc[wad.pairs.index, "is_wadati_outlier"] = wad.pairs["outlier"]
    for c in ("is_tt_outlier", "tt_invalid", "is_wadati_outlier"):
        t[c] = t[c].astype(bool)
    t["outlier"] = t["is_tt_outlier"] | t["is_wadati_outlier"]
    reason = np.where(
        t["tt_invalid"],
        "tt_invalid",
        np.where(
            t["is_tt_outlier"] & t["is_wadati_outlier"],
            "tt+wadati",
            np.where(t["is_tt_outlier"], "tt", np.where(t["is_wadati_outlier"], "wadati", "")),
        ),
    )
    t["reason"] = reason

    if wad is not None:
        lo, hi = wad.vpvs_ci
        findings.append(
            QCFinding(
                "wadati_vpvs",
                "info",
                f"Vp/Vs = {wad.vpvs:.3f} (95% CI {lo:.3f}-{hi:.3f}) from {wad.n_pairs} S-P pairs "
                f"in {wad.n_events} events; residual sigma {wad.sigma_s:.3f} s; "
                f"per-event median {wad.event_vpvs_median:.3f} "
                f"(spread {wad.event_vpvs_spread:.3f}, {len(wad.events)} events).",
                wad.n_pairs,
                {
                    "vpvs": wad.vpvs,
                    "ci": [lo, hi],
                    "event_median": wad.event_vpvs_median,
                    "event_spread": wad.event_vpvs_spread,
                },
            )
        )
        bad = wad.events[wad.events["inconsistent"]] if len(wad.events) else wad.events
        if len(bad):
            findings.append(
                QCFinding(
                    "wadati_inconsistent_events",
                    "warn",
                    f"{len(bad)} events have a per-event Vp/Vs inconsistent with the median "
                    f"({wad.event_vpvs_median:.3f}).",
                    len(bad),
                    {"events": {str(r.event_id): round(r.vpvs, 3) for r in bad.itertuples()}},
                )
            )
    for ph, c in _counts(t).items():
        frac = c["any"] / c["n"] if c["n"] else 0.0
        findings.append(
            QCFinding(
                f"outliers_{ph}",
                "warn" if frac > max_outlier_fraction else "info",
                f"{c['any']} of {c['n']} {ph} picks flagged as outliers ({100 * frac:.1f}%): "
                f"{c['tt']} travel-time, {c['wadati']} Wadati.",
                c["any"],
                c,
            )
        )
    stn = _station_stats(
        df, tt, min_station_picks, station_min_abs_s or {"P": 0.15, "S": 0.25}, station_k
    )
    flagged = stn[stn["systematic"]] if len(stn) else stn
    if len(flagged):
        findings.append(
            QCFinding(
                "station_systematic_residuals",
                "info",
                f"{flagged['station'].nunique()} stations have a median travel-time residual "
                "exceeding the threshold (possible station delays or 1-D curve misfit).",
                int(flagged["station"].nunique()),
                {
                    "stations": {
                        f"{r.station}/{r.phase}": round(r.median_residual_s, 3)
                        for r in flagged.itertuples()
                    }
                },
            )
        )
    for f in findings:
        logger.info("%s", f)
    return OutlierAnalysis(wad, tt, t, stn, findings, sigma)


def _counts(t: pd.DataFrame) -> dict[str, dict[str, int]]:
    out = {}
    for ph in ("P", "S"):
        m = t["phase"] == ph
        if m.any():
            out[ph] = {
                "n": int(m.sum()),
                "tt": int((t.is_tt_outlier & m).sum()),
                "wadati": int((t.is_wadati_outlier & m).sum()),
                "any": int((t.outlier & m).sum()),
            }
    return out


def outlier_flags(picks: pd.DataFrame | Any, sigma: float = 3.0, **kwargs: Any) -> pd.Series:
    """Boolean outlier flag per pick (union of travel-time and Wadati outliers).

    Aligned with the index of ``picks`` (a DataFrame or ``AssociatedPicks``); use
    :func:`analyse_outliers` for reasons, residuals and fits.
    """
    return analyse_outliers(picks, sigma, **kwargs).flags.rename("outlier")
