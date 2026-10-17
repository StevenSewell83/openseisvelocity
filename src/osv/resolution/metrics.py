"""Recovery metrics and the combined resolvable mask of the resolution tests (Stage 6).

All fields are on the full velocity grid ``(depth, lat, lon)``; cushion nodes are never valid.

Per-node (local) metrics compare the TRUE and RECOVERED perturbation (percent) in a 3-D
Gaussian-weighted moving window (``window_km`` is the full width, i.e. Gaussian
``sigma = window_km / 2`` per axis, in km):

``corr``
    weighted Pearson correlation (window means removed); 0 where the recovered field is flat
    but the true one is not, NaN where the true field is flat.
``amp``
    amplitude recovery = weighted regression slope of recovered on true (1 = full amplitude,
    0.5 = half, negative = inverted).
``rms_ratio``
    local rms(recovered) / rms(true).
``sign_agree``
    weighted fraction of nodes with ``|true| >= sign_floor * max|true|`` whose sign matches.

Global metrics (inside the coverage mask): Pearson ``corr``, Zelt (1998) ``semblance``
``S = sum (t + r)^2 / (2 sum (t^2 + r^2))`` (1 = identical, 0.5 = nothing recovered, 0 =
opposite) and the ``rms_ratio``.

Smearing (:func:`spike_smearing`): offset of the centroid of the recovered spike from the true
one and the principal axes of the second-moment (covariance) tensors of both; the excess of the
recovered over the true spread is the smearing.  :func:`pattern_anisotropy` estimates, for
checkerboards, the per-axis loss of wavenumber content of the recovered pattern: the gradient
energy along an axis equals the (negative) curvature of the pattern's autocorrelation at zero
lag, so the ratio recovered/true gives how much the pattern is stretched along that axis.

Resolvable mask (:func:`resolvable_from_checkerboards`): a node is resolvable if for at least one
checkerboard cell size ``corr >= min_corr`` AND ``amp >= min_amplitude`` (the offset copy of a
cell size is averaged with the unshifted one); ``resolved_scale_km`` is the smallest such cell
size.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from osv.resolution.synth import NETCDF_LOCK

log = logging.getLogger(__name__)

__all__ = [
    "DIMS",
    "global_metrics",
    "local_metrics",
    "pattern_anisotropy",
    "resolvable_from_checkerboards",
    "run_metrics",
    "spike_smearing",
    "evaluate_test",
]

DIMS = ("depth", "lat", "lon")
#: variance below this fraction of the (valid-node) total variance counts as "flat"
FLAT_FRACTION = 1e-6
#: sign agreement is evaluated where |true| exceeds this fraction of its maximum
SIGN_FLOOR = 0.1


def _spacing(grid: Any) -> tuple[float, float, float]:
    """``(dz, dy, dx)`` node spacing in km."""
    return float(grid.dz_km), float(grid.dy_km), float(grid.dx_km)


def interior_mask(grid: Any) -> np.ndarray:
    """True on the interior (non-cushion) nodes of the full grid."""
    return ~np.asarray(grid.cushion_mask().values, dtype=bool)


def _smooth(a: np.ndarray, sigma_nodes: Sequence[float]) -> np.ndarray:
    return ndimage.gaussian_filter(a, sigma=sigma_nodes, mode="constant", cval=0.0, truncate=3.0)


def _sigma_nodes(window_km: float, spacing: Sequence[float]) -> list[float]:
    if window_km <= 0:
        raise ValueError("window_km must be positive")
    return [0.5 * window_km / s for s in spacing]


def local_metrics(
    true: np.ndarray,
    rec: np.ndarray,
    spacing_km: Sequence[float],
    window_km: float,
    valid: np.ndarray | None = None,
    *,
    sign_floor: float = SIGN_FLOOR,
) -> dict[str, np.ndarray]:
    """Per-node correlation, amplitude recovery, rms ratio and sign agreement.

    Parameters
    ----------
    true, rec
        True and recovered perturbation (percent), same shape.
    spacing_km
        Node spacing along each array axis, km.
    window_km
        Full width of the Gaussian window (``sigma = window_km / 2``).
    valid
        Boolean array of nodes that take part (default all); invalid nodes get zero weight and
        NaN output.
    """
    t = np.asarray(true, dtype=float)
    r = np.asarray(rec, dtype=float)
    if t.shape != r.shape:
        raise ValueError(f"true {t.shape} and recovered {r.shape} shapes differ")
    ok = np.ones(t.shape, bool) if valid is None else np.asarray(valid, bool)
    ok = ok & np.isfinite(t) & np.isfinite(r)
    sg = _sigma_nodes(window_km, spacing_km)
    w = ok.astype(float)
    tt = np.where(ok, t, 0.0)
    rr = np.where(ok, r, 0.0)
    sw = _smooth(w, sg)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(sw > 1e-12, 1.0 / sw, np.nan)
        mt = _smooth(tt, sg) * inv
        mr = _smooth(rr, sg) * inv
        vt = np.maximum(_smooth(tt * tt, sg) * inv - mt**2, 0.0)
        vr = np.maximum(_smooth(rr * rr, sg) * inv - mr**2, 0.0)
        cv = _smooth(tt * rr, sg) * inv - mt * mr
        qt = _smooth(tt * tt, sg) * inv
        qr = _smooth(rr * rr, sg) * inv
        total = float(np.mean(tt[ok] ** 2)) if ok.any() else 0.0
        eps = FLAT_FRACTION * total
        flat_t = vt <= eps
        flat_r = vr <= eps
        corr = cv / np.sqrt(vt * vr)
        corr = np.where(flat_r & ~flat_t, 0.0, corr)
        corr = np.where(flat_t, np.nan, np.clip(corr, -1.0, 1.0))
        amp = np.where(flat_t, np.nan, cv / vt)
        rms_ratio = np.where(qt > eps, np.sqrt(qr / qt), np.nan)
        big = (np.abs(tt) >= sign_floor * (np.abs(tt).max() if ok.any() else 0.0)) & ok
        agree = (np.sign(tt) == np.sign(rr)) & big
        denom = _smooth(big.astype(float), sg)
        sign = np.where(denom > 1e-6, _smooth(agree.astype(float), sg) / denom, np.nan)
    out = {"corr": corr, "amp": amp, "rms_ratio": rms_ratio, "sign_agree": sign}
    for k, v in out.items():
        out[k] = np.where(ok & (sw > 1e-6), v, np.nan)
    return out


def global_metrics(true: np.ndarray, rec: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Correlation, Zelt (1998) semblance and rms ratio over the nodes where ``mask`` is True."""
    t = np.asarray(true, float)
    r = np.asarray(rec, float)
    ok = np.isfinite(t) & np.isfinite(r)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    n = int(ok.sum())
    nan = math.nan
    if n < 2:
        return {"corr": nan, "semblance": nan, "rms_ratio": nan, "amp": nan, "n_nodes": n}
    a, b = t[ok], r[ok]
    sa, sb = float(np.sum(a * a)), float(np.sum(b * b))
    da, db = a - a.mean(), b - b.mean()
    va, vb = float(np.sum(da * da)), float(np.sum(db * db))
    corr = (
        float(np.sum(da * db) / math.sqrt(va * vb))
        if va > 0 and vb > 0
        else (nan if va == 0 else 0.0)
    )
    sem = float(np.sum((a + b) ** 2) / (2.0 * (sa + sb))) if sa + sb > 0 else nan
    return {
        "corr": corr,
        "semblance": sem,
        "rms_ratio": math.sqrt(sb / sa) if sa > 0 else nan,
        "amp": float(np.sum(da * db) / va) if va > 0 else nan,
        "n_nodes": n,
    }


# ---------------------------------------------------------------------------------------
# smearing
# ---------------------------------------------------------------------------------------
def _moments(
    f: np.ndarray, pos: np.ndarray, sel: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    wgt = np.where(sel, np.clip(f, 0.0, None), 0.0)
    tot = wgt.sum()
    if tot <= 0:
        return None
    c = np.array([(wgt * p).sum() / tot for p in pos])
    d = [p - ci for p, ci in zip(pos, c, strict=True)]
    cov = np.array([[(wgt * d[i] * d[j]).sum() / tot for j in range(3)] for i in range(3)])
    return c, cov


def _axes(cov: np.ndarray) -> dict[str, Any]:
    """Principal axes of a covariance tensor in (x east, y north, z down)."""
    val, vec = np.linalg.eigh(cov)
    order = np.argsort(val)[::-1]
    val, vec = np.clip(val[order], 0.0, None), vec[:, order]
    major = vec[:, 0]
    if major[0] < 0 or (major[0] == 0 and major[1] < 0):
        major = -major
    horiz = math.hypot(major[0], major[1])
    return {
        "sigma_km": np.sqrt(val),
        "vectors": vec,
        "major_azimuth_deg": math.degrees(math.atan2(major[0], major[1])) % 180.0,
        "major_plunge_deg": math.degrees(math.atan2(abs(major[2]), horiz)),
    }


def spike_smearing(
    true: np.ndarray,
    rec: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    centre: Sequence[float] | None = None,
    search_radius_km: float,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Smearing of one spike from the moments of the recovered and the true spike.

    Moments use the positive part of ``sign * field`` (``sign`` of the true peak) in a sphere
    of ``search_radius_km`` around the true peak (or ``centre = (x, y, z)``).  Coordinates
    ``x`` (east), ``y`` (north), ``z`` (depth down) are arrays (broadcastable) in km.

    Returns a dict with ``centroid_offset_km`` (east, north, down), ``offset_km``,
    ``true_sigma_km`` / ``rec_sigma_km`` (principal standard deviations, major first),
    ``excess_sigma_km`` (``sqrt(max(rec^2 - true^2, 0))``), ``major_azimuth_deg`` (clockwise
    from north, 0-180), ``major_plunge_deg`` and ``elongation`` (major/minor of the recovered
    spread).  ``nan`` entries when nothing is recovered.
    """
    t, r = (np.asarray(a, float) for a in (true, rec))
    pos = [np.broadcast_to(np.asarray(a, float), t.shape) for a in (x, y, z)]
    ok = np.isfinite(t) & np.isfinite(r)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    if centre is None:
        i = np.unravel_index(np.argmax(np.where(ok, np.abs(t), -1.0)), t.shape)
        centre = [float(p[i]) for p in pos]
        sgn = 1.0 if t[i] >= 0 else -1.0
    else:
        i = np.unravel_index(
            np.argmin(sum((p - c) ** 2 for p, c in zip(pos, centre, strict=True))), t.shape
        )
        sgn = 1.0 if t[i] >= 0 else -1.0
    r2 = sum((p - c) ** 2 for p, c in zip(pos, centre, strict=True))
    sel = ok & (r2 <= search_radius_km**2)
    mt = _moments(sgn * np.where(ok, t, 0.0), pos, sel)
    mr = _moments(sgn * np.where(ok, r, 0.0), pos, sel)
    nan3 = np.full(3, math.nan)
    out: dict[str, Any] = {
        "true_centre_km": mt[0] if mt else nan3,
        "centroid_offset_km": nan3,
        "offset_km": math.nan,
        "true_sigma_km": nan3,
        "rec_sigma_km": nan3,
        "excess_sigma_km": nan3,
        "major_azimuth_deg": math.nan,
        "major_plunge_deg": math.nan,
        "elongation": math.nan,
        "peak_ratio": float(sgn * r[i] / abs(t[i])) if t[i] != 0 else math.nan,
    }
    if mt is None:
        return out
    at = _axes(mt[1])
    out["true_sigma_km"] = at["sigma_km"]
    if mr is None:
        return out
    ar = _axes(mr[1])
    off = mr[0] - mt[0]
    out.update(
        centroid_offset_km=off,
        offset_km=float(np.linalg.norm(off)),
        rec_sigma_km=ar["sigma_km"],
        excess_sigma_km=np.sqrt(np.clip(ar["sigma_km"] ** 2 - at["sigma_km"] ** 2, 0.0, None)),
        major_azimuth_deg=ar["major_azimuth_deg"],
        major_plunge_deg=ar["major_plunge_deg"],
        elongation=float(ar["sigma_km"][0] / ar["sigma_km"][2])
        if ar["sigma_km"][2] > 0
        else math.inf,
    )
    return out


def pattern_anisotropy(
    true: np.ndarray,
    rec: np.ndarray,
    spacing_km: Sequence[float],
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-axis wavenumber retention of a recovered checkerboard (optional smearing estimate).

    ``retention_{z,y,x} = sqrt(G_rec / G_true)`` with ``G`` the mean squared gradient along the
    axis, taken after scaling the recovered field to the true rms so only the *shape* matters.
    1 = pattern as sharp as the truth; < 1 = smeared (stretched) along that axis.
    ``anisotropy`` = max/min retention; ``weakest_axis`` names the most smeared axis.
    """
    t = np.asarray(true, float)
    r = np.asarray(rec, float)
    ok = np.isfinite(t) & np.isfinite(r)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    # gradient pairs are only used where both neighbours are valid
    sa = float(np.sqrt(np.mean(t[ok] ** 2))) if ok.any() else 0.0
    sb = float(np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else 0.0
    names = ("z", "y", "x")
    if sa <= 0 or sb <= 0:
        return {f"retention_{n}": math.nan for n in names} | {
            "anisotropy": math.nan,
            "weakest_axis": "",
        }
    r = r * (sa / sb)
    ret: dict[str, float] = {}
    for ax, n in enumerate(names):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
        pair = ok[tuple(sl0)] & ok[tuple(sl1)]
        if not pair.any():
            ret[f"retention_{n}"] = math.nan
            continue
        gt = (t[tuple(sl1)] - t[tuple(sl0)])[pair] / spacing_km[ax]
        gr = (r[tuple(sl1)] - r[tuple(sl0)])[pair] / spacing_km[ax]
        g0 = float(np.mean(gt**2))
        ret[f"retention_{n}"] = math.sqrt(float(np.mean(gr**2)) / g0) if g0 > 0 else math.nan
    vals = {n: ret[f"retention_{n}"] for n in names if np.isfinite(ret[f"retention_{n}"])}
    if vals:
        lo = min(vals, key=vals.get)  # type: ignore[arg-type]
        hi = max(vals.values())
        ret["anisotropy"] = hi / vals[lo] if vals[lo] > 0 else math.inf
        ret["weakest_axis"] = lo  # type: ignore[assignment]
    else:
        ret["anisotropy"], ret["weakest_axis"] = math.nan, ""  # type: ignore[assignment]
    return ret


# ---------------------------------------------------------------------------------------
# per test
# ---------------------------------------------------------------------------------------
def _pct_fields(true_ds: xr.Dataset, rec_ds: xr.Dataset, phase: str) -> tuple[np.ndarray, ...]:
    """True and recovered perturbation (percent of the background) for ``phase``."""
    if phase.upper() == "P":
        t = true_ds["dvp_pct"].values
        r = rec_ds["dvp_pct"].values
    else:
        if "vs_bg" in true_ds and "vs" in true_ds:
            t = 100.0 * (true_ds["vs"].values / true_ds["vs_bg"].values - 1.0)
        else:
            t = true_ds["dvp_pct"].values
        r = rec_ds["dvs_pct"].values
    return np.asarray(t, float), np.asarray(r, float)


def evaluate_test(
    true: np.ndarray,
    rec: np.ndarray,
    grid: Any,
    *,
    window_km: float,
    coverage: np.ndarray | None = None,
    kind: str = "checkerboard",
    params: Mapping[str, Any] | None = None,
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Local fields (Dataset on the full grid) and a summary dict for one test.

    ``coverage`` (bool, full grid) restricts the global metrics; the local fields are computed
    on the interior and are NaN on the cushion (and on nodes outside ``coverage`` only when
    summarised, never masked in the fields so that the mask logic can combine them).
    """
    spacing = _spacing(grid)
    interior = interior_mask(grid)
    loc = local_metrics(true, rec, spacing, window_km, interior)
    gmask = interior if coverage is None else interior & np.asarray(coverage, bool)
    summary: dict[str, Any] = {"kind": kind, "window_km": float(window_km)}
    summary.update({f"global_{k}": v for k, v in global_metrics(true, rec, gmask).items()})
    with np.errstate(invalid="ignore"):
        for k in ("corr", "amp", "sign_agree"):
            v = loc[k][gmask]
            v = v[np.isfinite(v)]
            summary[f"median_{k}"] = float(np.median(v)) if v.size else math.nan
    if kind == "checkerboard":
        summary.update(
            {f"aniso_{k}": v for k, v in pattern_anisotropy(true, rec, spacing, gmask).items()}
        )
    if kind.startswith("spike"):
        if xyz is None:
            from osv.resolution.models import grid_xyz

            xyz = grid_xyz(grid)
        specs = (params or {}).get("spikes", [])
        sm = []
        for s in specs:
            c = (float(s["x_km"]), float(s["y_km"]), float(s["z_km"]))
            res = spike_smearing(
                true,
                rec,
                *xyz,
                centre=c,
                search_radius_km=max(3.0 * float(s["radius_km"]), 2.0 * max(spacing)),
                valid=interior,
            )
            sm.append(
                {
                    k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in res.items()
                }
            )
        summary["spikes"] = sm
    coords = {"depth": grid.depth_full, "lat": grid.lat_full, "lon": grid.lon_full}
    ds = xr.Dataset(
        {k: (DIMS, v) for k, v in loc.items()},
        coords=coords,
        attrs={"window_km": float(window_km), "kind": kind},
    )
    return ds, summary


# ---------------------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------------------
def resolvable_from_checkerboards(
    fields: Mapping[float, Sequence[Mapping[str, np.ndarray]]],
    *,
    min_corr: float = 0.5,
    min_amplitude: float = 0.3,
    coverage: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean resolvable mask and ``resolved_scale_km`` from checkerboard local fields.

    Parameters
    ----------
    fields
        ``{cell_km: [ {"corr": ..., "amp": ...}, ... ]}`` (one entry per copy, e.g. unshifted
        and half-cell-offset); per size the copies are averaged (``nanmean``).
    coverage
        Optional boolean field AND-ed into the mask (e.g. the DWS coverage mask).
    valid
        Optional boolean field (e.g. interior nodes) outside which the mask is False.

    Returns
    -------
    resolvable, resolved_scale_km
        ``resolved_scale_km`` is the smallest cell size passing both thresholds, NaN where none.
    """
    if not fields:
        raise ValueError("no checkerboard fields given")
    shape = np.shape(next(iter(fields.values()))[0]["corr"])
    scale = np.full(shape, np.nan)
    for cell in sorted(fields, reverse=True):  # largest first, smaller sizes overwrite
        copies = fields[cell]
        with np.errstate(invalid="ignore"), _quiet():
            c = np.nanmean([f["corr"] for f in copies], axis=0)
            a = np.nanmean([f["amp"] for f in copies], axis=0)
        passed = np.isfinite(c) & np.isfinite(a) & (c >= min_corr) & (a >= min_amplitude)
        scale = np.where(passed, float(cell), scale)
    res = np.isfinite(scale)
    for extra in (coverage, valid):
        if extra is not None:
            res &= np.asarray(extra, bool)
    scale = np.where(res, scale, np.nan)
    return res, scale


class _quiet:  # noqa: N801
    """Silence nanmean's all-NaN RuntimeWarning."""

    def __enter__(self) -> None:
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a: Any) -> None:
        self._w.__exit__(*a)


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def _cfg_mask(cfg: Any) -> Any:
    rc = getattr(cfg, "resolution", cfg)
    return getattr(rc, "mask", rc)


def run_metrics(
    result: Any,
    grid: Any,
    cfg: Any,
    out_dir: str | Path | None = None,
    *,
    coverage: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    """Metrics of every successful test of a :class:`ResolutionRunResult` and the resolvable mask.

    Parameters
    ----------
    result
        Output of :func:`osv.resolution.runner.run_resolution_tests` (table with ``name``,
        ``kind``, ``phase``, ``params``, ``success``, ``true_model``, ``recovered_model``).
    grid
        The :class:`~osv.model3d.grid.ModelGrid` of the tests.
    cfg
        :class:`~osv.config.schema.OSVConfig` (or the ``resolution`` / ``resolution.mask``
        section): ``min_corr``, ``min_amplitude``, ``window_km``, ``combine_with_dws``.
    out_dir
        Default ``<result.workdir>/metrics``.  Writes ``metrics.nc`` (per-test local fields,
        variables ``<field>__<test>``), ``summary.csv``, ``summary.json`` and
        ``resolvable_mask.nc``.
    coverage
        Boolean DWS coverage mask(s) (``DataArray`` for all phases or ``{"vp": da, "vs": da}``);
        used for the global metrics and, with ``combine_with_dws``, in the resolvable mask.

    Returns
    -------
    dict
        ``summary`` (DataFrame), ``metrics`` (Dataset), ``resolvable`` (Dataset, as written to
        ``resolvable_mask.nc``: bool ``vp`` [``vs``] and ``resolvable``, float
        ``resolved_scale_km_vp`` [``_vs``] and ``resolved_scale_km``) and ``paths``.
    """
    mc = _cfg_mask(cfg)
    out = Path(out_dir) if out_dir is not None else Path(result.workdir) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    spacing = _spacing(grid)
    interior = interior_mask(grid)
    from osv.resolution.models import grid_xyz

    xyz = grid_xyz(grid)

    def cov_for(phase: str) -> np.ndarray | None:
        if coverage is None:
            return None
        if isinstance(coverage, xr.DataArray):
            da = coverage
        else:
            da = coverage[f"v{phase.lower()}"] if f"v{phase.lower()}" in coverage else None
            if da is None:
                da = coverage.get("vp") if hasattr(coverage, "get") else None
            if da is None:
                return None
        arr = np.asarray(da.values if hasattr(da, "values") else da, bool)
        return arr if arr.shape == grid.shape_full else None

    table = result.table
    rows: list[dict[str, Any]] = []
    fields: dict[str, xr.DataArray] = {}
    cb: dict[str, dict[float, list[dict[str, np.ndarray]]]] = {}
    for _, row in table.iterrows():
        if not bool(row["success"]):
            continue
        name, kind, phase = str(row["name"]), str(row["kind"]), str(row["phase"])
        params = row["params"] if isinstance(row["params"], dict) else json.loads(row["params"])
        with NETCDF_LOCK:
            with xr.open_dataset(row["true_model"]) as d:
                tds = d.load()
            with xr.open_dataset(row["recovered_model"]) as d:
                rds = d.load()
        t, r = _pct_fields(tds, rds, phase)
        cell = params.get("cell_km")
        win = mc.window_km or (float(cell) if cell else 2.0 * max(spacing))
        ds, summ = evaluate_test(
            t, r, grid, window_km=win, coverage=cov_for(phase), kind=kind, params=params, xyz=xyz
        )
        for k in ds.data_vars:
            fields[f"{k}__{name}"] = ds[k]
        rows.append({"name": name, "phase": phase, **{k: v for k, v in summ.items()}})
        if kind == "checkerboard" and cell:
            cb.setdefault(phase, {}).setdefault(float(cell), []).append(
                {"corr": ds["corr"].values, "amp": ds["amp"].values}
            )
    metrics = xr.Dataset(fields) if fields else xr.Dataset()
    resolvable = xr.Dataset(
        coords={"depth": grid.depth_full, "lat": grid.lat_full, "lon": grid.lon_full}
    )
    first = None
    for phase in ("P", "S"):
        if phase not in cb:
            continue
        cov = cov_for(phase) if getattr(mc, "combine_with_dws", False) else None
        res, scale = resolvable_from_checkerboards(
            cb[phase],
            min_corr=mc.min_corr,
            min_amplitude=mc.min_amplitude,
            coverage=cov,
            valid=interior,
        )
        v = f"v{phase.lower()}"
        resolvable[v] = (DIMS, res)
        resolvable[f"resolved_scale_km_{v}"] = (DIMS, scale, {"units": "km"})
        first = first or v
    if first:
        resolvable["resolvable"] = resolvable[first]
        resolvable["resolved_scale_km"] = resolvable[f"resolved_scale_km_{first}"]
    resolvable.attrs.update(
        min_corr=float(mc.min_corr),
        min_amplitude=float(mc.min_amplitude),
        combine_with_dws=int(bool(getattr(mc, "combine_with_dws", False))),
    )
    summary = pd.DataFrame(rows)
    paths = {
        "metrics": out / "metrics.nc",
        "summary_csv": out / "summary.csv",
        "summary_json": out / "summary.json",
        "resolvable": out / "resolvable_mask.nc",
    }
    with NETCDF_LOCK:
        if fields:
            metrics.to_netcdf(paths["metrics"])
        if first:
            resolvable.to_netcdf(paths["resolvable"])
    summary.drop(columns=["spikes"], errors="ignore").to_csv(paths["summary_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(rows, indent=1, default=_json))
    return {
        "summary": summary,
        "metrics": metrics,
        "resolvable": resolvable,
        "paths": paths,
    }


def _json(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
