"""Shared machinery for slice and section figures of gridded velocity models.

Models are :class:`xarray.Dataset` s with coords ``depth`` (km positive down, ascending), ``lat``,
``lon`` and variables ``vp`` / ``vs`` (km/s), optionally ``vp_bg`` / ``vs_bg`` (the 1-D
background).  Datasets from Stage 3/4 include the FMTOMO cushion nodes (``attrs["n_cushion"]``);
the displayed region is the *interior* (cushion removed).

* :class:`ModelFields` evaluates derived quantities (``vp``, ``dvp_1d``, ``dvp_start``, ``vpvs``
  ...) at arbitrary points, either by **linear** interpolation between nodes (default, what a
  plot normally shows) or by the **cubic B-spline** that ``fm3d`` uses to evaluate the model
  (:func:`bspline_sample`, mirrors ``interpolate_velocity``); the latter smooths the node
  values, so a node-wise checkerboard looks weaker than the node values.
* :class:`ModelMasks` holds the "well sampled" masks (DWS threshold and optional resolvable
  mask) for P and S.
* Small drawing helpers (mask overlay, colour limits, swath projection) used by
  :mod:`osv.analysis.plots_model`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from osv.coords import FMTOMO_EARTH_RADIUS_KM

log = logging.getLogger(__name__)

__all__ = [
    "QUANTITIES",
    "ModelFields",
    "ModelMasks",
    "Quantity",
    "bspline_sample",
    "colour_limits",
    "crop_interior",
    "default_profiles",
    "draw_mask",
    "local_km",
    "project_swath",
]

KM_PER_DEG = FMTOMO_EARTH_RADIUS_KM * math.pi / 180.0
#: Points are kept this fraction of a node inside the upper valid edge of the B-spline.
_EDGE_EPS = 1e-6


# ----------------------------------------------------------------------------------------
# quantities
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """A plottable quantity.

    ``kind`` is ``"abs"`` (km/s, sequential colours), ``"pct"`` (percent, symmetric diverging
    colours centred on 0) or ``"ratio"`` (Vp/Vs, sequential).  ``phase`` selects the coverage
    mask: ``"p"``, ``"s"`` or ``"ps"`` (both).
    """

    key: str
    label: str
    kind: str
    phase: str
    title: str = ""


QUANTITIES: dict[str, Quantity] = {
    q.key: q
    for q in (
        Quantity("vp", "Vp (km/s)", "abs", "p", "Vp"),
        Quantity("vs", "Vs (km/s)", "abs", "s", "Vs"),
        Quantity("vpvs", "Vp/Vs", "ratio", "ps", "Vp/Vs"),
        Quantity("dvp_1d", "dVp vs 1-D (%)", "pct", "p", "dVp vs 1-D model"),
        Quantity("dvs_1d", "dVs vs 1-D (%)", "pct", "s", "dVs vs 1-D model"),
        Quantity("dvp_start", "dVp vs starting (%)", "pct", "p", "dVp vs starting model"),
        Quantity("dvs_start", "dVs vs starting (%)", "pct", "s", "dVs vs starting model"),
        Quantity("start_vp", "Vp (km/s)", "abs", "p", "Starting Vp"),
        Quantity("start_vs", "Vs (km/s)", "abs", "s", "Starting Vs"),
        Quantity("start_vpvs", "Vp/Vs", "ratio", "ps", "Starting Vp/Vs"),
    )
}


def crop_interior(ds: xr.Dataset) -> xr.Dataset:
    """Drop the cushion nodes (``attrs["n_cushion"]``, default 0) from a model dataset."""
    nc = int(ds.attrs.get("n_cushion", 0))
    if nc <= 0:
        return ds
    return ds.isel({d: slice(nc, ds.sizes[d] - nc) for d in ("depth", "lat", "lon")})


# ----------------------------------------------------------------------------------------
# interpolation
# ----------------------------------------------------------------------------------------


def _bspline_axis(x: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Node indices ``(n, 4)`` and uniform cubic B-spline weights ``(n, 4)`` along one axis.

    Valid from the second node to the second-to-last node (``fm3d`` rule); other points give
    NaN weights.
    """
    dx = (x[-1] - x[0]) / (x.size - 1)
    i = np.floor((pts - x[0]) / dx).astype(int)
    ok = (i >= 1) & (i <= x.size - 3)
    ic = np.clip(i, 1, x.size - 3)
    u = np.where(ok, (pts - x[ic]) / dx, np.nan)
    w = np.stack(
        [
            (1 - u) ** 3 / 6,
            (4 - 6 * u**2 + 3 * u**3) / 6,
            (1 + 3 * u + 3 * u**2 - 3 * u**3) / 6,
            u**3 / 6,
        ],
        axis=1,
    )
    idx = ic[:, None] - 1 + np.arange(4)[None, :]
    return idx, w


def bspline_sample(ds: xr.Dataset, name: str, depth: Any, lat: Any, lon: Any) -> np.ndarray:
    """Evaluate a variable like ``fm3d``: tensor-product uniform cubic B-spline of the nodes.

    Uses the *full* grid including cushion nodes; points outside ``[node 2, node n-1)`` of an
    axis give NaN.  Points are broadcast against each other.
    """
    d, la, lo = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (depth, lat, lon)))
    shape = d.shape
    vals = np.asarray(ds[name].transpose("depth", "lat", "lon").values, dtype=float)
    iz, wz = _bspline_axis(ds["depth"].values.astype(float), d.ravel())
    iy, wy = _bspline_axis(ds["lat"].values.astype(float), la.ravel())
    ix, wx = _bspline_axis(ds["lon"].values.astype(float), lo.ravel())
    out = np.empty(d.size)
    chunk = 200_000
    for s in range(0, d.size, chunk):
        sl = slice(s, s + chunk)
        v = vals[iz[sl, :, None, None], iy[sl, None, :, None], ix[sl, None, None, :]]
        w = wz[sl, :, None, None] * wy[sl, None, :, None] * wx[sl, None, None, :]
        out[sl] = np.sum(v * w, axis=(1, 2, 3))
    return out.reshape(shape)


def _linear_sample(ds: xr.Dataset, name: str, depth: Any, lat: Any, lon: Any) -> np.ndarray:
    f = RegularGridInterpolator(
        (ds["depth"].values, ds["lat"].values, ds["lon"].values),
        np.asarray(ds[name].transpose("depth", "lat", "lon").values, dtype=float),
        bounds_error=False,
        fill_value=np.nan,
    )
    d, la, lo = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (depth, lat, lon)))
    return f(np.stack([d.ravel(), la.ravel(), lo.ravel()], axis=-1)).reshape(d.shape)


def _clip_axis(v: Any, x: np.ndarray, lo_node: int, hi_node: int) -> np.ndarray:
    dx = (x[-1] - x[0]) / (x.size - 1)
    return np.clip(np.asarray(v, dtype=float), x[lo_node], x[hi_node] - _EDGE_EPS * dx)


class ModelFields:
    """Derived model quantities evaluated at arbitrary ``(depth, lat, lon)`` points.

    Parameters
    ----------
    model
        Final model dataset with ``vp`` and optionally ``vs`` (and ``vp_bg``, ``vs_bg``).
    reference_1d
        1-D reference: a dataset on any grid with ``vp_bg``/``vs_bg`` (or ``vp``/``vs``), or a
        table / mapping with ``depth`` (km) and ``vp`` (``vs``) columns interpolated linearly
        in depth.  Default: the ``*_bg`` variables of ``model`` when present.
    starting
        Starting model dataset (same grid as ``model`` is not required).
    interpolation
        ``"linear"`` (display) or ``"bspline"`` (as ``fm3d`` evaluates the model).

    Points outside the interior of the model are clipped onto it.
    """

    def __init__(
        self,
        model: xr.Dataset,
        reference_1d: Any = None,
        starting: xr.Dataset | None = None,
        interpolation: str = "linear",
    ) -> None:
        if interpolation not in ("linear", "bspline"):
            raise ValueError("interpolation must be 'linear' or 'bspline'")
        if "vp" not in model:
            raise ValueError("model dataset needs a 'vp' variable")
        self.model = model
        self.starting = starting
        self.interpolation = interpolation
        self._ref = self._reference(model, reference_1d)
        self.interior = crop_interior(model)
        self.depth = self.interior["depth"].values.astype(float)
        self.lat = self.interior["lat"].values.astype(float)
        self.lon = self.interior["lon"].values.astype(float)

    # -- reference ---------------------------------------------------------------------
    @staticmethod
    def _reference(model: xr.Dataset, ref: Any) -> Any:
        if ref is None:
            if "vp_bg" in model:
                return ("grid", model, {"vp": "vp_bg", "vs": "vs_bg"})
            return None
        if isinstance(ref, xr.Dataset):
            if "vp_bg" in ref:
                return ("grid", ref, {"vp": "vp_bg", "vs": "vs_bg"})
            return ("grid", ref, {"vp": "vp", "vs": "vs"})
        df = pd.DataFrame(ref)
        dcol = next((c for c in ("depth", "depth_km", "top_km") if c in df), None)
        if dcol is None or "vp" not in df:
            raise ValueError("reference_1d table needs 'depth' and 'vp' columns")
        df = df.sort_values(dcol)
        return ("table", df, dcol)

    # -- sampling ----------------------------------------------------------------------
    def _sample(self, ds: xr.Dataset, name: str, depth: Any, lat: Any, lon: Any) -> np.ndarray:
        x = {a: ds[a].values.astype(float) for a in ("depth", "lat", "lon")}
        nc = int(ds.attrs.get("n_cushion", 0))
        edge = max(nc, 1) if self.interpolation == "bspline" else nc
        d, la, lo_ = (
            _clip_axis(v, x[a], edge, x[a].size - 1 - edge)
            for v, a in ((depth, "depth"), (lat, "lat"), (lon, "lon"))
        )
        if self.interpolation == "bspline":
            return bspline_sample(ds, name, d, la, lo_)
        return _linear_sample(ds, name, d, la, lo_)

    def _ref_value(self, phase: str, depth: Any, lat: Any, lon: Any) -> np.ndarray:
        if self._ref is None:
            raise KeyError("no 1-D reference (give reference_1d or a model with vp_bg/vs_bg)")
        if self._ref[0] == "grid":
            _, ds, names = self._ref
            if names[phase] not in ds:
                raise KeyError(f"1-D reference has no {phase}")
            return self._sample(ds, names[phase], depth, lat, lon)
        _, df, dcol = self._ref
        if phase not in df:
            raise KeyError(f"1-D reference has no {phase}")
        d = np.asarray(depth, dtype=float)
        return np.interp(d, df[dcol].to_numpy(float), df[phase].to_numpy(float))

    # -- public ------------------------------------------------------------------------
    def available(self) -> list[str]:
        """Keys of :data:`QUANTITIES` computable from the inputs."""
        has_vs = "vs" in self.model
        has_start = self.starting is not None
        has_start_vs = has_start and "vs" in self.starting  # type: ignore[operator]
        ref_p = self._ref is not None and self._has_ref("vp")
        ref_s = self._ref is not None and self._has_ref("vs")
        out = ["vp"]
        if has_vs:
            out += ["vs", "vpvs"]
        if ref_p:
            out.append("dvp_1d")
        if ref_s and has_vs:
            out.append("dvs_1d")
        if has_start:
            out += ["start_vp", "dvp_start"]
            if has_vs and has_start_vs:
                out += ["start_vs", "dvs_start", "start_vpvs"]
        return out

    def _has_ref(self, phase: str) -> bool:
        if self._ref is None:
            return False
        if self._ref[0] == "grid":
            return self._ref[2][phase] in self._ref[1]
        return phase in self._ref[1]

    def evaluate(self, key: str, depth: Any, lat: Any, lon: Any) -> np.ndarray:
        """Value of quantity ``key`` at the points (broadcast against each other)."""
        if key not in QUANTITIES:
            raise KeyError(f"unknown quantity {key!r}; choose from {sorted(QUANTITIES)}")
        pts = (depth, lat, lon)
        with np.errstate(invalid="ignore", divide="ignore"):
            if key in ("vp", "vs"):
                return self._sample(self.model, key, *pts)
            if key == "vpvs":
                return self._sample(self.model, "vp", *pts) / self._sample(self.model, "vs", *pts)
            phase = "vp" if key.endswith("vp_1d") or key in ("dvp_1d", "dvp_start") else "vs"
            if key in ("dvp_1d", "dvs_1d"):
                ref = self._ref_value(phase, *pts)
                return 100.0 * (self._sample(self.model, phase, *pts) - ref) / ref
            if self.starting is None:
                raise KeyError("no starting model given")
            if key.startswith("start_"):
                nm = key[len("start_") :]
                if nm == "vpvs":
                    return self._sample(self.starting, "vp", *pts) / self._sample(
                        self.starting, "vs", *pts
                    )
                return self._sample(self.starting, nm, *pts)
            st = self._sample(self.starting, phase, *pts)
            return 100.0 * (self._sample(self.model, phase, *pts) - st) / st

    def note(self) -> str:
        """One-line description of the interpolation for figure footers."""
        if self.interpolation == "bspline":
            return "Model evaluated as fm3d sees it (uniform cubic B-spline of the node values)."
        return (
            "Linear interpolation between nodes (display only; fm3d evaluates a cubic B-spline,"
            " option interpolation='bspline')."
        )

    def node_arrays(self, key: str) -> np.ndarray:
        """Quantity at the interior nodes (used for colour limits)."""
        d, la, lo = np.meshgrid(self.depth, self.lat, self.lon, indexing="ij")
        return self.evaluate(key, d, la, lo)


# ----------------------------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------------------------


def _bool_interp(da: xr.DataArray) -> RegularGridInterpolator:
    da = da.transpose("depth", "lat", "lon")
    return RegularGridInterpolator(
        (da["depth"].values, da["lat"].values, da["lon"].values),
        da.values.astype(float),
        bounds_error=False,
        fill_value=0.0,
    )


@dataclass
class ModelMasks:
    """Well-sampled masks per phase (True = show) plus the DWS fields for contours.

    Build with :meth:`from_coverage`.  Empty masks (``None``) mean "everything shown".
    """

    covered: dict[str, xr.DataArray] = field(default_factory=dict)
    dws: dict[str, xr.DataArray] = field(default_factory=dict)
    resolvable_applied: bool = False

    @classmethod
    def from_coverage(
        cls,
        coverage: Any,
        threshold: float,
        resolvable_mask: Any = None,
        *,
        dilate: int = 0,
    ) -> ModelMasks:
        """Masks from DWS (``DataArray``, ``Dataset`` with ``dws``/``dws_vp``/``dws_vs``, or a
        ``{"vp": da, "vs": da}`` mapping) thresholded at ``threshold``, AND-ed with an optional
        ``resolvable_mask`` (boolean ``DataArray`` for all phases, a Dataset/mapping with ``vp``
        / ``vs`` entries, or one variable named ``resolvable``).
        """
        from osv.analysis.coverage import coverage_mask

        dws: dict[str, xr.DataArray] = {}
        if coverage is not None:
            if isinstance(coverage, xr.DataArray):
                dws = {"p": coverage, "s": coverage}
            else:
                get = (lambda k: coverage[k]) if hasattr(coverage, "__getitem__") else None
                names = list(coverage.keys()) if hasattr(coverage, "keys") else []
                for ph, cands in (("p", ("dws_vp", "vp", "dws", "dws_p")), ("s", ("dws_vs", "vs"))):
                    for c in cands:
                        if c in names:
                            dws[ph] = get(c)  # type: ignore[misc]
                            break
                if "p" in dws and "s" not in dws:
                    dws["s"] = dws["p"]
                if not dws:
                    raise ValueError("coverage: expected a DataArray, or variables dws[/_vp/_vs]")
        covered = {
            ph: coverage_mask(v, threshold, dilate=dilate).astype(bool) for ph, v in dws.items()
        }
        resolved = False
        if resolvable_mask is not None:
            res = cls._phase_masks(resolvable_mask)
            for ph in ("p", "s"):
                r = res.get(ph)
                if r is None:
                    continue
                if ph in covered:
                    covered[ph] = covered[ph] & r.reindex_like(
                        covered[ph], method="nearest"
                    ).astype(bool)
                else:
                    covered[ph] = r.astype(bool)
            resolved = True
        return cls(covered=covered, dws=dws, resolvable_applied=resolved)

    @staticmethod
    def _phase_masks(m: Any) -> dict[str, xr.DataArray]:
        if isinstance(m, xr.DataArray):
            return {"p": m, "s": m}
        names = list(m.keys())
        out: dict[str, xr.DataArray] = {}
        for ph, cands in (("p", ("vp", "p", "resolvable")), ("s", ("vs", "s", "resolvable"))):
            for c in cands:
                if c in names:
                    out[ph] = m[c]
                    break
        if not out:
            raise ValueError("resolvable_mask: need a DataArray or vp/vs/resolvable entries")
        return out

    def _for(self, phase: str) -> list[xr.DataArray]:
        phs = ["p", "s"] if phase == "ps" else [phase]
        return [self.covered[p] for p in phs if p in self.covered]

    def covered_at(self, phase: str, depth: Any, lat: Any, lon: Any) -> np.ndarray | None:
        """Boolean covered array at points, or ``None`` when no mask applies."""
        arrs = self._for(phase)
        if not arrs:
            return None
        d, la, lo = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (depth, lat, lon)))
        pts = np.stack([d.ravel(), la.ravel(), lo.ravel()], axis=-1)
        out = np.ones(d.shape, dtype=bool)
        for a in arrs:
            out &= (_bool_interp(a)(pts).reshape(d.shape)) >= 0.5
        return out

    def dws_at(
        self, phase: str, depth: Any, lat: Any, lon: Any, normalise: bool = False
    ) -> np.ndarray | None:
        """DWS (linear interpolation) at points, or ``None`` when absent.

        ``normalise`` divides by the maximum DWS of the phase (0-1).
        """
        ph = "s" if phase == "s" else "p"
        if ph not in self.dws:
            return None
        da = self.dws[ph].transpose("depth", "lat", "lon")
        f = RegularGridInterpolator(
            (da["depth"].values, da["lat"].values, da["lon"].values),
            da.values.astype(float),
            bounds_error=False,
            fill_value=0.0,
        )
        d, la, lo = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (depth, lat, lon)))
        out = f(np.stack([d.ravel(), la.ravel(), lo.ravel()], axis=-1)).reshape(d.shape)
        if normalise:
            top = float(np.nanmax(da.values))
            out = out / top if top > 0 else out
        return out


# ----------------------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------------------


def colour_limits(
    values: list[np.ndarray],
    kind: str,
    clip_percentile: float = 99.0,
    covered: list[np.ndarray | None] | None = None,
) -> tuple[float, float]:
    """Common ``(vmin, vmax)`` for a quantity across panels.

    ``kind="pct"`` gives symmetric limits ``(-m, m)`` with ``m`` the clip percentile of
    ``|value|``; other kinds use the (clipped) data range.  Only covered values count when
    ``covered`` masks are given.
    """
    parts = []
    for i, v in enumerate(values):
        v = np.asarray(v, dtype=float)
        m = covered[i] if covered is not None and covered[i] is not None else None
        sel = np.isfinite(v) if m is None else np.isfinite(v) & m
        parts.append(v[sel])
    a = np.concatenate(parts) if parts else np.array([])
    if a.size == 0:  # nothing covered: fall back to everything
        a = np.concatenate([np.asarray(v, float)[np.isfinite(v)] for v in values] or [np.array([])])
    if a.size == 0:
        return (-1.0, 1.0) if kind == "pct" else (0.0, 1.0)
    if kind == "pct":
        m = float(np.percentile(np.abs(a), clip_percentile))
        m = m if m > 1e-9 else 1.0
        return -m, m
    lo, hi = float(a.min()), float(a.max())
    # robust range: ignore the extreme 1-2 % tails (e.g. nodes above the ground surface)
    lo = float(np.percentile(a, 100 - clip_percentile))
    hi = float(np.percentile(a, clip_percentile))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def draw_mask(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    covered: np.ndarray | None,
    style: str = "grey",
    alpha: float = 0.65,
) -> None:
    """Overlay the not-covered region on ``ax`` (coordinates ``x`` by ``y``).

    ``style``: ``"grey"`` (opaque), ``"alpha"`` (translucent grey), ``"hatch"`` (hatching
    only) or ``"none"``.  ``covered`` has shape ``(len(y), len(x))``.
    """
    if covered is None or style == "none" or covered.all():
        return
    unc = (~covered).astype(float)
    kw: dict[str, Any] = {"levels": [0.5, 1.5], "zorder": 4}
    if style == "grey":
        ax.contourf(x, y, unc, colors=["#d9d9d9"], **kw)
    elif style == "alpha":
        ax.contourf(x, y, unc, colors=["#808080"], alpha=alpha, **kw)
    elif style == "hatch":
        cs = ax.contourf(x, y, unc, colors="none", hatches=["////"], **kw)
        cs.set_edgecolor("#555555")
        cs.set_linewidth(0.0)
    else:
        raise ValueError("mask style must be 'grey', 'alpha', 'hatch' or 'none'")


def local_km(lat: Any, lon: Any, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """East/north km of points relative to ``(lat0, lon0)`` on the FMTOMO sphere."""
    y = (np.asarray(lat, float) - lat0) * KM_PER_DEG
    x = (np.asarray(lon, float) - lon0) * KM_PER_DEG * math.cos(math.radians(lat0))
    return x, y


def project_swath(
    lat: Any, lon: Any, start: tuple[float, float], end: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Along-profile distance, absolute perpendicular distance (km) and profile length (km)."""
    lat0, lon0 = start
    x1, y1 = local_km(end[0], end[1], lat0, lon0)
    length = float(np.hypot(x1, y1))
    ux, uy = float(x1) / length, float(y1) / length
    px, py = local_km(lat, lon, lat0, lon0)
    return px * ux + py * uy, np.abs(-px * uy + py * ux), length


def default_profiles(
    lat: np.ndarray, lon: np.ndarray
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """E-W and N-S profiles through the centre of the ``lat`` / ``lon`` extent."""
    la0, la1, lo0, lo1 = float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())
    cla, clo = 0.5 * (la0 + la1), 0.5 * (lo0 + lo1)
    return [
        ("E-W", (cla, lo0), (cla, lo1)),
        ("N-S", (la0, clo), (la1, clo)),
    ]
