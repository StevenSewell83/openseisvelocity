"""FMTOMO grid definition from the configuration (Stage 3, task 4.1).

FMTOMO constraints that shape the grids (all verified in ``fortran/fmtomo/fmcode``)
-----------------------------------------------------------------------------------
* **Velocity grid** (``vgrids.in``) is a regular grid in ``(r, lat, lon)`` with radians for the
  angles.  The B-spline interpolation (``interpolate_velocity``, ``3dfmlib.f90:480-505``)
  needs ``2 <= i <= n-2`` (1-based, ``i = floor((x-x0)/dx)+1``), i.e. the evaluation point
  must lie in ``[x(2), x(n-1))``: **the cushion node on each side is never inside the
  usable region**, and the *upper* interior node is itself excluded.  The model region is
  therefore the grid *interior* (``n_cushion = 1`` node removed per side, ``grid3dg.f90:200-210``).
* **Propagation grid** (``propgrid.in``, ``3dfmlib.f90:98-135``) is evaluated from the velocity
  grids, so it must lie strictly *inside the interior* of the velocity grid (otherwise
  ``fm3d`` stops with "interpolation outside range ... your velocity parameter grid may not
  cover the entire propagation grid").  The same ``2..n-2`` rule applies to the interface
  grid (``interpolate_interface``, ``3dfmlib.f90:300-345``).  The propagation grid is
  therefore inset a fraction of a node from the interior on all sides.
* ``refinement_factor n_refine_cells`` (4th line of ``propgrid.in``) is *not* the refinement
  of the propagation grid relative to the velocity grid: it is the factor by which the
  propagation grid is locally refined around each source and the half-width of that refined
  patch in propagation cells (``3dfmlib.f90:2607-2625``).  ``propgrid_refine`` (config) is the
  ratio velocity-node spacing / propagation-node spacing.
* **Sources** must be within the lat/long range of the propagation grid (tolerance
  ``interface_tolerance*dr = 0.005 dr``, ``3dfmlib.f90:1575-1578``, ``mod_3dfm.f90:704``) and
  between the
  top and bottom interfaces (``3dfmlib.f90:1579-1582``).  **Receivers** must lie *inside* the
  propagation grid in lat/long with no tolerance (``3dfm_main.f90:423-427``) and between the
  interfaces (``3dfm_main.f90:431-436``).  The top interface must not be above the
  top propagation node and the bottom interface not below the bottom one
  (``3dfmlib.f90:241-242``).  Hence the model top lies ``top_margin_km`` above the highest
  station and the lateral extent is padded by at least half a node.

The in-memory layout follows :mod:`osv.io.fmtomo.grids`: ``depth`` ascends (shallowest
first), ``depth = 6371 - radius``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.coords import FMTOMO_EARTH_RADIUS_KM, spherical_distance_km
from osv.io.fmtomo.grids import (
    InterfaceGrids,
    PropGrid,
    VelocityGrids,
    interface_dataset,
    velocity_dataset,
)

log = logging.getLogger(__name__)

__all__ = ["GridError", "ModelGrid", "define_grid"]

KM_PER_DEG = FMTOMO_EARTH_RADIUS_KM * np.pi / 180.0  # FMTOMO is a sphere of 6371 km
#: Fraction of a (propagation) node by which the propagation grid is inset from the velocity
#: grid interior and from the interfaces, so that floating-point noise never trips the
#: ``floor`` tests of ``interpolate_velocity``.
INSET_FRACTION = 0.02
#: Rough fm3d memory per propagation node (bytes): node/region/pointer arrays of
#: ``Tpropagation_grid`` plus times, measured max RSS 2.0 GB for 10.3M nodes, 3 sources.
BYTES_PER_PROP_NODE = 200.0
#: ``interface_tolerance`` of ``mod_3dfm.f90:704`` (fraction of the radial spacing).
INTERFACE_TOLERANCE = 0.005


class GridError(ValueError):
    """The requested grid cannot satisfy FMTOMO's constraints (message says what to change)."""


def _axis(centre: float, extent: float, step: float) -> np.ndarray:
    """Regular axis of spacing ``step`` centred on ``centre`` whose span covers ``extent``."""
    n = int(np.ceil(extent / step - 1e-9)) + 1
    n = max(n, 2)
    lo = centre - 0.5 * (n - 1) * step
    return lo + step * np.arange(n)


def _extend(a: np.ndarray, step: float, n: int = 1) -> np.ndarray:
    """Add ``n`` cushion nodes on each side of an ascending axis."""
    k = np.arange(1, n + 1)
    return np.concatenate([a[0] - step * k[::-1], a, a[-1] + step * k])


def _prop_axis(lo: float, hi: float, target: float) -> tuple[float, int]:
    """Spacing (<= target) and node count of a propagation axis spanning ``[lo, hi]``."""
    n = int(np.ceil((hi - lo) / target - 1e-9)) + 1
    n = max(n, 2)
    return (hi - lo) / (n - 1), n


@dataclass
class ModelGrid:
    """FMTOMO velocity grid, propagation grid and helpers.

    Attributes
    ----------
    lat, lon, depth
        Node coordinates of the *interior* (the model region): degrees and km below sea level,
        ``depth`` ascending.  ``*_full`` include the cushion nodes.
    dlat_deg, dlon_deg, dz_km
        Node spacings actually used; ``dx_km``/``dy_km`` are the corresponding km spacings at
        the centre latitude (FMTOMO sphere).
    propgrid
        The :class:`~osv.io.fmtomo.grids.PropGrid` (inside the interior of the velocity grid).
    top_depth_km, bottom_depth_km
        Depths of the first/last interior node (top is negative = above sea level).
    """

    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray
    lat_full: np.ndarray
    lon_full: np.ndarray
    depth_full: np.ndarray
    dlat_deg: float
    dlon_deg: float
    dz_km: float
    dx_km: float
    dy_km: float
    propgrid: PropGrid
    centre_lat: float
    centre_lon: float
    n_cushion: int = 1
    warnings: list[str] = field(default_factory=list)

    # ---- geometry -------------------------------------------------------------------
    @property
    def top_depth_km(self) -> float:
        return float(self.depth[0])

    @property
    def bottom_depth_km(self) -> float:
        return float(self.depth[-1])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Interior ``(ndepth, nlat, nlon)``."""
        return (self.depth.size, self.lat.size, self.lon.size)

    @property
    def shape_full(self) -> tuple[int, int, int]:
        """Full ``(ndepth, nlat, nlon)`` including the cushion nodes (as written to vgrids.in)."""
        return (self.depth_full.size, self.lat_full.size, self.lon_full.size)

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.shape_full))

    @property
    def n_prop_nodes(self) -> int:
        pg = self.propgrid
        return pg.nr * pg.nlat * pg.nlon

    @property
    def fm3d_memory_gb(self) -> float:
        """Rough estimate of the fm3d memory for the propagation grid."""
        return self.n_prop_nodes * BYTES_PER_PROP_NODE / 1e9

    @property
    def surface_limits_km(self) -> tuple[float, float]:
        """``(top, bottom)`` depths between which all interfaces must lie (inside the propgrid)."""
        pg = self.propgrid
        inset = 0.5 * INSET_FRACTION * pg.dr_km
        return float(pg.depth_km[0] + inset), float(pg.depth_km[-1] - inset)

    def template(self, fill: float = np.nan, *, full: bool = True) -> xr.Dataset:
        """Dataset with coords ``depth``, ``lat``, ``lon`` and a ``velocity`` variable (km/s)."""
        d, la, lo = (
            (self.depth_full, self.lat_full, self.lon_full)
            if full
            else (self.depth, self.lat, self.lon)
        )
        ds = xr.Dataset(
            {
                "velocity": (
                    ("depth", "lat", "lon"),
                    np.full((d.size, la.size, lo.size), fill),
                    {"units": "km/s"},
                )
            },
            coords={
                "depth": ("depth", d, {"units": "km", "positive": "down"}),
                "lat": ("lat", la, {"units": "degrees_north"}),
                "lon": ("lon", lo, {"units": "degrees_east"}),
            },
            attrs={"n_cushion": self.n_cushion if full else 0},
        )
        return ds

    def cushion_mask(self) -> xr.DataArray:
        """True on cushion nodes of the full grid."""
        m = np.zeros(self.shape_full, dtype=bool)
        c = self.n_cushion
        m[:c] = m[-c:] = True
        m[:, :c] = m[:, -c:] = True
        m[:, :, :c] = m[:, :, -c:] = True
        return xr.DataArray(
            m,
            dims=("depth", "lat", "lon"),
            coords={"depth": self.depth_full, "lat": self.lat_full, "lon": self.lon_full},
        )

    # ---- containment ----------------------------------------------------------------
    def contains(self, lat: Any, lon: Any, depth: Any, *, region: str = "propgrid") -> np.ndarray:
        """Whether points lie inside ``region`` (``"propgrid"`` or ``"interior"``).

        ``"propgrid"`` is where ``fm3d`` accepts sources/receivers laterally and where the
        velocity is evaluated; ``"interior"`` is the model region of the velocity grid.
        Depths are km below sea level (negative above).
        """
        la, lo, de = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(lat, lon, depth))
        if region == "propgrid":
            pg = self.propgrid
            box = (pg.lat.min(), pg.lat.max(), pg.lon.min(), pg.lon.max())
            zr = (float(pg.depth_km[0]), float(pg.depth_km[-1]))
        elif region == "interior":
            box = (self.lat[0], self.lat[-1], self.lon[0], self.lon[-1])
            zr = (self.top_depth_km, self.bottom_depth_km)
        else:
            raise ValueError("region must be 'propgrid' or 'interior'")
        return (
            (la >= box[0]) & (la <= box[1]) & (lo >= box[2]) & (lo <= box[3])
            & (de >= zr[0]) & (de <= zr[1])
        )  # fmt: skip

    # ---- FMTOMO objects -------------------------------------------------------------
    def _full_values(self, v: Any, name: str) -> np.ndarray:
        a = np.asarray(v.values if hasattr(v, "values") else v, dtype=float)
        if a.ndim == 0:
            return np.full(self.shape_full, float(a))
        if a.shape == self.shape_full:
            return a
        if a.shape == self.shape:
            return np.pad(a, self.n_cushion, mode="edge")
        raise GridError(
            f"{name}: shape {a.shape} matches neither the full grid {self.shape_full} "
            f"nor the interior {self.shape} (axes: depth ascending, lat, lon)"
        )

    def to_vgrids(
        self, vp: Any, vs: Any | None = None, *, uncertainty: float | None = None
    ) -> VelocityGrids:
        """One-layer :class:`VelocityGrids` (type 1 = ``vp``, optional type 2 = ``vs``).

        Values are scalars, arrays or DataArrays with axes ``(depth, lat, lon)`` (depth
        ascending) of the full grid or of the interior (cushions filled by edge replication).
        ``uncertainty`` fills the second column (needed for ``vgridsref.in``).
        """
        out = []
        for name, v in (("vp", vp), ("vs", vs)):
            if v is None:
                continue
            a = self._full_values(v, name)
            if not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise GridError(f"{name}: values must be finite and positive (km/s)")
            unc = None if uncertainty is None else np.full(a.shape, float(uncertainty))
            out.append(
                [
                    velocity_dataset(
                        a[::-1],  # file order: radius ascending = depth descending
                        r0_km=float(FMTOMO_EARTH_RADIUS_KM - self.depth_full[-1]),
                        dr_km=self.dz_km,
                        lat0_rad=float(np.deg2rad(self.lat_full[0])),
                        dlat_rad=float(np.deg2rad(self.dlat_deg)),
                        lon0_rad=float(np.deg2rad(self.lon_full[0])),
                        dlon_rad=float(np.deg2rad(self.dlon_deg)),
                        uncertainty=None if unc is None else unc[::-1],
                        n_cushion=self.n_cushion,
                        layer=1,
                        vtype=len(out) + 1,
                    )
                ]
            )
        return VelocityGrids(out)

    def flat_interfaces(
        self, top_depth_km: float | None = None, bottom_depth_km: float | None = None
    ) -> InterfaceGrids:
        """Two flat interfaces (surface and base) on the lateral grid of the velocity model.

        Defaults: just inside the propagation grid top/bottom (:attr:`surface_limits_km`).
        A custom top must stay above all receivers (task 4.3 builds topographic surfaces).
        """
        lo, hi = self.surface_limits_km
        top = lo if top_depth_km is None else float(top_depth_km)
        bot = hi if bottom_depth_km is None else float(bottom_depth_km)
        if not (lo - 1e-9 <= top < bot <= hi + 1e-9):
            raise GridError(
                f"interfaces must satisfy {lo:g} <= top < bottom <= {hi:g} km depth "
                f"(inside the propagation grid), got top={top:g}, bottom={bot:g}"
            )
        shape = (2, self.lat_full.size, self.lon_full.size)
        rad = np.empty(shape)
        rad[0] = FMTOMO_EARTH_RADIUS_KM - top
        rad[1] = FMTOMO_EARTH_RADIUS_KM - bot
        return InterfaceGrids(
            interface_dataset(
                rad,
                lat0_rad=float(np.deg2rad(self.lat_full[0])),
                dlat_rad=float(np.deg2rad(self.dlat_deg)),
                lon0_rad=float(np.deg2rad(self.lon_full[0])),
                dlon_rad=float(np.deg2rad(self.dlon_deg)),
                n_cushion=self.n_cushion,
            )
        )

    def summary(self) -> str:
        pg = self.propgrid
        nz, ny, nx = self.shape
        return (
            f"velocity interior {nz}x{ny}x{nx} (+cushions {self.shape_full}), spacing "
            f"{self.dx_km:.3f} x {self.dy_km:.3f} x {self.dz_km:.3f} km, depth "
            f"{self.top_depth_km:.2f}..{self.bottom_depth_km:.2f} km; propgrid "
            f"{pg.nr}x{pg.nlat}x{pg.nlon} = {self.n_prop_nodes / 1e6:.1f}M nodes "
            f"(~{self.fm3d_memory_gb:.1f} GB), spacing {pg.dr_km:.3f} km"
        )


def _positions(
    stations: pd.DataFrame, events: pd.DataFrame | None, extent_from: str
) -> tuple[np.ndarray, np.ndarray]:
    lat: list[np.ndarray] = []
    lon: list[np.ndarray] = []
    if extent_from in ("both", "stations"):
        st = stations[stations["used"].astype(bool)] if "used" in stations else stations
        lat.append(st["lat"].to_numpy(float))
        lon.append(st["lon"].to_numpy(float))
    if extent_from in ("both", "events") and events is not None and len(events):
        lat.append(events["lat"].to_numpy(float))
        lon.append(events["lon"].to_numpy(float))
    if not lat or sum(len(x) for x in lat) == 0:
        raise GridError(f"no positions available for extent_from={extent_from!r}")
    return np.concatenate(lat), np.concatenate(lon)


def _event_depth(events: pd.DataFrame) -> np.ndarray:
    col = "depth_km_bsl" if "depth_km_bsl" in events else "depth_km"
    return events[col].to_numpy(float)


def define_grid(
    stations: pd.DataFrame,
    events: pd.DataFrame | None,
    cfg: Any,
    *,
    strict: bool = True,
) -> ModelGrid:
    """Define the FMTOMO velocity and propagation grids.

    Parameters
    ----------
    stations
        Table with ``lat``, ``lon`` (degrees), ``elevation_m`` (positive up) and optionally
        ``used`` (only used stations define the extent).
    events
        Table with ``lat``, ``lon`` and ``depth_km_bsl`` (or ``depth_km``); may be None.
    cfg
        ``Model3DConfig`` (or anything with a ``.model3d`` attribute, e.g. ``OSVConfig``).
    strict
        Raise :class:`GridError` when a station is above the top, an event below the
        bottom, or the grid has more nodes than fm3d can hold; False only warns.

    Returns
    -------
    ModelGrid
    """
    m3 = getattr(cfg, "model3d", cfg)
    g = m3.grid
    refine = int(m3.propgrid_refine)
    warns: list[str] = []

    def warn(msg: str) -> None:
        log.warning(msg)
        warns.append(msg)

    lat_p, lon_p = _positions(stations, events, g.extent_from)
    lat_c = 0.5 * (lat_p.min() + lat_p.max())
    lon_c = 0.5 * (lon_p.min() + lon_p.max())
    if lon_p.max() - lon_p.min() > 180:
        raise GridError("positions span more than 180 degrees of longitude (antimeridian?)")

    dlat = g.dy_km / KM_PER_DEG
    dlon = g.dx_km / (KM_PER_DEG * np.cos(np.deg2rad(lat_c)))
    # padding: at least half a node, so that every station/event is inside the propagation grid
    pad_x = max(g.pad_km, 0.5 * g.dx_km)
    pad_y = max(g.pad_km, 0.5 * g.dy_km)
    if pad_x > g.pad_km or pad_y > g.pad_km:
        warn(
            f"pad_km={g.pad_km:g} is below half a node; using {max(pad_x, pad_y):g} km so that "
            "sources and receivers lie inside the propagation grid"
        )
    pad_lat = pad_y / KM_PER_DEG
    pad_lon = pad_x / (KM_PER_DEG * np.cos(np.deg2rad(lat_c)))
    lat = _axis(lat_c, lat_p.max() - lat_p.min() + 2 * pad_lat, dlat)
    lon = _axis(lon_c, lon_p.max() - lon_p.min() + 2 * pad_lon, dlon)

    # vertical: top above the highest station, bottom at zmax (extended to a whole node)
    elev = stations["elevation_m"].to_numpy(float)
    max_elev_km = float(np.nanmax(elev)) / 1000.0
    top = -(max_elev_km + g.top_margin_km)
    nz = int(np.ceil((g.zmax_km - top) / g.dz_km - 1e-9)) + 1
    depth = top + g.dz_km * np.arange(nz)

    lat_full, lon_full = _extend(lat, dlat), _extend(lon, dlon)
    depth_full = _extend(depth, g.dz_km)

    # propagation grid inside the interior, inset by a fraction of a propagation node
    dz_t, dlat_t, dlon_t = g.dz_km / refine, dlat / refine, dlon / refine
    inset = INSET_FRACTION
    lat_lo, lat_hi = lat[0] + inset * dlat_t, lat[-1] - inset * dlat_t
    lon_lo, lon_hi = lon[0] + inset * dlon_t, lon[-1] - inset * dlon_t
    z_lo, z_hi = depth[0] + inset * dz_t, depth[-1] - inset * dz_t
    dzp, nzp = _prop_axis(z_lo, z_hi, dz_t)
    dlatp, nlatp = _prop_axis(lat_lo, lat_hi, dlat_t)
    dlonp, nlonp = _prop_axis(lon_lo, lon_hi, dlon_t)
    pg = PropGrid(
        nr=nzp,
        nlat=nlatp,
        nlon=nlonp,
        dr_km=dzp,
        dlat_deg=dlatp,
        dlon_deg=dlonp,
        top_height_km=-z_lo,
        lat0_deg=lat_lo,
        lon0_deg=lon_lo,
        refinement_factor=int(m3.source_refine_factor),
        n_refine_cells=int(m3.source_refine_cells),
    )
    grid = ModelGrid(
        lat=lat,
        lon=lon,
        depth=depth,
        lat_full=lat_full,
        lon_full=lon_full,
        depth_full=depth_full,
        dlat_deg=float(dlat),
        dlon_deg=float(dlon),
        dz_km=float(g.dz_km),
        dx_km=float(dlon * KM_PER_DEG * np.cos(np.deg2rad(lat_c))),
        dy_km=float(dlat * KM_PER_DEG),
        propgrid=pg,
        centre_lat=float(lat_c),
        centre_lon=float(lon_c),
        warnings=warns,
    )
    _validate(grid, stations, events, g, strict, warn)
    log.info("FMTOMO grid: %s", grid.summary())
    return grid


def _validate(
    grid: ModelGrid,
    stations: pd.DataFrame,
    events: pd.DataFrame | None,
    g: Any,
    strict: bool,
    warn: Any,
) -> None:
    problems: list[str] = []
    lo, hi = grid.surface_limits_km

    # receivers must lie inside the propagation grid laterally and below the top interface
    st = stations[stations["used"].astype(bool)] if "used" in stations else stations
    inside = grid.contains(
        st["lat"].to_numpy(float), st["lon"].to_numpy(float), np.zeros(len(st)), region="propgrid"
    )
    if not inside.all():
        problems.append(
            f"{int((~inside).sum())} station(s) lie outside the propagation grid laterally "
            "(extent_from='events' or too small pad_km?): "
            + ", ".join(map(str, st.loc[~inside, "sta"].tolist()[:5]) if "sta" in st else [])
        )
    if (-st["elevation_m"].to_numpy(float) / 1000.0 < lo).any():
        problems.append(
            f"stations above the model top ({lo:g} km depth); increase grid.top_margin_km"
        )
    if events is not None and len(events):
        ev_in = grid.contains(
            events["lat"].to_numpy(float),
            events["lon"].to_numpy(float),
            _event_depth(events),
            region="propgrid",
        )
        zev = _event_depth(events)
        if (zev > hi).any():
            problems.append(
                f"{int((zev > hi).sum())} event(s) deeper than the grid bottom ({hi:g} km); "
                f"increase grid.zmax_km (deepest event {zev.max():g} km)"
            )
        if (zev < lo).any():
            problems.append(f"{int((zev < lo).sum())} event(s) above the model top ({lo:g} km)")
        n_lat = int((~ev_in).sum()) - int((zev > hi).sum()) - int((zev < lo).sum())
        if g.extent_from == "stations" and n_lat > 0:
            warn(f"{n_lat} event(s) outside the grid laterally (extent_from='stations')")

    mem = grid.fm3d_memory_gb
    if mem > g.max_memory_gb:
        problems.append(
            f"propagation grid has {grid.n_prop_nodes / 1e6:.0f}M nodes (~{mem:.0f} GB for fm3d, "
            f"limit max_memory_gb={g.max_memory_gb:g}); increase the spacing or lower "
            "propgrid_refine"
        )

    # spacing vs station spacing
    if len(st) > 2:
        la, lo_ = st["lat"].to_numpy(float), st["lon"].to_numpy(float)
        d = spherical_distance_km(la[:, None], lo_[:, None], la[None, :], lo_[None, :])
        d = np.asarray(d)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        med = float(np.median(nn[np.isfinite(nn)]))
        h = max(grid.dx_km, grid.dy_km)
        if med > 0 and h > med / 3.0:
            warn(
                f"node spacing {h:.2f} km exceeds one third of the median nearest-station "
                f"spacing ({med:.2f} km); the data cannot constrain such a grid"
            )

    if problems:
        msg = "FMTOMO grid problems:\n  - " + "\n  - ".join(problems)
        if strict:
            raise GridError(msg)
        warn(msg)
