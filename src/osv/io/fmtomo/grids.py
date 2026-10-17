"""FMTOMO grid files: ``vgrids.in``, ``interfaces.in``, ``propgrid.in``, ``arrtimes.dat``.

Native conventions (verified against the Fortran readers)
---------------------------------------------------------
``vgrids.in`` (read ``3dfmlib.f90:12-86``, written by ``grid3dg.f90:171-461`` and
``invert3d.f90:905-935``)::

    n_layers n_types                       (n_layers = n_interfaces - 1; n_types 1 or 2)
    for each type, for each layer (type slowest):
        nr nlat nlon                       node counts INCLUDING the cushion nodes
        dr dlat dlon                       spacing: km, radians, radians
        r0 lat0 lon0                       origin (lowest, southern, western node): km, rad, rad
        nr*nlat*nlon records "value [uncertainty]"
            radius index slowest (r ascending), latitude next (ascending, south to north),
            longitude fastest (ascending, east)

The second column is the a priori uncertainty (``covvm``, squared inside
``invert3d.f90:568-590``).  ``vgridsref.in`` **must** have it (``invert3d.f90:346``
reads two values per record), ``vgrids.in`` need not (``invert3d.f90:407``).  Blank lines in
the node list are skipped by list-directed input; ``grid3dg`` writes one after every
longitude row and one more after every radius layer (``grid3dg.f90:459-461``).

``interfaces.in`` (``3dfmlib.f90:181-250``): ``n_interfaces``; ``nlat nlon``;
``dlat dlon`` (rad); ``lat0 lon0`` (rad); then per interface ``nlat*nlon`` records
"radius [uncertainty]" with latitude slowest and longitude fastest.  All interfaces share
one lateral grid; interface 1 is the top (surface).  ``fm3d`` modifies crossing interfaces
in memory ("pinching", ``3dfmlib.f90:236-270``) but never rewrites the file.

``propgrid.in`` (``3dfmlib.f90:98-135``): ``nr nlat nlon``; ``dr dlat dlon`` (km, **degrees**,
degrees); ``rtop lat0 lon0`` (the "radius" value is the height of the top node above the
6371 km sphere: ``r0 = R + rtop - (nr-1)*dr``; lat/lon in degrees); ``refinement_factor
n_refine_cells``.

Cushion nodes
-------------
``grid3dg`` adds one node outside the model volume on every side of every axis
(``grid3dg.f90:200-210``, manual p. 12: the B-spline surface spans only ``N-2`` nodes).  The
files do not say how many cushion nodes a grid has, so the datasets carry the attribute
``n_cushion`` (default 1; see :func:`interior`) and the mask helper :func:`cushion_mask`.

In-memory representation
------------------------
Grids are :class:`xarray.Dataset` objects with coordinates ``lat`` and ``lon`` (degrees,
east positive, unwrapped as in the file) and ``depth`` (km below the Earth-radius sphere,
``depth = R - r``, **ascending**, i.e. the first index is the shallowest node; the file stores
radius ascending, so the radius axis is flipped on reading and writing).  Native metadata are
kept in the attributes ``fmtomo_r0_km``, ``fmtomo_dr_km``, ``fmtomo_lat0_rad``,
``fmtomo_dlat_rad``, ``fmtomo_lon0_rad``, ``fmtomo_dlon_rad``, ``fmtomo_earth_radius_km``
and ``n_cushion``.  Writers recompute the geometry from the coordinates and use the native
attributes when they agree (so a read-modify-write cycle is exact, while a dataset whose
coordinates were edited is written consistently with its coordinates).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from osv.coords import FMTOMO_EARTH_RADIUS_KM

from ._fortran import FmtomoFormatError, RecordReader, fint, fnum, read_text, write_text

log = logging.getLogger(__name__)

__all__ = [
    "InterfaceGrids",
    "PropGrid",
    "VelocityGrids",
    "cushion_mask",
    "format_arrtimes",
    "format_interfaces",
    "format_propgrid",
    "format_vgrids",
    "interface_dataset",
    "interior",
    "parse_arrtimes",
    "parse_interfaces",
    "parse_propgrid",
    "parse_vgrids",
    "read_arrtimes",
    "read_interfaces",
    "read_propgrid",
    "read_vgrids",
    "validate_model",
    "velocity_dataset",
    "write_arrtimes",
    "write_interfaces",
    "write_propgrid",
    "write_vgrids",
]

_DEG = np.pi / 180.0


# ---------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Axis:
    n: int
    origin: float
    step: float


def _floats(tokens: list[str]) -> list[float]:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return [float(t.replace("d", "e").replace("D", "e")) for t in tokens]


def _axis_from_coords(
    values: np.ndarray,
    name: str,
    attr_origin: float | None,
    attr_step: float | None,
    scale: float,
    tol: float,
) -> _Axis:
    """Origin/step of a regular ascending axis; prefers the native attributes when consistent."""
    v = np.asarray(values, dtype=float)
    n = int(v.size)
    if n < 1:
        raise FmtomoFormatError(f"axis {name!r} is empty")
    if n == 1:
        if attr_step is None or attr_origin is None:
            raise FmtomoFormatError(
                f"axis {name!r} has a single node: the spacing cannot be derived from the "
                "coordinates; keep the fmtomo_* attributes of the dataset"
            )
        return _Axis(1, float(attr_origin), float(attr_step))
    if np.any(np.diff(v) <= 0):
        raise FmtomoFormatError(f"coordinate {name!r} must be strictly ascending")
    step = (v[-1] - v[0]) / (n - 1)
    if not np.allclose(np.diff(v), step, rtol=1e-6, atol=tol):
        raise FmtomoFormatError(
            f"coordinate {name!r} is not regularly spaced; FMTOMO grids need a constant spacing"
        )
    origin = float(v[0])
    if attr_origin is not None and attr_step is not None:
        o_n, s_n = float(attr_origin), float(attr_step)
        if abs(o_n * scale - origin) <= tol and abs(s_n * scale - step) <= tol:
            return _Axis(n, o_n, s_n)
    return _Axis(n, origin / scale, step / scale)


def _horizontal_axes(ds: xr.Dataset) -> tuple[_Axis, _Axis]:
    a = ds.attrs
    lat = _axis_from_coords(
        ds["lat"].values,
        "lat",
        a.get("fmtomo_lat0_rad"),
        a.get("fmtomo_dlat_rad"),
        1.0 / _DEG,
        1e-9,
    )
    lon = _axis_from_coords(
        ds["lon"].values,
        "lon",
        a.get("fmtomo_lon0_rad"),
        a.get("fmtomo_dlon_rad"),
        1.0 / _DEG,
        1e-9,
    )
    return lat, lon


def _radius_axis(ds: xr.Dataset) -> _Axis:
    a = ds.attrs
    earth = float(a.get("fmtomo_earth_radius_km", FMTOMO_EARTH_RADIUS_KM))
    depth = np.sort(np.asarray(ds["depth"].values, dtype=float))[::-1]
    radius_ascending = earth - depth
    return _axis_from_coords(
        radius_ascending, "depth", a.get("fmtomo_r0_km"), a.get("fmtomo_dr_km"), 1.0, 1e-6
    )


def _hor_attrs(lat: _Axis, lon: _Axis) -> dict[str, float]:
    return {
        "fmtomo_lat0_rad": lat.origin,
        "fmtomo_dlat_rad": lat.step,
        "fmtomo_lon0_rad": lon.origin,
        "fmtomo_dlon_rad": lon.step,
    }


def velocity_dataset(
    velocity: np.ndarray,
    *,
    r0_km: float,
    dr_km: float,
    lat0_rad: float,
    dlat_rad: float,
    lon0_rad: float,
    dlon_rad: float,
    uncertainty: np.ndarray | None = None,
    n_cushion: int = 1,
    earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM,
    layer: int | None = None,
    vtype: int | None = None,
) -> xr.Dataset:
    """Build a velocity-grid dataset from values in **file order** ``(nr, nlat, nlon)``.

    ``velocity[0]`` is the node at radius ``r0_km`` (the deepest); the returned dataset has
    the depth axis flipped so that depth ascends.  Variable ``velocity`` is in km/s and the
    optional ``uncertainty`` (the "covariance" column of the file) in km/s.
    """
    vel = np.asarray(velocity, dtype=float)
    if vel.ndim != 3:
        raise FmtomoFormatError(f"velocity must be 3-D (nr, nlat, nlon), got shape {vel.shape}")
    nr, nlat, nlon = vel.shape
    radius = r0_km + dr_km * np.arange(nr)
    depth = earth_radius_km - radius
    lat = np.rad2deg(lat0_rad + dlat_rad * np.arange(nlat))
    lon = np.rad2deg(lon0_rad + dlon_rad * np.arange(nlon))
    data = {"velocity": (("depth", "lat", "lon"), vel[::-1], {"units": "km/s"})}
    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=float)
        if unc.shape != vel.shape:
            raise FmtomoFormatError("uncertainty must have the same shape as velocity")
        data["uncertainty"] = (
            ("depth", "lat", "lon"),
            unc[::-1],
            {
                "units": "km/s",
                "note": "second column of vgrids.in ('covariance'); squared by invert3d",
            },
        )
    attrs: dict[str, object] = {
        "fmtomo_kind": "velocity_grid",
        "fmtomo_r0_km": float(r0_km),
        "fmtomo_dr_km": float(dr_km),
        "fmtomo_lat0_rad": float(lat0_rad),
        "fmtomo_dlat_rad": float(dlat_rad),
        "fmtomo_lon0_rad": float(lon0_rad),
        "fmtomo_dlon_rad": float(dlon_rad),
        "fmtomo_earth_radius_km": float(earth_radius_km),
        "n_cushion": int(n_cushion),
    }
    if layer is not None:
        attrs["layer"] = int(layer)
    if vtype is not None:
        attrs["vtype"] = int(vtype)
    return xr.Dataset(
        data,
        coords={
            "depth": ("depth", depth[::-1], {"units": "km", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs=attrs,
    )


def interface_dataset(
    radius: np.ndarray,
    *,
    lat0_rad: float,
    dlat_rad: float,
    lon0_rad: float,
    dlon_rad: float,
    uncertainty: np.ndarray | None = None,
    n_cushion: int = 1,
    earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM,
) -> xr.Dataset:
    """Build an interface dataset from radii (km) in file order ``(n_interfaces, nlat, nlon)``.

    Variable ``interface_depth`` (km, positive down) is ``R - radius``; interface 1 is the top.
    """
    rad = np.asarray(radius, dtype=float)
    if rad.ndim != 3:
        raise FmtomoFormatError(f"radius must be 3-D (n_interfaces, nlat, nlon), got {rad.shape}")
    n, nlat, nlon = rad.shape
    data = {
        "interface_depth": (
            ("interface", "lat", "lon"),
            earth_radius_km - rad,
            {"units": "km", "positive": "down"},
        )
    }
    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=float)
        if unc.shape != rad.shape:
            raise FmtomoFormatError("uncertainty must have the same shape as radius")
        data["uncertainty"] = (("interface", "lat", "lon"), unc, {"units": "km"})
    attrs = {
        "fmtomo_kind": "interfaces",
        "fmtomo_lat0_rad": float(lat0_rad),
        "fmtomo_dlat_rad": float(dlat_rad),
        "fmtomo_lon0_rad": float(lon0_rad),
        "fmtomo_dlon_rad": float(dlon_rad),
        "fmtomo_earth_radius_km": float(earth_radius_km),
        "n_cushion": int(n_cushion),
    }
    return xr.Dataset(
        data,
        coords={
            "interface": ("interface", np.arange(1, n + 1)),
            "lat": ("lat", np.rad2deg(lat0_rad + dlat_rad * np.arange(nlat))),
            "lon": ("lon", np.rad2deg(lon0_rad + dlon_rad * np.arange(nlon))),
        },
        attrs=attrs,
    )


def cushion_mask(ds: xr.Dataset) -> xr.DataArray:
    """Boolean mask, True on cushion nodes (``n_cushion`` outermost nodes of each axis)."""
    nc = int(ds.attrs.get("n_cushion", 1))
    dims = [d for d in ("depth", "lat", "lon") if d in ds.dims]
    out = None
    for d in dims:
        n = ds.sizes[d]
        m = np.zeros(n, dtype=bool)
        m[:nc] = True
        m[n - nc :] = True
        da = xr.DataArray(m, dims=[d], coords={d: ds[d]})
        out = da if out is None else (out | da)
    assert out is not None  # at least one of depth/lat/lon present
    return out.transpose(*dims)


def interior(ds: xr.Dataset) -> xr.Dataset:
    """The dataset without its cushion nodes (the region the B-spline surface actually spans)."""
    nc = int(ds.attrs.get("n_cushion", 1))
    sel = {d: slice(nc, ds.sizes[d] - nc) for d in ("depth", "lat", "lon") if d in ds.dims}
    return ds.isel(sel)


# ---------------------------------------------------------------------------------------
# vgrids.in
# ---------------------------------------------------------------------------------------


@dataclass
class VelocityGrids:
    """All velocity grids of a ``vgrids.in`` file.

    ``grids[t][k]`` is the dataset of velocity type ``t`` (0: first type, usually P; 1: second,
    usually S) in layer ``k`` (0 = the top layer, between interface 1 and 2).  Use
    :meth:`get` with FMTOMO's 1-based numbering.
    """

    grids: list[list[xr.Dataset]]

    @property
    def n_layers(self) -> int:
        return len(self.grids[0])

    @property
    def n_types(self) -> int:
        return len(self.grids)

    def get(self, layer: int, vtype: int = 1) -> xr.Dataset:
        """Grid of ``layer`` (1 = top) and velocity type ``vtype`` (1 or 2), FMTOMO numbering."""
        return self.grids[vtype - 1][layer - 1]

    def __iter__(self):  # (layer, vtype, dataset), file order
        for t, row in enumerate(self.grids, start=1):
            for k, ds in enumerate(row, start=1):
                yield k, t, ds


def _read_value_block(rd: RecordReader, n: int, what: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Read ``n`` single-statement value records; returns (values, second column or None)."""
    remaining = rd.remaining()
    if remaining < n:
        raise rd.error(
            f"{what}: the header declares {n} node values but only {remaining} value record(s) "
            "remain in the file (node counts and data disagree)"
        )
    first: list[str] = []
    second: list[str] = []
    for _ in range(n):
        toks = rd.record_tokens(1)
        first.append(toks[0])
        second.append(toks[1] if len(toks) > 1 else "")
    vals = np.array(_floats(first))
    if all(second):
        return vals, np.array(_floats(second))
    if any(second):
        raise rd.error(f"{what}: some records have an uncertainty column and some do not")
    return vals, None


def parse_vgrids(
    text: str,
    *,
    source: str | Path | None = None,
    earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM,
    n_cushion: int = 1,
) -> VelocityGrids:
    """Parse the text of a ``vgrids.in`` / ``vgridsref.in`` / ``vgridvpvs.in`` file."""
    rd = RecordReader(text, source)
    n_layers, n_types = rd.ints(2)
    if n_layers < 1 or n_types not in (1, 2):
        raise rd.error(
            f"invalid header: {n_layers} layers, {n_types} velocity types (need >=1 and 1 or 2)"
        )
    grids: list[list[xr.Dataset]] = []
    for t in range(1, n_types + 1):
        row = []
        for k in range(1, n_layers + 1):
            nr, nlat, nlon = rd.ints(3)
            if min(nr, nlat, nlon) < 1:
                raise rd.error(f"layer {k} type {t}: non-positive node count {nr}x{nlat}x{nlon}")
            dr, dlat, dlon = rd.floats(3)
            r0, lat0, lon0 = rd.floats(3)
            what = f"layer {k} type {t} ({nr}x{nlat}x{nlon} = {nr * nlat * nlon} nodes)"
            vals, unc = _read_value_block(rd, nr * nlat * nlon, what)
            shape = (nr, nlat, nlon)
            row.append(
                velocity_dataset(
                    vals.reshape(shape),
                    r0_km=r0,
                    dr_km=dr,
                    lat0_rad=lat0,
                    dlat_rad=dlat,
                    lon0_rad=lon0,
                    dlon_rad=dlon,
                    uncertainty=None if unc is None else unc.reshape(shape),
                    n_cushion=n_cushion,
                    earth_radius_km=earth_radius_km,
                    layer=k,
                    vtype=t,
                )
            )
        grids.append(row)
    rd.expect_eof("the last velocity grid (more values than the header node counts)")
    return VelocityGrids(grids)


def read_vgrids(path: str | Path, **kwargs: object) -> VelocityGrids:
    """Read a ``vgrids.in``-format file."""
    return parse_vgrids(read_text(path), source=path, **kwargs)  # type: ignore[arg-type]


def _geometry_3d(ds: xr.Dataset, what: str) -> tuple[_Axis, _Axis, _Axis]:
    if "velocity" not in ds:
        raise FmtomoFormatError(f"{what}: dataset has no 'velocity' variable")
    for d in ("depth", "lat", "lon"):
        if d not in ds.dims:
            raise FmtomoFormatError(f"{what}: dataset has no {d!r} dimension")
    lat, lon = _horizontal_axes(ds)
    return _radius_axis(ds), lat, lon


def _file_order(da: xr.DataArray) -> np.ndarray:
    """Array of a (depth, lat, lon) variable in file order: radius ascending (depth descending)."""
    da = (
        da.transpose("depth", "lat", "lon")
        .sortby("depth", ascending=False)
        .sortby("lat")
        .sortby("lon")
    )
    return np.asarray(da.values, dtype=float)


def format_vgrids(
    grids: VelocityGrids, *, precision: int = 8, uncertainty_default: float | None = None
) -> str:
    """Format as ``vgrids.in`` (``grid3dg.f90`` layout: ``(2f12.8)`` nodes, blank separators).

    The uncertainty column is written when a grid has an ``uncertainty`` variable (or
    ``uncertainty_default`` is given); a file mixing grids with and without it is rejected
    because ``invert3d`` reads every grid of a file the same way.  Values are rounded to
    ``precision`` decimals as in the Fortran format.
    """
    n_types = grids.n_types
    n_layers = grids.n_layers
    if n_types not in (1, 2) or any(len(r) != n_layers for r in grids.grids):
        raise FmtomoFormatError("need 1 or 2 velocity types, each with the same number of layers")
    has_unc = [("uncertainty" in ds) or uncertainty_default is not None for _, _, ds in grids]
    if any(has_unc) and not all(has_unc):
        raise FmtomoFormatError(
            "some grids have an uncertainty variable and some do not; give every grid one "
            "(or pass uncertainty_default)"
        )
    out = [fint(n_layers) + fint(n_types)]
    for layer, vtype, ds in grids:
        what = f"layer {layer} type {vtype}"
        rad, lat, lon = _geometry_3d(ds, what)
        out.append(fint(rad.n) + fint(lat.n) + fint(lon.n))
        out.append("  ".join(fnum(x) for x in (rad.step, lat.step, lon.step)))
        out.append("  ".join(fnum(x) for x in (rad.origin, lat.origin, lon.origin)))
        vel = _file_order(ds["velocity"])
        unc = _file_order(ds["uncertainty"]) if "uncertainty" in ds else None
        for i in range(rad.n):
            for j in range(lat.n):
                for k in range(lon.n):
                    s = f"{vel[i, j, k]:12.{precision}f}"
                    if unc is not None:
                        s += f"{unc[i, j, k]:12.{precision}f}"
                    elif uncertainty_default is not None:
                        s += f"{uncertainty_default:12.{precision}f}"
                    out.append(s)
                out.append("")
            out.append("")
    return "\n".join(out) + "\n"


def write_vgrids(path: str | Path, grids: VelocityGrids, **kwargs: object) -> Path:
    """Write a ``vgrids.in``-format file."""
    return write_text(path, format_vgrids(grids, **kwargs))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# interfaces.in
# ---------------------------------------------------------------------------------------


@dataclass
class InterfaceGrids:
    """All interfaces of an ``interfaces.in`` file as one dataset (dims ``interface, lat, lon``)."""

    data: xr.Dataset

    @property
    def n_interfaces(self) -> int:
        return int(self.data.sizes["interface"])


def parse_interfaces(
    text: str,
    *,
    source: str | Path | None = None,
    earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM,
    n_cushion: int = 1,
) -> InterfaceGrids:
    """Parse the text of an ``interfaces.in`` / ``interfacesref.in`` file."""
    rd = RecordReader(text, source)
    n = rd.int()
    nlat, nlon = rd.ints(2)
    dlat, dlon = rd.floats(2)
    lat0, lon0 = rd.floats(2)
    if n < 2:
        raise rd.error(f"{n} interface(s): fm3d needs at least a top and a bottom interface")
    if min(nlat, nlon) < 1:
        raise rd.error(f"non-positive node count {nlat}x{nlon}")
    rad = np.empty((n, nlat, nlon))
    unc: np.ndarray | None = np.empty((n, nlat, nlon))
    for i in range(n):
        vals, u = _read_value_block(rd, nlat * nlon, f"interface {i + 1} ({nlat}x{nlon} nodes)")
        rad[i] = vals.reshape(nlat, nlon)
        if u is None:
            unc = None
        elif unc is not None:
            unc[i] = u.reshape(nlat, nlon)
    rd.expect_eof("the last interface (more values than the header node counts)")
    ds = interface_dataset(
        rad,
        lat0_rad=lat0,
        dlat_rad=dlat,
        lon0_rad=lon0,
        dlon_rad=dlon,
        uncertainty=unc,
        n_cushion=n_cushion,
        earth_radius_km=earth_radius_km,
    )
    return InterfaceGrids(ds)


def read_interfaces(path: str | Path, **kwargs: object) -> InterfaceGrids:
    """Read an ``interfaces.in``-format file."""
    return parse_interfaces(read_text(path), source=path, **kwargs)  # type: ignore[arg-type]


def format_interfaces(
    interfaces: InterfaceGrids, *, precision: int = 5, uncertainty_default: float | None = None
) -> str:
    """Format as ``interfaces.in`` (``grid3dg.f90:640-666`` layout, ``(2f12.5)`` nodes)."""
    ds = interfaces.data
    if "interface_depth" not in ds:
        raise FmtomoFormatError("dataset has no 'interface_depth' variable")
    earth = float(ds.attrs.get("fmtomo_earth_radius_km", FMTOMO_EARTH_RADIUS_KM))
    lat, lon = _horizontal_axes(ds)
    n = ds.sizes["interface"]
    dep = (
        ds["interface_depth"]
        .transpose("interface", "lat", "lon")
        .sortby("interface")
        .sortby("lat")
        .sortby("lon")
    )
    radius = earth - np.asarray(dep.values, dtype=float)
    unc = None
    if "uncertainty" in ds:
        unc = np.asarray(
            ds["uncertainty"]
            .transpose("interface", "lat", "lon")
            .sortby("interface")
            .sortby("lat")
            .sortby("lon")
            .values
        )
    out = [fint(n), fint(lat.n) + fint(lon.n)]
    out.append("  ".join(fnum(x) for x in (lat.step, lon.step)))
    out.append("  ".join(fnum(x) for x in (lat.origin, lon.origin)))
    for i in range(n):
        for j in range(lat.n):
            for k in range(lon.n):
                s = f"{radius[i, j, k]:12.{precision}f}"
                if unc is not None:
                    s += f"{unc[i, j, k]:12.{precision}f}"
                elif uncertainty_default is not None:
                    s += f"{uncertainty_default:12.{precision}f}"
                out.append(s)
            out.append("")
        out.append("")
    return "\n".join(out) + "\n"


def write_interfaces(path: str | Path, interfaces: InterfaceGrids, **kwargs: object) -> Path:
    """Write an ``interfaces.in``-format file."""
    return write_text(path, format_interfaces(interfaces, **kwargs))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# propgrid.in
# ---------------------------------------------------------------------------------------


@dataclass
class PropGrid:
    """The propagation grid of ``propgrid.in`` (native units of the file).

    ``top_height_km`` is the file's "radius origin": the height of the *top* node above the
    Earth-radius sphere (negative = below), exactly as ``fm3d`` forms ``r0 = R + top -
    (nr-1) dr`` (``3dfmlib.f90:115``).  ``lat0_deg``/``lon0_deg`` are the south-west node and
    ``dlat_deg``/``dlon_deg`` the (positive) spacings, in degrees.
    """

    nr: int
    nlat: int
    nlon: int
    dr_km: float
    dlat_deg: float
    dlon_deg: float
    top_height_km: float
    lat0_deg: float
    lon0_deg: float
    refinement_factor: int
    n_refine_cells: int
    earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM

    @property
    def depth_km(self) -> np.ndarray:
        """Node depths (km, positive down), ascending (shallowest first)."""
        return -self.top_height_km + self.dr_km * np.arange(self.nr)

    @property
    def radius_km(self) -> np.ndarray:
        """Node radii in km, ascending (the order ``fm3d`` uses)."""
        return self.earth_radius_km - self.depth_km[::-1]

    @property
    def lat(self) -> np.ndarray:
        return self.lat0_deg + self.dlat_deg * np.arange(self.nlat)

    @property
    def lon(self) -> np.ndarray:
        return self.lon0_deg + self.dlon_deg * np.arange(self.nlon)

    def to_dataset(self) -> xr.Dataset:
        """A coordinate-only dataset (``depth``, ``lat``, ``lon``) of the propagation nodes."""
        return xr.Dataset(
            coords={"depth": self.depth_km, "lat": self.lat, "lon": self.lon},
            attrs={
                "fmtomo_kind": "propgrid",
                "refinement_factor": self.refinement_factor,
                "n_refine_cells": self.n_refine_cells,
                "fmtomo_earth_radius_km": self.earth_radius_km,
            },
        )

    def __post_init__(self) -> None:
        if min(self.nr, self.nlat, self.nlon) < 2:
            raise FmtomoFormatError(
                f"propagation grid needs at least 2 nodes per axis, "
                f"got {self.nr}x{self.nlat}x{self.nlon}"
            )
        if min(self.dr_km, self.dlat_deg, self.dlon_deg) <= 0:
            raise FmtomoFormatError("propagation grid spacings must be positive")


def parse_propgrid(text: str, *, source: str | Path | None = None) -> PropGrid:
    """Parse the text of ``propgrid.in``."""
    rd = RecordReader(text, source)
    nr, nlat, nlon = rd.ints(3)
    dr, dlat, dlon = rd.floats(3)
    top, lat0, lon0 = rd.floats(3)
    rf, nc = rd.ints(2)
    try:
        return PropGrid(nr, nlat, nlon, dr, dlat, dlon, top, lat0, lon0, rf, nc)
    except FmtomoFormatError as exc:
        raise FmtomoFormatError(f"{rd.source}: {exc}") from None


def read_propgrid(path: str | Path) -> PropGrid:
    """Read ``propgrid.in``."""
    return parse_propgrid(read_text(path), source=path)


def format_propgrid(pg: PropGrid) -> str:
    """Format as ``propgrid.in`` (``grid3dg.f90:153-162``)."""
    return (
        "\n".join(
            [
                fint(pg.nr) + fint(pg.nlat) + fint(pg.nlon),
                "  ".join(fnum(x) for x in (pg.dr_km, pg.dlat_deg, pg.dlon_deg)),
                "  ".join(fnum(x) for x in (pg.top_height_km, pg.lat0_deg, pg.lon0_deg)),
                fint(pg.refinement_factor) + fint(pg.n_refine_cells),
            ]
        )
        + "\n"
    )


def write_propgrid(path: str | Path, pg: PropGrid) -> Path:
    """Write ``propgrid.in``."""
    return write_text(path, format_propgrid(pg))


# ---------------------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------------------


def validate_model(
    vgrids: VelocityGrids | None = None,
    interfaces: InterfaceGrids | None = None,
    propgrid: PropGrid | None = None,
) -> None:
    """Check the cross-file constraints ``fm3d`` and the tools assume; raise one error listing all.

    * ``n_layers == n_interfaces - 1`` (``3dfm_main.f90`` pairs layer ``k`` with interfaces
      ``k``/``k+1``);
    * the **interiors** (without cushion nodes) of all velocity grids and of the interface grid
      span the same lateral box: ``grid3dg`` writes them from one extent, but the cushion
      nodes lie one layer-specific node spacing outside it (``grid3dg.f90:200-210``), so grids of
      different density have different total extents (example 2).  Node counts may differ
      between layers and between the two velocity types;
    * the propagation grid lies inside the second layer of interface nodes
      (``README_0.7``, interface section) and between the top and bottom interfaces
      (``3dfmlib.f90:240-241``).
    """
    problems: list[str] = []
    ref_box: tuple[float, float, float, float] | None = None

    def box(ds: xr.Dataset) -> tuple[float, float, float, float]:
        # the cushion nodes sit one (layer specific) node spacing outside the model volume, so the
        # grids only share the extent of their interior (grid3dg.f90:200-210)
        inner = interior(ds)
        return (
            float(inner["lat"].min()),
            float(inner["lat"].max()),
            float(inner["lon"].min()),
            float(inner["lon"].max()),
        )

    if interfaces is not None:
        ref_box = box(interfaces.data)
    if vgrids is not None:
        if interfaces is not None and vgrids.n_layers != interfaces.n_interfaces - 1:
            problems.append(
                f"{vgrids.n_layers} velocity layers but {interfaces.n_interfaces} interfaces "
                "(need n_interfaces - 1 layers)"
            )
        for layer, vtype, ds in vgrids:
            b = box(ds)
            if ref_box is None:
                ref_box = b
            elif not np.allclose(b, ref_box, atol=1e-6):
                problems.append(
                    f"layer {layer} type {vtype}: interior lateral extent {b} differs from "
                    f"{ref_box}"
                )
    if propgrid is not None and interfaces is not None:
        ds = interfaces.data
        lats, lons = ds["lat"].values, ds["lon"].values
        if len(lats) > 2 and (propgrid.lat.min() <= lats[1] or propgrid.lat.max() >= lats[-2]):
            problems.append(
                "propagation grid latitude range is not inside the second ring of interface nodes"
            )
        if len(lons) > 2 and (propgrid.lon.min() <= lons[1] or propgrid.lon.max() >= lons[-2]):
            problems.append(
                "propagation grid longitude range is not inside the second ring of interface nodes"
            )
        dep = ds["interface_depth"]
        top_depth = float(dep.sel(interface=1).max())
        bottom_depth = float(dep.sel(interface=int(ds.sizes["interface"])).min())
        pg_top, pg_bottom = float(propgrid.depth_km[0]), float(propgrid.depth_km[-1])
        if top_depth < pg_top - 1e-9:
            problems.append(
                f"top interface reaches {top_depth:g} km depth, above the propagation grid top "
                f"({pg_top:g} km)"
            )
        if bottom_depth > pg_bottom + 1e-9:
            problems.append(
                f"bottom interface reaches {bottom_depth:g} km depth, below the propagation grid "
                f"bottom ({pg_bottom:g} km)"
            )
    if problems:
        raise FmtomoFormatError("inconsistent FMTOMO model:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------------------
# arrtimes.dat  (fm3d output, save_timefields_mode)
# ---------------------------------------------------------------------------------------


def parse_arrtimes(text: str, *, source: str | Path | None = None) -> xr.Dataset:
    """Parse ``arrtimes.dat`` (``3dfm_main.f90:737-741``, ``3dfmlib.f90:5038-5044``).

    Header: ``nr nlat nlon``; ``dr dlat dlon`` (km, degrees, degrees); ``r0 lat0 lon0`` (km radius
    of the deepest node, degrees); ``n_fields``.  Per field: ``source path first_tf`` and then
    ``nr*nlat*nlon`` times (``f12.5``; ``-1.0`` marks a node never reached -> NaN here) with
    the radius index fastest and longitude slowest.  Returns a dataset ``arrival_time``
    with dims ``(field, depth, lat, lon)`` (depth ascending) and per-field coordinates
    ``source``, ``path`` and ``first_timefield``.
    """
    rd = RecordReader(text, source)
    nr, nlat, nlon = rd.ints(3)
    dr, dlat, dlon = rd.floats(3)
    r0, lat0, lon0 = rd.floats(3)
    nf = rd.int()
    n = nr * nlat * nlon
    if rd.remaining() < nf * (1 + n):
        raise rd.error(
            f"header declares {nf} time fields of {nr}x{nlat}x{nlon} nodes but the file has too "
            "few records"
        )
    times = np.empty((nf, nlon, nlat, nr))
    ids = np.empty((nf, 3), dtype=int)
    for f in range(nf):
        ids[f] = rd.ints(3)
        times[f] = np.array(_floats([rd.record_tokens(1)[0] for _ in range(n)])).reshape(
            nlon, nlat, nr
        )
    rd.expect_eof("the last time field")
    times[times == -1.0] = np.nan
    arr = times.transpose(0, 3, 2, 1)  # field, r, lat, lon
    radius = r0 + dr * np.arange(nr)
    depth = FMTOMO_EARTH_RADIUS_KM - radius
    return xr.Dataset(
        {"arrival_time": (("field", "depth", "lat", "lon"), arr[:, ::-1], {"units": "s"})},
        coords={
            "field": np.arange(1, nf + 1),
            "depth": depth[::-1],
            "lat": lat0 + dlat * np.arange(nlat),
            "lon": lon0 + dlon * np.arange(nlon),
            "source": ("field", ids[:, 0]),
            "path": ("field", ids[:, 1]),
            "first_timefield": ("field", ids[:, 2]),
        },
        attrs={"fmtomo_kind": "arrtimes", "fmtomo_r0_km": r0, "fmtomo_dr_km": dr},
    )


def read_arrtimes(path: str | Path) -> xr.Dataset:
    """Read ``arrtimes.dat``."""
    return parse_arrtimes(read_text(path), source=path)


def format_arrtimes(ds: xr.Dataset) -> str:
    """Format a dataset from :func:`parse_arrtimes` as ``arrtimes.dat`` (NaN written as -1)."""
    da = (
        ds["arrival_time"]
        .transpose("field", "depth", "lat", "lon")
        .sortby("depth", ascending=False)
    )
    nf, nr, nlat, nlon = da.shape
    r0 = float(ds.attrs["fmtomo_r0_km"])
    dr = float(ds.attrs["fmtomo_dr_km"])
    lat = ds["lat"].values
    lon = ds["lon"].values
    dlat = (lat[-1] - lat[0]) / (nlat - 1)
    dlon = (lon[-1] - lon[0]) / (nlon - 1)
    out = [
        fint(nr) + fint(nlat) + fint(nlon),
        "  ".join(fnum(x) for x in (dr, dlat, dlon)),
        "  ".join(fnum(x) for x in (r0, lat[0], lon[0])),
        fint(nf),
    ]
    vals = np.nan_to_num(np.asarray(da.values, dtype=float), nan=-1.0)
    for f in range(nf):
        out.append(
            fint(int(ds["source"].values[f]))
            + fint(int(ds["path"].values[f]))
            + fint(int(ds["first_timefield"].values[f]))
        )
        out.extend(f"{v:12.5f}" for v in vals[f].transpose(2, 1, 0).ravel())
    return "\n".join(out) + "\n"


def write_arrtimes(path: str | Path, ds: xr.Dataset) -> Path:
    """Write ``arrtimes.dat``."""
    return write_text(path, format_arrtimes(ds))
