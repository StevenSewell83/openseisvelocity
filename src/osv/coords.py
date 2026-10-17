"""Coordinate conversions (local projection, VELEST sign convention, FMTOMO spherical).

Canonical convention used everywhere inside osv (tables, config, in-memory objects):

* latitude/longitude in decimal degrees, WGS84, **north and east positive**;
* depth in km, **positive down** (below sea level unless stated otherwise);
* elevation in metres, **positive up** (above sea level).

Only the readers/writers in ``osv.io`` convert to program specific conventions, and they
must do so with the helpers in this module.  The conventions of each program, with source
references, are tabulated in ``docs/conventions.md``.

Summary of the program conventions implemented here (verified from the vendored sources):

* VELEST (``velest.f``): file lat/lon carry N/S and E/W letters (``INPUTCNV``, station
  reader); the control-file ``olat`` is north positive, ``olon`` is **west positive**
  (so an east longitude must be written negated); cartesian x is positive WEST, y positive
  north, z positive down; station elevation is an integer in metres (positive up) and the
  station z is ``-elev/1000`` km.
* FMTOMO (``fm3d``): spherical coordinates, Earth radius 6371 km; ``sources.in`` and
  ``receivers.in`` hold ``depth lat lon`` (depth km positive down, negative above the
  surface, lat/lon in degrees, east longitude); ``vgrids.in``/``interfaces.in`` use radius
  (km) with latitude and longitude in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Geod, Transformer

__all__ = [
    "FMTOMO_EARTH_RADIUS_KM",
    "LocalProjection",
    "VelestShortDistance",
    "azimuth_deg",
    "backazimuth_deg",
    "borehole_sensor_elevation_m",
    "depth_km_to_elevation_m",
    "depth_to_radius",
    "elevation_m_to_depth_km",
    "epicentral_distance_km",
    "fmtomo_lat_lon_to_radians",
    "fmtomo_receiver_depth_km",
    "format_velest_lat",
    "format_velest_lon",
    "hypocentral_distance_km",
    "legacy_mirror",
    "parse_velest_lat",
    "parse_velest_lon",
    "radius_to_depth",
    "spherical_distance_km",
    "to_sea_level_depth",
    "velest_depth_from_file",
    "velest_depth_to_file",
    "velest_elevation_field",
    "velest_olat",
    "velest_olon",
    "velest_station_z_km",
    "velest_zshift_for_reference_elevation",
    "wrap_lon",
]

#: Earth radius used by FMTOMO (``mod_3dfm.f90:709``, ``make_vgrids.f90:36``, manual p. 15).
FMTOMO_EARTH_RADIUS_KM = 6371.0

_GEOD = Geod(ellps="WGS84")

Array = NDArray[np.float64]


def _arr(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def _out(a: Array, scalar: bool) -> float | Array:
    """Return a python float for scalar input, an array otherwise."""
    return float(a) if scalar else a


# ----------------------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------------------


def wrap_lon(lon: ArrayLike, convention: Literal["180", "360"] = "180") -> float | Array:
    """Wrap longitudes into [-180, 180) (``"180"``) or [0, 360) (``"360"``) degrees east."""
    a = _arr(lon)
    scalar = a.ndim == 0
    if convention == "180":
        w = (a + 180.0) % 360.0 - 180.0
    elif convention == "360":
        w = a % 360.0
    else:
        raise ValueError(f"convention must be '180' or '360', got {convention!r}")
    return _out(w, scalar)


def elevation_m_to_depth_km(elevation_m: ArrayLike) -> float | Array:
    """Convert elevation (m, positive up) to depth below sea level (km, positive down)."""
    a = _arr(elevation_m)
    return _out(-a / 1000.0, a.ndim == 0)


def depth_km_to_elevation_m(depth_km: ArrayLike) -> float | Array:
    """Convert depth below sea level (km, positive down) to elevation (m, positive up)."""
    a = _arr(depth_km)
    return _out(-a * 1000.0, a.ndim == 0)


DepthDatum = Literal["sea_level", "surface", "reference"]


def to_sea_level_depth(
    depth_km: ArrayLike,
    datum: DepthDatum = "sea_level",
    reference_elevation_m: ArrayLike | None = None,
    surface_elevation_m: ArrayLike | None = None,
) -> float | Array:
    """Convert depths given relative to ``datum`` to km below sea level (positive down).

    Parameters
    ----------
    depth_km
        Depth(s) below the datum, km, positive down.
    datum
        ``"sea_level"``: no change; ``"surface"``: depth below the local ground surface
        (needs ``surface_elevation_m``, one value per depth or a scalar);
        ``"reference"``: depth below the fixed elevation ``reference_elevation_m``.
    reference_elevation_m, surface_elevation_m
        Elevation(s) in metres above sea level (positive up) of the datum.

    Returns
    -------
    float or ndarray
        ``depth_km - elevation_m / 1000`` (negative means above sea level).  NaN
        surface elevations give NaN depths.
    """
    a = _arr(depth_km)
    if datum == "sea_level":
        return _out(a.copy(), a.ndim == 0)
    if datum == "reference":
        if reference_elevation_m is None:
            raise ValueError("reference_elevation_m is required for datum 'reference'")
        elev = _arr(reference_elevation_m)
    elif datum == "surface":
        if surface_elevation_m is None:
            raise ValueError(
                "surface_elevation_m is required for datum 'surface' "
                "(interpolate station elevations or use a DEM)"
            )
        elev = _arr(surface_elevation_m)
    else:
        raise ValueError(f"unknown depth datum {datum!r}; use sea_level, surface or reference")
    out = a - elev / 1000.0
    return _out(out, out.ndim == 0)


def borehole_sensor_elevation_m(
    surface_elevation_m: ArrayLike, sensor_depth_m: ArrayLike = 0.0
) -> float | Array:
    """Absolute elevation (m above sea level) of a borehole sensor.

    Parameters
    ----------
    surface_elevation_m
        Elevation of the ground surface at the station (StationXML ``Elevation`` of the
        station/channel, metres, positive up).
    sensor_depth_m
        Depth of the sensor below the surface (StationXML channel ``Depth``, metres,
        positive down; 0 for a surface instrument).

    Returns
    -------
    float or ndarray
        ``surface_elevation_m - sensor_depth_m``.  Use :func:`elevation_m_to_depth_km` for
        the depth below sea level.
    """
    s = _arr(surface_elevation_m)
    d = _arr(sensor_depth_m)
    return _out(s - d, (s - d).ndim == 0)


# ----------------------------------------------------------------------------------------
# Local projection
# ----------------------------------------------------------------------------------------


class LocalProjection:
    """Map lat/lon (degrees, east positive) to local x (east) / y (north) in km.

    Parameters
    ----------
    origin_lat, origin_lon
        Origin of the local system, degrees (east positive).  Maps to ``(0, 0)``.
    method
        ``"local_tm"`` (default): transverse Mercator on WGS84 centred on the origin
        (``k0 = 1``), distances accurate to well below 1 m within ~30 km.  Any other
        string must be an EPSG code such as ``"EPSG:2193"`` (or just ``"2193"``); the
        projected coordinates are shifted so the origin maps to ``(0, 0)``.

    Notes
    -----
    All methods are vectorised over numpy arrays (and accept scalars).
    """

    def __init__(self, origin_lat: float, origin_lon: float, method: str = "local_tm") -> None:
        if not -90.0 <= origin_lat <= 90.0:
            raise ValueError(f"origin_lat must be within [-90, 90], got {origin_lat}")
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(wrap_lon(origin_lon, "180"))
        self.method = method
        geo = CRS.from_epsg(4326)
        if method == "local_tm":
            proj = CRS.from_proj4(
                f"+proj=tmerc +lat_0={self.origin_lat!r} +lon_0={self.origin_lon!r} "
                "+k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
            )
            self._x0 = self._y0 = 0.0
        else:
            code = method if ":" in method else f"EPSG:{method}"
            try:
                proj = CRS.from_user_input(code)
            except Exception as exc:  # pyproj raises CRSError
                raise ValueError(
                    f"method must be 'local_tm' or an EPSG code (e.g. 'EPSG:2193'), got "
                    f"{method!r}: {exc}"
                ) from exc
            if not proj.is_projected:
                raise ValueError(f"{method!r} is not a projected CRS; choose a projected EPSG")
            self._x0 = self._y0 = 0.0
        self.crs = proj
        self._fwd = Transformer.from_crs(geo, proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, geo, always_xy=True)
        if method != "local_tm":
            self._x0, self._y0 = self._fwd.transform(self.origin_lon, self.origin_lat)

    def forward(self, lat: ArrayLike, lon: ArrayLike) -> tuple[float | Array, float | Array]:
        """Return ``(x_km, y_km)`` (east, north) for latitude/longitude in degrees."""
        la, lo = np.broadcast_arrays(_arr(lat), _arr(lon))
        scalar = la.ndim == 0
        x, y = self._fwd.transform(lo, la)
        x = (np.asarray(x) - self._x0) / 1000.0
        y = (np.asarray(y) - self._y0) / 1000.0
        return _out(x, scalar), _out(y, scalar)

    def inverse(self, x_km: ArrayLike, y_km: ArrayLike) -> tuple[float | Array, float | Array]:
        """Return ``(lat, lon)`` in degrees for local ``x`` (east), ``y`` (north) in km."""
        xa, ya = np.broadcast_arrays(_arr(x_km), _arr(y_km))
        scalar = xa.ndim == 0
        lon, lat = self._inv.transform(xa * 1000.0 + self._x0, ya * 1000.0 + self._y0)
        return _out(np.asarray(lat), scalar), _out(np.asarray(lon), scalar)

    def __repr__(self) -> str:
        return (
            f"LocalProjection(origin_lat={self.origin_lat}, origin_lon={self.origin_lon}, "
            f"method={self.method!r})"
        )


# ----------------------------------------------------------------------------------------
# VELEST
# ----------------------------------------------------------------------------------------

# Constants of SETORG (velest.f:10123-10240). Note the truncated value of ``rad``.
_V_REARTH = 6378.135  # km, WGS72 equatorial radius (velest.f:10168)
_V_ELLIP = 298.26  # 1/flattening (velest.f:10169)
_V_RAD = 0.017453292  # velest.f:10158 (pi/180 truncated; kept for fidelity)


def velest_olat(lat: float) -> float:
    """Control-file ``olat`` for a true latitude: north positive, south negative.

    ``velest.f:1248-1251``: "For LAT N(orth) and LON W(est) their values are positive.
    For LAT S(outh) and LON E(ast) their values must be negative".
    """
    return float(lat)


def velest_olon(lon_east: float) -> float:
    """Control-file ``olon`` for a true (east positive) longitude: **west positive**.

    ``velest.f:1259-1263`` prints ``olon <= 0`` as an E longitude, so an east longitude of
    176.194 must be written as ``-176.194``.
    """
    return -float(wrap_lon(lon_east, "180"))


def format_velest_lat(lat: float) -> str:
    """Format a latitude as the VELEST ``f7.4,a1`` field, e.g. ``-38.5196 -> '38.5196S'``.

    The latitude must satisfy ``|lat| < 100`` (7 character field).  The result is exactly
    8 characters.
    """
    letter = "S" if lat < 0 else "N"
    s = f"{abs(lat):7.4f}"
    if len(s) > 7:
        raise ValueError(f"latitude {lat} does not fit VELEST format f7.4")
    return s + letter


def format_velest_lon(lon_east: float) -> str:
    """Format an east-positive longitude as the VELEST ``f8.4,a1`` field.

    East longitudes get the letter ``E`` and west longitudes ``W``: ``176.1624 ->
    '176.1624E'`` and ``7.4396 -> '  7.4396E'`` (9 characters, ``f8.4`` right aligned).
    VELEST (``INPUTCNV``, ``velest.f:3459-3460``; station reader ``velest.f:1538-1539``)
    negates the value for ``E`` internally, so true longitudes need no sign flip in the
    files; only the control-file ``olon`` does (:func:`velest_olon`).
    """
    lon = float(wrap_lon(lon_east, "180"))
    letter = "W" if lon < 0 else "E"
    return f"{abs(lon):8.4f}" + letter


def parse_velest_lat(text: str) -> float:
    """Parse a VELEST latitude field such as ``'38.5196S'`` to a signed latitude.

    Letters are case-insensitive here (VELEST itself upper-cases station letters but not
    CNV letters, so writers must emit upper case).
    """
    t = text.strip()
    if not t or t[-1].upper() not in "NS":
        raise ValueError(f"VELEST latitude must end in N or S, got {text!r}")
    v = float(t[:-1])
    return -v if t[-1].upper() == "S" else v


def parse_velest_lon(text: str) -> float:
    """Parse a VELEST longitude field such as ``'176.1624E'`` to a signed east longitude."""
    t = text.strip()
    if not t or t[-1].upper() not in "EW":
        raise ValueError(f"VELEST longitude must end in E or W, got {text!r}")
    v = float(t[:-1])
    return -v if t[-1].upper() == "W" else v


def legacy_mirror(lat: ArrayLike, lon: ArrayLike) -> tuple[float | Array, float | Array]:
    """Map between the legacy mirrored NZ files and true coordinates (an involution).

    The legacy Ngatamariki files (``MonteCarlo/nmstats.sta``, ``*.cnv``, ``velest.cmn``)
    write ``38.5196N 176.1624W`` for a site that really is ``38.5196 S, 176.1624 E``, and
    use ``olat = 38.55, olon = 176.194``.  Both axes are mirrored (point reflection
    through the origin of lat/lon).  VELEST only sees differences to the origin, and its
    short-distance conversion is symmetric under this reflection, so interstation
    distances are identical (to ~1e-5 relative, because ``aa`` is a one-arcminute finite
    difference towards +lat) and x/y are negated (``x' = -x, y' = -y``, a 180 degree
    rotation).  Pass signed values parsed with
    :func:`parse_velest_lat`/:func:`parse_velest_lon` to get true coordinates, and vice
    versa.
    """
    la, lo = _arr(lat), _arr(lon)
    return _out(-la, la.ndim == 0), _out(-lo, lo.ndim == 0)


def velest_elevation_field(elevation_m: float) -> str:
    """Station-file elevation field (``i4``): integer metres, positive up.

    Raises ``ValueError`` when the rounded value does not fit four characters
    (-999 .. 9999 m).
    """
    v = int(round(float(elevation_m)))
    if not -999 <= v <= 9999:
        raise ValueError(
            f"station elevation {elevation_m} m does not fit VELEST i4 field (-999..9999 m)"
        )
    return f"{v:4d}"


def velest_station_z_km(elevation_m: ArrayLike) -> float | Array:
    """Station z (km, positive down) as VELEST computes it: ``-ielev/1000`` (``velest.f:1551``).

    Only used with ``iuseelev = 1``; with ``iuseelev = 0`` VELEST sets z to 0.
    """
    return elevation_m_to_depth_km(np.rint(_arr(elevation_m)))


def velest_depth_from_file(depth_km: ArrayLike, zshift_km: float = 0.0) -> float | Array:
    """Internal VELEST depth for a CNV depth: ``depth + zshift`` (``velest.f:1865``)."""
    a = _arr(depth_km)
    return _out(a + zshift_km, a.ndim == 0)


def velest_depth_to_file(depth_km: ArrayLike, zshift_km: float = 0.0) -> float | Array:
    """CNV depth that VELEST will interpret as ``depth_km`` below sea level.

    Inverse of :func:`velest_depth_from_file`.  Note VELEST writes its *internal* depth
    (zshift included) to the output CNV file, so subtract ``zshift`` again when reading
    results back if zshift is not 0 (``velest.f:3280-3284`` writes ``e(4,i)``, which
    already includes it).
    """
    a = _arr(depth_km)
    return _out(a - zshift_km, a.ndim == 0)


def velest_zshift_for_reference_elevation(reference_elevation_m: float) -> float:
    """ZSHIFT (km) for hypocentre depths measured below a reference elevation.

    Manual p. 13: events located relative to an average station elevation of 900 m need
    ``zshift = 0.9`` so that depths become relative to sea level.
    """
    return float(reference_elevation_m) / 1000.0


@dataclass(frozen=True)
class _VelestConstants:
    aa: float  # km per minute of latitude
    bb: float  # km per minute of longitude, at the equator-equivalent (divided by cos lat1)
    rlatc: float
    olat_min: float
    olon_min: float
    sint: float
    cost: float


class VelestShortDistance:
    """Equivalent of VELEST's ``SETORG``/``DIST``/``REDIST`` (``icoordsystem = 0``).

    Reproduces VELEST's local cartesian coordinates from true (east positive) lat/lon.
    VELEST's own system has **x positive WEST**, y positive north, rotated clockwise
    by ``rotate`` degrees (``velest.f:10017-10120``).  The ellipsoid is WGS72 with
    ``REARTH = 6378.135`` km, ``1/f = 298.26``.

    Parameters
    ----------
    origin_lat, origin_lon
        Origin as true lat/lon, east longitude positive.  (The control-file values are
        :func:`velest_olat` and :func:`velest_olon`.)
    rotate
        ``rotate`` of the control file, degrees (axes rotated clockwise from north).

    Notes
    -----
    VELEST uses single-precision reals, so it differs from this double-precision
    implementation by up to about 1e-3 km; the algebra is otherwise identical.
    """

    def __init__(self, origin_lat: float, origin_lon: float, rotate: float = 0.0) -> None:
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(wrap_lon(origin_lon, "180"))
        self.rotate = float(rotate)
        rad = _V_RAD
        olat_min = self.origin_lat * 60.0
        olon_min = -self.origin_lon * 60.0  # minutes WEST
        phi = olat_min * rad / 60.0
        beta = phi - math.sin(phi * 2.0) / _V_ELLIP
        rlatc = math.tan(beta) / math.tan(phi)
        lat1 = math.atan(rlatc * math.tan(olat_min * rad / 60.0))
        lat2 = math.atan(rlatc * math.tan((olat_min + 1.0) * rad / 60.0))
        r = _V_REARTH * (1.0 - math.sin(lat1) ** 2 / _V_ELLIP)
        aa = r * (lat2 - lat1)
        delb = math.acos(math.sin(lat1) ** 2 + math.cos(rad / 60.0) * math.cos(lat1) ** 2)
        bb = r * delb / math.cos(lat1)
        self._c = _VelestConstants(
            aa=aa,
            bb=bb,
            rlatc=rlatc,
            olat_min=olat_min,
            olon_min=olon_min,
            sint=math.sin(self.rotate * rad),
            cost=math.cos(self.rotate * rad),
        )

    @property
    def km_per_degree_lat(self) -> float:
        """Kilometres per degree of latitude at the origin (``60 * aa``)."""
        return 60.0 * self._c.aa

    @property
    def km_per_degree_lon(self) -> float:
        """Kilometres per degree of longitude at the origin (``60 * bb * cos(lat1)``)."""
        c = self._c
        lat1 = math.atan(c.rlatc * math.tan(c.olat_min * _V_RAD / 60.0))
        return 60.0 * c.bb * math.cos(lat1)

    def to_xy(self, lat: ArrayLike, lon: ArrayLike) -> tuple[float | Array, float | Array]:
        """True lat/lon (degrees, east +) to VELEST ``(x, y)`` km (x positive WEST).

        Equivalent of ``SDC(x, y, lat, lon, -1)`` / ``DIST`` (``velest.f:10017-10051``).
        """
        c = self._c
        la, lo = np.broadcast_arrays(_arr(lat), _arr(lon))
        scalar = la.ndim == 0
        rad = _V_RAD
        q = 60.0 * la - c.olat_min
        yp = q + c.olat_min
        lat1 = np.arctan(c.rlatc * np.tan(rad * yp / 60.0))
        lat2 = math.atan(c.rlatc * math.tan(rad * c.olat_min / 60.0))
        lat3 = (lat2 + lat1) / 2.0
        xx = 60.0 * (-lo) - c.olon_min
        q = q * c.aa
        xx = xx * c.bb * np.cos(lat3)
        y = c.cost * q - c.sint * xx
        x = c.cost * xx + c.sint * q
        return _out(x, scalar), _out(y, scalar)

    def to_latlon(self, x: ArrayLike, y: ArrayLike) -> tuple[float | Array, float | Array]:
        """VELEST ``(x, y)`` km (x positive WEST) to true lat/lon (degrees, east +).

        Equivalent of ``SDC(x, y, lat, lon, 1)`` / ``REDIST`` (``velest.f:10053-10120``).
        """
        c = self._c
        xa, ya = np.broadcast_arrays(_arr(x), _arr(y))
        scalar = xa.ndim == 0
        rad = _V_RAD
        xr = xa * c.cost - ya * c.sint
        yr = ya * c.cost + xa * c.sint
        q = yr / c.aa
        lat_min = q + c.olat_min
        lat1 = np.arctan(c.rlatc * np.tan(lat_min * rad / 60.0))
        lat2 = math.atan(c.rlatc * math.tan(c.olat_min * rad / 60.0))
        lat3 = (lat1 + lat2) / 2.0
        p = xr / (c.bb * np.cos(lat3))
        lon_w_min = p + c.olon_min
        return _out(lat_min / 60.0, scalar), _out(-lon_w_min / 60.0, scalar)

    def to_xy_east(self, lat: ArrayLike, lon: ArrayLike) -> tuple[float | Array, float | Array]:
        """Like :meth:`to_xy` but with x positive EAST (the negated VELEST x).

        Only valid for ``rotate = 0``; for other rotations the axes are still
        VELEST's rotated axes with x negated.
        """
        x, y = self.to_xy(lat, lon)
        return -x, y


# ----------------------------------------------------------------------------------------
# FMTOMO
# ----------------------------------------------------------------------------------------


def depth_to_radius(
    depth_km: ArrayLike, earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM
) -> float | Array:
    """Radius (km from the Earth's centre) for a depth (km, positive down).

    ``fm3d`` does ``r = earth_radius - depth`` for sources and receivers
    (``3dfm_main.f90:209,222,420``).  Negative depth means above the surface.
    """
    a = _arr(depth_km)
    return _out(earth_radius_km - a, a.ndim == 0)


def radius_to_depth(
    radius_km: ArrayLike, earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM
) -> float | Array:
    """Depth (km, positive down) for a radius; inverse of :func:`depth_to_radius`."""
    a = _arr(radius_km)
    return _out(earth_radius_km - a, a.ndim == 0)


def fmtomo_receiver_depth_km(elevation_m: ArrayLike) -> float | Array:
    """FMTOMO receiver/source "depth" (km) for an elevation (m, positive up).

    FMTOMO depth is negative above the surface (manual p. 39, "negative values are above
    the surface"), so a station at +500 m has depth ``-0.5``.  Elevation is measured from
    the sphere of radius 6371 km (sea level); surface topography only enters via the
    interfaces.
    """
    return elevation_m_to_depth_km(elevation_m)


def fmtomo_lat_lon_to_radians(
    lat: ArrayLike, lon: ArrayLike
) -> tuple[float | Array, float | Array]:
    """Degrees to radians, as stored in ``vgrids.in``/``interfaces.in`` headers.

    Latitude is geographic (north +, *not* colatitude) and longitude east (``grid3dg.f90``
    lines 164-167; example1 ``vgrids.in`` origin ``-0.736 2.4387`` rad = -42.18, 139.73
    deg).  FMTOMO treats latitude as a spherical angle.
    """
    la, lo = _arr(lat), _arr(lon)
    return _out(np.deg2rad(la), la.ndim == 0), _out(np.deg2rad(lo), lo.ndim == 0)


def spherical_distance_km(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
    earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM,
) -> float | Array:
    """Great-circle distance on the FMTOMO sphere (haversine, radius 6371 km)."""
    p1, p2 = np.deg2rad(_arr(lat1)), np.deg2rad(_arr(lat2))
    dl = np.deg2rad(_arr(lon2) - _arr(lon1))
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    d = 2.0 * earth_radius_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return _out(d, np.ndim(d) == 0)


# ----------------------------------------------------------------------------------------
# Distances and azimuths (WGS84 geodesic)
# ----------------------------------------------------------------------------------------


def _inv(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> tuple[Array, Array, Array, bool]:
    a1, o1, a2, o2 = np.broadcast_arrays(_arr(lat1), _arr(lon1), _arr(lat2), _arr(lon2))
    scalar = a1.ndim == 0
    az, baz, dist = _GEOD.inv(o1, a1, o2, a2)
    return np.asarray(az), np.asarray(baz), np.asarray(dist), scalar


def epicentral_distance_km(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> float | Array:
    """WGS84 geodesic distance in km between two surface points (degrees)."""
    _, _, d, scalar = _inv(lat1, lon1, lat2, lon2)
    return _out(d / 1000.0, scalar)


def hypocentral_distance_km(
    lat1: ArrayLike,
    lon1: ArrayLike,
    depth1_km: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
    depth2_km: ArrayLike,
) -> float | Array:
    """Straight-line distance (km) between two points given as lat, lon, depth (km down).

    Computed as ``sqrt(epicentral**2 + (depth2 - depth1)**2)``.  For a station use
    ``depth = -elevation_m / 1000`` (see :func:`elevation_m_to_depth_km`).
    """
    _, _, d, scalar = _inv(lat1, lon1, lat2, lon2)
    dz = _arr(depth2_km) - _arr(depth1_km)
    return _out(np.hypot(d / 1000.0, dz), scalar and np.ndim(dz) == 0)


def azimuth_deg(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> float | Array:
    """Azimuth (degrees clockwise from north, 0-360) from point 1 to point 2."""
    az, _, _, scalar = _inv(lat1, lon1, lat2, lon2)
    return _out(az % 360.0, scalar)


def backazimuth_deg(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> float | Array:
    """Back-azimuth at point 2 pointing back to point 1 (degrees from north, 0-360).

    For the usual seismological use give the event as point 1 and the station as point 2:
    the result is the direction from the station towards the event.
    """
    _, baz, _, scalar = _inv(lat1, lon1, lat2, lon2)
    return _out(baz % 360.0, scalar)
