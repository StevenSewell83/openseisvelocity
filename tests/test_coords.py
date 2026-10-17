"""Tests for osv.coords."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Geod

from osv import coords as c

NZ_LAT, NZ_LON = -38.55, 176.194  # Ngatamariki

# Reference values produced by compiling SETORG/DIST/REDIST verbatim from
# velest/src/velest.f (velest.f:10017-10240) with gfortran and running them on inputs
# that are exactly representable in single precision.
# (olat_velest, olon_velest_west_positive, rotate, lat, lon_west_positive, x, y)
VELEST_REF = [
    (-38.5, -176.25, 0.0, -38.625, -176.125, 10.89471912, -13.87571526),
    (-38.5, -176.25, 0.0, -38.40625, -176.40625, -13.63892937, 10.40678692),
    (-38.5, -176.25, 30.0, -38.625, -176.125, 2.49724579, -17.46408081),
    (-38.5, -176.25, 30.0, -38.40625, -176.40625, -6.60826540, 15.83200645),
    (46.953125, -7.4375, 0.0, 47.25, -7.0, 33.21158218, 33.00355148),
    (46.953125, -7.4375, 0.0, 46.5, -8.0, -42.99856949, -50.37384415),
    (46.953125, -7.4375, 0.0, -33.0, -151.0, -15835.02343750, -8888.37792969),
]


@pytest.mark.parametrize(("olat", "olonw", "rot", "lat", "lonw", "x", "y"), VELEST_REF)
def test_velest_short_distance_matches_fortran(olat, olonw, rot, lat, lonw, x, y):
    # VELEST lon is west positive; osv is east positive.
    proj = c.VelestShortDistance(olat, -olonw, rot)
    xx, yy = proj.to_xy(lat, -lonw)
    tol = max(2e-3, 1e-6 * abs(x))  # VELEST uses single precision
    assert xx == pytest.approx(x, abs=tol)
    assert yy == pytest.approx(y, abs=tol)
    la, lo = proj.to_latlon(x, y)
    assert la == pytest.approx(lat, abs=1e-4)
    assert lo == pytest.approx(-lonw, abs=1e-4)


def test_velest_x_is_positive_west():
    proj = c.VelestShortDistance(NZ_LAT, NZ_LON)
    x_west_pt, _ = proj.to_xy(NZ_LAT, NZ_LON - 0.1)
    x_east_pt, _ = proj.to_xy(NZ_LAT, NZ_LON + 0.1)
    assert x_west_pt > 0 > x_east_pt
    _, y_north = proj.to_xy(NZ_LAT + 0.1, NZ_LON)
    assert y_north > 0
    assert proj.to_xy_east(NZ_LAT, NZ_LON + 0.1)[0] > 0


def test_velest_short_distance_roundtrip_vectorised():
    rng = np.random.default_rng(1)
    lat = NZ_LAT + rng.uniform(-0.3, 0.3, 50)
    lon = NZ_LON + rng.uniform(-0.3, 0.3, 50)
    for rot in (0.0, 17.5, -40.0):
        proj = c.VelestShortDistance(NZ_LAT, NZ_LON, rot)
        x, y = proj.to_xy(lat, lon)
        la, lo = proj.to_latlon(x, y)
        np.testing.assert_allclose(la, lat, atol=1e-10)
        np.testing.assert_allclose(lo, lon, atol=1e-10)


def test_velest_short_distance_scales():
    proj = c.VelestShortDistance(NZ_LAT, NZ_LON)
    assert proj.km_per_degree_lat == pytest.approx(111.1, abs=0.3)
    assert proj.km_per_degree_lon == pytest.approx(111.3 * np.cos(np.radians(38.55)), abs=0.3)


def test_velest_agrees_with_geodesic():
    geod = Geod(ellps="WGS84")
    proj = c.VelestShortDistance(NZ_LAT, NZ_LON)
    x, y = proj.to_xy(NZ_LAT + 0.1, NZ_LON + 0.12)
    _, _, d = geod.inv(NZ_LON, NZ_LAT, NZ_LON + 0.12, NZ_LAT + 0.1)
    assert np.hypot(x, y) == pytest.approx(d / 1000.0, rel=2e-3)


# ---------------------------------------------------------------- VELEST text formats
def test_nz_point_velest_formatting_roundtrip():
    lat_s, lon_s = c.format_velest_lat(NZ_LAT), c.format_velest_lon(NZ_LON)
    assert lat_s == "38.5500S"
    assert lon_s == "176.1940E"
    assert c.parse_velest_lat(lat_s) == pytest.approx(NZ_LAT)
    assert c.parse_velest_lon(lon_s) == pytest.approx(NZ_LON)
    assert c.velest_olat(NZ_LAT) == pytest.approx(-38.55)
    assert c.velest_olon(NZ_LON) == pytest.approx(-176.194)  # east -> negative


def test_velest_format_widths_and_hemispheres():
    assert c.format_velest_lat(5.5) == " 5.5000N"
    assert len(c.format_velest_lat(-0.5)) == 8
    assert c.format_velest_lon(-7.4396) == "  7.4396W"
    assert len(c.format_velest_lon(120.0)) == 9
    assert c.parse_velest_lat(" 36.6458N") == pytest.approx(36.6458)
    assert c.parse_velest_lon("121.0298W") == pytest.approx(-121.0298)
    with pytest.raises(ValueError):
        c.parse_velest_lat("38.5")
    with pytest.raises(ValueError):
        c.parse_velest_lon("176.1W5")


def test_legacy_nz_files_are_a_consistent_point_mirror():
    """legacy: '38.5196N 176.1624W', olat=38.55, olon=176.194 == true S/E mirrored."""
    legacy_lat = c.parse_velest_lat("38.5196N")
    legacy_lon = c.parse_velest_lon("176.1624W")
    lat, lon = c.legacy_mirror(legacy_lat, legacy_lon)
    assert (lat, lon) == pytest.approx((-38.5196, 176.1624))
    # origin: legacy control file olat=38.55, olon=176.194 (west positive)
    legacy = c.VelestShortDistance(38.55, -176.194)
    true = c.VelestShortDistance(*c.legacy_mirror(38.55, -176.194))
    assert true.origin_lat == pytest.approx(-38.55)
    assert true.origin_lon == pytest.approx(176.194)
    xl, yl = legacy.to_xy(legacy_lat, legacy_lon)
    xt, yt = true.to_xy(lat, lon)
    # same geometry rotated by 180 degrees: x -> -x, y -> -y.  Not bit-exact because the
    # km-per-arcminute of latitude is a finite difference taken towards +lat from the origin.
    assert xl == pytest.approx(-xt, abs=1e-4)
    assert yl == pytest.approx(-yt, abs=1e-4)
    # inter-station distance preserved
    l2 = legacy.to_xy(38.5082, -176.1770)
    t2 = true.to_xy(*c.legacy_mirror(38.5082, -176.1770))
    assert np.hypot(xl - l2[0], yl - l2[1]) == pytest.approx(
        np.hypot(xt - t2[0], yt - t2[1]), abs=1e-4
    )
    # mirror is an involution
    assert c.legacy_mirror(*c.legacy_mirror(1.5, -2.5)) == (1.5, -2.5)


def test_velest_elevation_and_depth_helpers():
    assert c.velest_elevation_field(519.4) == " 519"
    assert c.velest_elevation_field(-12) == " -12"
    with pytest.raises(ValueError):
        c.velest_elevation_field(10000)
    assert c.velest_station_z_km(519.0) == pytest.approx(-0.519)
    assert c.velest_depth_from_file(2.0, 0.9) == pytest.approx(2.9)
    assert c.velest_depth_to_file(2.9, 0.9) == pytest.approx(2.0)
    assert c.velest_zshift_for_reference_elevation(900.0) == pytest.approx(0.9)


# ---------------------------------------------------------------- local projection
def test_local_tm_origin_and_roundtrip():
    proj = c.LocalProjection(NZ_LAT, NZ_LON)
    assert proj.forward(NZ_LAT, NZ_LON) == pytest.approx((0.0, 0.0), abs=1e-9)
    rng = np.random.default_rng(0)
    lat = NZ_LAT + rng.uniform(-0.2, 0.2, 100)
    lon = NZ_LON + rng.uniform(-0.2, 0.2, 100)
    x, y = proj.forward(lat, lon)
    la, lo = proj.inverse(x, y)
    np.testing.assert_allclose(la, lat, atol=1e-9)
    np.testing.assert_allclose(lo, lon, atol=1e-9)
    assert isinstance(proj.forward(NZ_LAT, NZ_LON)[0], float)


def test_local_tm_axes_directions():
    proj = c.LocalProjection(NZ_LAT, NZ_LON)
    x, y = proj.forward(NZ_LAT + 0.05, NZ_LON + 0.05)
    assert x > 0
    assert y > 0


def test_local_tm_agrees_with_geodesic_over_20km():
    geod = Geod(ellps="WGS84")
    proj = c.LocalProjection(NZ_LAT, NZ_LON)
    rng = np.random.default_rng(42)
    # points within ~10 km of the origin => pair separations up to ~20 km
    lat = NZ_LAT + rng.uniform(-0.09, 0.09, 200)
    lon = NZ_LON + rng.uniform(-0.11, 0.11, 200)
    x, y = proj.forward(lat, lon)
    i, j = rng.integers(0, 200, 400), rng.integers(0, 200, 400)
    _, _, d = geod.inv(lon[i], lat[i], lon[j], lat[j])
    dproj = np.hypot(x[i] - x[j], y[i] - y[j]) * 1000.0
    assert np.max(np.abs(dproj - d)) < 1.0  # metres
    assert np.max(d) > 15000.0


def test_epsg_projection():
    proj = c.LocalProjection(NZ_LAT, NZ_LON, method="EPSG:2193")  # NZTM2000
    assert proj.forward(NZ_LAT, NZ_LON) == pytest.approx((0.0, 0.0), abs=1e-6)
    x, y = proj.forward(NZ_LAT + 0.1, NZ_LON + 0.1)
    la, lo = proj.inverse(x, y)
    assert (la, lo) == pytest.approx((NZ_LAT + 0.1, NZ_LON + 0.1), abs=1e-9)
    # grid axes are rotated by meridian convergence, but distances agree within scale error
    tm = c.LocalProjection(NZ_LAT, NZ_LON)
    xt, yt = tm.forward(NZ_LAT + 0.1, NZ_LON + 0.1)
    assert np.hypot(x, y) == pytest.approx(np.hypot(xt, yt), rel=2e-3)


def test_projection_bad_input():
    with pytest.raises(ValueError):
        c.LocalProjection(95.0, 0.0)
    with pytest.raises(ValueError):
        c.LocalProjection(0.0, 0.0, method="not-a-crs")
    with pytest.raises(ValueError):
        c.LocalProjection(0.0, 0.0, method="EPSG:4326")


# ---------------------------------------------------------------- depth / elevation
def test_elevation_depth_and_borehole():
    assert c.elevation_m_to_depth_km(500.0) == pytest.approx(-0.5)
    assert c.depth_km_to_elevation_m(-0.5) == pytest.approx(500.0)
    # NS12: surface 400 m, sensor 514 m deep -> -114 m elevation
    assert c.borehole_sensor_elevation_m(400.0, 514.0) == pytest.approx(-114.0)
    assert c.borehole_sensor_elevation_m(400.0) == pytest.approx(400.0)
    elev = c.borehole_sensor_elevation_m(np.array([400.0, 300.0]), np.array([514.0, 0.0]))
    np.testing.assert_allclose(c.elevation_m_to_depth_km(elev), [0.114, -0.3])


# ---------------------------------------------------------------- FMTOMO
def test_fmtomo_depth_radius():
    assert c.depth_to_radius(0.0) == 6371.0
    assert c.depth_to_radius(-1.3) == pytest.approx(6372.3)  # interface 1 of example1
    assert c.radius_to_depth(6372.3) == pytest.approx(-1.3)
    d = np.array([-1.0, 5.0, 80.0])
    np.testing.assert_allclose(c.radius_to_depth(c.depth_to_radius(d)), d)
    assert c.fmtomo_receiver_depth_km(250.0) == pytest.approx(-0.25)


def test_fmtomo_radians_and_wrap():
    la, lo = c.fmtomo_lat_lon_to_radians(-42.18, 139.73)
    assert la == pytest.approx(-0.7362, abs=1e-4)  # example1 vgrids.in origin
    assert lo == pytest.approx(2.4387, abs=1e-4)
    assert c.wrap_lon(-115.0, "360") == pytest.approx(245.0)
    assert c.wrap_lon(245.0, "180") == pytest.approx(-115.0)
    with pytest.raises(ValueError):
        c.wrap_lon(1.0, "90")  # type: ignore[arg-type]


def test_spherical_distance():
    assert c.spherical_distance_km(0, 0, 0, 1) == pytest.approx(2 * np.pi * 6371.0 / 360.0)
    assert c.spherical_distance_km(10, 20, 10, 20) == 0.0


# ---------------------------------------------------------------- distances / azimuths
def test_distances_and_azimuths():
    geod = Geod(ellps="WGS84")
    ev = (NZ_LAT, NZ_LON, 3.0)
    st = (NZ_LAT - 0.05, NZ_LON + 0.04, -0.4)
    epi = c.epicentral_distance_km(ev[0], ev[1], st[0], st[1])
    _, _, d = geod.inv(ev[1], ev[0], st[1], st[0])
    assert epi == pytest.approx(d / 1000.0)
    hyp = c.hypocentral_distance_km(*ev, *st)
    assert hyp == pytest.approx(np.hypot(epi, 3.4))
    assert c.azimuth_deg(0, 0, 1, 0) == pytest.approx(0.0)
    assert c.azimuth_deg(0, 0, 0, 1) == pytest.approx(90.0)
    assert c.azimuth_deg(0, 0, 0, -1) == pytest.approx(270.0)
    assert c.backazimuth_deg(0, 0, 0, 1) == pytest.approx(270.0)
    # vectorised
    out = c.epicentral_distance_km(np.zeros(3), np.zeros(3), np.zeros(3), [0.0, 1.0, 2.0])
    assert out.shape == (3,)
    assert out[0] == 0.0


def test_to_sea_level_depth() -> None:
    assert c.to_sea_level_depth(2.0) == 2.0
    assert c.to_sea_level_depth(2.0, "reference", reference_elevation_m=600.0) == pytest.approx(1.4)
    # reference above surface by 0.6 km: a "depth" of 0 is 0.6 km above sea level
    assert c.to_sea_level_depth(0.0, "reference", reference_elevation_m=600.0) == pytest.approx(
        -0.6
    )
    out = c.to_sea_level_depth([1.0, 2.0], "surface", surface_elevation_m=[300.0, -100.0])
    np.testing.assert_allclose(out, [0.7, 2.1])
    assert np.isnan(c.to_sea_level_depth(1.0, "surface", surface_elevation_m=np.nan))
    with pytest.raises(ValueError, match="reference_elevation_m"):
        c.to_sea_level_depth(1.0, "reference")
    with pytest.raises(ValueError, match="surface_elevation_m"):
        c.to_sea_level_depth(1.0, "surface")
    with pytest.raises(ValueError, match="unknown depth datum"):
        c.to_sea_level_depth(1.0, "moon")  # type: ignore[arg-type]
