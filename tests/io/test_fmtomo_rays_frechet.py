"""Tests for osv.io.fmtomo.rays (rays.dat) and .frechet (frechet.dat)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osv.io.fmtomo import frechet as fr
from osv.io.fmtomo import rays as ry
from osv.io.fmtomo import times as t
from osv.io.fmtomo._fortran import FmtomoFormatError
from osv.io.fmtomo.controls import FrechetSpec

E1 = Path(__file__).resolve().parents[1] / "data" / "fmtomo" / "example1"


# hand-built rays.dat in the format of 3dfmlib.f90:4752-4817: ray 1 has two sections (one point
# written as r lat lon in radians), ray 2 is invalid, ray 3 has a single 2-point section
def _pt(r: float, la: float, lo: float) -> str:
    return f"{r:17.8f}{la:17.8f}{lo:17.8f}\n"


RAYS = (
    "     1     1     1     0     2\n"
    "     3     1    F    F\n"
    + _pt(6371.0, -0.71, 2.45)
    + _pt(6350.0, -0.711, 2.451)
    + _pt(6340.0, -0.712, 2.452)
    + "     2     2    T    F\n"
    + _pt(6340.0, -0.712, 2.452)
    + _pt(6300.0, -0.715, 2.455)
    + "     2     1     1     0     0\n"
    "     3     1     1     0     1\n"
    "     2     1    F    T\n" + _pt(6372.0, -0.72, 2.46) + _pt(6360.0, -0.725, 2.465)
)


def test_rays_parse_structure_and_units():
    rp = ry.parse_rays(RAYS)
    assert list(rp.rays["valid"]) == [True, False, True]
    assert list(rp.rays["n_sections"]) == [2, 0, 1]
    assert list(rp.sections["npoints"]) == [3, 2, 2]
    assert list(rp.sections["region"]) == [1, 2, 1]
    assert list(rp.sections["diffracted"]) == [False, True, False]
    assert list(rp.sections["headwave"]) == [False, False, True]
    pts = rp.points
    assert len(pts) == 7
    # r in km, lat/lon in radians in the file; degrees and depth derived
    assert pts["r_km"].iloc[0] == 6371.0
    assert pts["depth_km"].iloc[0] == pytest.approx(0.0)
    assert pts["depth_km"].iloc[-1] == pytest.approx(11.0)
    assert pts["lat"].iloc[0] == pytest.approx(np.degrees(-0.71))
    assert pts["lon"].iloc[0] == pytest.approx(np.degrees(2.45))
    assert list(pts["ray"].unique()) == [0, 2]


def test_rays_roundtrip_text_equal():
    rp = ry.parse_rays(RAYS)
    assert ry.format_rays(rp) == RAYS


def test_rays_errors():
    with pytest.raises(FmtomoFormatError, match="declares 5 points"):
        ry.parse_rays("1 1 1 0 1\n 5 1 F F\n 1 2 3\n")
    with pytest.raises(FmtomoFormatError, match="bad section header"):
        ry.parse_rays("1 1 1 0 1\n x 1 F F\n")


def test_ray_hit_count():
    rp = ry.parse_rays(RAYS)
    hits = ry.ray_hit_count(
        rp,
        depth_edges=np.array([-1.0, 5.0, 100.0]),
        lat_edges=np.array([-45.0, -40.0]),
        lon_edges=np.array([135.0, 145.0]),
    )
    assert hits.dims == ("depth", "lat", "lon")
    assert float(hits.sum()) == 7.0  # all points of the valid rays
    # depths of the valid rays: 0, 21, 31, 31, 71 and -1, 11 km
    assert hits.sel(depth=2.0).item() == 2.0  # bin [-1, 5): depth 0 and -1
    assert hits.sel(depth=52.5).item() == 5.0


# ---- frechet --------------------------------------------------------------------------


def test_frechet_example1():
    f = fr.read_frechet(E1 / "frechet.dat")
    assert f.matrix.shape == (500, 20)  # 5 sources x 4 parameters (frechet.in: only sources)
    assert list(f.rays.columns) == ["receiver", "source", "ray", "subray", "n_deriv"]
    row0 = f.matrix.getrow(0).toarray().ravel()
    # source 1: dt/d(r, lat, lon) and -1 for the origin time (frechet.f90:568-583)
    np.testing.assert_allclose(row0[:4], [-0.22723977e-01, 0.86725453e-01, 0.96255239e-01, -1.0])
    assert not row0[4:].any()
    # a ray to source 3 only touches parameters 9-12
    k = f.rays.index[(f.rays.receiver == 1) & (f.rays.source == 3)][0]
    cols = f.matrix.getrow(k).indices
    assert sorted(cols) == [8, 9, 10, 11]
    # invalid rays (time -1 in arrivals.dat) have no derivatives
    arr = t.read_arrivals(E1 / "arrivals.dat")
    assert list(f.rays["n_deriv"] == 0) == list(~arr["valid"])


def test_frechet_roundtrip_text_equal(tmp_path):
    src = E1 / "frechet.dat"
    f = fr.read_frechet(src)
    out = fr.write_frechet(tmp_path / "frechet.dat", f)
    assert out.read_text() == src.read_text()


def test_frechet_parameter_layout_matches_frechet_in():
    spec = FrechetSpec.read(E1 / "frechet.in")
    assert (
        spec.compute
        and spec.velocity == []
        and spec.interfaces == []
        and spec.sources == [1, 2, 3, 4, 5]
    )
    blocks = fr.parameter_layout(sources=spec.sources)
    assert [b.start for b in blocks] == [0, 4, 8, 12, 16]
    f = fr.read_frechet(E1 / "frechet.dat", n_params=blocks[-1].start + blocks[-1].size)
    assert f.matrix.shape[1] == 20


def test_parameter_layout_velocity_and_interface_ordering():
    blocks = fr.parameter_layout(
        velocity=[(2, 1, (3, 4, 5))], interfaces=[(2, (4, 5))], sources=[7]
    )
    v, i, s = blocks
    assert (
        (v.start, v.size) == (0, 60)
        and (i.start, i.size) == (60, 20)
        and (s.start, s.size) == (80, 4)
    )
    # velocity nodes: radius index fastest (frechet.f90:287): column 1 -> ir=1, column 3 -> next lat
    ir, ilat, ilon = v.node_index(np.array([0, 1, 3, 12, 59]))
    assert list(ir) == [0, 1, 0, 0, 2]
    assert list(ilat) == [0, 0, 1, 0, 3]
    assert list(ilon) == [0, 0, 0, 1, 4]
    # interface nodes: latitude index fastest (frechet.f90:527)
    ilat2, ilon2 = i.node_index(np.array([60, 61, 64, 79]))
    assert list(ilat2) == [0, 1, 0, 3]
    assert list(ilon2) == [0, 0, 1, 4]
    with pytest.raises(ValueError):
        v.node_index(np.array([60]))


def test_frechet_e_format_and_errors():
    assert fr._e17_8(-0.22723977e-01).strip() == "-0.22723977E-01"
    assert fr._e17_8(0.0).strip() == "0.00000000E+00"
    assert fr._e17_8(123.456789).strip() == "0.12345679E+03"
    with pytest.raises(FmtomoFormatError, match="declares 3 derivatives"):
        fr.parse_frechet("1 1 1 0 3\n 1 0.5\n")
    with pytest.raises(FmtomoFormatError, match="exceeds the expected number"):
        fr.parse_frechet("1 1 1 0 1\n 9 0.5\n", n_params=4)


def test_rays_real_fm3d_output_excerpt():
    """Rays 1, 2 and the invalid ray 6 of ``fm3d`` run on example 1 (save_rays_mode = T)."""
    path = E1 / "rays_excerpt.dat"
    rp = ry.read_rays(path)
    assert list(rp.rays["receiver"]) == [1, 2, 6]
    assert list(rp.rays["valid"]) == [True, True, False]
    assert list(rp.rays["n_sections"]) == [2, 2, 0]
    # the reflection path 0 2 2 1: one section in region 2 (down to interface 2), one in region 1
    assert list(rp.sections["region"]) == [2, 1, 2, 1]
    pts = rp.points[rp.points["ray"] == 0]
    first, last = pts.iloc[0], pts.iloc[-1]
    # a ray starts at the source (source 1: depth 49.1324768, lat -41.0005379, lon 141.401596) ...
    assert (first.depth_km, first.lat, first.lon) == pytest.approx(
        (49.1324768, -41.0005379, 141.401596), abs=1e-5
    )
    # ... and ends at the receiver (receiver 1: depth -1 km, lat -41.8, lon 140.2)
    assert (last.depth_km, last.lat, last.lon) == pytest.approx((-1.0, -41.8, 140.2), abs=1e-5)
    # the section boundary lies on interface 2 (~ 30 km deep in example 1)
    sec1_last = rp.points[(rp.points["ray"] == 0) & (rp.points["section"] == 1)].iloc[-1]
    sec2_first = rp.points[(rp.points["ray"] == 0) & (rp.points["section"] == 2)].iloc[0]
    assert sec1_last.depth_km == pytest.approx(sec2_first.depth_km, abs=1e-6)
    assert ry.format_rays(rp) == path.read_text()
