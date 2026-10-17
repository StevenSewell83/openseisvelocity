"""Tests for osv.io.fmtomo.grids (vgrids.in, interfaces.in, propgrid.in, arrtimes.dat).

Fixtures in ``tests/data/fmtomo/example1`` are the files of the vendored FMTOMO example 1
(``fortran/fmtomo/example1``): a 2-layer model, 14x14x14 velocity nodes per layer (12 + 2
cushion nodes per axis), 3 interfaces on a 14x14 grid, 21x41x41 propagation grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from osv.coords import FMTOMO_EARTH_RADIUS_KM
from osv.io.fmtomo import grids as g
from osv.io.fmtomo._fortran import FmtomoFormatError

DATA = Path(__file__).resolve().parents[1] / "data" / "fmtomo" / "example1"


def test_vgrids_coordinates_example1():
    vg = g.read_vgrids(DATA / "vgrids.in")
    assert (vg.n_layers, vg.n_types) == (2, 1)
    ds = vg.get(1)
    assert ds["velocity"].dims == ("depth", "lat", "lon")
    assert ds["velocity"].shape == (14, 14, 14)
    # header of the file: origin (6283.48181818182 km, -0.736211632237290 rad, 2.43870103178602 rad)
    # spacing (7.41818181818182 km, 3.173326001022804e-3 rad, 4.759989001534202e-3 rad)
    assert ds.attrs["fmtomo_r0_km"] == pytest.approx(6283.48181818182)
    lat0 = np.degrees(-0.736211632237290)
    lon0 = np.degrees(2.43870103178602)
    assert ds["lat"].values[0] == pytest.approx(lat0)
    assert lat0 == pytest.approx(-42.1818, abs=1e-3)  # grid3dg: south edge -42 minus one node
    assert ds["lon"].values[0] == pytest.approx(lon0)
    assert lon0 == pytest.approx(139.7273, abs=1e-3)
    assert np.all(np.diff(ds["lat"]) > 0)
    assert np.all(np.diff(ds["lon"]) > 0)
    # depth = R - r, ascending: shallowest node first
    r = 6283.48181818182 + 7.41818181818182 * np.arange(14)
    assert np.allclose(ds["depth"].values, (FMTOMO_EARTH_RADIUS_KM - r)[::-1])
    assert np.all(np.diff(ds["depth"]) > 0)
    # the top cushion node is one step above the 1.5 km model top
    assert ds["depth"].values[0] == pytest.approx(-1.5 - 7.418181818, abs=1e-6)
    # the interior excludes the cushion ring and spans the requested 1.5 .. -80.1 km range
    inner = g.interior(ds)
    assert inner["velocity"].shape == (12, 12, 12)
    assert inner["depth"].values[0] == pytest.approx(-1.5)
    assert inner["depth"].values[-1] == pytest.approx(80.1)
    assert inner["lat"].values[[0, -1]] == pytest.approx([-42.0, -40.0])
    assert inner["lon"].values[[0, -1]] == pytest.approx([140.0, 143.0])
    mask = g.cushion_mask(ds)
    assert int(mask.sum()) == 14**3 - 12**3


def test_vgrids_values_follow_file_order():
    """First record of the file is the deepest, southern, western node."""
    vg = g.read_vgrids(DATA / "vgrids.in")
    ds = vg.get(1)
    lines = [ln.split() for ln in (DATA / "vgrids.in").read_text().splitlines()[4:] if ln.strip()]
    first = float(lines[0][0])
    second_lon = float(lines[1][0])
    assert ds["velocity"].values[-1, 0, 0] == pytest.approx(first)
    assert ds["velocity"].values[-1, 0, 1] == pytest.approx(second_lon)
    assert ds["uncertainty"].values[0, 0, 0] == pytest.approx(0.3)
    # layer 2 follows layer 1 in the file
    ds2 = vg.get(2)
    assert ds2["velocity"].shape == (14, 14, 14)
    assert float(ds["velocity"].mean()) < float(ds2["velocity"].mean())  # lower crust faster


def test_vgrids_roundtrip_text_and_parse_equal(tmp_path):
    src = DATA / "vgrids.in"
    vg = g.read_vgrids(src)
    out = g.write_vgrids(tmp_path / "vgrids.in", vg)
    # node lines and blank separators are byte identical (the headers are list-directed output
    # of ifort/gfortran, which differs from the shortest round-trip text we write)
    a = src.read_text().splitlines()
    b = out.read_text().splitlines()
    assert len(a) == len(b)
    body_a = [
        ln
        for ln in a
        if ln.strip() == "" or len(ln.split()) == 2 and "." in ln.split()[1] and "E" not in ln
    ]
    body_b = [
        ln
        for ln in b
        if ln.strip() == "" or len(ln.split()) == 2 and "." in ln.split()[1] and "E" not in ln
    ]
    assert body_a == body_b
    vg2 = g.read_vgrids(out)
    for (_, _, d1), (_, _, d2) in zip(vg, vg2, strict=True):
        xr.testing.assert_identical(d1, d2)


def test_vgrids_write_uses_native_header_exactly(tmp_path):
    vg = g.read_vgrids(DATA / "vgrids.in")
    out = g.write_vgrids(tmp_path / "v.in", vg)
    h = out.read_text().splitlines()[1:4]
    assert [int(x) for x in h[0].split()] == [14, 14, 14]
    assert float(h[1].split()[0]) == 7.41818181818182
    assert float(h[2].split()[1]) == -0.736211632237290


def test_vgrids_geometry_derived_from_edited_coordinates(tmp_path):
    vg = g.read_vgrids(DATA / "vgrids.in")
    ds = vg.get(1).copy(deep=True)
    ds = ds.assign_coords(lat=ds["lat"] + 1.0)  # shift grid one degree north
    vg.grids[0][0] = ds
    out = g.write_vgrids(tmp_path / "v.in", vg)
    back = g.read_vgrids(out)
    assert back.get(1)["lat"].values[0] == pytest.approx(ds["lat"].values[0])


def test_vgrids_node_count_mismatch_has_clear_error(tmp_path):
    text = (DATA / "vgrids.in").read_text().splitlines()
    # too few values: drop 5 node records of the last grid
    short = "\n".join(text[:-12]) + "\n"
    with pytest.raises(FmtomoFormatError, match=r"layer 2 type 1 .*2744 nodes.*value record"):
        g.parse_vgrids(short, source="vgrids.in")
    # too many values (header says 14 nodes in r, data has more)
    extra = (DATA / "vgrids.in").read_text() + "  5.00000000  0.30000000\n"
    with pytest.raises(FmtomoFormatError, match="unexpected extra data"):
        g.parse_vgrids(extra)
    with pytest.raises(FmtomoFormatError, match="invalid header"):
        g.parse_vgrids("2 3\n")


def test_vgrids_two_types_and_no_uncertainty_column(tmp_path):
    def grid(v0, layer, vtype):
        vel = v0 + np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        return g.velocity_dataset(
            vel,
            r0_km=6300.0,
            dr_km=10.0,
            lat0_rad=-0.7,
            dlat_rad=0.01,
            lon0_rad=2.4,
            dlon_rad=0.02,
            layer=layer,
            vtype=vtype,
        )

    vg = g.VelocityGrids([[grid(5, 1, 1)], [grid(3, 1, 2)]])
    text = g.format_vgrids(vg)
    assert text.splitlines()[0].split() == ["1", "2"]
    back = g.parse_vgrids(text)
    assert (back.n_layers, back.n_types) == (1, 2)
    assert "uncertainty" not in back.get(1, 2)
    np.testing.assert_allclose(back.get(1, 2)["velocity"].values, vg.get(1, 2)["velocity"].values)
    # mixing grids with and without uncertainty is refused
    vg.grids[1][0] = vg.grids[1][0].assign(uncertainty=xr.zeros_like(vg.grids[1][0]["velocity"]))
    with pytest.raises(FmtomoFormatError, match="uncertainty"):
        g.format_vgrids(vg)


def test_interfaces_example1():
    itf = g.read_interfaces(DATA / "interfaces.in")
    ds = itf.data
    assert itf.n_interfaces == 3
    assert ds["interface_depth"].shape == (3, 14, 14)
    # interface 1 is the 1.3 km high surface (radius 6372.3), the others are 30 and 80 km deep
    assert float(ds["interface_depth"].sel(interface=1).mean()) == pytest.approx(-1.3)
    assert float(ds["interface_depth"].sel(interface=2).mean()) == pytest.approx(30.0, abs=12.0)
    assert float(ds["interface_depth"].sel(interface=3).mean()) == pytest.approx(80.0, abs=0.01)
    vg = g.read_vgrids(DATA / "vgrids.in")
    assert np.allclose(ds["lat"], vg.get(1)["lat"]) and np.allclose(ds["lon"], vg.get(1)["lon"])


def test_interfaces_roundtrip(tmp_path):
    src = DATA / "interfaces.in"
    itf = g.read_interfaces(src)
    out = g.write_interfaces(tmp_path / "interfaces.in", itf)
    a = [ln for ln in src.read_text().splitlines()[4:]]
    b = [ln for ln in out.read_text().splitlines()[4:]]
    assert a == b
    xr.testing.assert_identical(itf.data, g.read_interfaces(out).data)


def test_interfaces_count_mismatch():
    text = (DATA / "interfaces.in").read_text().replace("           3\n", "           4\n", 1)
    with pytest.raises(FmtomoFormatError, match="interface 4"):
        g.parse_interfaces(text)


def test_propgrid_example1_roundtrip(tmp_path):
    pg = g.read_propgrid(DATA / "propgrid.in")
    assert (pg.nr, pg.nlat, pg.nlon) == (21, 41, 41)
    assert pg.dr_km == pytest.approx(4.08)
    assert (pg.refinement_factor, pg.n_refine_cells) == (5, 5)
    # model top 1.5 km above the sphere, bottom 78.6 km deep, as in grid3dg.in (1.5 .. -80.1 range
    # shrunk by the 0.05 degree cushion only laterally)
    assert pg.depth_km[0] == pytest.approx(-1.5)
    assert pg.depth_km[-1] == pytest.approx(-1.5 + 20 * 4.08)
    assert pg.radius_km[-1] == pytest.approx(6371.0 + 1.5)
    assert pg.lat[0] == pytest.approx(-41.95)
    assert pg.lat[-1] == pytest.approx(-40.05)
    assert pg.lon[[0, -1]] == pytest.approx([140.05, 142.95])
    out = g.write_propgrid(tmp_path / "propgrid.in", pg)
    assert g.read_propgrid(out) == pg
    ds = pg.to_dataset()
    assert ds.sizes == {"depth": 21, "lat": 41, "lon": 41}


def test_propgrid_validation():
    with pytest.raises(FmtomoFormatError, match="at least 2 nodes"):
        g.parse_propgrid("1 41 41\n 4.0 0.05 0.05\n 1.5 -41.95 140.05\n 5 5\n")


def test_validate_model_example1_consistent():
    vg = g.read_vgrids(DATA / "vgrids.in")
    itf = g.read_interfaces(DATA / "interfaces.in")
    pg = g.read_propgrid(DATA / "propgrid.in")
    g.validate_model(vg, itf, pg)


def test_validate_model_reports_problems():
    vg = g.read_vgrids(DATA / "vgrids.in")
    itf = g.read_interfaces(DATA / "interfaces.in")
    pg = g.read_propgrid(DATA / "propgrid.in")
    pg.lat0_deg = -42.1  # touches the outermost ring of interface nodes
    vg.grids[0] = vg.grids[0][:1]
    with pytest.raises(FmtomoFormatError) as exc:
        g.validate_model(vg, itf, pg)
    msg = str(exc.value)
    assert "1 velocity layers but 3 interfaces" in msg
    assert "second ring" in msg


def test_arrtimes_roundtrip(tmp_path):
    text = (
        "           3           2           2\n"
        "  4.0  0.5  0.25\n"
        "  6300.0  -41.0  140.0\n"
        "           2\n"
        "           1           1           1\n"
        + "".join(f"{v:12.5f}\n" for v in range(12))
        + "           2           1           1\n"
        + "".join(f"{v:12.5f}\n" for v in [-1.0] + list(range(1, 12)))
    )
    ds = g.parse_arrtimes(text)
    assert ds["arrival_time"].dims == ("field", "depth", "lat", "lon")
    assert ds["arrival_time"].shape == (2, 3, 2, 2)
    assert list(ds["source"].values) == [1, 2]
    # r fastest, then lat, then lon slowest; depth axis is flipped (ascending)
    assert (
        ds["arrival_time"].sel(field=1).values[2, 0, 0] == 0.0
    )  # r0 (deepest) -> last depth index
    assert ds["arrival_time"].sel(field=1).values[1, 0, 0] == 1.0
    assert ds["arrival_time"].sel(field=1).values[2, 1, 0] == 3.0
    assert ds["arrival_time"].sel(field=1).values[2, 0, 1] == 6.0
    assert np.isnan(ds["arrival_time"].sel(field=2).values[2, 0, 0])
    out = g.write_arrtimes(tmp_path / "arrtimes.dat", ds)
    back = g.read_arrtimes(out)
    np.testing.assert_allclose(back["arrival_time"].values, ds["arrival_time"].values)
    with pytest.raises(FmtomoFormatError, match="too few records"):
        g.parse_arrtimes(
            text.replace("           2\n           1", "           3\n           1", 1)
        )


def test_validate_model_allows_layers_of_different_density():
    """Cushion nodes sit one layer-specific spacing outside the model (as in example 2)."""

    def layer(nlat, nlon, k):
        dlat, dlon = 0.04 / (nlat - 3), 0.06 / (nlon - 3)
        return g.velocity_dataset(
            np.full((4, nlat, nlon), 5.0),
            r0_km=6300.0,
            dr_km=20.0,
            lat0_rad=-0.7 - dlat,
            dlat_rad=dlat,
            lon0_rad=2.4 - dlon,
            dlon_rad=dlon,
            layer=k,
        )

    vg = g.VelocityGrids([[layer(7, 8, 1), layer(12, 14, 2)]])
    assert vg.get(1)["lat"].min() != vg.get(2)["lat"].min()
    g.validate_model(vg)
    vg.grids[0][1] = layer(12, 14, 2).assign_coords(lat=layer(12, 14, 2)["lat"] + 0.5)
    with pytest.raises(FmtomoFormatError, match="interior lateral extent"):
        g.validate_model(vg)
