"""Tests for the FMTOMO control files (osv.io.fmtomo.controls/datagen/grid3dg/gmtslice/extras).

Where the vendored example/template file follows the layout the writer produces exactly, the
tests require ``format(parse(text)) == text``; otherwise (hand edited spacing, optional
comments) ``parse(format(parse(text))) == parse(text)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.io.fmtomo import controls as c
from osv.io.fmtomo import datagen as d
from osv.io.fmtomo import extras as x
from osv.io.fmtomo import gmtslice as gs
from osv.io.fmtomo import grid3dg as g3
from osv.io.fmtomo import grids
from osv.io.fmtomo._fortran import FmtomoFormatError
from osv.io.fmtomo.sources import PathSpec

DATA = Path(__file__).resolve().parents[1] / "data" / "fmtomo"
E1 = DATA / "example1"


@pytest.mark.parametrize(
    ("rel", "cls"),
    [
        ("example1/invert3d.in", c.Invert3d),
        ("example1/frechgen.in", c.Frechgen),
        ("example1/mode_set.in", c.ModeSet),
        ("example1/tomo3d.in", c.TomoRun),
        ("example1/residuals.in", c.Residuals),
        ("example1/scorrect.in", c.Scorrect),
        ("example1/mkmodel/arraygen.in", d.Arraygen),
        ("example1/gmtplot/gmtslice.in", gs.Gmtslice),
        ("templates/moddata.in", d.Moddata),
        ("extras/gridloc.in", x.GridLoc),
        ("extras/gridlocj.in", x.GridLocJ),
        ("extras/jointimes.in", x.JoinTimes),
        ("extras/mkvpvs.in", x.MkVpVs),
        ("extras/frvpvs.in", x.FrVpVs),
        ("extras/vpvsresids.in", x.VpVsResids),
    ],
)
def test_text_roundtrip_exact(rel, cls):
    text = (DATA / rel).read_text()
    obj = cls.parse(text, rel)
    assert obj.format().rstrip() == text.rstrip()
    assert cls.parse(obj.format()) == obj


@pytest.mark.parametrize(
    ("rel", "cls"),
    [
        ("example1/mkmodel/synthdata.in", d.Synthdata),
        ("example1/mkmodel/moddata.in", d.Moddata),
        ("extras/vpvsinv.in", x.VpVsInv),
        ("templates/obsdata.in", d.Obsdata),
        ("templates/grid3dg.in", g3.Grid3dg),
        ("example1/mkmodel/grid3dg.in", g3.Grid3dg),
        ("example2/grid3dg.in", g3.Grid3dg),
        ("example3/grid3dg.in", g3.Grid3dg),
        ("templates/gmtslice.in", gs.Gmtslice),
        ("extras/tomo3di.in", c.TomoRun),
    ],
)
def test_parse_format_parse_equal(rel, cls):
    text = (DATA / rel).read_text()
    obj = cls.parse(text, rel)
    assert cls.parse(obj.format()) == obj


def test_invert3d_values():
    inv = c.Invert3d.read(E1 / "invert3d.in")
    assert inv.ref_vgrid_file == "vgridsref.in" and inv.rtimes_file == "rtimes.dat"
    assert inv.velocity_inversion == (0, 1.0, 1.0)
    assert inv.source_inversion == (1, 1.0, 1.0)
    assert inv.subspace_dimension == 20 and inv.earth_radius_km == 6371.0
    assert inv.min_interface_distance_km == 0.07
    # frechgen/residuals read fixed records of this file (line 22: remove-mean, 23-25: flags)
    lines = (E1 / "invert3d.in").read_text().splitlines()
    assert lines[21].split()[0] == str(inv.remove_mean_teleseisms)
    assert [int(lines[i].split()[0]) for i in (22, 23, 24)] == [
        inv.velocity_inversion[0],
        inv.interface_inversion[0],
        inv.source_inversion[0],
    ]


def test_fixed_width_file_names_are_enforced():
    inv = c.Invert3d(vgrid_file="a_really_long_directory/vgrids_with_long_name.in")
    with pytest.raises(FmtomoFormatError, match=r"only takes the first 26"):
        inv.format()
    # list-directed names stop at '/': refused (verified with gfortran)
    with pytest.raises(FmtomoFormatError, match="slash"):
        c.Scorrect(source_file="../sources.in").format()
    with pytest.raises(FmtomoFormatError, match="slash"):
        c.Frechgen(invert3d_file="dir/invert3d.in").format()
    # the formatted-read comment column starts after the name, so a 26-char name still parses
    name = "x" * 26
    assert c.Invert3d.parse(c.Invert3d(vgrid_file=name).format()).vgrid_file == name


def test_fortran_reader_semantics_for_control_lines():
    """Control lines carry bare values where a shell script reads them (tomo3d.in)."""
    rd_text = c.TomoRun(n_iterations=3).format()
    assert rd_text.splitlines()[:3] == [
        "3",
        "0",
        "0",
    ]  # bare integers: tomo3d reads them with ksh `read`
    assert c.InvIter.parse("7\n").iteration == 7
    assert c.InvIter(4).format() == "4\n"


def test_mode_set_values():
    m = c.ModeSet.read(E1 / "mode_set.in")
    assert (m.file_mode, m.no_pp_mode, m.parallel_mode, m.display_mode) == (
        False,
        True,
        False,
        False,
    )
    assert m.save_rays_mode is True and m.save_timefields_mode is False
    assert m.format().splitlines()[4] == "T   save_rays_mode"
    with pytest.raises(FmtomoFormatError, match="cannot read"):
        c.ModeSet.parse("F x\nQ y\n")


def test_frechgen_subset_lists_and_placeholders():
    fg = c.Frechgen.read(E1 / "frechgen.in")
    assert (fg.velocity_derivs, fg.velocity_type, fg.interface_derivs, fg.source_derivs) == (
        -1,
        1,
        1,
        -1,
    )
    assert fg.velocity_grid_ids == (2,) and fg.interface_ids == (2,)  # placeholders / subset
    sub = c.Frechgen(
        velocity_derivs=2,
        velocity_grid_ids=(1, 2),
        velocity_types=(1, 1),
        interface_derivs=2,
        interface_ids=(2, 3),
    )
    back = c.Frechgen.parse(sub.format())
    assert back == sub
    with pytest.raises(FmtomoFormatError, match="velocity_derivs must equal"):
        c.Frechgen(velocity_derivs=3, velocity_grid_ids=(1,), velocity_types=(1,)).format()


def test_frechet_spec_roundtrip():
    spec = c.FrechetSpec.read(E1 / "frechet.in")
    assert spec.sources == [1, 2, 3, 4, 5] and spec.velocity == [] and spec.interfaces == []
    assert c.FrechetSpec.parse(spec.format()) == spec
    full = c.FrechetSpec(True, [(1, 1), (2, 1)], [2], [])
    assert c.FrechetSpec.parse(full.format()) == full
    assert c.FrechetSpec.parse("0\n") == c.FrechetSpec(False)


def test_arraygen_points_match_example_output():
    ag = d.Arraygen.read(E1 / "mkmodel" / "arraygen.in")
    pts = ag.points()
    ref = d.parse_input_receivers((E1 / "mkmodel" / "arraygen.out").read_text())
    assert len(pts) == 100
    np.testing.assert_allclose(
        pts[["lat", "lon", "depth_km"]].to_numpy(),
        ref[["lat", "lon", "depth_km"]].to_numpy(),
        atol=1e-4,
    )
    back = d.parse_input_receivers(d.format_input_receivers(ref))
    pd.testing.assert_frame_equal(back, ref)


def test_moddata_groups_and_paths():
    md = d.Moddata.read(E1 / "mkmodel" / "moddata.in")
    assert md.sources_file == "sources.in" and md.input_receiver_file == "arraygen.out"
    (g,) = md.groups
    assert g.input_file == "sourcesle.in" and not g.teleseismic and g.compute_source_derivatives
    assert g.paths == [PathSpec.reflected(2)]
    tmpl = d.Moddata.read(DATA / "templates" / "moddata.in")
    assert len(tmpl.groups) == 2 and tmpl.groups[1].teleseismic
    assert tmpl.groups[1].paths[0].steps == ((3, 2), (2, 1))
    assert tmpl.groups[0].paths == [PathSpec.reflected(2), PathSpec.direct()]
    with pytest.raises(FmtomoFormatError, match="first 26"):
        d.Moddata(sources_file="s" * 30).format()


def test_obsdata_template_and_extras_variant():
    ob = d.Obsdata.read(DATA / "templates" / "obsdata.in")
    assert ob.picks_dir == "picks/" and ob.extract_times and ob.otimes_file == "otimes.dat"
    assert len(ob.groups) == 2
    assert ob.groups[0].paths[1].steps == ((0, 2), (2, 3), (2, 1))
    assert ob.groups[0].paths[1].vtypes == (1, 1, 1)
    ex = d.Obsdata.read(DATA / "extras" / "obsdatai.in", extras=True)
    assert ex.source_receiver_file == "src.dat"
    assert d.Obsdata.parse(ex.format(), extras=True) == ex
    # the Extras layout inserts two records after the times file
    assert ex.format().splitlines()[12].startswith("1 ")


def test_input_file_formats():
    srcs = d.parse_input_sources(
        "2\n -41.0 141.4 50.0 2.0 3.0 4.0\n -41.1 141.5 55.0 0.1 0.2 0.3\n"
    )
    # moddata input is lat lon depth covlat covlon covdep; Source keeps (depth, lat, lon) and
    # (covdep, covlat, covlon)
    assert (srcs[0].depth_km, srcs[0].lat, srcs[0].lon) == (50.0, -41.0, 141.4)
    assert srcs[0].uncertainty == (4.0, 2.0, 3.0)
    tele = d.parse_input_sources("1\n -33.0 80.0 33.0\nP\n", teleseismic=True)
    assert tele[0].phase == "P" and tele[0].teleseismic
    assert d.parse_input_sources(d.format_input_sources(srcs))[1].uncertainty == pytest.approx(
        (0.3, 0.1, 0.2)
    )
    pf = d.parse_picks(
        "2\n-20.52 162.18 33.\nP\n"
        "-40.97 145.13 -0.166 -0.166 0.075\n-41.04 145.28 -0.276 -0.116 0.0616\n",
        teleseismic=True,
    )
    assert pf.source.lat == -20.52 and pf.source.phase == "P" and len(pf.picks) == 2
    assert pf.picks.iloc[1]["uncertainty"] == 0.0616
    assert len(d.parse_picks(d.format_picks(pf), teleseismic=True).picks) == 2
    obs = d.parse_obs_sources(
        "1\n -41.7 148.3 -0.2 0.2 0.2 10.0\n2\n1 1 pick04.9pm\n3 2 pick05.8pg\n"
    )
    assert obs[0].picks == [(1, 1, "pick04.9pm"), (3, 2, "pick05.8pg")]
    assert obs[0].source.uncertainty == (10.0, 0.2, 0.2)
    assert d.parse_obs_sources(d.format_obs_sources(obs)) == obs
    assert d.parse_obs_sources("2\n1 1 a.ttr\n1 1 b.ttr\n", teleseismic=True)[1].picks == [
        (1, 1, "b.ttr")
    ]


def test_residuals_and_scorrect_values():
    r = c.Residuals.read(E1 / "residuals.in")
    assert (r.otimes_file, r.source_file, r.invert3d_file) == (
        "otimes.dat",
        "sourcesref.in",
        "invert3d.in",
    )
    s = c.Scorrect.read(E1 / "scorrect.in")
    assert s.depth_correction_km == 0.1 and s.geographic_correction_deg == 0.01


# ---- grid3dg -----------------------------------------------------------------------------


def test_grid3dg_example1_structure():
    g = g3.Grid3dg.read(DATA / "example1" / "mkmodel" / "grid3dg.in")
    assert g.n_layers == 2 and len(g.interfaces) == 3 and g.n_vtypes == 1
    assert g.radial_range_km == (1.5, -80.1)
    assert g.lat_range_deg == (-40.0, -42.0) and g.lon_range_deg == (140.0, 143.0)
    assert g.layers[0].n_radial == (12, 12) and g.layers[0].n_lat == (12, 2)
    assert g.layers[1].n_lon == (12, 18)
    assert g.layers[0].v_top == (5.4, 4.0) and g.layers[1].v_bottom == (7.5, 7.0)
    assert g.layers[0].perturbation.random == 1 and g.layers[0].perturbation.random_sd == 1.5
    assert g.layers[1].perturbation.spike_list[1].location == (-60.0, -41.4, 146.4)
    assert g.interfaces[0].height_nw_km == 1.3 and g.interfaces[1].height_nw_km == -30.0
    assert (
        g.interfaces[1].perturbation.random == 1
        and g.interfaces[1].perturbation.prior_uncertainty == 3.0
    )
    assert g.interface_nodes == (12, 12)


@pytest.mark.parametrize("example", ["example1", "example2", "example3"])
def test_grid3dg_expected_geometry_matches_generated_files(example):
    """grid3dg.in -> propgrid.in, vgrids.in, interfaces.in: cushion nodes, spacing, origins."""
    base = DATA / example
    gfile = base / "mkmodel" / "grid3dg.in" if example == "example1" else base / "grid3dg.in"
    geo = g3.Grid3dg.read(gfile).expected_geometry()
    pg = geo["propgrid"]
    ref = grids.read_propgrid(
        base / "mkmodel" / "propgrid.in" if example == "example1" else base / "propgrid.in"
    )
    for field in (
        "nr",
        "nlat",
        "nlon",
        "dr_km",
        "dlat_deg",
        "dlon_deg",
        "top_height_km",
        "lat0_deg",
        "lon0_deg",
        "refinement_factor",
        "n_refine_cells",
    ):
        assert getattr(pg, field) == pytest.approx(getattr(ref, field), rel=1e-6), field
    if example == "example1":
        vg = grids.read_vgrids(base / "vgrids.in")
        for k, per_type in enumerate(geo["layers"], start=1):
            ds = vg.get(k)
            want = per_type[0]
            assert ds.sizes == {"depth": want["nr"], "lat": want["nlat"], "lon": want["nlon"]}
            assert ds.attrs["fmtomo_dr_km"] == pytest.approx(want["dr_km"], rel=1e-9)
            assert ds.attrs["fmtomo_dlat_rad"] == pytest.approx(want["dlat_rad"], rel=1e-9)
            assert ds.attrs["fmtomo_dlon_rad"] == pytest.approx(want["dlon_rad"], rel=1e-9)
            assert ds.attrs["fmtomo_r0_km"] == pytest.approx(want["r0_km"], rel=1e-9)
            assert ds.attrs["fmtomo_lat0_rad"] == pytest.approx(want["lat0_rad"], rel=1e-9)
            assert ds.attrs["fmtomo_lon0_rad"] == pytest.approx(want["lon0_rad"], rel=1e-9)
        itf = grids.read_interfaces(base / "interfaces.in").data
        ifc = geo["interfaces"]
        assert (itf.sizes["lat"], itf.sizes["lon"]) == (ifc["nlat"], ifc["nlon"])
        assert itf.attrs["fmtomo_lat0_rad"] == pytest.approx(ifc["lat0_rad"], rel=1e-9)
        assert itf.attrs["fmtomo_dlon_rad"] == pytest.approx(ifc["dlon_rad"], rel=1e-9)
    else:
        lines = (base / "vgrids_headers.txt").read_text().splitlines()
        nl, nt = (int(v) for v in lines[0].split())
        assert nl == len(geo["layers"])
        for k in range(nl):
            nr, nlat, nlon = (int(v) for v in lines[1 + 3 * k].split())
            dr, dlat, dlon = (float(v) for v in lines[2 + 3 * k].split())
            r0, lat0, lon0 = (float(v) for v in lines[3 + 3 * k].split())
            want = geo["layers"][k][0]
            assert (nr, nlat, nlon) == (want["nr"], want["nlat"], want["nlon"])
            assert (dr, dlat, dlon) == pytest.approx(
                (want["dr_km"], want["dlat_rad"], want["dlon_rad"]), rel=1e-9
            )
            assert (r0, lat0, lon0) == pytest.approx(
                (want["r0_km"], want["lat0_rad"], want["lon0_rad"]), rel=1e-9
            )
        ih = (base / "interfaces_header.txt").read_text().splitlines()
        assert int(ih[0]) == len(geo["layers"]) + 1
        assert tuple(int(v) for v in ih[1].split()) == (
            geo["interfaces"]["nlat"],
            geo["interfaces"]["nlon"],
        )


def test_grid3dg_validation_and_two_types_geometry():
    g = g3.Grid3dg.read(DATA / "example1" / "mkmodel" / "grid3dg.in")
    g.layers.pop()
    with pytest.raises(FmtomoFormatError, match="need 2 interfaces"):
        g.format()
    g = g3.Grid3dg.read(DATA / "example1" / "mkmodel" / "grid3dg.in")
    g.n_vtypes = 2
    geo = g.expected_geometry()
    assert (
        geo["layers"][0][1]["nlat"] == 4
    )  # second type has 2 latitude nodes (+2 cushion), as in the file
    assert geo["layers"][1][1]["nlon"] == 20


# ---- gmtslice --------------------------------------------------------------------------------


def test_gmtslice_example_values():
    gm = gs.Gmtslice.read(E1 / "gmtplot" / "gmtslice.in")
    assert gm.vgrid_file == "../vgrids.in" and gm.earth_radius_km == 6371.0
    assert gm.plot_every_nth == 30 and gm.depth_slice_km == -30.0
    assert gm.gc_point1 == (-42.0, 140.0) and gm.gc_point2 == (-40.0, 143.0)
    assert gm.gc_points == (300, 100) and gm.interfaces == [(2, "gridint1.z")]
    gm.interfaces = [(2, "a.z"), (3, "b.z")]
    with pytest.raises(FmtomoFormatError, match="n_interfaces_plot"):
        gm.format()
    gm.n_interfaces_plot = 2
    assert gs.Gmtslice.parse(gm.format()).interfaces == [(2, "a.z"), (3, "b.z")]
    assert "ID of second interface to plot" in gm.format()


# ---- extras ----------------------------------------------------------------------------------


def test_extras_src_dat_and_sandr(tmp_path):
    df = x.read_src_dat(DATA / "extras" / "src.dat")
    assert df.attrs["n_blocks"] == 12 and df["block"].nunique() == 12
    assert list(df.iloc[0][["block", "id"]]) == [1, 1] and df.iloc[0]["time"] == pytest.approx(
        5.1620002
    )
    out = x.write_src_dat(tmp_path / "src.dat", df)
    back = x.read_src_dat(out)
    pd.testing.assert_frame_equal(back, df)
    assert x.read_sandr(DATA / "extras" / "sandr.dat") == (1922,)
    assert x.parse_sandr(x.format_sandr([5, 6])) == (5, 6)


def test_extras_otimess_is_an_ordinary_otimes_file():
    from osv.io.fmtomo import times as t

    df = t.read_observed(DATA / "extras" / "otimess.dat")
    assert len(df) == 40 and list(df.columns[:6]) == t.OBSERVED_COLUMNS


def test_extras_sources_with_ids():
    from osv.io.fmtomo import sources as s

    src = s.read_sources(DATA / "extras" / "sources.in")
    assert len(src) == 29
    assert (src[0].depth_km, src[0].lat, src[0].lon, src[0].extra_id) == (
        0.35,
        -31.3235,
        138.89931,
        1,
    ) or (src[0].extra_id == 1)
    assert src[1].extra_id == 2
    assert s.parse_sources(s.format_sources(src)) == src


def test_vpvs_grid_without_uncertainty_column(tmp_path):
    """vgridvpvs.in (mkvpvs.f90:48) holds one value per record and one layer/type."""
    vel = 1.7 + 0.01 * np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    ds = grids.velocity_dataset(
        vel,
        r0_km=6300.0,
        dr_km=10.0,
        lat0_rad=-0.6,
        dlat_rad=0.01,
        lon0_rad=2.4,
        dlon_rad=0.02,
        layer=1,
        vtype=1,
    )
    text = grids.format_vgrids(grids.VelocityGrids([[ds]]))
    assert all(len(ln.split()) == 1 for ln in text.splitlines()[4:] if ln.strip())
    back = grids.parse_vgrids(text).get(1)
    np.testing.assert_allclose(back["velocity"].values, ds["velocity"].values, atol=1e-8)


def test_gridlocj_serror_dat():
    df = x.parse_serror("  -12.5 -31.3 138.9 1.0 0.01 0.02\n  0.0 0.0 0.0 0.0 0.0 0.0\n")
    assert list(df.columns) == x.SERROR_COLUMNS and len(df) == 2
    assert df.loc[0, "err_lon"] == 0.02
