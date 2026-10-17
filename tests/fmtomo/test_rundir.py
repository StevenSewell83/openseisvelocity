"""Assembling FMTOMO run directories (``osv.fmtomo.rundir``), incl. a real ``tomo3d`` run."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import FmtomoConfig
from osv.fmtomo.controls import ModelInfo, build_controls
from osv.fmtomo.rundir import (
    RunDirError,
    RunInputs,
    assemble_joint_run_dirs,
    assemble_run_dir,
    derive_src_dat,
    find_ak135_dir,
)
from osv.io.fmtomo import (
    Invert3d,
    InvIter,
    ModeSet,
    read_gridsave,
    read_interfaces,
    read_observed,
    read_propgrid,
    read_receivers,
    read_residuals,
    read_sources,
    read_vgrids,
)
from osv.io.fmtomo.extras import read_src_dat

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "tests" / "data" / "fmtomo" / "example1"
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))


@pytest.fixture(scope="module")
def ex1():
    return {
        "grids": read_vgrids(E1 / "vgrids.in"),
        "interfaces": read_interfaces(E1 / "interfaces.in"),
        "propgrid": read_propgrid(E1 / "propgrid.in"),
        "sources": read_sources(E1 / "sourcesref.in"),
        "receivers": read_receivers(E1 / "receivers.in"),
        "otimes": read_observed(E1 / "otimes.dat"),
    }


def _cfg(**kw) -> FmtomoConfig:
    base = {
        "package": "base",
        "relocate": "linear",
        "damping": 1.0,
        "smoothing": 0.0,
        "invert_velocity": False,
        "iterations": 6,
    }
    base.update(kw)
    return FmtomoConfig(**base)


def _assemble(ex, root, controls, **kw):
    return assemble_run_dir(
        root,
        ex["grids"],
        ex["interfaces"],
        ex["propgrid"],
        ex["sources"],
        ex["receivers"],
        ex["otimes"],
        controls,
        **kw,
    )


def test_example1_run_dir_matches_example_files(ex1, tmp_path):
    ctl = build_controls(_cfg(), ModelInfo.from_model(ex1["grids"]))
    man = _assemble(ex1, tmp_path, ctl)
    assert man.driver == "tomo3d" and man.tomo_input == "tomo3d.in"
    for name in (
        "invert3d.in",
        "residuals.in",
        "scorrect.in",
        "tomo3d.in",
        "mode_set.in",
        "sourcederivs.in",
    ):
        assert (tmp_path / name).read_text().rstrip() == (E1 / name).read_text().rstrip(), name
    # data files: same content as the example (the float layout of the writers differs)
    rec = read_receivers(tmp_path / "receivers.in")
    exp_rec = read_receivers(E1 / "receivers.in")
    pd.testing.assert_frame_equal(rec.positions, exp_rec.positions)
    pd.testing.assert_frame_equal(rec.rays, exp_rec.rays)
    pd.testing.assert_frame_equal(read_observed(tmp_path / "otimes.dat"), ex1["otimes"])
    assert read_sources(tmp_path / "sourcesref.in") == ex1["sources"]
    assert read_sources(tmp_path / "sources.in") == ex1["sources"]
    assert read_interfaces(tmp_path / "interfaces.in").data.identical(ex1["interfaces"].data)
    got_v = read_vgrids(tmp_path / "vgrids.in")
    for layer in (1, 2):
        np.testing.assert_allclose(
            got_v.get(layer)["velocity"].values, ex1["grids"].get(layer)["velocity"].values
        )
    assert read_propgrid(tmp_path / "propgrid.in") == ex1["propgrid"]
    # reference copies and the current files the drivers start from
    for name in ("vgridsref.in", "interfacesref.in", "sourcesref.in", "sources.in"):
        assert name in man.files and man.files[name].is_file()
    assert (tmp_path / "vgridsref.in").read_text() == (tmp_path / "vgrids.in").read_text()
    assert ModeSet.read(tmp_path / "mode_set.in").save_rays_mode
    assert Invert3d.read(tmp_path / "invert3d.in").source_inversion[0] == 1
    assert InvIter.read(tmp_path / "inviter.in").iteration == 1
    assert man.names == sorted(man.files)
    j = man.as_json()
    assert j["driver"] == "tomo3d" and "vgrids.in" in j["files"]
    # nothing the nonlinear driver needs
    for name in ("gridloc.in", "src.dat", "gridsave.in", "ak135.tbl", "otimesref.dat"):
        assert name not in man.files


def test_ref_false_skips_reference_files(ex1, tmp_path):
    ctl = build_controls(_cfg())
    man = _assemble(ex1, tmp_path, ctl, ref=False)
    assert not any(n.endswith("ref.in") for n in man.files)
    assert "vgrids.in" in man.files


def test_nonlinear_run_dir(ex1, tmp_path):
    ctl = build_controls(FmtomoConfig(invert_velocity=True), ModelInfo.from_model(ex1["grids"]))
    man = _assemble(ex1, tmp_path, ctl)
    assert man.tomo_input == "tomo3di.in"
    for name in ("gridloc.in", "src.dat", "gridsave.in", "receiversref.in", "otimesref.dat"):
        assert (tmp_path / name).is_file(), name
    assert "scorrect.in" not in man.files and "sourcederivs.in" not in man.files
    src = read_src_dat(tmp_path / "src.dat")
    assert src.attrs["n_blocks"] == 100 and len(src) == int((ex1["otimes"]["time"] > 0).sum())
    assert set(src["block"]) == {1, 2, 3, 4, 5} and set(src["id"]) <= set(range(1, 101))
    gs = read_gridsave(tmp_path / "gridsave.in")
    assert [(g.source, g.paths) for g in gs] == [(i, (1,)) for i in range(1, 6)]
    assert read_observed(tmp_path / "otimesref.dat").equals(
        ex1["otimes"][list(read_observed(tmp_path / "otimesref.dat").columns)]
    )


def test_nonlinear_does_not_need_source_uncertainty(ex1, tmp_path):
    srcs = copy.deepcopy(ex1["sources"])
    for s in srcs:
        s.uncertainty = None
    ctl = build_controls(FmtomoConfig(invert_velocity=True))
    assemble_run_dir(
        tmp_path,
        ex1["grids"],
        ex1["interfaces"],
        ex1["propgrid"],
        srcs,
        ex1["receivers"],
        ex1["otimes"],
        ctl,
    )


def test_derive_src_dat_skips_invalid_times(ex1):
    ot = ex1["otimes"].copy()
    ot.loc[0, "time"] = -100.0
    df = derive_src_dat(ot, 100)
    assert len(df) == int((ex1["otimes"]["time"] > 0).sum()) - 1 and df.attrs["n_blocks"] == 100


def test_derive_src_dat_blocks_are_sources_ids_are_events():
    ot = pd.DataFrame(
        {
            "receiver": [1, 2, 3],
            "source": [2, 1, 1],
            "time": [1.0, 2.0, -1.0],
            "uncertainty": [0.1, 0.1, 0.1],
        }
    )
    df = derive_src_dat(ot, 2, event_ids=[7, 9, 4])
    assert df["block"].tolist() == [1, 2] and df["id"].tolist() == [9, 7]
    assert derive_src_dat(ot, 2)["id"].tolist() == [2, 1]


def test_validation_lists_every_problem(ex1, tmp_path):
    ctl = build_controls(_cfg())
    bad = dict(ex1)
    bad["otimes"] = ex1["otimes"].iloc[:-3]
    srcs = copy.deepcopy(ex1["sources"])
    srcs[0].uncertainty = None
    bad["sources"] = srcs
    with pytest.raises(RunDirError) as ei:
        _assemble(bad, tmp_path, ctl)
    msg = str(ei.value)
    assert "otimes has 497 rows" in msg and "uncertainties" in msg


def test_validation_otimes_keys_and_receivers(ex1, tmp_path):
    ctl = build_controls(_cfg())
    bad = dict(ex1)
    ot = ex1["otimes"].copy()
    ot.loc[0, "source"] = 4
    bad["otimes"] = ot
    with pytest.raises(RunDirError, match="keys differ"):
        _assemble(bad, tmp_path, ctl)
    rec = copy.deepcopy(ex1["receivers"])
    rec.rays.loc[0, "source"] = 99
    bad = dict(ex1, receivers=rec)
    with pytest.raises(RunDirError, match="non-existent source 99"):
        _assemble(bad, tmp_path, ctl)


def test_validation_layers_and_source_ids(ex1, tmp_path):
    ctl = build_controls(_cfg(invert_velocity=True), ModelInfo(n_layers=5, invert_layers=(5,)))
    with pytest.raises(RunDirError, match="exceed the 2 velocity grids"):
        _assemble(ex1, tmp_path, ctl)
    ctl = build_controls(_cfg(), ModelInfo(invert_source_ids=(1, 9)))
    with pytest.raises(RunDirError, match="not among the 5 sources"):
        _assemble(ex1, tmp_path, ctl)


def test_subset_source_ids_written(ex1, tmp_path):
    ctl = build_controls(_cfg(), ModelInfo(invert_source_ids=(2, 5)))
    _assemble(ex1, tmp_path, ctl)
    assert (tmp_path / "sourcederivs.in").read_text().split() == ["2", "2", "5"]
    assert ctl.run.frechgen.source_derivs == 2


def test_station_terms_preapply(ex1, tmp_path):
    ctl = build_controls(_cfg(station_terms="preapply"))
    with pytest.raises(RunDirError, match="no station terms"):
        _assemble(ex1, tmp_path, ctl)
    with pytest.raises(RunDirError, match="unknown station"):
        _assemble(ex1, tmp_path, ctl, station_terms={999: 0.1})
    _assemble(ex1, tmp_path, ctl, station_terms={1: 0.25, 2: -0.5})
    got = read_observed(tmp_path / "otimes.dat")
    exp = ex1["otimes"]
    d = (exp["time"] - got["time"]).round(6)
    assert np.allclose(d[exp["receiver"] == 1], 0.25) and np.allclose(d[exp["receiver"] == 2], -0.5)
    assert np.allclose(d[~exp["receiver"].isin([1, 2])], 0.0)
    with pytest.raises(RunDirError, match="fmtomo.station_terms is 'none'"):
        _assemble(ex1, tmp_path, build_controls(_cfg()), station_terms={1: 0.1})


def test_teleseismic_copies_ak135(ex1, tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    for n in ("ak135.tbl", "ak135.hed"):
        (tables / n).write_text("x")
    srcs = copy.deepcopy(ex1["sources"])
    srcs[0].teleseismic, srcs[0].phase = True, "P"
    ctl = build_controls(_cfg())
    man = assemble_run_dir(
        tmp_path / "run",
        ex1["grids"],
        ex1["interfaces"],
        ex1["propgrid"],
        srcs,
        ex1["receivers"],
        ex1["otimes"],
        ctl,
        ak135_dir=tables,
    )
    assert (tmp_path / "run" / "ak135.tbl").read_text() == "x" and "ak135.hed" in man.files
    assert find_ak135_dir(tables) == tables


def test_joint_run_dirs(ex1, tmp_path):
    ctl = build_controls(FmtomoConfig(phases=["P", "S"], invert_velocity=True))
    inp = RunInputs(
        ex1["grids"],
        ex1["interfaces"],
        ex1["propgrid"],
        ex1["sources"],
        ex1["receivers"],
        ex1["otimes"],
    )
    man = assemble_joint_run_dirs(tmp_path, {"P": inp, "S": inp}, ctl)
    for n in (
        "tomo3dj.in",
        "gridlocj.in",
        "jointimes.in",
        "invert_p/gridloc.in",
        "invert_s/src.dat",
    ):
        assert n in man.files and man.files[n].is_file(), n
    with pytest.raises(RunDirError, match="no joint"):
        assemble_joint_run_dirs(tmp_path, {"P": inp, "S": inp}, build_controls(_cfg()))


@pytest.mark.integration
def test_tomo3d_driver_reduces_residuals(ex1, tmp_path):
    for exe in ("tomo3d", "fm3d", "frechgen", "invert3d", "residuals", "scorrect"):
        if not (BIN / exe).exists():
            pytest.skip(f"{BIN / exe} not built")
    ctl = build_controls(_cfg(iterations=1), ModelInfo.from_model(ex1["grids"]))
    man = _assemble(ex1, tmp_path, ctl)
    env = {**os.environ, "PATH": f"{BIN}{os.pathsep}{os.environ['PATH']}"}
    r = subprocess.run(
        [str(BIN / "tomo3d")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    res = read_residuals(tmp_path / "residuals.dat")
    assert len(res) == 2  # start + 1 iteration
    assert res["rms_ms"].iloc[-1] < res["rms_ms"].iloc[0]
    assert res["rms_ms"].is_monotonic_decreasing
    # frechgen produced what example1 shipped (sources only)
    assert (tmp_path / "frechet.in").read_text().split() == (E1 / "frechet.in").read_text().split()
    assert man.tomo_input == "tomo3d.in"
    shutil.rmtree(tmp_path, ignore_errors=True)
