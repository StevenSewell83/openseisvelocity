"""Config -> FMTOMO control-file mapping (``osv.fmtomo.controls``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from osv.config.schema import FmtomoConfig
from osv.fmtomo.controls import ModelInfo, build_controls, phase_path
from osv.io.fmtomo import Frechgen, Invert3d

E1 = Path(__file__).resolve().parents[1] / "data" / "fmtomo" / "example1"


def example1_cfg(**kw) -> FmtomoConfig:
    """Config that mirrors fortran/fmtomo/example1 (source relocation only, no smoothing)."""
    base = {
        "package": "base",
        "phases": ["P"],
        "relocate": "linear",
        "iterations": 6,
        "damping": 1.0,
        "smoothing": 0.0,
        "invert_velocity": False,
    }
    base.update(kw)
    return FmtomoConfig(**base)


def test_example1_control_files_reproduced():
    run = build_controls(example1_cfg(), ModelInfo(n_layers=2)).run
    texts = run.texts()
    assert run.driver == "tomo3d"
    for name in ("invert3d.in", "residuals.in", "scorrect.in", "tomo3d.in", "mode_set.in"):
        assert texts[name].rstrip() == (E1 / name).read_text().rstrip(), name
    assert texts["inviter.in"] == "1\n"


def test_example1_frechgen_effective_content_matches():
    # example1's frechgen.in selects derivative classes whose invert3d flag is 0; frechgen
    # ignores them, so only the flags that matter are compared
    mine = build_controls(example1_cfg(), ModelInfo(n_layers=2)).run
    ex = Frechgen.read(E1 / "frechgen.in")
    assert mine.frechgen.source_derivs == ex.source_derivs == -1
    assert mine.frechgen.source_ids_file == ex.source_ids_file
    assert Invert3d.read(E1 / "invert3d.in") == mine.invert3d


def test_default_extras_nonlinear():
    c = build_controls(FmtomoConfig(), ModelInfo(n_layers=3))
    r = c.run
    assert (c.driver, c.package, c.relocate) == ("tomo3di", "extras", "nonlinear")
    assert c.stations_are_sources
    assert r.tomo_run_name == "tomo3di.in"
    assert r.tomo_run.n_iterations == 6
    assert r.mode_set.save_timefields_mode and r.mode_set.no_pp_mode
    assert r.scorrect is None  # tomo3di does not call scorrect
    inv = r.invert3d
    assert inv.velocity_inversion == (1, 1.0, 1.0)
    assert inv.source_inversion[0] == 0  # gridloc relocates, not the subspace inversion
    assert (inv.global_damping, inv.global_smoothing) == (50.0, 100.0)
    assert inv.second_derivative_smoothing == 1
    assert (r.frechgen.velocity_derivs, r.frechgen.source_derivs) == (-1, 0)
    g = r.gridloc
    assert (g.min_stations, g.max_origin_time_perturbation) == (5, 5.0)
    assert (g.refined_search, g.subcell_dicing) == (1, 10)
    assert set(r.texts()) >= {"gridloc.in", "tomo3di.in", "mode_set.in", "invert3d.in"}


def test_damping_smoothing_subspace_mapping():
    r = build_controls(
        FmtomoConfig(damping=7.5, smoothing=0, subspace_dimension=33, relocate="none")
    ).run
    assert r.invert3d.global_damping == 7.5
    assert r.invert3d.second_derivative_smoothing == 0
    assert r.invert3d.subspace_dimension == 33
    assert r.invert3d.source_inversion[0] == 0
    assert r.gridloc is None and r.driver == "tomo3d"
    assert r.scorrect is not None


def test_linear_relocation_flags_and_source_damping():
    cfg = FmtomoConfig(
        package="base", relocate="linear", source_damping_position=2.0, source_damping_time=0.5
    )
    r = build_controls(cfg, ModelInfo(invert_source_ids=(2, 4))).run
    assert r.invert3d.source_inversion == (1, 2.0, 0.5)
    assert r.frechgen.source_derivs == 2 and r.source_ids == (2, 4)
    assert r.needs_source_uncertainty
    assert not r.mode_set.save_timefields_mode


def test_layer_subset_roundtrips_through_frechgen():
    cfg = FmtomoConfig(relocate="none", package="base")
    r = build_controls(cfg, ModelInfo(n_layers=4, invert_layers=(2, 3))).run
    assert (r.frechgen.velocity_derivs, r.frechgen.velocity_grid_ids) == (2, (2, 3))
    assert r.frechgen.velocity_types == (1, 1)
    assert Frechgen.parse(r.frechgen.format()) == r.frechgen
    # all layers -> -1
    r = build_controls(cfg, ModelInfo(n_layers=4)).run
    assert r.frechgen.velocity_derivs == -1


def test_interface_inversion():
    cfg = FmtomoConfig(
        relocate="none", package="base", invert_interfaces=[2], invert_velocity=False
    )
    r = build_controls(cfg, ModelInfo(n_layers=2)).run
    assert r.invert3d.interface_inversion[0] == 1 and r.invert3d.velocity_inversion[0] == 0
    assert (r.frechgen.interface_derivs, r.frechgen.interface_ids) == (1, (2,))
    with pytest.raises(ValueError, match="exceed"):
        build_controls(cfg.model_copy(update={"invert_interfaces": [3]}), ModelInfo(n_layers=1))


def test_subspace_dimension_raised_to_minimum():
    cfg = example1_cfg(subspace_dimension=1, invert_velocity=True, invert_interfaces=[2])
    inv = build_controls(cfg, ModelInfo(n_layers=2)).run.invert3d
    assert inv.subspace_dimension == 4  # velocity + interface + 2 * source flags


def test_nothing_to_invert_rejected():
    with pytest.raises(ValueError, match="nothing to invert"):
        build_controls(example1_cfg(relocate="none"))


def test_base_package_rejects_extras_features():
    with pytest.raises(ValidationError, match="base"):
        FmtomoConfig(package="base", relocate="nonlinear")
    with pytest.raises(ValidationError, match="base"):
        FmtomoConfig(package="base", relocate="none", phases=["vpvs"])
    fake = SimpleNamespace(**{**FmtomoConfig().model_dump(), "package": "base"})
    with pytest.raises(ValueError, match="base"):
        build_controls(fake)


def test_gridloc_config_mapping():
    cfg = FmtomoConfig(
        gridloc={
            "min_stations": 8,
            "max_origin_time_shift_s": 2.5,
            "refine": False,
            "subcell_dicing": 6,
        }
    )
    g = build_controls(cfg).run.gridloc
    assert (g.min_stations, g.max_origin_time_perturbation) == (8, 2.5)
    assert (g.refined_search, g.subcell_dicing) == (0, 6)


def test_separate_p_and_s_runs():
    c = build_controls(FmtomoConfig(package="base", relocate="linear", phases=["P", "S"]))
    assert set(c.runs) == {"P", "S"} and c.joint is None and not c.experimental
    assert c.driver == "tomo3d"
    assert any("independent" in n for n in c.notes)
    with pytest.raises(ValueError, match="pick one"):
        c.run  # noqa: B018
    # the S-only run holds S as velocity type 1
    s = build_controls(FmtomoConfig(phases=["P", "S"]), phases=["S"])
    assert list(s.runs) == ["S"]
    assert phase_path("S").vtypes == (1,)
    assert phase_path("S", layout="joint").vtypes == (2,)
    assert phase_path("P", layout="joint").vtypes == (1,)
    with pytest.raises(ValueError):
        phase_path("vpvs")


def test_joint_p_s_nonlinear_is_tomo3dj_experimental():
    c = build_controls(
        FmtomoConfig(phases=["P", "S"], joint={"s_weight": 4.0, "relocation_based_on": 4})
    )
    assert c.driver == "tomo3dj" and c.experimental
    j = c.joint
    assert j.gridlocj.s_weight == 4.0 and j.gridlocj.relocation_based_on == 4
    assert j.gridlocj.arrtimes_p_file == "invert_p/arrtimes.dat"
    assert set(j.texts()) == {"tomo3dj.in", "gridlocj.in", "jointimes.in"}
    assert set(c.runs) == {"P", "S"}


def test_vpvs_is_tomo3dps_experimental():
    c = build_controls(FmtomoConfig(phases=["vpvs"], damping=3, smoothing=0.5))
    assert c.driver == "tomo3dps" and c.experimental
    v = c.vpvs
    assert v.vpvsinv.damping == 3 and v.vpvsinv.smoothing == 0.5
    assert v.vpvsinv.second_derivative_smoothing == 1
    assert v.mkvpvs.max_depth_km == 50.0
    assert set(v.texts()) == {
        "tomo3dps.in",
        "mkvpvs.in",
        "frvpvs.in",
        "vpvsinv.in",
        "vpvsresids.in",
    }
    assert any("experimental" in n for n in c.notes)


def test_station_terms_noted():
    c = build_controls(FmtomoConfig(station_terms="preapply"))
    assert c.station_terms == "preapply"
    assert any("station terms" in n for n in c.notes)


def test_bad_phases_and_model_info():
    with pytest.raises(ValueError, match="phases"):
        build_controls(FmtomoConfig(), phases=["P", "P"])
    with pytest.raises(ValueError):
        ModelInfo(n_layers=2, invert_layers=(3,))
    with pytest.raises(ValidationError):
        FmtomoConfig(invert_interfaces=[0])
