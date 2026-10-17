from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from osv.cli import app
from osv.config import (
    ConfigError,
    OSVConfig,
    config_hash,
    dump_config,
    generate_template,
    load_config,
    resolve_config,
    write_template,
)

runner = CliRunner()


def _load_dict(tmp_path: Path, data: dict, **kw) -> OSVConfig:
    return resolve_config(data, tmp_path, **kw)


def _errors(data: dict) -> dict[str, str]:
    with pytest.raises(ConfigError) as ei:
        resolve_config(data, ".")
    return dict(ei.value.errors)


# ---------------------------------------------------------------- template / round trip


def test_template_round_trips(tmp_path: Path) -> None:
    target = write_template(tmp_path / "proj")
    cfg = load_config(target)
    assert cfg.project.name == "proj"
    assert cfg.project.workdir == (tmp_path / "proj" / "runs")
    # Template defaults equal schema defaults (apart from placeholder inputs).
    default = OSVConfig()
    assert cfg.velest == default.velest
    assert cfg.model3d == default.model3d
    assert cfg.fmtomo == default.fmtomo
    assert cfg.plotting == default.plotting


def test_template_mentions_every_key_with_comments() -> None:
    text = generate_template()
    keys = ["vp_bounds", "layers_top_km", "weight_source", "propgrid_refine", "dws_mask_threshold"]
    for key in keys:
        assert f"{key}:" in text
    assert "# Allowed: fixed | sampled | independent_s_model" in text
    assert "# Example:" in text
    yaml.safe_load(text)


def test_init_refuses_overwrite_and_force(tmp_path: Path) -> None:
    d = tmp_path / "p"
    assert runner.invoke(app, ["init", str(d)]).exit_code == 0
    r = runner.invoke(app, ["init", str(d)])
    assert r.exit_code == 1
    assert "--force" in r.output
    assert runner.invoke(app, ["init", str(d), "--force", "--name", "other"]).exit_code == 0
    assert load_config(d / "config.yaml").project.name == "other"


def test_config_show(tmp_path: Path) -> None:
    runner.invoke(app, ["init", str(tmp_path / "p")])
    r = runner.invoke(app, ["config", "show", "--config", str(tmp_path / "p" / "config.yaml")])
    assert r.exit_code == 0
    shown = yaml.safe_load(r.output)
    assert shown["velest"]["n_models"] == 1000
    assert str(tmp_path) in shown["project"]["workdir"]


def test_config_show_bad_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("velest: {n_modls: 3}\n")
    r = runner.invoke(app, ["config", "show", "-c", str(tmp_path / "config.yaml")])
    assert r.exit_code == 2
    assert "velest.n_modls" in r.output


# ---------------------------------------------------------------- errors


def test_unknown_key_suggests() -> None:
    errs = _errors({"velest": {"n_modls": 5}})
    assert "did you mean 'n_models'" in errs["velest.n_modls"]


def test_unknown_top_level_key() -> None:
    assert "unknown key" in _errors({"velst": {}})["velst"]


def test_bad_enum_names_key() -> None:
    errs = _errors({"inputs": {"weight_source": "magic"}})
    assert "inputs.weight_source" in errs


def test_multiple_errors_reported() -> None:
    errs = _errors({"fmtomo": {"iterations": 0}, "plotting": {"dpi": 5}})
    assert {"fmtomo.iterations", "plotting.dpi"} <= set(errs)


def test_error_message_text() -> None:
    with pytest.raises(ConfigError) as ei:
        resolve_config({"velest": {"n_models": 0}}, ".", source="x.yaml")
    assert "x.yaml" in str(ei.value)
    assert "velest.n_models" in str(ei.value)


def test_non_mapping_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        resolve_config([1, 2], ".")
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    (tmp_path / "bad.yaml").write_text("a: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(tmp_path / "bad.yaml")


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text("")
    assert load_config(tmp_path / "c.yaml").velest.n_models == 1000


# ---------------------------------------------------------------- per-section validators


def test_project() -> None:
    assert "project.name" in _errors({"project": {"name": "bad name!"}})
    assert "project.n_jobs" in _errors({"project": {"n_jobs": 0}})


def test_inputs() -> None:
    assert "inputs.origin_selection" in _errors({"inputs": {"origin_selection": "oldest"}})
    cfg = resolve_config({"inputs": {"origin_selection": "by_method:NLLoc"}}, ".")
    assert cfg.inputs.origin_selection == "by_method:NLLoc"
    cfg = resolve_config({"inputs": {"station_aliases": {"ABCDE": "ABCD"}}}, ".")
    assert cfg.inputs.station_aliases == {"ABCDE": "ABCD"}


def test_coordinates() -> None:
    assert "coordinates.origin.lat" in _errors({"coordinates": {"origin": {"lat": 95, "lon": 0}}})
    assert "coordinates.origin.lon" in _errors({"coordinates": {"origin": {"lat": 0, "lon": 200}}})
    assert "coordinates.projection" in _errors({"coordinates": {"projection": "utm"}})
    cfg = resolve_config({"coordinates": {"projection": "epsg:2193"}}, ".")
    assert cfg.coordinates.projection == "epsg:2193"


def test_data_selection() -> None:
    assert "data_selection.depth_km" in _errors({"data_selection": {"depth_km": [5, 1]}})
    assert "data_selection.validation_fraction" in _errors(
        {"data_selection": {"validation_fraction": 1.0}}
    )
    assert "data_selection.region_polygon" in _errors(
        {"data_selection": {"region_polygon": [[0, 0], [1, 1]]}}
    )
    e = _errors({"data_selection": {"region_polygon": [[0, 0], [1, 1], [95, 3]]}})
    assert "data_selection.region_polygon[2]" in e
    assert "data_selection.pick_uncertainty_to_weight" in _errors(
        {"data_selection": {"pick_uncertainty_to_weight": {0.1: 3, 0.2: 1}}}
    )
    cfg = resolve_config({"data_selection": {"pick_uncertainty_to_weight": {0.05: 0, 0.1: 2}}}, ".")
    assert cfg.data_selection.pick_uncertainty_to_weight == {0.05: 0, 0.1: 2}


def test_velest_layers_and_bounds() -> None:
    e = _errors({"velest": {"layers_top_km": [0, 1, 1], "vp_bounds": [[1, 2]] * 3}})
    assert "increasing" in e["velest.layers_top_km"]
    e = _errors({"velest": {"layers_top_km": [0, 1], "vp_bounds": [[1, 2]]}})
    assert "layers_top_km" in e["velest.vp_bounds"]
    e = _errors({"velest": {"layers_top_km": [0, 1, 2], "vp_bounds": [[1, 2], [3, 3], [4, 5]]}})
    assert e["velest.vp_bounds[1]"].startswith("lower bound must be < upper")


def test_velest_monotonic_infeasible() -> None:
    d = {"layers_top_km": [0, 1], "vp_bounds": [[4, 5], [1, 2]]}
    assert "velest.vp_bounds[1]" in _errors({"velest": d})
    assert resolve_config({"velest": {**d, "monotonic": False}}, ".")


def test_velest_misc() -> None:
    assert "velest.vpvs.range" in _errors({"velest": {"vpvs": {"range": [1.9, 1.7]}}})
    assert "velest.vpvs.value" in _errors({"velest": {"vpvs": {"value": 0.9}}})
    e = _errors({"velest": {"select": {"method": "best_percent", "value": 150}}})
    assert "velest.select.value" in e
    assert "velest.sampler" in _errors({"velest": {"sampler": "grid"}})
    assert "velest.damping.vthet" in _errors({"velest": {"damping": {"vthet": -1}}})


def test_model3d_grid() -> None:
    assert "model3d.grid.dx_km" in _errors({"model3d": {"grid": {"dx_km": 0}}})
    assert "model3d.propgrid_refine" in _errors({"model3d": {"propgrid_refine": 0}})


HORIZON = {"name": "A", "surface": "a.xyz", "vp": {"law": "constant", "v0": 3.0}}


@pytest.mark.parametrize(
    ("source", "user_model", "fragment"),
    [
        ("user_grid", {}, "user_model.grid_file"),
        ("geology", {}, "user_model.horizons"),
        ("wells", {}, "user_model.wells.las_dir"),
        ("geology+wells", {"horizons": [HORIZON]}, "user_model.wells.las_dir"),
    ],
)
def test_model3d_source_requires_inputs(source, user_model, fragment) -> None:
    e = _errors({"model3d": {"source": source, "user_model": user_model}})
    assert f"model3d.{fragment}" in e


def test_model3d_source_satisfied(tmp_path: Path) -> None:
    cfg = _load_dict(
        tmp_path,
        {
            "model3d": {
                "source": "geology+wells",
                "user_model": {
                    "horizons": [
                        HORIZON,
                        {
                            "name": "B",
                            "surface": "b.xyz",
                            "vp": {"law": "gradient", "v0": 2, "k": 0.5},
                        },
                    ],
                    "wells": {"las_dir": "wells"},
                },
            }
        },
    )
    um = cfg.model3d.user_model
    assert um.horizons[0].surface == tmp_path / "a.xyz"
    assert um.wells.las_dir == tmp_path / "wells"


def test_horizon_law_validation() -> None:
    bad = {"name": "A", "surface": "a", "vp": {"law": "gradient", "v0": 2}}
    e = _errors({"model3d": {"user_model": {"horizons": [bad]}}})
    assert any(k.startswith("model3d.user_model.horizons[0].vp") for k in e)
    bad = {"name": "A", "surface": "a", "vp": {"law": "quadratic", "v0": 2}}
    assert _errors({"model3d": {"user_model": {"horizons": [bad]}}})


def test_fmtomo() -> None:
    assert "fmtomo.phases" in _errors({"fmtomo": {"phases": ["P", "vpvs"]}})
    assert "fmtomo.phases" in _errors({"fmtomo": {"phases": ["P", "P"]}})
    assert "fmtomo.phases" in _errors({"fmtomo": {"phases": []}})
    assert "fmtomo.relocate" in _errors({"fmtomo": {"relocate": "full"}})
    assert "fmtomo.tradeoff.damping" in _errors({"fmtomo": {"tradeoff": {"damping": [1, -2]}}})
    assert "fmtomo.tradeoff.smoothing" in _errors({"fmtomo": {"tradeoff": {"smoothing": []}}})
    assert resolve_config({"fmtomo": {"phases": ["vpvs"]}}, ".")


def test_resolution() -> None:
    assert "resolution.checkerboard.amplitude_pct" in _errors(
        {"resolution": {"checkerboard": {"amplitude_pct": 150}}}
    )
    assert "resolution.checkerboard.cell_km" in _errors(
        {"resolution": {"checkerboard": {"cell_km": [0]}}}
    )
    e = _errors({"resolution": {"spikes": [{"x_km": 0, "y_km": 0, "z_km": 1}]}})
    assert "resolution.spikes[0].radius_km" in e
    assert "resolution.input_models[0]" in _errors({"resolution": {"input_models": ["bogus"]}})


def test_plotting() -> None:
    assert "plotting.dpi" in _errors({"plotting": {"dpi": 5000}})
    assert "plotting.format[0]" in _errors({"plotting": {"format": ["gif"]}})
    prof = {"name": "A", "start": [-38.6, 176.2], "end": [-38.5, 176.2]}
    assert resolve_config({"plotting": {"profiles": [prof]}}, ".")
    e = _errors({"plotting": {"profiles": [{**prof, "end": [-138.5, 176.2]}]}})
    assert "plotting.profiles[0].end" in e
    e = _errors({"plotting": {"profiles": [{**prof, "end": prof["start"]}]}})
    assert "plotting.profiles[0].end" in e


# ---------------------------------------------------------------- loader behaviour


def test_relative_paths_resolved_against_config_dir(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    cfgfile = tmp_path / "sub" / "config.yaml"
    cfgfile.write_text("project: {workdir: ./out}\ninputs: {quakeml: [cat.xml], dem: ../dem.tif}\n")
    cfg = load_config(cfgfile)
    assert cfg.project.workdir == tmp_path / "sub" / "out"
    assert cfg.inputs.quakeml == [tmp_path / "sub" / "cat.xml"]
    assert cfg.inputs.dem == tmp_path / "dem.tif"


def test_absolute_paths_untouched(tmp_path: Path) -> None:
    cfg = _load_dict(tmp_path, {"inputs": {"dem": "/data/dem.tif"}})
    assert cfg.inputs.dem == Path("/data/dem.tif")


def test_glob_expansion(tmp_path: Path) -> None:
    st = tmp_path / "st"
    st.mkdir()
    for n in ("b", "a", "c"):
        (st / f"{n}.xml").write_text("<x/>")
    (st / "skip.txt").write_text("")
    (tmp_path / "config.yaml").write_text(
        "inputs: {stationxml: ['st/*.xml', 'st/a.xml'], quakeml: [cat.xml]}\n"
    )
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.inputs.stationxml == [st / "a.xml", st / "b.xml", st / "c.xml"]  # sorted, deduped


def test_glob_no_match(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data = {"inputs": {"stationxml": ["none/*.xml"]}}
    cfg = _load_dict(tmp_path, data)
    assert cfg.inputs.stationxml == [tmp_path / "none" / "*.xml"]
    assert "no files match" in caplog.text
    with pytest.raises(ConfigError, match=r"inputs.stationxml\[0\]: no files match"):
        _load_dict(tmp_path, data, strict=True)


def test_strict_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        _load_dict(tmp_path, {"inputs": {"quakeml": ["x.xml"]}}, strict=True)


def test_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSV_TEST_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("OSV_TEST_UNSET", raising=False)
    cfg = _load_dict(
        tmp_path,
        {
            "project": {"name": "${OSV_TEST_UNSET:-fallback}"},
            "inputs": {"quakeml": ["${OSV_TEST_DIR}/cat.xml"]},
        },
    )
    assert cfg.project.name == "fallback"
    assert cfg.inputs.quakeml == [tmp_path / "data" / "cat.xml"]


def test_env_missing_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSV_TEST_UNSET", raising=False)
    with pytest.raises(ConfigError) as ei:
        _load_dict(tmp_path, {"inputs": {"quakeml": ["${OSV_TEST_UNSET}/c.xml"]}})
    assert ei.value.errors[0][0] == "inputs.quakeml[0]"
    assert "OSV_TEST_UNSET" in ei.value.errors[0][1]


# ---------------------------------------------------------------- hash / dump


def test_hash_stable_and_sensitive(tmp_path: Path) -> None:
    a = _load_dict(tmp_path, {"velest": {"n_models": 10}})
    b = _load_dict(tmp_path, {"velest": {"n_models": 10}})
    c = _load_dict(tmp_path, {"velest": {"n_models": 11}})
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 64


def test_hash_independent_of_key_order(tmp_path: Path) -> None:
    a = _load_dict(tmp_path, {"project": {"name": "x"}, "fmtomo": {"iterations": 3}})
    b = _load_dict(tmp_path, {"fmtomo": {"iterations": 3}, "project": {"name": "x"}})
    assert config_hash(a) == config_hash(b)


def test_dump_round_trips(tmp_path: Path) -> None:
    data = {"data_selection": {"pick_uncertainty_to_weight": {0.05: 0, 0.1: 2}}}
    cfg = _load_dict(tmp_path, data)
    out = tmp_path / "dumped.yaml"
    out.write_text(dump_config(cfg))
    again = load_config(out)
    assert config_hash(again) == config_hash(cfg)


def test_depth_datum() -> None:
    cfg = resolve_config({}, ".")
    assert cfg.coordinates.depth_datum == "sea_level"
    assert cfg.coordinates.reference_elevation_m is None
    assert "coordinates.depth_datum" in _errors({"coordinates": {"depth_datum": "moon"}})
    assert "coordinates.reference_elevation_m" in _errors(
        {"coordinates": {"depth_datum": "reference"}}
    )
    assert _errors({"coordinates": {"reference_elevation_m": 600.0}})
    cfg = resolve_config(
        {"coordinates": {"depth_datum": "reference", "reference_elevation_m": 600}}, "."
    )
    assert cfg.coordinates.reference_elevation_m == 600.0
    assert resolve_config({"coordinates": {"depth_datum": "surface"}}, ".")
