"""inputs.station_aliases: 'auto' | mapping | CSV path (schema, loader, Stage 1 pass-through)."""

from __future__ import annotations

from pathlib import Path

import pytest

from osv.config import ConfigError, load_config, resolve_config
from osv.data import stage as prep_stage


def test_schema_accepts_auto_dict_and_path(tmp_path: Path) -> None:
    assert resolve_config({}, tmp_path).inputs.station_aliases == "auto"
    cfg = resolve_config({"inputs": {"station_aliases": {"ABCDE": "ABCD"}}}, tmp_path)
    assert cfg.inputs.station_aliases == {"ABCDE": "ABCD"}
    cfg = resolve_config({"inputs": {"station_aliases": "aliases.csv"}}, tmp_path)
    assert cfg.inputs.station_aliases == tmp_path / "aliases.csv"


def test_schema_rejects_bad_type(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        resolve_config({"inputs": {"station_aliases": 5}}, tmp_path)


def test_loader_resolves_relative_to_config_dir(tmp_path: Path) -> None:
    (tmp_path / "cfg").mkdir()
    path = tmp_path / "cfg" / "config.yaml"
    path.write_text("inputs:\n  station_aliases: ../maps/a.csv\n")
    cfg = load_config(path)
    assert cfg.inputs.station_aliases == tmp_path / "maps" / "a.csv"
    path.write_text("inputs:\n  station_aliases: auto\n")
    assert load_config(path).inputs.station_aliases == "auto"


def test_stage_passes_aliases_through(tmp_path: Path, monkeypatch, restore_stage_registry) -> None:
    csv = tmp_path / "a.csv"
    csv.write_text("code,alias\nABCDE,ABCD\n")
    cfg = resolve_config(
        {"inputs": {"quakeml": ["q.xml"], "stationxml": ["s.xml"], "station_aliases": str(csv)}},
        tmp_path,
    )
    seen: dict = {}

    class Stop(Exception):
        pass

    def fake_read_catalog(*a, **k):
        return object()

    def fake_read_stations(paths, icfg, ccfg, aliases=None):
        seen["aliases"] = aliases
        raise Stop

    monkeypatch.setattr(prep_stage, "read_catalog", fake_read_catalog)
    monkeypatch.setattr(prep_stage, "read_stations", fake_read_stations)
    st = prep_stage.PrepStage()
    assert csv in st.inputs(_Ctx(cfg))
    with pytest.raises(Stop):
        st.run(_Ctx(cfg))
    assert seen["aliases"] == csv


class _Ctx:
    def __init__(self, config) -> None:
        self.config = config
        import logging

        self.logger = logging.getLogger("t")
