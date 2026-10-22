"""Documentation checks: the generated config reference must match the schema."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from osv.config.schema import OSVConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gen_config_reference.py"
REFERENCE = ROOT / "docs" / "config_reference.md"
EXAMPLE = ROOT / "config" / "config.example.yaml"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_config_reference", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_models(model: type[BaseModel], seen: set[type[BaseModel]]) -> None:
    if model in seen:
        return
    seen.add(model)
    gen = _load_generator()
    for field in model.model_fields.values():
        sub = gen._submodel(field.annotation)
        if sub is not None:
            _all_models(sub, seen)


def test_config_reference_is_up_to_date():
    gen = _load_generator()
    assert REFERENCE.exists(), "run: python scripts/gen_config_reference.py"
    assert REFERENCE.read_text() == gen.generate(), (
        "docs/config_reference.md is stale; run: python scripts/gen_config_reference.py"
    )


def test_config_example_is_up_to_date_and_loads(tmp_path):
    from osv.config import generate_template, load_config

    assert EXAMPLE.read_text() == generate_template(), (
        "config/config.example.yaml is stale; run: python scripts/gen_config_reference.py"
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(EXAMPLE.read_text())
    load_config(cfg)


@pytest.mark.parametrize("path", sorted((ROOT / "config").glob("*.yaml")), ids=lambda p: p.name)
def test_repo_configs_validate(path):
    """Every config kept in config/ still matches the schema (input files need not exist)."""
    from osv.config import load_config

    load_config(path)


def test_config_reference_covers_every_key():
    text = REFERENCE.read_text()
    models: set[type[BaseModel]] = set()
    _all_models(OSVConfig, models)
    missing = [
        f"{m.__name__}.{name}"
        for m in models
        for name in m.model_fields
        if f"`{name}`" not in text and f".{name}`" not in text and f".{name}[]`" not in text
    ]
    assert not missing, f"keys missing from the config reference: {missing}"


def test_check_mode_flags_stale_file(tmp_path):
    gen = _load_generator()
    stale = tmp_path / "ref.md"
    stale.write_text("old")
    assert gen.main(["--check", "--output", str(stale)]) == 1
    assert gen.main(["--output", str(stale)]) == 0
    assert gen.main(["--check", "--output", str(stale)]) == 0


def test_docs_internal_links_resolve():
    """Relative Markdown links between docs point at existing files."""
    bad = []
    for md in (ROOT / "docs").glob("*.md"):
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", md.read_text()):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (md.parent / target).resolve().exists():
                bad.append(f"{md.name}: {target}")
    assert not bad, bad


@pytest.mark.skipif(
    importlib.util.find_spec("mkdocs") is None, reason="mkdocs not installed (docs extra)"
)
def test_mkdocs_config_parses():
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("", lambda loader, suffix, node: None)
    cfg = yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=_Loader)
    assert cfg["site_name"]
    assert cfg["nav"]
