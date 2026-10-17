"""Pydantic schema and loader for config.yaml."""

from __future__ import annotations

from osv.config.loader import (
    ConfigError,
    config_hash,
    dump_config,
    load_config,
    resolve_config,
)
from osv.config.schema import OSVConfig
from osv.config.template import generate_template, write_template

__all__ = [
    "ConfigError",
    "OSVConfig",
    "config_hash",
    "dump_config",
    "generate_template",
    "load_config",
    "resolve_config",
    "write_template",
]
