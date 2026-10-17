"""Generate the commented ``config.yaml`` template from the schema."""

from __future__ import annotations

import re
import textwrap
import types
import typing
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from osv.config.schema import OSVConfig

_WIDTH = 96


def _unwrap(tp: Any) -> list[Any]:
    """Flatten Optional/Union/Annotated into its leaf types."""
    args = typing.get_args(tp)
    if typing.get_origin(tp) is typing.Annotated:
        return _unwrap(args[0])
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return [leaf for a in args for leaf in _unwrap(a)]
    return [tp]


def _literals(tp: Any) -> list[Any]:
    out: list[Any] = []
    for leaf in _unwrap(tp):
        if typing.get_origin(leaf) is typing.Literal:
            out.extend(typing.get_args(leaf))
        elif typing.get_origin(leaf) is list:
            for inner in _unwrap(typing.get_args(leaf)[0]):
                if typing.get_origin(inner) is typing.Literal:
                    out.extend(typing.get_args(inner))
    return out


def _submodel(tp: Any) -> type[BaseModel] | None:
    for leaf in _unwrap(tp):
        if isinstance(leaf, type) and issubclass(leaf, BaseModel):
            return leaf
    return None


def _scalar(value: Any) -> str:
    text = yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()
    return text[:-4].rstrip() if text.endswith("\n...") else text


def _comment(text: str, indent: str) -> list[str]:
    return [f"{indent}# {line}" for line in textwrap.wrap(text, _WIDTH - len(indent) - 2) or [""]]


def _emit(model: type[BaseModel], values: dict[str, Any], indent: str) -> list[str]:
    lines: list[str] = []
    for name, field in model.model_fields.items():
        value = values[name]
        sub = _submodel(field.annotation)
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        if sub is not None and isinstance(value, dict):
            lines.append("")
            lines += _comment(field.description or "", indent)
            lines.append(f"{indent}{name}:")
            lines += _emit(sub, value, indent + "  ")
            continue
        lines += _comment(field.description or "", indent)
        allowed = _literals(field.annotation)
        if allowed:
            lines += _comment("Allowed: " + " | ".join(str(a) for a in allowed), indent)
        lines.append(f"{indent}{name}: {_scalar(value)}")
        example = extra.get("example")
        if example:
            lines += _comment("Example:", indent)
            lines += [f"{indent}#   {ln}" for ln in str(example).splitlines()]
    return lines


def generate_template(
    project_name: str = "my-project",
    stationxml: list[str] | None = None,
    quakeml: list[str] | None = None,
) -> str:
    """Return a fully commented ``config.yaml`` for a new project.

    Every key of the schema appears with its description, allowed values (for
    enumerations) and default. The placeholder input paths can be overridden.
    """
    base = OSVConfig.model_validate(
        {
            "project": {"name": project_name},
            "inputs": {
                "stationxml": stationxml or ["./stations/*.xml"],
                "quakeml": quakeml or ["./catalogue.xml"],
            },
        }
    )
    values = base.model_dump(mode="json")
    head = [
        "# openSeisVelocity configuration.",
        "# All settings are validated when loaded; unknown keys are errors.",
        "# Relative paths are resolved against the directory containing this file.",
        "# ${VAR} and ${VAR:-default} are expanded from the environment.",
    ]
    return "\n".join([*head, *_emit(OSVConfig, values, "")]) + "\n"


def write_template(
    directory: str | Path,
    *,
    force: bool = False,
    project_name: str | None = None,
) -> Path:
    """Create ``directory`` (if needed) containing a template ``config.yaml``.

    Raises
    ------
    FileExistsError
        If ``config.yaml`` already exists and ``force`` is false.
    """
    directory = Path(directory)
    target = directory / "config.yaml"
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; use --force to overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    name = project_name or re.sub(r"[^A-Za-z0-9._-]+", "-", directory.resolve().name).strip("-._")
    name = name or "my-project"
    target.write_text(generate_template(name))
    return target
