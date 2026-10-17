#!/usr/bin/env python
"""Generate ``docs/config_reference.md`` from the pydantic config schema.

Usage::

    python scripts/gen_config_reference.py            # rewrite docs/config_reference.md
    python scripts/gen_config_reference.py --check    # exit 1 if the file is out of date

``tests/test_docs.py`` runs the check so schema changes keep the docs in sync.
"""

from __future__ import annotations

import argparse
import sys
import types
import typing
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from osv.config.schema import OSVConfig

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "config_reference.md"

_CONSTRAINTS = {
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
    "min_length": "min length",
    "max_length": "max length",
    "multiple_of": "multiple of",
}


def _leaves(tp: Any) -> list[Any]:
    args = typing.get_args(tp)
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        return _leaves(args[0])
    if origin in (typing.Union, types.UnionType):
        return [leaf for a in args for leaf in _leaves(a)]
    return [tp]


def _type_name(tp: Any) -> str:
    """Readable type, e.g. ``list[[lower, upper]] | null``."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Annotated:
        inner = _type_name(args[0])
        funcs = {getattr(getattr(m, "func", None), "__name__", "") for m in args[1:]}
        if "_check_latlon" in funcs:
            return "[lat, lon]"
        if "_check_lower_upper" in funcs:
            return "[lower, upper]"
        if "_check_positive_list" in funcs:
            return "list[number > 0]"
        return inner
    if origin in (typing.Union, types.UnionType):
        parts = [("null" if a is type(None) else _type_name(a)) for a in args]
        return " | ".join(parts)
    if origin is typing.Literal:
        return "string" if all(isinstance(a, str) for a in args) else "literal"
    if origin is list:
        return f"list[{_type_name(args[0])}]"
    if origin is dict:
        return f"dict[{_type_name(args[0])}, {_type_name(args[1])}]"
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f"list[{_type_name(args[0])}]"
        return "[" + ", ".join(_type_name(a) for a in args) + "]"
    if isinstance(tp, type):
        if issubclass(tp, BaseModel):
            return "section"
        return {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(
            tp.__name__, tp.__name__.lower()
        )
    return str(tp)


def _allowed(tp: Any) -> list[str]:
    out: list[str] = []
    for leaf in _leaves(tp):
        origin = typing.get_origin(leaf)
        if origin is typing.Literal:
            out.extend(str(a) for a in typing.get_args(leaf))
        elif origin is list:
            for inner in _leaves(typing.get_args(leaf)[0]):
                if typing.get_origin(inner) is typing.Literal:
                    out.extend(str(a) for a in typing.get_args(inner))
    return out


def _submodel(tp: Any) -> type[BaseModel] | None:
    for leaf in _leaves(tp):
        if isinstance(leaf, type) and issubclass(leaf, BaseModel):
            return leaf
        if typing.get_origin(leaf) is list:
            for inner in _leaves(typing.get_args(leaf)[0]):
                if isinstance(inner, type) and issubclass(inner, BaseModel):
                    return inner
    return None


def _is_list_of_sections(tp: Any) -> bool:
    return any(
        typing.get_origin(leaf) is list
        and any(
            isinstance(i, type) and issubclass(i, BaseModel)
            for i in _leaves(typing.get_args(leaf)[0])
        )
        for leaf in _leaves(tp)
    )


def _default(field: Any) -> str:
    if field.default is not PydanticUndefined:
        value = field.default
    elif field.default_factory is not None:
        value = field.default_factory()
    else:
        return "*required*"
    if isinstance(value, BaseModel):
        return "(see subsections)"
    if isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
        if value and all(isinstance(v, dict) for v in value):
            return f"{len(value)} entries (see below)"
    if isinstance(value, Path):
        value = str(value)
    text = yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()
    if text.endswith("\n..."):
        text = text[:-4].rstrip()
    return f"`{text}`"


def _constraints(field: Any) -> str:
    parts = []
    for m in field.metadata:
        for attr, label in _CONSTRAINTS.items():
            v = getattr(m, attr, None)
            if v is not None:
                parts.append(f"{label} {v}")
    return ", ".join(parts)


def _esc(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _section(model: type[BaseModel], path: str, desc: str, out: list[str], depth: int) -> None:
    """Append the heading, description, key table and subsections of ``model``."""
    out.append(f"{'#' * min(depth, 6)} `{path}`")
    out.append("")
    text = desc or (model.__doc__ or "").strip()
    if text:
        out.append(_esc(text))
        out.append("")
    scalars, subs = [], []
    for name, field in model.model_fields.items():
        sub = _submodel(field.annotation)
        (subs if sub is not None else scalars).append((name, field, sub))
    if scalars:
        out.append("| Key | Type | Default | Allowed / constraints | Description |")
        out.append("|---|---|---|---|---|")
        for name, field, _ in scalars:
            extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
            notes = []
            allowed = _allowed(field.annotation)
            if allowed:
                notes.append(" \\| ".join(f"`{a}`" for a in allowed))
            c = _constraints(field)
            if c:
                notes.append(c)
            if extra.get("is_path"):
                notes.append("path, relative to the config file")
            if extra.get("glob"):
                notes.append("globs expanded")
            desc_text = _esc(field.description or "")
            if extra.get("example"):
                desc_text += " Example: " + "<br>".join(
                    f"`{_esc(ln)}`" for ln in str(extra["example"]).splitlines()
                )
            out.append(
                f"| `{name}` | {_esc(_type_name(field.annotation))} | {_esc(_default(field))} "
                f"| {'; '.join(notes) or '-'} | {desc_text} |"
            )
        out.append("")
    for name, field, sub in subs:
        assert sub is not None
        label = f"{path}.{name}" + ("[]" if _is_list_of_sections(field.annotation) else "")
        _section(sub, label, field.description or "", out, depth + 1)


def generate() -> str:
    """Return the full Markdown text of the configuration reference."""
    out: list[str] = [
        "<!-- Generated by scripts/gen_config_reference.py from osv.config.schema. "
        "Do not edit by hand. -->",
        "",
        "# Configuration reference",
        "",
        "Every setting of `config.yaml`, generated from the pydantic schema "
        "(`src/osv/config/schema.py`). `osv init DIR` writes a template with the same "
        "descriptions as comments; `osv config show` prints the settings in use. Unknown keys "
        "are errors. Relative paths are resolved against the directory containing the config "
        "file, and `${VAR}` / `${VAR:-default}` are expanded from the environment.",
        "",
        "Types: *list[[lower, upper]]* is a list of two-number pairs; *[lat/lon]* is a "
        "`[latitude, longitude]` pair in degrees; *null* means the key may be unset. "
        "Nested sections are written as YAML mappings; entries marked `[]` are lists of "
        "mappings.",
        "",
    ]
    top = OSVConfig
    # Overview table of top-level sections, then one chapter per section.
    out.append("| Section | Purpose |")
    out.append("|---|---|")
    for name, field in top.model_fields.items():
        out.append(f"| [`{name}`](#{name}) | {_esc(field.description or '')} |")
    out.append("")
    for name, field in top.model_fields.items():
        sub = _submodel(field.annotation)
        assert sub is not None
        _section(sub, name, field.description or "", out, 2)
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="fail if docs/config_reference.md is stale"
    )
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args(argv)
    text = generate()
    if args.check:
        if not args.output.exists() or args.output.read_text() != text:
            print(
                f"{args.output} is out of date; run: python scripts/gen_config_reference.py",
                file=sys.stderr,
            )
            return 1
        return 0
    args.output.write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
