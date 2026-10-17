"""Load, validate and resolve ``config.yaml``."""

from __future__ import annotations

import difflib
import glob as _glob
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from osv.config.schema import OSVConfig

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


_PREFIX_RE = re.compile(r"^([a-z_][a-z_0-9]*(?:\.[a-z_][a-z_0-9]*|\[\d+\])*): (.*)$", re.S)


def _split_key(key: str) -> list[Any]:
    return [int(p) if p.isdigit() else p for p in re.split(r"[.\[\]]+", key) if p]


class ConfigError(Exception):
    """Invalid configuration. ``errors`` lists ``(key_path, message)`` pairs."""

    def __init__(self, errors: list[tuple[str, str]], source: str | Path | None = None) -> None:
        self.errors = errors
        self.source = str(source) if source else None
        head = f"Invalid configuration{f' in {self.source}' if self.source else ''}:"
        body = "\n".join(f"  {k}: {m}" if k else f"  {m}" for k, m in errors)
        super().__init__(f"{head}\n{body}")


# --------------------------------------------------------------------------- helpers


def _fmt_loc(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out


def _expand_env(obj: Any, loc: tuple[Any, ...], errors: list[tuple[str, str]]) -> Any:
    if isinstance(obj, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            errors.append((_fmt_loc(loc), f"environment variable ${{{name}}} is not set"))
            return m.group(0)

        return _ENV_RE.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v, (*loc, k), errors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v, (*loc, i), errors) for i, v in enumerate(obj)]
    return obj


def _model_at(loc: tuple[Any, ...]) -> type[BaseModel] | None:
    """Best-effort: the model class that owns the last key in ``loc``'s parent path."""
    model: type[BaseModel] = OSVConfig
    for part in loc[:-1]:
        if isinstance(part, int):
            continue
        field = model.model_fields.get(part)
        if field is None:
            return None
        nxt = None
        stack = [field.annotation]
        while stack:
            t = stack.pop()
            if isinstance(t, type) and issubclass(t, BaseModel):
                nxt = t
                break
            stack.extend(getattr(t, "__args__", ()))
        if nxt is None:
            return None
        model = nxt
    return model


def _convert_errors(exc: ValidationError) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for err in exc.errors():
        loc = tuple(err["loc"])
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = "unknown key"
            model = _model_at(loc)
            if model is not None:
                close = difflib.get_close_matches(str(loc[-1]), list(model.model_fields), n=1)
                if close:
                    msg += f" (did you mean '{close[0]}'?)"
                else:
                    msg += f" (allowed: {', '.join(model.model_fields)})"
        elif err["type"] == "missing":
            msg = "required key is missing"
        else:
            msg = re.sub(r"^Value error, ", "", msg)
            # Model-level validators prefix their message with the offending sub-key.
            m = _PREFIX_RE.match(msg)
            if m:
                loc = (*loc, *_split_key(m.group(1)))
                msg = m.group(2)
        # Drop union-branch noise from locs such as ``fmtomo.phases[0].literal['P']``.
        parts = [p for p in loc if not (isinstance(p, str) and p.endswith("]") and "[" in p)]
        out.append((_fmt_loc(tuple(parts)), msg))
    return out


def _resolve_path(p: Path, base: Path) -> Path:
    p = Path(os.path.expanduser(str(p)))
    if not p.is_absolute():
        p = base / p
    return Path(os.path.normpath(p))


def _resolve_model(
    model: BaseModel,
    base: Path,
    strict: bool,
    loc: tuple[Any, ...],
    errors: list[tuple[str, str]],
) -> None:
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        here = (*loc, name)
        if extra.get("is_path"):
            if isinstance(value, Path):
                setattr(model, name, _resolve_path(value, base))
            elif isinstance(value, list):
                resolved: list[Path] = []
                for i, item in enumerate(value):
                    rp = _resolve_path(item, base)
                    if extra.get("glob") and _glob.has_magic(str(rp)):
                        hits = sorted(_glob.glob(str(rp), recursive=True))
                        if not hits:
                            if strict:
                                errors.append((_fmt_loc((*here, i)), f"no files match '{rp}'"))
                            else:
                                logger.warning("%s: no files match '%s'", _fmt_loc(here), rp)
                            resolved.append(rp)
                        else:
                            resolved.extend(Path(h) for h in hits)
                    else:
                        if strict and extra.get("glob") and not rp.exists():
                            errors.append((_fmt_loc((*here, i)), f"file not found: {rp}"))
                        resolved.append(rp)
                setattr(model, name, list(dict.fromkeys(resolved)))
        elif isinstance(value, BaseModel):
            _resolve_model(value, base, strict, here, errors)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, BaseModel):
                    _resolve_model(item, base, strict, (*here, i), errors)


# --------------------------------------------------------------------------- public API


def resolve_config(
    data: dict[str, Any] | None,
    base_dir: str | Path = ".",
    *,
    strict: bool = False,
    source: str | Path | None = None,
) -> OSVConfig:
    """Validate a raw config mapping and resolve paths against ``base_dir``.

    Parameters
    ----------
    data
        Parsed YAML mapping (``None`` is treated as empty).
    base_dir
        Directory that relative paths are resolved against.
    strict
        If true, glob patterns matching nothing and missing input files are errors.
    source
        Name of the origin, used in error messages.

    Raises
    ------
    ConfigError
        On any validation problem, listing every bad key path.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError([("", "top level of the config must be a mapping")], source)
    errors: list[tuple[str, str]] = []
    data = _expand_env(data, (), errors)
    if errors:
        raise ConfigError(errors, source)
    try:
        cfg = OSVConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_convert_errors(exc), source) from None
    _resolve_model(cfg, Path(base_dir).resolve(), strict, (), errors)
    if errors:
        raise ConfigError(errors, source)
    return cfg


def load_config(path: str | Path, *, strict: bool = False) -> OSVConfig:
    """Load ``config.yaml`` into an :class:`OSVConfig`.

    Performs ``${VAR}`` / ``${VAR:-default}`` expansion, validation, resolution of relative
    paths against the config file's directory, and glob expansion of ``inputs.stationxml``
    and ``inputs.quakeml``.

    Parameters
    ----------
    path
        Path to the YAML file.
    strict
        Treat unmatched globs and missing input files as errors (default: warn).

    Raises
    ------
    ConfigError
        If the file is missing, is not valid YAML, or fails validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError([("", f"config file not found: {path}")], path)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError([("", f"invalid YAML: {exc}")], path) from None
    return resolve_config(data, path.resolve().parent, strict=strict, source=path)


def config_data(cfg: OSVConfig) -> dict[str, Any]:
    """The config as plain JSON-compatible data (paths as strings)."""
    return cfg.model_dump(mode="json")


def dump_config(cfg: OSVConfig) -> str:
    """Serialise a (resolved) config to YAML text."""
    return yaml.safe_dump(config_data(cfg), sort_keys=False, default_flow_style=None, width=100)


def config_hash(cfg: OSVConfig) -> str:
    """Stable sha256 hex digest of the resolved config (for manifests)."""
    blob = json.dumps(config_data(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
