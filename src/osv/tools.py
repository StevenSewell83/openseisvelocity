"""Locate and run external executables (VELEST, FMTOMO, FMTOMO Extras).

Executables are looked up by *logical name* (``"velest"``, ``"fm3d"``, ...).
Resolution order for a name:

1. explicit override (``overrides`` dict name -> path, e.g. from the config),
2. ``$OSV_TOOLS_DIR`` (a directory),
3. ``$CONDA_PREFIX/bin/osv-tools/``, then ``<sys.prefix>/bin/osv-tools/`` (venv installs),
4. repo-local ``build/bin/``,
5. ``PATH`` (last, with a warning: an unrelated old ``velest`` may be found).

Extras may be installed under prefixed file names to avoid clashes; use the
``name_map`` argument (logical name -> file name) or the environment variable
``OSV_EXTRAS_PREFIX`` (prepended to every Extras name).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_TOOLS_DIR = "OSV_TOOLS_DIR"
ENV_EXTRAS_PREFIX = "OSV_EXTRAS_PREFIX"

#: Repo-local build directory searched before PATH (monkeypatch-able in tests).
REPO_BUILD_BIN: Path = Path(__file__).resolve().parents[2] / "build" / "bin"

GROUP_VELEST = "velest"
GROUP_FMTOMO = "fmtomo"
GROUP_EXTRAS = "extras"

#: logical name -> group
TOOL_GROUPS: dict[str, str] = {
    "velest": GROUP_VELEST,
    **dict.fromkeys(
        [
            "fm3d",
            "tomo3d",
            "invert3d",
            "frechgen",
            "residuals",
            "grid3dg",
            "arraygen",
            "synthdata",
            "obsdata",
            "moddata",
            "scorrect",
            "gmtslice",
        ],
        GROUP_FMTOMO,
    ),
    **dict.fromkeys(
        [
            "tomo3di",
            "tomo3dj",
            "tomo3dps",
            "gridloc",
            "gridlocj",
            "obsdatai",
            "synthdatai",
            "vpvsinv",
            "mkvpvs",
            "jointimes",
            "vpvsresids",
            "gmtslicei",
            "frvpvs",
            "grid3dgi",
        ],
        GROUP_EXTRAS,
    ),
}

#: Patterns (regex, searched per line) signalling a failure even with exit code 0.
DEFAULT_ERROR_PATTERNS: tuple[str, ...] = (
    r"Fortran runtime error",
    r"forrtl: severe",
    r"Segmentation fault",
    r"Program received signal SIG",
    r"^\s*STOP\s+(?!0\s*$)\S",  # "STOP <message-or-nonzero-code>"; bare/zero STOP is normal
    r"^\s*ERROR STOP",
)

#: Per-tool overrides of the patterns above (name -> patterns). Empty tuple disables.
TOOL_ERROR_PATTERNS: dict[str, tuple[str, ...]] = {}


class ToolError(Exception):
    """Base class for tool-runner errors."""


class ToolNotFoundError(ToolError, FileNotFoundError):
    """Raised when an executable cannot be located."""


class ToolRunError(ToolError):
    """Raised when a tool fails (non-zero exit, timeout or detected runtime error)."""

    def __init__(self, message: str, result: ToolResult) -> None:
        super().__init__(message)
        self.result = result


class ToolTimeoutError(ToolRunError):
    """Raised when a tool exceeds its timeout (process group was killed)."""


@dataclass
class ToolResult:
    """Outcome of one executable run."""

    name: str
    command: list[str]
    cwd: Path
    returncode: int
    wall_time: float
    stdout_log: Path
    stderr_log: Path
    timed_out: bool = False
    detected_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if exit code was 0, no timeout and no error pattern was detected."""
        return self.returncode == 0 and not self.timed_out and not self.detected_errors

    def stdout_text(self) -> str:
        """Return the captured stdout."""
        return _read(self.stdout_log)

    def stderr_text(self) -> str:
        """Return the captured stderr."""
        return _read(self.stderr_log)


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _tail(text: str, n: int = 30) -> str:
    return "\n".join(text.splitlines()[-n:])


def _filename(name: str, name_map: Mapping[str, str] | None) -> str:
    if name_map and name in name_map:
        return name_map[name]
    if TOOL_GROUPS.get(name) == GROUP_EXTRAS:
        return os.environ.get(ENV_EXTRAS_PREFIX, "") + name
    return name


def _is_exec(p: Path) -> bool:
    return p.is_file() and os.access(p, os.X_OK)


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    if os.environ.get(ENV_TOOLS_DIR):
        dirs.append(Path(os.environ[ENV_TOOLS_DIR]).expanduser())
    if os.environ.get("CONDA_PREFIX"):
        dirs.append(Path(os.environ["CONDA_PREFIX"]) / "bin" / "osv-tools")
    # Environment the interpreter runs from (venv installs by install.sh --venv work
    # without activation).
    dirs.append(Path(sys.prefix) / "bin" / "osv-tools")
    dirs.append(REPO_BUILD_BIN)
    return dirs


def _not_found(name: str, tried: Sequence[str]) -> ToolNotFoundError:
    return ToolNotFoundError(
        f"Executable '{name}' not found (tried: {', '.join(tried) or 'nothing'}). "
        f"Fix: run ./install.sh (installs into $CONDA_PREFIX/bin/osv-tools/), or set "
        f"{ENV_TOOLS_DIR} to the directory containing the built executables, or give an "
        f"explicit path in the config. Run `osv doctor` to check the toolchain."
    )


def which_tool(
    name: str,
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
    name_map: Mapping[str, str] | None = None,
) -> Path:
    """Locate an executable by logical name.

    Parameters
    ----------
    name
        Logical tool name (e.g. ``"velest"``, ``"fm3d"``).
    overrides
        Optional mapping name -> explicit path (e.g. from the config).
    name_map
        Optional mapping logical name -> file name (for prefixed installs).

    Raises
    ------
    ToolNotFoundError
        If the tool cannot be found; the message says how to fix it.
    """
    tried: list[str] = []
    if overrides and overrides.get(name):
        p = Path(overrides[name]).expanduser()
        if _is_exec(p):
            return p
        raise _not_found(name, [f"override {p} (missing or not executable)"])

    fname = _filename(name, name_map)
    for d in _search_dirs():
        p = d / fname
        tried.append(str(p))
        if _is_exec(p):
            return p
    found = shutil.which(fname)
    tried.append(f"PATH ({fname})")
    if found:
        logger.warning(
            "Using '%s' found on PATH (%s), not from an osv tools directory; this may be an "
            "unrelated build. Run ./install.sh or set %s to avoid ambiguity.",
            fname,
            found,
            ENV_TOOLS_DIR,
        )
        return Path(found)
    raise _not_found(name, tried)


def available_tools(
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
    name_map: Mapping[str, str] | None = None,
) -> dict[str, Path | None]:
    """Return ``{logical name: resolved path or None}`` for every known tool."""
    out: dict[str, Path | None] = {}
    for name in TOOL_GROUPS:
        try:
            out[name] = which_tool(name, overrides, name_map)
        except ToolNotFoundError:
            out[name] = None
    return out


def tool_group(name: str) -> str:
    """Return ``"velest"``, ``"fmtomo"`` or ``"extras"`` for a known tool name."""
    try:
        return TOOL_GROUPS[name]
    except KeyError:
        raise KeyError(f"Unknown tool '{name}'. Known: {', '.join(TOOL_GROUPS)}") from None


def _detect_errors(patterns: Iterable[str], *texts: str) -> list[str]:
    regs = [re.compile(p) for p in patterns]
    hits: list[str] = []
    for text in texts:
        for line in text.splitlines():
            if any(r.search(line) for r in regs):
                hits.append(line.strip())
    return hits


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_tool(
    name: str,
    cwd: str | os.PathLike[str],
    args: Sequence[str | os.PathLike[str]] = (),
    stdin: str | bytes | None = None,
    timeout: float | None = None,
    log_prefix: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    check: bool = True,
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
    name_map: Mapping[str, str] | None = None,
    error_patterns: Sequence[str] | None = None,
) -> ToolResult:
    """Run a tool in ``cwd``, logging stdout/stderr to files in ``cwd``.

    Many programs read fixed-name control files from the working directory and
    some read from stdin; pass ``stdin`` as text/bytes for the latter.
    Logs go to ``<cwd>/<log_prefix or name>.stdout.log`` / ``.stderr.log``.

    Parameters
    ----------
    check
        If True (default), raise `ToolRunError` on non-zero exit, timeout or a
        detected Fortran runtime error in the output.
    error_patterns
        Regexes (per line) marking failure even with exit code 0. Defaults to
        ``TOOL_ERROR_PATTERNS[name]`` or `DEFAULT_ERROR_PATTERNS`.
    env
        Extra environment variables, merged over ``os.environ``.

    Raises
    ------
    ToolNotFoundError, ToolRunError, ToolTimeoutError
    """
    exe = which_tool(name, overrides, name_map)
    cwd_p = Path(cwd)
    cwd_p.mkdir(parents=True, exist_ok=True)
    prefix = log_prefix or name
    out_log = cwd_p / f"{prefix}.stdout.log"
    err_log = cwd_p / f"{prefix}.stderr.log"
    cmd = [str(exe), *(str(a) for a in args)]
    full_env = {**os.environ, **(env or {})}
    data = stdin.encode() if isinstance(stdin, str) else stdin
    patterns = (
        error_patterns
        if error_patterns is not None
        else TOOL_ERROR_PATTERNS.get(name, DEFAULT_ERROR_PATTERNS)
    )

    logger.info("Running %s in %s", " ".join(cmd), cwd_p)
    timed_out = False
    t0 = time.monotonic()
    with open(out_log, "wb") as fo, open(err_log, "wb") as fe:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd_p,
            env=full_env,
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=fo,
            stderr=fe,
            start_new_session=True,
        )
        try:
            proc.communicate(input=data, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            proc.communicate()
        except BrokenPipeError:  # tool exited without reading stdin
            proc.wait()
    wall = time.monotonic() - t0

    result = ToolResult(
        name=name,
        command=cmd,
        cwd=cwd_p,
        returncode=proc.returncode,
        wall_time=wall,
        stdout_log=out_log,
        stderr_log=err_log,
        timed_out=timed_out,
    )
    result.detected_errors = _detect_errors(patterns, result.stdout_text(), result.stderr_text())

    if check and not result.ok:
        if timed_out:
            raise ToolTimeoutError(
                f"'{name}' timed out after {timeout}s in {cwd_p} (process group killed). "
                f"Increase the timeout or check the inputs. Logs: {out_log}, {err_log}",
                result,
            )
        if result.returncode != 0:
            why = f"'{name}' exited with return code {result.returncode}"
        else:
            why = f"'{name}' reported a runtime error despite exit code 0"
        detail = ""
        if result.detected_errors:
            detail += "Detected: " + "; ".join(result.detected_errors[:5]) + "\n"
        detail += (
            f"--- last lines of stderr ---\n{_tail(result.stderr_text())}\n"
            f"--- last lines of stdout ---\n{_tail(result.stdout_text())}"
        )
        raise ToolRunError(
            f"{why} (cwd={cwd_p}, command={' '.join(cmd)}).\n{detail}\nLogs: {out_log}, {err_log}",
            result,
        )
    return result


def tool_versions(
    names: Iterable[str] | None = None,
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
    name_map: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str | float]]:
    """Return ``{name: {path, sha256, mtime}}`` for found tools (for manifests).

    Tools that cannot be found are omitted. ``names`` defaults to all known tools.
    """
    out: dict[str, dict[str, str | float]] = {}
    for name in names if names is not None else TOOL_GROUPS:
        try:
            p = which_tool(name, overrides, name_map)
        except ToolNotFoundError:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = {"path": str(p), "sha256": h.hexdigest(), "mtime": p.stat().st_mtime}
    return out
