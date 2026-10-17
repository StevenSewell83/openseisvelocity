"""``osv doctor``: check the installation, the external tools and run smoke tests.

Every check yields a `Check` with status ``ok``/``warn``/``fail`` and, unless ok, a
fix hint. `run_doctor` collects them; `render_text` and `to_json` format the report.

The VELEST array limits and compiler information come from
``<prefix>/share/osv-tools/build_info.json`` (written by the CMake build); when that file is
missing the limits are parsed from the ``velest_common.inc`` of the repo build directory.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from osv import tools

logger = logging.getLogger(__name__)

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_FREE_GB = 5.0
MIN_PYTHON = (3, 11)

#: (import name, distribution name, minimum version or None)
REQUIRED_PACKAGES: tuple[tuple[str, str, str | None], ...] = (
    ("obspy", "obspy", "1.4"),
    ("numpy", "numpy", "1.24"),
    ("pandas", "pandas", "2.0"),
    ("xarray", "xarray", "2023.1"),
    ("matplotlib", "matplotlib", "3.7"),
    ("pyproj", "pyproj", "3.4"),
    ("pydantic", "pydantic", "2.0"),
    ("scipy", "scipy", "1.10"),
    ("joblib", "joblib", "1.2"),
)
OPTIONAL_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("cartopy", "cartopy", "maps in figures"),
    ("pygmt", "pygmt", "GMT figures"),
    ("lasio", "lasio", "reading well logs (LAS)"),
    ("rasterio", "rasterio", "reading GeoTIFF surfaces"),
    ("pyvista", "pyvista", "3-D model views"),
)

# Smoke-test tolerances.
VELEST_RMS_REF = 0.202976  # last "RMS RESIDUAL" of the packaged 25-event dataset
VELEST_RMS_RTOL = 0.05
VELEST_N_EVENTS = 25
VELEST_MIN_STATIONS = 17
FM3D_ATOL = 0.02  # seconds, vs the packaged reference arrivals
FM3D_RTOL = 1e-3


@dataclass
class Check:
    """Result of one doctor check."""

    name: str
    status: str
    message: str
    fix: str = ""
    group: str = "general"
    seconds: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _ver_tuple(v: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in re.split(r"[.\-+]", v):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group()))
    return tuple(out)


# --------------------------------------------------------------------------- Python


def check_python() -> list[Check]:
    """Python version and package versions."""
    out: list[Check] = []
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        out.append(Check("python", OK, f"Python {ver} ({platform.platform()})", group="python"))
    else:
        out.append(
            Check(
                "python",
                FAIL,
                f"Python {ver}; {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required",
                "Create the environment with python>=3.11 (see ./install.sh).",
                "python",
            )
        )
    for imp, dist, minv in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(imp)
        except Exception as exc:  # noqa: BLE001 - any import failure is reportable
            out.append(
                Check(
                    f"pkg:{imp}",
                    FAIL,
                    f"cannot import {imp} ({type(exc).__name__}: {exc})",
                    f"Install it: conda install -c conda-forge {dist}  (or pip install {dist}).",
                    "python",
                )
            )
            continue
        ver = str(getattr(mod, "__version__", "unknown"))
        if minv and ver != "unknown" and _ver_tuple(ver) < _ver_tuple(minv):
            out.append(
                Check(
                    f"pkg:{imp}",
                    WARN,
                    f"{imp} {ver} is older than the tested minimum {minv}",
                    f"Upgrade: conda install -c conda-forge '{dist}>={minv}'.",
                    "python",
                )
            )
        else:
            out.append(Check(f"pkg:{imp}", OK, f"{imp} {ver}", group="python"))
    for imp, dist, why in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(imp)
            out.append(
                Check(
                    f"pkg:{imp}",
                    OK,
                    f"{imp} {getattr(mod, '__version__', '?')} (optional)",
                    group="python",
                )
            )
        except Exception:  # noqa: BLE001
            out.append(
                Check(
                    f"pkg:{imp}",
                    WARN,
                    f"optional package {imp} not available ({why} disabled)",
                    f"Install it if needed: conda install -c conda-forge {dist}.",
                    "python",
                )
            )
    return out


# --------------------------------------------------------------------------- tools


def _origin(path: Path) -> str:
    """Label where a resolved executable came from: ``tools-dir`` or ``PATH``."""
    try:
        parent = path.parent.resolve()
        for d in tools._search_dirs():
            if d.resolve() == parent:
                return "tools-dir"
    except OSError:
        pass
    return "PATH"


def _missing_libs(path: Path) -> list[str]:
    """Shared libraries ``ldd`` cannot resolve (empty if not an ELF file or no ldd)."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"\x7fELF":
                return []
    except OSError:
        return []
    ldd = shutil.which("ldd")
    if not ldd:
        return []
    try:
        res = subprocess.run([ldd, str(path)], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln.split()[0] for ln in res.stdout.splitlines() if "not found" in ln]


def check_proj() -> list[Check]:
    """EPSG lookups via pyproj and (if installed) rasterio; a stale PROJ_DATA breaks them."""
    env = {v: os.environ.get(v) for v in ("PROJ_DATA", "PROJ_LIB")}
    try:
        import pyproj

        datadir = pyproj.datadir.get_data_dir()
    except ImportError:
        datadir = "n/a"
    info = f"PROJ_DATA={env['PROJ_DATA']!r} PROJ_LIB={env['PROJ_LIB']!r} pyproj data dir={datadir}"
    fix = (
        "unset PROJ_DATA PROJ_LIB (a conda/system PROJ with an older proj.db is shadowing the "
        f"bundled one), or point PROJ_DATA at {datadir}. osv overrides a stale value "
        "automatically unless OSV_KEEP_PROJ_DATA=1."
    )
    out: list[Check] = []
    details = {**env, "pyproj_data_dir": datadir}
    try:
        import pyproj

        pyproj.CRS.from_epsg(2193)
        out.append(
            Check(
                "proj:pyproj", OK, f"pyproj EPSG:2193 ok; {info}", group="python", details=details
            )
        )
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        out.append(
            Check(
                "proj:pyproj",
                FAIL,
                f"pyproj EPSG:2193 failed ({str(exc)[:100]}); {info}",
                fix,
                "python",
                details=details,
            )
        )
    try:
        from osv._proj import ensure_proj_data

        ensure_proj_data()  # same guard the CLI applies, so the check reflects real runs
        import rasterio.crs
    except ImportError:
        return out
    try:
        rasterio.crs.CRS.from_epsg(2193)
        out.append(
            Check(
                "proj:rasterio",
                OK,
                f"rasterio EPSG:2193 ok; {info}",
                group="python",
                details=details,
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(
            Check(
                "proj:rasterio",
                FAIL,
                f"rasterio EPSG:2193 failed ({str(exc)[:100]}); {info}",
                fix,
                "python",
                details=details,
            )
        )
    return out


def check_tools() -> list[Check]:
    """Each known executable: found, runnable, and where it came from."""
    out: list[Check] = []
    for name, group in tools.TOOL_GROUPS.items():
        fix_missing = (
            "Run ./install.sh (or scripts/build_fortran.sh), or set "
            f"{tools.ENV_TOOLS_DIR} to the directory holding the built executables."
        )
        if group == tools.GROUP_EXTRAS:
            fix_missing += " FMTOMO Extras are optional; rebuild without --no-extras to get them."
        try:
            p = tools.which_tool(name)
        except tools.ToolNotFoundError:
            status = WARN if group == tools.GROUP_EXTRAS else FAIL
            out.append(Check(f"tool:{name}", status, "not found", fix_missing, "tools"))
            continue
        missing = _missing_libs(p)
        if missing:
            out.append(
                Check(
                    f"tool:{name}",
                    FAIL,
                    f"{p} needs missing shared libraries: {', '.join(missing)}",
                    "Activate the environment that was used to build it (conda activate osv), "
                    "or rebuild with ./install.sh (add --static-gfortran-libs if the "
                    "compiler runtime is not on the library path).",
                    "tools",
                    details={"path": str(p)},
                )
            )
            continue
        origin = _origin(p)
        if origin == "PATH":
            out.append(
                Check(
                    f"tool:{name}",
                    WARN,
                    f"{p} (found on PATH, not in an osv tools directory)",
                    "A stale or unrelated binary may be shadowing the osv build. Run "
                    f"./install.sh or set {tools.ENV_TOOLS_DIR}.",
                    "tools",
                    details={"path": str(p), "origin": origin},
                )
            )
        else:
            out.append(
                Check(
                    f"tool:{name}",
                    OK,
                    f"{p} ({origin})",
                    group="tools",
                    details={"path": str(p), "origin": origin},
                )
            )
    return out


# --------------------------------------------------------------------------- build info


_INC_NAMES = {
    "ieq": "ieq",
    "ist": "ist",
    "inltot": "inltot",
    "itotmodels": "itotmodels",
    "inshot": "inshot",
    "maxobsperevent": "maxobs",
}


def parse_velest_inc(text: str) -> dict[str, int]:
    """Parse array limits from the text of a ``velest_common.inc``."""
    limits: dict[str, int] = {}
    for line in text.splitlines():
        if line[:1].lower() == "c":
            continue
        for key, val in re.findall(r"\b([a-zA-Z]+)\s*=\s*(\d+)\b", line):
            if key.lower() in _INC_NAMES:
                limits.setdefault(_INC_NAMES[key.lower()], int(val))
    return limits


def find_build_info(velest: Path | None = None) -> dict[str, Any] | None:
    """Return the contents of ``build_info.json`` next to the installed tools, if any."""
    cands: list[Path] = []
    if velest is not None:
        b = velest.parent
        cands += [
            b.parent / "share" / "osv-tools",
            b.parent.parent / "share" / "osv-tools",
            b / "share" / "osv-tools",
        ]
    cands += [tools.REPO_BUILD_BIN.parent / "share" / "osv-tools"]
    for d in cands:
        f = d / "build_info.json"
        if f.is_file():
            try:
                return json.loads(f.read_text())  # type: ignore[no-any-return]
            except (OSError, ValueError):
                continue
    return None


def velest_limits(velest: Path | None = None) -> tuple[dict[str, int], str]:
    """VELEST array limits and where they came from (``""`` if unknown)."""
    info = find_build_info(velest)
    if info and info.get("velest_limits"):
        return {k: int(v) for k, v in info["velest_limits"].items()}, "build_info.json"
    inc = tools.REPO_BUILD_BIN.parent / "fortran" / "velest_src" / "velest_common.inc"
    if inc.is_file():
        lim = parse_velest_inc(inc.read_text(errors="replace"))
        if lim:
            return lim, str(inc)
    return {}, ""


def check_build() -> list[Check]:
    """Compiler/flags used for the build and the VELEST array limits."""
    try:
        velest = tools.which_tool("velest")
    except tools.ToolNotFoundError:
        velest = None
    out: list[Check] = []
    info = find_build_info(velest)
    if info:
        comp = f"{info.get('compiler_id', '?')} {info.get('compiler_version', '?')}"
        out.append(
            Check(
                "build:compiler",
                OK,
                f"{comp} ({info.get('compiler', '?')}); flags: {info.get('flags', '?')}; "
                f"VELEST flags: {info.get('velest_flags', '?')}; commit "
                f"{info.get('git_commit', '?')}, built {info.get('built_at', '?')}",
                group="build",
                details=info,
            )
        )
    else:
        out.append(
            Check(
                "build:compiler",
                WARN,
                "no build_info.json found; compiler and flags of the build are unknown",
                "Rebuild with scripts/build_fortran.sh (or ./install.sh) to record them.",
                "build",
            )
        )
    lim, src = velest_limits(velest)
    if lim:
        txt = ", ".join(f"{k}={v}" for k, v in lim.items())
        out.append(
            Check("build:velest_limits", OK, f"{txt} (from {src})", group="build", details=lim)
        )
    elif velest is not None:
        out.append(
            Check(
                "build:velest_limits",
                WARN,
                "VELEST array limits unknown",
                "Rebuild with scripts/build_fortran.sh to record them.",
                "build",
            )
        )
    return out


# --------------------------------------------------------------------------- system


def _existing_parent(p: Path) -> Path:
    p = p.expanduser().absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _free_gb(p: Path) -> float:
    return shutil.disk_usage(_existing_parent(p)).free / 1e9


def check_system(workdir: Path | None, n_jobs: int | None) -> list[Check]:
    """CPU count, free disk space and workdir write permission."""
    out: list[Check] = []
    ncpu = os.cpu_count() or 1
    if n_jobs is None:
        out.append(Check("cpu", OK, f"{ncpu} CPU cores", group="system"))
    else:
        eff = ncpu if n_jobs == -1 else n_jobs
        msg = f"{ncpu} CPU cores; config n_jobs={n_jobs} (uses {eff})"
        if eff > ncpu:
            out.append(
                Check(
                    "cpu",
                    WARN,
                    msg + ": more workers than cores",
                    f"Set project.n_jobs to {ncpu} or -1 in the config.",
                    "system",
                )
            )
        else:
            out.append(Check("cpu", OK, msg, group="system"))
    spots = [("tmp", Path(tempfile.gettempdir()))]
    if workdir is not None:
        spots.insert(0, ("workdir", workdir))
    for label, path in spots:
        try:
            free = _free_gb(path)
        except OSError as exc:
            out.append(Check(f"disk:{label}", FAIL, f"cannot stat {path}: {exc}", group="system"))
            continue
        msg = f"{free:.1f} GB free in {path}"
        if free < MIN_FREE_GB:
            fix = (
                "Free up space"
                if label == "workdir"
                else "Free up space in the temp dir or point TMPDIR at a larger disk"
            )
            out.append(
                Check(
                    f"disk:{label}",
                    WARN,
                    msg + f" (< {MIN_FREE_GB:g} GB)",
                    fix + ".",
                    "system",
                    details={"free_gb": round(free, 2)},
                )
            )
        else:
            out.append(
                Check(f"disk:{label}", OK, msg, group="system", details={"free_gb": round(free, 2)})
            )
    if workdir is not None:
        anc = _existing_parent(workdir)
        if os.access(anc, os.W_OK | os.X_OK):
            note = "" if anc == workdir.expanduser().absolute() else f" (can create it in {anc})"
            out.append(Check("workdir:write", OK, f"{workdir} is writable{note}", group="system"))
        else:
            out.append(
                Check(
                    "workdir:write",
                    FAIL,
                    f"no write permission for {anc} (needed for {workdir})",
                    "Choose another project.workdir or fix the permissions (chmod u+w).",
                    "system",
                )
            )
    return out


def check_project_limits(cfg: Any) -> list[Check]:
    """Compare VELEST array limits with the Stage 1 data size (if Stage 1 has run)."""
    try:
        from osv.stages import stage_dirname

        stage = Path(cfg.project.workdir) / stage_dirname("prep")
    except Exception:  # noqa: BLE001
        return []
    ev_f, st_f, pk_f = (stage / n for n in ("events.parquet", "stations.parquet", "picks.parquet"))
    if not (ev_f.is_file() and st_f.is_file()):
        return [
            Check(
                "project:data_size",
                OK,
                f"Stage 1 outputs not found in {stage}; array limits not compared",
                group="project",
            )
        ]
    try:
        import pandas as pd

        n_ev = len(pd.read_parquet(ev_f))
        n_st = len(pd.read_parquet(st_f))
        max_picks = 0
        if pk_f.is_file():
            picks = pd.read_parquet(pk_f)
            if "event_id" in picks:
                max_picks = int(picks.groupby("event_id").size().max())
    except Exception as exc:  # noqa: BLE001
        return [
            Check("project:data_size", WARN, f"cannot read Stage 1 outputs: {exc}", group="project")
        ]
    try:
        lim, _ = velest_limits(tools.which_tool("velest"))
    except tools.ToolNotFoundError:
        lim = {}
    if not lim:
        return [
            Check(
                "project:data_size", WARN, "VELEST limits unknown; cannot compare", group="project"
            )
        ]
    out: list[Check] = []
    for label, n, key in (
        ("events", n_ev, "ieq"),
        ("stations", n_st, "ist"),
        ("picks per event", max_picks, "maxobs"),
    ):
        if key not in lim or n == 0:
            continue
        cap = lim[key]
        flag = {"ieq": "--velest-ieq", "ist": "--velest-ist", "maxobs": "--velest-maxobs"}[key]
        msg = f"{n} {label} in the project vs VELEST {key}={cap}"
        if n > cap:
            out.append(
                Check(
                    f"project:{key}",
                    FAIL,
                    msg + ": too many",
                    f"Rebuild VELEST with a larger limit: ./install.sh {flag} {int(n * 1.2) + 1}.",
                    "project",
                    details={"n": n, "limit": cap},
                )
            )
        elif n > 0.9 * cap:
            out.append(
                Check(
                    f"project:{key}",
                    WARN,
                    msg + ": within 10% of the limit",
                    f"Consider rebuilding VELEST with {flag} {int(n * 1.5)}.",
                    "project",
                    details={"n": n, "limit": cap},
                )
            )
        else:
            out.append(
                Check(f"project:{key}", OK, msg, group="project", details={"n": n, "limit": cap})
            )
    return out


# --------------------------------------------------------------------------- smoke tests


def _smoke_dir(name: str) -> Path:
    return Path(str(resources.files("osv") / "_smoke" / name))


def _timed(
    name: str, group: str, fn: Callable[[], tuple[str, str, str]], skip_fix: str = ""
) -> Check:
    """Run ``fn() -> (status, message, fix)`` with timing; exceptions become failures."""
    t0 = time.monotonic()
    try:
        status, msg, fix = fn()
    except tools.ToolNotFoundError as exc:
        return Check(name, FAIL, f"tool not found: {exc}", skip_fix, group, time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        status, msg, fix = FAIL, f"{type(exc).__name__}: {exc}", skip_fix
    return Check(name, status, msg, fix, group, round(time.monotonic() - t0, 3))


def _tmp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="osv-doctor-"))


def smoke_velest() -> tuple[str, str, str]:
    """Run VELEST on the packaged 25-event dataset and compare the final RMS."""
    root = _tmp_root()
    try:
        shutil.copytree(_smoke_dir("velest"), root, dirs_exist_ok=True)
        res = tools.run_tool("velest", root, check=False, timeout=300)
        if not res.ok:
            tail = "\n".join((res.stderr_text() + res.stdout_text()).splitlines()[-5:])
            lim, _ = velest_limits(tools.which_tool("velest"))
            hint = ""
            if lim and (
                lim.get("ieq", 1e9) < VELEST_N_EVENTS or lim.get("ist", 1e9) < VELEST_MIN_STATIONS
            ):
                hint = " The build's array limits are smaller than the smoke dataset needs."
            return (
                FAIL,
                f"velest exited with code {res.returncode}: {tail or res.detected_errors}",
                "Rebuild with ./install.sh --clean; see the logs in " + str(root) + "." + hint,
            )
        out = (root / "velest.out").read_text(errors="replace")
        rms = [float(x) for x in re.findall(r"RMS RESIDUAL=\s*([\d.]+)", out)]
        n_hyp = len(
            re.findall(
                r"^[ 0-9]{6} [ 0-9]{4} +[\d.]+ +[\d.]+[NS] +[\d.]+[EW]",
                (root / "velout.cnv").read_text(),
                re.M,
            )
        )
        if not rms or n_hyp != VELEST_N_EVENTS:
            return (
                FAIL,
                f"unexpected output: {n_hyp} events, rms values {rms[-1:]}",
                ("Rebuild with ./install.sh --clean."),
            )
        if abs(rms[-1] - VELEST_RMS_REF) > VELEST_RMS_RTOL * VELEST_RMS_REF:
            return (
                FAIL,
                f"final RMS {rms[-1]:.4f} s differs from the reference {VELEST_RMS_REF:.4f} s",
                "The build may be miscompiled; rebuild with ./install.sh --clean "
                "(avoid -ffast-math/-march flags that change results).",
            )
        return (
            OK,
            f"25 events relocated, final RMS {rms[-1]:.4f} s (reference {VELEST_RMS_REF:.4f})",
            "",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _arrival_times(path: Path) -> list[float]:
    return [float(ln.split()[4]) for ln in path.read_text().splitlines() if len(ln.split()) >= 5]


def smoke_fm3d() -> tuple[str, str, str]:
    """Run fm3d on a packaged model and compare arrival times with the reference."""
    root = _tmp_root()
    try:
        shutil.copytree(_smoke_dir("fm3d"), root, dirs_exist_ok=True)
        res = tools.run_tool("fm3d", root, check=False, timeout=300)
        if not res.ok or not (root / "arrivals.dat").is_file():
            tail = "\n".join((res.stderr_text() + res.stdout_text()).splitlines()[-5:])
            return (
                FAIL,
                f"fm3d exited with code {res.returncode}: {tail}",
                ("Rebuild with ./install.sh --clean; logs in " + str(root) + "."),
            )
        new = _arrival_times(root / "arrivals.dat")
        ref = _arrival_times(root / "arrivals_ref.dat")
        if len(new) != len(ref):
            return (
                FAIL,
                f"{len(new)} arrivals, expected {len(ref)}",
                "Rebuild with ./install.sh --clean.",
            )
        bad = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(new, ref, strict=True))
            if (a < 0) != (b < 0) or abs(a - b) > max(FM3D_ATOL, FM3D_RTOL * abs(b))
        ]
        if bad:
            i, a, b = bad[0]
            return (
                FAIL,
                f"{len(bad)} of {len(ref)} arrival times differ from the reference "
                f"(first: #{i + 1} {a:.4f} vs {b:.4f} s)",
                "The build may be miscompiled; rebuild with ./install.sh --clean.",
            )
        return OK, f"{len(new)} arrival times match the reference (atol {FM3D_ATOL} s)", ""
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _extras_example() -> Path | None:
    p = Path(__file__).resolve().parents[2] / "fortran" / "fmtomo_extras" / "observed" / "invert_p"
    return p if p.is_dir() else None


def smoke_gridloc() -> tuple[str, str, str]:
    """fm3d saves time fields, then gridloc relocates the Extras example sources."""
    ex = _extras_example()
    if ex is None:
        return WARN, "skipped: the Extras example needs a source checkout of the repository", ""
    root = _tmp_root()
    try:
        shutil.copytree(
            ex, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns("gmtplot", "mkmodel")
        )
        r1 = tools.run_tool("fm3d", root, check=False, timeout=600)
        if not r1.ok:
            return FAIL, f"fm3d failed (code {r1.returncode})", "Rebuild with ./install.sh --clean."
        r2 = tools.run_tool("gridloc", root, check=False, timeout=600)
        text = r2.stdout_text()
        if not r2.ok or "successfully completed" not in text:
            return (
                FAIL,
                f"gridloc failed (code {r2.returncode})",
                "Rebuild with ./install.sh --clean.",
            )
        m = re.search(r"(\d+)\s+out of\s+(\d+)", text)
        return OK, "gridloc relocated events" + (f" ({m.group(0)})" if m else ""), ""
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_smoke_tests(*, gridloc: bool = False) -> list[Check]:
    """Smoke tests (each fails/skips clearly if its tool is missing).

    The gridloc test takes about 35 s (fm3d on a large grid), so it only runs with
    ``gridloc=True``.
    """
    out: list[Check] = []
    avail = tools.available_tools()
    plan: list[tuple[str, list[str], Callable[[], tuple[str, str, str]]]] = [
        ("smoke:velest", ["velest"], smoke_velest),
        ("smoke:fm3d", ["fm3d"], smoke_fm3d),
        ("smoke:gridloc", ["fm3d", "gridloc"], smoke_gridloc),
    ]
    for name, needs, fn in plan:
        if name == "smoke:gridloc" and not gridloc:
            out.append(Check(name, OK, "not run (about 35 s; use --gridloc)", group="smoke"))
            continue
        missing = [n for n in needs if avail.get(n) is None]
        if missing:
            status = WARN if name == "smoke:gridloc" else FAIL
            out.append(
                Check(
                    name,
                    status,
                    f"skipped: {', '.join(missing)} not found",
                    "Install the missing executables first (see the tool checks above).",
                    "smoke",
                )
            )
            continue
        out.append(_timed(name, "smoke", fn))
    return out


# --------------------------------------------------------------------------- driver


def run_doctor(
    *, quick: bool = False, config: Path | None = None, gridloc: bool = False
) -> list[Check]:
    """Run all checks. ``config`` enables project-specific checks."""
    workdir: Path | None = None
    n_jobs: int | None = None
    cfg: Any = None
    results: list[Check] = []
    if config is not None:
        try:
            from osv.config import load_config

            cfg = load_config(config)
            workdir, n_jobs = Path(cfg.project.workdir), int(cfg.project.n_jobs)
        except Exception as exc:  # noqa: BLE001
            results.append(
                Check(
                    "config",
                    FAIL,
                    f"cannot load {config}: {exc}",
                    "Fix the config (see `osv config show`).",
                    "project",
                )
            )
    results += check_python()
    results += check_proj()
    results += check_tools()
    results += check_build()
    results += check_system(workdir, n_jobs)
    if cfg is not None:
        results += check_project_limits(cfg)
    if not quick:
        results += run_smoke_tests(gridloc=gridloc)
    return results


def exit_code(checks: Iterable[Check]) -> int:
    """0 if no check failed, else 1."""
    return 1 if any(c.status == FAIL for c in checks) else 0


def summary_counts(checks: Iterable[Check]) -> dict[str, int]:
    """Number of checks per status."""
    c = {OK: 0, WARN: 0, FAIL: 0}
    for ch in checks:
        c[ch.status] += 1
    return c


def to_json(checks: list[Check], *, quick: bool = False) -> str:
    """Machine-readable report."""
    from osv import __version__

    return json.dumps(
        {
            "osv_version": __version__,
            "python": sys.version.split()[0],
            "quick": quick,
            "ok": exit_code(checks) == 0,
            "summary": summary_counts(checks),
            "checks": [asdict(c) for c in checks],
        },
        indent=2,
        default=str,
    )


_STYLE = {OK: "green", WARN: "yellow", FAIL: "red bold"}


def render_text(checks: list[Check], *, use_rich: bool = True) -> str:
    """Readable table (rich if available, else plain text)."""
    counts = summary_counts(checks)
    foot = f"{counts[OK]} ok, {counts[WARN]} warnings, {counts[FAIL]} failures"
    if use_rich:
        try:
            from io import StringIO

            from rich.console import Console
            from rich.table import Table

            table = Table(title="osv doctor", show_lines=False)
            for col in ("Status", "Check", "Result", "Time"):
                table.add_column(col, overflow="fold")
            for c in checks:
                msg = c.message + (f"\n  fix: {c.fix}" if c.fix and c.status != OK else "")
                t = f"{c.seconds:.2f}s" if c.seconds is not None else ""
                table.add_row(f"[{_STYLE[c.status]}]{c.status.upper()}[/]", c.name, msg, t)
            buf = StringIO()
            Console(
                file=buf,
                width=max(100, shutil.get_terminal_size((100, 20)).columns),
                force_terminal=False,
            ).print(table)
            return buf.getvalue() + foot
        except ImportError:
            pass
    lines = []
    for c in checks:
        t = f" [{c.seconds:.2f}s]" if c.seconds is not None else ""
        lines.append(f"{c.status.upper():<5} {c.name:<22} {c.message}{t}")
        if c.fix and c.status != OK:
            lines.append(f"      fix: {c.fix}")
    return "\n".join(lines) + "\n" + foot
