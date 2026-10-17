"""Guard against a stale ``PROJ_DATA``/``PROJ_LIB`` (e.g. an old conda ``proj.db``).

rasterio and pyproj each bundle their own PROJ library and database.  A user shell that
exports ``PROJ_DATA`` pointing at a different (older) PROJ install makes EPSG lookups fail
with "DATABASE.LAYOUT.VERSION.MINOR ... another PROJ installation".  :func:`ensure_proj_data`
detects that and points the environment at the bundled data.  Set ``OSV_KEEP_PROJ_DATA=1``
to disable the override.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sqlite3
import sys
from pathlib import Path

log = logging.getLogger(__name__)

ENV_KEEP = "OSV_KEEP_PROJ_DATA"
_VARS = ("PROJ_DATA", "PROJ_LIB")


def _pkg_dir(pkg: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def bundled_proj_data() -> Path | None:
    """PROJ data dir bundled with rasterio (preferred: it is the env-var consumer) or pyproj."""
    for pkg, rel in (
        ("rasterio", "proj_data"),
        ("pyproj", "proj_dir/share/proj"),
    ):
        base = _pkg_dir(pkg)
        if base is not None and (base / rel / "proj.db").is_file():
            return base / rel
    return None


def proj_db_minor(directory: Path | str) -> int | None:
    """``DATABASE.LAYOUT.VERSION.MINOR`` of ``directory/proj.db``, or None if unreadable."""
    db = Path(directory) / "proj.db"
    if not db.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
            ).fetchone()
        finally:
            con.close()
        return int(row[0]) if row else None
    except (sqlite3.Error, ValueError):
        return None


def _refresh_loaded_libs(good: Path) -> None:
    """Tell an already-imported rasterio/GDAL about the new search path."""
    if "rasterio" not in sys.modules:
        return
    try:
        from rasterio._env import set_proj_data_search_path

        set_proj_data_search_path(str(good))
    except Exception:  # noqa: BLE001 - best effort; private API
        log.debug("could not update rasterio's PROJ search path", exc_info=True)


def ensure_proj_data() -> str | None:
    """Override a stale PROJ_DATA/PROJ_LIB; return the new directory if changed, else None."""
    if os.environ.get(ENV_KEEP, "") not in ("", "0"):
        return None
    current = [os.environ[v] for v in _VARS if os.environ.get(v)]
    if not current:
        return None
    good = bundled_proj_data()
    if good is None:
        return None
    want = proj_db_minor(good)
    for cur in current:
        have = proj_db_minor(cur.split(os.pathsep)[0])
        if have is not None and (want is None or have >= want):
            continue
        for v in _VARS:
            os.environ[v] = str(good)
        _refresh_loaded_libs(good)
        log.warning(
            "PROJ_DATA/PROJ_LIB=%s has an incompatible proj.db (layout minor %s, need >= %s); "
            "using %s instead. Set %s=1 to keep your value.",
            cur,
            have,
            want,
            good,
            ENV_KEEP,
        )
        return str(good)
    return None
