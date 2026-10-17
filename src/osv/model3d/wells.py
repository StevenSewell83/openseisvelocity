"""Well-log velocity model: LAS sonic logs to a 3-D grid, and unit velocities from logs (task 4.6).

Processing chain per well
-------------------------
1. Read the sonic (``curve``), optional shear sonic and density curves from the LAS file with
   ``lasio`` (nulls such as -999.25 become NaN, non-physical values are removed).
2. Slowness (us/ft or us/m, ``units: auto`` uses the curve unit, else a median-based guess) is
   converted to km/s.
3. Spikes are removed with a running-median filter (``despike_window_m``, threshold in MADs),
   then the log is smoothed over ``smoothing_m`` (a running mean of *slowness*, i.e. a
   travel-time-consistent average).  Gaps stay gaps.
4. Measured depth (MD) is converted to true vertical depth (TVD) and lateral offset with the
   minimum-curvature method when a deviation survey is given (vertical well otherwise), and
   to depth below sea level with the depth reference (``KB``/``GL``: ``elevation_m - TVD``).
5. The log is upscaled to the model nodes (block of one node spacing centred on each node):

   * ``backus``: Backus (1962) average for a vertically propagating wave.  The P modulus
     ``M = rho vp^2`` and shear modulus ``mu = rho vs^2`` are harmonically averaged, the
     density arithmetically.  If no density log exists, **Gardner's relation**
     ``rho = 0.31 (1000 vp)^0.25`` (g/cc, vp in km/s) supplies it.
   * ``average``: slowness (travel-time) average, ``v = 1 / <1/v>``.
   * ``none``: the log is sampled at the node depth.

Lateral interpolation (per depth slice, using only wells that have data at that depth, at
the lateral position of the well *at that depth* for deviated wells) is IDW, ordinary kriging
(spherical/exponential/gaussian variogram) or nearest-well.  Beyond ``max_distance_km`` from
the nearest well the background is used, with a linear taper over ``taper_km``; the result
carries a ``weight`` (0..1) and a boolean ``mask`` (weight > 0).

Geology + wells
---------------
:func:`wells_horizons` fits the parameters of each unit's velocity law (the law *type* comes
from the horizon configuration: constant, gradient or exponential) to the well-log samples
inside the unit, and returns modified horizon configurations for
:func:`osv.model3d.geology.build_geology_model` (see :func:`build_geology_wells_model`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

from osv.config.schema import WellsConfig
from osv.model3d.geology import Background, GeologyFinding, apply_law, build_geology_model
from osv.model3d.grid import KM_PER_DEG, ModelGrid
from osv.model3d.horizons_io import read_horizon

log = logging.getLogger(__name__)

__all__ = [
    "WellProfile",
    "WellsResult",
    "backus_average",
    "build_geology_wells_model",
    "build_wells_model",
    "interpolate_lateral",
    "load_wells",
    "minimum_curvature",
    "read_las_curves",
    "read_wells_table",
    "slowness_to_velocity",
    "wells_horizons",
]

M_PER_FT = 0.3048
#: Conventional LAS null values that lasio does not always convert (header NULL mismatch).
LAS_NULLS = (-999.25, -999.0, -9999.0, -99999.0)
#: Physical plausibility limits of sonic-derived Vp (km/s); samples outside are nulls.
VP_LIMITS_KM_S = (0.3, 9.0)
#: A node is covered by a well when at least this fraction of its block holds valid samples.
MIN_BLOCK_COVERAGE = 0.5
#: Median slowness above which ``units: auto`` (without a curve unit) assumes us/m.
AUTO_US_PER_M_THRESHOLD = 200.0
#: Minimum number of log samples in a unit for fitting its velocity law.
MIN_UNIT_SAMPLES = 10
#: Fit RMS (km/s) above which the unit fit gets a warning.
UNIT_FIT_RMS_WARN = 0.3
#: Distances (km) below this are treated as coincident with a well.
EPS_KM = 1e-6


class WellsError(ValueError):
    """Invalid well data or configuration (message says what to fix)."""


@dataclass
class WellProfile:
    """Processed log of one well.

    ``depth_km`` is depth below sea level (km, positive down) of each valid log sample,
    ``east_km``/``north_km`` the lateral offset from the wellhead at that sample and
    ``thick_km`` the vertical thickness it represents.  ``vp``/``vs``/``rho`` are despiked and
    smoothed (km/s, km/s, g/cc).  ``x_km``/``y_km`` locate the wellhead in the local frame of
    the model grid centre.
    """

    name: str
    lon: float
    lat: float
    x_km: float
    y_km: float
    depth_km: np.ndarray
    east_km: np.ndarray
    north_km: np.ndarray
    thick_km: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    has_shear: bool = False
    has_density: bool = False


@dataclass
class WellsResult:
    """Output of :func:`build_wells_model`."""

    dataset: xr.Dataset
    profiles: pd.DataFrame
    logs: pd.DataFrame
    findings: list[GeologyFinding]
    wells: list[WellProfile] = field(default_factory=list)


# ---------------------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------------------
def _cfg(cfg: Any) -> WellsConfig:
    if isinstance(cfg, WellsConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return WellsConfig.model_validate(dict(cfg))
    raise TypeError("wells config must be a WellsConfig or a dict")


def _resolve(path: Path | str, base_dir: str | Path | None) -> Path:
    p = Path(path)
    return p if p.is_absolute() or base_dir is None else Path(base_dir) / p


# ---------------------------------------------------------------------------------------
# LAS and conversions
# ---------------------------------------------------------------------------------------
def slowness_to_velocity(dt: np.ndarray, units: str = "us/ft") -> np.ndarray:
    """Convert sonic slowness to velocity in km/s (``us/ft`` or ``us/m``)."""
    dt = np.asarray(dt, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if units == "us/ft":
            return (M_PER_FT * 1e3) / dt
        if units == "us/m":
            return 1e3 / dt
    raise WellsError(f"unknown slowness units {units!r}; use 'us/ft' or 'us/m'")


def _import_lasio() -> Any:
    try:
        import lasio
    except ImportError as exc:  # pragma: no cover - exercised only without the geo extra
        raise ImportError(
            "reading LAS well logs requires the 'lasio' package; install it with "
            "`pip install lasio` (or `pip install openseisvelocity[geo]`)"
        ) from exc
    return lasio


def _unit_from_curve(unit: str) -> str | None:
    u = (unit or "").strip().lower().replace("usec", "us").replace("µs", "us")
    if not u:
        return None
    if u.endswith(("/m", "/meter", "/metre")):
        return "us/m"
    if u.endswith(("/f", "/ft", "/foot", "/feet")):
        return "us/ft"
    return None


def read_las_curves(
    path: str | Path,
    curves: Mapping[str, str | None],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Read measured depth (m) and named curves from a LAS file.

    Parameters
    ----------
    curves
        ``{role: mnemonic}`` (e.g. ``{"dt": "DT", "dts": "DTS", "rho": "RHOB"}``); roles with a
        ``None`` mnemonic or a mnemonic missing in the file are omitted from the result.

    Returns
    -------
    depth_m, data, units, header
        Measured depth in metres, the curves (nulls as NaN), the unit string of each curve and
        the header values (``XCOORD``, ``YCOORD``, ``LAT``, ``LON``, ``ELEV``, ``WELL``) found.
    """
    lasio = _import_lasio()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {path}")
    las = lasio.read(str(path))
    idx = np.asarray(las.index, dtype=float)
    depth_unit = str(las.curves[0].unit or "").strip().lower()
    if depth_unit in ("f", "ft", "feet", "foot"):
        idx = idx * M_PER_FT
    data: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    upper = {c.mnemonic.upper(): c for c in las.curves}
    for role, mnem in curves.items():
        if mnem is None or mnem.upper() not in upper:
            continue
        c = upper[mnem.upper()]
        arr = np.asarray(las[c.mnemonic], dtype=float)
        data[role] = np.where(np.isin(np.round(arr, 2), LAS_NULLS), np.nan, arr)
        units[role] = str(c.unit or "")
    header: dict[str, Any] = {}
    for key in ("XCOORD", "YCOORD", "LAT", "LATI", "LON", "LONG", "ELEV", "EKB", "KB", "WELL"):
        for sect in (las.well, las.params):
            if key in sect:
                header[key] = sect[key].value
                break
    return idx, data, units, header


def _despike(y: np.ndarray, window: int, threshold: float) -> tuple[np.ndarray, int]:
    """Replace spikes (|y - running median| > threshold * MAD) by the running median."""
    if window < 3:
        return y, 0
    s = pd.Series(y)
    med = s.rolling(window, center=True, min_periods=1).median()
    dev = (s - med).abs()
    mad = float(np.nanmedian(dev)) or float(np.nanmean(dev)) or 0.0
    bad = (dev > threshold * max(mad, 1e-12)) & s.notna()
    out = np.where(bad, med, s)
    return out, int(bad.sum())


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centered running mean ignoring NaN; NaN positions stay NaN."""
    if window < 2:
        return y
    s = pd.Series(y)
    m = s.rolling(window, center=True, min_periods=1).mean().to_numpy()
    return np.where(np.isfinite(y), m, np.nan)


def minimum_curvature(
    md: np.ndarray, incl_deg: np.ndarray, azim_deg: np.ndarray, at_md: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-curvature survey: TVD, northing and easting offsets at ``at_md`` (same unit as md).

    The survey is assumed to start vertically at MD 0 and, beyond its last station, to continue
    along the last inclination/azimuth.
    """
    md = np.asarray(md, float)
    inc = np.radians(np.asarray(incl_deg, float))
    azi = np.radians(np.asarray(azim_deg, float))
    order = np.argsort(md)
    md, inc, azi = md[order], inc[order], azi[order]
    if md[0] > 0:
        md, inc, azi = np.r_[0.0, md], np.r_[0.0, inc], np.r_[0.0, azi]
    dmd = np.diff(md)
    i1, i2, a1, a2 = inc[:-1], inc[1:], azi[:-1], azi[1:]
    cosb = np.clip(np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1)), -1, 1)
    dl = np.arccos(cosb)
    rf = np.where(dl < 1e-6, 1.0, 2.0 / np.where(dl < 1e-6, 1.0, dl) * np.tan(dl / 2.0))
    dn = 0.5 * dmd * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    de = 0.5 * dmd * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    dz = 0.5 * dmd * (np.cos(i1) + np.cos(i2)) * rf
    tvd, north, east = (np.r_[0.0, np.cumsum(v)] for v in (dz, dn, de))
    q = np.asarray(at_md, float)
    out = []
    for arr, tail in (
        (tvd, np.cos(inc[-1])),
        (north, np.sin(inc[-1]) * np.cos(azi[-1])),
        (east, np.sin(inc[-1]) * np.sin(azi[-1])),
    ):
        v = np.interp(q, md, arr)
        v = np.where(q > md[-1], arr[-1] + (q - md[-1]) * tail, v)
        out.append(v)
    return out[0], out[1], out[2]


# ---------------------------------------------------------------------------------------
# wells table
# ---------------------------------------------------------------------------------------
_ALIASES = {
    "name": ("name", "well", "well_name", "wellname", "uwi"),
    "las_file": ("las_file", "las", "file", "lasfile"),
    "x": ("x", "easting", "e", "xcoord"),
    "y": ("y", "northing", "n", "ycoord"),
    "lon": ("lon", "longitude", "long"),
    "lat": ("lat", "latitude"),
    "elevation_m": ("elevation_m", "elev_m", "kb_m", "kb", "elevation", "gl_m", "elev"),
    "deviation_file": ("deviation_file", "deviation", "survey", "dev_file"),
}


def read_wells_table(path: str | Path) -> pd.DataFrame:
    """Read the wells CSV and normalise column names (see :class:`~osv.config.schema.WellsConfig`).

    Result columns: ``name``, ``las_file``, ``x``, ``y`` (or ``lon``/``lat``), ``elevation_m``
    (0 if missing) and ``deviation_file`` (NaN if missing).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"wells table not found: {path}")
    df = pd.read_csv(path)
    low = {c.strip().lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for key, names in _ALIASES.items():
        for n in names:
            if n in low:
                out[key] = df[low[n]]
                break
    for req in ("name", "las_file"):
        if req not in out:
            raise WellsError(
                f"{path}: wells table needs a '{req}' column (found {list(df.columns)})"
            )
    if not (("x" in out and "y" in out) or ("lon" in out and "lat" in out)):
        raise WellsError(f"{path}: wells table needs x/y (with crs) or lon/lat columns")
    if "elevation_m" not in out:
        out["elevation_m"] = 0.0
    if "deviation_file" not in out:
        out["deviation_file"] = np.nan
    out["name"] = out["name"].astype(str)
    return out


def _read_deviation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    low = {c.strip().lower(): c for c in df.columns}

    def col(*names: str) -> np.ndarray:
        for n in names:
            if n in low:
                return df[low[n]].to_numpy(float)
        raise WellsError(f"{path}: deviation survey needs a column named one of {names}")

    return col("md", "measured_depth"), col("incl", "inclination", "inc"), col("azi", "azimuth")


def _to_local_km(
    lon: np.ndarray, lat: np.ndarray, grid: ModelGrid
) -> tuple[np.ndarray, np.ndarray]:
    cl = np.cos(np.radians(grid.centre_lat))
    return (
        (np.asarray(lon) - grid.centre_lon) * KM_PER_DEG * cl,
        (np.asarray(lat) - grid.centre_lat) * KM_PER_DEG,
    )


def _from_local_km(x: np.ndarray, y: np.ndarray, grid: ModelGrid) -> tuple[np.ndarray, np.ndarray]:
    cl = np.cos(np.radians(grid.centre_lat))
    return grid.centre_lon + x / (KM_PER_DEG * cl), grid.centre_lat + y / KM_PER_DEG


def _header_table(las_dir: Path, curve: str) -> pd.DataFrame:
    rows = []
    for f in sorted(las_dir.glob("*.[lL][aA][sS]")):
        _, _, _, h = read_las_curves(f, {"dt": curve})
        rows.append(
            {
                "name": str(h.get("WELL") or f.stem),
                "las_file": f.name,
                "x": h.get("XCOORD", np.nan),
                "y": h.get("YCOORD", np.nan),
                "lon": h.get("LON", h.get("LONG", np.nan)),
                "lat": h.get("LAT", h.get("LATI", np.nan)),
                "elevation_m": h.get("EKB", h.get("KB", h.get("ELEV", 0.0))),
                "deviation_file": np.nan,
            }
        )
    if not rows:
        raise WellsError(f"{las_dir}: no .las files found")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# per-well processing
# ---------------------------------------------------------------------------------------
def _gardner(vp_km_s: np.ndarray) -> np.ndarray:
    """Gardner et al. (1974) density (g/cc) from Vp (km/s): 0.31 * (1000 vp)^0.25."""
    return 0.31 * (1000.0 * np.maximum(vp_km_s, 1e-3)) ** 0.25


def _velocity_curve(
    dt: np.ndarray, unit: str, units_cfg: str, label: str, well: str, findings: list[GeologyFinding]
) -> np.ndarray:
    use = units_cfg
    if use == "auto":
        use = _unit_from_curve(unit) or (
            "us/m" if np.nanmedian(dt) > AUTO_US_PER_M_THRESHOLD else "us/ft"
        )
        findings.append(
            GeologyFinding("info", "sonic_units", f"well {well}: {label} read as {use}")
        )
    dt = np.where(dt > 0, dt, np.nan)
    return slowness_to_velocity(dt, use)


def _process_well(
    row: pd.Series,
    cfg: WellsConfig,
    grid: ModelGrid,
    las_dir: Path | None,
    base_dir: str | Path | None,
    tr: Transformer | None,
    findings: list[GeologyFinding],
) -> WellProfile | None:
    name = str(row["name"])
    las_path = Path(str(row["las_file"]))
    if not las_path.is_absolute():
        las_path = (las_dir or Path(base_dir or ".")) / las_path
    md, data, units, _ = read_las_curves(
        las_path, {"dt": cfg.curve, "dts": cfg.shear_curve, "rho": cfg.density_curve}
    )
    if "dt" not in data:
        findings.append(
            GeologyFinding(
                "warning",
                "missing_curve",
                f"well {name}: curve '{cfg.curve}' not in {las_path.name}",
            )
        )
        return None
    vp = _velocity_curve(data["dt"], units["dt"], cfg.units, cfg.curve, name, findings)
    vp = np.where((vp >= VP_LIMITS_KM_S[0]) & (vp <= VP_LIMITS_KM_S[1]), vp, np.nan)
    has_shear = "dts" in data
    if has_shear:
        vs = _velocity_curve(data["dts"], units["dts"], cfg.units, str(cfg.shear_curve), name, [])
        vs = np.where((vs > 0.1) & (vs < VP_LIMITS_KM_S[1]), vs, np.nan)
    else:
        vs = np.full_like(vp, np.nan)
        if cfg.shear_curve:
            findings.append(
                GeologyFinding(
                    "info",
                    "missing_curve",
                    f"well {name}: no shear curve '{cfg.shear_curve}'; using Vp/Vs={cfg.vpvs}",
                )
            )
    has_density = "rho" in data
    rho = np.full_like(vp, np.nan)
    if has_density:
        rho = np.asarray(data["rho"], float)
        if np.nanmedian(rho) > 100:  # kg/m3
            rho = rho / 1000.0
        rho = np.where((rho > 0.5) & (rho < 5.0), rho, np.nan)

    valid = np.isfinite(md) & np.isfinite(vp)
    if valid.sum() < 2:
        findings.append(
            GeologyFinding("warning", "short_log", f"well {name}: fewer than 2 valid log samples")
        )
        return None
    md, vp, vs, rho = md[valid], vp[valid], vs[valid], rho[valid]
    step = float(np.median(np.diff(md))) if md.size > 1 else 1.0
    step = step if step > 0 else 1.0
    n_spike = 0
    win = int(round(cfg.despike_window_m / step))
    win += 1 - win % 2  # odd
    for arr in (vp, vs):
        # despike slowness-equivalent (velocity) series; vs may be all NaN
        if np.isfinite(arr).any():
            arr[:], k = _despike(arr, win, cfg.despike_threshold)
            n_spike += k
    if n_spike:
        findings.append(
            GeologyFinding(
                "info", "despiked", f"well {name}: {n_spike} spike samples replaced", n_spike
            )
        )
    sw = int(round(cfg.smoothing_m / step))
    sw += 1 - sw % 2 if sw > 1 else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = 1.0 / _smooth(1.0 / vp, sw)
        vs = 1.0 / _smooth(1.0 / vs, sw)
    rho = _smooth(rho, sw)

    # geometry: MD -> TVD, lateral offsets
    if pd.notna(row.get("deviation_file")):
        dpath = _resolve(str(row["deviation_file"]), base_dir)
        if not dpath.exists() and las_dir is not None and (las_dir / dpath.name).exists():
            dpath = las_dir / dpath.name
        dmd, dinc, dazi = _read_deviation(dpath)
        tvd, north, east = minimum_curvature(dmd, dinc, dazi, md)
        if md.max() > dmd.max() + 1e-6:
            findings.append(
                GeologyFinding(
                    "warning",
                    "deviation_short",
                    f"well {name}: log extends {md.max() - dmd.max():.0f} m below the last survey "
                    "station; the last inclination/azimuth is extended",
                )
            )
    else:
        tvd, north, east = md.copy(), np.zeros_like(md), np.zeros_like(md)
    datum = 0.0 if cfg.depth_reference == "MSL" else float(row.get("elevation_m", 0.0) or 0.0)
    depth_km = (tvd - datum) / 1e3
    thick = np.gradient(tvd) / 1e3 if tvd.size > 1 else np.zeros_like(tvd)
    thick = np.maximum(thick, 0.0)

    if "lon" in row and pd.notna(row.get("lon")) and "lat" in row and pd.notna(row.get("lat")):
        lon, lat = float(row["lon"]), float(row["lat"])
    elif tr is not None:
        lon, lat = (float(v) for v in tr.transform(float(row["x"]), float(row["y"])))
    else:
        lon, lat = float(row["x"]), float(row["y"])
    x, y = _to_local_km(np.array(lon), np.array(lat), grid)

    if not np.isfinite(vs).any():
        vs_fill = vp / cfg.vpvs
    else:
        vs_fill = np.where(np.isfinite(vs), vs, vp / cfg.vpvs)
    if has_shear and (~np.isfinite(vs)).any():
        findings.append(
            GeologyFinding(
                "info", "shear_gaps", f"well {name}: shear log gaps filled with Vp/Vs={cfg.vpvs}"
            )
        )
    if not has_density:
        findings.append(
            GeologyFinding(
                "info", "gardner", f"well {name}: no density log; Gardner's relation used"
            )
        )
    rho_fill = np.where(np.isfinite(rho), rho, _gardner(vp))
    if (depth_km.max() - depth_km.min()) < grid.dz_km:
        findings.append(
            GeologyFinding(
                "warning",
                "short_log",
                f"well {name}: log spans only {(depth_km.max() - depth_km.min()) * 1e3:.0f} m, "
                "less than one model node spacing",
            )
        )
    return WellProfile(
        name=name,
        lon=lon,
        lat=lat,
        x_km=float(x),
        y_km=float(y),
        depth_km=depth_km,
        east_km=east / 1e3,
        north_km=north / 1e3,
        thick_km=thick,
        vp=vp,
        vs=vs_fill,
        rho=rho_fill,
        has_shear=has_shear,
        has_density=has_density,
    )


def load_wells(
    cfg: Any,
    grid: ModelGrid,
    *,
    base_dir: str | Path | None = None,
) -> tuple[list[WellProfile], list[GeologyFinding]]:
    """Read and condition all wells of a ``model3d.user_model.wells`` configuration.

    Returns the processed :class:`WellProfile` list and findings (missing curves, short logs,
    wells outside the grid, ...).
    """
    c = _cfg(cfg)
    findings: list[GeologyFinding] = []
    las_dir = _resolve(c.las_dir, base_dir) if c.las_dir is not None else None
    if c.table is not None:
        table = read_wells_table(_resolve(c.table, base_dir))
    elif las_dir is not None:
        table = _header_table(las_dir, c.curve)
        if table[["x", "y", "lon", "lat"]].isna().all(axis=None):
            raise WellsError(
                f"{las_dir}: LAS headers hold no well coordinates; give wells.table with "
                "name, las_file, x, y (or lon, lat)"
            )
    else:
        raise WellsError("wells: set las_dir and/or table")
    tr = None
    if c.crs is not None and CRS.from_user_input(c.crs) != CRS.from_epsg(4326):
        tr = Transformer.from_crs(CRS.from_user_input(c.crs), CRS.from_epsg(4326), always_xy=True)
    wells: list[WellProfile] = []
    for _, row in table.iterrows():
        w = _process_well(row, c, grid, las_dir, base_dir, tr, findings)
        if w is None:
            continue
        inside = (
            grid.lat_full[0] <= w.lat <= grid.lat_full[-1]
            and grid.lon_full[0] <= w.lon <= grid.lon_full[-1]
        )
        if not inside:
            findings.append(
                GeologyFinding(
                    "warning",
                    "well_outside_grid",
                    f"well {w.name} ({w.lat:.4f}, {w.lon:.4f}) lies outside the model grid",
                )
            )
        wells.append(w)
    if not wells:
        raise WellsError("no usable wells were read; see the log for the reasons")
    for f in findings:
        log.log(logging.WARNING if f.level == "warning" else logging.INFO, f.message)
    return wells, findings


# ---------------------------------------------------------------------------------------
# upscaling
# ---------------------------------------------------------------------------------------
def backus_average(
    vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, w: np.ndarray
) -> tuple[float, float, float]:
    """Backus average of a layered block (weights ``w`` = thickness).

    ``M = rho vp^2`` and ``mu = rho vs^2`` are harmonically averaged, density arithmetically.
    Returns ``(vp, vs, rho)`` of the equivalent medium.
    """
    w = np.asarray(w, float) / np.sum(w)
    rho_e = float(np.sum(w * rho))
    m_e = 1.0 / float(np.sum(w / (rho * vp**2)))
    mu_e = 1.0 / float(np.sum(w / (rho * vs**2)))
    return float(np.sqrt(m_e / rho_e)), float(np.sqrt(mu_e / rho_e)), rho_e


def _node_blocks(nodes: np.ndarray) -> np.ndarray:
    """Block edges (n+1) halfway between ascending nodes (edge nodes use their one spacing)."""
    mid = 0.5 * (nodes[1:] + nodes[:-1])
    return np.r_[nodes[0] - (mid[0] - nodes[0]), mid, nodes[-1] + (nodes[-1] - mid[-1])]


def _upscale(w: WellProfile, nodes: np.ndarray, method: str) -> tuple[np.ndarray, ...]:
    n = nodes.size
    vp, vs, rho = (np.full(n, np.nan) for _ in range(3))
    edges = _node_blocks(nodes)
    z = w.depth_km
    if method == "none":
        inside = (nodes >= z.min()) & (nodes <= z.max())
        for arr, src in ((vp, w.vp), (vs, w.vs), (rho, w.rho)):
            arr[inside] = np.interp(nodes[inside], z, src)
        return vp, vs, rho
    which = np.clip(np.searchsorted(edges, z) - 1, -1, n)
    for k in range(n):
        sel = which == k
        if not sel.any():
            continue
        wt = w.thick_km[sel]
        if wt.sum() < MIN_BLOCK_COVERAGE * (edges[k + 1] - edges[k]):
            continue
        if method == "backus":
            vp[k], vs[k], rho[k] = backus_average(w.vp[sel], w.vs[sel], w.rho[sel], wt)
        else:  # slowness average
            ww = wt / wt.sum()
            vp[k] = 1.0 / float(np.sum(ww / w.vp[sel]))
            vs[k] = 1.0 / float(np.sum(ww / w.vs[sel]))
            rho[k] = float(np.sum(ww * w.rho[sel]))
    return vp, vs, rho


# ---------------------------------------------------------------------------------------
# lateral interpolation
# ---------------------------------------------------------------------------------------
def _variogram(h: np.ndarray, model: str, rng: float, sill: float, nugget: float) -> np.ndarray:
    """Semivariance; ``nugget`` applies for h > 0 (gamma(0) = 0)."""
    r = np.asarray(h, float) / rng
    psill = sill - nugget
    if model == "spherical":
        g = np.where(r < 1, 1.5 * r - 0.5 * r**3, 1.0)
    elif model == "exponential":
        g = 1.0 - np.exp(-3.0 * r)
    elif model == "gaussian":
        g = 1.0 - np.exp(-3.0 * r**2)
    else:
        raise WellsError(f"unknown variogram model {model!r}")
    return np.where(h > 0, nugget + psill * g, 0.0)


def interpolate_lateral(
    xn: np.ndarray,
    yn: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    values: np.ndarray,
    *,
    method: str = "kriging",
    power: float = 2.0,
    variogram: Mapping[str, Any] | Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate point ``values`` at ``(px, py)`` (km) to nodes ``(xn, yn)`` (km).

    Returns ``(value, distance_to_nearest_point)`` with the shape of ``xn``.
    ``method``: ``kriging`` (ordinary), ``idw`` (``linear`` is an alias) or ``nearest``.
    """
    shp = xn.shape
    q = np.column_stack([xn.ravel(), yn.ravel()])
    p = np.column_stack([np.atleast_1d(px), np.atleast_1d(py)])
    v = np.atleast_1d(values).astype(float)
    d = np.hypot(q[:, None, 0] - p[None, :, 0], q[:, None, 1] - p[None, :, 1])
    dmin = d.min(axis=1)
    if p.shape[0] == 1 or method == "nearest":
        out = v[np.argmin(d, axis=1)]
    elif method in ("idw", "linear"):
        wgt = 1.0 / np.maximum(d, EPS_KM) ** power
        out = (wgt * v).sum(axis=1) / wgt.sum(axis=1)
    elif method == "kriging":
        vg = variogram or {}

        def g(key: str, default: float | str) -> Any:
            return vg.get(key, default) if isinstance(vg, Mapping) else getattr(vg, key, default)

        args = (str(g("model", "spherical")), float(g("range_km", 10.0)))
        args += (float(g("sill", 1.0)), float(g("nugget", 0.0)))
        n = p.shape[0]
        dpp = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        a = np.ones((n + 1, n + 1))
        a[:n, :n] = _variogram(dpp, *args)
        a[n, n] = 0.0
        # tiny regularisation guards against (near-)coincident wells
        a[:n, :n] += np.eye(n) * 1e-9 * args[2]
        rhs = np.ones((n + 1, q.shape[0]))
        rhs[:n] = _variogram(d, *args).T
        lam = np.linalg.solve(a, rhs)[:n]
        out = lam.T @ v
        at = dmin < EPS_KM  # exact at the wells
        if at.any():
            out[at] = v[np.argmin(d[at], axis=1)]
    else:
        raise WellsError(f"unknown interpolation {method!r}; use kriging, idw or nearest")
    return out.reshape(shp), dmin.reshape(shp)


def _bg_vp(bg: Background, depth: np.ndarray) -> np.ndarray:
    if bg is None:
        return np.full(depth.shape, np.nan)
    if callable(bg):
        return np.asarray(bg(depth), dtype=float)
    return np.full(depth.shape, float(bg))


def build_wells_model(
    cfg: Any,
    grid: ModelGrid,
    background: Background = None,
    *,
    wells: Sequence[WellProfile] | None = None,
    base_dir: str | Path | None = None,
) -> WellsResult:
    """Interpolate upscaled well velocities onto the (full) model grid.

    Parameters
    ----------
    cfg
        ``user_model.wells`` configuration (:class:`~osv.config.schema.WellsConfig` or dict).
    grid
        Model grid; the output is on its full grid (including cushions).
    background
        ``f(depth_km) -> vp``, a constant Vp, or ``None`` (NaN where no well constrains).
        Background Vs is ``vp / cfg.vpvs``.
    wells
        Already processed wells (from :func:`load_wells`); read from ``cfg`` when omitted.

    Returns
    -------
    WellsResult
        ``dataset`` with ``vp``, ``vs`` (km/s), ``weight`` (0..1) and boolean ``mask``
        (weight > 0), dims ``(depth, lat, lon)``; ``profiles`` (upscaled values at the model
        nodes per well: ``well, depth_km, vp, vs, rho, lat, lon``); ``logs`` (conditioned
        log samples: ``well, depth_km, vp, vs, rho``); ``findings``.
    """
    c = _cfg(cfg)
    findings: list[GeologyFinding] = []
    if wells is None:
        wells, findings = load_wells(c, grid, base_dir=base_dir)
    depth = grid.depth_full
    lat, lon = grid.lat_full, grid.lon_full
    nz, nlat, nlon = depth.size, lat.size, lon.size
    xn, yn = _to_local_km(*np.meshgrid(lon, lat), grid)
    up = [_upscale(w, depth, c.upscale) for w in wells]

    vp = np.full((nz, nlat, nlon), np.nan)
    vs = np.full_like(vp, np.nan)
    wgt = np.zeros_like(vp)
    rows = []
    for k in range(nz):
        have = [i for i in range(len(wells)) if np.isfinite(up[i][0][k])]
        for i, w in enumerate(wells):
            if i in have:
                lo, la = _from_local_km(*_offset_xy(w, depth[k]), grid)
                rows.append(
                    {
                        "well": w.name,
                        "depth_km": depth[k],
                        "vp": up[i][0][k],
                        "vs": up[i][1][k],
                        "rho": up[i][2][k],
                        "lat": float(la),
                        "lon": float(lo),
                    }
                )
        if not have:
            continue
        pts = np.array([_offset_xy(wells[i], depth[k]) for i in have])
        res = []
        for j in (0, 1):
            vals = np.array([up[i][j][k] for i in have])
            v, dmin = interpolate_lateral(
                xn,
                yn,
                pts[:, 0],
                pts[:, 1],
                vals,
                method=c.interp,
                power=c.idw_power,
                variogram=c.variogram,
            )
            res.append(v)
        if c.taper_km > 0:
            w_k = np.clip((c.max_distance_km - dmin) / c.taper_km, 0.0, 1.0)
        else:
            w_k = (dmin <= c.max_distance_km).astype(float)
        bgp = _bg_vp(background, np.full((nlat, nlon), depth[k]))
        bgs = bgp / c.vpvs
        vp[k] = np.where(
            w_k > 0, np.where(np.isfinite(bgp), w_k * res[0] + (1 - w_k) * bgp, res[0]), bgp
        )
        vs[k] = np.where(
            w_k > 0, np.where(np.isfinite(bgs), w_k * res[1] + (1 - w_k) * bgs, res[1]), bgs
        )
        wgt[k] = w_k
    if background is not None:
        empty = wgt.sum(axis=(1, 2)) == 0
        for k in np.where(empty)[0]:
            bgp = _bg_vp(background, np.full((nlat, nlon), depth[k]))
            vp[k], vs[k] = bgp, bgp / c.vpvs
    n_uncov = int((wgt == 0).sum())
    if n_uncov:
        findings.append(
            GeologyFinding(
                "info",
                "uncovered_nodes",
                f"{n_uncov} of {wgt.size} nodes are not constrained by any well "
                f"(> {c.max_distance_km:g} km away or outside the logged depth range)",
                n_uncov,
            )
        )
    dims = ("depth", "lat", "lon")
    ds = xr.Dataset(
        {
            "vp": (dims, vp, {"units": "km/s"}),
            "vs": (dims, vs, {"units": "km/s"}),
            "weight": (dims, wgt, {"long_name": "well-constraint weight (1 = fully constrained)"}),
            "mask": (dims, wgt > 0, {"long_name": "node constrained by wells"}),
        },
        coords={
            "depth": ("depth", depth, {"units": "km", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"wells": [w.name for w in wells], "interp": c.interp, "upscale": c.upscale},
    )
    logs = pd.concat(
        [
            pd.DataFrame(
                {"well": w.name, "depth_km": w.depth_km, "vp": w.vp, "vs": w.vs, "rho": w.rho}
            )
            for w in wells
        ],
        ignore_index=True,
    )
    return WellsResult(ds, pd.DataFrame(rows), logs, findings, list(wells))


def _offset_xy(w: WellProfile, depth_km: float) -> tuple[float, float]:
    """Local-frame ``(x, y)`` km of the well at a depth (wellhead + deviation offset)."""
    order = np.argsort(w.depth_km)
    z = w.depth_km[order]
    e = float(np.interp(depth_km, z, w.east_km[order]))
    n = float(np.interp(depth_km, z, w.north_km[order]))
    return w.x_km + e, w.y_km + n


# ---------------------------------------------------------------------------------------
# geology + wells
# ---------------------------------------------------------------------------------------
def _gget(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in obj.items()}
    return obj.model_dump()


def _fit_law(kind: str, h: np.ndarray, v: np.ndarray, w: np.ndarray) -> dict[str, float]:
    if kind == "constant":
        return {"v0": float(np.average(v, weights=w))}
    a = np.column_stack([np.ones_like(h), h])
    sw = np.sqrt(w)
    if kind == "gradient":
        coef, *_ = np.linalg.lstsq(a * sw[:, None], v * sw, rcond=None)
        return {"v0": float(coef[0]), "k": float(coef[1])}
    if kind == "exponential":
        hmax = max(float(h.max()), 1e-3)

        def f(x: np.ndarray, v0: float, vmax: float, lam: float) -> np.ndarray:
            return vmax - (vmax - v0) * np.exp(-x / lam)

        p0 = (float(v[h <= np.percentile(h, 20)].mean()), float(v.max()), hmax / 2)
        popt, _ = curve_fit(
            f, h, v, p0=p0, sigma=1 / sw, bounds=([0.1, 0.1, 1e-3], [20.0, 20.0, 1e3]), maxfev=5000
        )
        return {"v0": float(popt[0]), "vmax": float(popt[1]), "lambda_km": float(popt[2])}
    raise WellsError(f"unknown velocity law {kind!r}")


def wells_horizons(
    horizons_cfg: Sequence[Any],
    grid: ModelGrid,
    wells: Sequence[WellProfile],
    *,
    surfaces: Mapping[str, np.ndarray] | None = None,
    surface_depth_km: np.ndarray | float = 0.0,
    base_dir: str | Path | None = None,
    fit_vs_ratio: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], pd.DataFrame, list[GeologyFinding]]:
    """Fit each unit's velocity law to the well-log samples inside the unit.

    The law type (``constant``/``gradient``/``exponential``) and ``reference`` stay as
    configured; ``v0`` (and ``k`` or ``vmax``/``lambda_km``) are replaced by least-squares
    fits (thickness-weighted) to the conditioned log samples between the unit's top and base
    at the (deviated) well position.  If shear logs exist, ``vpvs`` of the unit is the median
    log Vp/Vs (unless the unit has its own ``vs`` law).  Units with fewer than
    ``MIN_UNIT_SAMPLES`` samples keep their configured law (info finding).

    Returns
    -------
    (horizons, surfaces, fits, findings)
        Modified horizon dicts for :func:`~osv.model3d.geology.build_geology_model`, the
        gridded base surfaces (so they are not read twice), a table ``unit, n_samples, law,
        params, rms_km_s, r2, n_wells`` and findings.
    """
    findings: list[GeologyFinding] = []
    lat, lon = grid.lat_full, grid.lon_full
    names = [str(_gget(h, "name")) for h in horizons_cfg]
    bases: dict[str, np.ndarray] = {}
    for h, nm in zip(horizons_cfg, names, strict=True):
        if surfaces is not None and nm in surfaces:
            bases[nm] = np.asarray(surfaces[nm], float)
        else:
            bases[nm] = read_horizon(
                _resolve(Path(_gget(h, "surface")), base_dir),
                grid,
                crs=_gget(h, "crs"),
                z_convention=_gget(h, "z_convention", "depth"),
                units=_gget(h, "units", "m"),
            )
    surf = np.broadcast_to(np.asarray(surface_depth_km, float), (lat.size, lon.size))
    interps = {
        key: RegularGridInterpolator((lat, lon), arr, bounds_error=False, fill_value=np.nan)
        for key, arr in {**bases, "__surface__": surf}.items()
    }
    # per-well, per-sample base depths and surface depth
    sample_pos = []
    for w in wells:
        lo, la = _from_local_km(w.x_km + w.east_km, w.y_km + w.north_km, grid)
        sample_pos.append(np.column_stack([la, lo]))
    base_at = [{nm: interps[nm](p) for nm in names} for p in sample_pos]
    surf_at = [interps["__surface__"](p) for p in sample_pos]

    out: list[dict[str, Any]] = []
    rows = []
    prev = None
    for h, nm in zip(horizons_cfg, names, strict=True):
        hd = _as_dict(h)
        law = dict(hd["vp"])
        kind = law["law"]
        ref = law.get("reference", "unit_top")
        hs, vs_s, ws, n_w = [], [], [], 0
        vr = []
        for i, w in enumerate(wells):
            top = surf_at[i] if prev is None else base_at[i][prev]
            inside = (w.depth_km >= top) & (w.depth_km < base_at[i][nm]) & np.isfinite(w.vp)
            if not inside.any():
                continue
            n_w += 1
            origin = {"unit_top": top, "surface": surf_at[i], "sea_level": 0.0 * top}[ref]
            hs.append((w.depth_km - origin)[inside])
            vs_s.append(w.vp[inside])
            ws.append(np.maximum(w.thick_km[inside], 1e-9))
            vr.append((w.vp / w.vs)[inside]) if w.has_shear else None
        h_a = np.concatenate(hs) if hs else np.array([])
        n_s = h_a.size
        row: dict[str, Any] = {
            "unit": nm, "n_samples": n_s, "n_wells": n_w, "law": kind,
            "params": None, "rms_km_s": np.nan, "r2": np.nan,
        }  # fmt: skip
        if n_s < MIN_UNIT_SAMPLES:
            findings.append(
                GeologyFinding(
                    "info",
                    "unit_no_wells",
                    f"unit '{nm}': {n_s} well-log samples; configured velocity law kept",
                    n_s,
                )
            )
        else:
            v_a, w_a = np.concatenate(vs_s), np.concatenate(ws)
            h_a = np.maximum(h_a, 0.0)
            try:
                par = _fit_law(kind, h_a, v_a, w_a)
            except (RuntimeError, ValueError) as exc:
                findings.append(
                    GeologyFinding(
                        "warning", "unit_fit_failed", f"unit '{nm}': {kind} fit failed ({exc})"
                    )
                )
                par = None
            if par is not None:
                law.update(par)
                resid = v_a - apply_law(law, h_a)
                rms = float(np.sqrt(np.average(resid**2, weights=w_a)))
                var = float(np.average((v_a - np.average(v_a, weights=w_a)) ** 2, weights=w_a))
                r2 = 1.0 - rms**2 / var if var > 0 else np.nan
                row.update(params=par, rms_km_s=rms, r2=r2)
                lvl = "warning" if rms > UNIT_FIT_RMS_WARN else "info"
                findings.append(
                    GeologyFinding(
                        lvl,
                        "unit_fit",
                        f"unit '{nm}': {kind} law from {n_s} samples in {n_w} well(s), "
                        f"rms {rms:.3f} km/s, R2 {r2:.2f}",
                        n_s,
                    )
                )
                hd["vp"] = law
                if fit_vs_ratio and vr and hd.get("vs") is None:
                    hd["vpvs"] = max(float(np.median(np.concatenate(vr))), 1.01)
        out.append(hd)
        rows.append(row)
        prev = nm
    for f in findings:
        log.log(logging.WARNING if f.level == "warning" else logging.INFO, f.message)
    return out, bases, pd.DataFrame(rows), findings


def build_geology_wells_model(
    horizons_cfg: Sequence[Any],
    grid: ModelGrid,
    wells_cfg: Any,
    background: Background = None,
    *,
    wells: Sequence[WellProfile] | None = None,
    surfaces: Mapping[str, np.ndarray] | None = None,
    surface_depth_km: np.ndarray | float = 0.0,
    base_dir: str | Path | None = None,
) -> tuple[xr.Dataset, pd.DataFrame, list[GeologyFinding]]:
    """``geology+wells``: fit unit laws to the logs, then build the layer-cake model.

    Returns the geology dataset (see :func:`~osv.model3d.geology.build_geology_model`), the
    unit-fit table and the combined findings.
    """
    c = _cfg(wells_cfg)
    findings: list[GeologyFinding] = []
    if wells is None:
        wells, findings = load_wells(c, grid, base_dir=base_dir)
    hz, bases, fits, f_fit = wells_horizons(
        horizons_cfg,
        grid,
        wells,
        surfaces=surfaces,
        surface_depth_km=surface_depth_km,
        base_dir=base_dir,
    )
    ds, f_geo = build_geology_model(
        hz,
        grid,
        background,
        surfaces=bases,
        surface_depth_km=surface_depth_km,
        background_vpvs=c.vpvs,
        base_dir=base_dir,
    )
    ds.attrs["unit_fits"] = fits.to_json(orient="records")
    return ds, fits, findings + f_fit + f_geo
