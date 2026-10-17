"""Data misfit, model variance and model roughness, with FMTOMO's own definitions.

These are the three quantities of the trade-off (L-curve) plots of Stage 4: data variance against
model variance (damping, ``epsilon``) and against model roughness (smoothing, ``eta``).  Each one
replicates what the Fortran computes or minimises so the numbers agree with ``residuals.dat`` and
with the objective of ``invert3d``.

Data misfit (``fortran/fmtomo/source/residuals.f90``)
    A pair of observed/model times ``(rd1, rd2)`` with uncertainty ``eot`` is used when
    ``rd1 > 0 and rd2 > 0`` (``:139``); with ``n`` such pairs::

        rms      = 1000 * sqrt(sum (rd1-rd2)^2 / n)       [ms]        (:152)
        variance = sum (rd1-rd2)^2 / (n-1)                 [s^2]       (:153)
        chi2     = sum ((rd1-rd2)/eot)^2 / n               [-]         (:154)

    The printed ``chi2`` of ``residuals.dat`` is therefore already divided by ``n``;
    :func:`data_misfit` returns the plain sum as ``chi2`` and FMTOMO's number as
    ``chi2_reduced``.  (Teleseismic sources, ``residuals.f90:127-137``, are not supported.)

Model variance (``invert3d.f90``)
    The model vector holds every node of every velocity grid, cushions included
    (``:552-557``), and ``dm = m - m_ref`` (``:612``).  The damping term enters the subspace
    gradient as ``epsilon * dm * ecmi`` with ``ecmi = epsv / cm`` and ``cm = sigma**2`` where
    ``sigma`` is the second column of ``vgridsref.in`` (``:555-556``, ``:1138``).  I.e. the
    penalty is ``0.5 * epsilon * epsv * sum(dm**2 / sigma**2)``.  :func:`model_variance` is
    ``mean(dm**2)`` in km^2/s^2 (``prior_unc=None``) or ``mean(dm**2 / sigma**2)``
    (dimensionless) with ``prior_unc``.

Model roughness (``invert3d.f90:1354-1505``, subroutine ``smoothing``)
    ``D`` acts on ``dm`` and is the sum over the three axes of the 1-D second difference
    ``v[i-1] - 2 v[i] + v[i+1]`` *per node index* (no division by the node spacing), applied only at
    nodes that are not at the end of that axis (``:1371-1391``; zero there, but the other axes
    still contribute).  Grids are separate (``is2`` counter runs through one grid at a time), and
    the transposed operator is applied in ``:1425-1466``.  The gradient term is
    ``eta * etav * D^T D dm``, i.e. the penalty is ``0.5 * eta * etav * ||D dm||^2``.  We provide

    * ``index`` roughness: ``mean((D v)^2)`` in (km/s)^2, exactly FMTOMO's operator;
    * ``physical`` roughness: the same but each axis divided by its node spacing squared
      (km; horizontal spacings follow the sphere at the node's radius and latitude), so it is in
      (km/s / km^2)^2 and independent of the grid spacing.

    Roughness is of ``v`` unless a ``reference`` is given, in which case it is of ``v - v_ref``
    (the quantity ``invert3d`` actually smooths).

Averages are taken over the interior nodes only (cushion nodes excluded) by default, but the
operators are evaluated on the full grid so the cushions serve as neighbours.  The ``objective``
terms of :func:`model_metrics` are the sums over *all* nodes, as ``invert3d`` sees them.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.io.fmtomo import Invert3d, VelocityGrids, read_vgrids
from osv.io.fmtomo.grids import cushion_mask

if TYPE_CHECKING:
    from osv.fmtomo.invert import InversionResult

log = logging.getLogger(__name__)

__all__ = [
    "attach_metrics",
    "data_misfit",
    "iteration_metrics",
    "laplacian_index",
    "laplacian_physical",
    "model_metrics",
    "model_roughness",
    "model_variance",
]

#: default Earth radius for the physical spacing of horizontal axes (FMTOMO ``earthr``)
EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------------------
# data misfit
# ---------------------------------------------------------------------------------------


def _times(x: Any, what: str) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x["time"].to_numpy(float)
    if isinstance(x, pd.Series):
        return x.to_numpy(float)
    return np.asarray(x, dtype=float).ravel()


def data_misfit(
    otimes: Any,
    arrivals: Any,
    uncertainties: Any = None,
) -> dict[str, float]:
    """RMS, variance and chi-squared of observed against model times, as ``residuals`` does.

    Parameters
    ----------
    otimes
        Observed times: the table from :func:`osv.io.fmtomo.read_observed` (``time`` and
        ``uncertainty`` columns) or an array of seconds.
    arrivals
        Model times (``mtimes.dat``/``arrivals.dat`` table or array), same order.
    uncertainties
        Pick uncertainties (s); default the ``uncertainty`` column of ``otimes`` (else 1).

    Returns
    -------
    dict
        ``rms_s``, ``rms_ms`` (``residuals.f90:152``), ``variance_s2`` (``n-1``, ``:153``),
        ``chi2`` (plain sum of squared normalised residuals), ``chi2_reduced`` (sum / ``n``,
        the third number of ``residuals.dat``, ``:154``), ``n`` (pairs with both times > 0,
        ``:139``), ``mean_resid_s`` (observed - model).  NaN where undefined.
    """
    obs = _times(otimes, "otimes")
    mod = _times(arrivals, "arrivals")
    if obs.shape != mod.shape:
        raise ValueError(
            f"otimes has {obs.size} entries but arrivals {mod.size}: same order needed"
        )
    if uncertainties is None:
        if isinstance(otimes, pd.DataFrame) and "uncertainty" in otimes:
            unc = otimes["uncertainty"].to_numpy(float)
        else:
            unc = np.ones_like(obs)
    else:
        unc = np.broadcast_to(np.asarray(uncertainties, dtype=float), obs.shape)
    use = (obs > 0.0) & (mod > 0.0)
    n = int(use.sum())
    nan = math.nan
    if n == 0:
        return dict(
            rms_s=nan,
            rms_ms=nan,
            variance_s2=nan,
            chi2=nan,
            chi2_reduced=nan,
            n=0,
            mean_resid_s=nan,
        )
    r = obs[use] - mod[use]
    ss = float(np.sum(r**2))
    chi2 = float(np.sum((r / unc[use]) ** 2))
    return dict(
        rms_s=math.sqrt(ss / n),
        rms_ms=1000.0 * math.sqrt(ss / n),
        variance_s2=ss / (n - 1) if n > 1 else nan,
        chi2=chi2,
        chi2_reduced=chi2 / n,
        n=n,
        mean_resid_s=float(np.mean(r)),
    )


# ---------------------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------------------

_DIMS = ("depth", "lat", "lon")


def _datasets(model: Any) -> list[xr.Dataset]:
    """List of velocity datasets in file order from a path, VelocityGrids or Dataset(s)."""
    if isinstance(model, (str, Path)):
        model = read_vgrids(model)
    if isinstance(model, VelocityGrids):
        return [ds for _, _, ds in model]
    if isinstance(model, xr.Dataset):
        return [model]
    if isinstance(model, Iterable):
        return [d for d in model]
    raise TypeError(f"cannot interpret {type(model).__name__} as a velocity model")


def _vel(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["velocity"].transpose(*_DIMS).values, dtype=float)


def _include(ds: xr.Dataset, mask: Any) -> np.ndarray:
    """Boolean array (True = node is averaged over) in ``(depth, lat, lon)`` order."""
    shape = tuple(ds.sizes[d] for d in _DIMS)
    if mask is None:
        return ~np.asarray(cushion_mask(ds).transpose(*_DIMS).values, dtype=bool)
    if isinstance(mask, str):
        if mask == "all":
            return np.ones(shape, dtype=bool)
        raise ValueError("mask must be None (interior nodes), 'all' or a boolean array")
    if isinstance(mask, xr.DataArray):
        mask = mask.transpose(*_DIMS).values
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"mask has shape {m.shape}, grid has {shape}")
    return m


def _pair(a: Any, b: Any) -> list[tuple[xr.Dataset, xr.Dataset]]:
    da, db = _datasets(a), _datasets(b)
    if len(da) != len(db):
        raise ValueError(f"model has {len(da)} grids, reference {len(db)}")
    for x, y in zip(da, db, strict=True):
        if x["velocity"].shape != y["velocity"].shape:
            raise ValueError("model and reference grids differ in shape")
    return list(zip(da, db, strict=True))


def laplacian_index(f: np.ndarray) -> np.ndarray:
    """FMTOMO's discrete operator ``D`` (``invert3d.f90:1371-1391``) on a 3-D array.

    Sum over the three axes of ``f[i-1] - 2 f[i] + f[i+1]``, set to zero at the two end nodes of
    each axis (those nodes still receive the other axes' contributions).  Node-index units
    (no spacing).  For a 2-node-thick axis nothing contributes.
    """
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    for ax in range(f.ndim):
        if f.shape[ax] < 3:
            continue
        lo = [slice(None)] * f.ndim
        mid = list(lo)
        hi = list(lo)
        lo[ax], mid[ax], hi[ax] = slice(0, -2), slice(1, -1), slice(2, None)
        out[tuple(mid)] += f[tuple(lo)] - 2.0 * f[tuple(mid)] + f[tuple(hi)]
    return out


def _spacing_km(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node spacing ``(dz, dy, dx)`` in km, broadcast to the ``(depth, lat, lon)`` shape."""
    shape = tuple(ds.sizes[d] for d in _DIMS)
    depth = np.asarray(ds["depth"].values, float)
    lat = np.asarray(ds["lat"].values, float)
    lon = np.asarray(ds["lon"].values, float)
    big_r = float(ds.attrs.get("fmtomo_earth_radius_km", EARTH_RADIUS_KM))
    dz = float(np.mean(np.abs(np.diff(depth)))) if depth.size > 1 else 1.0
    dlat = float(ds.attrs.get("fmtomo_dlat_rad", np.deg2rad(np.mean(np.abs(np.diff(lat))))))
    dlon = float(ds.attrs.get("fmtomo_dlon_rad", np.deg2rad(np.mean(np.abs(np.diff(lon))))))
    r = (big_r - depth)[:, None, None]
    dy = np.broadcast_to(r * abs(dlat), shape)
    dx = np.broadcast_to(r * np.cos(np.deg2rad(lat))[None, :, None] * abs(dlon), shape)
    return np.broadcast_to(np.float64(dz), shape), dy, dx


def laplacian_physical(f: np.ndarray, ds: xr.Dataset) -> np.ndarray:
    """:func:`laplacian_index` with each axis divided by its squared node spacing in km.

    ``f`` is in ``(depth, lat, lon)`` order; the spacing comes from ``ds`` (depth step, and the
    latitude/longitude steps on the sphere at each node's radius and latitude).  Units:
    ``[f]/km^2``.  A linear field gives 0, ``f = z**2`` gives 2 in the interior.
    """
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    for ax, h in enumerate(_spacing_km(ds)):
        if f.shape[ax] < 3:
            continue
        lo = [slice(None)] * f.ndim
        mid = list(lo)
        hi = list(lo)
        lo[ax], mid[ax], hi[ax] = slice(0, -2), slice(1, -1), slice(2, None)
        out[tuple(mid)] += (f[tuple(lo)] - 2.0 * f[tuple(mid)] + f[tuple(hi)]) / h[tuple(mid)] ** 2
    return out


def _unc_array(ds: xr.Dataset, ref: xr.Dataset, prior_unc: Any) -> np.ndarray:
    shape = tuple(ds.sizes[d] for d in _DIMS)
    if isinstance(prior_unc, str):
        if prior_unc != "reference":
            raise ValueError("prior_unc must be None, 'reference', a number or an array")
        if "uncertainty" not in ref:
            raise ValueError("the reference grid has no 'uncertainty' (vgridsref.in column 2)")
        return np.asarray(ref["uncertainty"].transpose(*_DIMS).values, float)
    if isinstance(prior_unc, xr.DataArray):
        return np.asarray(prior_unc.transpose(*_DIMS).values, float)
    return np.broadcast_to(np.asarray(prior_unc, dtype=float), shape)


def model_variance(
    model: Any,
    reference: Any,
    prior_unc: Any = None,
    mask: Any = None,
) -> float:
    """Mean squared model perturbation ``mean((v - v_ref)**2)`` in km^2/s^2.

    With ``prior_unc`` the covariance-weighted version of ``invert3d``'s damping term,
    ``mean((v - v_ref)**2 / sigma**2)`` (dimensionless), is returned instead.  ``prior_unc`` is
    ``"reference"`` (the ``uncertainty`` of the reference grids, i.e. column 2 of
    ``vgridsref.in``; ``cm = sigma**2``, ``invert3d.f90:555``), a number or an array.

    Parameters
    ----------
    model, reference
        ``vgrids.in`` paths, :class:`~osv.io.fmtomo.VelocityGrids` or datasets.
    mask
        ``None``: interior nodes only (cushions excluded); ``"all"``; or a boolean array
        (True = include) matching the grid shape.

    Returns
    -------
    float
        Pooled over all grids; NaN when no node is selected.
    """
    total, count = 0.0, 0
    for ds, ref in _pair(model, reference):
        d2 = (_vel(ds) - _vel(ref)) ** 2
        if prior_unc is not None:
            d2 = d2 / _unc_array(ds, ref, prior_unc) ** 2
        m = _include(ds, mask)
        total += float(np.sum(d2[m]))
        count += int(m.sum())
    return total / count if count else math.nan


def model_roughness(
    model: Any,
    mask: Any = None,
    *,
    reference: Any = None,
    kind: str = "physical",
) -> float:
    """Mean squared second derivative (Laplacian) of the model.

    Parameters
    ----------
    model
        ``vgrids.in`` path, :class:`~osv.io.fmtomo.VelocityGrids` or dataset(s).
    mask
        As in :func:`model_variance` (selects the nodes averaged over; the operator itself is
        evaluated on the full grid).
    reference
        If given, the roughness of ``v - v_ref`` (what ``invert3d`` smooths) is returned.
    kind
        ``"physical"``: per km^2, result in (km/s/km^2)^2; ``"index"``: FMTOMO's per-node-index
        operator (``invert3d.f90:1354-1505``), result in (km/s)^2.

    Returns
    -------
    float
    """
    if kind not in ("physical", "index"):
        raise ValueError("kind must be 'physical' or 'index'")
    dss = _datasets(model)
    refs = _datasets(reference) if reference is not None else [None] * len(dss)
    if len(refs) != len(dss):
        raise ValueError("model and reference have different numbers of grids")
    total, count = 0.0, 0
    for ds, ref in zip(dss, refs, strict=True):
        f = _vel(ds) if ref is None else _vel(ds) - _vel(ref)
        lap = laplacian_index(f) if kind == "index" else laplacian_physical(f, ds)
        m = _include(ds, mask)
        total += float(np.sum(lap[m] ** 2))
        count += int(m.sum())
    return total / count if count else math.nan


def model_metrics(
    model: Any,
    reference: Any,
    *,
    start: Any = None,
    mask: Any = None,
    damping: float = 1.0,
    smoothing: float = 1.0,
) -> dict[str, float]:
    """All model-space metrics of one ``vgrids.in`` against the reference (and start) model.

    Keys: ``model_variance`` (km^2/s^2), ``model_variance_weighted`` (with the reference
    uncertainties, NaN if absent), ``roughness_index``/``roughness_phys`` (of ``v``),
    ``pert_roughness_index``/``pert_roughness_phys`` (of ``v - v_ref``), and, with ``start``,
    ``model_variance_start``, ``model_change_from_start_rms`` (km/s).  Averages use ``mask``.

    ``obj_damping = damping * sum(dm**2 / sigma**2)`` and ``obj_smoothing = smoothing *
    sum((D dm)**2)`` are the two regularisation sums over *all* nodes of ``0.5*epsilon*epsv*..``
    and ``0.5*eta*etav*..`` that ``invert3d`` adds to ``0.5*sum((dt/sigma_d)**2)``
    (``invert3d.f90:1138,1149``); ``damping`` is ``epsilon*epsv`` and ``smoothing`` is
    ``eta*etav``.  (The factor 0.5 is omitted.)
    """
    pairs = _pair(model, reference)
    try:
        wvar = model_variance(model, reference, "reference", mask)
    except ValueError:
        wvar = math.nan
    out: dict[str, float] = {
        "model_variance": model_variance(model, reference, None, mask),
        "model_variance_weighted": wvar,
        "roughness_index": model_roughness(model, mask, kind="index"),
        "roughness_phys": model_roughness(model, mask, kind="physical"),
        "pert_roughness_index": model_roughness(model, mask, reference=reference, kind="index"),
        "pert_roughness_phys": model_roughness(model, mask, reference=reference, kind="physical"),
    }
    dsum, ssum = 0.0, 0.0
    have_unc = True
    for ds, ref in pairs:
        dm = _vel(ds) - _vel(ref)
        if "uncertainty" in ref:
            dsum += float(np.sum(dm**2 / _unc_array(ds, ref, "reference") ** 2))
        else:
            have_unc = False
        ssum += float(np.sum(laplacian_index(dm) ** 2))
    out["obj_damping"] = damping * dsum if have_unc else math.nan
    out["obj_smoothing"] = smoothing * ssum
    if start is not None:
        out["model_variance_start"] = model_variance(model, start, None, mask)
        diff = np.concatenate(
            [(_vel(a) - _vel(b))[_include(a, mask)] for a, b in _pair(model, start)]
        )
        out["model_change_from_start_rms"] = (
            float(np.sqrt(np.mean(diff**2))) if diff.size else math.nan
        )
    return out


# ---------------------------------------------------------------------------------------
# per-iteration metrics of a run
# ---------------------------------------------------------------------------------------

_ITER_RE = re.compile(r"^\d+$")


def _regularisation(run: Path) -> tuple[float, float]:
    """``(epsilon*epsv, eta*etav)`` from ``invert3d.in`` (1, 1 if unreadable)."""
    try:
        c = Invert3d.read(run / "invert3d.in")
        _, epsv, etav = c.velocity_inversion
        damp = float(c.global_damping) * float(epsv)
        smooth = (
            float(c.global_smoothing) * float(etav) if int(c.second_derivative_smoothing) else 0.0
        )
        return damp, smooth
    except Exception as exc:  # noqa: BLE001 - metrics must not fail on a missing control file
        log.debug("cannot read invert3d.in in %s: %s", run, exc)
        return 1.0, 1.0


def iteration_metrics(run_dir: str | Path, mask: Any = None) -> pd.DataFrame:
    """Data and model metrics for every iteration snapshot of a run.

    Reads ``run_dir/iterations/NN/`` (``summary.json`` for the data misfit as computed by
    ``residuals``, ``vgrids.in`` when the run kept snapshots) against ``run_dir/vgridsref.in``
    and the starting model (iteration 0).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``iter``; columns ``rms_s``, ``variance_s2``, ``chi2`` (as in
        ``residuals.dat``, the last one divided by ``n``), then the :func:`model_metrics` keys
        (NaN for iterations without a snapshot).
    """
    run = Path(run_dir)
    idir = run / "iterations"
    dirs = sorted((int(p.name), p) for p in idir.iterdir() if p.is_dir() and _ITER_RE.match(p.name))
    if not dirs:
        raise FileNotFoundError(f"{idir} has no iteration directories")
    ref_path = run / "vgridsref.in"
    reference = read_vgrids(ref_path) if ref_path.exists() else None
    start_path = dirs[0][1] / "vgrids.in"
    start = read_vgrids(start_path) if start_path.exists() else reference
    damping, smoothing = _regularisation(run)
    rows: list[dict[str, float]] = []
    for it, d in dirs:
        row: dict[str, float] = {"iter": it}
        sj = d / "summary.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            if s.get("rms_ms") is not None:
                row["rms_s"] = float(s["rms_ms"]) / 1000.0
            row["variance_s2"] = _f(s.get("variance_s2"))
            row["chi2"] = _f(s.get("chi2"))
            row["n_valid"] = _f(s.get("n_valid"))
        vg = d / "vgrids.in"
        if vg.exists() and reference is not None:
            row.update(
                model_metrics(
                    read_vgrids(vg),
                    reference,
                    start=start,
                    mask=mask,
                    damping=damping,
                    smoothing=smoothing,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("iter")


def _f(x: object) -> float:
    return math.nan if x is None else float(x)  # type: ignore[arg-type]


def attach_metrics(result: InversionResult, mask: Any = None) -> pd.DataFrame:
    """Add the :func:`iteration_metrics` columns to ``result.iterations`` (in place).

    Columns already present in the table (``variance_s2``, ``chi2``, ...) are kept as they are;
    the new ones are joined on ``iter``.  Returns the updated table.
    """
    met = iteration_metrics(result.run_dir, mask)
    new = met[[c for c in met.columns if c not in result.iterations.columns]]
    table = result.iterations.drop(columns=list(new.columns), errors="ignore")
    joined = table.join(new, on="iter")
    result.iterations = joined
    return joined
