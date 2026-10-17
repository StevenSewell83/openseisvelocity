"""Vp/Vs handling for VELEST runs (Stage 2).

``velest.f`` has three ``nsp`` settings (``velest.f:1121-1123``):

* ``nsp = 1``: P model and **P data only**; S readings are dropped (``velest.f:1892-1894``).
* ``nsp = 2``: independent S model (``nmod = 2``, second block in the model file).
* ``nsp = 3``: only the P model (``nmod = 1``); S readings are used with the S velocity
  ``vp / vpvs`` in every layer (``velest.f:4357``) and S station delays ``ptcor * vpvs``.

The osv modes map to ``nsp = 3`` (``fixed``, ``sampled``) or ``nsp = 2``
(``independent_s_model``); see :func:`nsp_for_mode`.  :func:`vpvs_grid_search` picks the
best fixed ratio from a small grid of VELEST runs.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.io.velest import VelestControl, VelestModel, write_model
from osv.velest.models import TrialModel
from osv.velest.runner import VelestJob, run_jobs

__all__ = [
    "NSP_FOR_MODE",
    "apply_vpvs_mode",
    "model_vpvs",
    "nsp_for_mode",
    "vpvs_grid_search",
]

logger = logging.getLogger(__name__)

#: ``velest.vpvs.mode`` -> VELEST ``nsp``.
NSP_FOR_MODE = {"fixed": 3, "sampled": 3, "independent_s_model": 2}

#: Decimals of the ratio written to ``velest.cmn`` (VELEST echoes it with ``f6.3``).
VPVS_DECIMALS = 3


def nsp_for_mode(mode: str) -> int:
    """VELEST ``nsp`` for a ``velest.vpvs.mode``."""
    try:
        return NSP_FOR_MODE[mode]
    except KeyError:
        raise ValueError(f"unknown vpvs mode {mode!r}; use one of {sorted(NSP_FOR_MODE)}") from None


def model_vpvs(model: TrialModel) -> float:
    """The single Vp/Vs ratio of a model (mean over layers, rounded to the file precision)."""
    return round(float(np.mean(model.vpvs)), VPVS_DECIMALS)


def apply_vpvs_mode(
    control: VelestControl, mode: str, *, value: float | None = None
) -> VelestControl:
    """Copy of ``control`` with ``nsp``/``nmod``/``vpvs`` set for ``mode``.

    ``value`` is the ratio for ``nsp = 3`` (default: keep ``control.vpvs``).
    """
    c = copy.deepcopy(control)
    c.nsp = nsp_for_mode(mode)
    c.nmod = 2 if c.nsp == 2 else 1
    if value is not None and c.nsp == 3:
        c.vpvs = float(value)
    return c


def vpvs_grid_search(
    model: TrialModel | VelestModel | Path | str,
    inputs: Any,
    root: str | Path,
    grid: Sequence[float] | None = None,
    *,
    ittmax: int | None = None,
    n_jobs: int = -1,
    timeout: float | None = 3600.0,
    force: bool = False,
) -> pd.DataFrame:
    """Run VELEST once per fixed ``vpvs`` (``nsp = 3``) and rank the ratios by final RMS.

    The data (P and S) and weights are identical in every run, so the RMS of the
    weighted P+S residuals is comparable between ratios.  The Vp model is the same
    starting model for all runs; with ``ittmax`` high enough that all runs converge the
    minimum is a data-driven estimate to compare with the Wadati ratio.

    Caveat (real data, 8-layer model, 2500 P+S readings): the RMS curve is very flat
    (0.072-0.077 s between 1.60 and 1.84) and VELEST rarely stops before ``ittmax``, so the
    run-to-run jitter (~0.002 s) is as large as the curve's curvature; the minimum was
    1.72, 1.64 and 1.72 for the three best models.  Treat it as a consistency check of the
    Wadati ratio, not as a replacement.

    Parameters
    ----------
    model
        Starting model (Vp used; the S block, if any, is ignored).
    inputs
        ``cnv_path``, ``sta_path`` and ``control`` (e.g. ``VelestInputs``).
    grid
        Ratios to test (default 1.66 .. 1.80 step 0.02).
    ittmax
        Iterations per run (default: the control's).

    Returns
    -------
    pandas.DataFrame
        ``vpvs, id, status, final_rms, converged, n_iterations`` sorted by ``vpvs``; the
        ``attrs["best_vpvs"]`` entry is the ratio with the minimum RMS and
        ``attrs["best_vpvs_parabola"]`` the vertex of a parabola through the three
        lowest-RMS neighbours (NaN at the grid edge).
    """
    ratios = [
        round(float(r), VPVS_DECIMALS)
        for r in (np.arange(1.66, 1.801, 0.02) if grid is None else grid)
    ]
    if isinstance(model, VelestModel):  # P block only (nsp = 3 reads no S block)
        model = write_model(Path(root) / "_inputs" / "start.mod", VelestModel(model.title, model.p))
    jobs = []
    for r in ratios:
        ctrl = apply_vpvs_mode(inputs.control, "fixed", value=r)
        if ittmax is not None:
            ctrl.ittmax = int(ittmax)
        jobs.append(VelestJob(f"vpvs_{r:.3f}", model, ctrl, inputs.cnv_path, inputs.sta_path))
    df = run_jobs(jobs, root, n_jobs=n_jobs, timeout=timeout, force=force, progress=False)
    rows = []
    for r, row in zip(ratios, df.itertuples(), strict=True):
        p = getattr(row, "parsed", None)
        rows.append(
            {
                "vpvs": r,
                "id": row.id,
                "status": row.status,
                "final_rms": float(row.final_rms) if np.isfinite(row.final_rms) else math.nan,
                "converged": bool(getattr(p, "converged", False)) if p is not None else False,
                "n_iterations": int(getattr(p, "n_iterations", 0)) if p is not None else 0,
            }
        )
    out = pd.DataFrame(rows).sort_values("vpvs", ignore_index=True)
    good = out[np.isfinite(out["final_rms"])]
    best = math.nan
    vertex = math.nan
    if len(good):
        i = int(good["final_rms"].idxmin())
        best = float(out.loc[i, "vpvs"])
        if 0 < i < len(out) - 1 and np.isfinite(out.loc[[i - 1, i, i + 1], "final_rms"]).all():
            x = out.loc[i - 1 : i + 1, "vpvs"].to_numpy(dtype=float)
            y = out.loc[i - 1 : i + 1, "final_rms"].to_numpy(dtype=float)
            a, b, _ = np.polyfit(x, y, 2)
            if a > 0:
                vertex = float(-b / (2 * a))
    out.attrs["best_vpvs"] = best
    out.attrs["best_vpvs_parabola"] = vertex
    logger.info("vpvs grid search: best %.3f (parabola %.3f)", best, vertex)
    return out
