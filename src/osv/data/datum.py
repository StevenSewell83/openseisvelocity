"""Convert catalogue depths to the canonical depth below sea level."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from osv.coords import DepthDatum, to_sea_level_depth

__all__ = ["apply_depth_datum"]

logger = logging.getLogger(__name__)

SurfaceElevation = ArrayLike | Callable[[np.ndarray, np.ndarray], ArrayLike]


def apply_depth_datum(
    events: pd.DataFrame,
    datum: DepthDatum = "sea_level",
    reference_elevation_m: float | None = None,
    surface_elevation_m: SurfaceElevation | None = None,
) -> pd.DataFrame:
    """Return a copy of ``events`` with ``depth_km_bsl`` (km below sea level, positive down).

    Parameters
    ----------
    events
        Events table with ``depth_km`` (relative to ``datum``) and, for ``"surface"`` with a
        callable, ``lat`` and ``lon``.
    datum, reference_elevation_m
        As in ``coordinates.depth_datum`` / ``coordinates.reference_elevation_m``.
    surface_elevation_m
        For ``"surface"``: an array (one value per event, metres above sea level) or a
        callable ``f(lat, lon) -> elevation_m``.

    Notes
    -----
    ``df.attrs["depth_datum"]`` records the *source* datum and
    ``df.attrs["depth_km_bsl_datum"] = "sea_level"`` the datum of the new column.
    """
    out = events.copy()
    surf = None
    if datum == "surface":
        if surface_elevation_m is None:
            raise ValueError(
                "depth_datum 'surface' needs surface elevations at the epicentres: pass an "
                "array or a callable f(lat, lon) (station interpolation or DEM)"
            )
        if callable(surface_elevation_m):
            surf = np.asarray(
                surface_elevation_m(out["lat"].to_numpy(), out["lon"].to_numpy()),
                dtype=float,
            )
        else:
            surf = np.asarray(surface_elevation_m, dtype=float)
        if surf.ndim > 0 and surf.shape != (len(out),):
            raise ValueError(f"surface_elevation_m has shape {surf.shape}, expected ({len(out)},)")
    out["depth_km_bsl"] = np.asarray(
        to_sea_level_depth(
            out["depth_km"].to_numpy(dtype=float),
            datum,
            reference_elevation_m=reference_elevation_m,
            surface_elevation_m=surf,
        ),
        dtype=float,
    )
    out.attrs["depth_datum"] = datum
    out.attrs["depth_reference_elevation_m"] = reference_elevation_m
    out.attrs["depth_km_bsl_datum"] = "sea_level"
    n_neg = int((out["depth_km_bsl"] < 0).sum())
    if n_neg:
        logger.info("%d events are above sea level after applying datum %r", n_neg, datum)
    return out
