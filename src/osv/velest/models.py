"""Monte Carlo trial 1-D velocity models for VELEST (Stage 2).

Sampling
--------
Each trial model is one point in a unit hypercube drawn with a scrambled, seeded
:mod:`scipy.stats.qmc` Latin hypercube or Sobol sequence, or with plain
``numpy`` uniform random numbers.  Dimensions: one per layer for Vp, plus one for
the Vp/Vs ratio (``vpvs.mode == "sampled"``: **one ratio per model**, so Vs stays
monotonic whenever Vp is) or one per layer for Vs (``"independent_s_model"``).

Monotonic option
----------------
With ``monotonic=True`` the velocity must be non-decreasing with depth while still
covering every layer's bounds.  Method (sequential conditional sampling):

1. Tighten the bounds: ``lo_i' = max(lo_j, j <= i)`` and ``hi_i' = min(hi_j, j >= i)``.
   Any monotonic model satisfies these, and both sequences are non-decreasing.  If
   ``lo_i' > hi_i'`` for some layer, no monotonic model exists and
   :class:`InfeasibleBoundsError` is raised.
2. Layer by layer, ``v_i = L_i + u_i (hi_i' - L_i)`` with ``L_i = max(lo_i', v_{i-1})``.
   Feasibility is guaranteed since ``v_{i-1} <= hi_{i-1}' <= hi_i'``.

Every model is valid, no rejection is needed, the Latin-hypercube stratification of
the unit-cube coordinates is kept, and each layer's marginal spans ``[lo_i', hi_i']``.
The price is that deeper layers are biased towards the upper part of their range
when the bounds overlap strongly (the sorted-order effect); when bounds do not
overlap the method equals independent uniform sampling.  The non-monotonic mode samples
each layer independently, which allows low-velocity zones.

Output precision
----------------
VELEST model files hold velocities with two decimals (``f5.2``) and depths with
two decimals (``f7.2``): :func:`write_trial_models` rounds to that precision.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from osv.config.schema import VelestConfig
from osv.io.velest import ModelBlock, VelestModel, write_model

__all__ = [
    "InfeasibleBoundsError",
    "TrialModel",
    "generate_trial_models",
    "models_to_frame",
    "save_trial_models_parquet",
    "write_trial_models",
]

logger = logging.getLogger(__name__)


class InfeasibleBoundsError(ValueError):
    """The velocity bounds admit no model satisfying the requested constraints."""


@dataclass(eq=False)
class TrialModel:
    """One trial 1-D model.  Depths are layer tops (km, positive down, may be negative)."""

    id: str
    depths: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    sampler: str
    seed: int
    index: int = 0
    is_reference: bool = False
    vpvs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.depths = np.asarray(self.depths, dtype=float)
        self.vp = np.asarray(self.vp, dtype=float)
        self.vs = np.asarray(self.vs, dtype=float)
        self.vpvs = self.vp / self.vs

    @property
    def n_layers(self) -> int:
        return len(self.depths)


def _effective_bounds(bounds: np.ndarray, monotonic: bool, what: str) -> np.ndarray:
    lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
    if monotonic:
        lo = np.maximum.accumulate(lo)
        hi = np.minimum.accumulate(hi[::-1])[::-1]
        bad = np.flatnonzero(lo > hi)
        if bad.size:
            i = int(bad[0])
            raise InfeasibleBoundsError(
                f"{what}: no monotonic (non-decreasing) model fits the bounds; layer {i} "
                f"needs at least {lo[i]:.3f} (a shallower layer's lower bound) but at most "
                f"{hi[i]:.3f} (a deeper layer's upper bound). Widen the bounds or set "
                "velest.monotonic: false."
            )
    return np.column_stack([lo, hi])


def _map_unit(u: np.ndarray, bounds: np.ndarray, monotonic: bool, what: str) -> np.ndarray:
    """Map unit-cube samples ``(n, L)`` to velocities (see module docstring)."""
    eff = _effective_bounds(bounds, monotonic, what)
    lo, hi = eff[:, 0], eff[:, 1]
    if not monotonic:
        return lo + u * (hi - lo)
    v = np.empty_like(u)
    prev = np.full(u.shape[0], -np.inf)
    for i in range(u.shape[1]):
        low = np.maximum(lo[i], prev)
        v[:, i] = low + u[:, i] * (hi[i] - low)
        prev = v[:, i]
    return v


def _unit_samples(sampler: str, n: int, d: int, seed: int) -> np.ndarray:
    if sampler == "uniform":
        return np.random.default_rng(seed).random((n, d))
    if sampler == "latin_hypercube":
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    if sampler == "sobol":
        with warnings.catch_warnings():  # balance warning for n not a power of 2
            warnings.simplefilter("ignore", UserWarning)
            return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)
    raise ValueError(f"unknown sampler {sampler!r}")


def _s_bounds(cfg: VelestConfig, vp_bounds: np.ndarray) -> np.ndarray:
    if cfg.vpvs.s_bounds is not None:
        return np.asarray(cfg.vpvs.s_bounds, dtype=float)
    r_lo, r_hi = cfg.vpvs.range
    return np.column_stack([vp_bounds[:, 0] / r_hi, vp_bounds[:, 1] / r_lo])


def generate_trial_models(cfg: VelestConfig, seed: int) -> list[TrialModel]:
    """Generate the Monte Carlo trial models.

    Parameters
    ----------
    cfg
        The ``velest`` config section (``layers_top_km``, ``vp_bounds``, ``vpvs``,
        ``monotonic``, ``n_models``, ``sampler``, ``include_reference_model``).
    seed
        Seed (normally ``project.random_seed``); the same seed gives identical models.

    Returns
    -------
    list of TrialModel
        ``MC_0001 .. MC_nnnn`` and, if ``include_reference_model``, ``MC_0000`` first
        (mid-points of the (monotonically tightened) bounds, Vp/Vs at the fixed value or
        the middle of the range).

    Raises
    ------
    InfeasibleBoundsError
        If ``monotonic`` is set and the bounds cannot be satisfied.
    """
    depths = np.asarray(cfg.layers_top_km, dtype=float)
    nl = len(depths)
    vp_b = np.asarray(cfg.vp_bounds, dtype=float)
    mode = cfg.vpvs.mode
    n = cfg.n_models
    # Validate feasibility before sampling (also for the reference model).
    vp_eff = _effective_bounds(vp_b, cfg.monotonic, "vp_bounds")
    s_b = _s_bounds(cfg, vp_b) if mode == "independent_s_model" else None
    if s_b is not None:
        s_eff = _effective_bounds(s_b, cfg.monotonic, "vpvs.s_bounds")

    extra = {"fixed": 0, "sampled": 1, "independent_s_model": nl}[mode]
    u = _unit_samples(cfg.sampler, n, nl + extra, seed)
    vp = _map_unit(u[:, :nl], vp_b, cfg.monotonic, "vp_bounds")
    if mode == "fixed":
        vs = vp / cfg.vpvs.value
    elif mode == "sampled":
        r_lo, r_hi = cfg.vpvs.range
        vs = vp / (r_lo + u[:, nl : nl + 1] * (r_hi - r_lo))
    else:
        assert s_b is not None
        vs = _map_unit(u[:, nl:], s_b, cfg.monotonic, "vpvs.s_bounds")

    models: list[TrialModel] = []
    if cfg.include_reference_model:
        vp0 = vp_eff.mean(axis=1)
        if mode == "fixed":
            vs0 = vp0 / cfg.vpvs.value
        elif mode == "sampled":
            vs0 = vp0 / float(np.mean(cfg.vpvs.range))
        else:
            vs0 = s_eff.mean(axis=1)
        models.append(TrialModel("MC_0000", depths, vp0, vs0, cfg.sampler, seed, 0, True))
    for k in range(n):
        models.append(TrialModel(f"MC_{k + 1:04d}", depths, vp[k], vs[k], cfg.sampler, seed, k + 1))
    logger.info("generated %d trial models (%s, seed %d)", len(models), cfg.sampler, seed)
    return models


def models_to_frame(models: list[TrialModel]) -> pd.DataFrame:
    """Long table with one row per model and layer.

    Columns: ``id``, ``index``, ``is_reference``, ``sampler``, ``seed``, ``layer``,
    ``depth_km``, ``vp``, ``vs``, ``vpvs``.
    """
    rows = [
        pd.DataFrame(
            {
                "id": m.id,
                "index": m.index,
                "is_reference": m.is_reference,
                "sampler": m.sampler,
                "seed": m.seed,
                "layer": np.arange(m.n_layers),
                "depth_km": m.depths,
                "vp": m.vp,
                "vs": m.vs,
                "vpvs": m.vpvs,
            }
        )
        for m in models
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_trial_models_parquet(models: list[TrialModel], path: str | Path) -> Path:
    """Save :func:`models_to_frame` as Parquet (``trial_models.parquet``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    models_to_frame(models).to_parquet(p, index=False)
    return p


def _vdamp_array(vdamp: float | list[float] | np.ndarray, n: int) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(vdamp, dtype=float), (n,)).copy()
    return arr


def write_trial_models(
    models: list[TrialModel],
    outdir: str | Path,
    vdamp: float | list[float] | np.ndarray = 1.0,
    *,
    parquet: bool = True,
    nmod: int = 2,
) -> list[Path]:
    """Write one VELEST ``.mod`` file per model into ``outdir``.

    ``nmod = 2`` (``nsp = 2``) writes P and S blocks; ``nmod = 1`` (``nsp = 1``/``3``, S
    from ``vp / vpvs``) writes only the P block, which is all VELEST reads then.

    Files are named ``<id>.mod``.  ``vdamp`` is a scalar or one value per layer.
    Values are rounded to the file precision (0.01 km/s, 0.01 km).  With ``parquet``
    also writes ``trial_models.parquet`` in ``outdir``.
    """
    out = Path(outdir)
    paths: list[Path] = []
    for m in models:
        vd = _vdamp_array(vdamp, m.n_layers)
        blocks = []
        for label, vel in (("P", m.vp), ("S", m.vs)):
            if label == "S" and nmod != 2:
                blocks.append(None)
                continue
            lay = pd.DataFrame({"vel": vel, "depth": m.depths, "vdamp": vd})
            blocks.append(ModelBlock(lay, label=f"{label} {m.id}"))
        title = f"{m.id} {m.sampler}"[:40]
        s_block = blocks[1] if nmod == 2 else None
        paths.append(write_model(out / f"{m.id}.mod", VelestModel(title, blocks[0], s_block)))
    if parquet:
        save_trial_models_parquet(models, out / "trial_models.parquet")
    return paths
