"""Synthetic data for resolution tests: real geometry, pick-based noise.

:func:`make_synthetic_data` forward models a known (true) velocity model with ``fm3d`` for exactly
the source-receiver pairs of the real data set and writes a synthetic ``otimes.dat``:

``time = t_calc(true model, true hypocentres) + noise``

* ``noise ~ N(0, (scale * sigma)^2)`` with ``sigma`` the real pick uncertainty (``otimes.dat``
  column 6) when ``resolution.noise.from_pick_uncertainty`` else the constant
  ``resolution.noise.sigma_s``; the uncertainty column itself is kept, so the inversion weights
  equal those of the real inversion;
* observations that are invalid in the real file (time below -50) stay invalid, as do those for
  which ``fm3d`` found no ray (time <= 0); they are written with the invalid marker -100;
* the RNG is seeded (``numpy.random.default_rng``); noise and hypocentre perturbations use
  independent child streams, so changing one setting never changes the other realisation.

The synthetic times are computed from the TRUE hypocentres.  For the subsequent inversion the
starting hypocentres can be perturbed (``resolution.hypocentre_noise_km``) to simulate location
error; the perturbed ``sources.in``/``sourcesref.in`` are written next to the synthetic times.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.fmtomo.forward import run_fm3d_parallel
from osv.io.fmtomo import (
    InterfaceGrids,
    ModeSet,
    PropGrid,
    Receivers,
    Source,
    read_arrivals,
    read_interfaces,
    read_observed,
    read_propgrid,
    read_receivers,
    read_sources,
    write_interfaces,
    write_observed,
    write_propgrid,
    write_sources,
)
from osv.io.fmtomo import (
    write_vgrids as write_fmtomo_vgrids,
)
from osv.io.fmtomo.times import INVALID_OBSERVED_BELOW
from osv.model3d.grid import ModelGrid
from osv.resolution.models import write_vgrids

log = logging.getLogger(__name__)

__all__ = [
    "BaseInputs",
    "SyntheticData",
    "forward_times",
    "make_synthetic_data",
    "perturb_sources",
    "synthetic_observed",
]

#: marker for invalid observed times (``synthdata.f90:102``)
INVALID_TIME = -100.0
_KEY = ["receiver", "source", "ray", "subray"]
_KM_PER_DEG = 6371.0 * np.pi / 180.0


@dataclass
class BaseInputs:
    """Real-data geometry and uncertainties used for the synthetic data set.

    Each item is an object of :mod:`osv.io.fmtomo` or a path to the file.  ``observed`` is the
    real ``otimes.dat`` (rows in ``arrivals.dat`` order; only its keys, validity and uncertainty
    column are used).
    """

    interfaces: InterfaceGrids | str | Path
    propgrid: PropGrid | str | Path
    sources: list[Source] | str | Path
    receivers: Receivers | str | Path
    observed: pd.DataFrame | str | Path
    mode_set: ModeSet | None = None

    def load(self) -> tuple[InterfaceGrids, PropGrid, list[Source], Receivers, pd.DataFrame]:
        itf = self.interfaces
        if isinstance(itf, (str, Path)):
            itf = read_interfaces(itf)
        pg = self.propgrid
        if isinstance(pg, (str, Path)):
            pg = read_propgrid(pg)
        src = self.sources
        if isinstance(src, (str, Path)):
            src = read_sources(src)
        rec = self.receivers
        if isinstance(rec, (str, Path)):
            rec = read_receivers(rec)
        obs = self.observed
        if isinstance(obs, (str, Path)):
            obs = read_observed(obs)
        return itf, pg, list(src), rec, obs


@dataclass
class SyntheticData:
    """Result of :func:`make_synthetic_data`.

    ``otimes`` is the synthetic ``otimes.dat``; ``sources``/``sourcesref`` are the starting
    hypocentres for the inversion (true or perturbed); ``true_sources`` the true ones.
    ``stats`` holds the noise realisation and signal statistics, see :meth:`summary`.
    """

    workdir: Path
    otimes: Path
    otimes_noiseless: Path
    sources: Path
    sourcesref: Path
    true_sources: Path
    true_model: Path
    arrivals_true: Path
    observed: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def summary(self) -> dict[str, Any]:
        """Noise and signal statistics (seconds): noise rms, signal rms, signal-to-noise."""
        return dict(self.stats)


# ---------------------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------------------
def _res_cfg(cfg: Any) -> Any:
    return getattr(cfg, "resolution", cfg)


def forward_times(
    model: xr.Dataset,
    grid: ModelGrid,
    base: BaseInputs,
    workdir: str | Path,
    *,
    sources: list[Source] | None = None,
    n_jobs: int = 1,
    phase: str = "P",
) -> pd.DataFrame:
    """Run ``fm3d`` on ``model`` for the real geometry; returns the ``arrivals.dat`` table.

    ``sources`` override the source positions of ``base`` (same count and paths).
    """
    work = Path(workdir)
    inp = work / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    itf, pg, src, rec, _ = base.load()
    with_s = phase.upper() == "S" and "vs" in model
    write_fmtomo_vgrids(inp / "vgrids.in", write_vgrids(grid, model, with_s=with_s))
    write_interfaces(inp / "interfaces.in", itf)
    write_propgrid(inp / "propgrid.in", pg)
    (base.mode_set or ModeSet()).write(inp / "mode_set.in")
    (inp / "frechet.in").write_text("0\n")  # travel times only
    res = run_fm3d_parallel(inp, sources if sources is not None else src, rec, n_jobs, workdir=work)
    return res.arrivals()


def synthetic_observed(
    observed: pd.DataFrame,
    calc: pd.DataFrame,
    *,
    scale: float,
    from_pick_uncertainty: bool,
    sigma_s: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic ``otimes`` table (calc + noise) and the noiseless one, same row order as real.

    Returns ``(noisy, noiseless)``; both have the ``otimes`` columns plus ``valid`` and, for
    ``noisy``, the drawn ``noise`` (s; 0 for invalid rows).
    """
    obs = observed.reset_index(drop=True)
    c = calc[[*_KEY, "time"]].rename(columns={"time": "calc"})
    m = obs[_KEY].merge(c, on=_KEY, how="left", validate="one_to_one")
    calc_t = m["calc"].to_numpy(float)
    real_valid = obs["time"].to_numpy(float) >= INVALID_OBSERVED_BELOW
    found = np.isfinite(calc_t) & (calc_t > 0.0)
    valid = real_valid & found
    unc = obs["uncertainty"].to_numpy(float)
    if from_pick_uncertainty:
        sigma = np.where(unc > 0, unc, sigma_s)
    else:
        sigma = np.full(len(obs), float(sigma_s))
    # draw for every row so the stream does not depend on which rows are valid
    noise = rng.normal(0.0, 1.0, len(obs)) * scale * sigma
    noise = np.where(valid, noise, 0.0)
    free = obs.copy()
    clean = free.copy()
    clean["time"] = np.where(valid, calc_t, INVALID_TIME)
    free["time"] = np.where(valid, calc_t + noise, INVALID_TIME)
    free["noise"] = noise
    free["sigma"] = sigma * scale
    free["valid"] = valid
    clean["valid"] = valid
    return free, clean


def perturb_sources(
    sources: list[Source],
    sigma_xy_km: float,
    sigma_z_km: float,
    rng: np.random.Generator,
    grid: ModelGrid | None = None,
) -> tuple[list[Source], np.ndarray]:
    """Gaussian location error: ``sigma_xy_km`` per horizontal coordinate, ``sigma_z_km`` in depth.

    Positions are clipped to the interior of ``grid`` if given.  Returns the new sources and an
    ``(n, 3)`` array of the applied ``(dx_east, dy_north, dz_down)`` shifts in km after clipping.
    """
    n = len(sources)
    d = rng.normal(0.0, 1.0, (n, 3)) * np.array([sigma_xy_km, sigma_xy_km, sigma_z_km])
    out: list[Source] = []
    shifts = np.zeros((n, 3))
    for i, s in enumerate(sources):
        lat = s.lat + d[i, 1] / _KM_PER_DEG
        lon = s.lon + d[i, 0] / (_KM_PER_DEG * np.cos(np.deg2rad(s.lat)))
        depth = s.depth_km + d[i, 2]
        if grid is not None:
            lat = float(np.clip(lat, grid.lat[0], grid.lat[-1]))
            lon = float(np.clip(lon, grid.lon[0], grid.lon[-1]))
            depth = float(np.clip(depth, grid.depth[0], grid.depth[-1]))
        shifts[i] = [
            (lon - s.lon) * _KM_PER_DEG * np.cos(np.deg2rad(s.lat)),
            (lat - s.lat) * _KM_PER_DEG,
            depth - s.depth_km,
        ]
        out.append(replace(s, lat=float(lat), lon=float(lon), depth_km=float(depth)))
    return out, shifts


def _rms(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a**2))) if a.size else float("nan")


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def make_synthetic_data(
    true_model: xr.Dataset,
    grid: ModelGrid,
    base_inputs: BaseInputs,
    workdir: str | Path,
    cfg: Any,
    seed: int = 42,
    n_jobs: int = 1,
    phase: str = "P",
    perturb_hypocentres: bool | None = None,
    *,
    background_model: xr.Dataset | None = None,
    background_arrivals: pd.DataFrame | str | Path | None = None,
) -> SyntheticData:
    """Forward model ``true_model`` with the real geometry and add pick-based noise.

    Parameters
    ----------
    true_model
        Dataset with ``vp`` (and ``vs``) on ``grid`` (see :mod:`osv.resolution.models`).
    grid
        The model grid (:meth:`ModelGrid.to_vgrids` writes ``vgrids.in``).
    base_inputs
        Real interfaces, propagation grid, sources (true hypocentres), receivers (observed pairs
        only) and ``otimes`` (keys, validity and uncertainties).
    workdir
        Output directory (created).
    cfg
        :class:`~osv.config.schema.ResolutionConfig` (or a config with a ``resolution`` member):
        ``noise.{from_pick_uncertainty, scale, sigma_s}`` and ``hypocentre_noise_km.{xy, z}``.
    seed
        Seed of the random number generator (use ``cfg.project.random_seed``).
    n_jobs
        Parallel ``fm3d`` processes.
    phase
        ``"P"`` or ``"S"``; selects the output name ``otimes_<phase>.dat`` (S writes Vs too).
    perturb_hypocentres
        ``None``: perturb iff ``hypocentre_noise_km`` is non-zero; ``True``/``False`` force it.
    background_model, background_arrivals
        Background (unperturbed) model or its already computed arrivals, to report the signal
        ``calc_true - calc_background``.  Without either the signal statistics are NaN.
    """
    rc = _res_cfg(cfg)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    _, _, sources, _, observed = base_inputs.load()
    phase = phase.upper()

    # ---- true model and times
    true_path = work / "true_model.nc"
    true_model.to_netcdf(true_path)
    calc = forward_times(
        true_model, grid, base_inputs, work / "fm3d_true", n_jobs=n_jobs, phase=phase
    )
    calc_bg = None
    if background_arrivals is not None:
        calc_bg = (
            background_arrivals
            if isinstance(background_arrivals, pd.DataFrame)
            else read_arrivals(background_arrivals)
        )
    elif background_model is not None:
        calc_bg = forward_times(
            background_model,
            grid,
            base_inputs,
            work / "fm3d_background",
            n_jobs=n_jobs,
            phase=phase,
        )

    # ---- noise
    ss_noise, ss_hypo = np.random.SeedSequence(seed).spawn(2)
    rng_noise = np.random.default_rng(ss_noise)
    rng_hypo = np.random.default_rng(ss_hypo)
    nz = rc.noise
    noisy, clean = synthetic_observed(
        observed,
        calc,
        scale=nz.scale,
        from_pick_uncertainty=nz.from_pick_uncertainty,
        sigma_s=nz.sigma_s,
        rng=rng_noise,
    )
    otimes = work / f"otimes_{phase}.dat"
    write_observed(otimes, noisy[[*_KEY, "time", "uncertainty"]])
    otimes_clean = work / f"otimes_{phase}_noiseless.dat"
    write_observed(otimes_clean, clean[[*_KEY, "time", "uncertainty"]])

    # ---- hypocentres
    hn = rc.hypocentre_noise_km
    do_perturb = (hn.xy > 0 or hn.z > 0) if perturb_hypocentres is None else perturb_hypocentres
    true_src = work / "sources_true.in"
    write_sources(true_src, sources)
    shifts = np.zeros((len(sources), 3))
    start = sources
    if do_perturb:
        start, shifts = perturb_sources(sources, hn.xy, hn.z, rng_hypo, grid)
    src_path, ref_path = work / "sources.in", work / "sourcesref.in"
    write_sources(src_path, start)
    write_sources(ref_path, start)

    # ---- statistics
    v = noisy["valid"].to_numpy(bool)
    nv = noisy.loc[v, "noise"].to_numpy(float)
    sg = noisy.loc[v, "sigma"].to_numpy(float)
    stats: dict[str, Any] = {
        "phase": phase,
        "seed": int(seed),
        "n_obs": int(len(noisy)),
        "n_valid": int(v.sum()),
        "n_invalid": int((~v).sum()),
        "noise_scale": float(nz.scale),
        "noise_from_pick_uncertainty": bool(nz.from_pick_uncertainty),
        "noise_mean_s": float(nv.mean()) if nv.size else float("nan"),
        "noise_std_s": float(nv.std()) if nv.size else float("nan"),
        "noise_rms_s": _rms(nv),
        "noise_sigma_rms_s": _rms(sg),
        "noise_normalised_std": float((nv / sg).std()) if nv.size and (sg > 0).all() else None,
        "hypocentres_perturbed": bool(do_perturb),
        "hypocentre_shift_rms_km": {
            "xy": _rms(shifts[:, :2]),
            "z": _rms(shifts[:, 2]),
        },
        "signal_rms_s": float("nan"),
        "snr": float("nan"),
    }
    if calc_bg is not None:
        sig_clean, _ = synthetic_observed(
            observed, calc_bg, scale=0.0, from_pick_uncertainty=False, sigma_s=0.0, rng=rng_noise
        )
        bg = sig_clean["time"].to_numpy(float)
        tr = clean["time"].to_numpy(float)
        both = clean["valid"].to_numpy(bool) & sig_clean["valid"].to_numpy(bool)
        sig = tr[both] - bg[both]
        stats["signal_rms_s"] = _rms(sig)
        stats["signal_mean_s"] = float(sig.mean()) if sig.size else float("nan")
        stats["snr"] = (
            stats["signal_rms_s"] / stats["noise_rms_s"]
            if stats["noise_rms_s"] > 0
            else float("inf")
        )
    (work / "synthetic_summary.json").write_text(json.dumps(stats, indent=2, default=float))
    log.info(
        "synthetic %s data: %d valid obs, noise rms %.4f s, signal rms %.4f s",
        phase,
        stats["n_valid"],
        stats["noise_rms_s"],
        stats["signal_rms_s"],
    )
    return SyntheticData(
        workdir=work,
        otimes=otimes,
        otimes_noiseless=otimes_clean,
        sources=src_path,
        sourcesref=ref_path,
        true_sources=true_src,
        true_model=true_path,
        arrivals_true=work / "fm3d_true" / "arrivals.dat",
        observed=noisy,
        stats=stats,
        seed=int(seed),
    )
