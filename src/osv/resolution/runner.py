"""Resolution-test inversions: synthetic data in, recovered model out (Stage 6).

For every test (a checkerboard cell size, a spike set, an input model or an anomaly body, each
for P and, if configured, S) :func:`run_resolution_tests`

1. builds the TRUE model (:mod:`osv.resolution.models`),
2. makes synthetic data with the real geometry and pick-based noise
   (:func:`osv.resolution.synth.make_synthetic_data`; rows of ``otimes`` stay source-major in
   ``arrivals.dat`` order),
3. assembles a run directory whose STARTING model is the *background* model (the starting model
   of the real inversion) and whose controls are those of the real inversion: damping and
   smoothing from the Stage 4 trade-off (``tradeoff.json``, see
   :func:`osv.fmtomo.tradeoff.load_chosen`) or given explicitly, the same relocation mode,
   iterations and stopping,
4. runs :func:`osv.fmtomo.invert.run_inversion`,
5. writes ``true_model.nc``, ``recovered_model.nc`` (final velocity on the same grid, plus the
   perturbation in percent of the background), ``iterations.csv`` and ``summary.json`` in
   ``<workdir>/<test name>/`` and deletes the heavy files of the run directory.

Tests are independent; they run concurrently within an ``fm3d`` process budget and are
resumable: a test whose ``summary.json`` says ``success`` for the same settings is skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.fmtomo import forward as _fwd
from osv.fmtomo import invert as _inv
from osv.fmtomo.controls import ModelInfo, build_controls
from osv.fmtomo.metrics import attach_metrics
from osv.fmtomo.rundir import assemble_run_dir, derive_src_dat
from osv.fmtomo.tradeoff import HEAVY_FILES, load_chosen
from osv.io.fmtomo import PathSpec, read_observed, read_sources, read_vgrids
from osv.model3d.grid import ModelGrid
from osv.resolution import models as _models
from osv.resolution import synth as _synth
from osv.resolution.synth import NETCDF_LOCK, BaseInputs

log = logging.getLogger(__name__)

__all__ = [
    "ResolutionRunResult",
    "ResolutionTest",
    "build_tests",
    "run_resolution_tests",
]

SUMMARY = "summary.json"
#: default fm3d processes per concurrent test
MIN_JOBS_PER_TEST = 4
#: refuse to start a test with less free disk than this (GB)
MIN_FREE_DISK_GB = 1.0
TABLE_COLUMNS = (
    "name",
    "kind",
    "phase",
    "params",
    "success",
    "misfit_initial_s",
    "misfit_final_s",
    "n_iter",
    "runtime_s",
    "flags",
    "true_model",
    "recovered_model",
    "dir",
)


@dataclass
class ResolutionTest:
    """One resolution test: a name, a kind, and a recipe for the true model.

    ``build(grid, background_model)`` returns the true model Dataset (``vp``, ``vs``).
    ``params`` are the user-visible parameters (cell size, amplitude, ...), JSON-able.
    """

    name: str
    kind: str
    phase: str
    params: dict[str, Any]
    build: Callable[[ModelGrid, xr.Dataset], xr.Dataset]


@dataclass
class ResolutionRunResult:
    """Outcome of :func:`run_resolution_tests`.

    ``table`` has one row per test (see :data:`TABLE_COLUMNS`); ``misfit_*`` are the data rms
    (s) of iteration 0 (background model) and of the last iteration.
    """

    workdir: Path
    table: pd.DataFrame
    chosen_damping: float
    chosen_smoothing: float
    n_iter: int
    max_concurrent: int
    jobs_per_test: int
    skipped: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return [
            str(n)
            for n, ok in zip(self.table["name"], self.table["success"], strict=True)
            if not ok
        ]

    def path(self, name: str, what: str = "recovered") -> Path:
        """Path of the ``recovered`` or ``true`` NetCDF of test ``name``."""
        col = "recovered_model" if what == "recovered" else "true_model"
        return Path(self.table.set_index("name").loc[name, col])


# ---------------------------------------------------------------------------------------
# test specification
# ---------------------------------------------------------------------------------------
def _slug(v: float) -> str:
    return f"{float(v):g}".replace(".", "p").replace("-", "m")


def build_tests(
    cfg: Any,
    phases: Sequence[str] | None = None,
    *,
    input_model_kwargs: dict[str, dict[str, Any]] | None = None,
    vs_scale: float = 1.0,
) -> list[ResolutionTest]:
    """Tests of ``cfg.resolution``: one per cell size (and offset copy), spike set, input model.

    Parameters
    ----------
    cfg
        :class:`~osv.config.schema.OSVConfig` (or anything with ``resolution`` and ``fmtomo``).
    phases
        Default: the P and S entries of ``fmtomo.phases`` (P if neither).
    input_model_kwargs
        ``{kind: kwargs of models.input_model}`` for the configured ``input_models`` (e.g.
        ``{"final_3d": {"dataset": ds}, "user_model": {"cfg": model3d_cfg, ...}}``).  Kinds
        without an entry are skipped with a warning; ``"file"`` uses
        ``resolution.input_model_file`` and ``"anomaly"`` makes one test per configured body.
    vs_scale
        Passed to the pattern generators (Vs amplitude relative to Vp).
    """
    rc = getattr(cfg, "resolution", cfg)
    fm = getattr(cfg, "fmtomo", None)
    if phases is None:
        ph = [p for p in (fm.phases if fm is not None else ["P"]) if p in ("P", "S")] or ["P"]
    else:
        ph = [p.upper() for p in phases]
    imk = input_model_kwargs or {}
    out: list[ResolutionTest] = []

    def add(name: str, kind: str, params: dict[str, Any], build: Callable[..., xr.Dataset]) -> None:
        for p in ph:
            out.append(ResolutionTest(f"{name}_{p}", kind, p, dict(params), build))

    cb = rc.checkerboard
    for cell in cb.cell_km:
        for off in (False, True) if cb.offset_half_cell else (False,):

            def mk(grid, bg, cell=cell, off=off):  # noqa: ANN001, ANN202
                return _models.checkerboard(
                    grid,
                    bg,
                    float(cell),
                    cb.amplitude_pct,
                    off,
                    shape=cb.shape,
                    depth_cells=cb.depth_cells,
                    vs_scale=vs_scale,
                )

            add(
                f"checker_{_slug(cell)}km" + ("_off" if off else ""),
                "checkerboard",
                {
                    "cell_km": float(cell),
                    "amplitude_pct": cb.amplitude_pct,
                    "offset_half_cell": bool(off),
                    "shape": cb.shape,
                    "depth_cells": bool(cb.depth_cells),
                },
                mk,
            )
    if rc.spikes:
        specs = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in rc.spikes]
        add(
            "spikes",
            "spikes",
            {"spikes": specs},
            lambda grid, bg: _models.spikes(grid, bg, specs, vs_scale=vs_scale),
        )
    for kind in rc.input_models:
        if kind == "anomaly":
            for i, body in enumerate(rc.anomalies):
                b = body.model_dump() if hasattr(body, "model_dump") else dict(body)
                add(
                    f"anomaly{i + 1}_{b.get('kind', 'body')}",
                    "anomaly",
                    {"body": b},
                    lambda grid, bg, b=b: _models.anomaly_bodies(grid, bg, [b], vs_scale=vs_scale),
                )
            continue
        kw = dict(imk.get(kind, {}))
        if kind == "file" and "path" not in kw and rc.input_model_file is not None:
            kw["path"] = rc.input_model_file
        if not kw:
            log.warning("input model %r skipped: no source given in input_model_kwargs", kind)
            continue
        add(
            f"input_{kind}",
            f"input:{kind}",
            {k: (str(v) if isinstance(v, Path) else "<object>") for k, v in kw.items()},
            lambda grid, bg, kind=kind, kw=kw: _models.input_model(
                kind, grid, bg, vs_scale=vs_scale, **kw
            ),
        )
    names = [t.name for t in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate resolution test names: {names}")
    return out


# ---------------------------------------------------------------------------------------
# one test
# ---------------------------------------------------------------------------------------
def _f(x: Any) -> float:
    try:
        return math.nan if x is None else float(x)
    except (TypeError, ValueError):
        return math.nan


def _signature(
    test: ResolutionTest,
    d: float,
    s: float,
    n_iter: int,
    seed: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything that makes a cached test result stale (compared by :func:`_read_summary`).

    ``context`` holds the synthetic-data settings and the digest of the true model (see
    :func:`_test_context`); without it only the regularisation and the test's parameters count.
    """
    sig = {
        "name": test.name,
        "phase": test.phase,
        "damping": float(d),
        "smoothing": float(s),
        "n_iter": int(n_iter),
        "seed": int(seed),
        "params": json.loads(json.dumps(test.params, default=str)),
    }
    if context is not None:
        sig.update(context)
    return sig


def _station_terms(
    fm: Any, otimes: pd.DataFrame, stations_are_sources: bool, event_ids: Any
) -> _inv.StationTermSpec | None:
    """Same station-term procedure as the real inversion (``fmtomo.station_terms: iterate``)."""
    if getattr(fm, "station_terms", "none") != "iterate":
        return None
    o = otimes.reset_index(drop=True)
    if stations_are_sources:
        station, event = o["source"].to_numpy(), np.asarray(event_ids)
    else:
        station, event = o["receiver"].to_numpy(), o["source"].to_numpy()
    opt = fm.station_term_iteration
    return _inv.StationTermSpec(
        station=station,
        event=event,
        every=int(opt.every),
        start_iteration=int(opt.start_iteration),
        relaxation=float(opt.relaxation),
        min_picks=int(opt.min_picks),
    )


def _test_context(
    test: ResolutionTest, grid: ModelGrid, background: xr.Dataset, cfg: Any, uncertainty: float
) -> dict[str, Any]:
    """Synthetic-data settings and a digest of the true model (input models are objects)."""
    rc = getattr(cfg, "resolution", None)
    synth: dict[str, Any] = {"uncertainty": float(uncertainty)}
    fm = getattr(cfg, "fmtomo", None)
    if fm is not None and getattr(fm, "station_terms", "none") != "none":
        synth["station_terms"] = fm.station_terms
        synth["station_term_iteration"] = fm.station_term_iteration.model_dump(mode="json")
    if rc is not None:
        for k in ("noise", "hypocentre_noise_km"):
            v = getattr(rc, k, None)
            if v is not None:
                synth[k] = v.model_dump(mode="json") if hasattr(v, "model_dump") else v
    with NETCDF_LOCK:  # builders may read NetCDF files (HDF5 is not thread-safe)
        true = test.build(grid, background)
    h = hashlib.sha256()
    for var in sorted(true.data_vars):
        h.update(var.encode())
        h.update(np.ascontiguousarray(true[var].values, dtype=np.float64).tobytes())
    return {"synthetic_settings": json.loads(json.dumps(synth)), "true_digest": h.hexdigest()[:16]}


def _read_summary(path: Path, sig: dict[str, Any]) -> dict[str, Any] | None:
    try:
        sm = json.loads((path / SUMMARY).read_text())
    except (OSError, ValueError):
        return None
    if not sm.get("success"):
        return None
    for k, v in sig.items():
        if isinstance(v, float):
            if not math.isclose(_f(sm.get(k)), v):
                return None
        elif sm.get(k) != v:
            return None
    if not (path / "recovered_model.nc").exists() or not (path / "true_model.nc").exists():
        return None
    return sm


def _compact(*roots: Path) -> None:
    for root in roots:
        if not root.exists():
            continue
        for name in HEAVY_FILES:
            for p in root.rglob(name):
                p.unlink(missing_ok=True)
        for pat in ("parallel", "fm3d_*", "group*"):
            for p in root.glob(pat):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
        for p in root.rglob("*.dat"):
            if p.name.startswith(
                ("timefield", "tfield", "gridsave", "rays")
            ) or p.stat().st_size > (50 << 20):
                p.unlink(missing_ok=True)


def _recovered(
    run: Path, grid: ModelGrid, background: xr.Dataset, phase: str, true: xr.Dataset
) -> xr.Dataset:
    """Final model of the run as a Dataset on the full grid (same grid as the true model)."""
    vg = read_vgrids(run / "vgrids.in")
    vel = vg.get(1, 1)["velocity"].values
    if vel.shape != grid.shape_full:
        raise ValueError(f"final model shape {vel.shape} differs from the grid {grid.shape_full}")
    var = "vp" if phase == "P" else "vs"
    coords = {"depth": grid.depth_full, "lat": grid.lat_full, "lon": grid.lon_full}
    dims = ("depth", "lat", "lon")
    bg = background[var].values
    ds = xr.Dataset(
        {
            var: (dims, vel, {"units": "km/s"}),
            f"d{var}_pct": (dims, 100.0 * (vel / bg - 1.0), {"units": "percent"}),
            f"{var}_bg": (dims, bg, {"units": "km/s"}),
        },
        coords=coords,
        attrs={"phase": phase, "kind": "recovered"},
    )
    if "dvp_pct" in true:
        ds["dvp_pct_true"] = (dims, true["dvp_pct"].values, {"units": "percent"})
    return ds.assign_coords({k: (k, v, background[k].attrs) for k, v in coords.items()})


def _run_test(
    test: ResolutionTest,
    base: BaseInputs,
    grid: ModelGrid,
    background: xr.Dataset,
    bg_arrivals: pd.DataFrame | None,
    cfg: Any,
    d: float,
    s: float,
    n_iter: int,
    seed: int,
    workdir: Path,
    n_jobs: int,
    uncertainty: float,
    mem_per_job_gb: float | None,
    timeout: float | None,
) -> dict[str, Any]:
    tdir = workdir / test.name
    sig = _signature(
        test, d, s, n_iter, seed, _test_context(test, grid, background, cfg, uncertainty)
    )
    done = _read_summary(tdir, sig)
    if done is not None:
        log.info("resolution test %s already complete, skipping", test.name)
        done["skipped"] = True
        return done
    shutil.rmtree(tdir, ignore_errors=True)
    tdir.mkdir(parents=True)
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        **sig,
        "kind": test.kind,
        "params": test.params,
        "success": False,
        "dir": str(tdir),
        "true_model": str(tdir / "true_model.nc"),
        "recovered_model": str(tdir / "recovered_model.nc"),
        "flags": "",
    }
    try:
        fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
        fm_run = fm.model_copy(update={"damping": float(d), "smoothing": float(s)})
        ctl = build_controls(fm_run, None, phases=[test.phase])
        with NETCDF_LOCK:  # builders may read NetCDF files (HDF5 is not thread-safe)
            true = test.build(grid, background)
            true.to_netcdf(tdir / "true_model.nc")
        # starting hypocentres: nonlinear relocation grid-searches, the start is irrelevant
        perturb = None if not ctl.stations_are_sources else False
        sd = _synth.make_synthetic_data(
            true,
            grid,
            base,
            tdir / "synth",
            cfg,
            seed=seed,
            n_jobs=n_jobs,
            phase=test.phase,
            perturb_hypocentres=perturb,
            background_arrivals=bg_arrivals,
        )
        row["synthetic"] = sd.summary()
        itf, pg, _, rec, _ = base.load()
        sources = read_sources(sd.sources)
        if test.phase == "S":  # separate layout: S is velocity type 1 of its own run
            sources = [
                replace(src, paths=[PathSpec(p.steps, (1,) * len(p.vtypes)) for p in src.paths])
                for src in sources
            ]
        otimes = read_observed(sd.otimes)
        var = "vp" if test.phase == "P" else "vs"
        if var not in background:
            raise ValueError(f"background model has no {var!r}")
        grids = grid.to_vgrids(background[var], uncertainty=uncertainty)
        info = ModelInfo.from_model(grids, sources=sources, receivers=rec)
        ctl = build_controls(fm_run, info, phases=[test.phase])
        run = tdir / "run"
        src_dat = None
        if ctl.stations_are_sources:
            if base.event_ids is None or len(base.event_ids) != len(otimes):
                raise ValueError(
                    "nonlinear relocation needs the event number of every observation "
                    "(BaseInputs.event_ids) to write src.dat"
                )
            src_dat = derive_src_dat(otimes, len(sources), base.event_ids)
        assemble_run_dir(
            run, grids, itf, pg, sources, rec, otimes, ctl, phase=test.phase, src_dat=src_dat
        )
        res = _inv.run_inversion(
            run,
            ctl,
            n_iter,
            n_jobs=n_jobs,
            keep_iterations=True,
            timeout=timeout,
            mem_per_job_gb=mem_per_job_gb,
            stop_on_error=False,
            phase=test.phase,
            station_terms=_station_terms(fm, otimes, ctl.stations_are_sources, base.event_ids),
        )
        tab = res.iterations
        flags = list(res.flags)
        try:
            tab = attach_metrics(res)
        except Exception as exc:  # noqa: BLE001 - metrics are best effort
            flags.append(f"metrics failed: {exc}")
        tab.to_csv(tdir / "iterations.csv", index=False)
        if res.success:
            rec = _recovered(run, grid, background, test.phase, true)
            with NETCDF_LOCK:
                rec.to_netcdf(tdir / "recovered_model.nc")
        row.update(
            success=bool(res.success),
            misfit_initial_s=_f(tab["rms_ms"].iloc[0]) / 1000.0,
            misfit_final_s=_f(tab["rms_ms"].iloc[-1]) / 1000.0,
            flags="; ".join(flags),
        )
    except Exception as exc:  # noqa: BLE001 - one failed test must not kill the others
        log.error("resolution test %s failed: %s", test.name, exc)
        row.update(
            success=False,
            flags=f"error: {exc}",
            traceback=traceback.format_exc(limit=4),
            misfit_initial_s=math.nan,
            misfit_final_s=math.nan,
        )
    _compact(tdir)
    row["runtime_s"] = time.perf_counter() - t0
    (tdir / SUMMARY).write_text(json.dumps(row, indent=1, default=float))
    return row


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def _concurrency(
    cfg: Any,
    n_jobs: int,
    n_tests: int,
    override: int | None,
    mem_gb: float,
    avail_gb: float,
) -> tuple[int, int]:
    """``(max_concurrent, jobs_per_test)`` with ``max_concurrent * jobs_per_test <= n_jobs``."""
    n_jobs = max(1, int(n_jobs))
    fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
    k = (
        override
        or getattr(fm.tradeoff, "max_concurrent", None)
        or max(1, n_jobs // MIN_JOBS_PER_TEST)
    )
    k = max(1, min(int(k), n_tests, n_jobs))
    jobs = max(1, n_jobs // k)
    # memory guard: all fm3d processes together must fit
    while k > 1 and k * jobs * mem_gb > 0.8 * avail_gb:
        k -= 1
        jobs = max(1, n_jobs // k)
    return k, jobs


def _resolve_chosen(
    chosen: tuple[float, float] | str | Path | None, cfg: Any
) -> tuple[float, float]:
    fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
    if chosen is None:
        return float(fm.damping), float(fm.smoothing)
    if isinstance(chosen, (str, Path)):
        return load_chosen(chosen)
    return float(chosen[0]), float(chosen[1])


def run_resolution_tests(
    tests_spec: Sequence[ResolutionTest] | None,
    base_inputs: BaseInputs | dict[str, BaseInputs],
    grid: ModelGrid,
    background_model: xr.Dataset,
    cfg: Any,
    workdir: str | Path,
    n_jobs: int = 1,
    chosen: tuple[float, float] | str | Path | None = None,
    *,
    iterations: int | None = None,
    seed: int | None = None,
    max_concurrent: int | None = None,
    uncertainty_km_s: float = 0.3,
    mem_per_job_gb: float | None = None,
    min_free_disk_gb: float = MIN_FREE_DISK_GB,
    timeout: float | None = None,
    on_test: Callable[[dict[str, Any]], None] | None = None,
) -> ResolutionRunResult:
    """Invert synthetic data for every test with the chosen inversion settings.

    Parameters
    ----------
    tests_spec
        :class:`ResolutionTest` list (see :func:`build_tests`); ``None`` builds them from
        ``cfg.resolution``.
    base_inputs
        Real geometry (:class:`~osv.resolution.synth.BaseInputs`), or ``{phase: BaseInputs}``
        when P and S differ.
    grid, background_model
        The model grid and the background (starting) model with ``vp``/``vs`` on its full nodes:
        the starting model of every test inversion.
    cfg
        :class:`~osv.config.schema.OSVConfig`: ``fmtomo`` (relocation mode, iterations, stopping,
        solver settings) and ``resolution`` (noise, hypocentre noise).
    workdir
        Output directory; one subdirectory per test.
    n_jobs
        Total ``fm3d`` processes (``max_concurrent * jobs_per_test <= n_jobs``).
    chosen
        ``(damping, smoothing)``, a ``tradeoff.json`` (or its directory), or ``None`` for
        ``fmtomo.damping``/``fmtomo.smoothing``.
    iterations
        Inversion iterations (default ``fmtomo.iterations``, as the real inversion).
    seed
        Noise seed (default ``project.random_seed``); the same for all tests.
    max_concurrent
        Concurrent tests (default ``fmtomo.tradeoff.max_concurrent`` or ``n_jobs // 4``).
    uncertainty_km_s
        A-priori model uncertainty column of the starting ``vgrids``.
    min_free_disk_gb
        Raise before starting if the work disk has less free space.

    Returns
    -------
    ResolutionRunResult
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    fm = cfg.fmtomo if hasattr(cfg, "fmtomo") else cfg
    d, s = _resolve_chosen(chosen, cfg)
    n_iter = int(iterations if iterations is not None else fm.iterations)
    if seed is None:
        seed = int(getattr(getattr(cfg, "project", None), "random_seed", 42))
    tests = list(tests_spec) if tests_spec is not None else build_tests(cfg)
    if not tests:
        raise ValueError("no resolution tests to run (check resolution.* in the config)")
    free = shutil.disk_usage(wd).free / 1e9
    if free < min_free_disk_gb:
        raise OSError(
            f"only {free:.1f} GB free in {wd}; need {min_free_disk_gb:g} GB "
            "(free disk space or lower resolution.min_free_disk_gb)"
        )
    mem = mem_per_job_gb if mem_per_job_gb is not None else float(grid.fm3d_memory_gb)
    k, jobs = _concurrency(cfg, n_jobs, len(tests), max_concurrent, mem, _fwd.available_memory_gb())
    bases = base_inputs if isinstance(base_inputs, dict) else None
    log.info(
        "resolution tests: %d tests, damping %g smoothing %g, %d iterations, %d x %d fm3d jobs",
        len(tests),
        d,
        s,
        n_iter,
        k,
        jobs,
    )

    # background arrivals once per phase (signal statistics of every synthetic data set)
    bg_arr: dict[str, pd.DataFrame | None] = {}
    todo = [
        t
        for t in tests
        if _read_summary(
            wd / t.name,
            _signature(
                t,
                d,
                s,
                n_iter,
                seed,
                _test_context(t, grid, background_model, cfg, uncertainty_km_s),
            ),
        )
        is None
    ]
    for p in sorted({t.phase for t in todo}):
        base = bases[p] if bases else base_inputs  # type: ignore[index]
        try:
            bg_arr[p] = _synth.forward_times(
                background_model, grid, base, wd / f"_background_{p}", n_jobs=n_jobs, phase=p
            )
            _compact(wd / f"_background_{p}")
        except Exception as exc:  # noqa: BLE001 - only the signal statistics need it
            log.warning("background forward model (%s) failed: %s", p, exc)
            bg_arr[p] = None

    def work(t: ResolutionTest) -> dict[str, Any]:
        row = _run_test(
            t,
            bases[t.phase] if bases else base_inputs,  # type: ignore[index,arg-type]
            grid,
            background_model,
            bg_arr.get(t.phase),
            cfg,
            d,
            s,
            n_iter,
            seed,
            wd,
            jobs,
            uncertainty_km_s,
            mem_per_job_gb,
            timeout,
        )
        if on_test is not None:
            on_test(row)
        return row

    if k == 1:
        rows = [work(t) for t in tests]
    else:
        with ThreadPoolExecutor(max_workers=k) as ex:
            rows = list(ex.map(work, tests))

    skipped = [str(r["name"]) for r in rows if r.pop("skipped", False)]
    table = pd.DataFrame(
        [{c: r.get(c, math.nan) for c in TABLE_COLUMNS} for r in rows], columns=list(TABLE_COLUMNS)
    )
    table["params"] = table["params"].map(lambda p: json.dumps(p, default=str))
    res = ResolutionRunResult(wd, table, d, s, n_iter, k, jobs, skipped)
    (wd / "resolution_runs.json").write_text(
        json.dumps(
            {
                "chosen_damping": d,
                "chosen_smoothing": s,
                "n_iter": n_iter,
                "max_concurrent": k,
                "jobs_per_test": jobs,
                "skipped": skipped,
                "table": json.loads(table.to_json(orient="records")),
            },
            indent=1,
        )
    )
    return res
