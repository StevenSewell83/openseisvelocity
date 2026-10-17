"""Multi-stage damping, reseeding and repeat-until-stable VELEST workflow (Stage 2).

The minimum 1-D model procedure of Kissling et al. (1994) and the VELEST guidelines
(Kissling 1995) is iterative: invert simultaneously for velocities, hypocentres and station
delays with strong damping first, then repeat from the outputs with the final damping until
the model and hypocentres stop changing.  This module implements that on top of
:mod:`osv.velest.runner`.

Hand-off between runs
---------------------
* **Model**: the previous run's final model (``VelestRunResult.final_model``) is written to
  ``<stage>/_inputs/<id>.mod`` and becomes the next ``model.mod``.
* **Hypocentres**: the previous run's ``velout.cnv`` is the next input CNV.
* **Station delays**: VELEST ignores the ``ptcor``/``stcor`` columns of the station file
  unless ``iusestacorr = 1`` (``velest.f:1566-1569`` zeroes them when it is 0).  The
  workflow therefore writes a per-run copy of the original station file with ``ptcor`` (P)
  and ``stcor`` (S) replaced by the previous run's totals (VELEST reports absolute delays:
  ``ptcor = ptcor + cc``, ``velest.f:4935``) and sets ``iusestacorr = 1`` for that run.
  The original station file keeps its ``icc`` indices (reference station).
* With ``nsp = 3`` (fixed Vp/Vs) the S delay is ``ptcor * vpvs`` but S readings give no
  delay partials (``velest.f:4100``), so *inverting* delays while S data are used is
  inconsistent and makes the RMS erratic (``DampingStage.invert_station_delays`` can hold
  them fixed).  The repeat loop of :func:`iterate_final_model` therefore freezes the
  handed-on delays by default for ``nsp = 3``; the hand-off itself is exact.
* A model whose run fails (or whose output cannot be parsed) is dropped from later stages;
  its last row stays in the run table with ``status != "ok"``.

Directories: round 1 uses ``<root>/stage<k>/<id>``; reseeded round ``r >= 2`` uses
``<root>/round<r>/stage<k>/<id>``; the iterate loop uses ``<root>/iterate/cycleNN``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from osv.config.schema import DampingStage, VelestConfig
from osv.io.velest import (
    ModelBlock,
    VelestControl,
    VelestModel,
    read_model,
    read_stations,
    write_model,
    write_stations,
)
from osv.velest.models import TrialModel
from osv.velest.runner import STATUS_FAILED, STATUS_OK, VelestJob, run_jobs
from osv.velest.vpvs import model_vpvs

logger = logging.getLogger(__name__)

__all__ = [
    "IterationResults",
    "WorkflowResult",
    "iterate_final_model",
    "VpVsResolutionError",
    "perturb_model",
    "resolve_stages",
    "resolve_vdamp",
    "resolve_vpvs",
    "run_mc_workflow",
    "select_seeds",
]


class InputsLike(Protocol):
    """What the workflow needs from :class:`osv.data.velest_inputs.VelestInputs`."""

    cnv_path: Path
    sta_path: Path
    control: VelestControl


@dataclass
class WorkflowResult:
    """Result of :func:`run_mc_workflow`.

    Attributes
    ----------
    runs
        One row per model and round (its last attempted stage), with the columns of
        :func:`osv.velest.runner.run_jobs` (``id, status, ..., run_dir, parsed,
        final_rms``) plus ``round``, ``stage`` (name), ``origin`` (round-1 model id) and
        ``parent`` (seed id, ``None`` in round 1).  Feed this to the ensemble analysis.
    history
        One row per executed stage run: ``round, stage_index, stage, id, parent, status,
        initial_rms, final_rms, n_iterations, converged, run_dir``.
    final_models
        ``id -> TrialModel`` with the final 1-D model of every successful run.
    """

    runs: pd.DataFrame
    history: pd.DataFrame
    final_models: dict[str, TrialModel]
    root: Path
    stages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> pd.DataFrame:
        """Rows of successful runs."""
        return self.runs[self.runs["status"] == STATUS_OK]

    def final_run_dirs(self) -> dict[str, Path]:
        """``id -> run directory`` of the last stage of every run."""
        return {r.id: Path(r.run_dir) for r in self.runs.itertuples()}


class IterationResults(list):
    """List of per-cycle :class:`~osv.io.velest_out.VelestRunResult` with ``history``.

    ``history`` has ``cycle, rms, model_change_km_s, hypo_change_km, converged, run_dir``;
    ``converged`` is True when the last cycle was below both thresholds.
    """

    history: pd.DataFrame
    converged: bool
    best_cycle: int | None

    def __init__(self, items: Sequence[Any] = ()) -> None:
        super().__init__(items)
        self.history = pd.DataFrame()
        self.converged = False
        self.best_cycle = None  # 1-based cycle with the lowest RMS (the last one may be worse)


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def _vcfg(cfg: Any) -> VelestConfig:
    return getattr(cfg, "velest", cfg)


def _seed(cfg: Any, seed: int | None) -> int:
    if seed is not None:
        return seed
    return int(getattr(getattr(cfg, "project", None), "random_seed", 0))


class VpVsResolutionError(ValueError):
    """``velest.vpvs.value = 'wadati'`` but no usable Wadati ratio is available."""


def resolve_vpvs(cfg: Any, prep_stage_dir: str | Path) -> float:
    """Vp/Vs ratio for the VELEST control file (``nsp = 3``).

    A numeric ``velest.vpvs.value`` is returned unchanged.  With ``'wadati'`` the ratio is
    read from ``wadati.json`` (key ``vpvs``) in the Stage 1 (data preparation) directory.
    For ``vpvs.mode = 'sampled'`` the control ratio is the middle of ``range`` and this value
    is not used (the function still resolves it).

    Parameters
    ----------
    cfg
        :class:`~osv.config.schema.OSVConfig` or its ``velest`` section.
    prep_stage_dir
        Directory of the data-preparation stage (holds ``wadati.json``).

    Raises
    ------
    VpVsResolutionError
        If ``wadati.json`` is missing, empty (too few S-P pairs for a Wadati fit), or holds a
        ratio that is not finite and above 1.  Set ``velest.vpvs.value`` to a number then.
    """
    value = _vcfg(cfg).vpvs.value
    if not isinstance(value, str):
        return float(value)
    path = Path(prep_stage_dir) / "wadati.json"
    hint = "set velest.vpvs.value to a number or fix the Stage 1 Wadati fit"
    if not path.is_file():
        raise VpVsResolutionError(
            f"velest.vpvs.value is 'wadati' but {path} does not exist; {hint}"
        )
    try:
        ratio = float(json.loads(path.read_text()).get("vpvs", math.nan))
    except (ValueError, TypeError, AttributeError) as exc:
        raise VpVsResolutionError(f"cannot read a Vp/Vs ratio from {path} ({exc}); {hint}") from exc
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise VpVsResolutionError(f"{path}: Wadati Vp/Vs = {ratio!r} is unusable; {hint}")
    return ratio


def resolve_stages(cfg: Any) -> list[DampingStage]:
    """Stages with every ``None`` replaced by the top-level ``velest`` settings.

    An empty ``stages`` list yields one stage named ``"single"``.
    """
    v = _vcfg(cfg)
    stages = list(v.stages) or [DampingStage(name="single")]
    out = []
    for st in stages:
        d = v.damping
        out.append(
            st.model_copy(
                update={
                    "othet": d.othet if st.othet is None else st.othet,
                    "xythet": d.xythet if st.xythet is None else st.xythet,
                    "zthet": d.zthet if st.zthet is None else st.zthet,
                    "vthet": d.vthet if st.vthet is None else st.vthet,
                    "stathet": d.stathet if st.stathet is None else st.stathet,
                    "ittmax": v.ittmax if st.ittmax is None else st.ittmax,
                    "invertratio": v.invertratio if st.invertratio is None else st.invertratio,
                }
            )
        )
    return out


def resolve_vdamp(
    cfg: Any,
    n_layers: int | None = None,
    *,
    layers_top_km: Sequence[float] | None = None,
    station_z_km: Sequence[float] | None = None,
    event_depth_km: Sequence[float] | None = None,
) -> np.ndarray:
    """Per-layer VELEST velocity damping factors (``vdamp`` of the model file).

    VELEST damps layer ``i`` with ``vthet * vdamp[i]`` (user guide p. 21; ``vdamp = 999`` almost
    fixes a layer).  ``velest.vdamp`` is a number (all layers), a list (one value per layer) or
    ``"auto"``:

    * layers that start above the deepest station (``top < max(station_z)``): every ray crosses
      them at a steep angle, so their velocity trades off with the station delays;
      ``vdamp_auto.near_surface`` for these;
    * layers that start below the deepest hypocentre (``top > max(event_depth)``): only
      refracted rays sample them; ``vdamp_auto.below_events``;
    * all others 1.

    Without the geometry arguments (``station_z_km``, ``event_depth_km``) the corresponding rule
    is skipped, so ``"auto"`` gives ones.

    Parameters
    ----------
    cfg
        ``OSVConfig`` or its ``velest`` section.
    n_layers, layers_top_km
        Layer count (default ``len(cfg.layers_top_km)``) and the layer tops used by ``"auto"``
        (default ``cfg.layers_top_km``).
    station_z_km, event_depth_km
        Station depths (``-elevation / 1000``, km positive down) and hypocentre depths (km).
    """
    v = _vcfg(cfg)
    tops = np.asarray(layers_top_km if layers_top_km is not None else v.layers_top_km, float)
    n = len(tops) if n_layers is None else n_layers
    spec = v.vdamp
    if isinstance(spec, str):
        out = np.ones(n)
        if station_z_km is not None and len(station_z_km):
            out[tops < float(np.max(station_z_km))] = v.vdamp_auto.near_surface
        if event_depth_km is not None and len(event_depth_km):
            below = tops > float(np.max(event_depth_km))
            out[below] = np.maximum(out[below], v.vdamp_auto.below_events)
        return out
    arr = np.broadcast_to(np.asarray(spec, float), (n,)).copy()
    return arr


def _stage_control(
    base: VelestControl, st: DampingStage, use_stacorr: bool, vpvs: float | None = None
) -> VelestControl:
    c = copy.deepcopy(base)
    if vpvs is not None and c.nsp == 3:
        c.vpvs = float(vpvs)
    c.othet, c.xythet, c.zthet = float(st.othet), float(st.xythet), float(st.zthet)  # type: ignore[arg-type]
    c.vthet, c.stathet = float(st.vthet), float(st.stathet)  # type: ignore[arg-type]
    c.ittmax, c.invertratio = int(st.ittmax), int(st.invertratio)  # type: ignore[arg-type]
    if st.swtfac is not None:
        c.swtfac = float(st.swtfac)
    if use_stacorr:
        c.iusestacorr = 1
    if st.invert_station_delays is not None:
        c.nsinv = int(st.invert_station_delays)
    return c


def _trial_to_velest(
    m: TrialModel, nmod: int = 2, vdamp: float | Sequence[float] | np.ndarray = 1.0
) -> VelestModel:
    blocks = []
    for label, vel in (("P", m.vp), ("S", m.vs)):
        if label == "S" and nmod != 2:
            break
        vd = np.broadcast_to(np.asarray(vdamp, float), (len(vel),))
        lay = pd.DataFrame({"vel": vel, "depth": m.depths, "vdamp": vd})
        blocks.append(ModelBlock(lay, label=f"{label} {m.id}"))
    return VelestModel(
        f"{m.id} {m.sampler}"[:40], blocks[0], blocks[1] if len(blocks) > 1 else None
    )


def _vd_list(
    vdamp: float | Sequence[float] | np.ndarray, model: TrialModel | VelestModel
) -> list[float]:
    n = len(model.p.layers) if isinstance(model, VelestModel) else model.n_layers
    return [float(x) for x in np.broadcast_to(np.asarray(vdamp, float), (n,))]


def _with_vdamp(model: VelestModel, vdamp: float | Sequence[float] | np.ndarray) -> VelestModel:
    """Copy of ``model`` with ``vdamp`` (scalar or one value per layer) in every block."""
    blocks: list[ModelBlock] = []
    for blk in (model.p, model.s):
        if blk is None:
            continue
        lay = blk.layers.copy()
        lay["vdamp"] = np.broadcast_to(np.asarray(vdamp, float), (len(lay),)).copy()
        blocks.append(ModelBlock(lay, label=blk.label, comment=blk.comment))
    return VelestModel(model.title, blocks[0], blocks[1] if len(blocks) > 1 else None)


def _velest_to_trial(
    vm: VelestModel, id_: str, vpvs_fallback: float, seed: int, index: int
) -> TrialModel:
    vp = vm.p.layers["vel"].to_numpy()
    vs = vm.s.layers["vel"].to_numpy() if vm.s is not None else vp / vpvs_fallback
    return TrialModel(id_, vm.p.layers["depth"].to_numpy(), vp, vs, "velest", seed, index=index)


def perturb_model(
    model: VelestModel, sigma_km_s: float, rng: np.random.Generator, *, monotonic: bool = True
) -> VelestModel:
    """Copy of ``model`` with Gaussian noise (``sigma_km_s``) added to the layer Vp.

    Vs is scaled by the same factor (each layer keeps its Vp/Vs).  With ``monotonic`` the
    perturbed Vp is made non-decreasing by a running maximum.  Values are kept above
    0.5 km/s and rounded to the model file precision.
    """
    p = model.p.layers.copy()
    vp0 = p["vel"].to_numpy()
    vp = np.maximum(vp0 + rng.normal(0.0, sigma_km_s, len(vp0)), 0.5)
    if monotonic:
        vp = np.maximum.accumulate(vp)
    vp = np.round(vp, 2)
    p["vel"] = vp
    blocks: list[ModelBlock | None] = [ModelBlock(p, label=model.p.label, comment=model.p.comment)]
    if model.s is not None:
        s = model.s.layers.copy()
        if len(s) == len(vp0):
            s["vel"] = np.round(s["vel"].to_numpy() * vp / vp0, 2)
        blocks.append(ModelBlock(s, label=model.s.label, comment=model.s.comment))
    return VelestModel(model.title, blocks[0], blocks[1] if len(blocks) > 1 else None)  # type: ignore[arg-type]


def _usable(row: Any) -> bool:
    if row.status != STATUS_OK:
        return False
    parsed = getattr(row, "parsed", None)
    if parsed is None or getattr(parsed, "failed", False):
        return False
    return getattr(parsed, "final_model", None) is not None


def _rms(parsed: Any) -> float:
    v = getattr(parsed, "final_rms", math.nan)
    return float(v) if isinstance(v, int | float | np.floating) else math.nan


def select_seeds(runs: pd.DataFrame, best_percent: float) -> list[str]:
    """Ids of the best ``best_percent`` % of the usable runs by final RMS (at least one).

    Runs that failed, did not parse, diverged or have a non-finite RMS are not eligible; the
    count is ``ceil(best_percent / 100 * n_eligible)``.
    """
    ok = []
    for r in runs.itertuples():
        if _usable(r) and np.isfinite(r.final_rms) and not getattr(r.parsed, "diverged", False):
            ok.append((float(r.final_rms), str(r.id)))
    ok.sort()
    n = max(1, math.ceil(best_percent / 100.0 * len(ok))) if ok else 0
    return [i for _, i in ok[:n]]


@dataclass
class _Entry:
    id: str
    parent: str | None
    origin: str
    model: TrialModel | VelestModel
    base_cnv: Path
    base_corr: pd.DataFrame | None  # station delays inherited at the start of the round
    cur_cnv: Path | None = None
    cur_corr: pd.DataFrame | None = None
    alive: bool = True
    last_row: dict[str, Any] | None = None
    vpvs: float | None = None  # per-model ratio (nsp=3 runs); None: the control's value


def _corr_station_file(base: Path, corr: pd.DataFrame, out: Path) -> Path:
    """Copy of the station file ``base`` with ``ptcor``/``stcor`` from ``corr``."""
    table = read_stations(base)
    for col in ("ptcor", "stcor"):
        if col not in table.columns:
            table[col] = 0.0
        if col in corr.columns:
            m = dict(zip(corr["sta"], corr[col], strict=True))
            table[col] = [
                float(m[s]) if s in m and np.isfinite(m[s]) else float(v)
                for s, v in zip(table["sta"], table[col], strict=True)
            ]
    out.parent.mkdir(parents=True, exist_ok=True)
    write_stations(out, table)
    return out


def _model_path(entry_model: TrialModel | VelestModel, out: Path, nmod: int = 2) -> Path:
    vm = _trial_to_velest(entry_model, nmod) if isinstance(entry_model, TrialModel) else entry_model
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_model(out, vm)


def _history_row(round_no: int, k: int, st: str, e: _Entry, row: Any) -> dict[str, Any]:
    p = getattr(row, "parsed", None)
    return {
        "round": round_no,
        "stage_index": k,
        "stage": st,
        "id": e.id,
        "parent": e.parent,
        "status": row.status,
        "initial_rms": float(getattr(p, "initial_rms", math.nan)) if p is not None else math.nan,
        "final_rms": float(row.final_rms) if np.isfinite(row.final_rms) else math.nan,
        "n_iterations": int(getattr(p, "n_iterations", 0)) if p is not None else 0,
        "converged": bool(getattr(p, "converged", False)) if p is not None else False,
        "run_dir": row.run_dir,
    }


def _run_schedule(
    entries: list[_Entry],
    inputs: InputsLike,
    stages: list[DampingStage],
    root: Path,
    round_no: int,
    n_jobs: int,
    timeout: float | None,
    force: bool,
    compact: bool,
    vdamp: float | Sequence[float] | np.ndarray = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all stages for ``entries``; returns (history rows, last-row records)."""
    hist: list[dict[str, Any]] = []
    base_root = root if round_no == 1 else root / f"round{round_no}"
    for k, st in enumerate(stages, start=1):
        live = [e for e in entries if e.alive]
        if not live:
            break
        stage_root = base_root / f"stage{k}"
        inp = stage_root / "_inputs"
        jobs: list[VelestJob] = []
        for e in live:
            first = k == 1
            cnv = e.cur_cnv if (not first and st.use_previous_hypocentres) else None
            cnv = cnv or e.base_cnv
            corr = e.cur_corr if (not first and st.use_previous_station_corrections) else None
            if corr is None:
                corr = e.base_corr
            if corr is not None:
                sta = _corr_station_file(Path(inputs.sta_path), corr, inp / f"{e.id}.sta")
            else:
                sta = Path(inputs.sta_path)
            if first and isinstance(e.model, TrialModel):
                model: Path | TrialModel = e.model
            else:
                model = _model_path(e.model, inp / f"{e.id}.mod", inputs.control.nmod)
            ctrl = _stage_control(inputs.control, st, corr is not None, e.vpvs)
            jobs.append(VelestJob(e.id, model, ctrl, cnv, sta, vdamp=_vd_list(vdamp, e.model)))
        logger.info(
            "round %d stage %d/%d (%s): %d runs", round_no, k, len(stages), st.name, len(jobs)
        )
        df = run_jobs(
            jobs, stage_root, n_jobs=n_jobs, timeout=timeout, force=force, compact=compact
        )
        by_id = {e.id: e for e in live}
        for row in df.itertuples():
            e = by_id[row.id]
            usable = _usable(row)
            rec = row._asdict()
            rec.pop("Index", None)
            if row.status == STATUS_OK and not usable:
                rec["status"] = STATUS_FAILED
                parsed = getattr(row, "parsed", None)
                why = getattr(parsed, "failure_reason", None)
                rec["error"] = (
                    f"VELEST run failed: {why}"
                    if why
                    else "VELEST output could not be parsed (no final model)"
                )
            rec.update(
                round=round_no, stage=st.name, stage_index=k, origin=e.origin, parent=e.parent
            )
            e.last_row = rec
            hrow = _history_row(round_no, k, st.name, e, row)
            hrow["status"] = rec["status"]
            hist.append(hrow)
            if not usable:
                e.alive = False
                logger.warning("%s dropped at stage %s: %s", e.id, st.name, rec.get("error"))
                continue
            p = row.parsed
            e.model = p.final_model
            e.cur_cnv = Path(row.cnv_path) if row.cnv_path else None
            sc = getattr(p, "station_corrections", None)
            e.cur_corr = sc if isinstance(sc, pd.DataFrame) and len(sc) else None
    return hist, [e.last_row for e in entries if e.last_row is not None]


# ----------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------


def run_mc_workflow(
    models: Sequence[TrialModel],
    inputs: InputsLike,
    cfg: Any,
    root: str | Path,
    n_jobs: int = -1,
    *,
    seed: int | None = None,
    timeout: float | None = 3600.0,
    force: bool = False,
    compact: bool = False,
    vdamp: float | Sequence[float] | np.ndarray | None = None,
) -> WorkflowResult:
    """Run the damping schedule (and optional reseeded rounds) for every trial model.

    Each stage is one parallel :func:`~osv.velest.runner.run_jobs` batch over all models
    still alive; stage ``k`` starts from stage ``k-1``'s final model and, per the stage's
    ``use_previous_*`` flags, its relocated hypocentres and station delays (see the module
    docstring).  With ``velest.reseed.enabled`` the best ``best_percent`` % of the converged
    models (lowest final RMS) seed ``n_rounds`` further rounds; each seed spawns
    ``n_copies`` copies with Gaussian Vp noise ``perturb_vp_km_s`` and runs the full
    schedule again, starting from its own hypocentres and station delays when
    ``inherit_*`` is set.

    Parameters
    ----------
    models
        Trial models of round 1 (unique ids).
    inputs
        ``cnv_path``, ``sta_path`` and ``control`` (e.g. a ``VelestInputs``).
    cfg
        :class:`~osv.config.schema.OSVConfig` or its ``velest`` section.
    root
        Output directory.
    n_jobs
        Parallel workers.
    seed
        Seed of the reseed perturbations (default ``cfg.project.random_seed``).
    timeout, force, compact
        Passed to :func:`~osv.velest.runner.run_jobs`.
    vdamp
        Per-layer ``vdamp`` written into the first-stage model files (later stages inherit
        it).  Default: :func:`resolve_vdamp` of ``cfg`` without geometry (``"auto"`` gives
        ones; pass the resolved array from the stage for a geometry-based choice).

    Returns
    -------
    WorkflowResult
        Run table (compatible with the ensemble analysis), stage history, final models.
    """
    v = _vcfg(cfg)
    stages = resolve_stages(cfg)
    vd = resolve_vdamp(cfg) if vdamp is None else np.asarray(vdamp, float)
    rootp = Path(root)
    rootp.mkdir(parents=True, exist_ok=True)
    base_cnv = Path(inputs.cnv_path)
    nsp3 = inputs.control.nsp == 3
    entries = [
        _Entry(m.id, None, m.id, m, base_cnv, None, vpvs=model_vpvs(m) if nsp3 else None)
        for m in models
    ]
    hist, last = _run_schedule(
        entries, inputs, stages, rootp, 1, n_jobs, timeout, force, compact, vd
    )
    all_last = list(last)
    all_entries = list(entries)
    final_models = _collect_models(entries, inputs, _seed(cfg, seed))

    prev_rows = pd.DataFrame(last)
    if v.reseed.enabled:
        rs = v.reseed
        for rnd in range(2, rs.n_rounds + 2):
            seeds = select_seeds(prev_rows, rs.best_percent)
            if not seeds:
                logger.warning("reseed round %d: no usable models from the previous round", rnd)
                break
            n_copies = rs.n_copies or max(1, round(100.0 / rs.best_percent))
            if rs.perturb_vp_km_s == 0 and n_copies > 1:
                logger.info("perturb_vp_km_s = 0: one copy per seed instead of %d", n_copies)
                n_copies = 1
            rng = np.random.default_rng([_seed(cfg, seed), rnd])
            by_id = {r["id"]: r for r in prev_rows.to_dict("records")}
            new: list[_Entry] = []
            seed_vpvs = {e.id: e.vpvs for e in all_entries}
            for sid in seeds:
                r = by_id[sid]
                parsed = r["parsed"]
                sm: VelestModel = parsed.final_model
                sc = getattr(parsed, "station_corrections", None)
                corr = (
                    sc if rs.inherit_station_corrections and isinstance(sc, pd.DataFrame) else None
                )
                cnv = (
                    Path(r["cnv_path"])
                    if rs.inherit_hypocentres and r.get("cnv_path")
                    else base_cnv
                )
                origin = str(r.get("origin", sid))
                for c in range(1, n_copies + 1):
                    m = (
                        perturb_model(sm, rs.perturb_vp_km_s, rng, monotonic=v.monotonic)
                        if rs.perturb_vp_km_s > 0
                        else sm
                    )
                    new.append(
                        _Entry(
                            f"{sid}_r{rnd}_{c:02d}", sid, origin, m, cnv, corr, vpvs=seed_vpvs[sid]
                        )
                    )
            logger.info("reseed round %d: %d seeds x %d copies", rnd, len(seeds), n_copies)
            h, last = _run_schedule(
                new, inputs, stages, rootp, rnd, n_jobs, timeout, force, compact, vd
            )
            hist += h
            all_last += last
            all_entries += new
            final_models.update(_collect_models(new, inputs, _seed(cfg, seed)))
            prev_rows = pd.DataFrame(last)
            if prev_rows.empty:
                break

    runs = pd.DataFrame(all_last)
    for col in ("origin", "parent"):
        if col not in runs.columns:
            runs[col] = None
    history = pd.DataFrame(hist)
    if len(history):
        history.to_csv(rootp / "workflow_history.csv", index=False)
    return WorkflowResult(runs, history, final_models, rootp, [s.name for s in stages])


def _collect_models(entries: list[_Entry], inputs: InputsLike, seed: int) -> dict[str, TrialModel]:
    out: dict[str, TrialModel] = {}
    for i, e in enumerate(entries):
        r = e.last_row
        if r is not None and r.get("status") == STATUS_OK and isinstance(e.model, VelestModel):
            ratio = e.vpvs if e.vpvs is not None else float(inputs.control.vpvs)
            out[e.id] = _velest_to_trial(e.model, e.id, ratio, seed, i)
    return out


def _model_diff(a: VelestModel, b: VelestModel) -> float:
    d = [np.abs(a.p.layers["vel"].to_numpy() - b.p.layers["vel"].to_numpy())]
    if a.s is not None and b.s is not None and len(a.s.layers) == len(b.s.layers):
        d.append(np.abs(a.s.layers["vel"].to_numpy() - b.s.layers["vel"].to_numpy()))
    return float(max(x.max() for x in d))


def _hypo_diff(prev: pd.DataFrame | None, cur: pd.DataFrame | None) -> float:
    if prev is None or cur is None or not len(prev) or not len(cur):
        return math.nan
    m = prev.merge(cur, on="eq", suffixes=("_a", "_b"))
    if m.empty:
        return math.nan
    d = np.sqrt(sum((m[f"{c}_a"] - m[f"{c}_b"]) ** 2 for c in ("x", "y", "z")))
    return float(np.nanmean(d))


def iterate_final_model(
    model: TrialModel | VelestModel | Path | str,
    inputs: InputsLike,
    cfg: Any,
    root: str | Path,
    *,
    n_jobs: int = 1,
    cnv_path: Path | str | None = None,
    corrections: pd.DataFrame | None = None,
    timeout: float | None = 3600.0,
    force: bool = False,
    vpvs: float | None = None,
    freeze_station_delays: bool | None = None,
    vdamp: float | Sequence[float] | np.ndarray | None = None,
) -> IterationResults:
    """Repeat VELEST from its own outputs until the changes are negligible.

    Runs ``<root>/iterate/cycleNN`` with the damping of the last stage; cycle ``n`` starts
    from cycle ``n-1``'s final model, relocated hypocentres (``velout.cnv``) and station
    delays.  It stops when both the largest layer-velocity change of a cycle (P and S, vs.
    its input model) is below ``iterate_until.model_change_km_s`` and the mean 3-D
    hypocentre shift relative to the previous cycle is below ``hypo_change_km`` (the first
    cycle has no previous hypocentres, so at least two cycles run), or after ``max_cycles``
    or a failed run.

    Parameters
    ----------
    model
        Starting model (``TrialModel``, ``VelestModel`` or ``.mod`` path).
    cnv_path, corrections
        Optional starting hypocentres (CNV) and station delays (a ``station_corrections``
        table) instead of the original inputs.
    vpvs
        Vp/Vs ratio for ``nsp = 3`` runs (default: the ratio of a ``TrialModel``, else the
        control's).
    freeze_station_delays
        Hold the station delays fixed (``nsinv = 0``) at the values handed on from the
        previous cycle.  Default (``None``): frozen for ``nsp = 3`` whenever delays are
        available, inverted otherwise.  With ``nsp = 3`` VELEST applies the S delay
        ``ptcor * vpvs`` but gives S readings no delay partials, so inverting delays while S
        data are used is internally inconsistent: the RMS climbs for tens of iterations,
        successive cycles drift (layers collapse towards ~3 km/s) and some runs end with a
        NaN RMS.  Feeding the delays back is not the problem (the next cycle starts at the
        previous RMS); re-inverting them is.  The delays of the damped stage run are the
        ones estimated; freezing them makes the repeat cycles refine velocities and
        hypocentres only.

    vdamp
        Per-layer ``vdamp`` written into the model files (default: as in the model given).

    Returns
    -------
    IterationResults
        One ``VelestRunResult`` per cycle; ``.history`` holds the convergence table and
        ``.converged`` the verdict.
    """
    v = _vcfg(cfg)
    it = v.iterate_until
    st = resolve_stages(cfg)[-1]
    rootp = Path(root) / "iterate"
    results = IterationResults()
    if isinstance(model, TrialModel):
        cur = _trial_to_velest(model, inputs.control.nmod)
        if vpvs is None and inputs.control.nsp == 3:
            vpvs = model_vpvs(model)
    elif isinstance(model, VelestModel):
        cur = model
    else:
        cur = read_model(model, nmod=inputs.control.nmod)
    if vdamp is not None:
        cur = _with_vdamp(cur, vdamp)
    cnv = Path(cnv_path) if cnv_path is not None else Path(inputs.cnv_path)
    corr = corrections
    prev_hypo: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []
    for c in range(1, it.max_cycles + 1):
        cid = f"cycle{c:02d}"
        inp = rootp / "_inputs"
        sta = (
            _corr_station_file(Path(inputs.sta_path), corr, inp / f"{cid}.sta")
            if corr is not None
            else Path(inputs.sta_path)
        )
        mod = _model_path(cur, inp / f"{cid}.mod", inputs.control.nmod)
        ctrl = _stage_control(inputs.control, st, corr is not None, vpvs)
        freeze = (
            freeze_station_delays
            if freeze_station_delays is not None
            else (corr is not None and ctrl.nsp == 3)
        )
        if freeze:
            ctrl.nsinv = 0
        job = VelestJob(cid, mod, ctrl, cnv, sta)
        df = run_jobs([job], rootp, n_jobs=1, timeout=timeout, force=force, progress=False)
        row = next(df.itertuples())
        if not _usable(row):
            logger.warning("iterate cycle %d failed: %s", c, getattr(row, "error", None))
            rows.append(
                {
                    "cycle": c,
                    "rms": math.nan,
                    "model_change_km_s": math.nan,
                    "hypo_change_km": math.nan,
                    "converged": False,
                    "run_dir": row.run_dir,
                }
            )
            break
        p = row.parsed
        results.append(p)
        dm = _model_diff(cur, p.final_model)
        hypo = getattr(p, "final_hypocentres", None)
        dh = _hypo_diff(prev_hypo, hypo)
        done = bool(dm < it.model_change_km_s and np.isfinite(dh) and dh < it.hypo_change_km)
        rows.append(
            {
                "cycle": c,
                "rms": _rms(p),
                "model_change_km_s": dm,
                "hypo_change_km": dh,
                "converged": done,
                "run_dir": row.run_dir,
            }
        )
        logger.info("iterate cycle %d: rms %.4f dV %.3f dH %.3f", c, _rms(p), dm, dh)
        cur, prev_hypo = p.final_model, hypo
        if row.cnv_path:
            cnv = Path(row.cnv_path)
        sc = getattr(p, "station_corrections", None)
        corr = sc if isinstance(sc, pd.DataFrame) and len(sc) else corr
        if done:
            break
    results.history = pd.DataFrame(
        rows,
        columns=["cycle", "rms", "model_change_km_s", "hypo_change_km", "converged", "run_dir"],
    )
    results.converged = bool(rows and rows[-1]["converged"])
    rms_ok = results.history["rms"].dropna()
    if len(rms_ok):
        results.best_cycle = int(results.history.loc[rms_ok.idxmin(), "cycle"])
        last_rms = float(results.history["rms"].iloc[-1])
        if results.best_cycle != len(results.history) and not (
            last_rms <= float(rms_ok.min()) * 1.01  # within 1 %: not a drift
        ):
            logger.warning(
                "iterate: RMS is lowest in cycle %d, not the last (%d); VELEST returns its "
                "last accepted model, so later cycles drifted - use that cycle's model",
                results.best_cycle,
                len(results.history),
            )
    if rows:
        rootp.mkdir(parents=True, exist_ok=True)
        results.history.to_csv(rootp / "convergence.csv", index=False)
    return results
