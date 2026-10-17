"""Stability tests of the final 1-D model (Stage 2, task 3.5).

The ensemble median model (:mod:`osv.velest.ensemble`) is not itself a model VELEST ever
ran, so the delays and hypocentres of any single trial are not consistent with it.  This
module provides

* :func:`fixed_model_run` -- VELEST with the final model *held fixed*: the canonical source
  of the Stage 2 station delays and hypocentres;
* :func:`hypocentre_perturbation_test` -- random shifts of all starting hypocentres, then
  relocation with velocities fixed and with velocities free;
* :func:`reference_station_test` -- the free inversion repeated with other reference
  stations;
* :func:`lvz_test` -- the free inversion with ``lowveloclay`` on and off;
* :func:`run_stability` -- all of the above for an :class:`~osv.velest.ensemble.EnsembleResult`;
* :class:`StabilityResult` / :func:`write_stability` -- tidy tables and QC findings.

How velocities are held fixed (``fortran/velest/velest.f``)
-----------------------------------------------------------
``SETUNT`` (``velest.f:3984-4017``) scales the velocity columns of the design matrix by
``scale(6) = sqrt(othet / vthet)`` (or 0 for iterations that are not a multiple of
``invertratio``).  A column scale of 0 also removes the *station-delay* columns
(``scale(5)``), so ``invertratio > ittmax`` freezes velocities **and** delays -- not wanted
here, because the fixed run must produce delays.  A very large ``vthet`` shrinks only the
velocity scale: with ``vthet = 1e5`` the velocities of the real data stay within the
two-decimal precision of the model file (``1e3`` still moved one layer by 0.001 km/s), while
station delays and hypocentres remain free.  ``vthet >= 1e8`` makes VELEST's list-directed
read of ``velest.cmn`` fail, hence the config limit.  The default is
``velest.stability.fixed_vthet = 1e5``.  :func:`fixed_model_run` checks the result and
records ``max_velocity_change``.

``lowveloclay`` (``velest.f:4887-4901`` and ``5245-5262``)
-----------------------------------------------------------
With ``lowveloclay = 0`` every velocity update is checked layer by layer: if a layer would
become slower than the layer above, the update is rejected (``dvp = 0``) and the layer is
set to ``vp(layer above) + 0.001`` -- the model is forced to be **non-decreasing with
depth** (low-velocity layers are forbidden).  With ``lowveloclay = 1`` the inversion may
produce low-velocity layers.  VELEST also forces ``lowveloclay = 0`` when reflected phases
are used (``velest.f:1486-1497``).  Shear velocities are not checked.

Reference station
-----------------
In the station file the station with the highest ``icc`` has its P delay held fixed
(see :mod:`osv.data.velest_inputs`).  Moving another station to that position changes the
datum of the delays: all P delays should shift by one constant, ``-(delay of the new reference
in the old solution)``, while velocities stay unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from osv.config.schema import HypocentrePerturbation, StabilityConfig
from osv.data.velest_inputs import choose_reference_station
from osv.io.velest import (
    ModelBlock,
    VelestControl,
    VelestModel,
    read_cnv,
    read_model,
    read_stations,
    write_cnv,
    write_model,
    write_stations,
)
from osv.velest.ensemble import EnsembleResult, Finding
from osv.velest.runner import STATUS_OK, VelestJob, run_jobs

logger = logging.getLogger(__name__)

__all__ = [
    "FixedModelResult",
    "StabilityResult",
    "fixed_model_run",
    "hypocentre_perturbation_test",
    "lvz_test",
    "reference_station_test",
    "run_stability",
    "write_stability",
]

#: km per degree of latitude (same constant as :mod:`osv.velest.ensemble`).
KM_PER_DEG = 111.195

#: ``depth >= zmin + this``: VELEST moves hypocentres at/above ``zmin`` to ``zmin + 0.1``
#: (``velest.f:4385-4387``).
ZMIN_MARGIN_KM = 0.1


class StabilityError(ValueError):
    """A stability test cannot be set up (bad model, missing station, failed reference run)."""


class InputsLike(Protocol):
    """What the tests need from :class:`osv.data.velest_inputs.VelestInputs`."""

    cnv_path: Path
    sta_path: Path
    control: VelestControl


ModelLike = Any  # EnsembleResult | DataFrame | VelestModel | TrialModel | path

# ----------------------------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------------------------

_RUN_COLUMNS = [
    "test",
    "variant",
    "id",
    "status",
    "rms",
    "n_iterations",
    "converged",
    "stop_reason",
    "wall_time",
    "lowveloclay",
    "reference_station",
    "model_key",
    "pert_dh_km",
    "pert_dz_km",
    "dh_mean_km",
    "dh_median_km",
    "dh_p90_km",
    "dz_mean_abs_km",
    "dz_median_abs_km",
    "dz_p90_abs_km",
    "dz_bias_km",
    "max_dv_vs_input",
    "n_low_velocity_layers",
    "run_dir",
]
_EVENT_COLUMNS = [
    "test",
    "variant",
    "id",
    "event",
    "pert_dx_km",
    "pert_dy_km",
    "pert_dz_km",
    "dx_km",
    "dy_km",
    "dh_km",
    "dz_km",
]
_MODEL_COLUMNS = ["test", "variant", "id", "model_key", "layer", "depth_top", "vp", "vs", "hits"]
_LAYER_COLUMNS = [
    "test",
    "variant",
    "layer",
    "depth_top",
    "vp_start",
    "vp_mean",
    "vp_std",
    "vp_min",
    "vp_max",
    "vp_max_abs_dev",
    "vs_std",
    "hits",
]
_DELAY_COLUMNS = [
    "test",
    "id",
    "reference_station",
    "sta",
    "ptcor",
    "stcor",
    "nobs",
    "ptcor_diff",
    "ptcor_resid",
    "stcor_diff",
    "stcor_resid",
]
_SUMMARY_COLUMNS = ["test", "variant", "metric", "value"]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


@dataclass(eq=False)
class FixedModelResult:
    """VELEST run with the final model fixed (canonical delays and hypocentres).

    Attributes
    ----------
    run_id, run_dir, status
        Identity and outcome of the run.
    parsed
        :class:`~osv.io.velest_out.VelestRunResult` (``None`` when the run failed).
    rms
        Final RMS residual (s).
    model
        ``layer, depth_top, vp, vs`` as VELEST reports them after the run.
    max_velocity_change
        Largest ``|v_out - v_in|`` (km/s) over the Vp and Vs layers; should be < 0.01.
    station_delays
        ``sta, ptcor, stcor, nobs`` -- the canonical Stage 2 station delays.
    hypocentres
        ``event, lat, lon, depth_km, rms`` -- the canonical Stage 2 hypocentres.
    cnv_path, sta_path
        Output ``velout.cnv`` / ``velout.sta`` of the run.
    """

    run_id: str
    run_dir: Path
    status: str
    parsed: Any
    rms: float
    model: pd.DataFrame
    max_velocity_change: float
    station_delays: pd.DataFrame
    hypocentres: pd.DataFrame
    cnv_path: Path | None = None
    sta_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.parsed is not None


@dataclass(eq=False)
class StabilityResult:
    """Tables and QC findings of the stability tests.

    Attributes
    ----------
    fixed
        The :class:`FixedModelResult` (``None`` if not run).
    runs
        One row per VELEST run (``test, variant, id, status, rms, n_iterations, converged,
        wall_time, lowveloclay, reference_station, model_key``, the applied perturbation
        ``pert_dh_km, pert_dz_km``, hypocentre mislocation statistics against the unperturbed
        fixed-model solution ``dh_*`` / ``dz_*`` (km), ``max_dv_vs_input`` (largest Vp change
        of the run against the model it started from), ``n_low_velocity_layers``,
        ``run_dir``).
    events
        One row per perturbed run and event: applied offsets ``pert_d*_km`` and the recovered
        mislocation ``dx_km, dy_km, dh_km, dz_km`` (signed east, north, depth; horizontal
        distance).
    models
        Final model of every run (long: ``test, variant, id, model_key, layer, depth_top, vp,
        vs, hits``).
    layers
        Per test and variant and layer: ``vp_start`` (the model the runs started from),
        mean/std/min/max of the final Vp, the largest ``|vp - vp_start|``, ``vs_std`` and
        ``hits`` (ray hits of the baseline run).
    delays
        Reference-station test: per run and station the delays and their difference to the
        baseline run (``*_diff``) and the residual after removing the constant offset
        (``*_resid``).
    summary
        Long table ``test, variant, metric, value``.
    findings
        :class:`~osv.velest.ensemble.Finding` objects.
    """

    fixed: FixedModelResult | None = None
    runs: pd.DataFrame = field(default_factory=lambda: _empty(_RUN_COLUMNS))
    events: pd.DataFrame = field(default_factory=lambda: _empty(_EVENT_COLUMNS))
    models: pd.DataFrame = field(default_factory=lambda: _empty(_MODEL_COLUMNS))
    layers: pd.DataFrame = field(default_factory=lambda: _empty(_LAYER_COLUMNS))
    delays: pd.DataFrame = field(default_factory=lambda: _empty(_DELAY_COLUMNS))
    summary: pd.DataFrame = field(default_factory=lambda: _empty(_SUMMARY_COLUMNS))
    findings: list[Finding] = field(default_factory=list)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "info"]

    def metric(self, test: str, variant: str, name: str) -> float:
        """One value of :attr:`summary` (``nan`` when absent)."""
        s = self.summary
        m = s[(s["test"] == test) & (s["variant"] == variant) & (s["metric"] == name)]
        return float(m["value"].iloc[0]) if len(m) else float("nan")

    def merge(self, other: StabilityResult) -> StabilityResult:
        """Concatenate two results (``fixed`` of ``other`` wins when present)."""

        def cat(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
            parts = [x for x in (a, b) if len(x)]
            return pd.concat(parts, ignore_index=True) if parts else a

        return StabilityResult(
            fixed=other.fixed or self.fixed,
            runs=cat(self.runs, other.runs),
            events=cat(self.events, other.events),
            models=cat(self.models, other.models),
            layers=cat(self.layers, other.layers),
            delays=cat(self.delays, other.delays),
            summary=cat(self.summary, other.summary),
            findings=[*self.findings, *other.findings],
        )

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable summary (``summary.json``)."""
        fx = self.fixed
        return {
            "n_runs": len(self.runs),
            "n_failed": int((self.runs["status"] != STATUS_OK).sum()) if len(self.runs) else 0,
            "fixed_model": None
            if fx is None
            else {
                "run_id": fx.run_id,
                "status": fx.status,
                "rms": _f(fx.rms),
                "max_velocity_change": _f(fx.max_velocity_change),
            },
            "summary": [
                {**r, "value": _f(r["value"])} for r in self.summary.to_dict(orient="records")
            ],
            "findings": [
                {
                    "code": f.code,
                    "severity": f.severity,
                    "message": f.message,
                    "layer": f.layer,
                    "data": _jsonable(f.data),
                }
                for f in self.findings
            ],
        }


def _f(x: Any) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return _f(obj)
    return obj


# ----------------------------------------------------------------------------------------
# Models and parsed outputs
# ----------------------------------------------------------------------------------------


def _as_frame(model: ModelLike, control: VelestControl) -> pd.DataFrame:
    """Normalise a model to ``depth_top, vp, vs`` (Vs from ``control.vpvs`` when absent)."""
    if isinstance(model, EnsembleResult):
        model = model.final_model
    if isinstance(model, str | Path):
        model = read_model(model)
    if isinstance(model, VelestModel):
        p = model.p.layers
        vs = model.s.layers["vel"].to_numpy(float) if model.s is not None else None
        df = pd.DataFrame({"depth_top": p["depth"].to_numpy(float), "vp": p["vel"].to_numpy(float)})
        df["vs"] = vs if vs is not None else np.nan
    elif isinstance(model, pd.DataFrame):
        df = model.copy()
        if "depth_top" not in df and "depth" in df:
            df = df.rename(columns={"depth": "depth_top"})
        if not {"depth_top", "vp"} <= set(df.columns):
            raise StabilityError("model table needs the columns depth_top (or depth) and vp")
        if "vs" not in df:
            df["vs"] = np.nan
        df = df[["depth_top", "vp", "vs"]].astype(float)
    elif hasattr(model, "depths") and hasattr(model, "vp"):  # TrialModel
        df = pd.DataFrame({"depth_top": model.depths, "vp": model.vp, "vs": model.vs})
    else:
        raise StabilityError(f"unsupported model type {type(model).__name__}")
    df = df.reset_index(drop=True).astype(float)
    if control.nsp == 2:
        df["vs"] = df["vs"].where(np.isfinite(df["vs"]), df["vp"] / control.vpvs)
    return df


def _write_model_file(frame: pd.DataFrame, path: Path, vdamp: float = 1.0) -> Path:
    def block(col: str, label: str) -> ModelBlock:
        lay = pd.DataFrame(
            {
                "vel": frame[col].round(2),
                "depth": frame["depth_top"].round(2),
                "vdamp": vdamp,
            }
        )
        return ModelBlock(lay, label=label)

    p = block("vp", "P stability model")
    s = block("vs", "S stability model") if np.isfinite(frame["vs"]).all() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_model(path, VelestModel("osv stability", p, s))


def _model_of(parsed: Any) -> pd.DataFrame | None:
    fm = getattr(parsed, "final_model", None)
    if fm is None:
        return None
    p = fm.p.layers
    df = pd.DataFrame(
        {
            "layer": np.arange(len(p)),
            "depth_top": p["depth"].to_numpy(float),
            "vp": p["vel"].to_numpy(float),
        }
    )
    df["vs"] = fm.s.layers["vel"].to_numpy(float) if fm.s is not None else np.nan
    rs = getattr(parsed, "ray_statistics", None)
    hits = np.full(len(df), np.nan)
    if rs is not None and len(rs):
        h = pd.Series(rs["nhit"].to_numpy(float), index=rs["nlay"].astype(int) - 1)
        hits = h.reindex(range(len(df))).to_numpy()
    df["hits"] = hits
    return df


def _hypocentres_of(parsed: Any) -> pd.DataFrame:
    cnv = getattr(parsed, "cnv", None)
    if cnv is not None and len(cnv.events):
        ev = cnv.events
        out = pd.DataFrame(
            {
                "event": ev["event"].to_numpy(),
                "lat": ev["lat"].to_numpy(float),
                "lon": ev["lon"].to_numpy(float),
                "depth_km": ev["depth_km"].to_numpy(float),
                "rms": ev["rms"].to_numpy(float) if "rms" in ev else np.nan,
            }
        )
        return out
    fh = getattr(parsed, "final_hypocentres", None)
    if fh is not None and len(fh):
        return pd.DataFrame(
            {
                "event": fh["eq"].to_numpy() - 1,
                "lat": fh["lat"].to_numpy(float),
                "lon": fh["lon"].to_numpy(float),
                "depth_km": fh["depth_km"].to_numpy(float),
                "rms": fh["rms"].to_numpy(float),
            }
        )
    return pd.DataFrame(columns=["event", "lat", "lon", "depth_km", "rms"])


def _delays_of(parsed: Any) -> pd.DataFrame:
    sc = getattr(parsed, "station_corrections", None)
    if sc is None or not len(sc):
        return pd.DataFrame(columns=["sta", "ptcor", "stcor", "nobs"])
    return sc[["sta", "ptcor", "stcor", "nobs"]].reset_index(drop=True)


def _rms(parsed: Any) -> float:
    try:
        return float(parsed.final_rms)
    except (AttributeError, TypeError, ValueError):
        return float("nan")


def _usable(row: Any) -> bool:
    return row.status == STATUS_OK and getattr(row, "parsed", None) is not None


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", text)


def _control(inputs: InputsLike, control: VelestControl | None) -> VelestControl:
    return copy.deepcopy(control if control is not None else inputs.control)


def _job(
    rid: str,
    model_path: Path,
    ctrl: VelestControl,
    cnv: Path | str,
    sta: Path | str,
) -> VelestJob:
    return VelestJob(rid, model_path, ctrl, cnv, sta)


def _run(jobs: list[VelestJob], root: Path, n_jobs: int, timeout: float | None) -> pd.DataFrame:
    # always re-run: the inputs (perturbations, model) may have changed since a previous call
    return run_jobs(jobs, root, n_jobs=n_jobs, timeout=timeout, progress=False, force=True)


def _row_of(table: pd.DataFrame, rid: str) -> Any:
    return next(table[table["id"] == rid].itertuples())


def _dv(a: pd.DataFrame, b: pd.DataFrame, col: str = "vp") -> float:
    if len(a) != len(b):
        return float("nan")
    return float(np.max(np.abs(a[col].to_numpy(float) - b[col].to_numpy(float))))


def _n_lvz(vp: np.ndarray) -> int:
    return int((np.diff(vp) < -1e-9).sum())


# ----------------------------------------------------------------------------------------
# 1. Fixed-model run
# ----------------------------------------------------------------------------------------


def fixed_model_run(
    final_model: ModelLike,
    inputs: InputsLike,
    control: VelestControl | None,
    root: str | Path,
    *,
    vthet: float | None = None,
    cfg: StabilityConfig | None = None,
    run_id: str = "fixed",
    cnv_path: Path | str | None = None,
    timeout: float | None = 3600.0,
) -> FixedModelResult:
    """Run VELEST with ``final_model`` held fixed.

    Velocities are frozen with a very large ``vthet`` (module docstring); hypocentres and
    station delays are inverted as usual, so the result is the consistent final set of
    delays and hypocentres for the final model.  ``control`` (default ``inputs.control``)
    supplies everything else (iterations, damping of the other parameters, ``nsinv``).

    Parameters
    ----------
    final_model
        ``EnsembleResult``, table with ``depth_top, vp[, vs]``, ``VelestModel``,
        ``TrialModel`` or a ``.mod`` path.
    vthet
        Overrides ``cfg.fixed_vthet`` (default ``velest.stability.fixed_vthet``).
    cnv_path
        Alternative input CNV (default ``inputs.cnv_path``).

    Returns
    -------
    FixedModelResult
        ``ok`` is False (and ``parsed`` None) when the run failed.
    """
    scfg = cfg or StabilityConfig()
    ctrl = _control(inputs, control)
    ctrl.vthet = float(vthet if vthet is not None else scfg.fixed_vthet)
    frame = _as_frame(final_model, ctrl)
    root_p = Path(root)
    mpath = _write_model_file(frame, root_p / "_inputs" / f"{run_id}.mod")
    job = _job(run_id, mpath, ctrl, cnv_path or inputs.cnv_path, inputs.sta_path)
    table = _run([job], root_p, 1, timeout)
    row = _row_of(table, run_id)
    parsed = row.parsed if _usable(row) else None
    if parsed is None:
        logger.warning("fixed-model run failed: %s", row.error)
        return FixedModelResult(
            run_id,
            Path(row.run_dir),
            row.status,
            None,
            float("nan"),
            pd.DataFrame(columns=["layer", "depth_top", "vp", "vs", "hits"]),
            float("nan"),
            pd.DataFrame(columns=["sta", "ptcor", "stcor", "nobs"]),
            pd.DataFrame(columns=["event", "lat", "lon", "depth_km", "rms"]),
        )
    model = _model_of(parsed)
    assert model is not None
    dv = max(_dv(model, frame, "vp"), _dv(model, frame, "vs") if np.isfinite(model.vs).all() else 0)
    if not dv < 0.011:  # model files hold two decimals
        logger.warning("fixed-model run changed the velocities by up to %.3f km/s", dv)
    return FixedModelResult(
        run_id,
        Path(row.run_dir),
        row.status,
        parsed,
        _rms(parsed),
        model,
        dv,
        _delays_of(parsed),
        _hypocentres_of(parsed),
        Path(row.cnv_path) if row.cnv_path else None,
        Path(row.sta_path) if row.sta_path else None,
    )


# ----------------------------------------------------------------------------------------
# Shared bookkeeping
# ----------------------------------------------------------------------------------------


def _mislocation(out: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame | None:
    """Per-event mislocation of ``out`` against ``ref`` (merged on ``event``)."""
    if out.empty or ref.empty:
        return None
    m = out.merge(ref, on="event", suffixes=("", "_ref"))
    if m.empty:
        return None
    coslat = np.cos(np.radians(m["lat_ref"]))
    dx = (m["lon"] - m["lon_ref"]) * KM_PER_DEG * coslat
    dy = (m["lat"] - m["lat_ref"]) * KM_PER_DEG
    return pd.DataFrame(
        {
            "event": m["event"].to_numpy(),
            "dx_km": dx.to_numpy(),
            "dy_km": dy.to_numpy(),
            "dh_km": np.hypot(dx, dy).to_numpy(),
            "dz_km": (m["depth_km"] - m["depth_km_ref"]).to_numpy(),
        }
    )


def _run_record(
    test: str,
    variant: str,
    row: Any,
    start: pd.DataFrame,
    **extra: Any,
) -> dict[str, Any]:
    rec: dict[str, Any] = dict.fromkeys(_RUN_COLUMNS, np.nan)
    rec.update(
        test=test,
        variant=variant,
        id=row.id,
        status=row.status,
        wall_time=row.wall_time,
        run_dir=row.run_dir,
        converged=False,
    )
    if _usable(row):
        p = row.parsed
        model = _model_of(p)
        rec.update(
            rms=_rms(p),
            n_iterations=getattr(p, "n_iterations", np.nan),
            converged=bool(getattr(p, "converged", False)),
            stop_reason=getattr(p, "stop_reason", None),
        )
        if model is not None:
            rec["max_dv_vs_input"] = _dv(model, start, "vp")
            rec["n_low_velocity_layers"] = _n_lvz(model["vp"].to_numpy(float))
    rec.update(extra)
    return rec


def _model_rows(test: str, variant: str, rid: str, key: str, parsed: Any) -> pd.DataFrame | None:
    m = _model_of(parsed)
    if m is None:
        return None
    m.insert(0, "model_key", key)
    m.insert(0, "id", rid)
    m.insert(0, "variant", variant)
    m.insert(0, "test", test)
    return m[_MODEL_COLUMNS]


def _layer_summary(
    models: pd.DataFrame, start: pd.DataFrame, hits: pd.Series | None = None
) -> pd.DataFrame:
    """Per test/variant/layer statistics of the final Vp over the runs in ``models``."""
    rows = []
    for (test, variant), g in models.groupby(["test", "variant"], sort=False):
        for layer, gl in g.groupby("layer"):
            vp = gl["vp"].to_numpy(float)
            vs = gl["vs"].to_numpy(float)
            v0 = float(start["vp"].iloc[int(layer)])
            rows.append(
                {
                    "test": test,
                    "variant": variant,
                    "layer": int(layer),
                    "depth_top": float(gl["depth_top"].iloc[0]),
                    "vp_start": v0,
                    "vp_mean": float(vp.mean()),
                    "vp_std": float(vp.std(ddof=1)) if len(vp) > 1 else 0.0,
                    "vp_min": float(vp.min()),
                    "vp_max": float(vp.max()),
                    "vp_max_abs_dev": float(np.abs(vp - v0).max()),
                    "vs_std": float(np.nanstd(vs, ddof=1)) if np.isfinite(vs).sum() > 1 else np.nan,
                    "hits": float(hits.get(int(layer), np.nan)) if hits is not None else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=_LAYER_COLUMNS) if rows else _empty(_LAYER_COLUMNS)


def _hits_of(parsed: Any) -> pd.Series | None:
    m = _model_of(parsed) if parsed is not None else None
    if m is None:
        return None
    return pd.Series(m["hits"].to_numpy(float), index=m["layer"].to_numpy())


def _summary_rows(test: str, variant: str, metrics: Mapping[str, float]) -> list[dict[str, Any]]:
    return [
        {"test": test, "variant": variant, "metric": k, "value": float(v)}
        for k, v in metrics.items()
    ]


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns) if rows else _empty(columns)


# ----------------------------------------------------------------------------------------
# 2. Hypocentre perturbation
# ----------------------------------------------------------------------------------------


def _perturb_events(
    events: pd.DataFrame,
    pert: HypocentrePerturbation,
    rng: np.random.Generator,
    depth_floor: float,
) -> pd.DataFrame:
    """Events with random offsets; adds ``pert_dx_km, pert_dy_km, pert_dz_km`` (applied)."""
    n = len(events)
    if pert.distribution == "gaussian":
        dx, dy = rng.normal(0.0, pert.xy, n), rng.normal(0.0, pert.xy, n)
        dz = rng.normal(0.0, pert.z, n)
    else:
        dx, dy = rng.uniform(-pert.xy, pert.xy, n), rng.uniform(-pert.xy, pert.xy, n)
        dz = rng.uniform(-pert.z, pert.z, n)
    out = events.copy()
    depth0 = events["depth_km"].to_numpy(float)
    # never push an event above the model top / zmin; events already shallower stay put
    depth1 = np.where(depth0 < depth_floor, depth0, np.maximum(depth0 + dz, depth_floor))
    lat0 = events["lat"].to_numpy(float)
    out["lat"] = lat0 + dy / KM_PER_DEG
    out["lon"] = events["lon"].to_numpy(float) + dx / (KM_PER_DEG * np.cos(np.radians(lat0)))
    out["depth_km"] = depth1
    out["pert_dx_km"], out["pert_dy_km"], out["pert_dz_km"] = dx, dy, depth1 - depth0
    return out


def hypocentre_perturbation_test(
    final_model: ModelLike,
    inputs: InputsLike,
    control: VelestControl | None,
    cfg: Any,
    root: str | Path,
    seed: int = 0,
    *,
    reference: FixedModelResult | None = None,
    n_tests: int | None = None,
    n_jobs: int = -1,
    timeout: float | None = 3600.0,
) -> StabilityResult:
    """Perturb all starting hypocentres randomly and relocate (fixed and free velocities).

    Each of ``n_tests`` perturbations shifts every event of ``inputs.cnv_path`` by random
    offsets (``cfg.velest.hypocentre_perturbation_km``: ``xy`` per horizontal axis, ``z`` in
    depth, uniform or Gaussian; depth kept at or below ``zmin + 0.1`` km and the model
    top; origin times are kept, VELEST adjusts them).  Two runs per perturbation:

    * variant ``fixed``: velocities fixed (:func:`fixed_model_run` mechanism) -- measures
      how well the hypocentres are recovered for the final model;
    * variant ``free``: velocities free with the control's damping -- also measures the
      stability of the velocity model.

    A baseline free run from the unperturbed hypocentres (variant ``baseline_free``)
    provides the layer hit counts and the reference for the velocity changes.  The
    mislocation of every run is measured against the unperturbed fixed-model solution
    (``reference``; computed with :func:`fixed_model_run` when omitted).

    Parameters
    ----------
    cfg
        ``OSVConfig``, ``VelestConfig`` or :class:`~osv.config.schema.StabilityConfig`
        holder providing ``hypocentre_perturbation_km`` and the stability section.
    seed
        Random seed (Generator); the same seed gives the same perturbations.
    n_tests
        Overrides ``hypocentre_perturbation_km.n_tests``.

    Returns
    -------
    StabilityResult
        Tests ``hypocentre_perturbation``; summary metrics per variant: ``dh_mean_km``,
        ``dh_median_km``, ``dh_p90_km``, ``dz_mean_abs_km``, ``dz_median_abs_km``,
        ``dz_p90_abs_km`` (all over all events of all runs), ``rms_mean``, ``rms_std``,
        ``n_ok``; and for the free variant ``vp_spread_max`` (largest per-layer std) and
        ``vp_dev_max`` (largest deviation from the start model).
    """
    vcfg = getattr(cfg, "velest", cfg)
    pert: HypocentrePerturbation = vcfg.hypocentre_perturbation_km
    scfg: StabilityConfig = vcfg.stability
    n = pert.n_tests if n_tests is None else n_tests
    ctrl = _control(inputs, control)
    root_p = Path(root)
    test = "hypocentre_perturbation"
    start = _as_frame(final_model, ctrl)
    if reference is None:
        reference = fixed_model_run(
            final_model, inputs, ctrl, root_p / "_reference", cfg=scfg, timeout=timeout
        )
    if not reference.ok:
        raise StabilityError(
            f"the unperturbed fixed-model run failed ({reference.status}); see {reference.run_dir}"
        )

    cnv = read_cnv(inputs.cnv_path)
    floor = max(float(start["depth_top"].iloc[0]), ctrl.zmin + ZMIN_MARGIN_KM)
    rng = np.random.default_rng(seed)
    inp_dir = root_p / "_inputs"
    inp_dir.mkdir(parents=True, exist_ok=True)
    mpath = _write_model_file(start, inp_dir / "start.mod")

    fixed_ctrl = copy.deepcopy(ctrl)
    fixed_ctrl.vthet = scfg.fixed_vthet
    jobs = [_job("hypo_base_free", mpath, ctrl, inputs.cnv_path, inputs.sta_path)]
    applied: dict[str, pd.DataFrame] = {}
    meta: dict[str, tuple[str, int]] = {"hypo_base_free": ("baseline_free", -1)}
    cnv_name = ctrl.phasefile or Path(inputs.cnv_path).name
    for k in range(n):
        ev = _perturb_events(cnv.events, pert, rng, floor)
        p = inp_dir / f"perturbed_{k:03d}" / cnv_name
        p.parent.mkdir(parents=True, exist_ok=True)
        write_cnv(p, ev.drop(columns=["pert_dx_km", "pert_dy_km", "pert_dz_km"]), cnv.picks)
        for variant, c in (("fixed", fixed_ctrl), ("free", ctrl)):
            rid = f"hypo_{variant}_{k:03d}"
            jobs.append(_job(rid, mpath, c, p, inputs.sta_path))
            applied[rid] = ev
            meta[rid] = (variant, k)
    table = _run(jobs, root_p, n_jobs, timeout)

    run_rows: list[dict[str, Any]] = []
    ev_rows: list[pd.DataFrame] = []
    model_rows: list[pd.DataFrame] = []
    hits: pd.Series | None = None
    for row in table.itertuples():
        variant, _k = meta[row.id]
        extra: dict[str, Any] = {"lowveloclay": ctrl.lowveloclay, "model_key": "final"}
        if row.id in applied:
            a = applied[row.id]
            extra["pert_dh_km"] = float(np.hypot(a["pert_dx_km"], a["pert_dy_km"]).mean())
            extra["pert_dz_km"] = float(a["pert_dz_km"].abs().mean())
        rec = _run_record(test, variant, row, start, **extra)
        if _usable(row):
            if row.id == "hypo_base_free":
                hits = _hits_of(row.parsed)
            mr = _model_rows(test, variant, row.id, "final", row.parsed)
            if mr is not None:
                model_rows.append(mr)
            ml = _mislocation(_hypocentres_of(row.parsed), reference.hypocentres)
            if ml is not None and row.id in applied:
                a = applied[row.id]
                ml = ml.merge(
                    a[["event", "pert_dx_km", "pert_dy_km", "pert_dz_km"]], on="event", how="left"
                )
                ml.insert(0, "id", row.id)
                ml.insert(0, "variant", variant)
                ml.insert(0, "test", test)
                ev_rows.append(ml[_EVENT_COLUMNS])
            if ml is not None:
                rec.update(
                    dh_mean_km=ml["dh_km"].mean(),
                    dh_median_km=ml["dh_km"].median(),
                    dh_p90_km=ml["dh_km"].quantile(0.9),
                    dz_mean_abs_km=ml["dz_km"].abs().mean(),
                    dz_median_abs_km=ml["dz_km"].abs().median(),
                    dz_p90_abs_km=ml["dz_km"].abs().quantile(0.9),
                    dz_bias_km=ml["dz_km"].mean(),
                )
        run_rows.append(rec)

    runs = _frame(run_rows, _RUN_COLUMNS)
    events = pd.concat(ev_rows, ignore_index=True) if ev_rows else _empty(_EVENT_COLUMNS)
    models = pd.concat(model_rows, ignore_index=True) if model_rows else _empty(_MODEL_COLUMNS)
    pm = models[models["variant"].isin(["fixed", "free"])]
    layers = _layer_summary(pm, start, hits)

    summary: list[dict[str, Any]] = []
    for variant in ("fixed", "free"):
        ev = events[events["variant"] == variant]
        rr = runs[(runs["variant"] == variant) & (runs["status"] == STATUS_OK)]
        m: dict[str, float] = {"n_ok": float(len(rr)), "n_runs": float(n)}
        if len(ev):
            dh, dz = ev["dh_km"].astype(float), ev["dz_km"].astype(float).abs()
            m.update(
                dh_mean_km=dh.mean(),
                dh_median_km=dh.median(),
                dh_p90_km=dh.quantile(0.9),
                dz_mean_abs_km=dz.mean(),
                dz_median_abs_km=dz.median(),
                dz_p90_abs_km=dz.quantile(0.9),
                dz_bias_km=float(ev["dz_km"].astype(float).mean()),
            )
        if len(rr):
            rms = rr["rms"].astype(float)
            m.update(rms_mean=rms.mean(), rms_std=rms.std(ddof=1) if len(rr) > 1 else 0.0)
        lay = layers[layers["variant"] == variant]
        if variant == "free" and len(lay):
            m.update(vp_spread_max=lay["vp_std"].max(), vp_dev_max=lay["vp_max_abs_dev"].max())
            bm = models[models["variant"] == "baseline_free"].sort_values("layer")
            fr = models[models["variant"] == "free"]
            if len(bm) == len(lay):
                bvp = bm["vp"].to_numpy(float)
                dev = [
                    np.abs(g.sort_values("layer")["vp"].to_numpy(float) - bvp).max()
                    for _, g in fr.groupby("id")
                ]
                m["vp_dev_baseline_max"] = float(np.max(dev))
        summary += _summary_rows(test, variant, m)
    base = runs[runs["variant"] == "baseline_free"]
    if len(base) and np.isfinite(base["rms"].astype(float)).any():
        summary += _summary_rows(
            test,
            "baseline_free",
            {
                "rms": float(base["rms"].iloc[0]),
                "max_dv_vs_input": float(base["max_dv_vs_input"].iloc[0]),
            },
        )
    summary += _summary_rows(test, "reference_fixed", {"rms": reference.rms})
    out = StabilityResult(
        runs=runs,
        events=events,
        models=models,
        layers=layers,
        summary=_frame(summary, _SUMMARY_COLUMNS),
    )
    out.findings = _hypo_findings(out, scfg, n) + _convergence_findings(runs, test)
    return out


def _convergence_findings(runs: pd.DataFrame, test: str) -> list[Finding]:
    """Warn about runs VELEST abandoned after four step-length backups (not comparable)."""
    if runs.empty or "stop_reason" not in runs:
        return []
    bad = runs[runs["stop_reason"] == "four_backups"]
    if bad.empty:
        return []
    ids = sorted(bad["id"])
    return [
        Finding(
            "stability_runs_stopped_early",
            "warn",
            f"{test}: {len(ids)} runs stopped after four backups (RMS rose), so they end at "
            f"different iterations and their models are only roughly comparable: {ids[:5]}.",
            data={"test": test, "ids": ids},
        )
    ]


def _hypo_findings(res: StabilityResult, scfg: StabilityConfig, n: int) -> list[Finding]:
    f: list[Finding] = []
    test = "hypocentre_perturbation"
    n_failed = int((res.runs["status"] != STATUS_OK).sum())
    if n_failed:
        f.append(
            Finding(
                "stability_runs_failed",
                "warn",
                f"{n_failed} of {len(res.runs)} hypocentre-perturbation runs failed.",
                data={"test": test, "n_failed": n_failed},
            )
        )
    for variant in ("fixed", "free"):
        dz = res.metric(test, variant, "dz_mean_abs_km")
        dh = res.metric(test, variant, "dh_mean_km")
        if np.isfinite(dz) and dz > scfg.vertical_mislocation_warn_km:
            f.append(
                Finding(
                    "hypocentre_vertical_mislocation",
                    "warn",
                    f"mean vertical mislocation {dz:.2f} km after perturbation ({variant} "
                    f"velocities) exceeds {scfg.vertical_mislocation_warn_km} km: depths are "
                    "poorly constrained.",
                    data={"variant": variant, "value": dz},
                )
            )
        if np.isfinite(dh) and dh > scfg.horizontal_mislocation_warn_km:
            f.append(
                Finding(
                    "hypocentre_horizontal_mislocation",
                    "warn",
                    f"mean horizontal mislocation {dh:.2f} km after perturbation ({variant} "
                    f"velocities) exceeds {scfg.horizontal_mislocation_warn_km} km.",
                    data={"variant": variant, "value": dh},
                )
            )
    lay = res.layers[res.layers["variant"] == "free"]
    for r in lay.itertuples():
        hit = bool(np.isfinite(r.hits) and r.hits > 0)
        if hit and r.vp_std > scfg.velocity_spread_warn_km_s:
            f.append(
                Finding(
                    "velocity_unstable_hypocentres",
                    "warn",
                    f"layer {r.layer}: Vp spread {r.vp_std:.2f} km/s over the perturbed free "
                    f"runs exceeds {scfg.velocity_spread_warn_km_s} km/s "
                    f"({int(r.hits)} ray hits).",
                    layer=int(r.layer),
                    data={"vp_std": r.vp_std, "hits": r.hits},
                )
            )
    f.append(
        Finding(
            "hypocentre_perturbation_done",
            "info",
            f"{n} hypocentre perturbations, fixed and free velocities.",
            data={"n_tests": n},
        )
    )
    return f


# ----------------------------------------------------------------------------------------
# 3. Reference-station swap
# ----------------------------------------------------------------------------------------


def _reference_of(sta: pd.DataFrame) -> str:
    cand = sta[sta["icc"] > 0]
    if cand.empty:
        raise StabilityError("the station file has no station with icc > 0")
    return str(cand.loc[cand["icc"].idxmax(), "sta"])


def _with_reference(sta: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Station table with ``ref`` last among the ``icc > 0`` stations (highest icc)."""
    sta = sta.copy()
    use = sta[sta["icc"] > 0]
    if ref not in set(use["sta"]):
        raise StabilityError(
            f"reference station {ref!r} is not a station-delay station of the station file; "
            f"candidates: {sorted(use['sta'])[:10]}..."
        )
    order = [*[s for s in use["sta"] if s != ref], ref]
    icc = {s: i + 1 for i, s in enumerate(order)}
    sta["icc"] = [icc.get(s, 0) for s in sta["sta"]]
    return sta


def _auto_references(
    sta: pd.DataFrame, cnv_picks: pd.DataFrame, original: str, n: int
) -> list[str]:
    """Well-recorded stations nearest the network centre, excluding ``original``."""
    if n <= 0:
        return []
    use = sta[sta["icc"] > 0]
    counts = cnv_picks[cnv_picks["weight"] < 4].groupby("sta").size()
    chosen: list[str] = []
    excluded = {original}
    table = use.rename(columns={"sta": "alias"})
    for _ in range(n):
        npk = counts.reindex(table["alias"]).fillna(0).copy()
        npk[npk.index.isin(excluded)] = 0
        if not (npk > 0).any():
            break
        pick = choose_reference_station(table, npk)
        chosen.append(pick)
        excluded.add(pick)
    return chosen


def reference_station_test(
    final_model: ModelLike,
    inputs: InputsLike,
    control: VelestControl | None,
    cfg: Any,
    root: str | Path,
    seed: int = 0,
    *,
    reference_stations: Sequence[str] | None = None,
    n_jobs: int = -1,
    timeout: float | None = 3600.0,
) -> StabilityResult:
    """Repeat the free inversion with other reference stations.

    The baseline run uses the original reference station (highest ``icc`` of
    ``inputs.sta_path``); each alternative gets a station file in which that station has the
    highest ``icc``.  All runs start from ``final_model`` and the same hypocentres, with
    free velocities and the control's iteration settings (give a ``control`` with a small
    ``ittmax`` for a short test).  Alternatives default to
    ``velest.stability.reference_stations`` or, when empty, the
    ``n_reference_stations`` best recorded stations nearest the network centre.

    Returns
    -------
    StabilityResult
        Test ``reference_station``: ``runs`` (variant ``baseline`` / ``swap``,
        ``reference_station``), ``models``, ``layers`` (per swap variant: Vp spread over all
        reference choices), ``delays`` (differences to the baseline and residuals after
        removing the constant offset) and summary metrics for variant ``all``:
        ``vp_dev_max`` (largest Vp change against the baseline in any layer and run),
        ``ptcor_resid_std_max`` / ``stcor_resid_std_max`` (s), ``rms_range``.
    """
    del seed  # deterministic; kept for a uniform signature
    vcfg = getattr(cfg, "velest", cfg)
    scfg: StabilityConfig = vcfg.stability
    ctrl = _control(inputs, control)
    root_p = Path(root)
    test = "reference_station"
    start = _as_frame(final_model, ctrl)
    sta = read_stations(inputs.sta_path)
    original = _reference_of(sta)
    alts = list(reference_stations if reference_stations else scfg.reference_stations)
    if not alts:
        alts = _auto_references(
            sta, read_cnv(inputs.cnv_path).picks, original, scfg.n_reference_stations
        )
    alts = [a for a in dict.fromkeys(alts) if a != original]
    inp_dir = root_p / "_inputs"
    mpath = _write_model_file(start, inp_dir / "start.mod")
    sta_name = ctrl.stationfile or Path(inputs.sta_path).name

    jobs = [_job("ref_baseline", mpath, ctrl, inputs.cnv_path, inputs.sta_path)]
    meta = {"ref_baseline": ("baseline", original)}
    for a in alts:
        rid = f"ref_{_safe_id(a)}"
        p = inp_dir / rid / sta_name
        p.parent.mkdir(parents=True, exist_ok=True)
        write_stations(p, _with_reference(sta, a))
        jobs.append(_job(rid, mpath, ctrl, inputs.cnv_path, p))
        meta[rid] = ("swap", a)
    table = _run(jobs, root_p, n_jobs, timeout)

    base_row = _row_of(table, "ref_baseline")
    if not _usable(base_row):
        raise StabilityError(f"baseline run failed ({base_row.status}); see {base_row.run_dir}")
    base_model = _model_of(base_row.parsed)
    base_delay = _delays_of(base_row.parsed).set_index("sta")
    base_hyp = _hypocentres_of(base_row.parsed)
    hits = _hits_of(base_row.parsed)

    runs_rows, model_rows, delay_rows = [], [], []
    expected: dict[str, float] = {}
    for row in table.itertuples():
        variant, ref = meta[row.id]
        rec = _run_record(
            test, variant, row, start, lowveloclay=ctrl.lowveloclay, reference_station=ref,
            model_key="final",
        )  # fmt: skip
        if _usable(row):
            mr = _model_rows(test, variant, row.id, "final", row.parsed)
            if mr is not None:
                mr["variant"] = f"ref_{_safe_id(ref)}" if variant == "swap" else "baseline"
                model_rows.append(mr)
            ml = _mislocation(_hypocentres_of(row.parsed), base_hyp)
            if ml is not None:
                rec.update(
                    dh_mean_km=ml["dh_km"].mean(),
                    dh_median_km=ml["dh_km"].median(),
                    dh_p90_km=ml["dh_km"].quantile(0.9),
                    dz_mean_abs_km=ml["dz_km"].abs().mean(),
                    dz_median_abs_km=ml["dz_km"].abs().median(),
                    dz_p90_abs_km=ml["dz_km"].abs().quantile(0.9),
                    dz_bias_km=ml["dz_km"].mean(),
                )
            d = _delays_of(row.parsed).set_index("sta")
            common = d.index.intersection(base_delay.index)
            if len(common) and row.id != "ref_baseline":
                dp = (d.loc[common, "ptcor"] - base_delay.loc[common, "ptcor"]).astype(float)
                ds = (d.loc[common, "stcor"] - base_delay.loc[common, "stcor"]).astype(float)
                expected[ref] = (
                    -float(base_delay.loc[ref, "ptcor"]) if ref in base_delay.index else np.nan
                )
                delay_rows.append(
                    pd.DataFrame(
                        {
                            "test": test,
                            "id": row.id,
                            "reference_station": ref,
                            "sta": common,
                            "ptcor": d.loc[common, "ptcor"].to_numpy(),
                            "stcor": d.loc[common, "stcor"].to_numpy(),
                            "nobs": d.loc[common, "nobs"].to_numpy(),
                            "ptcor_diff": dp.to_numpy(),
                            "ptcor_resid": (dp - dp.median()).to_numpy(),
                            "stcor_diff": ds.to_numpy(),
                            "stcor_resid": (ds - ds.median()).to_numpy(),
                        }
                    )[_DELAY_COLUMNS]
                )
        runs_rows.append(rec)

    runs = _frame(runs_rows, _RUN_COLUMNS)
    models = pd.concat(model_rows, ignore_index=True) if model_rows else _empty(_MODEL_COLUMNS)
    delays = pd.concat(delay_rows, ignore_index=True) if delay_rows else _empty(_DELAY_COLUMNS)
    swaps = models[models["variant"] != "baseline"]
    layers = _layer_summary(models.assign(variant="all"), start, hits)
    dv_rows = []
    for rid, g in swaps.groupby("id"):
        dv_rows.append(
            {
                "id": rid,
                "dv": np.abs(g["vp"].to_numpy(float) - base_model["vp"].to_numpy(float)).max()
                if len(g) == len(base_model)
                else np.nan,
            }
        )
    metrics: dict[str, float] = {"n_alternatives": float(len(alts))}
    if dv_rows:
        metrics["vp_dev_max"] = float(np.nanmax([r["dv"] for r in dv_rows]))
    if len(delays):
        metrics["ptcor_resid_std_max"] = float(delays.groupby("id")["ptcor_resid"].std().max())
        metrics["stcor_resid_std_max"] = float(delays.groupby("id")["stcor_resid"].std().max())
        off = delays.groupby("reference_station")["ptcor_diff"].median()
        err = [off[r] - e for r, e in expected.items() if r in off.index and np.isfinite(e)]
        if err:
            metrics["ptcor_offset_error_max"] = float(np.max(np.abs(err)))
    rr = runs[runs["status"] == STATUS_OK]["rms"].astype(float)
    if len(rr):
        metrics["rms_range"] = float(rr.max() - rr.min())
    out = StabilityResult(
        runs=runs,
        models=models,
        layers=layers,
        delays=delays,
        summary=_frame(_summary_rows(test, "all", metrics), _SUMMARY_COLUMNS),
    )
    out.findings = _ref_findings(out, scfg, alts, original, base_model, dv_rows, hits)
    out.findings += _convergence_findings(runs, test)
    return out


def _ref_findings(
    res: StabilityResult,
    scfg: StabilityConfig,
    alts: list[str],
    original: str,
    base: pd.DataFrame,
    dv_rows: list[dict[str, Any]],
    hits: pd.Series | None,
) -> list[Finding]:
    f: list[Finding] = []
    bad = res.runs[(res.runs["status"] != STATUS_OK)]
    if len(bad):
        f.append(
            Finding(
                "stability_runs_failed",
                "warn",
                f"{len(bad)} reference-station runs failed: {sorted(bad['id'])}.",
                data={"ids": sorted(bad["id"])},
            )
        )
    for rid, g in res.models[res.models["variant"] != "baseline"].groupby("id"):
        if len(g) != len(base):
            continue
        dv = np.abs(g["vp"].to_numpy(float) - base["vp"].to_numpy(float))
        for layer in np.flatnonzero(dv > scfg.reference_swap_dv_warn_km_s):
            hit = hits.get(int(layer), np.nan) if hits is not None else np.nan
            f.append(
                Finding(
                    "reference_swap_velocity_change",
                    "warn",
                    f"{rid}: layer {layer} Vp changes by {dv[layer]:.2f} km/s against reference "
                    f"{original} (limit {scfg.reference_swap_dv_warn_km_s} km/s).",
                    layer=int(layer),
                    data={"id": rid, "dv": float(dv[layer]), "hits": _f(hit)},
                )
            )
    f.append(
        Finding(
            "reference_swap_done",
            "info",
            f"reference station swapped from {original} to {alts}.",
            data={"original": original, "alternatives": alts},
        )
    )
    return f


# ----------------------------------------------------------------------------------------
# 4. Low-velocity-layer switch
# ----------------------------------------------------------------------------------------


def lvz_test(
    final_model: ModelLike,
    inputs: InputsLike,
    control: VelestControl | None,
    cfg: Any,
    root: str | Path,
    seed: int = 0,
    *,
    extra_models: Mapping[str, ModelLike] | None = None,
    n_jobs: int = -1,
    timeout: float | None = 3600.0,
) -> StabilityResult:
    """Free inversion with ``lowveloclay`` 0 and 1 for the final and any extra models.

    ``lowveloclay = 0`` forbids layers slower than the layer above (module docstring);
    ``1`` allows them.  The models are the final model (key ``final``) plus
    ``extra_models`` (e.g. the best and the closest ensemble run).  Results: ``runs`` with
    ``lowveloclay`` and ``n_low_velocity_layers``, ``models`` (``model_key`` plus variant
    ``lvz0`` / ``lvz1``) and summary metrics per model key (``test = "lvz"``, variant =
    key): ``rms_lvz0``, ``rms_lvz1``, ``rms_gain`` (lvz0 - lvz1; positive when allowing LVZ
    fits better), ``vp_dev_max`` (largest Vp difference between the two) and
    ``n_lvz_layers`` of the lvz1 model.
    """
    del seed
    vcfg = getattr(cfg, "velest", cfg)
    scfg: StabilityConfig = vcfg.stability
    ctrl = _control(inputs, control)
    root_p = Path(root)
    test = "lvz"
    models_in = {"final": final_model, **(dict(extra_models) if extra_models else {})}
    inp_dir = root_p / "_inputs"
    jobs, meta, starts = [], {}, {}
    for key, m in models_in.items():
        frame = _as_frame(m, ctrl)
        starts[key] = frame
        mpath = _write_model_file(frame, inp_dir / f"{_safe_id(key)}.mod")
        for lvz in (0, 1):
            c = copy.deepcopy(ctrl)
            c.lowveloclay = lvz
            rid = f"lvz_{_safe_id(key)}_{lvz}"
            jobs.append(_job(rid, mpath, c, inputs.cnv_path, inputs.sta_path))
            meta[rid] = (key, lvz)
    table = _run(jobs, root_p, n_jobs, timeout)

    run_rows, model_rows = [], []
    parsed_of: dict[tuple[str, int], Any] = {}
    for row in table.itertuples():
        key, lvz = meta[row.id]
        rec = _run_record(test, f"lvz{lvz}", row, starts[key], lowveloclay=lvz, model_key=key)
        run_rows.append(rec)
        if _usable(row):
            parsed_of[(key, lvz)] = row.parsed
            mr = _model_rows(test, f"lvz{lvz}", row.id, key, row.parsed)
            if mr is not None:
                model_rows.append(mr)
    runs = _frame(run_rows, _RUN_COLUMNS)
    models = pd.concat(model_rows, ignore_index=True) if model_rows else _empty(_MODEL_COLUMNS)

    summary, findings = [], []
    for key in models_in:
        p0, p1 = parsed_of.get((key, 0)), parsed_of.get((key, 1))
        if p0 is None or p1 is None:
            findings.append(
                Finding(
                    "stability_runs_failed",
                    "warn",
                    f"LVZ test for model {key!r}: a run failed.",
                    data={"model_key": key},
                )
            )
            continue
        m0, m1 = _model_of(p0), _model_of(p1)
        assert m0 is not None and m1 is not None
        r0, r1 = _rms(p0), _rms(p1)
        dv = _dv(m0, m1, "vp")
        nl = _n_lvz(m1["vp"].to_numpy(float))
        summary += _summary_rows(
            test,
            key,
            {
                "rms_lvz0": r0,
                "rms_lvz1": r1,
                "rms_gain": r0 - r1,
                "vp_dev_max": dv,
                "n_lvz_layers": float(nl),
            },
        )
        if np.isfinite(dv) and dv > scfg.velocity_spread_warn_km_s:
            findings.append(
                Finding(
                    "lvz_switch_changes_model",
                    "warn",
                    f"model {key!r}: switching lowveloclay changes a layer's Vp by {dv:.2f} km/s "
                    f"(RMS {r0:.4f} s with lowveloclay=0, {r1:.4f} s with 1).",
                    data={"model_key": key, "dv": dv, "rms_lvz0": r0, "rms_lvz1": r1},
                )
            )
    findings.append(
        Finding(
            "lvz_test_done",
            "info",
            f"lowveloclay on/off for models {list(models_in)}.",
            data={"models": list(models_in)},
        )
    )
    layers = _lvz_layers(models, starts)
    findings += _convergence_findings(runs, test)
    return StabilityResult(
        runs=runs,
        models=models,
        layers=layers,
        summary=_frame(summary, _SUMMARY_COLUMNS),
        findings=findings,
    )


def _lvz_layers(models: pd.DataFrame, starts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (key, variant), g in models.groupby(["model_key", "variant"], sort=False):
        s = starts[key]
        for r in g.itertuples():
            rows.append(
                {
                    "test": "lvz",
                    "variant": f"{key}:{variant}",
                    "layer": int(r.layer),
                    "depth_top": r.depth_top,
                    "vp_start": float(s["vp"].iloc[int(r.layer)]),
                    "vp_mean": r.vp,
                    "vp_std": 0.0,
                    "vp_min": r.vp,
                    "vp_max": r.vp,
                    "vp_max_abs_dev": abs(r.vp - float(s["vp"].iloc[int(r.layer)])),
                    "vs_std": np.nan,
                    "hits": r.hits,
                }
            )
    return _frame(rows, _LAYER_COLUMNS)


# ----------------------------------------------------------------------------------------
# Orchestration and output
# ----------------------------------------------------------------------------------------


def _ensemble_run_model(ens: EnsembleResult, rid: str) -> pd.DataFrame | None:
    fm = ens.final_models
    g = fm[fm["id"] == rid]
    if g.empty:
        return None
    p = g[g["phase"] == "P"].sort_values("layer")
    s = g[g["phase"] == "S"].sort_values("layer")
    if p.empty:
        return None
    df = pd.DataFrame(
        {"depth_top": p["depth_top"].to_numpy(float), "vp": p["v_final"].to_numpy(float)}
    )
    df["vs"] = s["v_final"].to_numpy(float) if len(s) == len(p) else np.nan
    return df


def run_stability(
    ensemble: EnsembleResult,
    inputs: InputsLike,
    cfg: Any,
    root: str | Path,
    seed: int = 0,
    *,
    control: VelestControl | None = None,
    n_tests: int | None = None,
    n_jobs: int = -1,
    timeout: float | None = 3600.0,
) -> StabilityResult:
    """Run the fixed-model run and all stability tests for an ensemble's final model.

    Layout: ``root/fixed_model``, ``root/hypocentre_perturbation``, ``root/reference_station``
    and ``root/lvz``.  The LVZ test also covers the ensemble's closest and best run models.
    """
    vcfg = getattr(cfg, "velest", cfg)
    scfg: StabilityConfig = vcfg.stability
    root_p = Path(root)
    model = ensemble.final_model
    fixed = fixed_model_run(
        model, inputs, control, root_p / "fixed_model", cfg=scfg, timeout=timeout
    )
    if not fixed.ok:
        raise StabilityError(f"fixed-model run failed ({fixed.status}); see {fixed.run_dir}")
    res = StabilityResult(fixed=fixed)
    res = res.merge(
        hypocentre_perturbation_test(
            model, inputs, control, cfg, root_p / "hypocentre_perturbation", seed,
            reference=fixed, n_tests=n_tests, n_jobs=n_jobs, timeout=timeout,
        )
    )  # fmt: skip
    res = res.merge(
        reference_station_test(
            model, inputs, control, cfg, root_p / "reference_station", seed,
            n_jobs=n_jobs, timeout=timeout,
        )
    )  # fmt: skip
    if scfg.lvz_test:
        extra: dict[str, ModelLike] = {}
        for label, rid in (("closest", ensemble.closest_run_id), ("best", ensemble.best_run_id)):
            m = _ensemble_run_model(ensemble, rid)
            if m is not None and rid not in extra:
                extra[f"{label}_{rid}"] = m
        res = res.merge(
            lvz_test(
                model, inputs, control, cfg, root_p / "lvz", seed,
                extra_models=extra, n_jobs=n_jobs, timeout=timeout,
            )
        )  # fmt: skip
    if fixed.max_velocity_change >= 0.011:
        res.findings.append(
            Finding(
                "fixed_model_not_fixed",
                "warn",
                f"the fixed-model run changed velocities by {fixed.max_velocity_change:.3f} km/s; "
                "increase velest.stability.fixed_vthet.",
                data={"max_velocity_change": fixed.max_velocity_change},
            )
        )
    return res


def write_stability(result: StabilityResult, outdir: str | Path) -> dict[str, Path]:
    """Write the tables (Parquet + CSV summary), ``summary.json`` and the fixed-run outputs.

    Files: ``stability_runs.parquet``, ``stability_events.parquet``,
    ``stability_models.parquet``, ``stability_layers.parquet``, ``stability_delays.parquet``,
    ``stability_summary.csv``, ``stability_findings.json`` and ``stability.json``; with a
    fixed-model run also ``fixed_station_delays.csv``, ``fixed_hypocentres.csv`` and
    ``fixed_model.csv``.  Returns ``{name: path}``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in (
        ("runs", result.runs),
        ("events", result.events),
        ("models", result.models),
        ("layers", result.layers),
        ("delays", result.delays),
    ):
        p = out / f"stability_{name}.parquet"
        df.infer_objects().to_parquet(p, index=False)
        paths[p.name] = p
    p = out / "stability_summary.csv"
    result.summary.to_csv(p, index=False)
    paths[p.name] = p
    p = out / "stability_findings.json"
    p.write_text(json.dumps(result.to_json()["findings"], indent=2))
    paths[p.name] = p
    p = out / "stability.json"
    p.write_text(json.dumps(result.to_json(), indent=2))
    paths[p.name] = p
    fx = result.fixed
    if fx is not None and fx.ok:
        for name, df in (
            ("fixed_station_delays", fx.station_delays),
            ("fixed_hypocentres", fx.hypocentres),
            ("fixed_model", fx.model),
        ):
            p = out / f"{name}.csv"
            df.to_csv(p, index=False)
            paths[p.name] = p
    return paths
