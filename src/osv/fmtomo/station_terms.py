"""P and S station terms estimated in the 3-D starting model (``model3d.estimate_station_terms``).

FMTOMO has no station parameters.  Iterating station terms while the velocities are inverted
(``fmtomo.station_terms: iterate``) is ill-posed on compact arrays: a station constant trades off
with the velocities along that station's ray bundle, and ``invert3d`` damps the velocities toward
the reference model every iteration, so the terms keep growing.  Here the events are relocated
as configured in ``fmtomo`` (``relocate``, ``joint``, ``gridloc``) with the velocities held fixed
(very large damping) and the terms iterated until they stop changing: they are the per-station
offsets that the starting model plus relocation cannot explain - the rule VELEST's delays follow,
but estimated in the 3-D model (topography, grid-search relocation) and with S independent of P.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from osv.config.schema import StationTermIterationConfig
from osv.report.checks import QCFinding

log = logging.getLogger(__name__)

__all__ = ["StationTermEstimate", "delay_table", "estimate_station_terms"]

#: file in the Stage 3 ``fmtomo/`` directory
TERMS_FILE = "station_terms.csv"
HISTORY_FILE = "station_terms_history.csv"


@dataclass
class StationTermEstimate:
    """Result of :func:`estimate_station_terms`."""

    terms: pd.DataFrame  # phase, station, term_s, n
    history: pd.DataFrame  # phase, iter, station, delta_s, term_s, n
    iterations: pd.DataFrame  # phase, iter, rms_ms, ...
    #: largest per-station mean change of the last updates (ms), see :func:`drift_ms`
    last_step_ms: dict[str, float] = field(default_factory=dict)
    findings: list[QCFinding] = field(default_factory=list)


#: convergence is judged on the mean change of the last updates: grid-search relocation moves
#: events between grid nodes, so single updates jitter by ~1 ms even when the terms have settled
DRIFT_UPDATES = 3


def drift_ms(history: pd.DataFrame, n: int = DRIFT_UPDATES) -> float:
    """Largest per-station mean change (ms) over the last ``n`` updates of ``history``."""
    if not len(history):
        return 0.0
    its = sorted(history["iter"].unique())[-n:]
    h = history[history["iter"].isin(its)]
    return 1000 * float(h.groupby("station")["delta_s"].mean().abs().max())


def _estimation_fm(cfg: Any) -> Any:
    est = cfg.model3d.station_term_estimation
    return cfg.fmtomo.model_copy(
        update={
            "damping": float(est.damping),
            "iterations": int(est.iterations) + 1,  # terms are updated after all but the last
            "station_terms": "iterate",
            "station_term_iteration": StationTermIterationConfig(min_picks=int(est.min_picks)),
            "save_rays": False,
            "invert_interfaces": [],
        }
    )


def estimate_station_terms(
    stage3_dir: str | Path, cfg: Any, workdir: str | Path, n_jobs: int = 1
) -> StationTermEstimate:
    """Relocate with fixed velocities and iterate P and S station terms to convergence.

    Reads the (uncorrected) FMTOMO hand-off files of ``stage3_dir`` (``fmtomo/``) and runs the
    phases as the invert stage would: P and S coupled for ``fmtomo.joint.relocation: joint``,
    else one run per phase.  ``workdir`` holds the runs (heavy files removed).

    Raises
    ------
    ValueError
        For ``joint.relocation: p_fixed_for_s`` or ``phases: [vpvs]`` (not supported).
    """
    from osv.fmtomo import stage as st

    s3 = Path(stage3_dir)
    work = Path(workdir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    est = cfg.model3d.station_term_estimation
    fm = _estimation_fm(cfg)
    phases = st._phases(cfg, s3)
    datas = {ph: st.load_phase_data(s3, ph, fm) for ph in phases}
    mode = st.coupled_mode(fm, phases) if len(phases) > 1 else None
    if mode not in (None, "joint"):
        raise ValueError(
            f"model3d.estimate_station_terms supports separate runs and joint.relocation: joint, "
            f"not {mode!r}"
        )
    d, s = float(fm.damping), float(fm.smoothing)
    if mode == "joint":
        reg = {p: {"damping": d, "smoothing": s, "source": "estimation"} for p in phases}
        co = st.invert_coupled(work, datas, fm, reg, n_jobs, mode="joint", dws_threshold=1e30)
        outcomes = co.outcomes
    else:
        outcomes = {
            ph: st.invert_phase(
                work / f"run_{ph}", datas[ph], fm, d, s, n_jobs, phase=ph, dws_threshold=1e30
            )
            for ph in phases
        }
    terms, hist, its, findings = [], [], [], []
    last: dict[str, float] = {}
    for ph, oc in outcomes.items():
        r = oc.result
        if r.station_terms is None:
            raise RuntimeError(f"{ph}: the estimation run returned no station terms")
        terms.append(r.station_terms.assign(phase=ph))
        h = r.station_term_history.assign(phase=ph)
        hist.append(h)
        its.append(oc.iterations.assign(phase=ph))
        step = drift_ms(h)
        last[ph] = step
        if step > est.tolerance_ms:
            findings.append(
                QCFinding(
                    "station_terms_not_converged",
                    "warn",
                    f"{ph}: the station terms still drift by {step:.1f} ms per update (mean of "
                    f"the last {DRIFT_UPDATES} updates, > {est.tolerance_ms:g} ms); raise "
                    "model3d.station_term_estimation.iterations",
                )
            )
        log.info("%s station terms: drift %.2f ms per update", ph, step)
    t = pd.concat(terms, ignore_index=True)[["phase", "station", "term_s", "n"]]
    return StationTermEstimate(
        terms=t,
        history=pd.concat(hist, ignore_index=True),
        iterations=pd.concat(its, ignore_index=True),
        last_step_ms=last,
        findings=findings,
    )


def delay_table(terms: pd.DataFrame) -> pd.DataFrame:
    """Terms in the layout of the VELEST ``station_delays.csv`` (``sta``, ``alias``, ``ptcor``,
    ``stcor``), for :func:`osv.model3d.sources.build_fmtomo_data`."""
    wide = terms.pivot_table(index="station", columns="phase", values="term_s", aggfunc="first")
    out = pd.DataFrame(
        {
            "sta": wide.index.astype(str),
            "alias": wide.index.astype(str),
            "ptcor": wide["P"].to_numpy() if "P" in wide else float("nan"),
            "stcor": wide["S"].to_numpy() if "S" in wide else float("nan"),
        }
    )
    return out.reset_index(drop=True)
