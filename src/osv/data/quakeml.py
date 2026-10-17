"""QuakeML reader producing clean event and pick tables.

The reader is deliberately defensive: real catalogues contain duplicated arrivals,
picks that no arrival references, placeholder magnitudes, unset evaluation modes and
asymmetric uncertainties.  Each such quirk is reported as a
:class:`~osv.report.checks.QCFinding` instead of silently being "fixed".
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.config.schema import DataSelectionConfig, InputsConfig
from osv.report.checks import QCFinding

logger = logging.getLogger(__name__)

#: Magnitudes outside this range are treated as implausible (microseismic scale).
MAG_PLAUSIBLE_RANGE: tuple[float, float] = (-3.0, 10.0)
#: A catalogue whose magnitudes all lie within this spread is treated as placeholder values.
MAG_IDENTICAL_TOL: float = 1e-6
#: Lower/upper uncertainties differing by more than this (s) count as asymmetric.
ASYMMETRY_TOL_S: float = 1e-9

EVENT_COLUMNS = [
    "event_id",
    "origin_id",
    "time",
    "lat",
    "lon",
    "depth_km",
    "depth_unc_km",
    "rms",
    "azgap",
    "n_phases",
    "mag",
    "mag_type",
    "mag_valid",
    "earth_model_id",
    "method_id",
    "creation_time",
    "creation_version",
    "n_origins",
]
PICK_COLUMNS = [
    "event_id",
    "pick_id",
    "net",
    "sta",
    "loc",
    "cha",
    "phase",
    "phase_hint",
    "time",
    "unc_s",
    "unc_lower",
    "unc_upper",
    "comment_weight",
    "polarity",
    "onset",
    "evaluation_mode",
    "travel_time_s",
]

_WGT_RE = re.compile(r"(?:wgt|weight)\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)


@dataclass
class CatalogTables:
    """Result of :func:`read_catalog`."""

    events: pd.DataFrame
    picks: pd.DataFrame
    findings: list[QCFinding] = field(default_factory=list)

    def finding(self, code: str) -> QCFinding | None:
        """Return the first finding with ``code`` or ``None``."""
        return next((f for f in self.findings if f.code == code), None)

    def write(self, outdir: str | Path) -> dict[str, Path]:
        """Write ``events.parquet``, ``picks.parquet`` and ``catalog_findings.json``."""
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "events": out / "events.parquet",
            "picks": out / "picks.parquet",
            "findings": out / "catalog_findings.json",
        }
        self.events.to_parquet(paths["events"])
        self.picks.to_parquet(paths["picks"])
        paths["findings"].write_text(json.dumps([f.to_dict() for f in self.findings], indent=2))
        return paths


def normalise_phase(hint: str | None) -> str | None:
    """Map a phase label (``P``, ``Pg``, ``Sn``, ``s`` ...) to ``"P"``/``"S"`` or ``None``."""
    if not hint:
        return None
    first = hint.strip()[:1].upper()
    return first if first in ("P", "S") else None


def parse_comment_weight(comments: Iterable[Any]) -> float:
    """Parse a ``wgt = x`` value from QuakeML comments (NaN if absent)."""
    for c in comments:
        m = _WGT_RE.search(getattr(c, "text", None) or "")
        if m:
            return float(m.group(1))
    return float("nan")


def _ts(t: Any) -> pd.Timestamp | pd.NaT:
    if t is None:
        return pd.NaT
    return pd.Timestamp(int(t.ns), unit="ns", tz="UTC")


def _f(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _creation(obj: Any) -> Any:
    ci = getattr(obj, "creation_info", None)
    return getattr(ci, "creation_time", None) if ci else None


def _select_origin(event: Any, selection: str) -> tuple[Any | None, str | None]:
    """Return ``(origin, warning_code)`` for ``event`` according to ``selection``."""
    origins = list(event.origins)
    if not origins:
        return None, "no_origin"
    if selection == "preferred":
        pref = event.preferred_origin()
        if pref is not None:
            return pref, None
        return origins[-1], "no_preferred_origin"
    if selection == "latest":
        dated = [o for o in origins if _creation(o) is not None]
        if dated:
            return max(dated, key=_creation), None
        return origins[-1], None
    if selection.startswith("by_method:"):
        key = selection.split(":", 1)[1]
        for o in reversed(origins):
            if key in str(o.method_id or "") or key in str(o.earth_model_id or ""):
                return o, None
        return None, "origin_method_not_found"
    raise ValueError(
        f"origin_selection {selection!r} not understood; use 'preferred', 'latest' or "
        "'by_method:<id>'"
    )


def _event_magnitude(event: Any, origin: Any) -> tuple[float, str | None]:
    mags = list(event.magnitudes)
    chosen = None
    pref = event.preferred_magnitude()
    if pref is not None:
        chosen = pref
    else:
        for m in mags:
            if m.origin_id is not None and str(m.origin_id) == str(origin.resource_id):
                chosen = m
                break
        if chosen is None and mags:
            chosen = mags[0]
    if chosen is None or chosen.mag is None:
        return float("nan"), None
    return float(chosen.mag), chosen.magnitude_type


def _unc(t: Any) -> tuple[float, float, float]:
    """Return (symmetric uncertainty, lower, upper) from a pick's time errors."""
    te = t.time_errors
    lo, up, u = _f(te.lower_uncertainty), _f(te.upper_uncertainty), _f(te.uncertainty)
    if not np.isnan(u):
        return u, lo, up
    if not np.isnan(lo) and not np.isnan(up):
        return (lo + up) / 2.0, lo, up
    one = lo if not np.isnan(lo) else up
    return one, lo, up


def read_catalog(
    paths: Sequence[str | Path] | str | Path | None,
    inputs_cfg: InputsConfig | None = None,
    data_cfg: DataSelectionConfig | None = None,
) -> CatalogTables:
    """Read QuakeML file(s) into event and pick tables plus QC findings.

    Parameters
    ----------
    paths
        QuakeML file(s). ``None`` uses ``inputs_cfg.quakeml``.
    inputs_cfg
        ``inputs`` config section (``origin_selection``, ``evaluation_mode``,
        ``use_arrivals``); defaults are used when omitted.  ``weight_source`` is *not*
        applied here: both ``unc_s`` and ``comment_weight`` are returned and Stage 1.3
        picks one.
    data_cfg
        ``data_selection`` config section; only ``magnitude.ignore_if_invalid`` is used.

    Returns
    -------
    CatalogTables
        ``events`` (one row per event, selected origin), ``picks`` (``travel_time_s`` is
        pick time minus origin time) and ``findings``.
    """
    from obspy import read_events

    cfg = inputs_cfg or InputsConfig()
    ignore_invalid_mag = (data_cfg or DataSelectionConfig()).magnitude.ignore_if_invalid
    if paths is None:
        paths = cfg.quakeml
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("No QuakeML files given; set inputs.quakeml in config.yaml")

    ev_rows: list[dict[str, Any]] = []
    pk_rows: list[dict[str, Any]] = []
    c: Counter[str] = Counter()  # counts of quirks
    affected: dict[str, set[str]] = {}

    def hit(code: str, event_id: str, n: int = 1) -> None:
        c[code] += n
        affected.setdefault(code, set()).add(event_id)

    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"QuakeML file not found: {p}")
        logger.info("Reading QuakeML %s", p)
        cat = read_events(str(p), format="QUAKEML")
        for event in cat:
            eid = str(event.resource_id)
            origin, warn = _select_origin(event, cfg.origin_selection)
            if warn:
                hit(warn, eid)
            if origin is None or origin.time is None:
                hit("event_skipped", eid)
                continue
            if len(event.origins) > 1:
                hit("multiple_origins", eid)
            t0 = origin.time
            mag, mag_type = _event_magnitude(event, origin)
            q = origin.quality
            n_ph = None
            if q is not None:
                n_ph = (
                    q.used_phase_count
                    if q.used_phase_count is not None
                    else q.associated_phase_count
                )
            ci = origin.creation_info
            depth_unc = origin.depth_errors.uncertainty if origin.depth_errors else None
            ev_rows.append(
                {
                    "event_id": eid,
                    "origin_id": str(origin.resource_id),
                    "time": _ts(t0),
                    "lat": _f(origin.latitude),
                    "lon": _f(origin.longitude),
                    "depth_km": _f(origin.depth) / 1000.0,
                    "depth_unc_km": _f(depth_unc) / 1000.0,
                    "rms": _f(q.standard_error) if q else float("nan"),
                    "azgap": _f(q.azimuthal_gap) if q else float("nan"),
                    "n_phases": n_ph if n_ph is not None else pd.NA,
                    "mag": mag,
                    "mag_type": mag_type,
                    "mag_valid": True,
                    "earth_model_id": str(origin.earth_model_id) if origin.earth_model_id else None,
                    "method_id": str(origin.method_id) if origin.method_id else None,
                    "creation_time": _ts(ci.creation_time) if ci else pd.NaT,
                    "creation_version": ci.version if ci else None,
                    "n_origins": len(event.origins),
                }
            )

            # --- arrivals of the selected origin
            arrivals = list(origin.arrivals)
            arr_by_pick: dict[str, Any] = {}
            for a in arrivals:
                key = str(a.pick_id)
                if key in arr_by_pick:
                    c["duplicate_arrivals"] += 1
                    affected.setdefault("duplicate_arrivals", set()).add(eid)
                else:
                    arr_by_pick[key] = a
            picks = {str(pk.resource_id): pk for pk in event.picks}
            for key in arr_by_pick:
                if key not in picks:
                    hit("arrival_without_pick", eid)
            unref = [k for k in picks if k not in arr_by_pick]
            if unref:
                hit("unreferenced_picks", eid, len(unref))

            use = [k for k in arr_by_pick if k in picks] if cfg.use_arrivals else list(picks)
            for key in use:
                pk = picks[key]
                if cfg.use_arrivals:
                    hint = arr_by_pick[key].phase
                    raw = pk.phase_hint
                    raw = hint if hint else raw
                else:
                    raw = pk.phase_hint
                phase = normalise_phase(raw)
                if phase is None:
                    hit("unknown_phase", eid)
                em = pk.evaluation_mode
                if em is None:
                    hit("eval_mode_unset", eid)
                elif cfg.evaluation_mode != "any" and em != cfg.evaluation_mode:
                    c["eval_mode_filtered"] += 1
                    continue
                unc, lo, up = _unc(pk)
                if np.isnan(unc):
                    hit("missing_uncertainty", eid)
                if not np.isnan(lo) and not np.isnan(up) and abs(lo - up) > ASYMMETRY_TOL_S:
                    hit("asymmetric_uncertainty", eid)
                wid = pk.waveform_id
                tt = (pk.time - t0) if pk.time is not None else float("nan")
                pk_rows.append(
                    {
                        "event_id": eid,
                        "pick_id": key,
                        "net": (wid.network_code or "") if wid else "",
                        "sta": (wid.station_code or "") if wid else "",
                        "loc": (wid.location_code or "") if wid else "",
                        "cha": (wid.channel_code or "") if wid else "",
                        "phase": phase,
                        "phase_hint": raw,
                        "time": _ts(pk.time),
                        "unc_s": unc,
                        "unc_lower": lo,
                        "unc_upper": up,
                        "comment_weight": parse_comment_weight(pk.comments),
                        "polarity": pk.polarity,
                        "onset": pk.onset,
                        "evaluation_mode": em,
                        "travel_time_s": float(tt),
                    }
                )

    events = pd.DataFrame(ev_rows, columns=EVENT_COLUMNS)
    picks = pd.DataFrame(pk_rows, columns=PICK_COLUMNS)
    events["time"] = pd.to_datetime(events["time"], utc=True)
    events["creation_time"] = pd.to_datetime(events["creation_time"], utc=True)
    events["mag"] = events["mag"].astype(float)
    events["n_phases"] = events["n_phases"].astype("Int64")
    events["n_origins"] = events["n_origins"].astype("int64")
    picks["time"] = pd.to_datetime(picks["time"], utc=True)
    for col in ("phase", "polarity", "onset", "evaluation_mode"):
        picks[col] = picks[col].astype(object)

    findings = _build_findings(c, affected, events, picks, cfg, ignore_invalid_mag)
    # magnitude validity
    events["mag_valid"] = _mag_valid(events["mag"], ignore_invalid_mag, findings)
    dup_ids = events["event_id"].duplicated().sum()
    if dup_ids:
        findings.append(
            QCFinding(
                "duplicate_event_ids",
                "warn",
                f"{dup_ids} event id(s) appear more than once across the input files.",
                int(dup_ids),
            )
        )
    for f in findings:
        logger.log(logging.WARNING if f.severity != "info" else logging.INFO, "%s", f)
    return CatalogTables(events=events, picks=picks, findings=findings)


def _mag_valid(mag: pd.Series, ignore_invalid: bool, findings: list[QCFinding]) -> pd.Series:
    lo, hi = MAG_PLAUSIBLE_RANGE
    finite = mag.notna() & np.isfinite(mag)
    plausible = finite & (mag >= lo) & (mag <= hi)
    n = int(len(mag))
    identical = n > 1 and bool(finite.all()) and float(mag.max() - mag.min()) <= MAG_IDENTICAL_TOL
    problems = []
    if identical:
        problems.append(f"all {n} magnitudes are identical ({mag.iloc[0]:.3f})")
    n_implaus = int((~plausible).sum())
    if n_implaus:
        problems.append(f"{n_implaus} magnitude(s) missing or outside [{lo}, {hi}]")
    if problems:
        action = "marked invalid (mag_valid=False)" if ignore_invalid else "kept as valid"
        findings.append(
            QCFinding(
                "invalid_magnitudes",
                "warn",
                "; ".join(problems) + f"; {action}.",
                n_implaus if not identical else n,
                {"identical": identical, "ignore_if_invalid": ignore_invalid, "range": [lo, hi]},
            )
        )
    if not ignore_invalid:
        return finite
    return plausible & (not identical)


def _build_findings(
    c: Counter[str],
    affected: dict[str, set[str]],
    events: pd.DataFrame,
    picks: pd.DataFrame,
    cfg: InputsConfig,
    ignore_invalid_mag: bool,
) -> list[QCFinding]:
    out: list[QCFinding] = []

    def add(code: str, sev: str, msg: str, n: int | None = None, **details: Any) -> None:
        ev = sorted(affected.get(code, ()))
        d: dict[str, Any] = {"n_events": len(ev), "events": ev[:20], **details}
        out.append(QCFinding(code, sev, msg, c[code] if n is None else n, d))  # type: ignore[arg-type]

    if c["duplicate_arrivals"]:
        add(
            "duplicate_arrivals",
            "warn",
            f"{c['duplicate_arrivals']} duplicated arrival(s) (same pick_id repeated within an "
            f"origin) in {len(affected['duplicate_arrivals'])} event(s); duplicates dropped.",
        )
    if c["unreferenced_picks"]:
        sev = "info" if cfg.use_arrivals is False else "warn"
        add(
            "unreferenced_picks",
            sev,
            f"{c['unreferenced_picks']} pick(s) are not referenced by any arrival of the "
            "selected origin ("
            + ("dropped" if cfg.use_arrivals else "kept; use_arrivals is false")
            + ").",
        )
    if c["arrival_without_pick"]:
        add(
            "arrival_without_pick",
            "warn",
            f"{c['arrival_without_pick']} arrival(s) refer to a pick that is not in the event.",
        )
    if c["eval_mode_unset"]:
        mode = cfg.evaluation_mode
        if mode == "any":
            add(
                "eval_mode_unset",
                "info",
                f"{c['eval_mode_unset']} pick(s) have no evaluation_mode.",
            )
        else:
            add(
                "eval_mode_unset",
                "warn",
                f"{c['eval_mode_unset']} pick(s) have no evaluation_mode; kept although "
                f"evaluation_mode={mode!r} was requested.",
            )
    if c["eval_mode_filtered"]:
        add(
            "eval_mode_filtered",
            "info",
            f"{c['eval_mode_filtered']} pick(s) dropped by "
            f"evaluation_mode={cfg.evaluation_mode!r}.",
        )
    if c["missing_uncertainty"]:
        add(
            "missing_uncertainty",
            "warn",
            f"{c['missing_uncertainty']} pick(s) have no time uncertainty (unc_s is NaN).",
        )
    if c["asymmetric_uncertainty"]:
        add(
            "asymmetric_uncertainty",
            "info",
            f"{c['asymmetric_uncertainty']} pick(s) have different lower/upper uncertainties; "
            "unc_s is their mean.",
        )
    if c["unknown_phase"]:
        add(
            "unknown_phase",
            "warn",
            f"{c['unknown_phase']} pick(s) have a phase that is not P or S (phase is null).",
        )
    if c["multiple_origins"]:
        add(
            "multiple_origins",
            "info",
            f"{c['multiple_origins']} event(s) have several origins; "
            f"origin_selection={cfg.origin_selection!r} applied.",
        )
    if c["no_preferred_origin"]:
        add(
            "no_preferred_origin",
            "warn",
            f"{c['no_preferred_origin']} event(s) have no preferred origin; last origin used.",
        )
    if c["origin_method_not_found"]:
        add(
            "origin_method_not_found",
            "warn",
            f"{c['origin_method_not_found']} event(s) have no origin matching "
            f"{cfg.origin_selection!r}; skipped.",
        )
    if c["no_origin"]:
        add("no_origin", "warn", f"{c['no_origin']} event(s) have no origin; skipped.")
    if c["event_skipped"]:
        add("event_skipped", "warn", f"{c['event_skipped']} event(s) skipped (no usable origin).")
    out.append(
        QCFinding(
            "catalog_summary",
            "info",
            f"{len(events)} events, {len(picks)} picks "
            f"({int((picks['phase'] == 'P').sum())} P, {int((picks['phase'] == 'S').sum())} S).",
            len(picks),
        )
    )
    return out
