"""Parser for the VELEST main print output (the ``.OUT`` file, ``outfile`` of ``velest.cmn``).

:func:`parse_velest_out` / :func:`read_velest_out` turn the print file into a
:class:`VelestRunResult`; :func:`read_run_dir` additionally reads the output CNV and
station files (and ``velout.mod`` when present) with :mod:`osv.io.velest`.  The parser
never raises on partial or garbled output (truncated files of crashed runs, ``NaN`` and
``*****`` fields): such a run is marked ``failed`` with a reason instead.

Layout of the file (vendored ``fortran/velest/velest.f``; ``unit 16``)
----------------------------------------------------------------------
* Header: title ``velest.f`` (``INPUTPARAM``, control echo ``1163-1215``), the control
  parameters as ``*** name name ...`` header lines each followed by one line of values,
  file names as ``*** label:`` lines followed by the name.
* ``INPUT - M O D E L``: ``Velocity structure for model n`` (``1396-1440``) with layer rows
  ``layer vel depth vdamp reflector``.
* ``neqs= .. nshot= .. nltot= ..``; ``I N P U T - D A T A`` event table, the lines
  ``readings for station XXXX : tot= P: S:``, ``Total number of readings`` (``5969``) and
  ``Total number of observations is:`` (``5000``).  The P/S totals overflow their ``i7``
  field (``*******``) with ``-fdefault-integer-8`` style builds, so the counts are summed
  from the per-station lines instead.
* Per iteration (``velest.f:342``): ``ITERATION no n`` ... ``(Applied) Step length =``
  (``5426``) ... ``Iteration nr n obtained:`` + ``DATVAR= .. mean sqrd residual= ..  RMS
  RESIDUAL= ..`` (``CHECKSOLUTION``, ``4216-4223``) + the ``After n iterations we got:``
  block (``5080-5104``: average absolute [mean] residual of direct/refracted/reflected/all
  rays and the percentage change of the absolute residual).  If the mean squared residual
  rose (``varat < 0.99``, ``4250-4290``) VELEST prints ``msqrd res increased -->
  hypocenter and model backup.`` (``584``), halves the step (``BACKUP``,
  ``5229-5360``: ``Velocity readjustments:``, ``Half adjustments made``) and re-evaluates:
  ``(Iteration nr n)   BACKUP nr k obtained:`` with ``k = 1..4``.  Four backups stop the
  iterations (``613``).  ``NITTOUTPUT`` (``2895-3085``) prints, for the accepted state, the
  hypocentre table (``eq ot x y z rms avres dot dx dy dz``), ``A V E R A G E of
  [ABSOLUTE] ADJUSTMENTS``, ``Step length damping of x was applied`` / ``NO step length
  damping applied``, ``Velocity adjustments:`` (rows ``vp dvp hp reflector`` per model) and
  ``Adjusted station corrections:`` (P: ``stn ptcor dpcor``, S: ``stn stcor dscor``).
* Stop reasons (``velest.f:610-650``): ``4 TIMES BACKUP MADE``, ``STEP-LENGTH < DELMIN``,
  ``NITT = ITTMAX``, ``Changes in datvar < 1.e-6`` (single-event mode).
* Final: ``Final rms values``, ``output final hypocenters`` table (``3155-3190``),
  ``station statistics`` (``6062-6215``: ``sta phase nobs avres avwres std wsum delay``),
  ``RAY-STATISTICS FOR LAST ITERATION`` (``2263``), ``GAP`` statistics, magnitude and
  depth histograms, ``Rdiag of selected model parameters`` with ``iresolcalc > 0``
  (``RESOLCOVAR``, ``5440-5560``) and finally ``TOTAL CPU-sec`` (``837``), the marker of a
  run that reached the end.

Single-event mode (``isingle=1`` with ``iturbo=0``) repeats the iteration block for every
event; ``iterations["run"]`` numbers these (1 for a simultaneous run).  With ``iturbo=1``
VELEST writes no ``.OUT`` at all in single-event mode.

``STOP`` messages and fatal errors of ``OPENERROR`` (``velest.f:3318``) end the file
abruptly; the exit code of VELEST is 0 even then, so the print file is the only evidence.

Tables
------
* ``iterations``: ``run``, ``entry``, ``iteration``, ``kind`` (``initial``/``iteration``/
  ``backup``), ``backup_nr``, ``accepted`` (the last entry of an iteration), ``datvar``,
  ``mean_sq_res``, ``rms``, ``step_length``, ``damping``, ``n_depth_constrained``,
  ``n_direct``/``n_refracted``/``n_reflected``/``n_all`` rays, ``abs_res_direct``,
  ``abs_res_refracted``, ``abs_res_reflected``, ``abs_res_all``, ``mean_res_all``,
  ``pct_change_abs_res``.
* ``models``: ``entry``, ``iteration``, ``phase`` (``P``/``S``), ``layer`` (1-based),
  ``vel`` (km/s), ``dvel``, ``depth`` (km, top of layer, positive down), ``readjusted``
  (True for the ``Velocity readjustments`` printed by a backup, False for the accepted
  ``Velocity adjustments``).
* ``hypocentre_shifts``: mean and mean-absolute adjustment per entry (``dot``, ``dx``,
  ``dy``, ``dz``, ``adot``, ``adx``, ``ady``, ``adz``; seconds / km).
* ``station_corrections``: ``sta``, ``ptcor``, ``stcor`` (seconds), ``dptcor``,
  ``dstcor`` (last adjustment), ``nobs_p``, ``nobs_s``, ``nobs``.
* ``station_stats``: ``sta``, ``phase``, ``nobs``, ``avres``, ``avwres``, ``std``,
  ``wsum``, ``delay``.
* ``final_hypocentres``: ``eq``, ``time`` (UTC), ``lat``, ``lon`` (signed, as printed),
  ``depth_km``, ``mag``, ``nobs``, ``rms``, ``x``, ``y``, ``z``.
* ``ray_statistics``: layer table of the last iteration (``nlay``, ``top``, ``bottom``,
  ``vel``, ``nhyp``, ``nref``, ``pct_len``, ``nhit``, ``xy_km``, ``z_km``, ``rflx``).
* ``resolution``: ``param``, ``sigma``, ``rdiag`` (``iresolcalc > 0``).
"""

from __future__ import annotations

import gzip
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.coords import parse_velest_lat, parse_velest_lon
from osv.io.velest import (
    CnvData,
    ModelBlock,
    VelestFormatError,
    VelestModel,
    _year_from_yy,
    parse_cnv,
    parse_model,
    parse_stations,
)

__all__ = [
    "CONVERGED_REASONS",
    "VelestRunResult",
    "parse_velest_out",
    "read_run_dir",
    "read_velest_out",
]

logger = logging.getLogger(__name__)

#: Stop reasons that mean the step became negligible (a converged solution).
CONVERGED_REASONS = ("delmin", "datvar")

ITERATION_COLUMNS = [
    "run",
    "entry",
    "iteration",
    "kind",
    "backup_nr",
    "accepted",
    "datvar",
    "mean_sq_res",
    "rms",
    "step_length",
    "damping",
    "n_depth_constrained",
    "n_direct",
    "n_refracted",
    "n_reflected",
    "n_all",
    "abs_res_direct",
    "abs_res_refracted",
    "abs_res_reflected",
    "abs_res_all",
    "mean_res_all",
    "pct_change_abs_res",
]
MODEL_COLUMNS = ["entry", "iteration", "phase", "layer", "vel", "dvel", "depth", "readjusted"]
SHIFT_COLUMNS = [
    "entry", "iteration", "kind", "backup_nr", "dot", "dx", "dy", "dz",
    "adot", "adx", "ady", "adz",
]  # fmt: skip
STACORR_COLUMNS = ["sta", "ptcor", "stcor", "dptcor", "dstcor", "nobs_p", "nobs_s", "nobs"]
STATS_COLUMNS = ["sta", "phase", "nobs", "avres", "avwres", "std", "wsum", "delay"]
HYPO_COLUMNS = ["eq", "time", "lat", "lon", "depth_km", "mag", "nobs", "rms", "x", "y", "z"]
RAY_COLUMNS = [
    "nlay", "top", "bottom", "vel", "nhyp", "nref", "pct_len", "nhit", "xy_km", "z_km", "rflx",
]  # fmt: skip
RES_COLUMNS = ["param", "sigma", "rdiag"]

_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|NaN|Infinity|\*+"
_FNUM = re.compile(_NUM)

_INT_PARAMS = {
    "icoordsystem", "itrial", "ised", "neqs", "nshot", "isingle", "iresolcalc", "itopo",
    "lowveloclay", "nsp", "nmod", "nsinv", "nshcor", "nshfix", "iuseelev", "iusestacorr",
    "iturbo", "icnvout", "istaout", "ismpout", "irayout", "idrvout", "ialeout", "idspout",
    "irflout", "irfrout", "iresout", "ittmax", "invertratio",
}  # fmt: skip

_RE_ENTRY = re.compile(r"^\(?\s*Iteration nr\s+(\d+)\)?\s+(?:BACKUP nr\s+(\d+)\s+)?obtained:")
_RE_DATVAR = re.compile(r"DATVAR=\s*(\S+)\s+mean sqrd residual=\s*(\S+)\s+RMS RESIDUAL=\s*(\S+)")
_RE_RAYS = re.compile(
    r"^\s*(\d+)\s+(straight and direct|refracted|reflected)\s+rays\s*=\s*(\S+)\s*\[\s*(\S+)\]"
)
_RE_ALL = re.compile(r"^\s*(\d+)\s+ALL\s+RAYS\s*=\s*(\S+)\s*\[\s*(\S+)\]\s+(\S+)\s*%")
_RE_STEP = re.compile(r"\(Applied\) Step length =\s*(\S+)")
_RE_DAMP = re.compile(r"Step length damping of\s*(\S+)\s+was applied")
_RE_CONSTR = re.compile(r"\*{5} depth constrained for event")
_RE_AVG = re.compile(r"A V E R A G E\s+of\s+(ABSOLUTE\s+)?ADJUSTMENTS\s*:(.*)$")
# velest.f 5948-5957: ``tot= P: S:`` is printed only for nsp=2; every other nsp prints just
# `` readings for station XXXX :  NNN`` (the total).
_RE_READ = re.compile(
    r"readings for station\s+(\S+)\s*:\s*(?:tot=\s*(\d+)\s+P:\s*(\d+)\s+S:\s*(\d+)|(\d+)\s*$)"
)
_RE_HYPO = re.compile(
    r"^\s*(\d+)\s+(\d{6})\s+(\d{4})\s+(\d+\.\d+)\s+(\d+\.\d+[NSns])\s*(\d+\.\d+[EWew])\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)
_RE_STAT = re.compile(r"^\s*(\S{1,4})\s+([PS])\s+(\d+)\s+(.*)$")
_RE_RAYROW = re.compile(
    r"^\s*(\d+)\s+(-?\d+\.\d+)\.\.\.\s*(-?\d+\.\d+)\s+km\s+(-?\d+\.\d+)\s+km/s\s+"
    r"(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s*$"
)
_RE_LAYER = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*([mM]?)\s*$")

_STOP_REASONS = (
    ("4 TIMES BACKUP MADE", "four_backups"),
    ("STEP-LENGTH < DELMIN", "delmin"),
    ("NITT = ITTMAX", "ittmax"),
    ("Changes in datvar < 1.e-6", "datvar"),
    ("no iteration is made", "no_iterations"),
)
_WARN_PATTERNS = (
    "WARNING",
    "ERROR",
    "Event cannot be located",
    "knobs(i)-nobswithw0 < nvar",
    "Reflected wave did not converge",
    "probably a low velocity layer detected",
    "Phase therefore skipped",
    "4 times backup made",
    "Only   m   or   M   are allowed",
    "LOWVELOCLAY now set to 0",
    "STOP",
)


# ----------------------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------------------


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})


@dataclass(eq=False)
class VelestRunResult:
    """Everything read from one VELEST run (see the module docstring for the tables).

    Attributes
    ----------
    converged
        The iterations ended because the step became negligible (``STEP-LENGTH < DELMIN``
        or, in single-event mode, ``Changes in datvar``) in every run and the run did not
        fail.  ``False`` when ``ittmax`` was reached (``stop_reason == "ittmax"``), after
        four backups, or on failure.
    diverged
        The final RMS is not finite or larger than the initial RMS.
    finished
        The run reached the end of the program (``TOTAL CPU-sec`` printed).
    failed, failure_reason
        ``failed`` is set for a truncated print file (crash, open error, ``STOP``), a
        non-finite final RMS or a matrix decomposition error; ``failure_reason`` says why.
    stop_reason
        ``"delmin"``, ``"datvar"``, ``"ittmax"``, ``"four_backups"``, ``"no_iterations"``
        or ``None`` (``"mixed"`` if the single-event runs ended differently).
    warnings
        Messages with the line number, e.g. ``"line 123: WARNING: ..."``.
    cnv, stations, velout_model
        Filled by :func:`read_run_dir` (output CNV, output station file, ``velout.mod``).
    """

    source: str | None = None
    title: str = ""
    version: str | None = None
    control: dict[str, float | int] = field(default_factory=dict)
    control_files: dict[str, str] = field(default_factory=dict)
    n_events: int | None = None
    n_obs: int | None = None
    n_obs_p: int | None = None
    n_obs_s: int | None = None
    initial_model: VelestModel | None = None
    final_model: VelestModel | None = None
    iterations: pd.DataFrame = field(default_factory=lambda: _empty(ITERATION_COLUMNS))
    models: pd.DataFrame = field(default_factory=lambda: _empty(MODEL_COLUMNS))
    hypocentre_shifts: pd.DataFrame = field(default_factory=lambda: _empty(SHIFT_COLUMNS))
    station_corrections: pd.DataFrame = field(default_factory=lambda: _empty(STACORR_COLUMNS))
    station_stats: pd.DataFrame = field(default_factory=lambda: _empty(STATS_COLUMNS))
    final_hypocentres: pd.DataFrame = field(default_factory=lambda: _empty(HYPO_COLUMNS))
    ray_statistics: pd.DataFrame = field(default_factory=lambda: _empty(RAY_COLUMNS))
    resolution: pd.DataFrame | None = None
    summary: dict[str, float | int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    stop_reasons: list[str] = field(default_factory=list)
    converged: bool = False
    diverged: bool = False
    finished: bool = False
    failed: bool = True
    failure_reason: str | None = "empty output"
    n_lines: int = 0
    cnv: CnvData | None = None
    stations: pd.DataFrame | None = None
    velout_model: VelestModel | None = None

    @property
    def single_event_mode(self) -> bool:
        """True for ``isingle`` runs (the iteration block is repeated per event)."""
        return bool(self.control.get("isingle", 0))

    @property
    def accepted(self) -> pd.DataFrame:
        """Rows of ``iterations`` that are the accepted state of their iteration."""
        return self.iterations[self.iterations["accepted"]]

    @property
    def n_iterations(self) -> int:
        """Number of the last iteration reached (0 if none; highest over all runs)."""
        it = self.iterations["iteration"]
        return int(it.max()) if len(it) else 0

    @property
    def n_backups(self) -> int:
        """Number of backup (step-halving) evaluations over the whole run."""
        return int((self.iterations["kind"] == "backup").sum())

    @property
    def backup_iterations(self) -> list[int]:
        """Iteration numbers in which at least one backup was made (ascending, unique)."""
        b = self.iterations.loc[self.iterations["kind"] == "backup", "iteration"]
        return sorted({int(v) for v in b})

    @property
    def initial_rms(self) -> float:
        """RMS residual before the first iteration (``nan`` when not printed)."""
        r = self.iterations.loc[self.iterations["kind"] == "initial", "rms"]
        return float(r.iloc[0]) if len(r) else math.nan

    @property
    def final_rms(self) -> float:
        """RMS residual of the last entry (``nan`` when not printed or not finite)."""
        r = self.iterations["rms"]
        return float(r.iloc[-1]) if len(r) else math.nan

    @property
    def rms_history(self) -> pd.Series:
        """Accepted RMS residual indexed by iteration (first run of a single-event file)."""
        a = self.accepted
        if len(a):
            a = a[a["run"] == a["run"].iloc[0]]
        return pd.Series(a["rms"].to_numpy(), index=a["iteration"].to_numpy(), name="rms")


# ----------------------------------------------------------------------------------------
# Low level helpers
# ----------------------------------------------------------------------------------------


def _num(tok: str | None) -> float:
    """Fortran number to float; ``NaN``, ``*****`` (field overflow) and junk give ``nan``."""
    if tok is None:
        return math.nan
    t = tok.strip().replace("D", "E").replace("d", "e")
    try:
        return float(t)
    except ValueError:
        return math.nan


def _numbers(text: str) -> list[float]:
    return [_num(t) for t in _FNUM.findall(text)]


def _read_text(path: str | Path) -> str:
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.decompress(p.read_bytes()).decode("latin-1")
    return p.read_text(encoding="latin-1")


def _is_header_names(tokens: list[str]) -> bool:
    return bool(tokens) and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t) for t in tokens)


def _parse_control_echo(lines: list[str]) -> tuple[dict[str, float | int], dict[str, str]]:
    """Control parameters (``*** name name`` + value line) and file names (``*** label:``)."""
    params: dict[str, float | int] = {}
    files: dict[str, str] = {}
    n = len(lines)
    for i, ln in enumerate(lines):
        if not ln.startswith("***") or i + 1 >= n:
            continue
        body = ln[3:].strip()
        if not body or body.startswith("*"):
            continue
        nxt = lines[i + 1]
        if body.endswith(":"):
            val = nxt.strip()
            if val and not val.startswith("***"):
                files[body[:-1].strip()] = val
            continue
        names = body.split()
        if not _is_header_names(names):
            continue
        vals = _FNUM.findall(nxt)
        if len(vals) != len(names):
            continue
        for name, tok in zip(names, vals, strict=True):
            v = _num(tok)
            if math.isnan(v):
                continue
            params[name.lower()] = int(round(v)) if name.lower() in _INT_PARAMS else v
    return params, files


def _parse_initial_model(lines: list[str]) -> VelestModel | None:
    """Initial model from ``model file title`` and the ``Velocity structure`` blocks."""
    title = ""
    blocks: dict[int, list[tuple[float, float, float, str]]] = {}
    cur: int | None = None
    for i, ln in enumerate(lines):
        if "model file title" in ln and i + 1 < len(lines):
            title = lines[i + 1].strip()
            continue
        m = re.match(r"\s*Velocity structure for model\s+(\d+)", ln)
        if m:
            cur = int(m.group(1))
            blocks[cur] = []
            continue
        if cur is None:
            continue
        if "Calculation of average velocity" in ln or "INPUT - S T A T I O N S" in ln:
            cur = None
            continue
        r = _RE_LAYER.match(ln)
        if r:
            blocks[cur].append(
                (float(r.group(2)), float(r.group(3)), float(r.group(4)), r.group(5))
            )
    if 1 not in blocks or not blocks[1]:
        return None

    def block(rows: list[tuple[float, float, float, str]]) -> ModelBlock:
        df = pd.DataFrame(rows, columns=["vel", "depth", "vdamp", "reflector"])
        df["reflector"] = df["reflector"].replace("", " ")
        return ModelBlock(df, label=title)

    try:
        s = block(blocks[2]) if blocks.get(2) else None
        return VelestModel(title=title, p=block(blocks[1]), s=s)
    except VelestFormatError as exc:  # pragma: no cover - defensive
        logger.warning("initial model in VELEST output is invalid: %s", exc)
        return None


def _parse_velocity_block(lines: list[str], i: int) -> tuple[list[dict[str, Any]], int]:
    """Rows after a ``Velocity adjustments:`` / ``readjustments:`` line.

    Returns the rows (``model``, ``layer``, ``vel``, ``dvel``, ``depth``) and the index of
    the first line that does not belong to the block.
    """
    rows: list[dict[str, Any]] = []
    model = 0
    layer = 0
    n = len(lines)
    i += 1
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if s.startswith("Adjusted station") or s.startswith("~~~") or s.startswith("-----"):
            break
        if s.startswith("Velocity adjustments") or s.startswith("Velocity readjustments"):
            break
        if _RE_ENTRY.match(s) or s.startswith("ITERATION no") or "Half adjustments" in s:
            break
        m = re.match(r"Velocity model\s+(\d+)", s)
        if m:
            model = int(m.group(1))
            layer = 0
        elif model and re.fullmatch(r"[-+\d.\sNaN*mM]+", s) and len(_FNUM.findall(s)) >= 3:
            vals = _numbers(s)
            layer += 1
            rows.append(
                {"model": model, "layer": layer, "vel": vals[0], "dvel": vals[1], "depth": vals[2]}
            )
        i += 1
    return rows, i


def _parse_stacorr_block(lines: list[str], i: int) -> tuple[list[tuple[str, float, float]], int]:
    """Rows after an ``Adjusted station corrections:`` header (4 per line, 20 chars each)."""
    out: list[tuple[str, float, float]] = []
    n = len(lines)
    i += 2  # the title line and the "stn ptcor dpcor" header
    while i < n:
        ln = lines[i]
        if not ln.strip() or "Adjusted" in ln or ln.lstrip().startswith("~~~"):
            break
        for k in range(0, len(ln), 20):
            chunk = ln[k : k + 20]
            if not chunk.strip():
                continue
            name = chunk[2:6].strip()
            vals = chunk[6:].replace("NaN", " NaN ").split()
            if not name or len(vals) < 2:
                # NaN / fused fields: fall back to the generic number scanner
                nums = _numbers(chunk[6:])
                if not name or len(nums) < 2:
                    continue
                vals = [str(v) for v in nums[:2]]
            out.append((name, _num(vals[0]), _num(vals[1])))
        i += 1
    return out, i


def _entry_extra(lines: list[str], i: int) -> dict[str, float]:
    """``DATVAR`` line and ray-residual lines that follow an ``Iteration nr`` line."""
    d: dict[str, float] = {}
    for j in range(i + 1, min(i + 16, len(lines))):
        ln = lines[j]
        m = _RE_DATVAR.search(ln)
        if m:
            d["datvar"], d["mean_sq_res"], d["rms"] = (_num(m.group(k)) for k in (1, 2, 3))
            continue
        m = _RE_RAYS.match(ln)
        if m:
            key = {
                "straight and direct": "direct",
                "refracted": "refracted",
                "reflected": "reflected",
            }[m.group(2)]
            d[f"n_{key}"] = float(m.group(1))
            d[f"abs_res_{key}"] = _num(m.group(3))
            continue
        m = _RE_ALL.match(ln)
        if m:
            d["n_all"] = float(m.group(1))
            d["abs_res_all"] = _num(m.group(2))
            d["mean_res_all"] = _num(m.group(3))
            d["pct_change_abs_res"] = _num(m.group(4))
            break
        if ln.lstrip().startswith("~~~") or _RE_ENTRY.match(ln.strip()):
            break
    return d


def _model_from_rows(rows: pd.DataFrame, template: VelestModel | None) -> VelestModel | None:
    """Final model from the ``phase``/``layer`` rows of the last velocity block."""
    if rows.empty or not np.isfinite(rows[["vel", "depth"]].to_numpy()).all():
        return None
    blocks: dict[str, ModelBlock] = {}
    for ph, grp in rows.groupby("phase", sort=True):
        grp = grp.sort_values("layer")
        base = template.p if (ph == "P" or template is None or template.s is None) else template.s
        vdamp = np.ones(len(grp))
        refl = [" "] * len(grp)
        if template is not None and len(base.layers) == len(grp):
            vdamp = base.layers["vdamp"].to_numpy()
            refl = list(base.layers["reflector"])
        df = pd.DataFrame(
            {"vel": grp["vel"].to_numpy(), "depth": grp["depth"].to_numpy(), "vdamp": vdamp,
             "reflector": refl}
        )  # fmt: skip
        try:
            blocks[str(ph)] = ModelBlock(df, label=base.label if template is not None else "")
        except VelestFormatError:
            return None
    if "P" not in blocks:
        return None
    return VelestModel(
        title=template.title if template is not None else "",
        p=blocks["P"],
        s=blocks.get("S"),
    )


# ----------------------------------------------------------------------------------------
# Main parser
# ----------------------------------------------------------------------------------------


def parse_velest_out(text: str, *, source: str | None = None) -> VelestRunResult:
    """Parse the text of a VELEST print file into a :class:`VelestRunResult`.

    Never raises on malformed or truncated input; the result is then ``failed`` with a
    ``failure_reason`` (an empty text gives ``failure_reason == "empty output"``).
    """
    res = VelestRunResult(source=source)
    lines = [ln.rstrip("\r\n") for ln in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    n = len(lines)
    res.n_lines = n
    if not any(ln.strip() for ln in lines):
        return res
    try:
        _parse_into(res, lines)
    except Exception as exc:  # noqa: BLE001 - the parser must never raise on odd output
        logger.warning("unexpected error while parsing %s: %s", source or "<text>", exc)
        res.failed = True
        res.failure_reason = f"parser error: {exc}"
    return res


def _parse_into(res: VelestRunResult, lines: list[str]) -> None:  # noqa: C901
    n = len(lines)
    # ---- header -----------------------------------------------------------------------
    for i, ln in enumerate(lines[:40]):
        if "Version" in ln and "V E L E S T" in ln:
            res.version = ln.split(":", 1)[-1].strip()
        if "Title of this VELEST run" in ln:
            for k in range(i + 1, min(i + 4, n)):
                if lines[k].strip():
                    res.title = lines[k].strip()
                    break
    first_iter = next((i for i, ln in enumerate(lines) if _RE_ENTRY.match(ln.strip())), n)
    head = lines[: min(first_iter, n)]
    res.control, res.control_files = _parse_control_echo(head)
    res.initial_model = _parse_initial_model(head)

    # ---- sequential scan --------------------------------------------------------------
    entries: list[dict[str, Any]] = []
    shifts: list[dict[str, Any]] = []
    vel_rows: list[dict[str, Any]] = []
    sta_tables: list[dict[str, Any]] = []  # per NITTOUTPUT: {"entry", "P": rows, "S": rows}
    current: dict[str, Any] | None = None
    pending_step = math.nan
    run = 1
    seen_initial = False
    stop_reasons: list[str] = []
    n_cpu = 0
    obs_totals: list[int] = []
    read_totals: list[int] = []
    readings: dict[str, list[int]] = {}
    warnings: list[str] = []
    suspicious = 0
    neg_var: list[str] = []
    in_final_hypo = False
    in_stats = False
    hypo_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    ray_rows: list[dict[str, Any]] = []
    summary: dict[str, float | int] = {}
    res_rdiag: list[float] = []
    sigma: list[float] = []
    n_cannot_locate = 0
    ludecp_error = False
    open_error: str | None = None
    i = 0
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue

        m = _RE_ENTRY.match(s)
        if m:
            it = int(m.group(1))
            bk = int(m.group(2)) if m.group(2) else 0
            if it == 0 and not bk:
                if seen_initial:
                    run += 1
                seen_initial = True
            kind = "backup" if bk else ("initial" if it == 0 else "iteration")
            rec: dict[str, Any] = {
                "run": run,
                "entry": len(entries),
                "iteration": it,
                "kind": kind,
                "backup_nr": bk,
                "step_length": pending_step if kind != "initial" else math.nan,
                "damping": math.nan,
                "n_depth_constrained": 0,
            }
            rec.update(_entry_extra(lines, i))
            entries.append(rec)
            current = rec
            i += 1
            continue

        m = _RE_STEP.search(s)
        if m:
            pending_step = _num(m.group(1))
            i += 1
            continue
        m = _RE_DAMP.search(s)
        if m and current is not None:
            current["damping"] = _num(m.group(1))
            i += 1
            continue
        if s.startswith("NO step length damping applied") and current is not None:
            current["damping"] = 1.0
            i += 1
            continue
        if _RE_CONSTR.search(s):
            if current is not None:
                current["n_depth_constrained"] += 1
            i += 1
            continue
        m = _RE_AVG.search(ln)
        if m and current is not None:
            vals = _numbers(m.group(2))
            key = "abs" if m.group(1) else "mean"
            sh = next((x for x in shifts if x["entry"] == current["entry"]), None)
            if sh is None:
                sh = {
                    "entry": current["entry"],
                    "iteration": current["iteration"],
                    "kind": current["kind"],
                    "backup_nr": current["backup_nr"],
                }
                shifts.append(sh)
            names = ["dot", "dx", "dy", "dz"]
            for k, name in enumerate(names):
                sh[("a" + name) if key == "abs" else name] = vals[k] if k < len(vals) else math.nan
            i += 1
            continue
        if s.startswith("Velocity adjustments") or s.startswith("Velocity readjustments"):
            rows, i2 = _parse_velocity_block(lines, i)
            if s.startswith("Velocity adjustments") and current is not None:
                for r in rows:
                    r.update(
                        entry=current["entry"], iteration=current["iteration"], readjusted=False
                    )
                    vel_rows.append(r)
            elif s.startswith("Velocity readjustments"):
                pending_readj = rows  # attached to the backup entry that follows
                for r in pending_readj:
                    r.update(entry=len(entries), iteration=-1, readjusted=True)
                    vel_rows.append(r)
            i = max(i2, i + 1)
            continue
        if s.startswith("Adjusted station corrections") and current is not None:
            rows, i2 = _parse_stacorr_block(lines, i)
            kind_p = i + 1 < n and "ptcor" in lines[i + 1]
            if (
                not sta_tables
                or sta_tables[-1]["entry"] != current["entry"]
                or (kind_p and "P" in sta_tables[-1])
            ):
                sta_tables.append({"entry": current["entry"]})
            sta_tables[-1]["P" if kind_p else "S"] = rows
            i = max(i2, i + 1)
            continue

        # ---- input data section -------------------------------------------------------
        m = _RE_READ.search(s)
        if m:
            c = readings.setdefault(m.group(1), [0, 0, 0])
            if m.group(2) is not None:
                c[0] += int(m.group(2))
                c[1] += int(m.group(3))
                c[2] += int(m.group(4))
            else:  # total only (nsp != 2); a 4th element marks "P/S split unknown"
                c[0] += int(m.group(5))
                if len(c) == 3:
                    c.append(-1)
            i += 1
            continue
        m = re.match(r"Total number of readings:\s*(\d+)", s)
        if m:
            read_totals.append(int(m.group(1)))
        m = re.match(r"Total number of observations is:\s*(\d+)", s)
        if m:
            obs_totals.append(int(m.group(1)))
        m = re.match(r"neqs=\s*(\d+)\s+nshot=\s*(\d+)\s+nltot=\s*(\d+)", s)
        if m and "nshot" not in summary:
            res.n_events = int(m.group(1))
            summary["nshot"] = int(m.group(2))
            summary["nltot"] = int(m.group(3))
        m = re.match(r"Number of unknowns.*nvar\s*=\s*(\d+)", s)
        if m and "nvar" not in summary:
            summary["nvar"] = int(m.group(1))

        # ---- stop reasons and counters -----------------------------------------------
        for pat, reason in _STOP_REASONS:
            if pat in s and not s.startswith("c "):
                if reason == "four_backups" and s.startswith("4 times"):
                    continue
                stop_reasons.append(reason)
                break
        if "Event cannot be located" in s:
            n_cannot_locate += 1
        if "error in ludecp" in s:
            ludecp_error = True
        if "ERROR OPENING FILE" in s:
            open_error = s
        if s.startswith("TOTAL CPU-sec"):
            n_cpu += 1
            mm = re.search(r"=\s*(\S+)", s)
            if mm:
                summary["cpu_sec"] = _num(mm.group(1))
        if s.startswith("Events with") and "suspicious" in s:
            j = i + 1
            k_sus = 0
            while j < n and not lines[j].lstrip().startswith(("^^^", "ZERO", "~~~")):
                if lines[j].strip().startswith("Event#"):
                    k_sus += 1
                j += 1
                if j - i > 5000:
                    break
            if k_sus:
                suspicious += k_sus
                state = f"iteration {current['iteration']}" if current else "initial state"
                warnings.append(
                    f"line {i + 1}: {k_sus} event(s) with |AVRES| > 1.0 s are suspicious "
                    f"(evaluation after {state})"
                )
        if "*** WARNING: the data variance has increased" in s:
            warnings.append(f"line {i + 1}: {s}")

        # ---- final sections ----------------------------------------------------------
        if "output final hypocenters" in s:
            in_final_hypo = True
        elif in_final_hypo:
            hm = _RE_HYPO.match(ln)
            if hm:
                hypo_rows.append(_hypo_row(hm))
            elif hypo_rows and not s:
                in_final_hypo = False
            elif hypo_rows and s.startswith("~~~"):
                in_final_hypo = False
        if "station statistics, remember nsp" in s:
            in_stats = True
        elif s.startswith("~~~ output statistics"):
            in_stats = False
        sm = _RE_STAT.match(ln) if in_stats else None
        if sm:
            nums = _numbers(sm.group(4))
            if len(nums) >= 5:
                stat_rows.append(
                    {
                        "sta": sm.group(1),
                        "phase": sm.group(2),
                        "nobs": int(sm.group(3)),
                        "avres": nums[0],
                        "avwres": nums[1],
                        "std": nums[2],
                        "wsum": nums[3],
                        "delay": nums[4],
                    }
                )
        rm = _RE_RAYROW.match(ln)
        if rm and "km/s" in ln:
            row = {
                "nlay": int(rm.group(1)),
                "top": float(rm.group(2)),
                "bottom": float(rm.group(3)),
                "vel": float(rm.group(4)),
                "nhyp": int(rm.group(5)),
                "nref": int(rm.group(6)),
                "pct_len": _num(rm.group(7)),
                "nhit": int(rm.group(8)),
                "xy_km": _num(rm.group(9)),
                "z_km": _num(rm.group(10)),
                "rflx": int(rm.group(11)),
            }
            if row["vel"] > 0:
                ray_rows.append(row)
        if s.startswith("RAY-STATISTICS FOR"):
            ray_rows = []  # single-event mode repeats the block; keep the last
        for key, pat in (
            ("n_refracted", r"Total nr of refracted rays =\s*(\d+)"),
            ("n_reflected", r"Total nr of reflected rays =\s*(\d+)"),
            ("n_direct", r"Total nr of\s+other\s+rays =\s*(\d+)"),
            ("n_rays", r"Total nr of\s+all\s+rays =\s*(\d+)"),
        ):
            mm = re.search(pat, s)
            if mm:
                summary[key] = int(mm.group(1))
        mm = re.match(r"GAPs were between\s+(\d+)\s+and\s+(\d+)", s)
        if mm:
            summary["gap_min"], summary["gap_max"] = int(mm.group(1)), int(mm.group(2))
        mm = re.search(r"average GAP was\s+(\d+)", s)
        if mm:
            summary["gap_mean"] = int(mm.group(1))
        mm = re.match(r"Straight and direct rays :\s*(\S+)\s+meters", s)
        if mm:
            summary["raytracer_error_m"] = _num(mm.group(1))
        mm = re.match(r"Average horizontal ray length =\s*(\S+)\s+km", s)
        if mm:
            summary["mean_horizontal_ray_km"] = _num(mm.group(1))
        mm = re.match(r"Average\s+vertical\s+ray length =\s*(\S+)\s+km", s)
        if mm:
            summary["mean_vertical_ray_km"] = _num(mm.group(1))
        if s.startswith("standard deviation of selected model parameters"):
            j = i + 1
            sigma = []
            while j < n and lines[j].strip() and not lines[j].strip().startswith("Rdiag"):
                sigma += _numbers(lines[j])
                j += 1
        if s.startswith("Rdiag of selected model parameters"):
            j = i + 1
            res_rdiag = []
            while j < n and lines[j].strip():
                res_rdiag += _numbers(lines[j])
                j += 1
        if s.startswith("Final rms values"):
            vals = _numbers(ln.split(":", 1)[1])
            j = i + 1
            while j < n and lines[j].strip() and _RE_FNUM_ONLY.fullmatch(lines[j].strip()):
                vals += _numbers(lines[j])
                j += 1
            summary["n_final_rms"] = len(vals)
            summary["max_event_rms"] = float(np.nanmax(vals)) if vals else math.nan

        # ---- warnings ---------------------------------------------------------------
        m = re.search(r"WARNING: Station = (\S+)\s+!!! Variance", s)
        if m:
            neg_var.append(m.group(1))
        elif any(p in s for p in _WARN_PATTERNS) and not s.startswith("c "):
            if s in ("WARNING:", "WARNING"):
                nxt = lines[i + 1].strip() if i + 1 < n else ""
                warnings.append(f"line {i + 1}: WARNING: {nxt}".rstrip())
                if nxt:
                    i += 1  # the message line belongs to this warning
            elif "STOP" in s and not s.startswith("STOP"):
                pass
            else:
                warnings.append(f"line {i + 1}: {s}")
        i += 1

    if neg_var:
        warnings.append(
            f"negative station residual variance (data inconsistent) for {len(neg_var)} "
            f"station-phase(s): {', '.join(sorted(set(neg_var)))}"
        )
    _finalise(
        res,
        entries,
        shifts,
        vel_rows,
        sta_tables,
        stop_reasons,
        n_cpu,
        obs_totals,
        read_totals,
        readings,
        warnings,
        hypo_rows,
        stat_rows,
        ray_rows,
        summary,
        res_rdiag,
        sigma,
        n_cannot_locate,
        ludecp_error,
        open_error,
        lines,
        suspicious,
    )


_RE_FNUM_ONLY = re.compile(r"(?:\s*(?:" + _NUM + r"))+")


def _hypo_row(hm: re.Match[str]) -> dict[str, Any]:
    yymmdd, hhmm = hm.group(2), hm.group(3)
    try:
        year = _year_from_yy(int(yymmdd[:2]), 1970)
        t = pd.Timestamp(
            year=year,
            month=int(yymmdd[2:4]),
            day=int(yymmdd[4:6]),
            hour=int(hhmm[:2]),
            minute=int(hhmm[2:]),
            tz="UTC",
        ) + pd.Timedelta(seconds=float(hm.group(4)))
    except (ValueError, OverflowError):
        t = pd.NaT
    return {
        "eq": int(hm.group(1)),
        "time": t,
        "lat": parse_velest_lat(hm.group(5)),
        "lon": parse_velest_lon(hm.group(6)),
        "depth_km": _num(hm.group(7)),
        "mag": _num(hm.group(8)),
        "nobs": int(hm.group(9)),
        "rms": _num(hm.group(10)),
        "x": _num(hm.group(11)),
        "y": _num(hm.group(12)),
        "z": _num(hm.group(13)),
    }


def _finalise(  # noqa: C901
    res: VelestRunResult,
    entries: list[dict[str, Any]],
    shifts: list[dict[str, Any]],
    vel_rows: list[dict[str, Any]],
    sta_tables: list[dict[str, Any]],
    stop_reasons: list[str],
    n_cpu: int,
    obs_totals: list[int],
    read_totals: list[int],
    readings: dict[str, list[int]],
    warnings: list[str],
    hypo_rows: list[dict[str, Any]],
    stat_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
    summary: dict[str, float | int],
    res_rdiag: list[float],
    sigma: list[float],
    n_cannot_locate: int,
    ludecp_error: bool,
    open_error: str | None,
    lines: list[str],
    suspicious: int,
) -> None:
    # iterations
    it = pd.DataFrame(entries)
    for c in ITERATION_COLUMNS:
        if c not in it:
            it[c] = math.nan if c not in ("kind",) else ""
    if len(it):
        it = it[ITERATION_COLUMNS].copy()
        nxt_iter = it["iteration"].shift(-1)
        nxt_run = it["run"].shift(-1)
        it["accepted"] = (nxt_iter != it["iteration"]) | (nxt_run != it["run"])
        for c in ("run", "entry", "iteration", "backup_nr", "n_depth_constrained"):
            it[c] = it[c].astype("int64")
        for c in ("n_direct", "n_refracted", "n_reflected", "n_all"):
            it[c] = it[c].astype("Int64") if it[c].notna().any() else it[c]
        it["accepted"] = it["accepted"].astype(bool)
    else:
        it = _empty(ITERATION_COLUMNS)
        it["accepted"] = it["accepted"].astype(bool)
    res.iterations = it
    # readjustment velocity rows belong to the following (backup) entry
    if vel_rows and len(it):
        for r in vel_rows:
            if r["iteration"] == -1:
                e = min(r["entry"], len(it) - 1)
                r["entry"] = e
                r["iteration"] = int(it.loc[e, "iteration"])
    models = pd.DataFrame(vel_rows)
    if len(models):
        models["phase"] = np.where(models.pop("model") == 1, "P", "S")
        models = models[MODEL_COLUMNS].sort_values(["entry", "phase", "layer"], kind="stable")
        models = models.reset_index(drop=True)
    else:
        models = _empty(MODEL_COLUMNS)
    res.models = models
    sh = pd.DataFrame(shifts)
    for c in SHIFT_COLUMNS:
        if c not in sh:
            sh[c] = math.nan
    res.hypocentre_shifts = sh[SHIFT_COLUMNS].reset_index(drop=True)

    # final model: last "Velocity adjustments" rows that belong to a printed (accepted) state
    final_rows = pd.DataFrame()
    printed = models[~models["readjusted"]] if len(models) else models
    if len(printed):
        final_rows = printed[printed["entry"] == printed["entry"].max()]
    res.final_model = _model_from_rows(final_rows, res.initial_model)
    if res.final_model is None and not len(printed) and len(it):
        res.final_model = res.initial_model  # no velocity adjustment printed: unchanged

    # events / observations
    res.n_obs = sum(obs_totals) if obs_totals else (sum(read_totals) if read_totals else None)
    if "neqs" in res.control:  # single-event mode prints neqs=1 for every event
        res.n_events = int(res.control["neqs"])
    elif res.n_events is None:
        res.n_events = len(hypo_rows) or None

    # stations
    st = pd.DataFrame(stat_rows)
    if len(st):
        st = st[STATS_COLUMNS]
    else:
        st = _empty(STATS_COLUMNS)
    res.station_stats = st.reset_index(drop=True)
    res.station_corrections = _station_corrections(sta_tables, readings, res.station_stats)
    if readings and len(res.station_corrections):
        res.n_obs_p = int(res.station_corrections["nobs_p"].sum())
        res.n_obs_s = int(res.station_corrections["nobs_s"].sum())

    hyp = pd.DataFrame(hypo_rows)
    if len(hyp):
        hyp = hyp[HYPO_COLUMNS]
        hyp["time"] = pd.to_datetime(hyp["time"], utc=True)
    else:
        hyp = _empty(HYPO_COLUMNS)
    res.final_hypocentres = hyp.reset_index(drop=True)
    ray = pd.DataFrame(ray_rows)
    res.ray_statistics = ray[RAY_COLUMNS] if len(ray) else _empty(RAY_COLUMNS)
    if res_rdiag or sigma:
        res.resolution = _resolution_table(res, sigma, res_rdiag)

    res.summary = summary
    res.warnings = warnings
    res.finished = n_cpu >= 1 and (
        not res.single_event_mode or n_cpu >= int(it["run"].max() if len(it) else 1)
    )

    # stop reasons: one per run
    res.stop_reasons = stop_reasons
    if stop_reasons:
        res.stop_reason = stop_reasons[-1] if len(set(stop_reasons)) == 1 else "mixed"
    else:
        res.stop_reason = None

    # failure analysis
    reasons: list[str] = []
    last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
    if open_error is None:
        open_error = next((ln.strip() for ln in lines if "ERROR OPENING FILE" in ln), None)
    if open_error:
        reasons.append(f"file open error: {open_error}")
    if not res.finished:
        reasons.append(f"output truncated (run did not finish); last line: {last[:100]!r}")
    if ludecp_error:
        reasons.append("error in matrix decomposition (WARNING: error in ludecp)")
    fr = res.final_rms
    if len(it) and not math.isfinite(fr):
        reasons.append("non-finite RMS residual in the last iteration (numerical divergence)")
    if n_cannot_locate and (not res.single_event_mode or n_cannot_locate >= (res.n_events or 1)):
        reasons.append("insufficient data: event(s) cannot be located")
    res.failed = bool(reasons)
    res.failure_reason = "; ".join(reasons) if reasons else None
    r0 = res.initial_rms
    res.diverged = bool(len(it)) and (
        not math.isfinite(fr) or (math.isfinite(r0) and fr > r0 * (1.0 + 1e-9))
    )
    res.converged = (
        not res.failed
        and not res.diverged
        and bool(stop_reasons)
        and all(r in CONVERGED_REASONS for r in stop_reasons)
    )
    if suspicious:
        summary["n_suspicious_events"] = suspicious


def _station_corrections(
    tables: list[dict[str, Any]], readings: dict[str, list[int]], stats: pd.DataFrame
) -> pd.DataFrame:
    last = next((t for t in reversed(tables) if "P" in t), None)
    rows: dict[str, dict[str, Any]] = {}
    if last is not None:
        for name, v, dv in last["P"]:
            rows[name] = {"sta": name, "ptcor": v, "dptcor": dv}
        for name, v, dv in last.get("S", []):
            rows.setdefault(name, {"sta": name})
            rows[name].update(stcor=v, dstcor=dv)
    elif len(stats):
        for sta, grp in stats.groupby("sta", sort=False):
            r: dict[str, Any] = {"sta": sta}
            for _, row in grp.iterrows():
                r["ptcor" if row["phase"] == "P" else "stcor"] = row["delay"]
            rows[str(sta)] = r
    if not rows:
        return _empty(STACORR_COLUMNS)
    df = pd.DataFrame(list(rows.values()))
    for c in ("ptcor", "stcor", "dptcor", "dstcor"):
        if c not in df:
            df[c] = math.nan
    _fill_from_stats(df, stats)
    _fill_counts(df, readings, stats)
    return df[STACORR_COLUMNS].reset_index(drop=True)


def _fill_from_stats(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Fill ``ptcor``/``stcor`` still missing from the station statistics ``delay`` column.

    With ``nsp=3`` VELEST prints no S station-correction table (only the P one); the S delay
    is only available in the statistics (``velest.f`` 6150-6215).
    """
    if not len(stats):
        return
    for ph, col in (("P", "ptcor"), ("S", "stcor")):
        sub = stats[stats["phase"] == ph].set_index("sta")["delay"]
        if len(sub):
            df[col] = df[col].fillna(df["sta"].map(sub))


def _fill_counts(df: pd.DataFrame, readings: dict[str, list[int]], stats: pd.DataFrame) -> None:
    """Set ``nobs_p``, ``nobs_s``, ``nobs`` per station.

    ``readings for station`` lines carry P/S counts only for ``nsp=2`` (``velest.f``
    5948-5957); otherwise the split is taken from the station statistics (``nobs`` per phase,
    printed only when a station has at least two readings of that phase, 6062-6215) and the
    total from the ``readings`` line: a phase missing from the statistics gets ``total`` minus
    the other phase.  A station whose split cannot be recovered keeps 0 for both and only
    ``nobs`` (the total).
    """
    cnt = {
        (str(a), str(b)): int(n)
        for a, b, n in zip(stats["sta"], stats["phase"], stats["nobs"], strict=True)
    }
    cols: dict[str, list[int]] = {"nobs_p": [], "nobs_s": [], "nobs": []}
    for sta in df["sta"]:
        r = readings.get(sta)
        if r is not None and len(r) == 3:  # P/S printed explicitly (nsp=2)
            p, s, tot = r[1], r[2], r[0]
        else:
            p0, s0 = cnt.get((sta, "P")), cnt.get((sta, "S"))
            tot = r[0] if r is not None else (p0 or 0) + (s0 or 0)
            if p0 is not None and s0 is None:
                s0 = max(tot - p0, 0)
            elif s0 is not None and p0 is None:
                p0 = max(tot - s0, 0)
            p, s = p0 or 0, s0 or 0
        cols["nobs_p"].append(p)
        cols["nobs_s"].append(s)
        cols["nobs"].append(tot)
    for k, v in cols.items():
        df[k] = pd.Series(v, index=df.index, dtype="int64")


def _resolution_table(res: VelestRunResult, sigma: list[float], rdiag: list[float]) -> pd.DataFrame:
    n = max(len(sigma), len(rdiag))
    names = ["ot1", "x1", "y1", "z1"]
    m = res.initial_model
    if m is not None:
        names += [f"vp{k + 1}" for k in range(len(m.p.layers))]
        if m.s is not None:
            names += [f"vs{k + 1}" for k in range(len(m.s.layers))]
    if len(names) != n:
        names = [f"p{k + 1}" for k in range(n)]
    sg = sigma + [math.nan] * (n - len(sigma))
    rd = rdiag + [math.nan] * (n - len(rdiag))
    return pd.DataFrame({"param": names, "sigma": sg[:n], "rdiag": rd[:n]})


# ----------------------------------------------------------------------------------------
# File / directory readers
# ----------------------------------------------------------------------------------------


def read_velest_out(path: str | Path) -> VelestRunResult:
    """Read and parse a VELEST ``.OUT`` print file (``.gz`` is decompressed).

    A missing or unreadable file gives a failed result instead of an exception.
    """
    p = Path(path)
    try:
        text = _read_text(p)
    except OSError as exc:
        res = VelestRunResult(source=str(p))
        res.failure_reason = f"cannot read {p}: {exc}"
        return res
    return parse_velest_out(text, source=str(p))


_OUT_NAMES = ("velest.out", "velest.OUT", "VELEST.OUT", "velest.OUT.gz", "velest.out.gz")


def _find_file(d: Path, names: list[str]) -> Path | None:
    for nm in names:
        for cand in (d / nm, d / (nm + ".gz")):
            if nm and cand.is_file():
                return cand
    return None


def read_run_dir(
    path: str | Path,
    *,
    out_file: str | Path | None = None,
    legacy_mirror: bool = False,
) -> VelestRunResult:
    """Combine ``.OUT`` + output CNV + output STA (+ ``velout.mod``) of a run directory.

    The print file is ``out_file`` if given, else the file named in ``velest.cmn``,
    else ``velest.out`` / ``*.OUT`` / ``*.out`` in ``path``.  The CNV and station file
    names are taken from the control echo of the print file (``velout.cnv`` /
    ``velout.sta`` by default).  Missing or unreadable companion files add a warning;
    nothing here raises.

    Parameters
    ----------
    legacy_mirror
        Passed to :func:`osv.io.velest.read_cnv` / :func:`~osv.io.velest.read_stations`
        for point-mirrored legacy files.

    Returns
    -------
    VelestRunResult
        With ``cnv``, ``stations`` and ``velout_model`` filled where the files exist.
    """
    d = Path(path)
    out: Path | None
    if out_file is not None:
        out = Path(out_file)
        if not out.is_absolute() and not out.exists():
            out = d / out
    else:
        names: list[str] = []
        cmn = d / "velest.cmn"
        if cmn.is_file():
            try:
                from osv.io.velest import read_control

                names.append(read_control(cmn).outfile)
            except Exception as exc:  # noqa: BLE001
                logger.debug("cannot read %s: %s", cmn, exc)
        names += list(_OUT_NAMES)
        out = _find_file(d, [nm for nm in names if nm])
        if out is None:
            hits = sorted(
                [*d.glob("*.OUT"), *d.glob("*.out"), *d.glob("*.OUT.gz"), *d.glob("*.out.gz")]
            )
            out = hits[0] if hits else None
    if out is None:
        res = VelestRunResult(source=str(d))
        res.failure_reason = f"no VELEST print output (velest.out / *.OUT) in {d}"
        return res
    res = read_velest_out(out)

    files = {k.lower(): v for k, v in res.control_files.items()}
    cnv_name = next((v for k, v in files.items() if "final hypocenters" in k), "velout.cnv")
    sta_name = next((v for k, v in files.items() if "new station corrections" in k), "velout.sta")
    cnv_path = _find_file(d, [cnv_name, "velout.cnv"])
    sta_path = _find_file(d, [sta_name, "velout.sta"])
    mod_path = _find_file(d, ["velout.mod"])
    if cnv_path is not None:
        try:
            res.cnv = parse_cnv(
                _read_text(cnv_path), legacy_mirror=legacy_mirror, source=str(cnv_path)
            )
        except Exception as exc:  # noqa: BLE001
            res.warnings.append(f"cannot read output CNV {cnv_path.name}: {exc}")
    if sta_path is not None:
        try:
            res.stations = parse_stations(
                _read_text(sta_path), legacy_mirror=legacy_mirror, source=str(sta_path)
            )
        except Exception as exc:  # noqa: BLE001
            res.warnings.append(f"cannot read output station file {sta_path.name}: {exc}")
    if mod_path is not None:
        try:
            res.velout_model = parse_model(_read_text(mod_path), source=str(mod_path))
        except Exception as exc:  # noqa: BLE001
            res.warnings.append(f"cannot read {mod_path.name}: {exc}")
    return res
