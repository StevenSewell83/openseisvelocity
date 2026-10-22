"""Assemble a complete FMTOMO run directory (model, geometry, data and control files).

:func:`assemble_run_dir` writes every file the ``tomo3d``/``tomo3di`` drivers read, validates the
cross-file consistency first (so mistakes surface here, not as a Fortran crash) and returns a
:class:`RunManifest`.

Geometry conventions: in a ``tomo3d`` (relocate none/linear) run the *sources* of the files
are the events and the *receivers* the stations; in a ``tomo3di`` (nonlinear relocation) run the
roles swap (stations are FMTOMO sources, events are FMTOMO receivers, see
:mod:`osv.fmtomo.controls`).  The function does not care which is which; it only checks that the
three tables agree.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from osv.fmtomo.controls import DIR_P, DIR_S, FmtomoControls, RunControls
from osv.io.fmtomo import (
    FmtomoFormatError,
    GridSave,
    InterfaceGrids,
    PropGrid,
    Receivers,
    Source,
    VelocityGrids,
    validate_model,
    validate_receivers,
    validate_sources,
    write_gridsave,
    write_interfaces,
    write_observed,
    write_propgrid,
    write_receivers,
    write_source_ids,
    write_sources,
    write_vgrids,
)
from osv.io.fmtomo.extras import format_src_dat
from osv.io.fmtomo.times import INVALID_OBSERVED_BELOW

log = logging.getLogger(__name__)

__all__ = [
    "AK135_FILES",
    "RunInputs",
    "RunManifest",
    "RunDirError",
    "assemble_joint_run_dirs",
    "assemble_run_dir",
    "derive_src_dat",
    "find_ak135_dir",
]

AK135_FILES = ("ak135.tbl", "ak135.hed")
ENV_AK135_DIR = "OSV_AK135_DIR"


class RunDirError(ValueError):
    """The inputs of a run directory are inconsistent (the message lists every problem)."""


@dataclass
class RunManifest:
    """Files written by :func:`assemble_run_dir` (relative name -> absolute path)."""

    root: Path
    driver: str
    tomo_input: str
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return sorted(self.files)

    def as_json(self) -> dict[str, object]:
        """JSON-able summary for the stage ``manifest.json``."""
        return {
            "root": str(self.root),
            "driver": self.driver,
            "tomo_input": self.tomo_input,
            "files": {k: str(v) for k, v in sorted(self.files.items())},
            "warnings": self.warnings,
        }


@dataclass
class RunInputs:
    """Model, geometry and data of one run directory."""

    grids: VelocityGrids
    interfaces: InterfaceGrids
    propgrid: PropGrid
    sources: list[Source]
    receivers: Receivers
    otimes: pd.DataFrame


def find_ak135_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Directory holding ``ak135.tbl``/``ak135.hed`` (only needed for teleseismic sources)."""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get(ENV_AK135_DIR):
        cands.append(Path(os.environ[ENV_AK135_DIR]))
    cands.append(Path(sys.prefix) / "share" / "osv-tools")
    cands.append(Path(__file__).resolve().parents[3] / "build" / "share" / "osv-tools")
    for c in cands:
        if all((c / n).is_file() for n in AK135_FILES):
            return c
    raise RunDirError(
        "teleseismic sources need the ak135 tables but none were found (tried "
        f"{', '.join(str(c) for c in cands)}); build/install the tools or set {ENV_AK135_DIR}"
    )


def derive_src_dat(
    otimes: pd.DataFrame, n_blocks: int, event_ids: Sequence[int] | None = None
) -> pd.DataFrame:
    """``src.dat`` table of the ``tomo3di`` geometry from the valid observed times.

    One block per FMTOMO source (a station); ``gridloc`` reads each pick as
    ``idtto, tto(i, idtto)``, so ``id`` is the event number.  ``event_ids`` gives it per
    ``otimes`` row (``mapping["event_idx"]`` of :func:`osv.fmtomo.stage.to_nonlinear_geometry`);
    by default the ``receiver`` column is used, which is the event only if the receivers are
    numbered by event.  Rows are sorted by block then id.
    """
    ids = otimes["receiver"] if event_ids is None else np.asarray(event_ids)
    ids = pd.Series(np.asarray(ids), otimes.index)
    ok = (otimes["time"] > 0).to_numpy()
    df = pd.DataFrame(
        {
            "block": otimes.loc[ok, "source"].astype("int64").to_numpy(),
            "id": ids[ok].astype("int64").to_numpy(),
            "time": otimes.loc[ok, "time"].astype(float).to_numpy(),
            "uncertainty": otimes.loc[ok, "uncertainty"].astype(float).to_numpy(),
        }
    ).sort_values(["block", "id"], kind="stable")
    df.attrs["n_blocks"] = int(n_blocks)
    return df.reset_index(drop=True)


def _expected_rays(receivers: Receivers) -> set[tuple[int, int, int]]:
    rays = receivers.rays
    k = rays.groupby("receiver").cumcount() + 1
    return set(zip(rays["receiver"], rays["source"], k, strict=True))


def _validate(
    ctl: RunControls,
    stations_are_sources: bool,
    inp: RunInputs,
    station_terms: Mapping[int, float] | None,
    terms_mode: str,
) -> list[str]:
    problems: list[str] = []
    ni = inp.interfaces.n_interfaces
    try:
        validate_model(inp.grids, inp.interfaces, inp.propgrid)
    except FmtomoFormatError as e:
        problems.append(str(e))
    try:
        validate_sources(inp.sources, n_interfaces=ni, n_vtypes=inp.grids.n_types)
    except FmtomoFormatError as e:
        problems.append(str(e))
    try:
        validate_receivers(inp.receivers, inp.sources)
    except FmtomoFormatError as e:
        problems.append(str(e))
    n_rays = len(inp.receivers.rays)
    n_rec = len(inp.receivers.positions)
    n_src = len(inp.sources)
    if len(inp.otimes) != n_rays:
        problems.append(
            f"otimes has {len(inp.otimes)} rows but receivers.in requests {n_rays} rays "
            f"({n_src} sources, {n_rec} receivers); fm3d writes one arrival per ray and "
            "residuals compares them row by row"
        )
    else:
        got = set(
            zip(
                inp.otimes["receiver"].astype(int),
                inp.otimes["source"].astype(int),
                inp.otimes["ray"].astype(int),
                strict=True,
            )
        )
        exp = _expected_rays(inp.receivers)
        if got != exp:
            miss, extra = sorted(exp - got)[:3], sorted(got - exp)[:3]
            problems.append(
                f"otimes (receiver, source, ray) keys differ from receivers.in: missing {miss}, "
                f"unexpected {extra} (first three shown)"
            )
    if ctl.invert3d.source_inversion[0]:
        missing = [i for i, s in enumerate(inp.sources, 1) if s.uncertainty is None]
        if missing:
            problems.append(
                f"source inversion needs a priori uncertainties (Source.uncertainty) for every "
                f"source; missing for {len(missing)} (first: {missing[:5]})"
            )
        if ctl.source_ids is not None:
            bad = [i for i in ctl.source_ids if not 1 <= i <= n_src]
            if bad:
                problems.append(f"invert_source_ids {bad[:5]} are not among the {n_src} sources")
    nl = inp.grids.n_layers
    fg = ctl.frechgen
    if ctl.invert3d.velocity_inversion[0] and fg.velocity_derivs > 0:
        bad = [i for i in fg.velocity_grid_ids if not 1 <= i <= nl]
        if bad:
            problems.append(f"frechgen velocity layers {bad} exceed the {nl} velocity grids")
    if ctl.invert3d.interface_inversion[0]:
        bad = [i for i in fg.interface_ids if not 1 <= i <= ni]
        if bad:
            problems.append(f"frechgen interfaces {bad} exceed the {ni} interfaces")
    if terms_mode == "preapply" and station_terms is None:
        problems.append("station_terms='preapply' but no station terms were given")
    if terms_mode != "preapply" and station_terms is not None:
        problems.append("station terms given but fmtomo.station_terms is 'none'")
    if station_terms is not None:
        ids = set(station_terms)
        col = "source" if stations_are_sources else "receiver"
        unknown = sorted(ids - set(inp.otimes[col].astype(int)))
        if unknown:
            problems.append(f"station terms for unknown station ids (otimes {col}): {unknown[:5]}")
    return problems


def _apply_terms(
    otimes: pd.DataFrame, terms: Mapping[int, float], stations_are_sources: bool
) -> pd.DataFrame:
    col = "source" if stations_are_sources else "receiver"
    corr = otimes[col].astype(int).map(lambda i: float(terms.get(int(i), 0.0)))
    out = otimes.copy()
    valid = out["time"] >= INVALID_OBSERVED_BELOW
    out.loc[valid, "time"] = out.loc[valid, "time"] - corr[valid]
    return out


def assemble_run_dir(
    root: str | os.PathLike[str],
    grids: VelocityGrids,
    interfaces: InterfaceGrids,
    propgrid: PropGrid,
    sources: list[Source],
    receivers: Receivers,
    otimes: pd.DataFrame,
    controls: FmtomoControls,
    ref: bool = True,
    *,
    phase: str | None = None,
    station_terms: Mapping[int, float] | None = None,
    src_dat: pd.DataFrame | None = None,
    ak135_dir: str | os.PathLike[str] | None = None,
) -> RunManifest:
    """Write all files of one FMTOMO run directory and return the manifest.

    Parameters
    ----------
    root
        Directory to fill (created; existing files with the same names are overwritten).
    grids, interfaces, propgrid
        The model (written as ``vgrids.in``, ``interfaces.in``, ``propgrid.in``).
    sources, receivers, otimes
        ``sources.in``, ``receivers.in`` and ``otimes.dat``; ``otimes`` has the columns of
        :data:`osv.io.fmtomo.times.OBSERVED_COLUMNS`.
    controls
        From :func:`osv.fmtomo.controls.build_controls`.
    ref
        Also write the reference copies the drivers copy to the current files at the start
        (``vgridsref.in``, ``interfacesref.in``, ``sourcesref.in`` and, for the nonlinear
        driver, ``otimesref.dat``/``receiversref.in``).  Pass False to continue a run whose
        reference files already exist.
    phase
        Which phase's controls to use (needed when several were built).
    station_terms
        ``{station id: seconds}`` subtracted from the observed times (only with
        ``fmtomo.station_terms = 'preapply'``; ids are receiver ids, or source ids for the
        nonlinear geometry).
    src_dat
        ``src.dat`` table for nonlinear runs (default: derived from ``otimes``).
    ak135_dir
        Where to find the ak135 tables (copied only for teleseismic sources).

    Raises
    ------
    RunDirError
        If the inputs are inconsistent; the message lists every problem found.
    """
    root = Path(root)
    ctl = controls.runs[phase] if phase else controls.run
    sar = controls.stations_are_sources
    inp = RunInputs(grids, interfaces, propgrid, sources, receivers, otimes)
    problems = _validate(ctl, sar, inp, station_terms, controls.station_terms)
    if ctl.nonlinear and src_dat is None and otimes["time"].gt(0).sum() == 0:
        problems.append("nonlinear relocation without any valid observed time")
    if problems:
        raise RunDirError("cannot assemble the run directory:\n  - " + "\n  - ".join(problems))
    obs = otimes if station_terms is None else _apply_terms(otimes, station_terms, sar)

    root.mkdir(parents=True, exist_ok=True)
    man = RunManifest(root, ctl.driver, ctl.tomo_run_name)

    def put(name: str, path: Path) -> None:
        man.files[name] = path

    def write(name: str, fn, *args) -> None:  # type: ignore[no-untyped-def]
        put(name, Path(fn(root / name, *args)))

    write("propgrid.in", write_propgrid, propgrid)
    write("vgrids.in", write_vgrids, grids)
    write("interfaces.in", write_interfaces, interfaces)
    write("sources.in", write_sources, sources)
    write("receivers.in", write_receivers, receivers)
    write("otimes.dat", write_observed, obs)
    if ref:
        write("vgridsref.in", write_vgrids, grids)
        write("interfacesref.in", write_interfaces, interfaces)
        write("sourcesref.in", write_sources, sources)
        if ctl.nonlinear:
            write("receiversref.in", write_receivers, receivers)
            write("otimesref.dat", write_observed, obs)
    for name, text in ctl.texts().items():
        (root / name).write_text(text)
        put(name, root / name)
    if ctl.invert3d.source_inversion[0]:
        ids = ctl.source_ids if ctl.source_ids is not None else tuple(range(1, len(sources) + 1))
        write("sourcederivs.in", write_source_ids, list(ids))
    if ctl.nonlinear:
        sd = src_dat if src_dat is not None else derive_src_dat(obs, len(sources))
        (root / "src.dat").write_text(format_src_dat(sd))
        put("src.dat", root / "src.dat")
        entries = [
            GridSave(i, tuple(range(1, len(s.paths) + 1)), (1,) * len(s.paths))
            for i, s in enumerate(sources, start=1)
            if s.paths
        ]
        write("gridsave.in", write_gridsave, entries)
    if any(s.teleseismic for s in sources):
        src_dir = find_ak135_dir(ak135_dir)
        for n in AK135_FILES:
            shutil.copy2(src_dir / n, root / n)
            put(n, root / n)
    man.warnings.extend(controls.notes)
    return man


def assemble_joint_run_dirs(
    root: str | os.PathLike[str],
    inputs: Mapping[str, RunInputs],
    controls: FmtomoControls,
    *,
    station_terms: Mapping[str, Mapping[int, float]] | None = None,
) -> RunManifest:
    """Experimental: ``invert_p/``, ``invert_s/`` and the coupling files of ``tomo3dj``.

    ``inputs`` maps ``"P"``/``"S"`` to the :class:`RunInputs` of each phase.  Requires
    controls built for ``phases = [P, S]`` with nonlinear relocation.
    """
    if controls.joint is None:
        raise RunDirError(
            "controls hold no joint P+S setup: build them with phases=['P','S'] and "
            "relocate='nonlinear' (package extras)"
        )
    if set(inputs) != {"P", "S"}:
        raise RunDirError(f"need inputs for P and S, got {sorted(inputs)}")
    root = Path(root)
    man = RunManifest(root, controls.driver, "tomo3dj.in")
    for ph, sub in (("P", DIR_P), ("S", DIR_S)):
        i = inputs[ph]
        m = assemble_run_dir(
            root / sub,
            i.grids,
            i.interfaces,
            i.propgrid,
            i.sources,
            i.receivers,
            i.otimes,
            controls,
            phase=ph,
            station_terms=None if station_terms is None else station_terms.get(ph),
        )
        for n, p in m.files.items():
            man.files[f"{sub}/{n}"] = p
    for name, text in controls.joint.texts().items():
        (root / name).write_text(text)
        man.files[name] = root / name
    man.warnings.extend(controls.notes)
    return man
