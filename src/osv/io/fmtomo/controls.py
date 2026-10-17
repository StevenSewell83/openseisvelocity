"""Run-control files of ``fm3d`` and ``tomo3d``/``invert3d``.

Each class is a dataclass with ``parse``/``read`` (positional, as the Fortran reads) and
``format``/``write`` (the layout of ``fortran/fmtomo/source/inputfiles``).  Fortran references
give the ``read`` statements the layout is derived from.

* :class:`ModeSet` ``mode_set.in`` - ``3dfm_main.f90:44-51`` (six list-directed logicals, the
  trailing name is ignored).
* :class:`TomoRun` ``tomo3d.in`` - read by the ``tomo3d`` ksh script (``bin/tomo3d:77-81``) with
  ``read -u3 NI`` into integer variables: **values must stand alone on their line** (a
  ``c:`` comment would break the script), so none is written.
* :class:`InvIter` ``inviter.in`` - one integer, ``invert3d.f90:254-255``; written by tomo3d.
* :class:`Invert3d` ``invert3d.in`` - ``invert3d.f90:187-217`` (file names ``(a26)``).  The
  receivers line (8th name) is skipped by ``invert3d`` (``:199``).
* :class:`Frechgen` ``frechgen.in`` - ``frechgen.f90:42-74``.  It also reads lines 23-25 of
  ``invert3d.in`` (the 0/1 flags for velocity/interface/source inversion, ``:49-53``) and
  ignores its own derivative selection for a class whose flag is 0.
* :class:`FrechetSpec` ``frechet.in`` - written by ``frechgen`` (``:80-156``), read by ``fm3d``
  (``frechet.f90:23-99``) and ``invert3d`` (``:261-284``).
* :class:`Residuals` ``residuals.in`` - ``residuals.f90:40-44`` (``(a27)``, ``(a27)``, ``(a26)``,
  ``(a26)``, ``(a26)``).
* :class:`Scorrect` ``scorrect.in`` - ``scorrect.f90:25-31`` (list-directed: file names must
  not contain ``/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._control import ControlFile, F, L
from ._fortran import FmtomoFormatError, RecordReader, fint, fints, read_text, write_text
from ._texts import FRECHGEN_TAIL, MODE_SET_TAIL, RESIDUALS_TAIL, TOMO3D_TAIL

__all__ = [
    "FrechetSpec",
    "Frechgen",
    "InvIter",
    "Invert3d",
    "ModeSet",
    "Residuals",
    "Scorrect",
    "TomoRun",
]

_C70 = "c" * 70
_NAMES = "c: "


@dataclass
class ModeSet(ControlFile):
    """``mode_set.in``: run modes of ``fm3d`` (see the explanatory tail of the file)."""

    file_mode: bool = False
    no_pp_mode: bool = True
    parallel_mode: bool = False
    display_mode: bool = False
    save_rays_mode: bool = False
    save_timefields_mode: bool = False

    _layout = tuple(
        F(n, "b", n, pad=4, sep="")
        for n in (
            "file_mode",
            "no_pp_mode",
            "parallel_mode",
            "display_mode",
            "save_rays_mode",
            "save_timefields_mode",
        )
    )
    _tail = MODE_SET_TAIL


@dataclass
class TomoRun(ControlFile):
    """``tomo3d.in`` (also ``tomo3di/j/ps.in`` of the Extras): iterations and start mode.

    ``begin_from_current``: 0 start from the reference model, 1 continue from the current
    iteration.  ``start_with_forward``: only used if the former is 1: 0 start with an
    inversion step, 1 with a forward step.
    """

    n_iterations: int = 6
    begin_from_current: int = 0
    start_with_forward: int = 0

    _layout = (
        F("n_iterations", "i"),
        F("begin_from_current", "i"),
        F("start_with_forward", "i"),
    )
    _tail = TOMO3D_TAIL


@dataclass
class InvIter:
    """``inviter.in``: the current inversion iteration number."""

    iteration: int = 1

    @classmethod
    def parse(cls, text: str, source: str | Path | None = None) -> InvIter:
        return cls(RecordReader(text, source).int())

    @classmethod
    def read(cls, path: str | Path) -> InvIter:
        return cls.parse(read_text(path), path)

    def format(self) -> str:
        return f"{int(self.iteration)}\n"

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.format())


_INVERT3D_BANNER = (
    L("c" * 71),
    L("c This file contains all required input parameters and files for the"),
    L('c inversion program "invert3d.f90"'),
    L("c" * 71),
)


@dataclass
class Invert3d(ControlFile):
    """``invert3d.in``: files and regularisation parameters of the subspace inversion.

    The ``*_inversion`` tuples are ``(flag 0/1, damping, smoothing)``; for sources the last two
    are ``damp1`` (position) and ``damp2`` (origin time).
    """

    ref_vgrid_file: str = "vgridsref.in"
    vgrid_file: str = "vgrids.in"
    ref_interface_file: str = "interfacesref.in"
    interface_file: str = "interfaces.in"
    ref_source_file: str = "sourcesref.in"
    source_file: str = "sources.in"
    stimes_file: str = "stimes.dat"
    receivers_file: str = "receivers.in"
    otimes_file: str = "otimes.dat"
    mtimes_file: str = "mtimes.dat"
    rtimes_file: str = "rtimes.dat"
    frechet_in_file: str = "frechet.in"
    frechet_dat_file: str = "frechet.dat"
    inviter_file: str = "inviter.in"
    propgrid_file: str = "propgrid.in"
    min_interface_distance_km: float = 0.07
    min_velocity: float = 0.5
    remove_mean_teleseisms: int = 0
    velocity_inversion: tuple[int, float, float] = (0, 1.0, 1.0)
    interface_inversion: tuple[int, float, float] = (0, 1.0, 1.0)
    source_inversion: tuple[int, float, float] = (1, 1.0, 1.0)
    subspace_dimension: int = 20
    global_damping: float = 1.0
    second_derivative_smoothing: int = 0
    global_smoothing: float = 1.0
    earth_radius_km: float = 6371.0

    _J = ("     ", "    ")
    _layout = (
        *_INVERT3D_BANNER,
        F("ref_vgrid_file", "a", "Reference velocity grid", 26),
        F("vgrid_file", "a", "Current velocity grid", 26),
        F("ref_interface_file", "a", "Reference interface grid", 26),
        F("interface_file", "a", "Current interface grid", 26),
        F("ref_source_file", "a", "Reference source coords", 26),
        F("source_file", "a", "Current source coordinates", 26),
        F("stimes_file", "a", "Current source time perturbations", 26),
        F("receivers_file", "a", "Receiver coordinates", 26),
        F("otimes_file", "a", "Observed traveltimes", 26),
        F("mtimes_file", "a", "Model traveltimes", 26),
        F("rtimes_file", "a", "Reference teleseismic traveltimes", 26),
        F("frechet_in_file", "a", "Frechet derivative parameters", 26),
        F("frechet_dat_file", "a", "Frechet derivatives", 26),
        F("inviter_file", "a", "File indicating current inversion step", 26),
        F("propgrid_file", "a", "File containing propagation grid parameters", 26),
        F("min_interface_distance_km", "f", "Minimum distance between interfaces (km)"),
        F("min_velocity", "f", "Minimum permitted velocity (km/s)"),
        F("remove_mean_teleseisms", "i", "Remove mean from predicted teleseisms (0=no,1=yes)"),
        F("velocity_inversion", "iff", "Velocity inversion (0=no,1=yes),damp,smooth", joins=_J),
        F("interface_inversion", "iff", "Interface inversion (0=no, 1=yes),damp,smooth", joins=_J),
        F("source_inversion", "iff", "Source inversion (0=no, 1=yes),damp1,damp2", joins=_J),
        F("subspace_dimension", "i", "Subspace dimension (max=50)"),
        F("global_damping", "f", "Global damping factor (epsilon)"),
        F("second_derivative_smoothing", "i", "Apply second derivative smoothing (0=no,1=yes)"),
        F("global_smoothing", "f", "Global smoothing factor (eta)"),
        F("earth_radius_km", "f", "Earth radius in km"),
    )


@dataclass
class Residuals(ControlFile):
    """``residuals.in``: times files used by the ``residuals`` program."""

    otimes_file: str = "otimes.dat"
    mtimes_file: str = "mtimes.dat"
    source_file: str = "sourcesref.in"
    rtimes_file: str = "rtimes.dat"
    invert3d_file: str = "invert3d.in"

    _pad = 33
    _layout = (
        F("otimes_file", "a", "Observed times", 27),
        F("mtimes_file", "a", "Model times", 27),
        F("source_file", "a", "Source file", 26),
        F("rtimes_file", "a", "Reference teleseismic times", 26),
        F("invert3d_file", "a", "Inversion parameters (indicates if mean is", 26),
    )
    _tail = RESIDUALS_TAIL


@dataclass
class Scorrect(ControlFile):
    """``scorrect.in``: source correction after an inversion step.

    ``depth_correction_km`` and ``geographic_correction_deg`` are the distances by which sources
    that left the model are pulled back inside (``scorrect.f90:25-31``).
    """

    source_file: str = "sources.in"
    propgrid_file: str = "propgrid.in"
    interface_file: str = "interfaces.in"
    depth_correction_km: float = 0.1
    geographic_correction_deg: float = 0.01
    earth_radius_km: float = 6371.0

    _pad = 9
    _layout = (
        F("source_file", "w"),
        F("propgrid_file", "w"),
        F("interface_file", "w"),
        F("depth_correction_km", "f", "Depth correction (km)"),
        F("geographic_correction_deg", "f", "geographic corrrection (degrees)"),
        F("earth_radius_km", "f", "Earth radius (km)"),
        L(""),
    )


@dataclass
class Frechgen:
    """``frechgen.in``: which derivatives ``fm3d`` has to compute.

    ``velocity_derivs``/``interface_derivs``/``source_derivs``: ``-1`` all, ``>0`` the number of
    entries in the subset lists, ``0`` none.  ``velocity_type`` (``dvt``): 1, 2 or 3 (both) when
    all velocity grids are used.  The id lists are only used for a positive count; otherwise the
    two placeholder lines are skipped by the program (``frechgen.f90:60-71``).
    """

    invert3d_file: str = "invert3d.in"
    frechet_file: str = "frechet.in"
    ref_vgrid_file: str = "vgridsref.in"
    ref_source_file: str = "sourcesref.in"
    velocity_derivs: int = -1
    velocity_type: int = 1
    velocity_grid_ids: tuple[int, ...] = (2,)
    velocity_types: tuple[int, ...] = (1,)
    interface_derivs: int = 1
    interface_ids: tuple[int, ...] = (2,)
    source_derivs: int = -1
    source_ids_file: str = "sourcederivs.in"

    _PAD = 29

    @classmethod
    def parse(cls, text: str, source: str | Path | None = None) -> Frechgen:
        rd = RecordReader(text, source)
        files = [rd.word() for _ in range(4)]
        vder = rd.int()
        dvt = rd.int()
        if vder > 0:
            gids, vts = tuple(rd.ints(vder)), tuple(rd.ints(vder))
        else:
            gids, vts = _placeholder(rd.next_record()), _placeholder(rd.next_record())
        ider = rd.int()
        iids = tuple(rd.ints(ider)) if ider > 0 else _placeholder(rd.next_record())
        sder = rd.int()
        sfile = rd.word()
        return cls(*files, vder, dvt, gids, vts, ider, iids, sder, sfile)

    @classmethod
    def read(cls, path: str | Path) -> Frechgen:
        return cls.parse(read_text(path), path)

    def format(self) -> str:
        def check(n: str, v: str) -> str:
            if not v or any(c.isspace() or c in ",/;'\"" for c in v):
                raise FmtomoFormatError(
                    f"Frechgen.{n}: {v!r} cannot be read by list-directed input (blank, comma or "
                    "slash ends the value)"
                )
            return v

        if self.velocity_derivs > 0 and not (
            len(self.velocity_grid_ids) == len(self.velocity_types) == self.velocity_derivs
        ):
            raise FmtomoFormatError(
                "Frechgen: velocity_derivs must equal the number of velocity_grid_ids and "
                "velocity_types"
            )
        if self.interface_derivs > 0 and len(self.interface_ids) != self.interface_derivs:
            raise FmtomoFormatError(
                "Frechgen: interface_derivs must equal the number of interface_ids"
            )

        def line(v: str, c: str) -> str:
            return f"{v:<{self._PAD}}c: {c}"

        def ids(v: tuple[int, ...]) -> str:
            return " ".join(str(i) for i in v)

        rows = [
            line(check("invert3d_file", self.invert3d_file), "Input file for invert3d"),
            line(check("frechet_file", self.frechet_file), "File specifying derivatives for fm3d"),
            line(
                check("ref_vgrid_file", self.ref_vgrid_file),
                "File containing reference velocity grid",
            ),
            line(
                check("ref_source_file", self.ref_source_file),
                "File containing reference source locations",
            ),
            line(str(self.velocity_derivs), "Velocity derivatives (>0=subset, -1=all)"),
            line(str(self.velocity_type), "If all, type 1 (1), type 2 (2) or both (3)"),
            line(ids(self.velocity_grid_ids), "Indices of velocity grids (if subset chosen)"),
            line(ids(self.velocity_types), "Velocity types for inversion (1 or 2)"),
            line(str(self.interface_derivs), "Interface derivatives (>0=subset, -1=all)"),
            line(ids(self.interface_ids), "Indices of interface grids (if subset chosen)"),
            line(str(self.source_derivs), "Source derivatives (>0=subset, -1=all) "),
            line(
                check("source_ids_file", self.source_ids_file),
                "File specifying sources for inversion (if subset",
            ),
            " " * 32 + "chosen)",
        ]
        return "\n".join(rows) + "\n" + FRECHGEN_TAIL

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.format())


def _placeholder(record: str) -> tuple[int, ...]:
    out = []
    for tok in record.replace(",", " ").split():
        try:
            out.append(int(tok))
        except ValueError:
            break
    return tuple(out)


@dataclass
class FrechetSpec:
    """``frechet.in``: derivatives requested from ``fm3d`` (written by ``frechgen``).

    ``compute``: master switch (``frechgen`` always writes 1).  ``velocity``: ``(layer, vtype)``
    pairs, ``interfaces``: interface ids, ``sources``: source ids; the global parameter vector
    of ``frechet.dat`` follows this order (see :mod:`osv.io.fmtomo.frechet`).
    """

    compute: bool = True
    velocity: list[tuple[int, int]] = field(default_factory=list)
    interfaces: list[int] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str, source: str | Path | None = None) -> FrechetSpec:
        rd = RecordReader(text, source)
        flag = rd.int()
        if flag == 0:
            return cls(False)
        nv = rd.int()
        vel: list[tuple[int, int]] = []
        if nv > 0:
            gids, vts = rd.ints(nv), rd.ints(nv)
            vel = list(zip(gids, vts, strict=True))
        ni = rd.int()
        ifs = rd.ints(ni) if ni > 0 else []
        ns = rd.int()
        srcs = rd.ints(ns) if ns > 0 else []
        return cls(True, vel, ifs, srcs)

    @classmethod
    def read(cls, path: str | Path) -> FrechetSpec:
        return cls.parse(read_text(path), path)

    def format(self) -> str:
        if not self.compute:
            return fint(0) + "\n"
        out = [fint(1), fint(len(self.velocity))]
        if self.velocity:
            out.extend(fints([g for g, _ in self.velocity]))
            out.extend(fints([t for _, t in self.velocity]))
        out.append(fint(len(self.interfaces)))
        if self.interfaces:
            out.extend(fints(self.interfaces))
        out.append(fint(len(self.sources)))
        if self.sources:
            out.extend(fints(self.sources))
        return "\n".join(out) + "\n"

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.format())
