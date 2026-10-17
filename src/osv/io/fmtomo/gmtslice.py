"""``gmtslice.in``: control file of the ``gmtslice`` plotting-data extractor.

Read by ``gmtslice.f90:171-232``, ``:399-416`` and ``:1266-1277,1322-1325`` as one sequential
stream (the three pieces are consecutive reads of unit 10; the flags only guard the processing).
File names are fixed-width: ``(a27)`` in the first part, ``(a22)`` for the velocity/interface
output names of the second part and ``(a26)`` for the interface-contour names, so the comment
column (28) must follow the name.  The number of ``(id, file)`` pairs at the end is ``nip``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._control import ControlFile, F, L, _fmt_one
from ._fortran import FmtomoFormatError, RecordReader

__all__ = ["Gmtslice"]

_ORDINALS = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
]
_C62 = "c" * 62
_C47 = "c" * 47


def _banner(*lines: str, rule: str = _C62) -> tuple[L, ...]:
    return (L(rule), *(L(t) for t in lines), L(rule))


@dataclass
class Gmtslice(ControlFile):
    """``gmtslice.in`` (see module docstring for the read statements)."""

    vgrid_file: str = "../vgrids.in"
    ref_vgrid_file: str = "../vgridsref.in"
    interface_file: str = "../interfaces.in"
    ref_interface_file: str = "../interfacesref.in"
    earth_radius_km: float = 6371.0
    velocity_type: int = 0  # 0 = P, 1 = S
    plot_rays: int = 1
    ray_file: str = "../rays.dat"
    rays_depth_file: str = "raysd.xy"
    rays_ew_file: str = "raysew.xy"
    rays_ns_file: str = "raysns.xy"
    rays_gc_file: str = "raysgc.xy"
    plot_every_nth: int = 30
    plot_sources_receivers: int = 1
    sources_file: str = "../sources.in"
    receivers_file: str = "../receivers.in"
    sources_depth_file: str = "sourcesd.xy"
    sources_ew_file: str = "sourcesew.xy"
    sources_ns_file: str = "sourcesns.xy"
    sources_gc_file: str = "sourcesgc.xy"
    receivers_depth_file: str = "receiversd.xy"
    receivers_ew_file: str = "receiversew.xy"
    receivers_ns_file: str = "receiversns.xy"
    receivers_gc_file: str = "receiversgc.xy"
    extract_depth_slice: int = 1
    depth_slice_km: float = -30.0
    depth_bounds_file: str = "bounddp.gmt"
    depth_contour_file: str = "contint.d"
    extract_ns_slice: int = 1
    ns_slice_lon: float = 141.5
    ns_bounds_file: str = "boundns.gmt"
    extract_ew_slice: int = 1
    ew_slice_lat: float = -41.0
    ew_bounds_file: str = "boundew.gmt"
    extract_gc_slice: int = 1
    gc_point1: tuple[float, float] = (-42.0, 140.0)
    gc_point2: tuple[float, float] = (-40.0, 143.0)
    gc_bounds_file: str = "boundgc.gmt"
    relative_velocity: int = 0
    dicing_depth_slice: tuple[int, int] = (30, 30)
    depth_velocity_file: str = "grid2dvd.z"
    dicing_ns_slice: tuple[int, int] = (40, 20)
    ns_velocity_file: str = "grid2dvns.z"
    dicing_ew_slice: tuple[int, int] = (40, 20)
    ew_velocity_file: str = "grid2dvew.z"
    gc_points: tuple[int, int] = (300, 100)
    gc_velocity_file: str = "grid2dvgc.z"
    ns_interface_file_out: str = "intns.xy"
    ew_interface_file_out: str = "intew.xy"
    gc_interface_file_out: str = "intgc.xy"
    relative_depth: int = 0
    n_interfaces_plot: int = 1
    interface_dicing: tuple[int, int] = (20, 20)
    interface_bounds_file: str = "boundsint.gmt"
    interfaces: list[tuple[int, str]] = field(default_factory=lambda: [(2, "gridint1.z")])

    _layout = (
        *_banner("c Input file names"),
        F("vgrid_file", "a", "Inversion model velocity vertices", 27),
        F("ref_vgrid_file", "a", "Reference model velocity vertices", 27),
        F("interface_file", "a", "Inversion model interface vertices", 27),
        F("ref_interface_file", "a", "Reference model interface vertices", 27),
        F("earth_radius_km", "f", "Earth radius (km)"),
        F("velocity_type", "i", "Generate P(0) or S(1) velocities"),
        *_banner("c Set parameters for plotting ray paths if required"),
        F("plot_rays", "i", "Generate raypaths (0=no, 1=yes)"),
        F("ray_file", "a", "File containing raypahts", 27),
        F("rays_depth_file", "a", "File name for depth projection", 27),
        F("rays_ew_file", "a", "File name for EW projection", 27),
        F("rays_ns_file", "a", "File name for NS projection", 27),
        F("rays_gc_file", "a", "File name for GC projection", 27),
        F("plot_every_nth", "i", "Plot every nth ray point"),
        *_banner("c Set parameters for plotting sources and receivers"),
        F("plot_sources_receivers", "i", "Generate sources & receivers (0=no, 1=yes)"),
        F("sources_file", "a", "File containing sources", 27),
        F("receivers_file", "a", "File containing receivers", 27),
        F("sources_depth_file", "a", "Output source file for depth projection", 27),
        F("sources_ew_file", "a", "Output source file for EW projection", 27),
        F("sources_ns_file", "a", "Output source file for NS projection", 27),
        F("sources_gc_file", "a", "Output source file for GC projection", 27),
        F("receivers_depth_file", "a", "Output receiver file for depth projection", 27),
        F("receivers_ew_file", "a", "Output receiver file for EW projection", 27),
        F("receivers_ns_file", "a", "Output receiver file for NS projection", 27),
        F("receivers_gc_file", "a", "Output receiver file for GC projection", 27),
        L(_C62),
        L(_C62),
        L("c Set parameters for taking horizontal and vertical slices"),
        L(_C62),
        L(_C62),
        *_banner("c Slice parameters", rule=_C47),
        F("extract_depth_slice", "i", "Extract depth slice? (0=no, 1=yes)"),
        F("depth_slice_km", "f", "Depth of slice (km)"),
        F("depth_bounds_file", "a", "GMT plotting bounds for depth slice", 27),
        F("depth_contour_file", "a", "GMT depth contouring file", 27),
        F("extract_ns_slice", "i", "Extract N-S slice? (0=no, 1=yes)"),
        F("ns_slice_lon", "f", "Longitude of slice (degrees)"),
        F("ns_bounds_file", "a", "GMT plotting bounds for N-S slice", 27),
        F("extract_ew_slice", "i", "Extract E-W slice? (0=no, 1=yes)"),
        F("ew_slice_lat", "f", "Latitude of slice (degrees)"),
        F("ew_bounds_file", "a", "GMT plotting bounds for E-W slice", 27),
        F("extract_gc_slice", "i", "Extract great circle slice? (0=no,1=yes)"),
        F("gc_point1", "ff", "Lat,Long of first GC point", joins=("       ",)),
        F("gc_point2", "ff", "Lat,Long of second GC point", joins=("       ",)),
        F("gc_bounds_file", "a", "GMT plotting bounds for great circle slice", 27),
        *_banner("c Velocity grid parameters", rule=_C47),
        F("relative_velocity", "i", "0=absolute, 1=relative velocity"),
        F("dicing_depth_slice", "ii", "Dicing in theta,phi", joins=("  ",)),
        F("depth_velocity_file", "a", "GMT output file for depth slice", 22),
        F("dicing_ns_slice", "ii", "Dicing in r,theta", joins=("  ",)),
        F("ns_velocity_file", "a", "GMT output file for N-S slice", 22),
        F("dicing_ew_slice", "ii", "Dicing in r,phi", joins=("  ",)),
        F("ew_velocity_file", "a", "GMT output file for E-W slice", 22),
        F("gc_points", "ii", "Total number of GC points (horizontal, vertical)", joins=("  ",)),
        F("gc_velocity_file", "a", "GMT output file for great circle slice", 22),
        *_banner("c Interface grid parameters", rule=_C47),
        F("ns_interface_file_out", "a", "GMT output file for N-S slice", 22),
        F("ew_interface_file_out", "a", "GMT output file for E-W slice ", 22),
        F("gc_interface_file_out", "a", "GMT output file for GC slice", 22),
        L(_C62),
        L(_C62),
        L("c Set parameters for contour plots of interfaces"),
        L(_C62),
        L(_C62),
        F("relative_depth", "i", "0=absolute, 1=relative depth"),
        F("n_interfaces_plot", "i", "Number of interfaces to plot"),
        F("interface_dicing", "ii", "Dicing in theta,phi", joins=("  ",)),
        F("interface_bounds_file", "a", "GMT plotting bounds for surface", 26),
    )

    @classmethod
    def _parse_extra(cls, rd: RecordReader, kwargs: dict) -> None:
        n = kwargs["n_interfaces_plot"]
        out = []
        for _ in range(max(n, 0)):
            out.append((rd.int(), rd.a(26)))
        kwargs["interfaces"] = out

    def _extra_lines(self) -> list[str]:
        if len(self.interfaces) != self.n_interfaces_plot:
            raise FmtomoFormatError(
                f"Gmtslice: n_interfaces_plot={self.n_interfaces_plot} but "
                f"{len(self.interfaces)} interface entries"
            )
        out: list[str] = []
        for i, (iid, name) in enumerate(self.interfaces):
            word = _ORDINALS[i] if i < len(_ORDINALS) else f"{i + 1}th"
            out.append(self._with_comment(str(iid), F("i", "i", f"ID of {word} interface to plot")))
            out.append(self._with_comment(self._check(name), F("n", "a", "GMT output file", 26)))
        return out

    @staticmethod
    def _check(name: str) -> str:
        return _fmt_one("a", name, F("interfaces", "a", width=26), "Gmtslice")
