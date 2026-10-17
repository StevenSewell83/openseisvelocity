"""``grid3dg.in``: control file of the model generator ``grid3dg``.

Positional layout from ``grid3dg.f90:113-262`` (velocity layers) and ``:474-540`` (interfaces).
Notable points (verified against the code):

* Names are ``(a20)``.  The radial/lateral extent is given as **heights** (km relative to the
  Earth-radius sphere, top first), latitudes north first (degrees, N positive) and longitudes
  west first (degrees east).  ``grid3dg`` adds one cushion node on every side
  (``:200-210``): the node counts in the file are the counts *inside* the model.
* With two velocity types the first column of ``nr``, ``nlat``, ``nlon``, ``v_top`` and ``v_bottom``
  belongs to type 1 and the second to type 2 (``:190-196,221-226``).  The second pass restarts
  at the top of the file and skips 29 records (``:181-183``) whereas the shipped template needs 31
  (the layer block starts at line 32): **two velocity types do not work with the shipped template
  without editing ``grid3dg.f90``**; the Extras ``grid3dgi.f90:183`` skips 32.
* ``vtop``/``vbot`` are the velocities at the top and bottom of the model for the linear gradient
  option (``umocvg = 1``); the first number is read for type 1, the second for type 2.

:meth:`Grid3dg.expected_geometry` reproduces the node geometry that ``grid3dg`` writes to
``propgrid.in``, ``vgrids.in`` and ``interfaces.in`` (``:152-162,208-210,485-489``) so that the
three files can be cross-checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from pathlib import Path

import numpy as np

from ._control import cfloat
from ._fortran import FmtomoFormatError, RecordReader, read_text, write_text
from .grids import PropGrid

__all__ = ["Grid3dg", "InterfaceSpec", "Perturbation", "Spike", "VelocityLayerSpec"]

# ``REAL(KIND=i10), PARAMETER :: pi=3.141592653589793`` (grid3dg.f90:37) has no ``_i10``
# suffix, so the literal is a *single precision* constant (3.14159274...).  All radian values
# grid3dg writes are therefore scaled by 1 + 2.78e-8 relative to true radians (<= 4e-6 degree
# at 140 E).
PI_GRID3DG = float(np.float32(pi))

_C61 = "c" * 61
_C62 = "c" * 62
_PAD = 22


@dataclass
class Spike:
    """A spike: ``amplitude`` at ``location`` (velocity: depth, lat, lon; interface: lat, lon)."""

    amplitude: float
    location: tuple[float, ...]


@dataclass
class Perturbation:
    """Random structure, checkerboard and spikes of one layer or interface."""

    random: int = 0
    random_sd: float = 1.0
    add_prior: int = 1
    prior_uncertainty: float = 0.3
    checkerboard: int = 0
    checker_amplitude: float = 1.0
    checker_size: int = 2
    checker_use_spacing: int = 1
    spikes: int = 0
    spike_list: list[Spike] = field(default_factory=list)


@dataclass
class VelocityLayerSpec:
    """One velocity layer; tuples hold the type-1 (and optionally type-2) values."""

    n_radial: tuple[int, ...] = (12,)
    n_lat: tuple[int, ...] = (12,)
    n_lon: tuple[int, ...] = (12,)
    constant_gradient: int = 1  # 1: linear gradient between v_top and v_bottom, 0: model file
    phase: str = "P"
    model_file: str = "ak135.vel"
    model_dimension: int = 1
    v_top: tuple[float, ...] = (5.4,)
    v_bottom: tuple[float, ...] = (7.0,)
    perturbation: Perturbation = field(default_factory=Perturbation)


@dataclass
class InterfaceSpec:
    """One interface: external file or constant heights of the NW, NE and SW corner nodes (km)."""

    from_file: int = 0
    file: str = "interface1.z"
    height_nw_km: float = 0.0
    height_ne_km: float = 0.0
    height_sw_km: float = 0.0
    perturbation: Perturbation = field(default_factory=Perturbation)


def _kv(value: str, comment: str, pad: int = _PAD) -> str:
    return f"{value:<{pad}}c: {comment}"


def _pair(vals: tuple, fmt) -> str:
    return "".join(f"{fmt(v):<10}" if i < len(vals) - 1 else fmt(v) for i, v in enumerate(vals))


def _numeric_prefix(tokens: list[str], cast) -> tuple:
    out = []
    for t in tokens:
        try:
            out.append(cast(t))
        except ValueError:
            break
    return tuple(out)


def _dec(v: float, min_decimals: int) -> str:
    """Shortest exact text of ``v`` with at least ``min_decimals`` decimals (``1.50``)."""
    s = repr(float(v))
    if "e" in s or "E" in s:
        return s
    whole, _, frac = s.partition(".")
    return f"{whole}.{frac.ljust(min_decimals, '0')}"


def _file_name(name: str, what: str) -> str:
    if len(name) > 20:
        raise FmtomoFormatError(f"Grid3dg.{what}: {name!r} exceeds the (a20) field of grid3dg.f90")
    return name


@dataclass
class Grid3dg:
    """``grid3dg.in``; see the module docstring."""

    n_vtypes: int = 1
    pinchout_km: float = 0.2
    vgrids_file: str = "vgrids.in"
    interfaces_file: str = "interfaces.in"
    seed: int = 145678
    min_velocity: float = 0.2
    radial_range_km: tuple[float, float] = (1.5, -80.1)  # (top, bottom) heights
    lat_range_deg: tuple[float, float] = (-40.0, -42.0)  # (north, south)
    lon_range_deg: tuple[float, float] = (140.0, 143.0)  # (west, east)
    earth_radius_km: float = 6371.0
    propgrid_file: str = "propgrid.in"
    prop_nodes: tuple[int, int, int] = (21, 41, 41)
    prop_refinement: tuple[int, int] = (5, 10)
    prop_cushion: float = 0.05
    layers: list[VelocityLayerSpec] = field(default_factory=list)
    interface_nodes: tuple[int, int] = (12, 12)
    interfaces: list[InterfaceSpec] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    # -- parsing ---------------------------------------------------------------------------
    @classmethod
    def read(cls, path: str | Path) -> Grid3dg:
        return cls.parse(read_text(path), path)

    @classmethod
    def parse(cls, text: str, source: str | Path | None = None) -> Grid3dg:
        rd = RecordReader(text, source)
        rd.skip(3)
        n_layers = rd.int()
        n_vtypes = rd.int()
        pinch = rd.float()
        vfile, ifile = rd.a(20), rd.a(20)
        seed = rd.int()
        vmin = rd.float()
        rd.skip(6)
        radial = tuple(rd.floats(2))
        lat = tuple(rd.floats(2))
        lon = tuple(rd.floats(2))
        earth = rd.float()
        rd.skip(3)
        pfile = rd.a(20)
        pn = tuple(rd.ints(3))
        pr = tuple(rd.ints(2))
        cush = rd.float()
        rd.skip(4)
        layers = [cls._parse_layer(rd) for _ in range(n_layers)]
        rd.skip(5)
        nlat, nlon = rd.int(), rd.int()
        ifaces = [cls._parse_interface(rd) for _ in range(n_layers + 1)]
        return cls(
            n_vtypes,
            pinch,
            vfile,
            ifile,
            seed,
            vmin,
            radial,
            lat,
            lon,
            earth,
            pfile,
            pn,
            pr,
            cush,
            layers,
            (nlat, nlon),
            ifaces,
        )  # type: ignore[arg-type]

    @staticmethod
    def _parse_perturbation(rd: RecordReader, spike_dim: int) -> Perturbation:
        rd.skip(3)
        p = Perturbation()
        p.random = rd.int()
        p.random_sd = rd.float()
        p.add_prior = rd.int()
        p.prior_uncertainty = rd.float()
        rd.skip(3)
        p.checkerboard = rd.int()
        p.checker_amplitude = rd.float()
        p.checker_size = rd.int()
        p.checker_use_spacing = rd.int()
        rd.skip(3)
        p.spikes = rd.int()
        n = rd.int()
        for _ in range(n):
            amp = rd.float()
            loc = tuple(rd.floats(spike_dim))
            p.spike_list.append(Spike(amp, loc))
        return p

    @classmethod
    def _parse_layer(cls, rd: RecordReader) -> VelocityLayerSpec:
        rd.skip(3)
        nr = _numeric_prefix(rd.record_tokens(1), int)
        nlat = _numeric_prefix(rd.record_tokens(1), int)
        nlon = _numeric_prefix(rd.record_tokens(1), int)
        grad = rd.int()
        phase = rd.a(1)
        mfile = rd.a(20)
        dim = rd.int()
        vtop = _numeric_prefix(rd.record_tokens(1), float)
        vbot = _numeric_prefix(rd.record_tokens(1), float)
        pert = cls._parse_perturbation(rd, 3)
        return VelocityLayerSpec(nr, nlat, nlon, grad, phase, mfile, dim, vtop, vbot, pert)

    @classmethod
    def _parse_interface(cls, rd: RecordReader) -> InterfaceSpec:
        rd.skip(3)
        ue = rd.int()
        fname = rd.a(20)
        nw, ne, sw = rd.float(), rd.float(), rd.float()
        pert = cls._parse_perturbation(rd, 2)
        return InterfaceSpec(ue, fname, nw, ne, sw, pert)

    # -- formatting ------------------------------------------------------------------------
    def format(self) -> str:
        if not self.layers:
            raise FmtomoFormatError("Grid3dg needs at least one velocity layer")
        if len(self.interfaces) != self.n_layers + 1:
            raise FmtomoFormatError(
                f"Grid3dg: {self.n_layers} layers need {self.n_layers + 1} interfaces, "
                f"got {len(self.interfaces)}"
            )
        f = cfloat
        two = lambda a, b, fm: f"{fm(a):<8}{fm(b)}"  # noqa: E731
        out = [
            _C61,
            "c Specify number of layers (= number of interfaces -1)",
            _C61,
            _kv(str(self.n_layers), "Number of layers in model"),
            _kv(str(self.n_vtypes), "Number of velocity grid types (1 or 2)"),
            _kv(f(self.pinchout_km), "Pinchout distance (km) (>=0.0)"),
            _kv(_file_name(self.vgrids_file, "vgrids_file"), "Output velocity grid file"),
            _kv(_file_name(self.interfaces_file, "interfaces_file"), "Output interface grid file"),
            _kv(str(self.seed), "Seed for random number generation (int)"),
            _kv(f(self.min_velocity), "Minimum permitted velocity (km/s)"),
            _C61,
            "c Set 3-D grid size and location. Note that all layer",
            "c velocity grids have the same spatial dimension, but can",
            "c have different node densities. Interface grids have the",
            "c same node distribution.",
            _C61,
            _kv(
                two(*self.radial_range_km, lambda v: _dec(v, 2)),
                "Radial range (top-bottom) of grid (km)",
            ),
            _kv(
                two(*self.lat_range_deg, lambda v: _dec(v, 1)),
                "Latitudinal range (N-S) of grid (degrees)",
            ),
            _kv(
                two(*self.lon_range_deg, lambda v: _dec(v, 1)),
                "Longitudinal range (E-W) of grid (degrees)",
            ),
            _kv(f(self.earth_radius_km), "Earth radius"),
            _C62,
            "c Set up propagation grid file",
            _C62,
            _kv(_file_name(self.propgrid_file, "propgrid_file"), "Name of propagation grid file"),
            _kv(
                f"{self.prop_nodes[0]}   {self.prop_nodes[1]}    {self.prop_nodes[2]}",
                "Number of points in rad lat, long",
            ),
            _kv(
                "     ".join(str(v) for v in self.prop_refinement),
                "Refine factor & no. of local cells",
            ),
            _kv(f(self.prop_cushion), "Cushion factor for prop grid (<<1)"),
            _C62,
            _C62,
            "c First, set up the velocity grids",
            _C62,
        ]
        for k, lay in enumerate(self.layers, start=1):
            out.extend(self._format_layer(k, lay))
        out.extend(
            [
                _C62,
                _C62,
                "c Now, set up the interface grids",
                _C62,
                _C62,
                _kv(str(self.interface_nodes[0]), "Number of grid points in theta (N-S)"),
                _kv(str(self.interface_nodes[1]), "Number of grid points in phi (E-W)"),
            ]
        )
        for k, itf in enumerate(self.interfaces, start=1):
            out.extend(self._format_interface(k, itf))
        return "\n".join(out) + "\n"

    @staticmethod
    def _format_perturbation(
        p: Perturbation, what: str, size_label: str, spike_dim: int
    ) -> list[str]:
        f = cfloat
        if p.spikes and len(p.spike_list) == 0:
            raise FmtomoFormatError(f"{what}: spikes enabled but spike_list is empty")
        out = [
            _C62,
            f"c Optionally apply random structure to {what}",
            _C62,
            _kv(str(p.random), "Add random structure (0=no,1=yes)"),
            _kv(f(p.random_sd), "Standard deviation of Gaussian noise"),
            _kv(str(p.add_prior), "Add a priori model covariance (0=no,1=yes)?"),
            _kv(f(p.prior_uncertainty), "Diagonal elements of covariance matrix"),
            _C62,
            f"c Optionally apply checkerboard to {what}",
            _C62,
            _kv(str(p.checkerboard), "Add checkerboard (0=no,1=yes)"),
            _kv(f(p.checker_amplitude), "Maximum perturbation of vertices"),
            _kv(str(p.checker_size), f"Checkerboard size ({size_label})"),
            _kv(str(p.checker_use_spacing), "Use spacing (0=no,1=yes)"),
            _C62,
            f"c Optionally, apply spikes to {what}",
            _C62,
            _kv(str(p.spikes), "Apply spikes (0=no,1=yes)"),
            _kv(str(len(p.spike_list)), "Number of spikes"),
        ]
        for i, sp in enumerate(p.spike_list, start=1):
            if len(sp.location) != spike_dim:
                raise FmtomoFormatError(f"{what}: spike {i} needs {spike_dim} location values")
            out.append(_kv(_dec(sp.amplitude, 2), f"Amplitude of spike {i}"))
            label = "(depth, lat, long)" if spike_dim == 3 else "(lat, long)"
            out.append(
                _kv("  ".join(_dec(v, 1) for v in sp.location), f"Coordinates of spike {i} {label}")
            )
        return out

    def _format_layer(self, k: int, lay: VelocityLayerSpec) -> list[str]:
        f = cfloat
        suffix = " (type 1 & 2)" if k == 1 else ""
        out = [
            _C62,
            f"c Set velocity grid values for layer {k}",
            _C62,
            _kv(_pair(lay.n_radial, str), f"Number of radial grid points{suffix}"),
            _kv(_pair(lay.n_lat, str), "Number of grid points in theta (N-S)"),
            _kv(_pair(lay.n_lon, str), "Number of grid points in phi (E-W)"),
            _kv(str(lay.constant_gradient), "Use model (0) or constant gradient (1)"),
            _kv(lay.phase[:1] or "P", "Use P or S velocity model"),
            _kv(_file_name(lay.model_file, "model_file"), "Velocity model (option 0)"),
            _kv(str(lay.model_dimension), "Dimension of velocity model (1=1-D,3=3-D)"),
            _kv(_pair(lay.v_top, f), "Velocity at origin (km/s) (option 1)"),
            _kv(_pair(lay.v_bottom, f), "Velocity at maximum depth (km/s) (option 1)"),
        ]
        out.extend(self._format_perturbation(lay.perturbation, f"layer {k}", "NxNxN", 3))
        return out

    def _format_interface(self, k: int, itf: InterfaceSpec) -> list[str]:
        f = cfloat
        out = [
            _C62,
            f"c Set up interface grid for interface {k}",
            _C62,
            _kv(str(itf.from_file), "Obtain grid from external file (0=no,1=yes)"),
            _kv(_file_name(itf.file, "interface file"), "External interface grid file (option 1)"),
            _kv(f(itf.height_nw_km), "Height of NW grid point (option 0)"),
            _kv(f(itf.height_ne_km), "Height of NE grid point (option 0)"),
            _kv(f(itf.height_sw_km), "Height of SW grid point (option 0)"),
        ]
        out.extend(self._format_perturbation(itf.perturbation, f"interface {k}", "NxN", 2))
        return out

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.format())

    # -- expected output geometry ------------------------------------------------------------
    def expected_geometry(self) -> dict:
        """Node geometry written by ``grid3dg`` (``grid3dg.f90:152-162,200-210,474-489``).

        Returns ``{"propgrid": PropGrid, "layers": [per type: dict(nr, nlat, nlon, dr_km, dlat_rad,
        dlon_rad, r0_km, lat0_rad, lon0_rad)], "interfaces": dict(nlat, nlon, dlat_rad, dlon_rad,
        lat0_rad, lon0_rad)}``; node counts include the two cushion nodes per axis.
        """
        gor, glr = self.radial_range_km
        got, glt = self.lat_range_deg
        gop, glp = self.lon_range_deg
        npr, npt, npp = self.prop_nodes
        cush = self.prop_cushion
        gotp, gopp = glt + cush, gop + cush
        pg = PropGrid(
            npr,
            npt,
            npp,
            (gor - glr) / (npr - 1),
            (got - cush - gotp) / (npt - 1),
            (glp - cush - gopp) / (npp - 1),
            gor,
            gotp,
            gopp,
            self.prop_refinement[0],
            self.prop_refinement[1],
            self.earth_radius_km,
        )
        got_r, gop_r, glt_r, glp_r = (v * PI_GRID3DG / 180.0 for v in (got, gop, glt, glp))
        layers = []
        for lay in self.layers:
            per_type = []
            for t in range(self.n_vtypes):
                nvr, nvt, nvp = (
                    lay.n_radial[min(t, len(lay.n_radial) - 1)],
                    lay.n_lat[min(t, len(lay.n_lat) - 1)],
                    lay.n_lon[min(t, len(lay.n_lon) - 1)],
                )
                gsr = (gor - glr) / (nvr - 1)
                gst = (got_r - glt_r) / (nvt - 1)
                gsp = (glp_r - gop_r) / (nvp - 1)
                per_type.append(
                    {
                        "nr": nvr + 2,
                        "nlat": nvt + 2,
                        "nlon": nvp + 2,
                        "dr_km": gsr,
                        "dlat_rad": gst,
                        "dlon_rad": gsp,
                        "r0_km": gor - nvr * gsr + self.earth_radius_km,
                        "lat0_rad": got_r - gst * nvt,
                        "lon0_rad": gop_r - gsp,
                    }
                )
            layers.append(per_type)
        nvt, nvp = self.interface_nodes
        gst = (got_r - glt_r) / (nvt - 1)
        gsp = (glp_r - gop_r) / (nvp - 1)
        ifc = {
            "nlat": nvt + 2,
            "nlon": nvp + 2,
            "dlat_rad": gst,
            "dlon_rad": gsp,
            "lat0_rad": got_r - gst * nvt,
            "lon0_rad": gop_r - gsp,
        }
        return {"propgrid": pg, "layers": layers, "interfaces": ifc}
