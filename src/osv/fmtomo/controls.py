"""Map the ``fmtomo`` config section onto the FMTOMO control-file dataclasses (Stage 4).

:func:`build_controls` turns ``cfg.fmtomo`` plus a small description of the model
(:class:`ModelInfo`) into every control file a run needs.  The files are the dataclasses of
:mod:`osv.io.fmtomo` (``ModeSet``, ``Frechgen``, ``Invert3d`` ...); :mod:`osv.fmtomo.rundir`
writes them.

What FMTOMO can and cannot do (verified in ``fortran/fmtomo/source/invert3d.f90``,
``frechgen.f90`` and ``fortran/fmtomo_extras/source/gridloc.f90``):

* **Parameter classes** inverted by ``invert3d``: velocity (per velocity grid ``layer`` and
  type; ``frechgen`` writes the subset), interface depths (per interface), and per source the
  position (depth, lat, lon) plus origin time.  The classes are switched on by the three
  0/1 flags of ``invert3d.in``, which ``frechgen`` reads too.
* **Regularisation**: damping is ``epsilon * eps_class / sigma^2`` on the diagonal (``epsilon`` =
  ``fmtomo.damping``; ``eps_class`` the per-class factor of ``invert3d.in``, here 1 for velocity and
  interfaces and ``source_damping_*`` for sources; ``sigma`` the a priori uncertainty in
  ``vgridsref.in``/``sourcesref.in``).  Smoothing is a **second-derivative (Laplacian-like)
  operator** on velocity and interface nodes only, weighted ``eta * eta_class`` (``eta`` =
  ``fmtomo.smoothing``); it is switched on globally by the ``second_derivative_smoothing`` flag
  and cannot be applied to sources.  ``smoothing = 0`` therefore writes flag 0.  There is no
  first-derivative/norm-damping variant.
* **Station terms: not supported** (there is no receiver parameter class; ``scorrect`` is a
  *source* correction).  ``fmtomo.station_terms = "preapply"`` is implemented outside FMTOMO:
  :func:`osv.fmtomo.rundir.assemble_run_dir` subtracts supplied per-station terms from the
  observed times before they are written.
* **Relocation**: ``linear`` = hypocentre parameters in the subspace inversion (events are
  FMTOMO *sources*, stations are *receivers*, driver ``tomo3d``); ``nonlinear`` = Extras
  ``gridloc`` grid search over time fields computed from the stations (stations are FMTOMO
  *sources*, events are *receivers*, driver ``tomo3di``).  ``gridloc`` always searches the
  whole propagation grid (no window around the event), refined by sub-cell dicing.
* **P and S**: ``tomo3d`` can only invert one velocity type per run (``frechgen.in`` note),
  so P and S are *separate runs* (each with its own velocity set as type 1).  ``["P", "S"]``
  with nonlinear relocation maps to the Extras ``tomo3dj`` (``gridlocj``, coupled relocation;
  experimental); ``["vpvs"]`` to ``tomo3dps`` (``mkvpvs``/``frvpvs``/``vpvsinv``; experimental).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from osv.io.fmtomo import (
    Frechgen,
    GridLoc,
    GridLocJ,
    Invert3d,
    InvIter,
    JoinTimes,
    MkVpVs,
    ModeSet,
    PathSpec,
    Residuals,
    Scorrect,
    TomoRun,
    VpVsInv,
    VpVsResids,
)
from osv.io.fmtomo.extras import FrVpVs

log = logging.getLogger(__name__)

__all__ = [
    "DIR_P",
    "DIR_S",
    "FmtomoControls",
    "JointControls",
    "ModelInfo",
    "RunControls",
    "VpVsControls",
    "build_controls",
    "phase_path",
]

DIR_P = "invert_p"
DIR_S = "invert_s"
_PHASES = ("P", "S")


@dataclass(frozen=True)
class ModelInfo:
    """What the control files need to know about the FMTOMO model of one run.

    ``n_layers``: number of velocity grids (layers; interfaces = layers + 1).  ``n_vtypes``:
    velocity types in ``vgrids.in``.  ``invert_layers``: 1-based layers (1 = top) whose velocity
    is inverted (default all).  ``invert_source_ids``: 1-based ids of the sources whose location
    is inverted by linear relocation (default all).
    """

    n_layers: int = 1
    n_vtypes: int = 1
    invert_layers: tuple[int, ...] | None = None
    invert_source_ids: tuple[int, ...] | None = None
    n_sources: int | None = None
    n_receivers: int | None = None

    def __post_init__(self) -> None:
        if self.n_layers < 1 or self.n_vtypes not in (1, 2):
            raise ValueError("ModelInfo: need n_layers >= 1 and n_vtypes of 1 or 2")
        if self.invert_layers is not None:
            bad = [i for i in self.invert_layers if not 1 <= i <= self.n_layers]
            if bad or len(set(self.invert_layers)) != len(self.invert_layers):
                raise ValueError(
                    f"ModelInfo.invert_layers {self.invert_layers}: need distinct layers in "
                    f"1..{self.n_layers}"
                )

    @property
    def n_interfaces(self) -> int:
        return self.n_layers + 1

    @property
    def layers(self) -> tuple[int, ...]:
        return (
            tuple(range(1, self.n_layers + 1))
            if self.invert_layers is None
            else tuple(self.invert_layers)
        )

    @classmethod
    def from_model(
        cls,
        vgrids: Any,
        *,
        invert_layers: tuple[int, ...] | None = None,
        sources: list[Any] | None = None,
        receivers: Any = None,
        invert_source_ids: tuple[int, ...] | None = None,
    ) -> ModelInfo:
        """Build from an :class:`osv.io.fmtomo.VelocityGrids` (and optionally sources/receivers)."""
        return cls(
            n_layers=vgrids.n_layers,
            n_vtypes=vgrids.n_types,
            invert_layers=invert_layers,
            invert_source_ids=invert_source_ids,
            n_sources=None if sources is None else len(sources),
            n_receivers=None if receivers is None else len(receivers.positions),
        )


@dataclass
class RunControls:
    """Control files of one ``fm3d``/``tomo3d`` run directory."""

    driver: str
    tomo_run_name: str
    mode_set: ModeSet
    tomo_run: TomoRun
    frechgen: Frechgen
    invert3d: Invert3d
    residuals: Residuals
    scorrect: Scorrect | None
    inviter: InvIter
    gridloc: GridLoc | None = None
    #: sources whose location is inverted (None: all); written to ``sourcederivs.in``
    source_ids: tuple[int, ...] | None = None
    #: True if the sources written to ``sources.in`` need a priori uncertainties
    needs_source_uncertainty: bool = False

    @property
    def nonlinear(self) -> bool:
        return self.gridloc is not None

    def texts(self) -> dict[str, str]:
        """File name -> text of every control file of this run."""
        out = {
            "mode_set.in": self.mode_set.format(),
            self.tomo_run_name: self.tomo_run.format(),
            "frechgen.in": self.frechgen.format(),
            "invert3d.in": self.invert3d.format(),
            "residuals.in": self.residuals.format(),
            "inviter.in": self.inviter.format(),
        }
        if self.scorrect is not None:
            out["scorrect.in"] = self.scorrect.format()
        if self.gridloc is not None:
            out["gridloc.in"] = self.gridloc.format()
        return out


@dataclass
class JointControls:
    """Coupled P/S relocation of the Extras ``tomo3dj`` (experimental)."""

    tomo_run: TomoRun
    gridlocj: GridLocJ
    jointimes: JoinTimes

    def texts(self) -> dict[str, str]:
        return {
            "tomo3dj.in": self.tomo_run.format(),
            "gridlocj.in": self.gridlocj.format(),
            "jointimes.in": self.jointimes.format(),
        }


@dataclass
class VpVsControls:
    """Joint Vp/Vs inversion of the Extras ``tomo3dps`` (experimental)."""

    tomo_run: TomoRun
    mkvpvs: MkVpVs
    frvpvs: FrVpVs
    vpvsinv: VpVsInv
    vpvsresids: VpVsResids

    def texts(self) -> dict[str, str]:
        return {
            "tomo3dps.in": self.tomo_run.format(),
            "mkvpvs.in": self.mkvpvs.format(),
            "frvpvs.in": self.frvpvs.format(),
            "vpvsinv.in": self.vpvsinv.format(),
            "vpvsresids.in": self.vpvsresids.format(),
        }


@dataclass
class FmtomoControls:
    """All control files for the configured inversion.

    ``runs`` maps a phase (``"P"``/``"S"``) to the controls of its run directory.  For
    ``phases == ["vpvs"]`` the P and S runs are the forward runs the Vp/Vs inversion builds on.
    """

    driver: str
    package: str
    phases: tuple[str, ...]
    relocate: str
    runs: dict[str, RunControls]
    joint: JointControls | None = None
    vpvs: VpVsControls | None = None
    station_terms: str = "none"
    experimental: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def run(self) -> RunControls:
        """The single run of a one-phase inversion."""
        if len(self.runs) != 1:
            raise ValueError(f"controls hold {sorted(self.runs)} runs; pick one via .runs[phase]")
        return next(iter(self.runs.values()))

    @property
    def stations_are_sources(self) -> bool:
        """True for the nonlinear (gridloc) geometry: stations are FMTOMO sources."""
        return self.relocate == "nonlinear"


def phase_path(phase: str, *, layout: str = "separate") -> PathSpec:
    """The direct-wave :class:`PathSpec` of a phase.

    ``layout="separate"`` (one velocity set per run directory, the supported mode): P and S are
    both velocity type 1.  ``layout="joint"`` (one ``vgrids.in`` holding P as type 1 and S as
    type 2, as ``grid3dg`` writes for ``n_vtypes = 2``): P is 1, S is 2.
    """
    if phase not in _PHASES:
        raise ValueError(f"phase must be 'P' or 'S', got {phase!r}")
    if layout not in ("separate", "joint"):
        raise ValueError(f"layout must be 'separate' or 'joint', got {layout!r}")
    return PathSpec.direct(2 if (phase == "S" and layout == "joint") else 1)


# ---------------------------------------------------------------------------------------


def _run_controls(fm: Any, info: ModelInfo, relocate: str, *, invert_velocity: bool) -> RunControls:
    nonlinear = relocate == "nonlinear"
    linear = relocate == "linear"
    layers = info.layers
    pvi = int(invert_velocity and bool(layers))
    pii = int(bool(fm.invert_interfaces))
    psi = int(linear)
    if pvi + pii + psi == 0:
        raise ValueError(
            "nothing to invert (invert3d would stop): fmtomo.invert_velocity is false or no "
            "layers are selected, no interfaces are inverted and relocate is not 'linear'; "
            "enable one of them"
        )
    bad_if = [i for i in fm.invert_interfaces if i > info.n_interfaces]
    if bad_if:
        raise ValueError(
            f"fmtomo.invert_interfaces {bad_if} exceed the {info.n_interfaces} interfaces"
        )
    smoothing = float(fm.smoothing)
    minsd = pvi + pii + 2 * psi
    inv = Invert3d(
        min_interface_distance_km=fm.min_interface_distance_km,
        min_velocity=fm.min_velocity_km_s,
        velocity_inversion=(pvi, 1.0, 1.0),
        interface_inversion=(pii, 1.0, 1.0),
        source_inversion=(psi, float(fm.source_damping_position), float(fm.source_damping_time)),
        subspace_dimension=max(int(fm.subspace_dimension), minsd),
        global_damping=float(fm.damping),
        second_derivative_smoothing=int(smoothing > 0),
        global_smoothing=smoothing if smoothing > 0 else 1.0,
    )
    # frechgen: ignores its own selection for a class whose invert3d flag is 0
    if pvi and layers == tuple(range(1, info.n_layers + 1)):
        vder, gids, vts = -1, (1,), (1,)
    elif pvi:
        vder, gids, vts = len(layers), tuple(layers), (1,) * len(layers)
    else:
        vder, gids, vts = 0, (1,), (1,)
    ifs = tuple(fm.invert_interfaces)
    src_ids = info.invert_source_ids if psi else None
    sder = 0 if not psi else (-1 if src_ids is None else len(src_ids))
    frech = Frechgen(
        velocity_derivs=vder,
        velocity_type=1,
        velocity_grid_ids=gids,
        velocity_types=vts,
        interface_derivs=len(ifs),
        interface_ids=ifs or (2,),
        source_derivs=sder,
    )
    if nonlinear:
        driver, tomo_name = "tomo3di", "tomo3di.in"
        g = fm.gridloc
        gridloc: GridLoc | None = GridLoc(
            min_stations=g.min_stations,
            max_origin_time_perturbation=g.max_origin_time_shift_s,
            refined_search=int(g.refine),
            subcell_dicing=g.subcell_dicing,
        )
    else:
        driver, tomo_name, gridloc = "tomo3d", "tomo3d.in", None
    return RunControls(
        driver=driver,
        tomo_run_name=tomo_name,
        mode_set=ModeSet(
            no_pp_mode=True, save_rays_mode=bool(fm.save_rays), save_timefields_mode=nonlinear
        ),
        tomo_run=TomoRun(n_iterations=int(fm.iterations)),
        frechgen=frech,
        invert3d=inv,
        residuals=Residuals(),
        scorrect=None if nonlinear else Scorrect(),
        inviter=InvIter(1),
        gridloc=gridloc,
        source_ids=src_ids,
        needs_source_uncertainty=bool(psi),
    )


def build_controls(
    cfg: Any, model_info: ModelInfo | None = None, phases: list[str] | tuple[str, ...] | None = None
) -> FmtomoControls:
    """Build every control file of a local-earthquake inversion from the config.

    Parameters
    ----------
    cfg
        :class:`osv.config.schema.OSVConfig` or its ``fmtomo`` section.
    model_info
        Layers/types/sources of the model (default: one layer, one type, all sources).
    phases
        Phases to invert (default ``cfg.fmtomo.phases``): ``["P"]``, ``["S"]``, ``["P", "S"]``
        or ``["vpvs"]``.

    Returns
    -------
    FmtomoControls
        See the module docstring for the config -> control field mapping.
    """
    fm = getattr(cfg, "fmtomo", cfg)
    info = model_info or ModelInfo()
    ph = tuple(phases if phases is not None else fm.phases)
    if not ph or len(set(ph)) != len(ph) or any(p not in (*_PHASES, "vpvs") for p in ph):
        raise ValueError(f"phases {list(ph)}: use P, S, P+S or vpvs")
    if "vpvs" in ph and len(ph) > 1:
        raise ValueError("phases: 'vpvs' cannot be combined with P or S")
    relocate = fm.relocate
    if fm.package == "base" and (relocate == "nonlinear" or "vpvs" in ph):
        raise ValueError(
            "fmtomo.package 'base' has no nonlinear relocation or Vp/Vs inversion; use "
            "package: extras or relocate: linear/none with phases P/S"
        )
    notes: list[str] = []
    if fm.station_terms != "none":
        notes.append(
            "FMTOMO has no station terms; 'preapply' subtracts station terms from the observed "
            "times in assemble_run_dir"
        )
    experimental = False
    joint: JointControls | None = None
    vpvs: VpVsControls | None = None
    runs: dict[str, RunControls]
    if ph == ("vpvs",):
        experimental = True
        notes.append(
            "vpvs: experimental. Forward P and S runs without relocation feed mkvpvs/frvpvs/"
            "vpvsinv (tomo3dps); only the control files are produced, not the run layout"
        )
        runs = {p: _run_controls(fm, info, "none", invert_velocity=True) for p in _PHASES}
        driver = "tomo3dps"
        j = fm.joint
        vpvs = VpVsControls(
            tomo_run=TomoRun(n_iterations=int(fm.iterations)),
            mkvpvs=MkVpVs(max_depth_km=j.vpvs_max_depth_km),
            frvpvs=FrVpVs(),
            vpvsinv=VpVsInv(
                damping=float(fm.damping),
                subspace_dimension=int(fm.subspace_dimension),
                second_derivative_smoothing=int(fm.smoothing > 0),
                smoothing=float(fm.smoothing) if fm.smoothing > 0 else 0.5,
                sparse_fraction=j.sparse_fraction,
            ),
            vpvsresids=VpVsResids(),
        )
    else:
        runs = {
            p: _run_controls(fm, info, relocate, invert_velocity=fm.invert_velocity) for p in ph
        }
        first = next(iter(runs.values()))
        driver = first.driver
        if len(ph) == 2 and relocate == "nonlinear":
            experimental = True
            driver = "tomo3dj"
            notes.append(
                "P+S with nonlinear relocation uses the Extras tomo3dj/gridlocj (experimental): "
                f"run directories {DIR_P}/ and {DIR_S}/ plus the coupling files in the root"
            )
            g, j = fm.gridloc, fm.joint
            joint = JointControls(
                tomo_run=TomoRun(n_iterations=int(fm.iterations)),
                gridlocj=GridLocJ(
                    arrtimes_p_file=f"{DIR_P}/arrtimes.dat",
                    arrtimes_s_file=f"{DIR_S}/arrtimes.dat",
                    sources_p_file=f"{DIR_P}/sources.in",
                    sources_s_file=f"{DIR_S}/sources.in",
                    min_combinations=g.min_stations,
                    max_origin_time_perturbation=g.max_origin_time_shift_s,
                    limits_file=f"{DIR_P}/interfacesref.in",
                    otimes_p_file=f"{DIR_P}/otimes.dat",
                    otimes_s_file=f"{DIR_S}/otimes.dat",
                    receivers_p_file=f"{DIR_P}/receivers.in",
                    receivers_s_file=f"{DIR_S}/receivers.in",
                    refined_search=int(g.refine),
                    subcell_dicing=g.subcell_dicing,
                    s_weight=j.s_weight,
                    sp_weight=j.sp_weight,
                    relocation_based_on=j.relocation_based_on,
                ),
                jointimes=JoinTimes(),
            )
        elif len(ph) == 2:
            notes.append("P and S are independent runs (one velocity type per FMTOMO run)")
    log.debug("fmtomo controls: driver=%s phases=%s relocate=%s", driver, ph, relocate)
    return FmtomoControls(
        driver=driver,
        package=fm.package,
        phases=ph,
        relocate=relocate,
        runs=runs,
        joint=joint,
        vpvs=vpvs,
        station_terms=fm.station_terms,
        experimental=experimental,
        notes=notes,
    )
