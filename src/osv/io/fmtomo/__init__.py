"""Readers and writers for the FMTOMO file formats (``fm3d``, ``tomo3d``, ``invert3d``, Extras).

See the module docstrings for the verified formats:

* :mod:`~osv.io.fmtomo.grids` - ``vgrids.in``, ``interfaces.in``, ``propgrid.in``, ``arrtimes.dat``
  (xarray datasets with ``lat``, ``lon`` in degrees and ``depth`` in km, positive down);
* :mod:`~osv.io.fmtomo.sources` - ``sources.in`` and ``receivers.in``;
* :mod:`~osv.io.fmtomo.times` - ``otimes.dat``, ``arrivals.dat`` (= ``mtimes.dat``/``rtimes.dat``),
  ``stimes.dat``, ``sourcederivs.in``, ``gridsave.in``, ``residuals.dat``;
* :mod:`~osv.io.fmtomo.rays` and :mod:`~osv.io.fmtomo.frechet` - ``rays.dat`` and ``frechet.dat``;
* :mod:`~osv.io.fmtomo.controls`, :mod:`~osv.io.fmtomo.datagen`, :mod:`~osv.io.fmtomo.grid3dg`,
  :mod:`~osv.io.fmtomo.gmtslice` - control files;
* :mod:`~osv.io.fmtomo.extras` - FMTOMO Extras (gridloc, vpvs) files.
"""

from __future__ import annotations

from ._fortran import FmtomoFormatError
from .controls import (
    FrechetSpec,
    Frechgen,
    Invert3d,
    InvIter,
    ModeSet,
    Residuals,
    Scorrect,
    TomoRun,
)
from .datagen import Arraygen, Moddata, Obsdata, SourceGroup, Synthdata
from .extras import (
    GridLoc,
    GridLocJ,
    JoinTimes,
    MkVpVs,
    VpVsInv,
    VpVsResids,
    format_src_dat,
    parse_src_dat,
    read_src_dat,
    write_src_dat,
)
from .frechet import Frechet, ParameterBlock, parameter_layout, read_frechet, write_frechet
from .gmtslice import Gmtslice
from .grid3dg import Grid3dg
from .grids import (
    InterfaceGrids,
    PropGrid,
    VelocityGrids,
    cushion_mask,
    interior,
    read_arrtimes,
    read_interfaces,
    read_propgrid,
    read_vgrids,
    validate_model,
    write_arrtimes,
    write_interfaces,
    write_propgrid,
    write_vgrids,
)
from .rays import RayPaths, ray_hit_count, read_rays, write_rays
from .sources import (
    PathSpec,
    Receivers,
    Source,
    read_receivers,
    read_sources,
    sources_frame,
    validate_receivers,
    validate_sources,
    write_receivers,
    write_sources,
)
from .times import (
    GridSave,
    read_arrivals,
    read_gridsave,
    read_observed,
    read_residuals,
    read_source_ids,
    read_stimes,
    write_arrivals,
    write_gridsave,
    write_observed,
    write_residuals,
    write_source_ids,
    write_stimes,
)

__all__ = [
    "Arraygen",
    "FmtomoFormatError",
    "Frechet",
    "FrechetSpec",
    "Frechgen",
    "Gmtslice",
    "Grid3dg",
    "GridLoc",
    "GridLocJ",
    "GridSave",
    "InterfaceGrids",
    "InvIter",
    "Invert3d",
    "JoinTimes",
    "MkVpVs",
    "Moddata",
    "ModeSet",
    "Obsdata",
    "ParameterBlock",
    "PathSpec",
    "PropGrid",
    "RayPaths",
    "Receivers",
    "Residuals",
    "Scorrect",
    "Source",
    "SourceGroup",
    "Synthdata",
    "TomoRun",
    "VelocityGrids",
    "VpVsInv",
    "VpVsResids",
    "cushion_mask",
    "format_src_dat",
    "interior",
    "parameter_layout",
    "parse_src_dat",
    "ray_hit_count",
    "read_arrivals",
    "read_arrtimes",
    "read_frechet",
    "read_gridsave",
    "read_interfaces",
    "read_observed",
    "read_propgrid",
    "read_rays",
    "read_receivers",
    "read_residuals",
    "read_source_ids",
    "read_sources",
    "read_src_dat",
    "read_stimes",
    "read_vgrids",
    "sources_frame",
    "validate_model",
    "validate_receivers",
    "validate_sources",
    "write_arrivals",
    "write_arrtimes",
    "write_frechet",
    "write_gridsave",
    "write_interfaces",
    "write_observed",
    "write_propgrid",
    "write_rays",
    "write_receivers",
    "write_residuals",
    "write_source_ids",
    "write_sources",
    "write_src_dat",
    "write_stimes",
    "write_vgrids",
]
