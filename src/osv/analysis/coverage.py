"""Ray coverage of a tomographic model: derivative weight sum, hit counts and ray length.

Measures (all on the **interior** nodes of a velocity grid, i.e. without the cushion nodes, as
``xarray.DataArray`` with dims ``(depth, lat, lon)``, depth ascending, km positive down):

* Derivative weight sum (DWS; Toomey & Foulger 1989, as in SIMUL/tomoDD): for node ``j``
  ``DWS_j = sum_i |dt_i/dv_j|`` over all ray records ``i`` of ``frechet.dat``.  The FMTOMO
  derivative is the sensitivity of the travel time to the velocity at the node including the
  B-spline basis weight, so it is proportional to the path length of the ray in the node's
  support, weighted by the basis function and divided by ``v**2`` (unit: s per km/s).  With
  ``per_volume=True`` the sum is divided by the node volume ``dr * r dlat * r cos(lat) dlon``
  (km**3) to remove the dependence on node spacing (unit: s per km/s per km**3).
* Hit count from Frechet: number of ray records with a non-zero derivative at the node.
* Hit count from ``rays.dat``: number of distinct rays passing through the node's cell.  Cells
  are node-centred boxes (edges halfway between nodes, outer edges half a spacing outside).
* Ray length per cell from ``rays.dat``: path length in km, summing great-circle chords of ray
  segments that are subdivided so that each piece is shorter than a fraction of the smallest
  cell side and assigned to the cell of its midpoint.

``coverage_mask`` turns any of them into a boolean "well sampled" mask (True where covered);
the plotting threshold ``plotting.dws_mask_threshold`` is an absolute DWS threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from osv.io.fmtomo.controls import FrechetSpec
from osv.io.fmtomo.frechet import Frechet, ParameterBlock, parameter_layout
from osv.io.fmtomo.grids import VelocityGrids, interior
from osv.io.fmtomo.rays import RayPaths

__all__ = [
    "coverage_mask",
    "derivative_weight_sum",
    "dws_by_block",
    "hit_count_from_frechet",
    "ray_hit_count",
    "ray_length_per_cell",
    "velocity_layout",
]

log = logging.getLogger(__name__)

Grid = xr.Dataset | VelocityGrids


def velocity_layout(
    spec: FrechetSpec,
    grids: VelocityGrids,
    *,
    interface_shape: tuple[int, int] | None = None,
) -> list[ParameterBlock]:
    """Parameter layout of ``frechet.dat`` for the requests in ``frechet.in``.

    ``interface_shape`` ``(nlat, nlon)`` is needed only when interfaces are inverted.
    """
    vel = []
    for layer, vtype in spec.velocity:
        ds = grids.get(layer, vtype)
        vel.append((layer, vtype, (ds.sizes["depth"], ds.sizes["lat"], ds.sizes["lon"])))
    if spec.interfaces and interface_shape is None:
        raise ValueError("frechet.in inverts interfaces: pass interface_shape=(nlat, nlon)")
    ifaces = [(i, interface_shape) for i in spec.interfaces]
    return parameter_layout(velocity=vel, interfaces=ifaces, sources=spec.sources)  # type: ignore[arg-type]


def _velocity_blocks(layout: Iterable[ParameterBlock]) -> list[ParameterBlock]:
    return [b for b in layout if b.kind == "velocity"]


def _pick_block(layout: Iterable[ParameterBlock], layer: int | None, vtype: int | None):
    blocks = _velocity_blocks(layout)
    if not blocks:
        raise ValueError("the parameter layout has no velocity block")
    sel = [
        b
        for b in blocks
        if (layer is None or b.ident == layer) and (vtype is None or b.vtype == vtype)
    ]
    if not sel:
        have = [(b.ident, b.vtype) for b in blocks]
        raise ValueError(f"no velocity block for layer={layer}, vtype={vtype}; layout has {have}")
    if len(sel) > 1:
        have = [(b.ident, b.vtype) for b in sel]
        raise ValueError(f"layer/vtype are ambiguous, pass both explicitly (candidates {have})")
    return sel[0]


def _template(grid: Grid, layer: int | None, vtype: int | None) -> xr.Dataset:
    """Full (cushion included) dataset of the grid of ``layer``/``vtype``."""
    if isinstance(grid, xr.Dataset):
        return grid
    return grid.get(layer or 1, vtype or 1)


def _column_stat(frechet: Frechet, block: ParameterBlock, kind: str) -> np.ndarray:
    """Per-column sum of |value| (``dws``) or of the non-zero count (``hits``) of one block."""
    mat = frechet.matrix.tocsc()
    stop = block.start + block.size
    out = np.zeros(block.size)
    hi = min(stop, mat.shape[1])
    if hi > block.start:
        sub = mat[:, block.start : hi]
        if kind == "dws":
            out[: hi - block.start] = np.asarray(abs(sub).sum(axis=0)).ravel()
        else:
            sub = sub.copy()
            sub.eliminate_zeros()
            out[: hi - block.start] = np.diff(sub.indptr)
    return out


def _to_grid(vec: np.ndarray, block: ParameterBlock, ds: xr.Dataset, name: str) -> xr.DataArray:
    """Block vector (radius fastest) to a (depth, lat, lon) array of the full grid."""
    nlon, nlat, nr = block.shape
    want = (ds.sizes["depth"], ds.sizes["lat"], ds.sizes["lon"])
    if (nr, nlat, nlon) != want:
        raise ValueError(
            f"block of layer {block.ident} has shape (nr, nlat, nlon)={(nr, nlat, nlon)} but the "
            f"grid has {want}; pass the grid the Frechet derivatives were computed on"
        )
    cube = vec.reshape(nlon, nlat, nr).transpose(2, 1, 0)[::-1]  # depth ascending
    return xr.DataArray(
        cube,
        dims=("depth", "lat", "lon"),
        coords={d: ds[d] for d in ("depth", "lat", "lon")},
        name=name,
        attrs={
            "layer": block.ident,
            "vtype": block.vtype,
            "n_cushion": ds.attrs.get("n_cushion", 1),
        },
    )


def _crop(da: xr.DataArray, ds: xr.Dataset, interior_only: bool) -> xr.DataArray:
    if not interior_only:
        return da
    nc = int(ds.attrs.get("n_cushion", 1))
    return da.isel({d: slice(nc, da.sizes[d] - nc) for d in ("depth", "lat", "lon")})


def node_volume(ds: xr.Dataset) -> xr.DataArray:
    """Volume (km**3) of the box around each node: ``dr * r dlat * r cos(lat) dlon``."""
    r_e = float(ds.attrs.get("fmtomo_earth_radius_km", 6371.0))
    dr = float(ds.attrs["fmtomo_dr_km"])
    dlat, dlon = float(ds.attrs["fmtomo_dlat_rad"]), float(ds.attrs["fmtomo_dlon_rad"])
    r = r_e - ds["depth"]
    coslat = np.cos(np.deg2rad(ds["lat"]))
    vol = dr * r * dlat * r * coslat * dlon
    return (vol + xr.zeros_like(ds["lon"], dtype=float).reset_coords(drop=True)).transpose(
        "depth", "lat", "lon"
    )


def _stat_array(
    frechet: Frechet,
    layout: Iterable[ParameterBlock],
    grid: Grid,
    layer: int | None,
    vtype: int | None,
    kind: str,
    name: str,
    interior_only: bool,
) -> xr.DataArray:
    block = _pick_block(layout, layer, vtype)
    ds = _template(grid, block.ident, block.vtype)
    return _crop(_to_grid(_column_stat(frechet, block, kind), block, ds, name), ds, interior_only)


def derivative_weight_sum(
    frechet: Frechet,
    layout: Iterable[ParameterBlock],
    grid: Grid,
    *,
    layer: int | None = None,
    vtype: int | None = None,
    per_volume: bool = False,
    interior_only: bool = True,
) -> xr.DataArray:
    """Derivative weight sum ``sum_i |dt_i/dv_j|`` per velocity node of one block.

    Parameters
    ----------
    frechet, layout
        Parsed ``frechet.dat`` and its parameter layout (see :func:`velocity_layout`).
    grid
        The velocity grid (Dataset from ``read_vgrids``) or a ``VelocityGrids`` (then the grid
        of the block is used); defines the node coordinates and cushion.
    layer, vtype
        Block to use (FMTOMO numbering); may be omitted when the layout has one velocity block.
    per_volume
        Divide by the node volume in km**3 (see :func:`node_volume`).
    interior_only
        Drop the cushion nodes (default).

    Returns
    -------
    xarray.DataArray
        Dims ``(depth, lat, lon)``.  Nodes never touched by a ray are 0.
    """
    block = _pick_block(layout, layer, vtype)
    ds = _template(grid, block.ident, block.vtype)
    da = _to_grid(_column_stat(frechet, block, "dws"), block, ds, "dws")
    if per_volume:
        da = da / node_volume(ds)
        da.name = "dws"
        da.attrs["units"] = "s/(km/s)/km3"
    else:
        da.attrs["units"] = "s/(km/s)"
    return _crop(da, ds, interior_only)


def dws_by_block(
    frechet: Frechet,
    layout: Iterable[ParameterBlock],
    grids: VelocityGrids,
    **kwargs: object,
) -> dict[tuple[int, int], xr.DataArray]:
    """DWS of every velocity block, keyed by ``(layer, vtype)``."""
    layout = list(layout)
    return {
        (b.ident, b.vtype): derivative_weight_sum(  # type: ignore[arg-type]
            frechet, layout, grids, layer=b.ident, vtype=b.vtype, **kwargs
        )
        for b in _velocity_blocks(layout)
    }


def hit_count_from_frechet(
    frechet: Frechet,
    layout: Iterable[ParameterBlock],
    grid: Grid,
    *,
    layer: int | None = None,
    vtype: int | None = None,
    interior_only: bool = True,
) -> xr.DataArray:
    """Number of ray records with a non-zero derivative at each node of one block."""
    return _stat_array(frechet, layout, grid, layer, vtype, "hits", "hits", interior_only)


# ---------------------------------------------------------------------------------------
# rays.dat based measures
# ---------------------------------------------------------------------------------------


def _edges(centres: np.ndarray) -> np.ndarray:
    c = np.asarray(centres, dtype=float)
    if c.size < 2:
        raise ValueError("need at least two nodes per axis to define cells")
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _cell_grid(grid: Grid, layer: int | None, vtype: int | None):
    ds = interior(_template(grid, layer, vtype))
    edges = [_edges(ds[d].to_numpy()) for d in ("depth", "lat", "lon")]
    return ds, edges


def _sample_paths(
    rp: RayPaths, min_cell_km: float, oversample: float, region: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Midpoints (depth, lat, lon), lengths (km) and ray row of sub-pieces of all segments."""
    pts = rp.points
    if region is not None:
        sec = rp.sections[rp.sections["region"] == region]
        key = pd.MultiIndex.from_arrays([pts["ray"], pts["section"]])
        pts = pts[key.isin(pd.MultiIndex.from_arrays([sec["ray"], sec["section"]]))]
    if len(pts) < 2:
        z = np.empty(0)
        return z, z, z, z, z.astype(int)
    ray = pts["ray"].to_numpy()
    sec = pts["section"].to_numpy()
    r = pts["r_km"].to_numpy()
    la, lo = pts["lat_rad"].to_numpy(), pts["lon_rad"].to_numpy()
    same = (ray[1:] == ray[:-1]) & (sec[1:] == sec[:-1])
    i0 = np.flatnonzero(same)
    i1 = i0 + 1

    def xyz(i: np.ndarray) -> np.ndarray:
        c = r[i] * np.cos(la[i])
        return np.column_stack([c * np.cos(lo[i]), c * np.sin(lo[i]), r[i] * np.sin(la[i])])

    length = np.linalg.norm(xyz(i1) - xyz(i0), axis=1)
    n = np.maximum(1, np.ceil(length * oversample / min_cell_km)).astype(np.int64)
    seg = np.repeat(np.arange(len(i0)), n)
    start = np.cumsum(n) - n
    k = np.arange(len(seg)) - start[seg]
    frac = (k + 0.5) / n[seg]
    a, b = i0[seg], i1[seg]
    rr = r[a] + frac * (r[b] - r[a])
    lat = la[a] + frac * (la[b] - la[a])
    lon = lo[a] + frac * (lo[b] - lo[a])
    return (
        rp.earth_radius_km - rr,
        np.rad2deg(lat),
        np.rad2deg(lon),
        length[seg] / n[seg],
        ray[a],
    )


def _min_cell_km(ds: xr.Dataset) -> float:
    r_e = float(ds.attrs.get("fmtomo_earth_radius_km", 6371.0))
    rmax = r_e - float(ds["depth"].min())
    side_h = rmax * np.deg2rad(
        min(float(ds["lat"].diff("lat").min()), float(ds["lon"].diff("lon").min()))
    )
    side_h *= min(1.0, float(np.cos(np.deg2rad(np.abs(ds["lat"]).max()))))
    return float(min(float(ds["depth"].diff("depth").min()), side_h))


def _locate(
    vals: tuple[np.ndarray, np.ndarray, np.ndarray], edges: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    idx, ok = [], np.ones(len(vals[0]), dtype=bool)
    for v, e in zip(vals, edges, strict=True):
        i = np.searchsorted(e, v, side="right") - 1
        ok &= (i >= 0) & (i < len(e) - 1)
        idx.append(i)
    return np.stack(idx), ok


def _da(cube: np.ndarray, ds: xr.Dataset, name: str, units: str) -> xr.DataArray:
    return xr.DataArray(
        cube,
        dims=("depth", "lat", "lon"),
        coords={d: ds[d] for d in ("depth", "lat", "lon")},
        name=name,
        attrs={"units": units},
    )


def ray_length_per_cell(
    rp: RayPaths,
    grid: Grid,
    *,
    layer: int | None = None,
    vtype: int | None = None,
    region: int | None = None,
    oversample: float = 4.0,
) -> xr.DataArray:
    """Ray path length (km) in the node-centred cell of every interior node.

    ``region``: only ray sections inside this region (= velocity layer; FMTOMO numbers regions
    like layers, 1 = top), as needed to compare with the Frechet derivatives of one layer;
    ``None`` uses all sections.  ``oversample``: segments are cut into pieces shorter
    than ``min_cell_side / oversample``.
    Ray length outside the interior grid is discarded.
    """
    ds, edges = _cell_grid(grid, layer, vtype)
    d, la, lo, w, _ = _sample_paths(rp, _min_cell_km(ds), oversample, region)
    shape = tuple(len(e) - 1 for e in edges)
    idx, ok = _locate((d, la, lo), edges)
    cube = np.bincount(
        np.ravel_multi_index(tuple(idx[:, ok]), shape), weights=w[ok], minlength=int(np.prod(shape))
    ).reshape(shape)
    return _da(cube, ds, "ray_length", "km")


def ray_hit_count(
    rp: RayPaths,
    grid: Grid,
    *,
    layer: int | None = None,
    vtype: int | None = None,
    region: int | None = None,
    oversample: float = 4.0,
) -> xr.DataArray:
    """Number of distinct rays (rows of ``rp.rays``) passing through each interior node's cell.

    Differs from :func:`osv.io.fmtomo.rays.ray_hit_count` (ray *points* per box of arbitrary
    edges): paths are interpolated between points so no crossed cell is missed, and a ray
    counts once per cell however long it stays in it.
    """
    ds, edges = _cell_grid(grid, layer, vtype)
    d, la, lo, _, ray = _sample_paths(rp, _min_cell_km(ds), oversample, region)
    shape = tuple(len(e) - 1 for e in edges)
    idx, ok = _locate((d, la, lo), edges)
    flat = np.ravel_multi_index(tuple(idx[:, ok]), shape)
    pairs = np.unique(np.column_stack([ray[ok], flat]), axis=0)
    cube = np.bincount(pairs[:, 1], minlength=int(np.prod(shape))).reshape(shape)
    return _da(cube.astype(float), ds, "ray_hits", "rays")


# ---------------------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------------------


def coverage_mask(
    measure: xr.DataArray,
    threshold: float,
    *,
    mode: str = "absolute",
    dilate: int = 0,
) -> xr.DataArray:
    """Boolean mask, True where the model is covered (``measure >= threshold``).

    Parameters
    ----------
    measure
        DWS, hit count or ray length.
    threshold
        ``mode="absolute"``: value in the units of ``measure`` (``plotting.dws_mask_threshold``
        for the DWS); ``mode="percentile"``: percentage 0-100 of the positive finite values
        below which nodes count as uncovered.
    dilate
        Grow the covered region by this many nodes (face connectivity) to avoid ragged edges.
        The mask to hide uncovered model is ``~coverage_mask(...)``.
    """
    if dilate < 0:
        raise ValueError("dilate must be >= 0")
    vals = measure.to_numpy()
    if mode == "absolute":
        thr = float(threshold)
    elif mode == "percentile":
        if not 0 <= threshold <= 100:
            raise ValueError("percentile threshold must be between 0 and 100")
        pos = vals[np.isfinite(vals) & (vals > 0)]
        thr = float(np.percentile(pos, threshold)) if pos.size else np.inf
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'absolute' or 'percentile'")
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(vals) & (vals >= thr) & (vals > 0 if thr <= 0 else True)
    if dilate:
        mask = ndimage.binary_dilation(mask, iterations=dilate)
    out = measure.copy(data=mask)
    out.name = "covered"
    out.attrs = {"threshold": thr, "mode": mode, "dilate": dilate}
    return out
