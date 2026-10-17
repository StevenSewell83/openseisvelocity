"""``frechet.dat``: sparse Frechet derivatives written by ``fm3d``.

Format (``3dfmlib.f90:4875-4915``, ``README_0.7``)::

    receiver source ray subray n_deriv                 (5i6; n_deriv = 0 for invalid rays)
    n_deriv records "index value"                     (i10,e17.8)

(for the sub-rays of reflection-fit paths the source writes ``(i10,f17.8)``,
``3dfmlib.f90:4885`` - fixed point with only 8 decimals, an inconsistency of the code that this
reader tolerates; the writer always uses ``e17.8``).  Rows appear in the order of
``arrivals.dat``.  ``index`` is the 1-based global parameter number (``frechet.f90:19-113``):
blocks in the order of ``frechet.in`` - velocity grids (radius index fastest, then latitude,
then longitude; ``frechet.f90:287``, over **all** nodes of the grid including cushion nodes),
interfaces (latitude fastest then longitude, ``frechet.f90:527``), then 4 values per inverted
source (three position derivatives along r, lat, lon in s/km, then ``-1`` for the origin
time).  Units: s/(km/s) for velocities, s/km for interfaces and positions.

The reader returns a :class:`scipy.sparse.csr_matrix` with one row per ray record and
0-based columns (``index - 1``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ._fortran import FmtomoFormatError, RecordReader, read_text, write_text

__all__ = [
    "Frechet",
    "ParameterBlock",
    "format_frechet",
    "parameter_layout",
    "parse_frechet",
    "read_frechet",
    "write_frechet",
]


@dataclass
class Frechet:
    """``rays``: DataFrame (``receiver``, ``source``, ``ray``, ``subray``, ``n_deriv``).

    ``matrix`` is the CSR matrix of derivatives (one row per ray record)."""

    rays: pd.DataFrame
    matrix: sparse.csr_matrix


def parse_frechet(
    text: str, *, source: str | Path | None = None, n_params: int | None = None
) -> Frechet:
    """Parse the text of ``frechet.dat``.

    ``n_params`` fixes the number of matrix columns (otherwise the largest index found).
    """
    rd = RecordReader(text, source)
    heads: list[tuple[int, int, int, int, int]] = []
    indptr = [0]
    cols: list[int] = []
    vals: list[float] = []
    while not rd.at_eof():
        h = rd.ints(5)
        n = h[4]
        if n < 0:
            raise rd.error(f"ray {len(heads) + 1}: negative derivative count")
        if rd.remaining() < n:
            raise rd.error(f"ray {len(heads) + 1} declares {n} derivatives but the file ends early")
        for _ in range(n):
            parts = rd.next_record().split()
            while not parts:
                parts = rd.next_record().split()
            try:
                cols.append(int(parts[0]) - 1)
                vals.append(float(parts[1].replace("d", "e").replace("D", "e")))
            except (ValueError, IndexError):
                raise rd.error(f"bad derivative record {parts!r}") from None
        heads.append((h[0], h[1], h[2], h[3], n))
        indptr.append(len(cols))
    if cols and min(cols) < 0:
        raise FmtomoFormatError(f"{rd.source}: parameter index below 1")
    ncol = max(cols) + 1 if cols else 0
    if n_params is not None:
        if ncol > n_params:
            raise FmtomoFormatError(
                f"{rd.source}: parameter index {ncol} exceeds the expected number of parameters "
                f"({n_params})"
            )
        ncol = n_params
    mat = sparse.csr_matrix(
        (np.array(vals), np.array(cols, dtype=np.int64), np.array(indptr, dtype=np.int64)),
        shape=(len(heads), ncol),
    )
    rays = pd.DataFrame(
        heads, columns=["receiver", "source", "ray", "subray", "n_deriv"], dtype="int64"
    )
    return Frechet(rays, mat)


def read_frechet(path: str | Path, **kwargs: object) -> Frechet:
    """Read ``frechet.dat``."""
    return parse_frechet(read_text(path), source=path, **kwargs)  # type: ignore[arg-type]


def _e17_8(x: float) -> str:
    """Fortran ``e17.8``: ``-0.22723977E-01``."""
    if x == 0.0:
        return f"{'0.00000000E+00':>17}"
    s = f"{abs(x):.7E}"
    mant, exp = s.split("E")
    digits = mant.replace(".", "")
    e10 = int(exp) + 1
    return f"{('-' if x < 0 else '') + '0.' + digits + 'E' + format(e10, '+03d'):>17}"


def format_frechet(fr: Frechet) -> str:
    """Format as ``frechet.dat`` (explicit zeros are dropped; ``n_deriv`` is recomputed)."""
    mat = fr.matrix.tocsr()
    mat.eliminate_zeros()
    mat.sort_indices()
    if mat.shape[0] != len(fr.rays):
        raise FmtomoFormatError("matrix rows and ray table length differ")
    out: list[str] = []
    for i, r in enumerate(fr.rays.itertuples(index=False)):
        lo, hi = mat.indptr[i], mat.indptr[i + 1]
        out.append(f"{r.receiver:6d}{r.source:6d}{r.ray:6d}{r.subray:6d}{hi - lo:6d}")
        out.extend(
            f"{int(c) + 1:10d}{_e17_8(float(v))}"
            for c, v in zip(mat.indices[lo:hi], mat.data[lo:hi], strict=True)
        )
    return "\n".join(out) + ("\n" if out else "")


def write_frechet(path: str | Path, fr: Frechet) -> Path:
    """Write ``frechet.dat``."""
    return write_text(path, format_frechet(fr))


@dataclass(frozen=True)
class ParameterBlock:
    """A contiguous block of the global parameter vector (0-based ``start``)."""

    kind: str  # "velocity", "interface" or "source"
    ident: int  # layer, interface or source number (1-based)
    vtype: int | None
    start: int
    size: int
    shape: tuple[
        int, ...
    ]  # velocity: (nlon, nlat, nr) C-order == radius fastest; interface: (nlon, nlat)

    def node_index(self, columns: np.ndarray) -> tuple[np.ndarray, ...]:
        """Local node indices for matrix ``columns`` of this block.

        Velocity: ``(ir, ilat, ilon)`` with ``ir`` counting radius **ascending** from the grid
        origin (file order).  Interface: ``(ilat, ilon)``.  Source: ``(component,)`` with 0-2
        the position derivatives (r, lat, lon) and 3 the origin time.
        """
        loc = np.asarray(columns) - self.start
        if loc.size and (loc.min() < 0 or loc.max() >= self.size):
            raise ValueError("column outside the block")
        if self.kind == "source":
            return (loc,)
        dims = np.unravel_index(loc, self.shape)
        return tuple(dims[::-1])


def parameter_layout(
    *,
    velocity: list[tuple[int, int, tuple[int, int, int]]] = (),  # type: ignore[assignment]
    interfaces: list[tuple[int, tuple[int, int]]] = (),  # type: ignore[assignment]
    sources: list[int] = (),  # type: ignore[assignment]
) -> list[ParameterBlock]:
    """Blocks of the global parameter vector in ``frechet.in`` order.

    ``velocity``: ``(layer, vtype, (nr, nlat, nlon))`` per inverted velocity grid;
    ``interfaces``: ``(interface, (nlat, nlon))``; ``sources``: source ids.
    """
    blocks: list[ParameterBlock] = []
    pos = 0
    for layer, vtype, (nr, nlat, nlon) in velocity:
        size = nr * nlat * nlon
        blocks.append(ParameterBlock("velocity", layer, vtype, pos, size, (nlon, nlat, nr)))
        pos += size
    for iface, (nlat, nlon) in interfaces:
        size = nlat * nlon
        blocks.append(ParameterBlock("interface", iface, None, pos, size, (nlon, nlat)))
        pos += size
    for sid in sources:
        blocks.append(ParameterBlock("source", sid, None, pos, 4, (4,)))
        pos += 4
    return blocks
