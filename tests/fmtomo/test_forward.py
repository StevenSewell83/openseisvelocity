"""Parallel fm3d by source partitioning (``osv.fmtomo.forward``)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.fmtomo.forward import (
    Fm3dError,
    _make_groups,
    partition_sources,
    run_fm3d_parallel,
)
from osv.io.fmtomo import Receivers, read_receivers, read_sources
from osv.io.fmtomo.frechet import read_frechet
from osv.io.fmtomo.rays import read_rays
from osv.tools import ToolNotFoundError, which_tool

REPO = Path(__file__).resolve().parents[2]
EX1 = REPO / "tests" / "data" / "fmtomo" / "example1"
SHARED = ("vgrids.in", "interfaces.in", "propgrid.in", "mode_set.in", "frechet.in")


def test_partition_balances_and_covers() -> None:
    w = [10.0, 1, 1, 1, 9, 2, 8, 3]
    groups = partition_sources(w, 3)
    assert sorted(i for g in groups for i in g) == list(range(8))
    loads = [sum(w[i] for i in g) for g in groups]
    assert max(loads) - min(loads) <= 3
    assert len(partition_sources(w, 50)) == 8
    assert partition_sources([], 3) == []


def test_group_ray_index_remap(tmp_path: Path) -> None:
    sources = read_sources(EX1 / "sources.in")
    rec = read_receivers(EX1 / "receivers.in")
    groups = _make_groups(sources, rec, [[0, 2, 4], [1, 3]], tmp_path)
    g = groups[0]
    assert g.sources == [1, 3, 5]
    first = g.rays[g.rays["receiver"] == 1]
    # global list of receiver 1 is sources 1..5 -> group has rays 1, 3, 5 as local 1, 2, 3
    assert first["ray"].tolist() == [1, 3, 5]
    assert first["local_ray"].tolist() == [1, 2, 3]
    assert first["local_source"].tolist() == [1, 2, 3]
    # output order is source-major
    assert g.rays["local_source"].is_monotonic_increasing


def test_rejects_reflection_fit_mode(tmp_path: Path) -> None:
    for n in SHARED:
        shutil.copy(EX1 / n, tmp_path / n)
    (tmp_path / "mode_set.in").write_text(
        (EX1 / "mode_set.in").read_text().replace("T   no_pp_mode", "F   no_pp_mode")
    )
    with pytest.raises(Fm3dError, match="no_pp_mode"):
        run_fm3d_parallel(
            tmp_path,
            read_sources(EX1 / "sources.in"),
            read_receivers(EX1 / "receivers.in"),
            2,
            workdir=tmp_path / "w",
        )


def _have_fm3d() -> bool:
    try:
        which_tool("fm3d")
    except ToolNotFoundError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(not _have_fm3d(), reason="fm3d binary not built")
@pytest.mark.parametrize(("n_jobs", "src_derivs"), [(2, False), (3, False), (3, True)])
def test_parallel_equals_single_run(tmp_path: Path, n_jobs: int, src_derivs: bool) -> None:
    sources = read_sources(EX1 / "sources.in")
    rec = read_receivers(EX1 / "receivers.in")
    # make the receiver ray lists non-trivial: reverse the order of the sources at odd receivers
    rays = rec.rays.copy()
    odd = (rays["receiver"] % 2 == 1).to_numpy()
    parts = [g.iloc[::-1] if odd[g.index[0]] else g for _, g in rays.groupby("receiver", sort=True)]
    rays = pd.concat(parts, ignore_index=True)
    rec = Receivers(rec.positions, rays)

    base = tmp_path / "base"
    base.mkdir()
    for n in SHARED:
        shutil.copy(EX1 / n, base / n)
    from osv.io.fmtomo import write_receivers, write_sources
    from osv.tools import run_tool

    if src_derivs:  # also relocate sources 4 and 2 (parameter blocks follow this order)
        (base / "frechet.in").write_text("1\n0\n0\n2\n4 2\n")
    write_sources(base / "sources.in", sources)
    write_receivers(base / "receivers.in", rec)
    run_tool("fm3d", base)

    res = run_fm3d_parallel(base, sources, rec, n_jobs, workdir=tmp_path / "par")
    assert res.n_jobs == n_jobs
    assert (base / "arrivals.dat").read_text() == res.arrivals_path.read_text()

    a, b = read_frechet(base / "frechet.dat"), read_frechet(res.frechet_path)
    assert a.rays.equals(b.rays)
    assert (a.matrix != b.matrix).nnz == 0
    assert (base / "frechet.dat").read_text() == res.frechet_path.read_text()

    ra, rb = read_rays(base / "rays.dat"), read_rays(res.rays_path)
    assert ra.rays.equals(rb.rays)
    np.testing.assert_array_equal(ra.points["r_km"], rb.points["r_km"])
    assert (base / "rays.dat").read_text() == res.rays_path.read_text()
