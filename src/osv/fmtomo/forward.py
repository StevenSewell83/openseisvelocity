"""Parallel ``fm3d`` by source partitioning.

``fm3d`` treats every source independently (time field, ray tracing and Frechet derivatives are
computed source by source, ``3dfm_main.f90:1040-1185``) so a run with many sources can be split
into groups that run as separate processes.  :func:`run_fm3d_parallel` does that and merges the
outputs so that they are *identical* to those of one run over all sources:

* each group directory symlinks the shared model files (``vgrids.in``, ``interfaces.in``,
  ``propgrid.in``, ``mode_set.in`` and ``frechet.in``) and holds its own ``sources.in`` and
  ``receivers.in``.  Group sources are renumbered ``1..g`` (in global order); receivers without
  any ray in the group are dropped and the others renumbered; the ``source`` column of the
  receiver association lines is remapped.  Path ids (position in the source's own path list) are
  unchanged;
* the third column of ``arrivals.dat``/``frechet.dat``/``rays.dat`` is the position of the ray in
  the *receiver's own* association list (``3dfm_main.f90:1095-1113``).  Filtering a receiver's
  list to the group's sources changes that position, so ray indices are mapped back to the
  global list;
* the output rows of ``fm3d`` appear source by source (sources.in order), then by receiver, then
  by ray.  The merged files are written in that same global order, as text with the original
  number formatting, so downstream programs (``frechgen``, ``residuals``, ``invert3d``) read them
  unchanged;
* parameter numbers in ``frechet.dat`` are grid based and global, except for the 4 parameters per
  relocated source (``frechet.in`` source list), whose columns are remapped.

The number of parallel processes is capped so that the estimated memory (per process from the
propagation grid size) fits into the available RAM.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from osv.io.fmtomo import (
    FmtomoFormatError,
    Frechet,
    FrechetSpec,
    ModeSet,
    RayPaths,
    Receivers,
    Source,
    read_arrivals,
    read_frechet,
    read_interfaces,
    read_propgrid,
    read_rays,
    read_vgrids,
    write_receivers,
    write_sources,
)
from osv.tools import run_tool

log = logging.getLogger(__name__)

__all__ = [
    "Fm3dError",
    "Fm3dResult",
    "available_memory_gb",
    "estimate_job_memory_gb",
    "partition_sources",
    "run_fm3d_parallel",
]

#: files shared (symlinked) by all group directories
SHARED_FILES = ("vgrids.in", "interfaces.in", "propgrid.in", "mode_set.in")
#: bytes of RSS per propagation node (``fm3d`` run with ``file_mode`` off, see benchmark) and the
#: fixed overhead (vgrids, interfaces, executable) in GB
BYTES_PER_PROP_NODE = 200.0
BASE_JOB_GB = 0.3
#: fraction of the available memory that jobs may use
MEM_SAFETY = 0.85
#: fixed cost of one source (time field) in units of one traced ray, for load balancing
SOURCE_COST_RAYS = 5.0


class Fm3dError(RuntimeError):
    """A parallel ``fm3d`` run or the merge of its outputs failed."""


@dataclass
class Fm3dResult:
    """Merged outputs of a (parallel) ``fm3d`` run.

    The files are in ``out_dir`` in the standard formats; ``arrivals`` is parsed on demand.
    """

    out_dir: Path
    arrivals_path: Path
    frechet_path: Path | None
    rays_path: Path | None
    n_jobs: int
    groups: list[list[int]]  # global (1-based) source ids per group
    wall_s: float
    job_wall_s: list[float] = field(default_factory=list)
    job_mem_gb: float = 0.0

    def arrivals(self) -> pd.DataFrame:
        return read_arrivals(self.arrivals_path)

    def frechet(self, **kwargs: object) -> Frechet:
        if self.frechet_path is None:
            raise Fm3dError("this run produced no frechet.dat")
        return read_frechet(self.frechet_path, **kwargs)

    def rays(self) -> RayPaths:
        if self.rays_path is None:
            raise Fm3dError("this run produced no rays.dat")
        return read_rays(self.rays_path)


# ---------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------
def available_memory_gb() -> float:
    """Available RAM in GB (``psutil`` if installed, else ``MemAvailable`` of ``/proc/meminfo``)."""
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.virtual_memory().available) / 1e9
    except ImportError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return float("inf")


def estimate_job_memory_gb(propgrid_file: str | Path) -> float:
    """Rough RSS (GB) of one ``fm3d`` process from the size of ``propgrid.in``."""
    pg = read_propgrid(propgrid_file)
    return BASE_JOB_GB + pg.nr * pg.nlat * pg.nlon * BYTES_PER_PROP_NODE / 1e9


# ---------------------------------------------------------------------------------------
# partitioning
# ---------------------------------------------------------------------------------------
def partition_sources(weights: Sequence[float], k: int) -> list[list[int]]:
    """Split items ``0..n-1`` into at most ``k`` groups of balanced total weight.

    Greedy longest-processing-time assignment.  Each group is sorted ascending; empty groups are
    dropped.
    """
    k = max(1, min(int(k), len(weights)))
    loads = [0.0] * k
    groups: list[list[int]] = [[] for _ in range(k)]
    for i in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
        g = int(np.argmin(loads))
        groups[g].append(i)
        loads[g] += weights[i]
    return [sorted(g) for g in groups if g]


@dataclass
class _Group:
    index: int
    sources: list[int]  # global 1-based ids, ascending (local id = position + 1)
    rays: pd.DataFrame  # in fm3d output order; global_/local_ receiver, source, ray (+ path)
    receivers: list[int]  # global ids of the kept receivers (local id = position + 1)
    dir: Path


def _ray_table(receivers: Receivers) -> pd.DataFrame:
    rays = receivers.rays.copy()
    rays["ray"] = rays.groupby("receiver", sort=False).cumcount() + 1
    return rays


def _make_groups(
    sources: list[Source], receivers: Receivers, groups: list[list[int]], workdir: Path
) -> list[_Group]:
    allrays = _ray_table(receivers)
    out: list[_Group] = []
    for gi, idx in enumerate(groups):
        gsrc = [i + 1 for i in idx]
        lsrc = {g: n for n, g in enumerate(gsrc, start=1)}
        t = allrays[allrays["source"].isin(lsrc)].copy()
        # local ray index: position in the receiver's list filtered to this group
        t["local_ray"] = t.groupby("receiver", sort=False).cumcount() + 1
        recs = sorted(t["receiver"].unique().tolist())
        lrec = {g: n for n, g in enumerate(recs, start=1)}
        t["local_receiver"] = t["receiver"].map(lrec)
        t["local_source"] = t["source"].map(lsrc)
        t = t.sort_values(["local_source", "local_receiver", "local_ray"], kind="stable")
        out.append(_Group(gi, gsrc, t.reset_index(drop=True), recs, workdir / f"group_{gi:03d}"))
    return out


def _write_group_inputs(
    grp: _Group,
    inputs: Path,
    sources: list[Source],
    receivers: Receivers,
    frechet_spec: FrechetSpec | None,
) -> None:
    d = grp.dir
    d.mkdir(parents=True, exist_ok=True)
    for name in SHARED_FILES:
        link = d / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((inputs / name).resolve())
    fr = d / "frechet.in"
    if fr.is_symlink() or fr.exists():
        fr.unlink()
    if (inputs / "frechet.in").exists():
        if frechet_spec is not None and frechet_spec.compute and frechet_spec.sources:
            lsrc = {g: n for n, g in enumerate(grp.sources, start=1)}
            mine = FrechetSpec(
                True,
                frechet_spec.velocity,
                frechet_spec.interfaces,
                [lsrc[s] for s in frechet_spec.sources if s in lsrc],
            )
            mine.write(fr)
        else:
            fr.symlink_to((inputs / "frechet.in").resolve())
    write_sources(d / "sources.in", [sources[g - 1] for g in grp.sources])
    pos = receivers.positions.loc[grp.receivers].copy()
    pos.index = pd.RangeIndex(1, len(pos) + 1, name="receiver")
    rays = pd.DataFrame(
        {
            "receiver": grp.rays["local_receiver"].to_numpy(),
            "source": grp.rays["local_source"].to_numpy(),
            "path": grp.rays["path"].to_numpy(),
        }
    )
    # association lines keep the receiver's own order (= local_ray order within a receiver)
    rays = rays.assign(_k=grp.rays["local_ray"].to_numpy()).sort_values(
        ["receiver", "_k"], kind="stable"
    )[["receiver", "source", "path"]]
    write_receivers(d / "receivers.in", Receivers(pos, rays.reset_index(drop=True)))


# ---------------------------------------------------------------------------------------
# text level readers of fm3d output (rows keep their original number formatting)
# ---------------------------------------------------------------------------------------
_HEAD4 = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")


def _head4(line: str, what: str) -> tuple[int, int, int, int]:
    m = _HEAD4.match(line)
    if not m:
        raise Fm3dError(f"{what}: cannot parse record {line!r}")
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def _check_rows(
    heads: list[tuple[int, int, int, int]], grp: _Group, what: str, with_src: bool = False
) -> None:
    exp = grp.rays
    if len(heads) != len(exp):
        raise Fm3dError(
            f"group {grp.index}: {what} has {len(heads)} ray records but {len(exp)} were requested"
        )
    got_rec = np.array([h[0] for h in heads])
    got_ray = np.array([h[2] for h in heads])
    if not (
        np.array_equal(got_rec, exp["local_receiver"].to_numpy())
        and np.array_equal(got_ray, exp["local_ray"].to_numpy())
    ):
        raise Fm3dError(f"group {grp.index}: {what} ray records are not in the expected order")


def _fmt_head(r: int, s: int, m: int, k: int) -> str:
    return f"{r:6d}{s:6d}{m:6d}{k:6d}"


def _global_ids(grp: _Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = grp.rays
    return e["receiver"].to_numpy(), e["source"].to_numpy(), e["ray"].to_numpy()


def _merge_arrivals(groups: list[_Group], out: Path) -> None:
    recs: list[tuple[int, int, int, str]] = []  # (src, rec, ray) keys + text
    for grp in groups:
        lines = (grp.dir / "arrivals.dat").read_text().splitlines()
        lines = [ln for ln in lines if ln.strip()]
        heads = [_head4(ln, "arrivals.dat") for ln in lines]
        _check_rows(heads, grp, "arrivals.dat")
        gr, gs, gm = _global_ids(grp)
        for i, ln in enumerate(lines):
            tail = ln[24:]
            recs.append(
                (
                    int(gs[i]),
                    int(gr[i]),
                    int(gm[i]),
                    _fmt_head(gr[i], gs[i], gm[i], heads[i][3]) + tail,
                )
            )
    recs.sort(key=lambda t: t[:3])
    out.write_text("".join(t[3] + "\n" for t in recs))


def _non_source_params(inputs: Path, spec: FrechetSpec) -> int:
    n = 0
    if spec.velocity:
        vg = read_vgrids(inputs / "vgrids.in")
        for layer, vtype in spec.velocity:
            n += int(np.prod(list(vg.get(layer, vtype)["velocity"].shape)))
    if spec.interfaces:
        it = read_interfaces(inputs / "interfaces.in").data
        n += len(spec.interfaces) * it.sizes["lat"] * it.sizes["lon"]
    return n


def _merge_frechet(groups: list[_Group], out: Path, inputs: Path, spec: FrechetSpec | None) -> None:
    src_map: dict[int, dict[int, int]] = {}  # group -> {local source offset: global offset}
    p0 = 0
    if spec is not None and spec.sources:
        p0 = _non_source_params(inputs, spec)
        gpos = {s: n for n, s in enumerate(spec.sources)}
        for grp in groups:
            mine = [s for s in spec.sources if s in set(grp.sources)]
            src_map[grp.index] = {p0 + 4 * n: p0 + 4 * gpos[s] for n, s in enumerate(mine)}
    recs: list[tuple[int, int, int, list[str]]] = []
    for grp in groups:
        lines = (grp.dir / "frechet.dat").read_text().splitlines()
        gr, gs, gm = _global_ids(grp)
        smap = src_map.get(grp.index)
        pos = 0
        i = 0
        while pos < len(lines):
            if not lines[pos].strip():
                pos += 1
                continue
            h = _head4(lines[pos], "frechet.dat")
            n = int(lines[pos].split()[4])
            body = lines[pos + 1 : pos + 1 + n]
            if len(body) < n:
                raise Fm3dError(f"group {grp.index}: frechet.dat ends early")
            if smap:
                body = [_remap_param(b, p0, smap) for b in body]
            if i >= len(grp.rays):
                raise Fm3dError(f"group {grp.index}: more frechet records than rays")
            _check_one(h, grp, i, "frechet.dat")
            head = _fmt_head(gr[i], gs[i], gm[i], h[3]) + f"{n:6d}"
            recs.append((int(gs[i]), int(gr[i]), int(gm[i]), [head, *body]))
            pos += 1 + n
            i += 1
        if i != len(grp.rays):
            raise Fm3dError(
                f"group {grp.index}: frechet.dat has {i} ray records, expected {len(grp.rays)}"
            )
    recs.sort(key=lambda t: t[:3])
    with out.open("w") as f:
        for t in recs:
            f.write("\n".join(t[3]) + "\n")


def _check_one(h: tuple[int, int, int, int], grp: _Group, i: int, what: str) -> None:
    e = grp.rays
    if h[0] != e["local_receiver"].iat[i] or h[2] != e["local_ray"].iat[i]:
        raise Fm3dError(f"group {grp.index}: {what} ray record {i + 1} is out of order")


def _remap_param(line: str, p0: int, smap: dict[int, int]) -> str:
    idx = int(line[:10])
    if idx <= p0:
        return line
    base = p0 + ((idx - p0 - 1) // 4) * 4
    comp = idx - base  # 1..4
    return f"{smap[base] + comp:10d}{line[10:]}"


def _merge_rays(groups: list[_Group], out: Path) -> None:
    recs: list[tuple[int, int, int, list[str]]] = []
    for grp in groups:
        lines = (grp.dir / "rays.dat").read_text().splitlines()
        gr, gs, gm = _global_ids(grp)
        pos = 0
        i = 0
        while pos < len(lines):
            if not lines[pos].strip():
                pos += 1
                continue
            h = _head4(lines[pos], "rays.dat")
            nsec = int(lines[pos].split()[4])
            end = pos + 1
            for _ in range(nsec):
                npts = int(lines[end].split()[0])
                end += 1 + npts
            if i >= len(grp.rays):
                raise Fm3dError(f"group {grp.index}: more ray records than rays")
            _check_one(h, grp, i, "rays.dat")
            head = _fmt_head(gr[i], gs[i], gm[i], h[3]) + f"{nsec:6d}"
            recs.append((int(gs[i]), int(gr[i]), int(gm[i]), [head, *lines[pos + 1 : end]]))
            pos = end
            i += 1
        if i != len(grp.rays):
            raise Fm3dError(
                f"group {grp.index}: rays.dat has {i} ray records, expected {len(grp.rays)}"
            )
    recs.sort(key=lambda t: t[:3])
    with out.open("w") as f:
        for t in recs:
            f.write("\n".join(t[3]) + "\n")


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def run_fm3d_parallel(
    run_dir_inputs: str | Path,
    sources: list[Source],
    receivers: Receivers,
    n_jobs: int,
    mem_per_job_gb: float | None = None,
    workdir: str | Path | None = None,
    *,
    timeout: float | None = None,
    keep_group_dirs: bool = False,
) -> Fm3dResult:
    """Run ``fm3d`` on source subsets in parallel and merge the outputs.

    Parameters
    ----------
    run_dir_inputs
        Directory with the shared model files ``vgrids.in``, ``interfaces.in``, ``propgrid.in``,
        ``mode_set.in`` and optionally ``frechet.in``.  Its own ``sources.in``/``receivers.in``
        are ignored (``sources`` and ``receivers`` are used).
    sources, receivers
        The complete ``sources.in``/``receivers.in`` contents (global numbering).
    n_jobs
        Maximum number of parallel ``fm3d`` processes (also bounded by the number of sources and
        the memory guard).
    mem_per_job_gb
        RSS of one ``fm3d`` process; default: estimated from ``propgrid.in``.
    workdir
        Where group directories and the merged ``arrivals.dat`` (``frechet.dat``, ``rays.dat``)
        are written (default ``<run_dir_inputs>/parallel``).
    keep_group_dirs
        Keep the per-group directories (with logs) after a successful merge; they are always kept
        on failure.

    Returns
    -------
    Fm3dResult
    """
    t_start = time.perf_counter()
    inputs = Path(run_dir_inputs)
    work = Path(workdir) if workdir is not None else inputs / "parallel"
    for name in SHARED_FILES:
        if not (inputs / name).exists():
            raise Fm3dError(f"{inputs / name} is missing (needed by fm3d)")
    ms = ModeSet.read(inputs / "mode_set.in")
    if not ms.no_pp_mode:
        raise Fm3dError("parallel fm3d needs no_pp_mode = T (sources must be independent)")
    if ms.parallel_mode:
        raise Fm3dError("mode_set.in must have parallel_mode = F")
    try:
        from osv.io.fmtomo import validate_receivers

        validate_receivers(receivers, sources)
    except FmtomoFormatError as exc:
        raise Fm3dError(str(exc)) from exc
    if not sources:
        raise Fm3dError("no sources")

    spec = FrechetSpec.read(inputs / "frechet.in") if (inputs / "frechet.in").exists() else None
    if spec is not None and not spec.compute:
        spec = None

    # ---- concurrency and memory guard
    mem_job = (
        mem_per_job_gb
        if mem_per_job_gb is not None
        else estimate_job_memory_gb(inputs / "propgrid.in")
    )
    avail = available_memory_gb()
    cap = max(1, int(MEM_SAFETY * avail // mem_job)) if np.isfinite(avail) else n_jobs
    k = max(1, min(int(n_jobs), len(sources), cap))
    if k < min(int(n_jobs), len(sources)):
        log.warning(
            "fm3d parallelism limited to %d jobs by memory (%.1f GB per job, %.1f GB available)",
            k,
            mem_job,
            avail,
        )

    nrays = _ray_table(receivers).groupby("source").size()
    weights = [float(nrays.get(i, 0)) + SOURCE_COST_RAYS for i in range(1, len(sources) + 1)]
    part = partition_sources(weights, k)
    work.mkdir(parents=True, exist_ok=True)
    groups = _make_groups(sources, receivers, part, work)
    for g in groups:
        _write_group_inputs(g, inputs, sources, receivers, spec)

    env = {"OMP_NUM_THREADS": "1"}
    walls = [0.0] * len(groups)

    def job(g: _Group) -> None:
        t0 = time.perf_counter()
        for stale in ("arrivals.dat", "frechet.dat", "rays.dat"):
            (g.dir / stale).unlink(missing_ok=True)
        run_tool("fm3d", g.dir, timeout=timeout, env=env)
        walls[g.index] = time.perf_counter() - t0

    if len(groups) == 1:
        job(groups[0])
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            for fut in [ex.submit(job, g) for g in groups]:
                fut.result()

    # ---- merge
    arr = work / "arrivals.dat"
    _merge_arrivals(groups, arr)
    fre = None
    if all((g.dir / "frechet.dat").exists() for g in groups) and spec is not None:
        fre = work / "frechet.dat"
        _merge_frechet(groups, fre, inputs, spec)
    rays = None
    if ms.save_rays_mode and all((g.dir / "rays.dat").exists() for g in groups):
        rays = work / "rays.dat"
        _merge_rays(groups, rays)
    if not keep_group_dirs:
        for g in groups:
            shutil.rmtree(g.dir, ignore_errors=True)
    return Fm3dResult(
        out_dir=work,
        arrivals_path=arr,
        frechet_path=fre,
        rays_path=rays,
        n_jobs=len(groups),
        groups=[g.sources for g in groups],
        wall_s=time.perf_counter() - t_start,
        job_wall_s=walls,
        job_mem_gb=mem_job,
    )
