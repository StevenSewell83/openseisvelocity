"""Stage framework: numbered stage directories, manifests, resume/force/from/to.

A *stage* is a subclass of `Stage` registered with `register`. Each stage runs
in ``<project.workdir>/<NN>_<name>/`` containing ``figures/``, ``logs/``,
``manifest.json`` and (reserved, rendered by the report task) ``report.html``.

Canonical order and numbering (`STAGE_ORDER`): prep 01, velest_mc 02,
velest_analyse 03, build_model 04, tradeoff 05, invert 06, analyse 07,
resolution 08, validation 09, report 10. Numbers are simply the position in
this list, so directories sort in execution order.

A stage is **up to date** (skipped) when its manifest says ``complete`` and

* the hash of its config sections (`Stage.config_sections`) is unchanged,
* the sha256 of every declared input (`Stage.inputs`) is unchanged,
* the run ids of its upstream stages are unchanged (re-running an upstream
  stage gives it a new run id, which makes all downstream stages stale),
* the hashes of the tools it uses (`Stage.tools`) are unchanged,
* all declared outputs still exist.

Implementing a stage::

    from osv.stages import Stage, StageContext, register

    @register
    class PrepStage(Stage):
        name = "prep"
        requires = ()
        config_sections = ("inputs", "coordinates", "data_selection")
        outputs = ("events.parquet", "stations.parquet")

        def inputs(self, ctx):
            return [*ctx.config.inputs.quakeml, *ctx.config.inputs.stationxml]

        def run(self, ctx):
            ctx.logger.info("working")
            events.to_parquet(ctx.path("events.parquet"))
            ctx.add_check({"name": "n_events", "status": "pass", "message": "..."})

        def plot(self, ctx):
            fig.savefig(ctx.figures_dir / "map.png")

The CLI command ``osv prep`` is bound automatically once the stage is
registered (see `STAGE_MODULES` for where the CLI imports stage classes from).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import importlib
import json
import logging
import os
import time
import traceback
import uuid
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

from osv import __version__
from osv.config import OSVConfig, config_hash
from osv.config.loader import config_data

logger = logging.getLogger(__name__)

#: Canonical stage order; the stage number is the 1-based position.
STAGE_ORDER: tuple[str, ...] = (
    "prep",
    "velest_mc",
    "velest_analyse",
    "build_model",
    "tradeoff",
    "invert",
    "analyse",
    "resolution",
    "validation",
    "report",
)

#: Default config sections (dotted paths into the config) each stage depends on.
DEFAULT_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "prep": ("project.random_seed", "inputs", "coordinates", "data_selection"),
    "velest_mc": ("project.random_seed", "velest"),
    "velest_analyse": ("velest",),
    "build_model": ("model3d", "coordinates"),
    "tradeoff": ("project.random_seed", "fmtomo"),
    "invert": ("project.random_seed", "fmtomo"),
    "analyse": ("plotting",),
    "resolution": ("project.random_seed", "resolution"),
    "validation": ("project.random_seed", "velest", "fmtomo"),
    "report": ("project", "plotting"),
}

#: Modules the CLI imports (if present) so their stages get registered.
STAGE_MODULES: dict[str, str] = {
    "prep": "osv.data.stage",
    "velest_mc": "osv.velest.stage",
    "velest_analyse": "osv.velest.stage",
    "build_model": "osv.model3d.stage",
    "tradeoff": "osv.fmtomo.tradeoff_stage",
    "invert": "osv.fmtomo.stage",
    "analyse": "osv.analysis.stage",
    "resolution": "osv.resolution.stage",
    "validation": "osv.validation.stage",
    "report": "osv.report.stage",
}

MANIFEST_NAME = "manifest.json"
RUNNING, COMPLETE, FAILED = "running", "complete", "failed"


class StageError(Exception):
    """Base class for stage-framework errors."""


class StageFailed(StageError):
    """A stage raised during `run_pipeline`; the pipeline stopped."""

    def __init__(self, stage: str, message: str, stage_dir: Path) -> None:
        super().__init__(
            f"Stage '{stage}' failed: {message}\nSee {stage_dir / MANIFEST_NAME} "
            f"and {stage_dir / 'logs'} for details; fix the problem and re-run."
        )
        self.stage = stage
        self.stage_dir = stage_dir


class MissingUpstreamError(StageError):
    """A required upstream stage has not completed."""


def normalise_name(name: str) -> str:
    """Accept ``velest-mc`` as well as ``velest_mc``."""
    return name.strip().replace("-", "_")


def stage_number(name: str) -> int:
    """1-based stage number from `STAGE_ORDER`."""
    try:
        return STAGE_ORDER.index(name) + 1
    except ValueError:
        raise StageError(f"Unknown stage '{name}'. Known: {', '.join(STAGE_ORDER)}") from None


def stage_dirname(name: str) -> str:
    """Directory name ``NN_<name>``."""
    return f"{stage_number(name):02d}_{name}"


# ---------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(path: Path) -> str:
    """sha256 of a file, of a directory tree (names + contents), or ``"missing"``."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        h = hashlib.sha256()
        for p in sorted(q for q in path.rglob("*") if q.is_file()):
            h.update(str(p.relative_to(path)).encode())
            h.update(sha256_file(p).encode())
        return h.hexdigest()
    return "missing"


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def config_subset(cfg: OSVConfig, sections: Iterable[str]) -> dict[str, Any]:
    """Pick dotted-path sections out of the config as plain data."""
    data = config_data(cfg)
    out: dict[str, Any] = {}
    for sec in sections:
        node: Any = data
        for part in sec.split("."):
            node = node[part]
        out[sec] = node
    return out


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))
    os.replace(tmp, path)


def read_manifest(stage_dir: Path) -> dict[str, Any] | None:
    """Return the parsed manifest of a stage directory, or None if absent/corrupt."""
    p = Path(stage_dir) / MANIFEST_NAME
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- stage + context


class Stage:
    """Base class for pipeline stages.

    Attributes
    ----------
    name
        Registry name; must be in `STAGE_ORDER`.
    requires
        Names of upstream stages whose outputs this stage reads.
    config_sections
        Dotted config paths this stage depends on (default from
        `DEFAULT_CONFIG_SECTIONS`).
    outputs
        Output paths relative to the stage directory; verified after `run`.
    tools
        Logical tool names (see `osv.tools`) whose binary hashes are recorded
        and make the stage stale when they change.
    """

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()
    config_sections: ClassVar[tuple[str, ...] | None] = None
    outputs: ClassVar[tuple[str, ...]] = ()
    tools: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def sections(cls) -> tuple[str, ...]:
        """Config sections used for hashing."""
        if cls.config_sections is not None:
            return cls.config_sections
        return DEFAULT_CONFIG_SECTIONS.get(cls.name, ())

    def inputs(self, ctx: StageContext) -> Sequence[Path]:
        """External input files/directories to hash (default: none)."""
        return ()

    def run(self, ctx: StageContext) -> None:
        """Do the work. Raise on failure."""
        raise NotImplementedError(f"Stage '{self.name}' does not implement run()")

    def plot(self, ctx: StageContext) -> None:
        """Regenerate figures from the stage outputs (default: nothing)."""


class UpstreamStage:
    """Read access to a completed upstream stage's outputs."""

    def __init__(self, name: str, stage_dir: Path, manifest: dict[str, Any]) -> None:
        self.name = name
        self.dir = stage_dir
        self.manifest = manifest

    def path(self, relpath: str) -> Path:
        """Absolute path of an output file; raises if it does not exist."""
        p = self.dir / relpath
        if not p.exists():
            raise FileNotFoundError(
                f"Upstream stage '{self.name}' has no output '{relpath}' (looked in {self.dir})"
            )
        return p


class StageContext:
    """Everything a stage needs while running."""

    def __init__(self, config: OSVConfig, stage: Stage, project_dir: Path | None = None) -> None:
        self.config = config
        self.stage = stage
        self.project_dir = Path(project_dir or config.project.workdir)
        self.stage_dir = self.project_dir / stage_dirname(stage.name)
        self.figures_dir = self.stage_dir / "figures"
        self.logs_dir = self.stage_dir / "logs"
        self.report_path = self.stage_dir / "report.html"
        self.logger = logging.getLogger(f"osv.stage.{stage.name}")
        self.checks: list[dict[str, Any]] = []
        #: Key-value pairs shown in the report's summary table (stored in the manifest).
        self.summary: dict[str, Any] = {}

    @property
    def figures(self) -> Any:
        """`osv.plotting.registry.FigureRegistry`: ``ctx.figures.save(fig, name, caption)``."""
        from osv.plotting.registry import figures_for

        return figures_for(self)

    @property
    def n_jobs(self) -> int:
        """Worker count with -1 resolved to the number of CPU cores."""
        n = self.config.project.n_jobs
        return (os.cpu_count() or 1) if n == -1 else n

    def path(self, relpath: str) -> Path:
        """Path inside this stage's directory (parents created)."""
        p = self.stage_dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def upstream(self, name: str) -> UpstreamStage:
        """Access a completed upstream stage (``ctx.upstream("prep").path("events.parquet")``)."""
        name = normalise_name(name)
        d = self.project_dir / stage_dirname(name)
        m = read_manifest(d)
        if m is None or m.get("status") != COMPLETE:
            raise MissingUpstreamError(
                f"Stage '{self.stage.name}' needs stage '{name}' to be complete "
                f"(status: {m.get('status') if m else 'not run'}). Run `osv run --to {name}` first."
            )
        return UpstreamStage(name, d, m)

    def add_check(self, check: Any) -> None:
        """Record a QC finding (a dict, or an object with ``to_dict()``/``asdict``)."""
        if hasattr(check, "to_dict"):
            check = check.to_dict()
        elif not isinstance(check, dict):
            from dataclasses import asdict, is_dataclass

            check = asdict(check) if is_dataclass(check) else {"message": str(check)}
        self.checks.append(check)

    def prepare_dirs(self) -> None:
        """Create the stage directory skeleton."""
        for d in (self.stage_dir, self.figures_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _file_logging(ctx: StageContext, filename: str) -> Iterator[None]:
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(ctx.logs_dir / filename, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("osv")
    old_level = root.level
    if old_level == logging.NOTSET or old_level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()


# ---------------------------------------------------------------- registry

_REGISTRY: dict[str, type[Stage]] = {}


def register(cls: type[Stage]) -> type[Stage]:
    """Class decorator registering a stage (replaces an earlier one of the same name)."""
    stage_number(cls.name)  # validates the name
    _REGISTRY[cls.name] = cls
    return cls


def unregister(name: str) -> None:
    """Remove a stage from the registry (used by tests)."""
    _REGISTRY.pop(name, None)


def discover() -> None:
    """Import the modules in `STAGE_MODULES` that exist, registering their stages."""
    for modname in STAGE_MODULES.values():
        try:
            importlib.import_module(modname)
        except ModuleNotFoundError as exc:
            if exc.name is None or not modname.startswith(exc.name):
                raise


def registered_stages() -> list[type[Stage]]:
    """Registered stage classes in canonical order."""
    return [_REGISTRY[n] for n in STAGE_ORDER if n in _REGISTRY]


def get_stage(name: str) -> type[Stage]:
    """Registered stage class by name."""
    name = normalise_name(name)
    stage_number(name)
    try:
        return _REGISTRY[name]
    except KeyError:
        raise StageError(f"Stage '{name}' is not implemented yet.") from None


# ---------------------------------------------------------------- up-to-date logic


def _tool_hashes(stage: Stage) -> dict[str, Any]:
    if not stage.tools:
        return {}
    from osv.tools import tool_versions

    return {
        k: {"path": v["path"], "sha256": v["sha256"]} for k, v in tool_versions(stage.tools).items()
    }


def _fingerprint(ctx: StageContext, stage: Stage) -> dict[str, Any]:
    """Everything whose change makes the stage stale."""
    sections = config_subset(ctx.config, stage.sections())
    ups: dict[str, str] = {}
    for req in stage.requires:
        m = read_manifest(ctx.project_dir / stage_dirname(req))
        ups[req] = (
            (m or {}).get("run_id", "none") if (m or {}).get("status") == COMPLETE else "none"
        )
    inputs = {str(Path(p)): hash_path(Path(p)) for p in stage.inputs(ctx)}
    return {
        "config_section": sections,
        "config_section_hash": _digest(sections),
        "input_hashes": inputs,
        "upstream": ups,
        "tools": _tool_hashes(stage),
    }


def stale_reason(ctx: StageContext, stage: Stage) -> str | None:
    """Why the stage must run, or None if it is up to date."""
    m = read_manifest(ctx.stage_dir)
    if m is None:
        return "not run yet"
    if m.get("status") != COMPLETE:
        return f"previous status was '{m.get('status')}'"
    fp = _fingerprint(ctx, stage)
    for key, label in (
        ("config_section_hash", "config section changed"),
        ("input_hashes", "inputs changed"),
        ("upstream", "upstream stage changed"),
    ):
        if m.get(key) != fp[key]:
            return label
    old_tools = {k: v.get("sha256") for k, v in (m.get("tools") or {}).items()}
    if old_tools != {k: v["sha256"] for k, v in fp["tools"].items()}:
        return "tool binaries changed"
    for rel in stage.outputs:
        if not (ctx.stage_dir / rel).exists():
            return f"output '{rel}' is missing"
    return None


# ---------------------------------------------------------------- execution


def _output_records(ctx: StageContext, stage: Stage) -> list[dict[str, Any]]:
    recs = []
    for rel in stage.outputs:
        p = ctx.stage_dir / rel
        if not p.exists():
            raise StageError(f"Stage '{stage.name}' did not produce declared output '{rel}'")
        recs.append(
            {
                "path": rel,
                "sha256": hash_path(p),
                "size": p.stat().st_size if p.is_file() else None,
            }
        )
    return recs


def render_report(ctx: StageContext) -> None:
    """Render ``report.html`` from the manifest and figures; never raises."""
    try:
        from osv.report.html import render_stage_report

        render_stage_report(ctx.stage_dir)
    except Exception:  # a broken report must not fail the stage
        logger.warning("Could not render %s", ctx.report_path, exc_info=True)


def _run_stage(cfg: OSVConfig, stage: Stage, project_dir: Path) -> None:
    ctx = StageContext(cfg, stage, project_dir)
    ctx.prepare_dirs()
    started = _now()
    t0 = time.monotonic()
    run_id = uuid.uuid4().hex
    manifest: dict[str, Any] = {
        "stage": stage.name,
        "number": stage_number(stage.name),
        "status": RUNNING,
        "run_id": run_id,
        "started": started,
        "finished": None,
        "wall_time_s": None,
        "osv_version": __version__,
        "config_hash": config_hash(cfg),
        "outputs": [],
        "checks": [],
        "error": None,
    }
    try:
        manifest.update(_fingerprint(ctx, stage))
        _write_json(ctx.stage_dir / MANIFEST_NAME, manifest)
        with _file_logging(ctx, "stage.log"):
            ctx.logger.info("Starting stage %s", stage.name)
            stage.run(ctx)
            ctx.logger.info("Finished stage %s", stage.name)
        manifest["outputs"] = _output_records(ctx, stage)
        if stage.tools:
            from osv.tools import tool_versions

            manifest["tools"] = tool_versions(stage.tools)
        manifest["checks"] = ctx.checks
        manifest["summary"] = ctx.summary
        manifest["status"] = COMPLETE
    except Exception as exc:
        manifest["status"] = FAILED
        manifest["checks"] = ctx.checks
        manifest["summary"] = ctx.summary
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        manifest["finished"] = _now()
        manifest["wall_time_s"] = round(time.monotonic() - t0, 3)
        _write_json(ctx.stage_dir / MANIFEST_NAME, manifest)
        render_report(ctx)
        raise StageFailed(stage.name, f"{type(exc).__name__}: {exc}", ctx.stage_dir) from exc
    manifest["finished"] = _now()
    manifest["wall_time_s"] = round(time.monotonic() - t0, 3)
    _write_json(ctx.stage_dir / MANIFEST_NAME, manifest)
    render_report(ctx)


def select_stages(
    from_: str | None = None, to: str | None = None, only: str | Sequence[str] | None = None
) -> list[type[Stage]]:
    """Registered stages selected by ``from_``/``to`` (inclusive) or ``only``."""
    if only is not None:
        names = [only] if isinstance(only, str) else list(only)
        wanted = {normalise_name(n) for n in names}
        for n in wanted:
            get_stage(n)
        return [c for c in registered_stages() if c.name in wanted]
    lo = stage_number(normalise_name(from_)) if from_ else 1
    hi = stage_number(normalise_name(to)) if to else len(STAGE_ORDER)
    if lo > hi:
        raise StageError(f"--from '{from_}' comes after --to '{to}' in the stage order")
    return [c for c in registered_stages() if lo <= stage_number(c.name) <= hi]


def run_pipeline(
    cfg: OSVConfig,
    from_: str | None = None,
    to: str | None = None,
    force: bool = False,
    only: str | Sequence[str] | None = None,
    project_dir: Path | None = None,
) -> dict[str, str]:
    """Run the selected stages in order, skipping those that are up to date.

    Parameters
    ----------
    cfg
        Resolved configuration.
    from_, to
        First/last stage (inclusive) by name; ``None`` means the ends.
    force
        Re-run selected stages even if up to date. Because a re-run gives a new
        run id, downstream stages become stale automatically.
    only
        Run just this stage (or these stages); overrides ``from_``/``to``.
    project_dir
        Defaults to ``cfg.project.workdir``.

    Returns
    -------
    dict
        ``{stage name: "ran" | "skipped"}`` in execution order.

    Raises
    ------
    StageFailed
        When a stage raises; its manifest holds status ``failed`` and the traceback.
    """
    project_dir = Path(project_dir or cfg.project.workdir)
    selected = select_stages(from_, to, only)
    result: dict[str, str] = {}
    for cls in selected:
        stage = cls()
        ctx = StageContext(cfg, stage, project_dir)
        for req in stage.requires:
            if req in {c.name for c in selected} and req not in result:
                continue
            up = read_manifest(project_dir / stage_dirname(req))
            if up is None or up.get("status") != COMPLETE:
                raise MissingUpstreamError(
                    f"Stage '{stage.name}' requires stage '{req}', which has not completed. "
                    f"Run `osv run --to {req}` (or include it in --from/--to)."
                )
        reason = "forced" if force else stale_reason(ctx, stage)
        if reason is None:
            logger.info("Stage %s is up to date; skipping", stage.name)
            result[stage.name] = "skipped"
            continue
        logger.info("Running stage %s (%s)", stage.name, reason)
        _run_stage(cfg, stage, project_dir)
        result[stage.name] = "ran"
    return result


def plot_only(cfg: OSVConfig, stage: str, project_dir: Path | None = None) -> None:
    """Regenerate a completed stage's figures without re-running it."""
    project_dir = Path(project_dir or cfg.project.workdir)
    inst = get_stage(stage)()
    ctx = StageContext(cfg, inst, project_dir)
    m = read_manifest(ctx.stage_dir)
    if m is None or m.get("status") != COMPLETE:
        raise MissingUpstreamError(
            f"Stage {inst.name!r} has not completed; run `osv run --to {inst.name}` first."
        )
    ctx.prepare_dirs()
    with _file_logging(ctx, "plot.log"):
        inst.plot(ctx)
    m["plotted"] = _now()
    _write_json(ctx.stage_dir / MANIFEST_NAME, m)
    render_report(ctx)
