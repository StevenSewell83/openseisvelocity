"""Project report aggregator: one ``index.html`` (and ``summary.md``) over all stage directories.

`collect_project` reads every ``NN_<stage>/`` directory of a work directory (manifest, figure
registry, files) into plain data; `render_project_report` turns that into the HTML page and
`render_project_summary` into a compact Markdown summary.  Missing stages and corrupt
manifests are tolerated: the page just shows "not run".
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import fnmatch
import html as _html
import json
import mimetypes
from pathlib import Path
from typing import Any

from osv import __version__
from osv.report.checks import SEVERITIES
from osv.report.html import _ENV, normalise_check
from osv.stages import STAGE_ORDER, stage_dirname

#: Summary keys (glob patterns, in display order) shown as headline numbers per stage.
HEADLINES: dict[str, tuple[str, ...]] = {
    "prep": (
        "events_in",
        "events_selected",
        "events_calibration",
        "events_validation",
        "picks_P_selected",
        "picks_S_selected",
        "stations_used",
        "vpvs",
        "vpvs_ci_95",
        "median_azgap_deg",
    ),
    "velest_mc": ("trial_models", "runs_ok", "runs_failed", "rms_range_s", "vpvs"),
    "velest_analyse": (
        "final_model_rms_s",
        "ensemble_rms_range_s",
        "vpvs",
        "runs_selected",
        "closest_run",
        "hypocentres",
        "station_delays",
    ),
    "build_model": (
        "source",
        "grid_interior",
        "spacing_km",
        "grid_nodes",
        "sources",
        "receivers",
        "observations_*",
        "vp_range_km_s",
        "rms_P_starting_s",
    ),
    "tradeoff": ("chosen_damping_*", "chosen_smoothing_*", "mode", "runs_*"),
    "invert": (
        "damping_*",
        "smoothing_*",
        "iterations_*",
        "rms_start_ms_*",
        "rms_final_ms_*",
        "variance_reduction_pct_*",
        "covered_fraction_*",
        "hypocentre_shift_median_km_*",
    ),
    "analyse": (
        "variance_reduction_pct_*",
        "final_rms_ms_*",
        "covered_fraction_*",
        "resolvable_fraction_*",
        "n_picks_*",
    ),
    "resolution": ("*resolv*", "*recover*", "*covered*", "n_*"),
    "validation": (
        "model_3d",
        "events",
        "phases",
        "median_rms_1d_ms",
        "median_rms_3d_ms",
        "median_rms_improvement_ms",
        "fraction_3d_better",
        "fraction_3d_worse",
        "wilcoxon_p",
    ),
}

#: Figure names (glob patterns, in preference order) embedded per stage; fallback: first figure.
KEY_FIGURES: dict[str, tuple[str, ...]] = {
    "prep": ("map_before_after", "travel_times"),
    "velest_mc": ("velocity_density", "rms_*"),
    "velest_analyse": ("velocity_density", "convergence_summary", "final_hypocentres"),
    "build_model": ("vp_slices", "section_ew", "grid_geometry"),
    "tradeoff": ("tradeoff*", "lcurve*", "l_curve*"),
    "invert": ("convergence*", "misfit*", "rms*"),
    "analyse": ("dvp_slices*", "dvp_section*", "variance*"),
    "resolution": ("checkerboard*", "resolvable*", "coverage*"),
    "validation": ("rms_1d_vs_3d*", "*rms*", "*improvement*"),
}
MAX_KEY_FIGURES = 2

_EXPORT_SUFFIXES = {".nc", ".csv", ".mod", ".in", ".xyz", ".vtk", ".vtu", ".vtr", ".parquet"}
_MAX_EXPORTS = 25


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list | tuple):
        return ", ".join(_fmt(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items())
    return str(value)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        return _dt.datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(iso)


def headline_numbers(stage: str, summary: dict[str, Any]) -> list[tuple[str, str]]:
    """Pick the headline ``(key, formatted value)`` pairs of a stage summary.

    Keys matching `HEADLINES` patterns come first (pattern order); when nothing matches the
    first few scalar entries are used.
    """
    chosen: list[str] = []
    for pat in HEADLINES.get(stage, ()):
        for k in summary:
            if k not in chosen and fnmatch.fnmatchcase(k, pat):
                chosen.append(k)
    if not chosen:
        chosen = [k for k, v in summary.items() if not isinstance(v, dict) and len(_fmt(v)) < 60][
            :8
        ]
    return [(k, _fmt(summary[k])) for k in chosen]


def _model_table(stage_dir: Path) -> list[dict[str, str]] | None:
    """Final 1-D model rows (depth, Vp, Vs) from ``final_model.csv`` if present."""
    p = stage_dir / "final_model.csv"
    try:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    out = []
    for r in rows:
        try:
            out.append(
                {
                    "depth": f"{float(r['depth_top_km']):.2f}",
                    "vp": f"{float(r['vp']):.3f}",
                    "vs": f"{float(r['vs']):.3f}",
                }
            )
        except (KeyError, ValueError):
            return None
    return out or None


def _pick_figures(stage: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for pat in KEY_FIGURES.get(stage, ()):
        for e in entries:
            if e not in picked and fnmatch.fnmatchcase(e.get("name", ""), pat):
                picked.append(e)
                break
        if len(picked) >= MAX_KEY_FIGURES:
            break
    if not picked and entries:
        picked = [entries[0]]
    return picked[:MAX_KEY_FIGURES]


def _raster(entry: dict[str, Any]) -> str:
    files = entry.get("files", [])
    return next((f for f in files if f.endswith((".png", ".svg", ".jpg"))), "")


def _exports(stage_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Relative paths (to the stage directory) of exportable result files."""
    found: list[str] = []
    for rec in manifest.get("outputs") or []:
        rel = rec.get("path", "")
        p = stage_dir / rel
        if p.is_file() and Path(rel).suffix in _EXPORT_SUFFIXES:
            found.append(rel)
    exp = stage_dir / "exports"
    if exp.is_dir():
        found += [str(p.relative_to(stage_dir)) for p in sorted(exp.rglob("*")) if p.is_file()]
    return list(dict.fromkeys(found))[:_MAX_EXPORTS]


def collect_project(workdir: Path, config: Any = None) -> dict[str, Any]:
    """Read all stage directories of ``workdir`` into plain data for rendering.

    Parameters
    ----------
    workdir
        Project work directory containing ``NN_<stage>`` directories.
    config
        Optional resolved `osv.config.OSVConfig` (project name, hash, snapshot).
    """
    workdir = Path(workdir)
    stages: list[dict[str, Any]] = []
    tools: dict[str, dict[str, str]] = {}
    versions: set[str] = set()
    starts: list[str] = []
    ends: list[str] = []
    for name in STAGE_ORDER:
        if name == "report":
            continue
        d = workdir / stage_dirname(name)
        m = _read_json(d / "manifest.json") if d.is_dir() else None
        m = m if isinstance(m, dict) else None
        checks = [normalise_check(c) for c in (m or {}).get("checks") or []]
        checks.sort(key=lambda c: -SEVERITIES.index(c["severity"]))
        summary = (m or {}).get("summary") or {}
        figs = (_read_json(d / "figures" / "figures.json") or {}).get("figures", []) if m else []
        key_figs = [
            {
                "name": e.get("name", ""),
                "caption": e.get("caption", ""),
                "rel": f"{d.name}/figures/{_raster(e)}",
                "path": d / "figures" / _raster(e),
            }
            for e in _pick_figures(name, figs)
            if _raster(e)
        ]
        for t, rec in ((m or {}).get("tools") or {}).items():
            tools[t] = {"path": str(rec.get("path", "")), "sha": str(rec.get("sha256", ""))[:12]}
        if m and m.get("osv_version"):
            versions.add(str(m["osv_version"]))
        if m and m.get("started"):
            starts.append(m["started"])
        if m and m.get("finished"):
            ends.append(m["finished"])
        stages.append(
            {
                "name": name,
                "dir": d.name,
                "status": (m or {}).get("status", "not run"),
                "wall": (m or {}).get("wall_time_s"),
                "wall_text": _fmt_duration((m or {}).get("wall_time_s")),
                "finished": _fmt_date((m or {}).get("finished")),
                "checks": checks,
                "n_fail": sum(c["severity"] == "fail" for c in checks),
                "n_warn": sum(c["severity"] == "warn" for c in checks),
                "report_rel": f"{d.name}/report.html" if (d / "report.html").exists() else None,
                "headlines": headline_numbers(name, summary) if m else [],
                "model_table": _model_table(d) if name == "velest_analyse" else None,
                "figures": key_figs,
                "exports": _exports(d, m) if m else [],
                "error": (m or {}).get("error"),
            }
        )
    cfg_json = ""
    cfg_hash = ""
    project_name = workdir.parent.name if workdir.name == "runs" else workdir.name
    if config is not None:
        from osv.config import config_hash
        from osv.config.loader import config_data

        cfg_hash = config_hash(config)
        cfg_json = json.dumps(config_data(config), indent=2, default=str)
        project_name = config.project.name
    return {
        "project": project_name,
        "workdir": str(workdir),
        "config_hash": cfg_hash,
        "config_json": cfg_json,
        "osv_version": __version__,
        "stage_versions": sorted(versions),
        "tools": dict(sorted(tools.items())),
        "first_start": _fmt_date(min(starts)) if starts else "-",
        "last_finish": _fmt_date(max(ends)) if ends else "-",
        "generated": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "stages": stages,
        "n_fail": sum(s["n_fail"] for s in stages),
        "n_warn": sum(s["n_warn"] for s in stages),
        "n_run": sum(s["status"] == "complete" for s in stages),
    }


def _data_uri(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def render_project_report(
    data: dict[str, Any],
    out_path: Path,
    *,
    base: str = "",
    embed_images: bool = False,
) -> Path:
    """Write the HTML project report to ``out_path``.

    Parameters
    ----------
    data
        Result of `collect_project`.
    out_path
        Output file.
    base
        Prefix of every link, i.e. the path from ``out_path``'s directory to the work
        directory (``""`` for ``<workdir>/index.html``, ``"../"`` for ``<workdir>/10_report/``).
    embed_images
        Embed figures as base64 data URIs (single portable file; links to stage reports and
        exports stay relative).
    """
    stages = []
    for s in data["stages"]:
        figs = []
        for f in s["figures"]:
            src = (_data_uri(f["path"]) if embed_images else None) or base + f["rel"]
            figs.append({**f, "src": src})
        stages.append(
            {
                **s,
                "figures": figs,
                "report_href": base + s["report_rel"] if s["report_rel"] else None,
                "export_links": [(e, f"{base}{s['dir']}/{e}") for e in s["exports"]],
            }
        )
    page = _ENV.get_template("project_report.html").render(
        **{**data, "stages": stages}, base=base, esc=_html.escape
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def render_project_summary(data: dict[str, Any], out_path: Path) -> Path:
    """Write the compact Markdown summary (for pasting into notes)."""
    L = [f"# {data['project']}", ""]
    L.append(
        f"openSeisVelocity {data['osv_version']}; run {data['first_start']} to "
        f"{data['last_finish']}; {data['n_run']} stages complete; "
        f"{data['n_fail']} fail, {data['n_warn']} warn."
    )
    if data["config_hash"]:
        L.append(f"Config hash: `{data['config_hash'][:12]}`")
    L += ["", "| Stage | Status | Time | Fail | Warn |", "|---|---|---|---|---|"]
    for s in data["stages"]:
        L.append(
            f"| {s['name']} | {s['status']} | {s['wall_text']} | {s['n_fail']} | {s['n_warn']} |"
        )
    for s in data["stages"]:
        if not s["headlines"] and not s["model_table"]:
            continue
        L += ["", f"## {s['name']}", ""]
        L += [f"- {k}: {v}" for k, v in s["headlines"]]
        if s["model_table"]:
            L += ["", "| depth_top_km | Vp | Vs |", "|---|---|---|"]
            L += [f"| {r['depth']} | {r['vp']} | {r['vs']} |" for r in s["model_table"]]
    flagged = [
        (s["name"], c) for s in data["stages"] for c in s["checks"] if c["severity"] != "info"
    ]
    if flagged:
        L += ["", "## QC findings", ""]
        L += [f"- [{c['severity'].upper()}] {n}/{c['code']}: {c['message']}" for n, c in flagged]
    out = Path(out_path)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


__all__ = [
    "HEADLINES",
    "KEY_FIGURES",
    "collect_project",
    "headline_numbers",
    "render_project_report",
    "render_project_summary",
]
