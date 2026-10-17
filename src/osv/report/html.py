"""Render self-contained HTML reports (per stage and a project index stub).

Entry points: `render_stage_report` (called by the stage framework after
``run`` and after ``plot``) and `render_project_index`.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

import jinja2

from osv.report.checks import SEVERITIES

_ENV = jinja2.Environment(
    loader=jinja2.PackageLoader("osv.report", "templates"),
    autoescape=jinja2.select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

_RANK = {"fail": 0, "warn": 1, "info": 2}
_STATUS_ALIASES = {
    "pass": "info",
    "ok": "info",
    "passed": "info",
    "warning": "warn",
    "error": "fail",
    "failed": "fail",
}


def normalise_check(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce a check dict (QCFinding or legacy ``name``/``status`` form) to display form."""
    sev = str(raw.get("severity", raw.get("status", "info"))).lower()
    sev = _STATUS_ALIASES.get(sev, sev)
    if sev not in SEVERITIES:
        sev = "info"
    return {
        "code": raw.get("code", raw.get("name", "")),
        "severity": sev,
        "message": raw.get("message", ""),
        "count": raw.get("count", 0) or 0,
        "details": raw.get("details") or {},
    }


def sort_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise and order checks fail, warn, info (stable within a level)."""
    return sorted((normalise_check(c) for c in checks), key=lambda c: _RANK[c["severity"]])


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _log_tail(stage_dir: Path, n: int = 40) -> str:
    log = stage_dir / "logs" / "stage.log"
    try:
        return "\n".join(log.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict | list):
        return json.dumps(value, default=str)
    return str(value)


def _group_figures(entries: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        files = e.get("files", [])
        raster = next((f for f in files if f.endswith((".png", ".svg"))), files[0] if files else "")
        vector = next((f for f in files if f.endswith(".pdf")), None)
        groups.setdefault(e.get("section") or "Figures", []).append(
            {
                "name": e.get("name", ""),
                "caption": e.get("caption", ""),
                "src": f"figures/{raster}" if raster else "",
                "link": f"figures/{vector or raster}" if (vector or raster) else "",
            }
        )
    return list(groups.items())


def render_stage_report(
    stage_dir: Path,
    summary: dict[str, Any] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write ``report.html`` for a stage directory from its manifest and figures.json.

    Parameters
    ----------
    stage_dir
        The ``NN_<name>`` directory (reads ``manifest.json``, ``figures/figures.json``,
        ``logs/stage.log``).
    summary
        Key-value table; defaults to ``manifest["summary"]``.
    out_path
        Defaults to ``<stage_dir>/report.html``.
    """
    stage_dir = Path(stage_dir)
    manifest = _read_json(stage_dir / "manifest.json") or {}
    figs = (_read_json(stage_dir / "figures" / "figures.json") or {}).get("figures", [])
    checks = sort_checks(manifest.get("checks") or [])
    summary = summary if summary is not None else manifest.get("summary") or {}
    status = manifest.get("status", "unknown")
    error = manifest.get("error")
    counts = {s: sum(c["severity"] == s for c in checks) for s in SEVERITIES}
    page = _ENV.get_template("stage_report.html").render(
        stage=manifest.get("stage", stage_dir.name),
        number=manifest.get("number"),
        status=status,
        manifest=manifest,
        checks=checks,
        counts=counts,
        summary=[(k, _fmt(v)) for k, v in summary.items()],
        figure_groups=_group_figures(figs),
        config_json=json.dumps(manifest.get("config_section") or {}, indent=2, default=str),
        error=error,
        log_tail=_log_tail(stage_dir) if status == "failed" else "",
        esc=_html.escape,
    )
    out = Path(out_path) if out_path else stage_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def render_project_index(workdir: Path, out_path: Path | None = None) -> Path:
    """Write a minimal ``index.html`` listing stage directories, status and report links.

    The full project report aggregator is a later task (8.3).
    """
    workdir = Path(workdir)
    rows = []
    dirs = workdir.iterdir() if workdir.is_dir() else []
    for d in sorted(p for p in dirs if p.is_dir() and p.name[:2].isdigit()):
        m = _read_json(d / "manifest.json") or {}
        checks = [normalise_check(c) for c in m.get("checks") or []]
        rows.append(
            {
                "dir": d.name,
                "status": m.get("status", "not run"),
                "wall": m.get("wall_time_s"),
                "finished": m.get("finished") or "",
                "n_fail": sum(c["severity"] == "fail" for c in checks),
                "n_warn": sum(c["severity"] == "warn" for c in checks),
                "report": f"{d.name}/report.html" if (d / "report.html").exists() else None,
            }
        )
    page = _ENV.get_template("project_index.html").render(rows=rows, workdir=str(workdir))
    out = Path(out_path) if out_path else workdir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out
