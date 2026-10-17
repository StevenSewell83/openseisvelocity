"""The ``report`` stage: aggregate all stage directories into the project report.

Outputs (in ``<workdir>/10_report/``): ``index.html``, ``summary.md``; a copy of the page is
written to ``<workdir>/index.html``.  The stage always re-runs when any stage manifest changed
(the manifests are its hashed inputs), so it never needs ``--force`` after other stages ran.
"""

from __future__ import annotations

from pathlib import Path

from osv.report.project import (
    collect_project,
    render_project_report,
    render_project_summary,
)
from osv.stages import (
    STAGE_ORDER,
    Stage,
    StageContext,
    register,
    stage_dirname,
)


@register
class ReportStage(Stage):
    """Project report; needs no stage strictly, shows whatever exists."""

    name = "report"
    requires = ()
    outputs = ("index.html", "summary.md")

    def inputs(self, ctx: StageContext) -> list[Path]:
        return [
            ctx.project_dir / stage_dirname(n) / "manifest.json"
            for n in STAGE_ORDER
            if n != "report"
        ]

    def run(self, ctx: StageContext) -> None:
        data = collect_project(ctx.project_dir, ctx.config)
        embed = ctx.config.report.embed_images
        render_project_report(data, ctx.path("index.html"), base="../", embed_images=embed)
        render_project_report(data, ctx.project_dir / "index.html", base="", embed_images=embed)
        render_project_summary(data, ctx.path("summary.md"))
        ctx.summary.update(
            stages_complete=data["n_run"],
            stages_total=len(data["stages"]),
            qc_fail=data["n_fail"],
            qc_warn=data["n_warn"],
            embed_images=embed,
        )
        if data["n_fail"]:
            ctx.add_check(
                {
                    "code": "stage_failures",
                    "severity": "warn",
                    "message": f"{data['n_fail']} QC failure(s) in upstream stages",
                }
            )
        for s in data["stages"]:
            if s["status"] not in ("complete", "not run"):
                ctx.add_check(
                    {
                        "code": "stage_incomplete",
                        "severity": "warn",
                        "message": f"stage {s['name']} has status {s['status']}",
                    }
                )
