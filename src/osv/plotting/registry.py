"""Per-stage figure registry: save in all configured formats and record metadata."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

import osv.plotting.style  # noqa: F401  (selects headless backend)

if TYPE_CHECKING:
    from osv.stages import StageContext

FIGURES_JSON = "figures.json"


@dataclass
class FigureRegistry:
    """Saves figures into ``figures_dir`` and tracks them in ``figures.json``.

    Entries are keyed by figure name: saving a name again replaces its entry, so
    regenerating figures (``osv plot``) is idempotent.
    """

    figures_dir: Path
    formats: tuple[str, ...] = ("png", "pdf")
    dpi: int = 200

    def __post_init__(self) -> None:
        self.figures_dir = Path(self.figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @property
    def json_path(self) -> Path:
        return self.figures_dir / FIGURES_JSON

    def entries(self) -> list[dict[str, Any]]:
        """Recorded figures (in save order)."""
        try:
            return json.loads(self.json_path.read_text())["figures"]
        except (OSError, ValueError, KeyError):
            return []

    def save(self, fig, name: str, caption: str, section: str | None = None) -> list[str]:
        """Write ``fig`` as ``<name>.<fmt>`` for each format, record it and close it.

        Returns the file names written (relative to the figures directory).
        """
        files = []
        try:
            for fmt in self.formats:
                fname = f"{name}.{fmt}"
                fig.savefig(self.figures_dir / fname, dpi=self.dpi, format=fmt)
                files.append(fname)
        finally:
            plt.close(fig)
        entry = {"name": name, "caption": caption, "section": section, "files": files}
        entries = [e for e in self.entries() if e.get("name") != name] + [entry]
        tmp = self.json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"figures": entries}, indent=2))
        os.replace(tmp, self.json_path)
        return files

    def clear(self) -> None:
        """Forget all recorded figures (files are left in place)."""
        self.json_path.unlink(missing_ok=True)


def figures_for(ctx: StageContext) -> FigureRegistry:
    """Registry for a stage context, using ``config.plotting`` format and dpi (cached on ctx)."""
    reg = getattr(ctx, "_figure_registry", None)
    if reg is None:
        p = ctx.config.plotting
        reg = FigureRegistry(ctx.figures_dir, tuple(p.format), int(p.dpi))
        ctx._figure_registry = reg  # type: ignore[attr-defined]
    return reg
