"""Plotting helpers (matplotlib by default, optional PyGMT)."""

from osv.plotting.registry import FigureRegistry, figures_for
from osv.plotting.style import apply_style, new_figure

__all__ = ["FigureRegistry", "apply_style", "figures_for", "new_figure"]
