"""Static reporting site generation for EQR experiments."""
# pyright: reportImportCycles=false

from __future__ import annotations

from .renderer import RenderResult, render_site

__all__ = ["RenderResult", "render_site"]
