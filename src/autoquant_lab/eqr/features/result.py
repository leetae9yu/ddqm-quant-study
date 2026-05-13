"""Shared feature builder result types."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureBuildResult:
    """Feature frame plus per-column metadata."""

    family: str
    frame: pd.DataFrame
    metadata: list[dict[str, str]]
