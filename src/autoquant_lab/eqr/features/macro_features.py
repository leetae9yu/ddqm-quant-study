"""Macro/market feature snapshots available by month end."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import pandas as pd

from .result import FeatureBuildResult


def build_macro_features(panel: pd.DataFrame, macro_features: pd.DataFrame | None = None, **_: object) -> FeatureBuildResult:
    """Aggregate business-day macro rows to month-end PIT snapshots."""

    if macro_features is None or macro_features.empty:
        return FeatureBuildResult("macro", pd.DataFrame(columns=("permno", "formation_date")), [])
    if "date" not in macro_features.columns:
        raise ValueError("macro_features must contain a date column")

    macro = macro_features.copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    macro = macro.loc[macro["date"].notna()].sort_values("date")
    value_cols = [col for col in macro.columns if col != "date"]
    macro["formation_date"] = macro["date"] + pd.offsets.MonthEnd(0)
    snapshots = macro.groupby("formation_date", as_index=False)[value_cols].last()
    snapshots = snapshots.rename(columns={col: f"macro__{col}" for col in value_cols})

    keys = panel[["permno", "formation_date"]].copy()
    keys["formation_date"] = pd.to_datetime(keys["formation_date"], errors="coerce")
    features = keys.merge(snapshots, on="formation_date", how="left")
    metadata = [
        {
            "feature": f"macro__{col}",
            "source_table": "macro_features",
            "availability_rule": "last non-future observation within same calendar month, stamped at month end",
            "feature_family": "macro",
            "leakage_rule": "macro rows with date after formation month-end are never used",
        }
        for col in value_cols
    ]
    return FeatureBuildResult("macro", features, metadata)
