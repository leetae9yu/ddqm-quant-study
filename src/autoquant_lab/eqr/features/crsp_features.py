"""CRSP monthly price-derived features."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import numpy as np
import pandas as pd

from .result import FeatureBuildResult


FEATURES = {
    "crsp__mom_12_2": "12-to-2 month compounded return momentum",
    "crsp__mom_6_2": "6-to-2 month compounded return momentum",
    "crsp__reversal_1m": "most recent monthly return reversal signal",
    "crsp__log_size": "log market capitalization",
}


def build_crsp_features(panel: pd.DataFrame, **_: object) -> FeatureBuildResult:
    """Build momentum, reversal, and size from the monthly panel only."""

    required = {"permno", "formation_date", "ret_1m", "market_cap"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Panel missing CRSP feature columns: {missing}")

    df = panel[["permno", "formation_date", "ret_1m", "market_cap"]].copy()
    df["formation_date"] = pd.to_datetime(df["formation_date"], errors="coerce")
    df = df.sort_values(["permno", "formation_date"])
    grouped = df.groupby("permno", group_keys=False)
    shifted_log_return = np.log1p(grouped["ret_1m"].shift(2))
    df["crsp__mom_12_2"] = np.expm1(shifted_log_return.groupby(df["permno"]).rolling(11, min_periods=8).sum().reset_index(level=0, drop=True))
    df["crsp__mom_6_2"] = np.expm1(shifted_log_return.groupby(df["permno"]).rolling(5, min_periods=4).sum().reset_index(level=0, drop=True))
    df["crsp__reversal_1m"] = df["ret_1m"]
    df["crsp__log_size"] = np.log(df["market_cap"].where(df["market_cap"] > 0))
    out = df[["permno", "formation_date", *FEATURES.keys()]].copy()
    metadata = [
        {
            "feature": feature,
            "source_table": "monthly_labels_panel",
            "availability_rule": "uses CRSP monthly values available at formation month end",
            "feature_family": "crsp",
            "leakage_rule": description + "; no future monthly returns are referenced",
        }
        for feature, description in FEATURES.items()
    ]
    return FeatureBuildResult("crsp", out, metadata)
