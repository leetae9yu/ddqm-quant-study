"""DDQM2 factor long-short return labels."""
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportArgumentType=false

from __future__ import annotations

import numpy as np
import pandas as pd


def _bucket_returns(group: pd.DataFrame, quantile: float) -> pd.Series:
    clean = group.dropna(subset=["factor_score", "forward_return"])
    if len(clean) < 10:
        return pd.Series({"long_return": np.nan, "short_return": np.nan, "factor_long_short_ret_1m": np.nan, "long_count": 0, "short_count": 0})
    long_cut = clean["factor_score"].quantile(1.0 - quantile)
    short_cut = clean["factor_score"].quantile(quantile)
    long_leg = clean.loc[clean["factor_score"] >= long_cut, "forward_return"]
    short_leg = clean.loc[clean["factor_score"] <= short_cut, "forward_return"]
    return pd.Series(
        {
            "long_return": float(long_leg.mean()) if not long_leg.empty else np.nan,
            "short_return": float(short_leg.mean()) if not short_leg.empty else np.nan,
            "factor_long_short_ret_1m": float(long_leg.mean() - short_leg.mean()) if not long_leg.empty and not short_leg.empty else np.nan,
            "long_count": int(long_leg.count()),
            "short_count": int(short_leg.count()),
        }
    )


def build_factor_long_short_returns(
    factor_scores: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    return_column: str = "ret_1m_fwd",
    quantile: float = 0.2,
) -> pd.DataFrame:
    """Build DDQM2 labels: one forward long-short return per factor/date."""

    if not (0.0 < quantile < 0.5):
        raise ValueError("quantile must be between 0 and 0.5")
    required_scores = {"formation_date", "permno", "factor_id", "factor_score"}
    missing_scores = sorted(required_scores.difference(factor_scores.columns))
    if missing_scores:
        raise ValueError(f"Factor scores missing columns: {missing_scores}")
    required_panel = {"formation_date", "permno", return_column}
    missing_panel = sorted(required_panel.difference(panel.columns))
    if missing_panel:
        raise ValueError(f"Panel missing return label columns: {missing_panel}")

    returns = panel[["formation_date", "permno", return_column]].copy()
    returns["formation_date"] = pd.to_datetime(returns["formation_date"], errors="coerce")
    returns["permno"] = pd.to_numeric(returns["permno"], errors="coerce").astype("Int64")
    returns = returns.rename(columns={return_column: "forward_return"})
    scores = factor_scores.copy()
    scores["formation_date"] = pd.to_datetime(scores["formation_date"], errors="coerce")
    scores["permno"] = pd.to_numeric(scores["permno"], errors="coerce").astype("Int64")
    merged = scores.merge(returns, on=["formation_date", "permno"], how="inner")
    rows: list[dict[str, object]] = []
    for (formation_date, factor_id), group in merged.groupby(["formation_date", "factor_id"], sort=True):
        bucket = _bucket_returns(group[["factor_score", "forward_return"]], quantile).to_dict()
        bucket["formation_date"] = formation_date
        bucket["factor_id"] = factor_id
        rows.append(bucket)
    labels = pd.DataFrame(rows)
    labels["label_horizon"] = return_column
    labels["quantile"] = quantile
    return labels.sort_values(["formation_date", "factor_id"]).reset_index(drop=True)
