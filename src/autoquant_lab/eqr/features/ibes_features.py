"""IBES point-in-time estimate, revision, surprise, and target-price features."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import numpy as np
import pandas as pd

from .result import FeatureBuildResult


FEATURES = {
    "ibes__mean_estimate": "estimates",
    "ibes__estimate_dispersion": "estimates",
    "ibes__num_estimates": "estimates",
    "ibes__revision_1m": "revisions",
    "ibes__surprise": "surprises",
    "ibes__target_price_mean": "target_price",
    "ibes__target_revision_balance_1m": "target_price",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator.abs() > 1e-12)
    return numerator / denom


def _clean_ticker(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    return result


def _active_ibes_link(rows: pd.DataFrame, ibes_link: pd.DataFrame, date_col: str) -> pd.DataFrame:
    links = _clean_ticker(ibes_link)
    links["permno"] = pd.to_numeric(links["permno"], errors="coerce")
    links["sdate"] = pd.to_datetime(links["sdate"], errors="coerce")
    links["edate"] = pd.to_datetime(links["edate"], errors="coerce").fillna(pd.Timestamp.max.normalize())
    if "score" in links.columns:
        links["_score"] = pd.to_numeric(links["score"], errors="coerce").fillna(999)
    else:
        links["_score"] = 0
    merged = rows.merge(links[["ticker", "permno", "sdate", "edate", "_score"]], on="ticker", how="inner")
    active = merged.loc[
        merged["permno"].notna()
        & merged[date_col].notna()
        & (merged["sdate"] <= merged[date_col])
        & (merged[date_col] <= merged["edate"])
    ].copy()
    return active.sort_values(["ticker", date_col, "_score"]).drop_duplicates(["ticker", date_col, "permno"], keep="first")


def _merge_asof_by_permno(panel_keys: pd.DataFrame, feature_rows: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in feature_rows.columns if col.startswith("ibes__")]
    if feature_rows.empty:
        return panel_keys.assign(**{col: np.nan for col in feature_cols})
    left = panel_keys.copy()
    left["_row_order"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values(["formation_date", "permno"]),
        feature_rows[["permno", "available_date", *feature_cols]].sort_values(["available_date", "permno"]),
        by="permno",
        left_on="formation_date",
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    ).drop(columns=["available_date"])
    return merged.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)


def _summary_rows(ibes_summary: pd.DataFrame) -> pd.DataFrame:
    summary = _clean_ticker(ibes_summary)
    summary["statpers"] = pd.to_datetime(summary["statpers"], errors="coerce")
    for col in ["meanest", "stdev", "numest", "actual", "actdats_act", "anndats_act"]:
        if col in ["actdats_act", "anndats_act"]:
            summary[col] = pd.to_datetime(summary.get(col), errors="coerce")
        else:
            summary[col] = pd.to_numeric(summary.get(col), errors="coerce")
    if "measure" in summary.columns:
        summary = summary.loc[summary["measure"].astype(str).str.upper() == "EPS"]
    summary = summary.sort_values(["ticker", "statpers"])
    summary["ibes__mean_estimate"] = summary["meanest"]
    summary["ibes__estimate_dispersion"] = _safe_divide(summary["stdev"], summary["meanest"].abs())
    summary["ibes__num_estimates"] = summary["numest"]
    prior_mean = summary.groupby("ticker")["meanest"].shift(1)
    summary["ibes__revision_1m"] = _safe_divide(summary["meanest"] - prior_mean, prior_mean.abs())
    available_actual = summary["actdats_act"].fillna(summary["anndats_act"])
    summary["ibes__surprise"] = np.where(available_actual.notna() & (available_actual <= summary["statpers"]), summary["actual"] - summary["meanest"], np.nan)
    summary["available_date"] = summary["statpers"]
    return summary[["ticker", "available_date", "ibes__mean_estimate", "ibes__estimate_dispersion", "ibes__num_estimates", "ibes__revision_1m", "ibes__surprise"]]


def _target_rows(ibes_target: pd.DataFrame | None) -> pd.DataFrame:
    if ibes_target is None or ibes_target.empty:
        return pd.DataFrame(columns=["ticker", "available_date", "ibes__target_price_mean", "ibes__target_revision_balance_1m"])
    target = _clean_ticker(ibes_target)
    target["statpers"] = pd.to_datetime(target["statpers"], errors="coerce")
    for col in ["meanptg", "numup1m", "numdown1m", "numest"]:
        target[col] = pd.to_numeric(target.get(col), errors="coerce")
    target["ibes__target_price_mean"] = target["meanptg"]
    target["ibes__target_revision_balance_1m"] = _safe_divide(target["numup1m"].fillna(0.0) - target["numdown1m"].fillna(0.0), target["numest"])
    target["available_date"] = target["statpers"]
    return target[["ticker", "available_date", "ibes__target_price_mean", "ibes__target_revision_balance_1m"]]


def build_ibes_features(
    panel: pd.DataFrame,
    ibes_link: pd.DataFrame | None = None,
    ibes_summary: pd.DataFrame | None = None,
    ibes_target: pd.DataFrame | None = None,
    **_: object,
) -> FeatureBuildResult:
    """Build IBES PIT features using statpers and announcement availability."""

    if ibes_link is None or ibes_summary is None or ibes_link.empty or ibes_summary.empty:
        return FeatureBuildResult("ibes", panel[["permno", "formation_date"]].copy(), [])

    summary = _summary_rows(ibes_summary)
    targets = _target_rows(ibes_target)
    rows = summary.merge(targets, on=["ticker", "available_date"], how="outer")
    rows = rows.loc[rows["available_date"].notna()].sort_values(["ticker", "available_date"])
    linked = _active_ibes_link(rows, ibes_link, "available_date")
    feature_rows = linked[["permno", "available_date", *FEATURES.keys()]].copy()
    feature_rows["permno"] = feature_rows["permno"].astype("int64")
    feature_rows = feature_rows.sort_values(["permno", "available_date"]).drop_duplicates(["permno", "available_date"], keep="last")

    panel_keys = panel[["permno", "formation_date"]].copy()
    panel_keys["formation_date"] = pd.to_datetime(panel_keys["formation_date"], errors="coerce")
    panel_keys["permno"] = pd.to_numeric(panel_keys["permno"], errors="coerce").astype("int64")
    out = _merge_asof_by_permno(panel_keys, feature_rows)
    metadata = [
        {
            "feature": feature,
            "source_table": "ibes_statsum_epsus_by_year / ibes_ptgsum_by_year via ibes_link",
            "availability_rule": "summary and target snapshots become available at statpers; surprises require actual actdats/anndats not after statpers",
            "feature_family": family,
            "leakage_rule": "IBES records are joined only when statpers/anndats/revdats/actdats availability is <= formation_date",
        }
        for feature, family in FEATURES.items()
    ]
    return FeatureBuildResult("ibes", out, metadata)
