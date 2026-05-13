"""Compustat quarterly PIT valuation, quality, and growth features."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import numpy as np
import pandas as pd

from .result import FeatureBuildResult


FEATURES = {
    "compustat__pb": "valuation",
    "compustat__pe_proxy": "valuation",
    "compustat__roe": "quality",
    "compustat__debt_to_assets": "quality",
    "compustat__liabilities_to_assets": "quality",
    "compustat__revenue_yoy": "growth",
    "compustat__net_income_yoy": "growth",
    "compustat__available_lag_days": "availability",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator.abs() > 1e-12)
    return numerator / denom


def _link_compustat_to_permno(fund: pd.DataFrame, ccm_link: pd.DataFrame) -> pd.DataFrame:
    links = ccm_link.copy()
    links["gvkey"] = links["gvkey"].astype(str).str.zfill(6)
    links["permno"] = pd.to_numeric(links.get("lpermno"), errors="coerce")
    links["linkdt"] = pd.to_datetime(links["linkdt"], errors="coerce")
    links["linkenddt"] = pd.to_datetime(links["linkenddt"], errors="coerce").fillna(pd.Timestamp.max.normalize())
    if "usedflag" in links.columns:
        links = links.loc[pd.to_numeric(links["usedflag"], errors="coerce").fillna(1).astype(int) == 1]
    if "linktype" in links.columns:
        links = links.loc[links["linktype"].astype(str).isin(["LC", "LU", "LS"])]
    if "linkprim" in links.columns:
        links["_link_rank"] = links["linkprim"].astype(str).map({"P": 0, "C": 1}).fillna(2)
    else:
        links["_link_rank"] = 0

    merged = fund.merge(links[["gvkey", "permno", "linkdt", "linkenddt", "_link_rank"]], on="gvkey", how="inner")
    active = merged.loc[
        merged["permno"].notna()
        & merged["datadate"].notna()
        & (merged["linkdt"] <= merged["datadate"])
        & (merged["datadate"] <= merged["linkenddt"])
    ].copy()
    return active.sort_values(["gvkey", "datadate", "_link_rank"]).drop_duplicates(["gvkey", "datadate", "permno"], keep="first")


def _merge_asof_by_permno(panel_keys: pd.DataFrame, feature_rows: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in feature_rows.columns if col.startswith("compustat__")]
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


def build_compustat_features(
    panel: pd.DataFrame,
    comp_fundq: pd.DataFrame | None = None,
    ccm_link: pd.DataFrame | None = None,
    **_: object,
) -> FeatureBuildResult:
    """Build Compustat features using rdq availability or a 90-day fallback."""

    if comp_fundq is None or ccm_link is None or comp_fundq.empty or ccm_link.empty:
        return FeatureBuildResult("compustat", panel[["permno", "formation_date"]].copy(), [])

    fund = comp_fundq.copy()
    fund["gvkey"] = fund["gvkey"].astype(str).str.zfill(6)
    fund["datadate"] = pd.to_datetime(fund["datadate"], errors="coerce")
    fund["rdq"] = pd.to_datetime(fund["rdq"], errors="coerce")
    numeric_cols = ["oiadpq", "niq", "ceqq", "revtq", "atq", "ltq", "dlttq", "dlcq", "cshoq", "prccq", "epspxq", "saleq"]
    for col in numeric_cols:
        if col in fund.columns:
            fund[col] = pd.to_numeric(fund[col], errors="coerce")
        else:
            fund[col] = np.nan

    fund["available_date"] = fund["rdq"].fillna(fund["datadate"] + pd.Timedelta(days=90))
    fund["compustat__available_lag_days"] = (fund["available_date"] - fund["datadate"]).dt.days.astype(float)
    book_equity = fund["ceqq"]
    market_equity = fund["prccq"] * fund["cshoq"]
    fund["compustat__pb"] = _safe_divide(market_equity, book_equity)
    fund["compustat__pe_proxy"] = _safe_divide(fund["prccq"], fund["epspxq"])
    fund["compustat__roe"] = _safe_divide(fund["niq"], book_equity)
    total_debt = fund["dlttq"].fillna(0.0) + fund["dlcq"].fillna(0.0)
    fund["compustat__debt_to_assets"] = _safe_divide(total_debt, fund["atq"])
    fund["compustat__liabilities_to_assets"] = _safe_divide(fund["ltq"], fund["atq"])
    fund = fund.sort_values(["gvkey", "datadate"])
    prior_revenue = fund.groupby("gvkey")["revtq"].shift(4)
    prior_income = fund.groupby("gvkey")["niq"].shift(4)
    fund["compustat__revenue_yoy"] = _safe_divide(fund["revtq"] - prior_revenue, prior_revenue.abs())
    fund["compustat__net_income_yoy"] = _safe_divide(fund["niq"] - prior_income, prior_income.abs())

    linked = _link_compustat_to_permno(fund, ccm_link)
    feature_cols = list(FEATURES)
    feature_rows = linked[["permno", "available_date", *feature_cols]].copy()
    feature_rows["permno"] = feature_rows["permno"].astype("int64")
    feature_rows = feature_rows.loc[feature_rows["available_date"].notna()].sort_values(["permno", "available_date"])
    feature_rows = feature_rows.drop_duplicates(["permno", "available_date"], keep="last")

    panel_keys = panel[["permno", "formation_date"]].copy()
    panel_keys["formation_date"] = pd.to_datetime(panel_keys["formation_date"], errors="coerce")
    panel_keys["permno"] = pd.to_numeric(panel_keys["permno"], errors="coerce").astype("int64")
    out = _merge_asof_by_permno(panel_keys, feature_rows)
    metadata = [
        {
            "feature": feature,
            "source_table": "comp_fundq via ccm_link",
            "availability_rule": "rdq when present, otherwise datadate plus 90 calendar days",
            "feature_family": family,
            "leakage_rule": "quarterly record is joined only when availability date is <= formation_date",
        }
        for feature, family in FEATURES.items()
    ]
    return FeatureBuildResult("compustat", out, metadata)
