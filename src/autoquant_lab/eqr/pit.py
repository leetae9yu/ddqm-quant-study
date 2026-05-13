"""Point-in-time entity mapping utilities for EQR datasets."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


CRSP_COMMON_SHRCD = frozenset({10, 11})
CRSP_MAJOR_EXCHCD = frozenset({1, 2, 3})
OPEN_END_DATE = pd.Timestamp.max.normalize()


IdentifierKind = Literal["permno", "gvkey", "ibes_ticker"]


def _to_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Invalid point-in-time date: {value!r}")
    return pd.Timestamp(timestamp)


def _normalized_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _with_interval_bounds(df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
    result = df.copy()
    result[start_col] = pd.to_datetime(result[start_col], errors="coerce")
    result[end_col] = pd.to_datetime(result[end_col], errors="coerce")
    result["_pit_start"] = result[start_col]
    result["_pit_end"] = result[end_col].fillna(OPEN_END_DATE)
    return result


def _active_interval_mask(df: pd.DataFrame, date: pd.Timestamp, start_col: str, end_col: str) -> pd.Series:
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce").fillna(OPEN_END_DATE)
    return start.notna() & (start <= date) & (date <= end)


def filter_crsp_common_stocks(names: pd.DataFrame) -> pd.DataFrame:
    """Return CRSP name rows for common stocks on NYSE/AMEX/NASDAQ only."""

    mask = names["shrcd"].isin(CRSP_COMMON_SHRCD) & names["exchcd"].isin(CRSP_MAJOR_EXCHCD)
    return names.loc[mask].copy()


def unmatched_rate(total_count: int, matched_count: int) -> dict[str, int | float]:
    """Summarize unmatched observations as counts and a rate."""

    unmatched_count = max(total_count - matched_count, 0)
    rate = float(unmatched_count / total_count) if total_count else 0.0
    return {"total_count": int(total_count), "matched_count": int(matched_count), "unmatched_count": int(unmatched_count), "unmatched_rate": rate}


def overlap_count(df: pd.DataFrame, group_cols: list[str], start_col: str, end_col: str) -> int:
    """Count overlapping validity-window pairs within each identifier group."""

    if df.empty:
        return 0

    intervals = _with_interval_bounds(df, start_col, end_col).dropna(subset=["_pit_start"])
    overlaps = 0
    for _, group in intervals.sort_values([*group_cols, "_pit_start", "_pit_end"]).groupby(group_cols, dropna=False):
        previous_end: pd.Timestamp | None = None
        for _, row in group.iterrows():
            start = row["_pit_start"]
            end = row["_pit_end"]
            if previous_end is not None and start <= previous_end:
                overlaps += 1
            previous_end = end if previous_end is None or end > previous_end else previous_end
    return int(overlaps)


def active_duplicate_count(df: pd.DataFrame, group_cols: list[str], start_col: str, end_col: str) -> int:
    """Count rows that create duplicate active mappings for an identifier."""

    return overlap_count(df, group_cols, start_col, end_col)


def diagnostics_summary(
    source: pd.DataFrame,
    resolved: pd.DataFrame,
    *,
    source_id_col: str,
    resolved_source_row_col: str,
    group_cols: list[str],
    start_col: str,
    end_col: str,
) -> dict[str, int | float]:
    """Report unmatched rate and duplicate active link diagnostics."""

    matched_count = resolved[resolved_source_row_col].nunique() if resolved_source_row_col in resolved else 0
    summary = unmatched_rate(int(source[source_id_col].notna().sum()), int(matched_count))
    summary["overlap_count"] = overlap_count(source, group_cols, start_col, end_col)
    summary["duplicate_active_link_count"] = active_duplicate_count(source, group_cols, start_col, end_col)
    return summary


@dataclass(frozen=True)
class CRSPNameResolver:
    """Resolve active CRSP name rows for a PERMNO at a point in time."""

    names: pd.DataFrame
    common_stock_only: bool = True

    def __post_init__(self) -> None:
        names = self.names.copy()
        names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
        names = _with_interval_bounds(names, "namedt", "nameenddt")
        if self.common_stock_only:
            names = filter_crsp_common_stocks(names)
        object.__setattr__(self, "names", names.sort_values(["permno", "namedt", "nameenddt"], na_position="last"))

    def active(self, permno: int, date: Any) -> pd.DataFrame:
        timestamp = _to_timestamp(date)
        mask = (self.names["permno"] == int(permno)) & _active_interval_mask(self.names, timestamp, "namedt", "nameenddt")
        return self.names.loc[mask].copy()

    def resolve(self, permno: int, date: Any) -> pd.Series | None:
        active = self.active(permno, date)
        if active.empty:
            return None
        return active.iloc[0]

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "row_count": int(len(self.names)),
            "overlap_count": overlap_count(self.names, ["permno"], "namedt", "nameenddt"),
            "duplicate_active_link_count": active_duplicate_count(self.names, ["permno"], "namedt", "nameenddt"),
        }


@dataclass(frozen=True)
class CCMLinkResolver:
    """Resolve GVKEY to active PERMNO using CCM link validity windows."""

    links: pd.DataFrame

    def __post_init__(self) -> None:
        links = self.links.copy()
        links["gvkey"] = links["gvkey"].astype("string")
        links["lpermno"] = pd.to_numeric(links["lpermno"], errors="coerce").astype("Int64")
        links = _with_interval_bounds(links, "linkdt", "linkenddt")
        sort_cols = ["gvkey", "linkdt", "linkenddt"]
        if "linkprim" in links:
            links["_linkprim_rank"] = links["linkprim"].map({"P": 0, "C": 1}).fillna(2)
            sort_cols.append("_linkprim_rank")
        if "usedflag" in links:
            links["_usedflag_rank"] = -pd.to_numeric(links["usedflag"], errors="coerce").fillna(-999)
            sort_cols.append("_usedflag_rank")
        object.__setattr__(self, "links", links.sort_values(sort_cols, na_position="last"))

    def active(self, gvkey: str, date: Any) -> pd.DataFrame:
        timestamp = _to_timestamp(date)
        mask = (self.links["gvkey"] == str(gvkey)) & self.links["lpermno"].notna() & _active_interval_mask(self.links, timestamp, "linkdt", "linkenddt")
        return self.links.loc[mask].copy()

    def resolve(self, gvkey: str, date: Any) -> int | None:
        active = self.active(gvkey, date)
        if active.empty:
            return None
        return int(active.iloc[0]["lpermno"])

    def diagnostics(self) -> dict[str, int | float]:
        valid = self.links.loc[self.links["lpermno"].notna()].copy()
        return {
            "row_count": int(len(self.links)),
            "linkable_row_count": int(len(valid)),
            "overlap_count": overlap_count(valid, ["gvkey"], "linkdt", "linkenddt"),
            "duplicate_active_link_count": active_duplicate_count(valid, ["gvkey"], "linkdt", "linkenddt"),
        }


@dataclass(frozen=True)
class IBESLinkResolver:
    """Resolve IBES ticker to active PERMNO using score-filtered date windows."""

    links: pd.DataFrame
    max_score: int = 1

    def __post_init__(self) -> None:
        links = self.links.copy()
        links["ticker"] = links["ticker"].map(_normalized_text).astype("string")
        links["permno"] = pd.to_numeric(links["permno"], errors="coerce").astype("Int64")
        if "score" in links:
            links["score"] = pd.to_numeric(links["score"], errors="coerce")
            links = links.loc[links["score"].le(self.max_score)].copy()
        links = _with_interval_bounds(links, "sdate", "edate")
        sort_cols = ["ticker", "sdate", "edate"]
        if "score" in links:
            sort_cols.append("score")
        object.__setattr__(self, "links", links.sort_values(sort_cols, na_position="last"))

    def active(self, ticker: str, date: Any) -> pd.DataFrame:
        timestamp = _to_timestamp(date)
        normalized = _normalized_text(ticker)
        mask = (self.links["ticker"] == normalized) & self.links["permno"].notna() & _active_interval_mask(self.links, timestamp, "sdate", "edate")
        return self.links.loc[mask].copy()

    def resolve(self, ticker: str, date: Any) -> int | None:
        active = self.active(ticker, date)
        if active.empty:
            return None
        return int(active.iloc[0]["permno"])

    def diagnostics(self) -> dict[str, int | float]:
        valid = self.links.loc[self.links["permno"].notna()].copy()
        return {
            "row_count": int(len(self.links)),
            "linkable_row_count": int(len(valid)),
            "overlap_count": overlap_count(valid, ["ticker"], "sdate", "edate"),
            "duplicate_active_link_count": active_duplicate_count(valid, ["ticker"], "sdate", "edate"),
        }


@dataclass(frozen=True)
class EntityMapper:
    """Unified point-in-time identifier-to-PERMNO mapper."""

    crsp_names: CRSPNameResolver
    ccm_links: CCMLinkResolver
    ibes_links: IBESLinkResolver

    def resolve_permno(self, identifier: str | int, date: Any, kind: IdentifierKind) -> int | None:
        if kind == "permno":
            permno = int(identifier)
            return permno if self.crsp_names.resolve(permno, date) is not None else None
        if kind == "gvkey":
            permno = self.ccm_links.resolve(str(identifier), date)
            return permno if permno is not None and self.crsp_names.resolve(permno, date) is not None else None
        if kind == "ibes_ticker":
            permno = self.ibes_links.resolve(str(identifier), date)
            return permno if permno is not None and self.crsp_names.resolve(permno, date) is not None else None
        raise ValueError(f"Unsupported identifier kind: {kind!r}")


__all__ = [
    "CCMLinkResolver",
    "CRSPNameResolver",
    "EntityMapper",
    "IBESLinkResolver",
    "active_duplicate_count",
    "diagnostics_summary",
    "filter_crsp_common_stocks",
    "overlap_count",
    "unmatched_rate",
]
