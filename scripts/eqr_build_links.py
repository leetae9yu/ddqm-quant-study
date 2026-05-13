#!/usr/bin/env python3
"""Build point-in-time CCM/IBES to CRSP PERMNO link tables."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from autoquant_lab.eqr.path_resolver import resolve_data_paths  # noqa: E402
from autoquant_lab.eqr.pit import (  # noqa: E402
    OPEN_END_DATE,
    active_duplicate_count,
    filter_crsp_common_stocks,
    overlap_count,
    unmatched_rate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "prepared" / "links"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time EQR link tables from local offline Parquet files.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Local raw data directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prepared link outputs.")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _prepare_crsp_names(names: pd.DataFrame) -> pd.DataFrame:
    columns = ["permno", "comnam", "ncusip", "ticker", "shrcd", "exchcd", "namedt", "nameenddt"]
    names = names.loc[:, columns].copy()
    names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameenddt"] = pd.to_datetime(names["nameenddt"], errors="coerce")
    names = filter_crsp_common_stocks(names)
    names = names.loc[names["permno"].notna() & names["namedt"].notna()].copy()
    names["_crsp_end"] = names["nameenddt"].fillna(OPEN_END_DATE)
    return names


def _interval_join_to_crsp(
    links: pd.DataFrame,
    names: pd.DataFrame,
    *,
    permno_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    link_frame = links.copy()
    link_frame["_source_row"] = link_frame.index.astype("int64")
    link_frame[permno_col] = pd.to_numeric(link_frame[permno_col], errors="coerce").astype("Int64")
    link_frame[start_col] = pd.to_datetime(link_frame[start_col], errors="coerce")
    link_frame[end_col] = pd.to_datetime(link_frame[end_col], errors="coerce")
    link_frame["_link_end"] = link_frame[end_col].fillna(OPEN_END_DATE)
    link_frame = link_frame.loc[link_frame[permno_col].notna() & link_frame[start_col].notna()].copy()

    merged = link_frame.merge(names, left_on=permno_col, right_on="permno", how="inner", suffixes=("", "_crsp"))
    if merged.empty:
        return merged

    active = (merged[start_col] <= merged["_crsp_end"]) & (merged["namedt"] <= merged["_link_end"])
    merged = merged.loc[active].copy()
    if merged.empty:
        return merged

    merged["valid_start"] = merged[[start_col, "namedt"]].max(axis=1)
    merged["valid_end"] = merged[["_link_end", "_crsp_end"]].min(axis=1)
    both_open = merged[end_col].isna() & merged["nameenddt"].isna()
    merged.loc[both_open, "valid_end"] = pd.NaT
    merged = merged.drop(columns=["_link_end", "_crsp_end"])
    return merged.sort_values(["valid_start", "permno", "_source_row"]).reset_index(drop=True)


def build_ccm_links(ccm: pd.DataFrame, crsp_names: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    columns = ["gvkey", "linkprim", "liid", "linktype", "lpermno", "lpermco", "usedflag", "linkdt", "linkenddt"]
    ccm = ccm.loc[:, columns].copy()
    resolved = _interval_join_to_crsp(ccm, crsp_names, permno_col="lpermno", start_col="linkdt", end_col="linkenddt")
    if not resolved.empty:
        resolved["permno"] = resolved["lpermno"].astype("Int64")
        resolved = resolved[
            [
                "gvkey",
                "permno",
                "lpermco",
                "linkprim",
                "liid",
                "linktype",
                "usedflag",
                "linkdt",
                "linkenddt",
                "valid_start",
                "valid_end",
                "comnam",
                "ncusip",
                "ticker",
                "shrcd",
                "exchcd",
                "namedt",
                "nameenddt",
                "_source_row",
            ]
        ]
    linkable = ccm.loc[ccm["lpermno"].notna()].copy()
    diagnostics = unmatched_rate(int(len(linkable)), int(resolved["_source_row"].nunique() if not resolved.empty else 0))
    diagnostics["overlap_count"] = overlap_count(linkable, ["gvkey"], "linkdt", "linkenddt")
    diagnostics["duplicate_active_link_count"] = active_duplicate_count(linkable, ["gvkey"], "linkdt", "linkenddt")
    diagnostics["resolved_row_count"] = int(len(resolved))
    return resolved, diagnostics


def build_ibes_links(ibes: pd.DataFrame, crsp_names: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    columns = ["ticker", "permno", "ncusip", "sdate", "edate", "score"]
    ibes = ibes.loc[:, columns].copy()
    ibes["score"] = pd.to_numeric(ibes["score"], errors="coerce")
    eligible = ibes.loc[ibes["score"].le(1)].copy()
    resolved = _interval_join_to_crsp(eligible, crsp_names, permno_col="permno", start_col="sdate", end_col="edate")
    if not resolved.empty:
        resolved = resolved.rename(columns={"ticker": "ibes_ticker", "ncusip": "ibes_ncusip", "ticker_crsp": "crsp_ticker"})
        resolved = resolved[
            [
                "ibes_ticker",
                "permno",
                "ibes_ncusip",
                "score",
                "sdate",
                "edate",
                "valid_start",
                "valid_end",
                "comnam",
                "crsp_ticker",
                "shrcd",
                "exchcd",
                "namedt",
                "nameenddt",
                "_source_row",
            ]
        ]
    linkable = eligible.loc[eligible["permno"].notna()].copy()
    diagnostics = unmatched_rate(int(len(linkable)), int(resolved["_source_row"].nunique() if not resolved.empty else 0))
    diagnostics["overlap_count"] = overlap_count(linkable, ["ticker"], "sdate", "edate")
    diagnostics["duplicate_active_link_count"] = active_duplicate_count(linkable, ["ticker"], "sdate", "edate")
    diagnostics["resolved_row_count"] = int(len(resolved))
    diagnostics["score_filtered_out_count"] = int(len(ibes) - len(eligible))
    return resolved, diagnostics


def main() -> int:
    args = parse_args()
    resolved_paths = resolve_data_paths(args.data_dir, include_date_ranges=False)
    required = {"crsp_names", "ccm_link", "ibes_link"}
    missing = sorted(required - set(resolved_paths))
    if missing:
        raise FileNotFoundError(f"Missing required raw artifacts: {missing}")

    crsp_names = _prepare_crsp_names(pd.read_parquet(resolved_paths["crsp_names"].path))
    ccm = pd.read_parquet(resolved_paths["ccm_link"].path)
    ibes = pd.read_parquet(resolved_paths["ibes_link"].path)

    ccm_links, ccm_diagnostics = build_ccm_links(ccm, crsp_names)
    ibes_links, ibes_diagnostics = build_ibes_links(ibes, crsp_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ccm_links.to_parquet(args.output_dir / "ccm_permno_links.parquet", index=False)
    ibes_links.to_parquet(args.output_dir / "ibes_permno_links.parquet", index=False)

    diagnostics = {
        "inputs": {name: resolved_paths[name].to_dict() for name in sorted(required)},
        "crsp_filter": {"shrcd": [10, 11], "exchcd": [1, 2, 3], "filtered_name_rows": int(len(crsp_names))},
        "ccm": ccm_diagnostics,
        "ibes": ibes_diagnostics,
    }
    diagnostics_path = args.output_dir / "link_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "diagnostics": str(diagnostics_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
