#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportMissingTypeStubs=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Assemble canonical macro-factor model-ready datasets with as-of macro joins."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoquant_lab import schemas


DEFAULT_FACTOR_RETURNS = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "factor_long_short_returns.parquet"
DEFAULT_MACRO_WORKBOOK = PROJECT_ROOT / "expanded_macro_market_features.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "macro_factor_model_ready.parquet"
DATASET_FACTOR_RETURNS = "canonical factor_long_short_returns"
DATASET_MACRO_FACTOR = "canonical macro_factor_model_ready"
MACRO_SOURCE_NAME = "expanded_macro_market_features.xlsx"
PROTOTYPE_ONLY = True

FACTOR_METADATA_COLUMNS: tuple[str, ...] = (
    "factor_family",
    "lookback_days",
    "direction",
    "rank_method",
    "winsorization_method",
    "long_quantile",
    "short_quantile",
    "long_count",
    "short_count",
    "forward_return_start",
    "forward_return_end",
    "return_source",
    "source",
)
STAGING_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "asset_id",
        "ticker",
        "permno",
        "permco",
        "gvkey",
        "long_asset_id",
        "short_asset_id",
        "long_ticker",
        "short_ticker",
        "long_permno",
        "short_permno",
        "long_permnos",
        "short_permnos",
        "long_tickers",
        "short_tickers",
        "long_constituents",
        "short_constituents",
        "constituents",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble macro-factor model-ready CSV or Parquet datasets.")
    parser.add_argument("--factor-returns", type=Path, default=DEFAULT_FACTOR_RETURNS, help="Input factor long-short returns.")
    parser.add_argument(
        "--macro-workbook",
        type=Path,
        default=DEFAULT_MACRO_WORKBOOK,
        help="Excel macro feature workbook with dates in the first column.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path ending in .csv or .parquet.")
    return parser.parse_args()


def prepare_factor_returns(df: pd.DataFrame) -> pd.DataFrame:
    schemas.require_columns(df, schemas.FACTOR_LONG_SHORT_REQUIRED_COLUMNS, DATASET_FACTOR_RETURNS)
    schemas.require_prototype_only(df, DATASET_FACTOR_RETURNS)

    data = df.copy()
    data["formation_date"] = pd.to_datetime(data["formation_date"], errors="coerce")
    invalid_dates = int(data["formation_date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"{DATASET_FACTOR_RETURNS} contains invalid formation_date rows: {invalid_dates}")
    data["formation_date"] = data["formation_date"].dt.tz_localize(None)
    schemas.require_unique_key(data, schemas.FACTOR_LONG_SHORT_KEY_COLUMNS, DATASET_FACTOR_RETURNS)

    data["long_short_return"] = pd.to_numeric(data["long_short_return"], errors="coerce")
    non_finite_targets = int((~np.isfinite(data["long_short_return"].to_numpy(dtype=float))).sum())
    if non_finite_targets:
        raise ValueError(f"{DATASET_FACTOR_RETURNS} contains non-finite long_short_return rows: {non_finite_targets}")

    leaking_columns = sorted(STAGING_LEAKAGE_COLUMNS.intersection(data.columns))
    if leaking_columns:
        raise ValueError(f"{DATASET_FACTOR_RETURNS} contains asset-level staging columns: {leaking_columns}")
    return data.sort_values(["formation_date", "factor_name", "horizon_trading_days"]).reset_index(drop=True)


def read_macro_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Macro workbook not found: {path}")

    macro = pd.read_excel(path, index_col=0)
    macro.index = pd.to_datetime(macro.index, errors="coerce")
    if macro.index.isna().any():
        raise ValueError("Macro workbook contains invalid date index rows.")
    if macro.index.duplicated().any():
        raise ValueError("Macro workbook contains duplicate dates.")
    if macro.empty:
        raise ValueError("Macro workbook contains no rows.")

    macro = macro.sort_index()
    macro.index.name = "macro_asof_date"
    macro = macro.add_prefix(schemas.MACRO_FEATURE_PREFIX)
    macro_features = [column for column in macro.columns if column.startswith(schemas.MACRO_FEATURE_PREFIX)]
    if not macro_features:
        raise ValueError("Macro workbook contains no feature columns to prefix as macro__.")
    return macro.reset_index()


def assemble_macro_factor_dataset(factor_returns: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    labels = prepare_factor_returns(factor_returns)
    macro_data = macro.copy()
    macro_data["macro_asof_date"] = pd.to_datetime(macro_data["macro_asof_date"], errors="coerce").dt.tz_localize(None)

    joined = pd.merge_asof(
        labels.sort_values("formation_date"),
        macro_data.sort_values("macro_asof_date"),
        left_on="formation_date",
        right_on="macro_asof_date",
        direction="backward",
    )
    macro_columns = [column for column in joined.columns if column.startswith(schemas.MACRO_FEATURE_PREFIX)]
    missing_macro_rows = int(joined["macro_asof_date"].isna().sum())
    if missing_macro_rows:
        raise ValueError(f"Rows without as-of macro features: {missing_macro_rows}")
    rows_with_missing_macro_values = int(joined.loc[:, macro_columns].isna().any(axis=1).sum())
    if rows_with_missing_macro_values:
        raise ValueError(f"Rows with missing macro feature values after as-of join: {rows_with_missing_macro_values}")

    joined["target_long_short_return"] = joined["long_short_return"]
    joined["macro_source"] = MACRO_SOURCE_NAME
    joined["prototype_only"] = PROTOTYPE_ONLY

    present_metadata = [column for column in FACTOR_METADATA_COLUMNS if column in joined.columns]
    output_columns = [
        *schemas.MACRO_FACTOR_MODEL_REQUIRED_COLUMNS,
        *present_metadata,
        *macro_columns,
    ]
    output = joined.loc[:, output_columns].copy()
    leaking_columns = sorted(STAGING_LEAKAGE_COLUMNS.intersection(output.columns))
    if leaking_columns:
        raise ValueError(f"{DATASET_MACRO_FACTOR} would contain asset-level staging columns: {leaking_columns}")
    schemas.require_columns(output, schemas.MACRO_FACTOR_MODEL_REQUIRED_COLUMNS, DATASET_MACRO_FACTOR)
    schemas.require_unique_key(output, schemas.MACRO_FACTOR_MODEL_KEY_COLUMNS, DATASET_MACRO_FACTOR)
    return output.sort_values(list(schemas.MACRO_FACTOR_MODEL_KEY_COLUMNS)).reset_index(drop=True)


def print_build_summary(data: pd.DataFrame, output: Path) -> None:
    macro_columns = [column for column in data.columns if column.startswith(schemas.MACRO_FEATURE_PREFIX)]
    print(f"Wrote {len(data):,} rows and {data.shape[1]} columns to {output}")
    print(f"Macro feature columns: {len(macro_columns)}")
    print(f"Factors: {data['factor_name'].nunique() if not data.empty else 0}")
    print(f"Start formation date: {data['formation_date'].min().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print(f"End formation date: {data['formation_date'].max().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print(f"Start macro as-of date: {data['macro_asof_date'].min().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print(f"End macro as-of date: {data['macro_asof_date'].max().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print("Rows by factor:")
    print(data["factor_name"].value_counts().sort_index().to_string() if not data.empty else "none: 0")


def main() -> None:
    args = parse_args()
    factor_returns = schemas.read_dataset(args.factor_returns)
    macro = read_macro_workbook(args.macro_workbook)
    assembled = assemble_macro_factor_dataset(factor_returns, macro)
    schemas.write_dataset(assembled, args.output)
    print("Warning: macro-factor model dataset is prototype_only public-data plumbing, not research-grade data.")
    print_build_summary(assembled, args.output)


if __name__ == "__main__":
    main()
