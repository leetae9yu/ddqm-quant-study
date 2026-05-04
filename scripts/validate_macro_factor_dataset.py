#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportMissingTypeStubs=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Validate canonical macro-factor model-ready datasets."""

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


DEFAULT_INPUT = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "macro_factor_model_ready.parquet"
DATASET_NAME = "canonical macro_factor_model_ready"
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
    parser = argparse.ArgumentParser(description="Validate a canonical macro_factor_model_ready CSV or Parquet artifact.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_INPUT, help="CSV or Parquet dataset path.")
    return parser.parse_args()


def _require_nonblank(df: pd.DataFrame, column: str) -> None:
    blank_mask = df[column].isna() | df[column].astype(str).str.strip().eq("")
    blank_count = int(blank_mask.sum())
    if blank_count:
        raise ValueError(f"{DATASET_NAME} column {column!r} contains blank values: {blank_count}")


def validate_macro_factor_dataset(df: pd.DataFrame) -> pd.DataFrame:
    schemas.require_columns(df, schemas.MACRO_FACTOR_MODEL_REQUIRED_COLUMNS, DATASET_NAME)
    schemas.require_prototype_only(df, DATASET_NAME)

    data = df.copy()
    if data.empty:
        raise ValueError(f"{DATASET_NAME} must contain at least one row")

    leaking_columns = sorted(STAGING_LEAKAGE_COLUMNS.intersection(data.columns))
    if leaking_columns:
        raise ValueError(f"{DATASET_NAME} contains asset-level staging columns: {leaking_columns}")

    data["formation_date"] = pd.to_datetime(data["formation_date"], errors="coerce")
    data["macro_asof_date"] = pd.to_datetime(data["macro_asof_date"], errors="coerce")
    invalid_formation_dates = int(data["formation_date"].isna().sum())
    invalid_macro_dates = int(data["macro_asof_date"].isna().sum())
    if invalid_formation_dates:
        raise ValueError(f"{DATASET_NAME} contains invalid formation_date rows: {invalid_formation_dates}")
    if invalid_macro_dates:
        raise ValueError(f"{DATASET_NAME} contains invalid/missing macro_asof_date rows: {invalid_macro_dates}")
    data["formation_date"] = data["formation_date"].dt.tz_localize(None)
    data["macro_asof_date"] = data["macro_asof_date"].dt.tz_localize(None)

    future_macro_rows = int((data["macro_asof_date"] > data["formation_date"]).sum())
    if future_macro_rows:
        raise ValueError(f"{DATASET_NAME} has macro_asof_date > formation_date rows: {future_macro_rows}")

    schemas.require_unique_key(data, schemas.MACRO_FACTOR_MODEL_KEY_COLUMNS, DATASET_NAME)

    macro_columns = [column for column in data.columns if column.startswith(schemas.MACRO_FEATURE_PREFIX)]
    if not macro_columns:
        raise ValueError(f"{DATASET_NAME} has no {schemas.MACRO_FEATURE_PREFIX} feature columns")
    rows_with_missing_macro_values = int(data.loc[:, macro_columns].isna().any(axis=1).sum())
    if rows_with_missing_macro_values:
        raise ValueError(f"{DATASET_NAME} contains rows with missing macro feature values: {rows_with_missing_macro_values}")

    for column in ("factor_name", "label_source", "macro_source"):
        _require_nonblank(data, column)

    data["target_long_short_return"] = pd.to_numeric(data["target_long_short_return"], errors="coerce")
    non_finite_targets = int((~np.isfinite(data["target_long_short_return"].to_numpy(dtype=float))).sum())
    if non_finite_targets:
        raise ValueError(f"{DATASET_NAME} contains non-finite target_long_short_return rows: {non_finite_targets}")
    if "long_short_return" in data.columns:
        source_target = pd.to_numeric(data["long_short_return"], errors="coerce")
        target_values = data["target_long_short_return"].to_numpy(dtype=float)
        source_values = source_target.to_numpy(dtype=float)
        target_mismatches = int((~np.isclose(target_values, source_values, rtol=1e-12, atol=1e-15)).sum())
        if target_mismatches:
            raise ValueError(f"{DATASET_NAME} target_long_short_return != long_short_return rows: {target_mismatches}")

    return data.sort_values(list(schemas.MACRO_FACTOR_MODEL_KEY_COLUMNS)).reset_index(drop=True)


def print_validation_summary(data: pd.DataFrame) -> None:
    macro_columns = [column for column in data.columns if column.startswith(schemas.MACRO_FEATURE_PREFIX)]
    duplicate_rows = int(data.duplicated(list(schemas.MACRO_FACTOR_MODEL_KEY_COLUMNS)).sum())
    prototype_rows = schemas.prototype_true_count(data["prototype_only"])
    target_stats = data["target_long_short_return"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])

    print(f"Rows: {len(data)}")
    print(f"Columns: {data.shape[1]}")
    print(f"Factors: {data['factor_name'].nunique()}")
    print(f"Start formation date: {data['formation_date'].min().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print(f"End formation date: {data['formation_date'].max().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print(f"Duplicate (formation_date, factor_name, horizon_trading_days) rows: {duplicate_rows}")
    print(f"Macro feature columns: {len(macro_columns)}")
    print(f"Macro feature missing values: {int(data.loc[:, macro_columns].isna().sum().sum())}")
    print(f"Prototype-only rows: {prototype_rows}")
    print("Rows by factor:")
    print(data["factor_name"].value_counts().sort_index().to_string())
    print("Rows by macro as-of date:")
    print(data["macro_asof_date"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_string())
    print("Target long-short return stats:")
    print(target_stats.to_string())
    print("Provenance counts:")
    print(data[["label_source", "macro_source"]].value_counts().sort_index().to_string())


def main() -> None:
    args = parse_args()
    df = schemas.read_dataset(args.dataset)
    validated = validate_macro_factor_dataset(df)
    print_validation_summary(validated)


if __name__ == "__main__":
    main()
