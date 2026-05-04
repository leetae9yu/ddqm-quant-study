#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Validate assembled yfinance plus macro prototype datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "sp500_yfinance_macro_model_ready.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an assembled yfinance plus macro prototype dataset.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_INPUT, help="CSV or Parquet dataset path.")
    parser.add_argument("--label-column", default=None, help="Label column to validate. Defaults to first forward_return_* column.")
    parser.add_argument("--top-n", type=int, default=10, help="Rows to print for top missing feature counts.")
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Dataset must end in .csv or .parquet")


def resolve_label_column(df: pd.DataFrame, label_column: str | None) -> str:
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Label column not found: {label_column}")
        return label_column
    candidates = [column for column in df.columns if column.startswith("forward_return_")]
    if not candidates:
        raise ValueError("No forward_return_* label column found.")
    return candidates[0]


def validate_required_schema(df: pd.DataFrame, label_column: str) -> None:
    required = {
        "date",
        "ticker",
        "adj_close",
        "volume",
        label_column,
        "prototype_only",
        "macro_source",
        "macro_workbook",
        "assembly_source",
        "assembled_at_utc",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    macro_columns = [column for column in df.columns if column.startswith("macro__")]
    if not macro_columns:
        raise ValueError("Dataset has no macro__ feature columns.")


def prototype_true_count(series: pd.Series) -> int:
    if series.dtype == bool:
        return int(series.sum())
    return int(series.astype(str).str.lower().eq("true").sum())


def print_validation_stats(df: pd.DataFrame, label_column: str, top_n: int) -> None:
    validate_required_schema(df, label_column)
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if bool(data["date"].isna().any()):
        raise ValueError("Dataset contains invalid dates.")

    duplicate_count = int(data.duplicated(["date", "ticker"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate date/ticker rows: {duplicate_count}")

    macro_columns = [column for column in data.columns if column.startswith("macro__")]
    macro_missing = data.loc[:, macro_columns].isna().sum().sort_values(ascending=False)
    missing_macro_columns = macro_missing[macro_missing > 0]
    finite_numeric = data.loc[:, macro_columns + ["adj_close", "volume", label_column]].select_dtypes(include="number")

    print(f"Rows: {len(data)}")
    print(f"Columns: {data.shape[1]}")
    print(f"Tickers: {data['ticker'].nunique()}")
    print(f"Start date: {data['date'].min().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print(f"End date: {data['date'].max().strftime('%Y-%m-%d') if not data.empty else 'n/a'}")
    print(f"Duplicate date/ticker rows: {duplicate_count}")
    print(f"Macro feature columns: {len(macro_columns)}")
    print(f"Macro feature missing values: {int(macro_missing.sum())}")
    if not missing_macro_columns.empty:
        print("Macro missing values by column:")
        print(missing_macro_columns.head(top_n).to_string())
    print(f"Prototype-only rows: {prototype_true_count(data['prototype_only'])}")
    print("Label stats:")
    print(data[label_column].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
    print("Provenance counts:")
    print(data[["macro_source", "assembly_source"]].value_counts().to_string())
    print(f"Numeric cells checked for finite values: {finite_numeric.size}")


def main() -> None:
    args = parse_args()
    df = read_dataset(args.dataset)
    label_column = resolve_label_column(df, args.label_column)
    print_validation_stats(df, label_column, args.top_n)


if __name__ == "__main__":
    main()
