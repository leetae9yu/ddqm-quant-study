#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Join yfinance prototype labels with date-level macro features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "sp500_yfinance_labels.csv"
DEFAULT_MACRO_WORKBOOK = PROJECT_ROOT / "expanded_macro_market_features.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "sp500_yfinance_macro_model_ready.csv"
PROTOTYPE_ONLY = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble prototype yfinance labels with macro features for LightGBM plumbing checks."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="CSV or Parquet yfinance label dataset.")
    parser.add_argument(
        "--macro-workbook",
        type=Path,
        default=DEFAULT_MACRO_WORKBOOK,
        help="Macro feature workbook with dates in the first column.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path ending in .csv or .parquet.")
    parser.add_argument(
        "--drop-missing-labels",
        action="store_true",
        help="Drop rows whose forward_return_* labels are missing after the horizon shift.",
    )
    return parser.parse_args()


def read_tabular_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Dataset must end in .csv or .parquet")


def read_macro_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Macro workbook not found: {path}")
    macro = pd.read_excel(path, index_col=0)
    macro.index = pd.to_datetime(macro.index, errors="coerce")
    if macro.index.isna().any():
        raise ValueError("Macro workbook contains invalid date index rows.")
    if macro.index.duplicated().any():
        raise ValueError("Macro workbook contains duplicate dates.")
    macro = macro.sort_index()
    macro.index.name = "date"
    macro = macro.add_prefix("macro__")
    return macro.reset_index()


def label_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("forward_return_")]


def normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"date", "ticker", "adj_close", "volume", "prototype_only"}
    missing_columns = sorted(required_columns.difference(labels.columns))
    if missing_columns:
        raise ValueError(f"Label dataset is missing required columns: {missing_columns}")
    if not label_columns(labels):
        raise ValueError("Label dataset has no forward_return_* columns.")

    normalized = labels.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if bool(normalized["date"].isna().any()):
        raise ValueError("Label dataset contains invalid dates.")
    duplicate_count = int(normalized.duplicated(["date", "ticker"]).sum())
    if duplicate_count:
        raise ValueError(f"Label dataset contains duplicate date/ticker rows: {duplicate_count}")
    normalized["prototype_only"] = True
    return normalized.sort_values(["ticker", "date"])


def assemble_dataset(labels: pd.DataFrame, macro: pd.DataFrame, macro_workbook: Path, drop_missing_labels: bool) -> pd.DataFrame:
    joined = labels.merge(macro, on="date", how="left", validate="many_to_one")
    macro_columns = [column for column in joined.columns if column.startswith("macro__")]
    missing_macro_rows = int(joined.loc[:, macro_columns].isna().any(axis=1).sum())
    if missing_macro_rows:
        raise ValueError(f"Rows without matching macro features: {missing_macro_rows}")

    if drop_missing_labels:
        labels_to_check = label_columns(joined)
        joined = joined.dropna(subset=labels_to_check).copy()

    joined["prototype_only"] = PROTOTYPE_ONLY
    joined["macro_source"] = "expanded_macro_market_features.xlsx"
    joined["macro_workbook"] = str(macro_workbook)
    joined["assembly_source"] = "yfinance_labels_plus_macro_features"
    joined["assembled_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return joined


def write_dataset(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError("Output must end in .csv or .parquet")
    print(f"Wrote {len(df):,} rows and {df.shape[1]} columns to {output}")


def main() -> None:
    args = parse_args()
    labels = normalize_labels(read_tabular_dataset(args.labels))
    macro = read_macro_workbook(args.macro_workbook)
    assembled = assemble_dataset(labels, macro, args.macro_workbook, args.drop_missing_labels)
    write_dataset(assembled, args.output)
    print("Warning: assembled dataset is prototype_only and not survivorship-bias-free research data.")


if __name__ == "__main__":
    main()
