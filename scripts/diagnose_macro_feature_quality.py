#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Print deeper quality diagnostics for the expanded macro market feature workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = PROJECT_ROOT / "expanded_macro_market_features.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print quality diagnostics for a macro feature workbook.")
    parser.add_argument(
        "workbook",
        nargs="?",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help="Workbook path to diagnose.",
    )
    parser.add_argument(
        "--near-constant-threshold",
        type=float,
        default=0.999,
        help="Top value frequency at or above this level is flagged as near-constant.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.98,
        help="Absolute Pearson correlation at or above this level is flagged.",
    )
    parser.add_argument(
        "--outlier-z-threshold",
        type=float,
        default=8.0,
        help="Robust MAD z-score absolute value at or above this level is flagged.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Maximum rows to print per diagnostic section.")
    return parser.parse_args()


def load_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    return pd.read_excel(path, index_col=0)


def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized.sort_index()
    return normalized


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number")


def print_section(title: str) -> None:
    print(f"\n## {title}")


def print_frame_or_message(frame: pd.DataFrame, message: str) -> None:
    if frame.empty:
        print(message)
    else:
        print(frame.to_string(index=False))


def print_series_or_message(series: pd.Series, message: str) -> None:
    if series.empty:
        print(message)
    else:
        print(series.to_string())


def date_coverage(df: pd.DataFrame, top_n: int) -> None:
    print_section("Date coverage, gaps, and duplicates")
    valid_dates = pd.Series(df.index).dropna().sort_values()
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Invalid date index rows: {int(pd.Series(df.index).isna().sum())}")
    print(f"Start date: {valid_dates.min().strftime('%Y-%m-%d') if not valid_dates.empty else 'n/a'}")
    print(f"End date: {valid_dates.max().strftime('%Y-%m-%d') if not valid_dates.empty else 'n/a'}")

    duplicate_dates = pd.Series(df.index[df.index.duplicated(keep=False)]).value_counts().sort_index()
    print(f"Duplicate date rows: {int(duplicate_dates.sum())}")
    if not duplicate_dates.empty:
        print("Duplicate dates:")
        print(duplicate_dates.head(top_n).to_string())

    day_gaps = valid_dates.diff().dropna()
    large_gaps = day_gaps[day_gaps > pd.Timedelta(days=1)].sort_values(ascending=False).head(top_n)
    print(f"Calendar gaps over 1 day: {large_gaps.shape[0]}")
    if not large_gaps.empty:
        gap_rows = pd.DataFrame(
            {
                "gap_start": [valid_dates.loc[index - 1].strftime("%Y-%m-%d") for index in large_gaps.index],
                "gap_end": [valid_dates.loc[index].strftime("%Y-%m-%d") for index in large_gaps.index],
                "gap_days": large_gaps.dt.days.to_list(),
            }
        )
        print(gap_rows.to_string(index=False))


def missing_values(df: pd.DataFrame, top_n: int) -> None:
    print_section("Missing values")
    null_counts = df.isna().sum().sort_values(ascending=False)
    missing_columns = null_counts[null_counts > 0]
    print(f"Total missing values: {int(null_counts.sum())}")
    print(f"Columns with missing values: {missing_columns.shape[0]}")
    print_series_or_message(missing_columns.head(top_n), "No missing values found.")


def constant_columns(numeric: pd.DataFrame, near_constant_threshold: float, top_n: int) -> None:
    print_section("Constant and near-constant numeric columns")
    if numeric.empty:
        print("No numeric columns found.")
        return

    distinct_counts = numeric.nunique(dropna=False)
    constants = distinct_counts[distinct_counts <= 1].sort_index()
    print(f"Constant numeric columns: {constants.shape[0]}")
    print_series_or_message(constants.head(top_n), "No constant numeric columns found.")

    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        value_frequencies = numeric[column].value_counts(dropna=False, normalize=True)
        if value_frequencies.empty:
            continue
        top_frequency = float(value_frequencies.iloc[0])
        if distinct_counts[column] > 1 and top_frequency >= near_constant_threshold:
            rows.append(
                {
                    "column": column,
                    "top_value_frequency": top_frequency,
                    "distinct_values": int(distinct_counts[column]),
                }
            )

    near_constants = pd.DataFrame(rows).sort_values("top_value_frequency", ascending=False) if rows else pd.DataFrame()
    print(f"Near-constant numeric columns: {near_constants.shape[0]}")
    print_frame_or_message(near_constants.head(top_n), "No near-constant numeric columns found.")


def exact_duplicate_columns(numeric: pd.DataFrame, top_n: int) -> None:
    print_section("Exact duplicate numeric columns")
    rows: list[dict[str, str]] = []
    seen: dict[int, str] = {}
    for column in numeric.columns:
        fingerprint = int(pd.util.hash_pandas_object(numeric[column], index=False).sum())
        original = seen.get(fingerprint)
        if original is None:
            seen[fingerprint] = column
        elif numeric[column].equals(numeric[original]):
            rows.append({"column": column, "duplicate_of": original})

    duplicates = pd.DataFrame(rows)
    print(f"Exact duplicate numeric columns: {duplicates.shape[0]}")
    print_frame_or_message(duplicates.head(top_n), "No exact duplicate numeric columns found.")


def high_correlations(numeric: pd.DataFrame, corr_threshold: float, top_n: int) -> None:
    print_section("High absolute pairwise correlations")
    if numeric.shape[1] < 2:
        print("Need at least two numeric columns for pairwise correlations.")
        return

    corr = numeric.corr().abs()
    rows: list[dict[str, object]] = []
    columns = list(corr.columns)
    for left_index, left_column in enumerate(columns):
        for right_column in columns[left_index + 1 :]:
            value = corr.loc[left_column, right_column]
            if pd.notna(value) and float(value) >= corr_threshold:
                rows.append({"column_a": left_column, "column_b": right_column, "abs_corr": float(value)})

    flagged = pd.DataFrame(rows).sort_values("abs_corr", ascending=False) if rows else pd.DataFrame()
    print(f"Pairs at or above threshold: {flagged.shape[0]}")
    print_frame_or_message(flagged.head(top_n), "No high-correlation pairs found.")


def robust_mad_outliers(numeric: pd.DataFrame, outlier_z_threshold: float, top_n: int) -> None:
    print_section("Robust MAD z-score outliers")
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        median = values.median()
        mad = (values - median).abs().median()
        if pd.isna(mad) or mad == 0:
            continue
        robust_z = 0.6745 * (values - median) / mad
        flagged = robust_z[robust_z.abs() >= outlier_z_threshold]
        for index, z_score in flagged.abs().sort_values(ascending=False).head(top_n).items():
            rows.append(
                {
                    "date": index.strftime("%Y-%m-%d") if hasattr(index, "strftime") else str(index),
                    "column": column,
                    "value": values.loc[index],
                    "robust_abs_z": float(z_score),
                }
            )

    outliers = pd.DataFrame(rows).sort_values("robust_abs_z", ascending=False) if rows else pd.DataFrame()
    print(f"Outlier cells at or above threshold: {outliers.shape[0]}")
    print_frame_or_message(outliers.head(top_n), "No robust MAD z-score outliers found.")


def biggest_one_day_changes(numeric: pd.DataFrame, top_n: int) -> None:
    print_section("Biggest one-day absolute changes")
    rows: list[dict[str, object]] = []
    changes = numeric.diff().abs()
    for column_index, column in enumerate(changes.columns):
        largest_changes = changes.iloc[:, column_index].dropna().nlargest(top_n)
        for date, abs_change in largest_changes.items():
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                    "column": column,
                    "abs_change": float(abs_change),
                }
            )

    if not rows:
        print("No one-day changes found.")
        return

    largest = pd.DataFrame(rows).sort_values("abs_change", ascending=False).head(top_n)
    print(largest.to_string(index=False))


def print_quality_diagnostics(
    df: pd.DataFrame,
    near_constant_threshold: float,
    corr_threshold: float,
    outlier_z_threshold: float,
    top_n: int,
) -> None:
    if top_n < 1:
        raise ValueError("top-n must be at least 1")
    if not 0 < near_constant_threshold <= 1:
        raise ValueError("near-constant-threshold must be in (0, 1]")
    if not 0 < corr_threshold <= 1:
        raise ValueError("corr-threshold must be in (0, 1]")
    if outlier_z_threshold <= 0:
        raise ValueError("outlier-z-threshold must be positive")

    normalized = normalize_date_index(df)
    numeric = numeric_frame(normalized)

    print("Macro feature quality diagnostics")
    date_coverage(normalized, top_n)
    missing_values(normalized, top_n)
    constant_columns(numeric, near_constant_threshold, top_n)
    exact_duplicate_columns(numeric, top_n)
    high_correlations(numeric, corr_threshold, top_n)
    robust_mad_outliers(numeric, outlier_z_threshold, top_n)
    biggest_one_day_changes(numeric, top_n)


def main() -> None:
    args = parse_args()
    df = load_workbook(args.workbook)
    print_quality_diagnostics(
        df,
        args.near_constant_threshold,
        args.corr_threshold,
        args.outlier_z_threshold,
        args.top_n,
    )


if __name__ == "__main__":
    main()
