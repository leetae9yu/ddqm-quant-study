"""Canonical EQR artifact schemas and validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


EQR_IDENTIFIER_COLUMNS: tuple[str, ...] = ("permno", "permco", "gvkey", "ticker")
ARTIFACT_PROVENANCE_COLUMNS: tuple[str, ...] = ("source", "asof_date", "generated_at_utc")

EQR_PANEL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "asset_id",
    "asset_id_type",
    *EQR_IDENTIFIER_COLUMNS,
    "return_1m",
    "market_cap",
    "universe_source",
    *ARTIFACT_PROVENANCE_COLUMNS,
)
EQR_PANEL_KEY_COLUMNS: tuple[str, ...] = ("date", "asset_id")

MODEL_DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "asset_id",
    "target_return",
    "target_horizon",
    "split",
    "label_source",
    "feature_set",
    "asof_date",
)
MODEL_DATASET_KEY_COLUMNS: tuple[str, ...] = ("date", "asset_id", "target_horizon")
MACRO_FEATURE_PREFIX = "macro__"

EXPERIMENT_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "input_artifact_path",
    "output_dir",
    "split_method",
    "train_start_date",
    "train_end_date",
    "validation_start_date",
    "validation_end_date",
    "target_column",
    "feature_columns",
    "metrics_artifact",
    "predictions_artifact",
    "feature_importance_artifact",
    "config_artifact",
)
EXPERIMENT_ARTIFACT_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "metrics.json",
    "predictions.parquet",
    "feature_importance.csv",
    "manifest.json",
)
EXPERIMENT_PREDICTIONS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "asset_id",
    "target_return",
    "prediction",
    "split",
    "run_id",
)
EXPERIMENT_FEATURE_IMPORTANCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "feature",
    "importance",
    "importance_type",
    "run_id",
)


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet dataset from disk."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    raise ValueError("Dataset must end in .csv or .parquet")


def write_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV or Parquet, creating parent directories."""

    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.suffix == ".parquet":
        df.to_parquet(dataset_path, index=False)
        return
    if dataset_path.suffix == ".csv":
        df.to_csv(dataset_path, index=False)
        return
    raise ValueError("Output must end in .csv or .parquet")


def require_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    """Raise when a DataFrame is missing required columns."""

    required_columns = tuple(required)
    missing_columns = sorted(set(required_columns).difference(df.columns))
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


def require_unique_key(df: pd.DataFrame, key: Iterable[str], dataset_name: str) -> None:
    """Raise when key columns are missing or key rows are duplicated."""

    key_columns = tuple(key)
    if not key_columns:
        raise ValueError(f"{dataset_name} unique key must include at least one column")

    require_columns(df, key_columns, dataset_name)
    duplicate_count = int(df.duplicated(list(key_columns)).sum())
    if duplicate_count:
        raise ValueError(f"{dataset_name} contains duplicate {key_columns} rows: {duplicate_count}")


def summarize_dataframe(df: pd.DataFrame, date_column: str | None = None) -> dict[str, object]:
    """Return deterministic row, column, date, and missing-value summary data."""

    summary: dict[str, object] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": {column: int(df[column].isna().sum()) for column in df.columns},
    }
    if date_column is None or date_column not in df.columns:
        return summary

    dates = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = dates.dropna()
    summary["invalid_dates"] = int(dates.isna().sum())
    summary["start_date"] = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else "n/a"
    summary["end_date"] = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else "n/a"
    return summary
