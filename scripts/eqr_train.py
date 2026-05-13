#!/usr/bin/env python3
"""Train and evaluate CPU EQR models on prepared monthly panels."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from autoquant_lab.eqr.evaluation import evaluate_model  # noqa: E402
from autoquant_lab.eqr.metrics import REQUIRED_METRIC_KEYS  # noqa: E402
from autoquant_lab.eqr.models.registry import available_models, create_model  # noqa: E402
from autoquant_lab.eqr.schemas import (  # noqa: E402
    EXPERIMENT_ARTIFACT_REQUIRED_FILES,
    EXPERIMENT_FEATURE_IMPORTANCE_REQUIRED_COLUMNS,
    EXPERIMENT_MANIFEST_REQUIRED_FIELDS,
    EXPERIMENT_PREDICTIONS_REQUIRED_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "golden_path.yaml"
DEFAULT_PANEL = PROJECT_ROOT / "experiments" / "prepared" / "panel" / "monthly_labels.parquet"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "experiments" / "prepared" / "features"
DEFAULT_RUN_DIR = PROJECT_ROOT / "experiments" / "runs"
TARGET_COLUMN = "ret_1m_fwd"
PERIOD_COLUMN = "formation_date"
ID_COLUMN = "permno"
EXCLUDED_FEATURE_COLUMNS = {
    "permno",
    "permco",
    "formation_date",
    "forward_return_start",
    "forward_return_end",
    "forward_return_end_3m",
    "forward_return_end_6m",
    "ret_1m_fwd",
    "ret_3m_fwd",
    "ret_6m_fwd",
    "universe_source",
    "label_source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU EQR model and persist evaluation artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Golden-path YAML config.")
    parser.add_argument("--model", required=True, choices=available_models(), help="Registered model name.")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL, help="Prepared monthly labels parquet.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR, help="Optional prepared feature parquet directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR, help="Experiment runs root directory.")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Forward-return target column.")
    parser.add_argument("--run-id", default=None, help="Optional run id; defaults to UTC timestamp and model name.")
    parser.add_argument("--max-rows", type=int, default=10_000, help="Deterministic smoke cap after chronological sorting; <=0 uses all rows.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Recent non-holdout period fraction for validation.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Most recent period fraction for locked holdout.")
    return parser.parse_args()


def _parse_scalar(value: str) -> Any:
    stripped = value.strip().strip('"').strip("'")
    if stripped == "":
        return ""
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped.startswith("[") and stripped.endswith("]"):
        return [_parse_scalar(item.strip()) for item in stripped[1:-1].split(",") if item.strip()]
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_prepared_features(feature_dir: Path) -> pd.DataFrame | None:
    if not feature_dir.exists():
        return None
    frames: list[pd.DataFrame] = []
    for path in sorted(feature_dir.glob("*.parquet")):
        if path.name.startswith("."):
            continue
        frame = pd.read_parquet(path)
        if {ID_COLUMN, PERIOD_COLUMN}.issubset(frame.columns):
            if frame.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
                raise ValueError(f"Prepared feature file has duplicate (permno, formation_date) keys: {path}")
            frames.append(frame)
    if not frames:
        return None
    merged = frames[0]
    for frame in frames[1:]:
        new_columns = [col for col in frame.columns if col not in merged.columns or col in {ID_COLUMN, PERIOD_COLUMN}]
        if merged.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
            raise ValueError("Prepared feature merge produced duplicate (permno, formation_date) keys")
        merged = merged.merge(frame[new_columns], on=[ID_COLUMN, PERIOD_COLUMN], how="left")
    return merged


def load_training_frame(panel_path: Path, feature_dir: Path) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"Prepared panel does not exist: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel[PERIOD_COLUMN] = pd.to_datetime(panel[PERIOD_COLUMN], errors="coerce")
    if panel.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
        raise ValueError("Prepared panel has duplicate (permno, formation_date) keys")
    features = _read_prepared_features(feature_dir)
    if features is None:
        return panel
    features[PERIOD_COLUMN] = pd.to_datetime(features[PERIOD_COLUMN], errors="coerce")
    feature_columns = [col for col in features.columns if col not in panel.columns or col in {ID_COLUMN, PERIOD_COLUMN}]
    if features.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
        raise ValueError("Prepared features have duplicate (permno, formation_date) keys")
    return panel.merge(features[feature_columns], on=[ID_COLUMN, PERIOD_COLUMN], how="left")


def select_feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    excluded = set(EXCLUDED_FEATURE_COLUMNS)
    excluded.add(target_column)
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns are available for training")
    return columns


def _model_params(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = config.get("models", {})
    if isinstance(models, dict):
        raw = models.get(model_name, {})
        if isinstance(raw, dict):
            return dict(raw.get("params", raw))
    return {}


def _smoke_cap(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    sorted_frame = frame.sort_values([PERIOD_COLUMN, ID_COLUMN]).reset_index(drop=True)
    if max_rows <= 0 or len(sorted_frame) <= max_rows:
        return sorted_frame
    period_counts = sorted_frame.groupby(PERIOD_COLUMN, sort=True).size()
    chosen: list[pd.Timestamp] = []
    running = 0
    for period, count_value in period_counts.items():
        count = int(count_value)
        if running > 0 and running + count > max_rows and len(chosen) >= 3:
            break
        chosen.append(pd.Timestamp(period))
        running += count
    capped = sorted_frame[sorted_frame[PERIOD_COLUMN].isin(chosen)].copy()
    return capped if capped[PERIOD_COLUMN].nunique() >= 3 else sorted_frame.head(max_rows).copy()


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, underscore, dash, and dot")
    if run_id in {".", ".."}:
        raise ValueError("run_id must name a child run directory")
    return run_id


def _safe_run_dir(output_dir: Path, run_id: str) -> Path:
    output_root = output_dir.resolve()
    run_dir = (output_root / _safe_run_id(run_id)).resolve()
    if output_root != run_dir and output_root not in run_dir.parents:
        raise ValueError("run_id must stay within output_dir")
    return run_dir


def _feature_importance_frame(model: object, feature_columns: list[str], run_id: str) -> pd.DataFrame:
    estimator = getattr(model, "model", None)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        pipeline = getattr(model, "pipeline", None)
        if pipeline is not None:
            final_estimator = pipeline.named_steps.get("estimator")
            importances = getattr(final_estimator, "feature_importances_", None)
            if importances is None:
                coefs = getattr(final_estimator, "coef_", None)
                if coefs is not None:
                    importances = np.abs(np.asarray(coefs, dtype=float))
    if importances is None:
        values = np.full(len(feature_columns), np.nan)
        importance_type = "unavailable"
    else:
        values = np.asarray(importances, dtype=float).reshape(-1)[: len(feature_columns)]
        if len(values) < len(feature_columns):
            values = np.pad(values, (0, len(feature_columns) - len(values)), constant_values=np.nan)
        importance_type = "model"
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
            "importance_type": importance_type,
            "run_id": run_id,
        },
        columns=list(EXPERIMENT_FEATURE_IMPORTANCE_REQUIRED_COLUMNS),
    )


def main() -> int:
    args = parse_args()
    config = load_simple_yaml(args.config)
    frame = load_training_frame(args.panel, args.feature_dir)
    if args.target not in frame.columns:
        raise ValueError(f"Target column '{args.target}' is missing from training frame")
    frame = _smoke_cap(frame.dropna(subset=[args.target, PERIOD_COLUMN]), args.max_rows)
    feature_columns = select_feature_columns(frame, args.target)

    model = create_model(args.model, _model_params(config, args.model))
    result = evaluate_model(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        target_column=args.target,
        period_column=PERIOD_COLUMN,
        id_column=ID_COLUMN,
        validation_fraction=args.validation_fraction,
        holdout_fraction=args.holdout_fraction,
    )
    run_id = _safe_run_id(args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.model}")
    run_dir = _safe_run_dir(args.output_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_id": run_id,
        "model": args.model,
        "target": args.target,
        "feature_columns": feature_columns,
        "required_metric_keys": list(REQUIRED_METRIC_KEYS),
        **result.metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    predictions = result.predictions.rename(
        columns={PERIOD_COLUMN: "date", ID_COLUMN: "asset_id", "actual": "target_return"}
    )
    predictions["run_id"] = run_id
    predictions = predictions[list(EXPERIMENT_PREDICTIONS_REQUIRED_COLUMNS)]
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    feature_importance = _feature_importance_frame(model, feature_columns, run_id)
    feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)
    config_artifact = {"config_path": str(args.config), "config": config, "model_params": _model_params(config, args.model)}
    (run_dir / "config.json").write_text(json.dumps(config_artifact, indent=2, default=_json_default), encoding="utf-8")
    joblib.dump(model, run_dir / "model.joblib")
    train_periods = result.split_definition.train_periods
    validation_periods = result.split_definition.validation_periods
    manifest = {
        "run_id": run_id,
        "input_artifact_path": str(args.panel),
        "output_dir": str(run_dir),
        "split_method": "chronological_train_validation_holdout",
        "train_start_date": str(min(train_periods).date()),
        "train_end_date": str(max(train_periods).date()),
        "validation_start_date": str(min(validation_periods).date()),
        "validation_end_date": str(max(validation_periods).date()),
        "target_column": args.target,
        "feature_columns": feature_columns,
        "metrics_artifact": "metrics.json",
        "predictions_artifact": "predictions.parquet",
        "feature_importance_artifact": "feature_importance.csv",
        "config_artifact": "config.json",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "panel": str(args.panel),
        "feature_dir": str(args.feature_dir),
        "row_count": int(len(frame)),
        "period_count": int(frame[PERIOD_COLUMN].nunique(dropna=True)),
        "artifacts": [*EXPERIMENT_ARTIFACT_REQUIRED_FILES, "model.joblib"],
    }
    missing_manifest_fields = [field for field in EXPERIMENT_MANIFEST_REQUIRED_FIELDS if field not in manifest]
    if missing_manifest_fields:
        raise ValueError(f"Manifest is missing required fields: {missing_manifest_fields}")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "model": args.model, "rows": len(frame)}, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
