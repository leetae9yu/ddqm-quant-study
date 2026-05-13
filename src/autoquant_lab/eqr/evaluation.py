"""Time-series evaluation with a locked holdout split."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .metrics import evaluate_predictions
from .models.base import EQRModel


@dataclass(frozen=True)
class SplitDefinition:
    train_periods: tuple[pd.Timestamp, ...]
    validation_periods: tuple[pd.Timestamp, ...]
    holdout_periods: tuple[pd.Timestamp, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": [str(period.date()) for period in self.train_periods],
            "validation": [str(period.date()) for period in self.validation_periods],
            "holdout": [str(period.date()) for period in self.holdout_periods],
        }


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    split_definition: SplitDefinition
    selected_model_name: str | None = None
    selection_metric: str | None = None


def time_series_split(
    periods: pd.Series | np.ndarray,
    *,
    validation_fraction: float = 0.2,
    holdout_fraction: float = 0.2,
    min_train_periods: int = 1,
) -> SplitDefinition:
    """Build explicit chronological train/validation/holdout period splits."""

    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1")
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be between 0 and 1")
    if validation_fraction + holdout_fraction >= 1.0:
        raise ValueError("validation_fraction plus holdout_fraction must be less than 1")
    if min_train_periods < 1:
        raise ValueError("min_train_periods must be at least 1")
    unique_periods = tuple(pd.Timestamp(period) for period in sorted(pd.Series(periods).dropna().unique()))
    if len(unique_periods) < min_train_periods + 2:
        raise ValueError("At least three periods are required for train/validation/holdout evaluation")
    holdout_count = max(1, int(round(len(unique_periods) * holdout_fraction)))
    validation_count = max(1, int(round(len(unique_periods) * validation_fraction)))
    if holdout_count + validation_count > len(unique_periods) - min_train_periods:
        excess = holdout_count + validation_count - (len(unique_periods) - min_train_periods)
        validation_count = max(1, validation_count - excess)
    train_end = len(unique_periods) - holdout_count - validation_count
    if train_end < min_train_periods:
        raise ValueError("Not enough training periods after validation/holdout split")
    return SplitDefinition(
        train_periods=unique_periods[:train_end],
        validation_periods=unique_periods[train_end : train_end + validation_count],
        holdout_periods=unique_periods[train_end + validation_count :],
    )


def _split_mask(periods: pd.Series, split_periods: tuple[pd.Timestamp, ...]) -> pd.Series:
    return pd.to_datetime(periods).isin(split_periods)


def evaluate_model(
    *,
    model: EQRModel,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    period_column: str = "formation_date",
    id_column: str = "permno",
    split_definition: SplitDefinition | None = None,
    validation_fraction: float = 0.2,
    holdout_fraction: float = 0.2,
) -> EvaluationResult:
    """Fit on train only and evaluate train/validation/locked-holdout splits."""

    working = frame.dropna(subset=[target_column, period_column]).sort_values([period_column, id_column]).reset_index(drop=True)
    split_definition = split_definition or time_series_split(
        working[period_column], validation_fraction=validation_fraction, holdout_fraction=holdout_fraction
    )
    train_mask = _split_mask(working[period_column], split_definition.train_periods)
    if not train_mask.any():
        raise ValueError("Training split is empty")

    X_train = working.loc[train_mask, feature_columns]
    y_train = working.loc[train_mask, target_column]
    train_periods = working.loc[train_mask, period_column]

    fit_start = perf_counter()
    model.fit(X_train, y_train, periods=train_periods)
    fit_seconds = perf_counter() - fit_start

    split_masks = {
        "train": train_mask,
        "validation": _split_mask(working[period_column], split_definition.validation_periods),
        "holdout": _split_mask(working[period_column], split_definition.holdout_periods),
    }
    metrics: dict[str, Any] = {}
    predictions: list[pd.DataFrame] = []
    for split_name, mask in split_masks.items():
        split_frame = working.loc[mask].copy()
        if split_frame.empty:
            raise ValueError(f"{split_name} split is empty")
        predict_start = perf_counter()
        y_pred = model.predict(split_frame[feature_columns])
        predict_seconds = perf_counter() - predict_start
        metrics[split_name] = evaluate_predictions(
            y_true=split_frame[target_column],
            y_pred=y_pred,
            periods=split_frame[period_column],
            ids=split_frame[id_column] if id_column in split_frame.columns else None,
            features=split_frame[feature_columns],
            runtime_seconds=(fit_seconds if split_name == "train" else 0.0) + predict_seconds,
        )
        predictions.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    id_column: split_frame[id_column].to_numpy() if id_column in split_frame.columns else np.arange(len(split_frame)),
                    period_column: split_frame[period_column].to_numpy(),
                    "actual": split_frame[target_column].to_numpy(dtype=float),
                    "prediction": y_pred,
                }
            )
        )

    metrics["split_periods"] = split_definition.as_dict()
    metrics["holdout_lock"] = {
        "enabled": True,
        "selection_uses": "validation",
        "holdout_used_for_selection": False,
    }
    return EvaluationResult(metrics=metrics, predictions=pd.concat(predictions, ignore_index=True), split_definition=split_definition)


def select_by_validation(
    results: dict[str, EvaluationResult],
    *,
    metric: str = "rank_ic",
    higher_is_better: bool = True,
) -> EvaluationResult:
    """Select a candidate using validation metrics only, preserving holdout lock."""

    if not results:
        raise ValueError("No evaluation results supplied")
    scored: list[tuple[str, float]] = []
    for name, result in results.items():
        value = result.metrics.get("validation", {}).get(metric)
        if value is None or not np.isfinite(float(value)):
            continue
        scored.append((name, float(value)))
    if not scored:
        raise ValueError(f"No finite validation metric available for '{metric}'")
    selected_name, _ = max(scored, key=lambda item: item[1]) if higher_is_better else min(scored, key=lambda item: item[1])
    selected = results[selected_name]
    metrics = dict(selected.metrics)
    metrics["holdout_lock"] = dict(metrics.get("holdout_lock", {}), selected_by="validation", selection_metric=metric)
    return EvaluationResult(
        metrics=metrics,
        predictions=selected.predictions,
        split_definition=selected.split_definition,
        selected_model_name=selected_name,
        selection_metric=metric,
    )
