"""DDQM2-style factor-return modeling, allocation, and backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..models.registry import create_model


@dataclass(frozen=True)
class FactorModelResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def macro_design_matrix(feature_panel: pd.DataFrame) -> pd.DataFrame:
    macro_cols = [col for col in feature_panel.columns if col.startswith("macro__")]
    if not macro_cols:
        raise ValueError("feature_panel contains no macro__ columns for DDQM2 modeling")
    macro = feature_panel[["formation_date", *macro_cols]].copy()
    macro["formation_date"] = pd.to_datetime(macro["formation_date"], errors="coerce")
    macro = macro.dropna(subset=["formation_date"]).groupby("formation_date", as_index=False)[macro_cols].last()
    return macro.sort_values("formation_date").reset_index(drop=True)


def _split_dates(dates: pd.Series, validation_fraction: float, holdout_fraction: float) -> tuple[set[pd.Timestamp], set[pd.Timestamp], set[pd.Timestamp]]:
    unique = [pd.Timestamp(date) for date in sorted(pd.Series(dates).dropna().unique())]
    if len(unique) < 6:
        raise ValueError("At least six dates are required for DDQM2 factor modeling")
    holdout_n = max(1, int(round(len(unique) * holdout_fraction)))
    valid_n = max(1, int(round(len(unique) * validation_fraction)))
    train_end = max(1, len(unique) - valid_n - holdout_n)
    return set(unique[:train_end]), set(unique[train_end : train_end + valid_n]), set(unique[train_end + valid_n :])


def train_factor_return_models(
    factor_returns: pd.DataFrame,
    feature_panel: pd.DataFrame,
    *,
    model_name: str = "lightgbm",
    model_params: dict[str, Any] | None = None,
    validation_fraction: float = 0.15,
    holdout_fraction: float = 0.15,
    min_observations: int = 24,
) -> FactorModelResult:
    """Train one CPU model per factor to forecast next 1M factor return."""

    macro = macro_design_matrix(feature_panel)
    frame = factor_returns.merge(macro, on="formation_date", how="inner")
    frame = frame.dropna(subset=["factor_long_short_ret_1m"])
    feature_cols = [col for col in macro.columns if col != "formation_date"]
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for factor_id, group in frame.groupby("factor_id", sort=True):
        group = group.sort_values("formation_date").reset_index(drop=True)
        if len(group) < min_observations:
            continue
        train_dates, valid_dates, holdout_dates = _split_dates(group["formation_date"], validation_fraction, holdout_fraction)
        train = group.loc[group["formation_date"].isin(train_dates)].copy()
        valid = group.loc[group["formation_date"].isin(valid_dates)].copy()
        holdout = group.loc[group["formation_date"].isin(holdout_dates)].copy()
        if train.empty or valid.empty or holdout.empty:
            continue
        model = create_model(model_name, model_params or {})
        start = perf_counter()
        model.fit(train[feature_cols], train["factor_long_short_ret_1m"], periods=train["formation_date"])
        train_pred = model.predict(train[feature_cols])
        valid_pred = model.predict(valid[feature_cols])
        holdout_pred = model.predict(holdout[feature_cols])
        runtime = perf_counter() - start
        for split_name, split, pred in (("train", train, train_pred), ("validation", valid, valid_pred), ("holdout", holdout, holdout_pred)):
            out = split[["formation_date", "factor_id", "factor_long_short_ret_1m"]].copy()
            out["prediction"] = pred
            out["split"] = split_name
            out["model"] = model_name
            predictions.append(out)
        metrics.append(
            {
                "factor_id": factor_id,
                "model": model_name,
                "train_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "holdout_rows": int(len(holdout)),
                "validation_correlation": float(pd.Series(valid_pred).corr(valid["factor_long_short_ret_1m"], method="spearman")) if len(valid) > 1 else None,
                "holdout_correlation": float(pd.Series(holdout_pred).corr(holdout["factor_long_short_ret_1m"], method="spearman")) if len(holdout) > 1 else None,
                "validation_mae": float(np.mean(np.abs(np.asarray(valid_pred) - valid["factor_long_short_ret_1m"].to_numpy()))),
                "holdout_mae": float(np.mean(np.abs(np.asarray(holdout_pred) - holdout["factor_long_short_ret_1m"].to_numpy()))),
                "runtime_seconds": float(runtime),
            }
        )
    if not predictions:
        return FactorModelResult(pd.DataFrame(), pd.DataFrame(metrics))
    return FactorModelResult(pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics))


def build_factor_allocations(predictions: pd.DataFrame, *, min_weight: float = 0.0) -> pd.DataFrame:
    """Convert predicted factor returns into DDQM2 non-negative factor weights."""

    if predictions.empty:
        return pd.DataFrame(columns=["formation_date", "factor_id", "prediction", "weight", "split"])
    frame = predictions.copy()
    frame["positive_prediction"] = pd.to_numeric(frame["prediction"], errors="coerce").clip(lower=0.0)
    rows: list[pd.DataFrame] = []
    for (date, split), group in frame.groupby(["formation_date", "split"], sort=True):
        denom = group["positive_prediction"].sum()
        out = group[["formation_date", "factor_id", "prediction", "split"]].copy()
        if pd.isna(denom) or denom <= 0:
            out["weight"] = 1.0 / len(out)
        else:
            out["weight"] = group["positive_prediction"] / denom
            if min_weight > 0:
                out["weight"] = out["weight"].clip(lower=min_weight)
                out["weight"] = out["weight"] / out["weight"].sum()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def backtest_factor_allocations(allocations: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Backtest the predicted factor-weight portfolio using realized factor returns."""

    if allocations.empty:
        return pd.DataFrame(columns=["formation_date", "split", "portfolio_return", "cumulative_return"])
    realized = factor_returns[["formation_date", "factor_id", "factor_long_short_ret_1m"]].copy()
    merged = allocations.merge(realized, on=["formation_date", "factor_id"], how="left")
    merged["weighted_return"] = merged["weight"] * merged["factor_long_short_ret_1m"]
    portfolio = merged.groupby(["formation_date", "split"], as_index=False)["weighted_return"].sum().rename(columns={"weighted_return": "portfolio_return"})
    portfolio = portfolio.sort_values("formation_date").reset_index(drop=True)
    portfolio["cumulative_return"] = (1.0 + portfolio["portfolio_return"].fillna(0.0)).cumprod() - 1.0
    return portfolio
