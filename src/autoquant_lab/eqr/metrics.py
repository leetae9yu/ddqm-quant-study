"""Multi-metric evaluation for EQR cross-sectional predictions."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportReturnType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportIndexIssue=false

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "rank_ic",
    "pearson_ic",
    "decile_long_short_return",
    "hit_rate",
    "mse",
    "mae",
    "turnover_proxy",
    "max_drawdown",
    "stability",
    "feature_coverage",
    "runtime",
)


def _valid_frame(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, periods: pd.Series | np.ndarray) -> pd.DataFrame:
    lengths = {len(y_true), len(y_pred), len(periods)}
    if len(lengths) != 1:
        raise ValueError("y_true, y_pred, and periods must have identical lengths")
    frame = pd.DataFrame(
        {
            "actual": pd.Series(y_true).reset_index(drop=True),
            "prediction": pd.Series(y_pred).reset_index(drop=True),
            "period": pd.to_datetime(pd.Series(periods).reset_index(drop=True), errors="coerce"),
        }
    )
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    return frame.dropna(subset=["actual", "prediction", "period"]).reset_index(drop=True)


def _safe_mean(values: list[float] | pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _period_correlations(frame: pd.DataFrame, method: str) -> pd.Series:
    values: dict[Any, float] = {}
    for period, group in frame.groupby("period", sort=True):
        if len(group) < 2 or group["actual"].nunique(dropna=True) < 2 or group["prediction"].nunique(dropna=True) < 2:
            continue
        corr = group["prediction"].corr(group["actual"], method=method)
        if pd.notna(corr):
            values[period] = float(corr)
    return pd.Series(values, dtype=float)


def rank_ic(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, periods: pd.Series | np.ndarray) -> float | None:
    """Mean period-level Spearman information coefficient."""

    return _safe_mean(_period_correlations(_valid_frame(y_true, y_pred, periods), "spearman"))


def pearson_ic(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, periods: pd.Series | np.ndarray) -> float | None:
    """Mean period-level Pearson information coefficient."""

    return _safe_mean(_period_correlations(_valid_frame(y_true, y_pred, periods), "pearson"))


def decile_long_short_return(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, periods: pd.Series | np.ndarray) -> float | None:
    """Mean return of top predicted decile minus bottom predicted decile by period."""

    frame = _valid_frame(y_true, y_pred, periods)
    returns: list[float] = []
    for _, group in frame.groupby("period", sort=True):
        if len(group) < 2:
            continue
        ranked = group.sort_values("prediction")
        decile_size = max(1, int(np.ceil(len(ranked) * 0.1)))
        bottom = ranked.head(decile_size)["actual"].mean()
        top = ranked.tail(decile_size)["actual"].mean()
        returns.append(float(top - bottom))
    return _safe_mean(returns)


def hit_rate(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float | None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical lengths")
    actual = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    pred = pd.to_numeric(pd.Series(y_pred).reset_index(drop=True), errors="coerce")
    mask = actual.notna() & pred.notna() & (actual != 0) & (pred != 0)
    if not mask.any():
        return None
    return float((np.sign(actual[mask]) == np.sign(pred[mask])).mean())


def mse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float | None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical lengths")
    actual = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    pred = pd.to_numeric(pd.Series(y_pred).reset_index(drop=True), errors="coerce")
    mask = actual.notna() & pred.notna()
    if not mask.any():
        return None
    return float(np.mean(np.square(actual[mask].to_numpy(dtype=float) - pred[mask].to_numpy(dtype=float))))


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float | None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical lengths")
    actual = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    pred = pd.to_numeric(pd.Series(y_pred).reset_index(drop=True), errors="coerce")
    mask = actual.notna() & pred.notna()
    if not mask.any():
        return None
    return float(np.mean(np.abs(actual[mask].to_numpy(dtype=float) - pred[mask].to_numpy(dtype=float))))


def period_long_short_returns(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, periods: pd.Series | np.ndarray) -> pd.Series:
    frame = _valid_frame(y_true, y_pred, periods)
    values: dict[Any, float] = {}
    for period, group in frame.groupby("period", sort=True):
        if len(group) < 2:
            continue
        ranked = group.sort_values("prediction")
        decile_size = max(1, int(np.ceil(len(ranked) * 0.1)))
        values[period] = float(ranked.tail(decile_size)["actual"].mean() - ranked.head(decile_size)["actual"].mean())
    return pd.Series(values, dtype=float)


def turnover_proxy(ids: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, periods: pd.Series | np.ndarray, top_fraction: float = 0.1) -> float | None:
    """Average symmetric-difference turnover of top-score names across periods."""

    if len(ids) != len(y_pred) or len(ids) != len(periods):
        raise ValueError("ids, y_pred, and periods must have identical lengths")
    frame = pd.DataFrame({"id": ids, "prediction": y_pred, "period": periods}).dropna(subset=["id", "prediction", "period"])
    previous: set[Any] | None = None
    turnovers: list[float] = []
    for _, group in frame.groupby("period", sort=True):
        count = max(1, int(np.ceil(len(group) * top_fraction)))
        current = set(group.sort_values("prediction").tail(count)["id"].tolist())
        if previous is not None:
            denom = max(1, len(current | previous))
            turnovers.append(len(current.symmetric_difference(previous)) / denom)
        previous = current
    return _safe_mean(turnovers)


def max_drawdown(returns: pd.Series | np.ndarray) -> float | None:
    series = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if series.empty:
        return None
    equity = (1.0 + series).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def stability(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, periods: pd.Series | np.ndarray) -> dict[str, float | None]:
    """Return IC stability summaries overall and by calendar year when possible."""

    frame = _valid_frame(y_true, y_pred, periods)
    ic_by_period = _period_correlations(frame, "spearman")
    ic_std = None if ic_by_period.empty else float(ic_by_period.std(ddof=0))
    positive_share = None if ic_by_period.empty else float((ic_by_period > 0).mean())
    by_year: dict[int, float] = {}
    if not ic_by_period.empty:
        index = pd.to_datetime(pd.Series(ic_by_period.index), errors="coerce")
        yearly = pd.DataFrame({"year": index.dt.year.to_numpy(), "ic": ic_by_period.to_numpy(dtype=float)}).dropna()
        by_year = {int(year): float(group["ic"].mean()) for year, group in yearly.groupby("year")}
    yearly_std = None if len(by_year) < 2 else float(np.std(list(by_year.values()), ddof=0))
    return {"ic_std": ic_std, "positive_ic_share": positive_share, "yearly_ic_std": yearly_std}


def feature_coverage(features: pd.DataFrame) -> float | None:
    if features.empty:
        return None
    return float(features.notna().mean().mean())


def evaluate_predictions(
    *,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    periods: pd.Series | np.ndarray,
    ids: pd.Series | np.ndarray | None = None,
    features: pd.DataFrame | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute all required EQR metrics for one split."""

    valid_frame = _valid_frame(y_true, y_pred, periods)
    ls_returns = period_long_short_returns(y_true, y_pred, periods)
    identifier = ids if ids is not None else np.arange(len(pd.Series(y_true)))
    return {
        "rank_ic": rank_ic(y_true, y_pred, periods),
        "pearson_ic": pearson_ic(y_true, y_pred, periods),
        "decile_long_short_return": _safe_mean(ls_returns),
        "hit_rate": hit_rate(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "turnover_proxy": turnover_proxy(identifier, y_pred, periods),
        "max_drawdown": max_drawdown(ls_returns),
        "stability": stability(y_true, y_pred, periods),
        "feature_coverage": feature_coverage(features) if features is not None else None,
        "runtime": {"seconds": runtime_seconds},
        "row_count": int(len(valid_frame)),
        "period_count": int(pd.Series(periods).nunique(dropna=True)),
    }
