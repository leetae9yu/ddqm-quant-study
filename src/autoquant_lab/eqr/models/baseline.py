"""Naive EQR baselines."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import EQRModel


class ConstantBaselineModel(EQRModel):
    """Predict the training mean or median forward return."""

    def __init__(self, strategy: Literal["mean", "median"] = "mean", **params: Any) -> None:
        super().__init__(strategy=strategy, **params)
        self.strategy = strategy
        self.value_: float = 0.0

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.float64],
        *,
        periods: pd.Series | NDArray[Any] | None = None,
    ) -> "ConstantBaselineModel":
        del periods
        self._remember_features(X)
        values = np.asarray(y, dtype=float)
        self.value_ = float(np.nanmedian(values) if self.strategy == "median" else np.nanmean(values))
        if not np.isfinite(self.value_):
            self.value_ = 0.0
        return self

    def predict(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(len(X), self.value_, dtype=float)


class MeanBaselineModel(ConstantBaselineModel):
    def __init__(self, **params: Any) -> None:
        super().__init__(strategy="mean", **params)


class MedianBaselineModel(ConstantBaselineModel):
    def __init__(self, **params: Any) -> None:
        super().__init__(strategy="median", **params)


class RandomBaselineModel(EQRModel):
    """Deterministic random ranking baseline."""

    def __init__(self, random_state: int = 42, **params: Any) -> None:
        super().__init__(random_state=random_state, **params)
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.float64],
        *,
        periods: pd.Series | NDArray[Any] | None = None,
    ) -> "RandomBaselineModel":
        del y, periods
        self._remember_features(X)
        return self

    def predict(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        rng = np.random.default_rng(self.random_state)
        return rng.normal(size=len(X)).astype(float)
