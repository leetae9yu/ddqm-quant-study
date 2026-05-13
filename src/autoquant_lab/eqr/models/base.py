"""Common model interface for EQR cross-sectional regression and ranking."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class EQRModel(ABC):
    """Abstract estimator used by EQR trainers.

    Models receive a dense numeric feature matrix and a forward-return target.
    Optional period labels allow ranking models to construct cross-sectional
    groups without exposing library-specific APIs to the evaluator.
    """

    supports_ranking: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.feature_names_: list[str] = []

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.float64],
        *,
        periods: pd.Series | NDArray[Any] | None = None,
    ) -> "EQRModel":
        """Fit the model and return ``self``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one numeric score per row."""

    def _remember_features(self, X: pd.DataFrame | NDArray[np.float64]) -> None:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)


def period_group_sizes(periods: pd.Series | NDArray[Any] | None) -> list[int] | None:
    """Return contiguous LightGBM-style group sizes for period labels."""

    if periods is None:
        return None
    series = pd.Series(periods).reset_index(drop=True)
    if series.empty:
        return []
    run_ids = series.ne(series.shift()).cumsum()
    runs = pd.DataFrame({"period": series, "run": run_ids}).groupby("run", sort=False)["period"].first()
    duplicated_periods = runs[runs.duplicated()].unique().tolist()
    if duplicated_periods:
        raise ValueError("Period labels must be contiguous for ranking group construction")
    return [int(size) for size in series.groupby(run_ids, sort=False).size().tolist()]
