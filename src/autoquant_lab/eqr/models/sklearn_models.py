"""Scikit-learn CPU model wrappers for EQR experiments."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import EQRModel


class SklearnRegressorModel(EQRModel):
    """Base wrapper around a scikit-learn regressor."""

    estimator_class: type[Any]
    scale_features: bool = False
    default_params: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        merged = dict(self.default_params)
        merged.update(params)
        super().__init__(**merged)
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("estimator", self.estimator_class(**merged)))
        self.pipeline = Pipeline(steps)

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.float64],
        *,
        periods: pd.Series | NDArray[Any] | None = None,
    ) -> "SklearnRegressorModel":
        del periods
        self._remember_features(X)
        self.pipeline.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.pipeline.predict(X), dtype=float)


class RidgeModel(SklearnRegressorModel):
    estimator_class = Ridge
    scale_features = True
    default_params = {"alpha": 1.0, "random_state": 42}


class ElasticNetModel(SklearnRegressorModel):
    estimator_class = ElasticNet
    scale_features = True
    default_params = {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 2000, "random_state": 42}


class RandomForestModel(SklearnRegressorModel):
    estimator_class = RandomForestRegressor
    default_params = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42}


class ExtraTreesModel(SklearnRegressorModel):
    estimator_class = ExtraTreesRegressor
    default_params = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42}
