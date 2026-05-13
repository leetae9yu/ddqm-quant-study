"""LightGBM wrapper for EQR regression and cross-sectional ranking."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportUnknownMemberType=false, reportCallIssue=false

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import EQRModel, period_group_sizes


class LightGBMModel(EQRModel):
    """LightGBM CPU estimator with optional LambdaRank grouping by period."""

    supports_ranking = True

    def __init__(self, objective: Literal["regression", "rank_xendcg", "lambdarank"] = "regression", **params: Any) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise ImportError("LightGBM is required for model 'lightgbm'. Install lightgbm to use it.") from exc

        defaults: dict[str, Any] = {
            "objective": objective,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        forbidden_gpu_keys = {"device", "device_type", "gpu_platform_id", "gpu_device_id"}
        requested_gpu = {key: value for key, value in params.items() if key in forbidden_gpu_keys and str(value).lower() != "cpu"}
        if requested_gpu:
            raise ValueError("GPU LightGBM parameters are not allowed in the CPU-only EQR registry")
        defaults.update(params)
        super().__init__(**defaults)
        self.objective = str(defaults.get("objective", objective))
        self.model = lgb.LGBMRanker(**defaults) if self.objective in {"rank_xendcg", "lambdarank"} else lgb.LGBMRegressor(**defaults)

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.float64],
        *,
        periods: pd.Series | NDArray[Any] | None = None,
    ) -> "LightGBMModel":
        self._remember_features(X)
        target = np.asarray(y, dtype=float)
        if self.objective in {"rank_xendcg", "lambdarank"}:
            groups = period_group_sizes(periods)
            if not groups:
                raise ValueError("LightGBM ranking objectives require period labels for group construction")
            ranked_target = pd.Series(target).groupby(pd.Series(periods).reset_index(drop=True), sort=False).rank(method="first").to_numpy()
            self.model.fit(X, ranked_target, group=groups)
        else:
            self.model.fit(X, target)
        return self

    def predict(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(X), dtype=float)
