"""Model registry for CPU-only EQR experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import EQRModel
from .baseline import MeanBaselineModel, MedianBaselineModel, RandomBaselineModel
from .lightgbm_model import LightGBMModel
from .sklearn_models import ElasticNetModel, ExtraTreesModel, RandomForestModel, RidgeModel


def model_registry() -> dict[str, type[EQRModel]]:
    """Return supported model names mapped to model classes."""

    return {
        "baseline_linear": RidgeModel,
        "ridge": RidgeModel,
        "elasticnet": ElasticNetModel,
        "random_forest": RandomForestModel,
        "extratrees": ExtraTreesModel,
        "extra_trees": ExtraTreesModel,
        "baseline_mean": MeanBaselineModel,
        "baseline_median": MedianBaselineModel,
        "baseline_random": RandomBaselineModel,
        "lightgbm": LightGBMModel,
    }


def available_models() -> list[str]:
    return sorted(model_registry())


def get_model_class(name: str) -> type[EQRModel]:
    registry = model_registry()
    if name not in registry:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(available_models())}")
    return registry[name]


def create_model(name: str, params: Mapping[str, Any] | None = None) -> EQRModel:
    return get_model_class(name)(**dict(params or {}))
