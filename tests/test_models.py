from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false

import numpy as np
import pandas as pd

from autoquant_lab.eqr.models.base import EQRModel
from autoquant_lab.eqr.models.registry import available_models, create_model, get_model_class


def test_registry_contains_required_cpu_models() -> None:
    names = set(available_models())
    assert {"baseline_linear", "ridge", "elasticnet", "random_forest", "extra_trees", "lightgbm"}.issubset(names)
    assert issubclass(get_model_class("baseline_linear"), EQRModel)


def test_baseline_linear_fit_predict_interface() -> None:
    X = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "quality": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0.01, 0.02, 0.03, 0.04])
    periods = pd.Series(pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]))

    model = create_model("baseline_linear", {"alpha": 0.1}).fit(X, y, periods=periods)
    predictions = model.predict(X)

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_naive_baselines_predict_expected_shapes() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([0.01, -0.02, 0.03])
    for name in ("baseline_mean", "baseline_median", "baseline_random"):
        predictions = create_model(name).fit(X, y).predict(X)
        assert predictions.shape == (3,)
        assert np.isfinite(predictions).all()
