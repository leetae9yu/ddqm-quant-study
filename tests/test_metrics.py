from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import math

import pandas as pd
import pytest

from autoquant_lab.eqr.metrics import REQUIRED_METRIC_KEYS, evaluate_predictions, max_drawdown, rank_ic, turnover_proxy


def test_rank_ic_is_period_level_spearman_average() -> None:
    y_true = pd.Series([0.1, 0.2, 0.3, 0.3, 0.2, 0.1])
    y_pred = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    periods = pd.Series(["2020-01-31"] * 3 + ["2020-02-29"] * 3)

    assert math.isclose(rank_ic(y_true, y_pred, periods) or 0.0, 0.0, abs_tol=1e-12)


def test_evaluate_predictions_contains_required_metric_keys() -> None:
    frame = pd.DataFrame(
        {
            "permno": [1, 2, 3, 1, 2, 3],
            "period": pd.to_datetime(["2020-01-31"] * 3 + ["2020-02-29"] * 3),
            "actual": [0.1, -0.2, 0.3, 0.2, -0.1, 0.05],
            "prediction": [0.2, -0.1, 0.4, 0.1, -0.2, 0.03],
            "feature": [1.0, None, 3.0, 4.0, 5.0, 6.0],
        }
    )

    metrics = evaluate_predictions(
        y_true=frame["actual"],
        y_pred=frame["prediction"],
        periods=frame["period"],
        ids=frame["permno"],
        features=frame[["feature"]],
        runtime_seconds=0.25,
    )

    for key in REQUIRED_METRIC_KEYS:
        assert key in metrics
    assert metrics["runtime"]["seconds"] == 0.25
    assert metrics["feature_coverage"] == frame[["feature"]].notna().mean().mean()


def test_turnover_and_drawdown_proxies() -> None:
    ids = pd.Series([1, 2, 3, 1, 2, 3])
    pred = pd.Series([0.1, 0.9, 0.2, 0.8, 0.1, 0.2])
    periods = pd.Series(["2020-01-31"] * 3 + ["2020-02-29"] * 3)

    assert turnover_proxy(ids, pred, periods, top_fraction=1 / 3) == 1.0
    assert max_drawdown(pd.Series([0.1, -0.2, 0.05])) < 0.0


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="identical lengths"):
        rank_ic(pd.Series([0.1, 0.2]), pd.Series([0.1]), pd.Series(["2020-01-31", "2020-01-31"]))
