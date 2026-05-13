from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false

import pandas as pd
import pytest

from autoquant_lab.eqr.evaluation import evaluate_model, select_by_validation, time_series_split
from autoquant_lab.eqr.models.baseline import MeanBaselineModel


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in range(1, 7):
        period = pd.Timestamp(2020, month, 29 if month == 2 else 28)
        for permno in range(1, 5):
            rows.append(
                {
                    "permno": permno,
                    "formation_date": period,
                    "feature": float(permno + month),
                    "ret_1m_fwd": float((permno - 2) * 0.01 + month * 0.001),
                }
            )
    return pd.DataFrame(rows)


def test_time_series_split_is_chronological_and_disjoint() -> None:
    split = time_series_split(_frame()["formation_date"], validation_fraction=0.2, holdout_fraction=0.2)

    assert max(split.train_periods) < min(split.validation_periods)
    assert max(split.validation_periods) < min(split.holdout_periods)
    assert not (set(split.train_periods) & set(split.holdout_periods))


def test_evaluate_model_reports_all_splits_and_holdout_lock() -> None:
    result = evaluate_model(
        model=MeanBaselineModel(),
        frame=_frame(),
        feature_columns=["feature"],
        target_column="ret_1m_fwd",
    )

    assert {"train", "validation", "holdout"}.issubset(result.metrics)
    assert result.metrics["holdout_lock"]["holdout_used_for_selection"] is False
    assert set(result.predictions["split"]) == {"train", "validation", "holdout"}


def test_select_by_validation_ignores_better_holdout_metric() -> None:
    first = evaluate_model(model=MeanBaselineModel(), frame=_frame(), feature_columns=["feature"], target_column="ret_1m_fwd")
    second = evaluate_model(model=MeanBaselineModel(), frame=_frame(), feature_columns=["feature"], target_column="ret_1m_fwd")
    first.metrics["validation"]["rank_ic"] = 0.5
    first.metrics["holdout"]["rank_ic"] = -1.0
    second.metrics["validation"]["rank_ic"] = 0.1
    second.metrics["holdout"]["rank_ic"] = 1.0

    selected = select_by_validation({"first": first, "second": second}, metric="rank_ic")

    assert selected.selected_model_name == "first"
    assert selected.metrics["holdout_lock"]["selected_by"] == "validation"


def test_time_series_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="less than 1"):
        time_series_split(_frame()["formation_date"], validation_fraction=0.6, holdout_fraction=0.5)
