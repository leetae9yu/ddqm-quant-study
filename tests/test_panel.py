from __future__ import annotations
# pyright: reportMissingImports=false

import math

import pandas as pd
import pytest

from autoquant_lab.eqr.panel import build_forward_returns, build_monthly_universe, build_panel, validate_labels


def _monthly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": [10001, 10001, 10001, 10001, 10001, 10001, 10002, 10002],
            "permco": [20001, 20001, 20001, 20001, 20001, 20001, 20002, 20002],
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",
                    "2020-04-30",
                    "2020-05-31",
                    "2020-06-30",
                    "2020-01-31",
                    "2020-02-29",
                ]
            ),
            "prc": [-10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0],
            "ret": [0.01, 0.02, -0.01, 0.03, 0.04, 0.05, 0.10, 0.20],
            "retx": [0.009, 0.019, -0.011, 0.029, 0.039, 0.049, 0.09, 0.19],
            "shrout": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 50.0, 51.0],
            "vol": [1000, 1100, 1200, 1300, 1400, 1500, 500, 600],
            "cfacpr": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
            "cfacshr": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "market_cap": [0.0] * 8,
        }
    )


def _names() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": [10001, 10002, 10003, 10004],
            "comnam": ["Included", "Excluded share", "Excluded exchange", "Future only"],
            "ncusip": ["11111111", "22222222", "33333333", "44444444"],
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "shrcd": [10, 12, 10, 11],
            "exchcd": [1, 1, 4, 3],
            "namedt": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-01", "2021-01-01"]),
            "nameenddt": pd.to_datetime([None, None, None, None]),
        }
    )


def test_forward_return_calculation_correctness() -> None:
    labels = build_forward_returns(_monthly(), horizons=(1, 3, 6))
    row = labels.loc[(labels["permno"] == 10001) & (labels["formation_date"] == pd.Timestamp("2020-01-31"))].iloc[0]

    assert math.isclose(row["ret_1m_fwd"], 0.02)
    assert math.isclose(row["ret_3m_fwd"], (1.02 * 0.99 * 1.03) - 1.0)
    assert pd.Timestamp(row["forward_return_start"]) == pd.Timestamp("2020-02-29")
    assert pd.Timestamp(row["forward_return_end"]) == pd.Timestamp("2020-02-29")
    assert pd.Timestamp(row["forward_return_end_3m"]) == pd.Timestamp("2020-04-30")
    assert pd.isna(row["ret_6m_fwd"])


def test_leakage_window_rejection() -> None:
    labels = build_panel(_monthly(), _names())
    bad = labels.copy()
    bad.loc[bad["ret_1m_fwd"].notna(), "forward_return_start"] = bad.loc[bad["ret_1m_fwd"].notna(), "formation_date"]

    with pytest.raises(ValueError, match="start after formation_date"):
        validate_labels(bad)


def test_feature_columns_cannot_include_label_data() -> None:
    labels = build_panel(_monthly(), _names())

    with pytest.raises(ValueError, match="Feature columns contain label"):
        validate_labels(labels, feature_columns=["market_cap", "ret_1m_fwd"])


def test_market_cap_calculation() -> None:
    universe = build_monthly_universe(_monthly(), _names())
    row = universe.loc[(universe["permno"] == 10001) & (universe["formation_date"] == pd.Timestamp("2020-01-31"))].iloc[0]

    assert row["market_cap"] == 10.0 * 100.0 * 1000.0
    assert row["adjusted_price"] == 5.0


def test_crsp_filter_application() -> None:
    universe = build_monthly_universe(_monthly(), _names())

    assert set(universe["permno"]) == {10001}
    assert set(universe["shrcd"]) == {10}
    assert set(universe["exchcd"]) == {1}


def test_label_availability_for_different_horizons() -> None:
    labels = build_panel(_monthly(), _names())
    first = labels.loc[labels["formation_date"] == pd.Timestamp("2020-01-31")].iloc[0]
    april = labels.loc[labels["formation_date"] == pd.Timestamp("2020-04-30")].iloc[0]
    june = labels.loc[labels["formation_date"] == pd.Timestamp("2020-06-30")].iloc[0]

    assert pd.notna(first["ret_1m_fwd"])
    assert pd.notna(first["ret_3m_fwd"])
    assert pd.isna(first["ret_6m_fwd"])
    assert pd.notna(april["ret_1m_fwd"])
    assert pd.isna(april["ret_3m_fwd"])
    assert pd.isna(june["ret_1m_fwd"])
