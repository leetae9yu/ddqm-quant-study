from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAttributeAccessIssue=false

import math

import pandas as pd

from autoquant_lab.eqr.features.compustat_features import build_compustat_features
from autoquant_lab.eqr.features.feature_registry import build_feature_families, feature_metadata_records
from autoquant_lab.eqr.features.ibes_features import build_ibes_features
from autoquant_lab.eqr.features.macro_features import build_macro_features


def _panel(dates: list[str] | None = None) -> pd.DataFrame:
    dates = dates or ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31", "2020-06-30"]
    return pd.DataFrame(
        {
            "permno": [10001] * len(dates),
            "formation_date": pd.to_datetime(dates),
            "ret_1m": [0.01, 0.02, -0.01, 0.03, 0.04, 0.05][: len(dates)],
            "market_cap": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0][: len(dates)],
        }
    )


def test_macro_month_end_aggregation_uses_last_value_per_month() -> None:
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-30", "2020-02-03"]),
            "sp500": [100.0, 105.0, 99.0],
            "vix": [20.0, 18.0, 22.0],
        }
    )

    result = build_macro_features(_panel(["2020-01-31", "2020-02-29"]), macro_features=macro)

    jan = result.frame.loc[result.frame["formation_date"] == pd.Timestamp("2020-01-31")].iloc[0]
    feb = result.frame.loc[result.frame["formation_date"] == pd.Timestamp("2020-02-29")].iloc[0]
    assert jan["macro__sp500"] == 105.0
    assert jan["macro__vix"] == 18.0
    assert feb["macro__sp500"] == 99.0


def test_compustat_rdq_availability_and_90_day_fallback() -> None:
    panel = _panel(["2020-02-29", "2020-05-31", "2020-06-30"])
    fundq = pd.DataFrame(
        {
            "gvkey": ["001001", "001001"],
            "datadate": pd.to_datetime(["2019-12-31", "2020-03-31"]),
            "rdq": pd.to_datetime(["2020-02-15", None]),
            "oiadpq": [10.0, 12.0],
            "niq": [8.0, 9.0],
            "ceqq": [40.0, 45.0],
            "revtq": [100.0, 120.0],
            "atq": [80.0, 90.0],
            "ltq": [30.0, 35.0],
            "dlttq": [10.0, 12.0],
            "dlcq": [2.0, 3.0],
            "cshoq": [10.0, 10.0],
            "prccq": [20.0, 21.0],
            "epspxq": [1.0, 1.5],
            "saleq": [100.0, 120.0],
        }
    )
    ccm = pd.DataFrame(
        {
            "gvkey": ["001001"],
            "lpermno": [10001],
            "linkdt": pd.to_datetime(["2010-01-01"]),
            "linkenddt": pd.to_datetime([None]),
            "linktype": ["LC"],
            "linkprim": ["P"],
            "usedflag": [1],
        }
    )

    result = build_compustat_features(panel, comp_fundq=fundq, ccm_link=ccm).frame.sort_values("formation_date")

    assert result.iloc[0]["compustat__available_lag_days"] == 46.0
    assert result.iloc[1]["compustat__available_lag_days"] == 46.0
    assert result.iloc[2]["compustat__available_lag_days"] == 90.0
    assert math.isclose(result.iloc[2]["compustat__roe"], 9.0 / 45.0)


def test_ibes_statpers_window_behavior() -> None:
    panel = _panel(["2020-01-31", "2020-02-29", "2020-03-31"])
    link = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "permno": [10001],
            "sdate": pd.to_datetime(["2019-01-01"]),
            "edate": pd.to_datetime([None]),
            "score": [1],
        }
    )
    summary = pd.DataFrame(
        {
            "ticker": ["ABC", "ABC"],
            "statpers": pd.to_datetime(["2020-02-15", "2020-03-15"]),
            "measure": ["EPS", "EPS"],
            "meanest": [1.00, 1.20],
            "stdev": [0.10, 0.12],
            "numest": [5, 6],
            "actual": [1.10, 1.30],
            "actdats_act": pd.to_datetime(["2020-02-20", "2020-04-15"]),
            "anndats_act": pd.to_datetime(["2020-02-20", "2020-04-15"]),
        }
    )

    result = build_ibes_features(panel, ibes_link=link, ibes_summary=summary).frame.sort_values("formation_date")

    assert pd.isna(result.iloc[0]["ibes__mean_estimate"])
    assert result.iloc[1]["ibes__mean_estimate"] == 1.00
    assert result.iloc[2]["ibes__mean_estimate"] == 1.20
    assert pd.isna(result.iloc[2]["ibes__surprise"])


def test_feature_metadata_completeness() -> None:
    panel = _panel(["2020-01-31", "2020-02-29"])
    macro = pd.DataFrame({"date": pd.to_datetime(["2020-01-30", "2020-02-27"]), "sp500": [100.0, 101.0]})

    result = build_feature_families(panel=panel, inputs={"macro_features": macro}, families=["macro", "crsp"])
    metadata = feature_metadata_records(result)
    described = {record["feature"] for record in metadata}
    feature_cols = {col for col in result.frame.columns if "__" in col}

    assert feature_cols <= described
    assert all({"source_table", "availability_rule", "feature_family", "leakage_rule"} <= set(record) for record in metadata)
