#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportMissingImports=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportReturnType=false
"""Build expanded macro and market features from FRED/ALFRED/yfinance data."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoquant_lab.config import Config  # noqa: E402


START_DATE = "1980-01-01"
DEFAULT_OUTPUT = PROJECT_ROOT / "expanded_macro_market_features.xlsx"

FRED_SERIES = {
    "10Y_Yield": "DGS10",
    "2Y_Yield": "DGS2",
    "10Y_3M_Spread": "T10Y3M",
    "TIPS_10Y": "DFII10",
    "Credit_Spread": "BAA10Y",
    "Expectation_Infla": "T10YIE",
    "NFCI": "NFCI",
    "Fed_Total_Assets": "WALCL",
}

ALFRED_SERIES = {
    "Nonfarm_Payrolls": "PAYEMS",
    "Unemployment_Rate": "UNRATE",
    "Housing_Starts": "HOUST",
    "Retail_Sales_Raw": "RSXFS",
    "Core_PCE_Raw": "PCEPILFE",
    "CPI_Raw": "CPIAUCSL",
    "Avg_Hourly_Earnings_Raw": "AHETPI",
    "Consumer_Sentiment": "UMCSENT",
}

YF_TICKERS = {
    "SP500_Close": "^GSPC",
    "Nasdaq_Close": "^IXIC",
    "Russell2000_Close": "^RUT",
    "VIX": "^VIX",
    "Copper": "HG=F",
    "WTI_Oil": "CL=F",
    "Gold": "GC=F",
    "Dollar_Index": "DX-Y.NYB",
    "USD_JPY": "JPY=X",
}

MARKET_DAILY_WEEKLY_FEATURES = list(YF_TICKERS.keys()) + list(FRED_SERIES.keys())

MACRO_MONTHLY_FEATURES = [
    "Nonfarm_Payrolls_MoM_Chg",
    "Unemployment_Rate_YoY_Chg",
    "Housing_Starts",
    "Retail_Sales_YoY",
    "Core_PCE_YoY",
    "CPI_YoY",
    "Avg_Hourly_Earnings_YoY",
    "Consumer_Sentiment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build expanded macro/market features. Live mode calls FRED and yfinance."
    )
    parser.add_argument("--start-date", default=START_DATE, help="Start date in YYYY-MM-DD format.")
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output workbook path.",
    )
    parser.add_argument(
        "--alfred-sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between ALFRED requests to avoid rate limits.",
    )
    return parser.parse_args()


def collect_fred_series(start_date: str, end_date: str) -> dict[str, pd.Series]:
    import pandas_datareader.data as web

    print("Standard FRED collecting...")
    data: dict[str, pd.Series] = {}
    for name, series_id in FRED_SERIES.items():
        try:
            df = web.DataReader(series_id, "fred", start_date, end_date)
            data[name] = df[series_id]
        except Exception as exc:  # noqa: BLE001 - keep collection resilient across series.
            print(f"API error ({name}): {exc}")
    return data


def get_alfred_initial_release(
    series_id: str,
    api_key: str,
    start_date: str,
    *,
    calc_yoy: bool = False,
    calc_diff: bool = False,
    months_lag: int = 1,
) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": start_date,
        "realtime_end": "9999-12-31",
        "units": "lin",
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        print(f"ALFRED request failed for {series_id}: HTTP {response.status_code}")
        return pd.Series(dtype=float)

    df = pd.DataFrame(response.json().get("observations", []))
    if df.empty:
        return pd.Series(dtype=float)

    df = df[df["value"] != "."].copy()
    df["value"] = df["value"].astype(float)
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["realtime_end"] = pd.to_datetime(df["realtime_end"].replace("9999-12-31", "2200-01-01"))
    df["date"] = pd.to_datetime(df["date"])

    idx_first_release = df.groupby("date")["realtime_start"].idxmin()
    first_releases = df.loc[idx_first_release, ["date", "realtime_start", "value"]]

    results: list[dict[str, object]] = []
    if calc_yoy:
        results = calculate_alfred_yoy(df, first_releases)
    elif calc_diff:
        results = calculate_alfred_diff(df, first_releases, months_lag)
    else:
        results = first_releases[["realtime_start", "value"]].to_dict("records")

    if not results:
        return pd.Series(dtype=float)

    result_df = pd.DataFrame(results).set_index("realtime_start")
    return result_df.groupby(result_df.index).last()["value"]


def calculate_alfred_yoy(df: pd.DataFrame, first_releases: pd.DataFrame) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for _, row in first_releases.iterrows():
        obs_date = row["date"]
        release_date = row["realtime_start"]
        current_value = row["value"]
        prev_obs_date = obs_date - pd.DateOffset(years=1)
        prev_row = df[
            (df["date"] == prev_obs_date)
            & (df["realtime_start"] <= release_date)
            & (df["realtime_end"] >= release_date)
        ]
        if not prev_row.empty:
            prev_value = prev_row["value"].values[0]
            if prev_value != 0:
                results.append({"realtime_start": release_date, "value": (current_value / prev_value - 1) * 100})
    return results


def calculate_alfred_diff(
    df: pd.DataFrame, first_releases: pd.DataFrame, months_lag: int
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for _, row in first_releases.iterrows():
        obs_date = row["date"]
        release_date = row["realtime_start"]
        current_value = row["value"]
        prev_obs_date = obs_date - pd.DateOffset(months=months_lag)
        prev_row = df[
            (df["date"] == prev_obs_date)
            & (df["realtime_start"] <= release_date)
            & (df["realtime_end"] >= release_date)
        ]
        if not prev_row.empty:
            results.append({"realtime_start": release_date, "value": current_value - prev_row["value"].values[0]})
    return results


def collect_alfred_series(api_key: str, start_date: str, sleep_seconds: float) -> dict[str, pd.Series]:
    print("ALFRED collecting...")
    data: dict[str, pd.Series] = {}
    for name, series_id in ALFRED_SERIES.items():
        calc_yoy = name in {"Retail_Sales_Raw", "Core_PCE_Raw", "CPI_Raw", "Avg_Hourly_Earnings_Raw"}
        calc_diff = name in {"Nonfarm_Payrolls", "Unemployment_Rate"}
        months_lag = 12 if name == "Unemployment_Rate" else 1
        output_name = name

        if calc_yoy:
            output_name = name.replace("_Raw", "_YoY")
        elif name == "Nonfarm_Payrolls":
            output_name = "Nonfarm_Payrolls_MoM_Chg"
        elif name == "Unemployment_Rate":
            output_name = "Unemployment_Rate_YoY_Chg"

        series = get_alfred_initial_release(
            series_id,
            api_key,
            start_date,
            calc_yoy=calc_yoy,
            calc_diff=calc_diff,
            months_lag=months_lag,
        )
        series.name = output_name
        data[output_name] = series
        time.sleep(sleep_seconds)
    return data


def build_macro_frame(data: dict[str, pd.Series], start_date: str, end_date: str) -> pd.DataFrame:
    print("Combining macro data with business-day calendar...")
    raw_combined = pd.concat(data.values(), axis=1, keys=data.keys())
    business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    macro_df = pd.DataFrame(index=business_days).join(raw_combined, how="outer")
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.ffill()
    macro_df = macro_df[macro_df.index.dayofweek < 5]
    return macro_df.dropna(how="any")


def collect_market_features(start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf

    print("yfinance market features collecting...")
    try:
        yf_data = yf.download(list(YF_TICKERS.values()), start=start_date, end=end_date)
        yf_close = yf_data["Close"] if isinstance(yf_data.columns, pd.MultiIndex) else yf_data
        yf_close = yf_close.rename(columns={value: key for key, value in YF_TICKERS.items()})
        yf_close.index = yf_close.index.tz_localize(None)
        return yf_close
    except Exception as exc:  # noqa: BLE001 - market data can fail independently.
        print(f"yfinance download error: {exc}")
        return pd.DataFrame()


def merge_macro_market(macro_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty:
        print("Warning: yfinance data is empty. Proceeding with macro data only.")
        return macro_df.copy()

    merged = macro_df.join(market_df, how="left")
    valid_market_cols = [col for col in YF_TICKERS if col in merged.columns]
    if valid_market_cols:
        merged[valid_market_cols] = merged[valid_market_cols].ffill()
    return merged.dropna(how="any")


def calc_normalized_slope(y_series: np.ndarray) -> float:
    if np.isnan(y_series).any():
        return np.nan

    x_values = np.arange(len(y_series))
    slope, _ = np.polyfit(x_values, y_series, 1)
    mean_abs = np.mean(np.abs(y_series))
    if mean_abs == 0:
        return 0.0
    return float(slope / mean_abs)


def expand_features(macro_market_df: pd.DataFrame) -> pd.DataFrame:
    print("Expanding features...")
    features = macro_market_df.copy()
    for column in MARKET_DAILY_WEEKLY_FEATURES:
        if column in features.columns:
            features[f"{column}_c20"] = features[column].rolling(window=20).apply(calc_normalized_slope, raw=True)
            features[f"{column}_c60"] = features[column].rolling(window=60).apply(calc_normalized_slope, raw=True)

    for column in MACRO_MONTHLY_FEATURES:
        if column in features.columns:
            features[f"{column}_m1"] = features[column].diff(21)
            features[f"{column}_m3"] = features[column].diff(63)

    return features.dropna(how="any")


def print_summary(df: pd.DataFrame) -> None:
    print(f"Total trading days (rows): {df.shape[0]}")
    print(f"Total features (cols): {df.shape[1]}")
    print(f"Start date: {df.index.min().strftime('%Y-%m-%d')}")
    print(f"End date: {df.index.max().strftime('%Y-%m-%d')}")


def main() -> None:
    args = parse_args()
    fred_api_key = Config().FRED_API_KEY
    if not fred_api_key:
        raise ValueError("FRED_API_KEY is required for live macro feature builds.")

    fred_data = collect_fred_series(args.start_date, args.end_date)
    alfred_data = collect_alfred_series(fred_api_key, args.start_date, args.alfred_sleep_seconds)
    macro_df = build_macro_frame({**fred_data, **alfred_data}, args.start_date, args.end_date)
    market_df = collect_market_features(macro_df.index.min().strftime("%Y-%m-%d"), args.end_date)
    quant_features = expand_features(merge_macro_market(macro_df, market_df))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quant_features.to_excel(args.output)
    print_summary(quant_features)


if __name__ == "__main__":
    main()
