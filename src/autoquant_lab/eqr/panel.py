"""Monthly EQR security panel and forward-return label builders."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .pit import OPEN_END_DATE, filter_crsp_common_stocks
from .schemas import require_columns, require_unique_key, summarize_dataframe


DEFAULT_FORWARD_HORIZONS: tuple[int, ...] = (1, 3, 6)
MONTHLY_UNIVERSE_COLUMNS: tuple[str, ...] = (
    "permno",
    "permco",
    "formation_date",
    "price",
    "adjusted_price",
    "ret_1m",
    "retx_1m",
    "shrout",
    "vol",
    "market_cap",
    "shrcd",
    "exchcd",
)
LABEL_WINDOW_COLUMNS: tuple[str, ...] = ("forward_return_start", "forward_return_end")
PRIMARY_LABEL_COLUMN = "ret_1m_fwd"


def _coerce_monthly_crsp(crsp_monthly: pd.DataFrame) -> pd.DataFrame:
    required = ("permno", "permco", "date", "prc", "ret", "retx", "shrout", "vol", "cfacpr", "cfacshr")
    require_columns(crsp_monthly, required, "CRSP monthly data")

    monthly = crsp_monthly.copy()
    monthly["permno"] = pd.to_numeric(monthly["permno"], errors="coerce").astype("Int64")
    monthly["permco"] = pd.to_numeric(monthly["permco"], errors="coerce").astype("Int64")
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    for column in ("prc", "ret", "retx", "shrout", "vol", "cfacpr", "cfacshr"):
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
    return monthly.loc[monthly["permno"].notna() & monthly["date"].notna()].copy()


def _coerce_crsp_names(crsp_names: pd.DataFrame) -> pd.DataFrame:
    required = ("permno", "shrcd", "exchcd", "namedt", "nameenddt")
    require_columns(crsp_names, required, "CRSP names data")

    names = crsp_names.copy()
    names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
    names["shrcd"] = pd.to_numeric(names["shrcd"], errors="coerce").astype("Int64")
    names["exchcd"] = pd.to_numeric(names["exchcd"], errors="coerce").astype("Int64")
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameenddt"] = pd.to_datetime(names["nameenddt"], errors="coerce")
    names = filter_crsp_common_stocks(names)
    return names.loc[names["permno"].notna() & names["namedt"].notna()].copy()


def _date_filtered(df: pd.DataFrame, date_column: str, start_date: Any | None, end_date: Any | None) -> pd.DataFrame:
    result = df
    if start_date is not None:
        result = result.loc[result[date_column] >= pd.Timestamp(start_date)]
    if end_date is not None:
        result = result.loc[result[date_column] <= pd.Timestamp(end_date)]
    return result.copy()


def build_monthly_universe(
    crsp_monthly: pd.DataFrame,
    crsp_names: pd.DataFrame,
    *,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Filter CRSP monthly rows to active common stocks on major exchanges.

    The CRSP names interval is evaluated at each monthly observation date, avoiding
    current-membership or future-membership universe construction.
    """

    monthly = _date_filtered(_coerce_monthly_crsp(crsp_monthly), "date", start_date, end_date)
    names = _coerce_crsp_names(crsp_names)
    names["_nameenddt"] = names["nameenddt"].fillna(OPEN_END_DATE)

    joined = monthly.merge(
        names[["permno", "shrcd", "exchcd", "namedt", "_nameenddt"]],
        on="permno",
        how="inner",
    )
    active = joined.loc[(joined["namedt"] <= joined["date"]) & (joined["date"] <= joined["_nameenddt"])].copy()
    active = active.sort_values(["permno", "date", "namedt", "_nameenddt"]).drop_duplicates(["permno", "date"], keep="last")

    active["formation_date"] = active["date"]
    active["price"] = active["prc"].abs()
    active["adjusted_price"] = np.where(active["cfacpr"].notna() & (active["cfacpr"] != 0), active["price"] / active["cfacpr"], np.nan)
    active["ret_1m"] = active["ret"]
    active["retx_1m"] = active["retx"]
    active["market_cap"] = active["price"] * active["shrout"] * 1000.0

    panel = active.loc[:, MONTHLY_UNIVERSE_COLUMNS].copy()
    panel["permno"] = panel["permno"].astype("int64")
    panel["permco"] = panel["permco"].astype("Int64")
    require_unique_key(panel, ("permno", "formation_date"), "monthly universe")
    return panel.sort_values(["formation_date", "permno"]).reset_index(drop=True)


def _compound_forward_returns(group: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    ordered = group.sort_values("formation_date").reset_index(drop=True)
    returns = ordered["ret"].astype(float).to_numpy()
    dates = pd.to_datetime(ordered["formation_date"]).reset_index(drop=True)

    labels = pd.DataFrame({"permno": ordered["permno"].astype("int64"), "formation_date": dates})
    labels["forward_return_start"] = dates.shift(-1)
    labels["forward_return_end"] = dates.shift(-1)

    for horizon in horizons:
        values: list[float] = []
        end_dates: list[object] = []
        for idx in range(len(ordered)):
            window = returns[idx + 1 : idx + 1 + horizon]
            if len(window) != horizon or np.isnan(window).any():
                values.append(np.nan)
                end_dates.append(pd.NaT)
                continue
            values.append(float(np.prod(1.0 + window) - 1.0))
            end_dates.append(dates.iloc[idx + horizon])

        labels[f"ret_{horizon}m_fwd"] = values
        if horizon != 1:
            labels[f"forward_return_end_{horizon}m"] = end_dates

    return labels


def build_forward_returns(crsp_monthly: pd.DataFrame, horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS) -> pd.DataFrame:
    """Compute monthly forward-return labels from CRSP monthly returns.

    Labels use returns strictly after the formation month.  The primary label is
    ``ret_1m_fwd`` with explicit start/end windows; longer horizons are included
    when enough future monthly rows exist for a security.
    """

    if not horizons:
        raise ValueError("At least one forward-return horizon is required")
    if 1 not in set(horizons):
        raise ValueError("The 1M forward-return horizon is required")

    monthly = _coerce_monthly_crsp(crsp_monthly)
    monthly = monthly.loc[:, ["permno", "date", "ret"]].rename(columns={"date": "formation_date"})
    labels = pd.concat([_compound_forward_returns(group, tuple(sorted(set(horizons)))) for _, group in monthly.groupby("permno")], ignore_index=True)
    return labels.sort_values(["formation_date", "permno"]).reset_index(drop=True)


def _default_feature_columns(panel: pd.DataFrame) -> list[str]:
    metadata_and_label_columns = {
        "permno",
        "formation_date",
        "forward_return_start",
        "forward_return_end",
        "forward_return_end_3m",
        "forward_return_end_6m",
        "label_source",
        "universe_source",
    }
    return [column for column in panel.columns if column not in metadata_and_label_columns and not column.endswith("_fwd")]


def validate_labels(panel: pd.DataFrame, *, feature_columns: Sequence[str] | None = None) -> dict[str, object]:
    """Validate label windows and feature/label separation for a prepared panel."""

    required = ("permno", "formation_date", PRIMARY_LABEL_COLUMN, *LABEL_WINDOW_COLUMNS)
    require_columns(panel, required, "monthly label panel")

    validated = panel.copy()
    validated["formation_date"] = pd.to_datetime(validated["formation_date"], errors="coerce")
    validated["forward_return_start"] = pd.to_datetime(validated["forward_return_start"], errors="coerce")
    validated["forward_return_end"] = pd.to_datetime(validated["forward_return_end"], errors="coerce")

    bad_window = validated[PRIMARY_LABEL_COLUMN].notna() & (validated["forward_return_start"].isna() | (validated["forward_return_start"] <= validated["formation_date"]))
    if bad_window.any():
        examples = validated.loc[bad_window, ["permno", "formation_date", "forward_return_start"]].head(5).to_dict("records")
        raise ValueError(f"Forward label windows must start after formation_date; examples={examples}")

    features = list(feature_columns) if feature_columns is not None else _default_feature_columns(validated)
    forbidden = [column for column in features if column.endswith("_fwd") or column.startswith("forward_return") or column in LABEL_WINDOW_COLUMNS]
    if forbidden:
        raise ValueError(f"Feature columns contain label or forward-window data: {forbidden}")

    return {
        "valid": True,
        "feature_columns": features,
        "label_columns": [column for column in validated.columns if column.endswith("_fwd")],
        "summary": summarize_dataframe(validated, "formation_date"),
        "primary_label_non_null": int(validated[PRIMARY_LABEL_COLUMN].notna().sum()),
    }


def build_panel(
    crsp_monthly: pd.DataFrame,
    crsp_names: pd.DataFrame,
    *,
    start_date: Any | None = None,
    end_date: Any | None = None,
    horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
) -> pd.DataFrame:
    """Assemble the monthly security universe with forward-return labels."""

    universe = build_monthly_universe(crsp_monthly, crsp_names, start_date=start_date, end_date=end_date)
    labels = build_forward_returns(crsp_monthly, horizons=horizons)
    panel = universe.merge(labels, on=("permno", "formation_date"), how="left")
    panel["universe_source"] = "crsp_msf_active_names"
    panel["label_source"] = "crsp_msf_monthly_returns"
    validate_labels(panel)
    return panel.sort_values(["formation_date", "permno"]).reset_index(drop=True)


__all__ = [
    "DEFAULT_FORWARD_HORIZONS",
    "build_forward_returns",
    "build_monthly_universe",
    "build_panel",
    "validate_labels",
]
