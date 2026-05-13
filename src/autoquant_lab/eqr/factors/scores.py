"""Build stock-level EQR factor scores from prepared feature panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .definitions import FactorDefinition, implemented_factor_definitions


def _standardize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return ((values - mean) / std).clip(-5.0, 5.0)


def _score_one_factor(frame: pd.DataFrame, definition: FactorDefinition) -> pd.DataFrame:
    if definition.source_column is None or definition.source_column not in frame.columns:
        return pd.DataFrame(columns=["formation_date", "permno", "factor_id", "factor_score", "raw_value", "scope", "status"])
    keys = ["formation_date", "permno", definition.source_column]
    if definition.scope == "local" and "exchcd" in frame.columns:
        keys.append("exchcd")
    work = frame[keys].copy()
    work["formation_date"] = pd.to_datetime(work["formation_date"], errors="coerce")
    group_keys = ["formation_date"] + (["exchcd"] if definition.scope == "local" and "exchcd" in work.columns else [])
    work["raw_value"] = pd.to_numeric(work[definition.source_column], errors="coerce")
    work["factor_score"] = work.groupby(group_keys, group_keys=False)["raw_value"].transform(_standardize) * definition.direction
    out = work.loc[work["formation_date"].notna() & work["permno"].notna(), ["formation_date", "permno", "factor_score", "raw_value"]].copy()
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    out["factor_id"] = definition.factor_id
    out["factor_name_ko"] = definition.name_ko
    out["factor_family"] = definition.family
    out["scope"] = definition.scope
    out["source_column"] = definition.source_column
    out["status"] = definition.status
    return out[["formation_date", "permno", "factor_id", "factor_name_ko", "factor_family", "scope", "source_column", "status", "raw_value", "factor_score"]]


def build_factor_scores(feature_panel: pd.DataFrame, definitions: tuple[FactorDefinition, ...] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long stock-level factor scores and definition metadata."""

    required = {"formation_date", "permno"}
    missing = sorted(required.difference(feature_panel.columns))
    if missing:
        raise ValueError(f"Feature panel missing required score keys: {missing}")
    definitions = definitions or implemented_factor_definitions()
    frames = [_score_one_factor(feature_panel, definition) for definition in definitions]
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metadata = pd.DataFrame([definition.to_dict() for definition in definitions])
    if not scores.empty:
        scores = scores.loc[scores["factor_score"].notna()].sort_values(["formation_date", "factor_id", "permno"]).reset_index(drop=True)
    return scores, metadata
