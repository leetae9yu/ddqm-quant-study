"""Registry for EQR point-in-time feature families."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from .result import FeatureBuildResult


Builder = Callable[..., FeatureBuildResult]


def _registry() -> dict[str, Builder]:
    from .compustat_features import build_compustat_features
    from .crsp_features import build_crsp_features
    from .ibes_features import build_ibes_features
    from .macro_features import build_macro_features

    return {
        "macro": build_macro_features,
        "crsp": build_crsp_features,
        "compustat": build_compustat_features,
        "ibes": build_ibes_features,
    }


def enabled_feature_families(config: Mapping[str, Any] | None = None) -> list[str]:
    """Return enabled family names from config, defaulting to all families."""

    registry = _registry()
    if not config:
        return list(registry)

    raw = config.get("families", {})
    if not isinstance(raw, Mapping):
        return list(registry)
    return [name for name in registry if bool(raw.get(name, True))]


def available_feature_families() -> list[str]:
    """Return supported point-in-time feature family names."""

    return sorted(_registry())


def build_feature_families(
    *,
    panel: pd.DataFrame,
    inputs: Mapping[str, pd.DataFrame],
    families: Sequence[str] | None = None,
) -> FeatureBuildResult:
    """Build and left-join selected feature families onto the monthly panel."""

    registry = _registry()
    requested = list(families) if families is not None else list(registry)
    unknown = sorted(set(requested).difference(registry))
    if unknown:
        raise ValueError(f"Unknown feature families: {unknown}")

    result = panel.copy()
    metadata: list[dict[str, str]] = []
    for family in requested:
        built = registry[family](panel=panel, **inputs)
        feature_columns = [col for col in built.frame.columns if col not in {"permno", "formation_date"}]
        if feature_columns:
            result = result.merge(built.frame, on=["permno", "formation_date"], how="left")
            metadata.extend(built.metadata)

    return FeatureBuildResult(family="all", frame=result, metadata=metadata)


def feature_metadata_records(result: FeatureBuildResult) -> list[dict[str, str]]:
    """Return metadata after checking all non-key feature columns are described."""

    keys = {"permno", "formation_date"}
    feature_cols = [col for col in result.frame.columns if "__" in col and col not in keys]
    described = {record["feature"] for record in result.metadata}
    missing = sorted(set(feature_cols).difference(described))
    if missing:
        raise ValueError(f"Missing metadata for features: {missing}")
    return result.metadata
