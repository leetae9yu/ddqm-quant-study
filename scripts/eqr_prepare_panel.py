#!/usr/bin/env python3
"""Prepare monthly EQR panels from local offline raw artifacts."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false, reportUnusedImport=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyarrow.lib import ArrowNotImplementedError
import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from autoquant_lab.eqr.panel import (  # noqa: E402
    MONTHLY_UNIVERSE_COLUMNS,
    _coerce_monthly_crsp,
    _date_filtered,
    validate_labels,
)
from autoquant_lab.eqr.path_resolver import resolve_data_paths  # noqa: E402
from autoquant_lab.eqr.pit import filter_crsp_common_stocks  # noqa: E402
from autoquant_lab.eqr.features.feature_registry import (  # noqa: E402
    build_feature_families,
    enabled_feature_families,
    feature_metadata_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "golden_path.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "prepared" / "panel"
DEFAULT_FEATURE_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "prepared" / "features"
FEATURE_INPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "ccm_link": ("gvkey", "lpermno", "linkdt", "linkenddt", "usedflag", "linktype", "linkprim"),
    "comp_fundq": ("gvkey", "datadate", "rdq", "oiadpq", "niq", "ceqq", "revtq", "atq", "ltq", "dlttq", "dlcq", "cshoq", "prccq", "epspxq", "saleq"),
    "ibes_link": ("ticker", "permno", "sdate", "edate", "score"),
    "ibes_summary": ("ticker", "statpers", "meanest", "stdev", "numest", "actual", "actdats_act", "anndats_act", "measure"),
    "ibes_target": ("ticker", "statpers", "meanptg", "numup1m", "numdown1m", "numest"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EQR monthly security panel artifacts.")
    parser.add_argument("--stage", choices=("labels", "features"), required=True, help="Panel preparation stage to run.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Golden-path YAML config.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Local raw data directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prepared panel outputs.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional deterministic row cap for smoke-sized panel artifacts (0=all).")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _parse_scalar(value: str) -> Any:
    stripped = value.strip().strip('"').strip("'")
    if stripped == "":
        return ""
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped.startswith("[") and stripped.endswith("]"):
        items = [item.strip() for item in stripped[1:-1].split(",") if item.strip()]
        return [_parse_scalar(item) for item in items]
    try:
        return int(stripped)
    except ValueError:
        return stripped


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the golden-path config, including block-list YAML syntax."""

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        return loaded
    if loaded is not None:
        raise ValueError(f"Config root must be a mapping: {path}")

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if separator != ":":
            raise ValueError(f"Unsupported config line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _is_string_dtype(dtype: object) -> bool:
    """Return True for ``object`` or ``pd.StringDtype()`` columns."""
    return isinstance(dtype, pd.StringDtype) or dtype == object  # type: ignore[comparison-overlap]


def _coerce_common_types(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-convert string columns that hold numeric data to compact types.

    CRSP monthly data stores ``ret``, ``retx`` and ``vol`` as strings in the
    partitioned Parquet files, bloating the in-memory DataFrame to ~1.2 GB.
    Converting them to float64 *before* :func:`_coerce_monthly_crsp` makes its
    internal ``.copy()`` ~3× smaller and keeps peak memory under control.
    """
    for col in df.columns:
        if col in ("ret", "retx", "vol") and _is_string_dtype(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _downcast_floats(df)


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 → float32 and int64 → smaller types to reduce memory.

    CRSP monthly data easily exceeds 1 GB as float64; halving that with
    float32 makes the difference between OOM and a clean exit on memory-
    constrained hosts.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _read_parquet_artifact(path_or_files: Path | Sequence[Path], columns: Sequence[str] | None = None) -> pd.DataFrame:
    # Phase 1 – resolve which actual Parquet files to read.
    # Avoid calling pd.read_parquet on a directory because pyarrow may
    # trip over non-Parquet artefacts such as Windows Zone.Identifier
    # streams that have been copied into the tree.
    col_list = list(columns) if columns is not None else None
    if isinstance(path_or_files, Path):
        if path_or_files.is_dir():
            files = sorted(
                child
                for child in path_or_files.iterdir()
                if child.is_file() and child.suffix == ".parquet"
            )
            if not files:
                raise FileNotFoundError(
                    f"No parquet files found in directory: {path_or_files}"
                )
        else:
            return pd.read_parquet(path_or_files, columns=col_list)
    else:
        files = [f for f in path_or_files if f.suffix == ".parquet"]

    if not files:
        raise FileNotFoundError("No parquet files were provided for the artifact")

    # Phase 2 – read data efficiently.
    # Some partitioned Parquet files contain null-type columns (all-null
    # schemas in empty year partitions) that cause pyarrow's read_table to
    # fail with ArrowNotImplementedError when merging schemas.  Fall back
    # to per-file pandas reads + concat when that happens.
    if len(files) == 1:
        return pd.read_parquet(files[0], columns=col_list)

    try:
        # Convert Path objects to strings – pyarrow does not accept Path objects
        # when reading multiple files via read_table.
        table = pq.read_table([str(f) for f in files], columns=col_list)
        df = table.to_pandas()
    except ArrowNotImplementedError:
        # Schema mismatch from null-type columns – read individually.
        frames: list[pd.DataFrame] = []
        for f in files:
            frame = pd.read_parquet(f, columns=col_list)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

    # Pre-coerce string → numeric so downstream .copy() calls are cheaper.
    return _coerce_common_types(df)


def _build_universe_chunked(
    crsp_monthly: pd.DataFrame,
    crsp_names: pd.DataFrame,
    *,
    start_date: object = None,
    end_date: object = None,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Build the monthly security universe with a chunked merge.

    CRSP names can have multiple rows per ``permno`` (one per name period).
    A naive inner merge on ``permno`` blows up memory because each monthly
    row is duplicated for every name period of that security.  Processing
    permnos one chunk at a time keeps peak allocation low and avoids OOM.
    """
    from autoquant_lab.eqr.pit import OPEN_END_DATE

    monthly = _date_filtered(_coerce_monthly_crsp(crsp_monthly), "date", start_date, end_date)

    # Coerce names
    names = crsp_names.copy()
    names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
    names["shrcd"] = pd.to_numeric(names["shrcd"], errors="coerce").astype("Int64")
    names["exchcd"] = pd.to_numeric(names["exchcd"], errors="coerce").astype("Int64")
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameenddt"] = pd.to_datetime(names["nameenddt"], errors="coerce")
    names = filter_crsp_common_stocks(names)
    names = names.loc[names["permno"].notna() & names["namedt"].notna()].copy()
    names["_nameenddt"] = names["nameenddt"].fillna(OPEN_END_DATE)
    n_subset = names[["permno", "shrcd", "exchcd", "namedt", "_nameenddt"]]
    del names

    # Chunked merge
    permnos = pd.Index(monthly["permno"].unique()).intersection(pd.Index(n_subset["permno"].unique()))
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(permnos), chunk_size):
        chunk_permnos = permnos[i : i + chunk_size]
        m_chunk = monthly[monthly["permno"].isin(chunk_permnos)]
        n_chunk = n_subset[n_subset["permno"].isin(chunk_permnos)]
        joined = m_chunk.merge(n_chunk, on="permno", how="inner")
        active = joined.loc[
            (joined["namedt"] <= joined["date"]) & (joined["date"] <= joined["_nameenddt"])
        ].copy()
        active = active.sort_values(["permno", "date", "namedt", "_nameenddt"]).drop_duplicates(
            ["permno", "date"], keep="last"
        )
        chunks.append(active)

    active = pd.concat(chunks, ignore_index=True)
    del chunks

    # Build universe columns (same as build_monthly_universe)
    active["formation_date"] = active["date"]
    active["price"] = active["prc"].abs()
    active["adjusted_price"] = np.where(
        active["cfacpr"].notna() & (active["cfacpr"] != 0),
        active["price"] / active["cfacpr"],
        np.nan,
    )
    active["ret_1m"] = active["ret"]
    active["retx_1m"] = active["retx"]
    active["market_cap"] = active["price"] * active["shrout"] * 1000.0

    panel = active.loc[:, MONTHLY_UNIVERSE_COLUMNS].copy()
    panel["permno"] = panel["permno"].astype("int64")
    panel["permco"] = panel["permco"].astype("Int64")
    return panel.sort_values(["formation_date", "permno"]).reset_index(drop=True)


def run_labels(args: argparse.Namespace) -> dict[str, Any]:
    config = load_simple_yaml(args.config)
    panel_config = config.get("panel", {}) if isinstance(config.get("panel", {}), dict) else {}
    date_config = panel_config.get("date_range", {}) if isinstance(panel_config.get("date_range", {}), dict) else {}
    horizons = tuple(panel_config.get("forward_horizons", [1, 3, 6]))

    resolved = resolve_data_paths(args.data_dir, include_date_ranges=False)
    missing = sorted({"crsp_monthly", "crsp_names"}.difference(resolved))
    if missing:
        raise FileNotFoundError(f"Missing required raw artifacts: {missing}")

    crsp_monthly = _read_parquet_artifact(resolved["crsp_monthly"].files)
    crsp_names = _read_parquet_artifact(resolved["crsp_names"].files)

    # Build universe – chunked merge avoids OOM
    universe = _build_universe_chunked(
        crsp_monthly,
        crsp_names,
        start_date=date_config.get("start"),
        end_date=date_config.get("end"),
    )

    # Build forward-return labels
    from autoquant_lab.eqr.panel import build_forward_returns

    labels = build_forward_returns(crsp_monthly, horizons=horizons)

    # Merge universe + labels
    panel = universe.merge(labels, on=("permno", "formation_date"), how="left")
    panel["universe_source"] = "crsp_msf_active_names"
    panel["label_source"] = "crsp_msf_monthly_returns"
    if args.max_rows > 0:
        panel = panel.sort_values(["formation_date", "permno"]).head(args.max_rows).copy()

    validation = validate_labels(panel)
    validation["inputs"] = {
        "crsp_monthly": resolved["crsp_monthly"].to_dict(),
        "crsp_names": resolved["crsp_names"].to_dict(),
    }
    validation["limitations"] = ["No daily-only labels were created.", "Delisting returns are not fabricated when unavailable in current monthly data."]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "monthly_labels.parquet"
    validation_path = args.output_dir / "monthly_labels_validation.json"
    panel.to_parquet(panel_path, index=False)
    validation["output_panel"] = str(panel_path)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return {"panel_path": str(panel_path), "validation_path": str(validation_path), "rows": int(len(panel))}


def _artifact_frame(resolved: dict[str, Any], artifact_name: str, columns: Sequence[str] | None = None) -> pd.DataFrame | None:
    artifact = resolved.get(artifact_name)
    if artifact is None:
        return None
    selected_columns = columns or FEATURE_INPUT_COLUMNS.get(artifact_name)
    return _read_parquet_artifact(artifact.files, columns=selected_columns)


def _filter_feature_inputs_to_panel(inputs: dict[str, pd.DataFrame], panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce raw feature inputs to the smoke panel's identifiers and date window."""

    if panel.empty:
        return inputs
    result = {name: frame.copy() for name, frame in inputs.items()}
    panel_permnos = set(pd.to_numeric(panel["permno"], errors="coerce").dropna().astype("int64"))
    panel_dates = pd.to_datetime(panel["formation_date"], errors="coerce")
    min_date = panel_dates.min() - pd.Timedelta(days=730)
    max_date = panel_dates.max()

    ccm_link = result.get("ccm_link")
    if ccm_link is not None and not ccm_link.empty and "lpermno" in ccm_link.columns:
        linked_permnos = pd.to_numeric(ccm_link["lpermno"], errors="coerce")
        result["ccm_link"] = ccm_link.loc[linked_permnos.isin(panel_permnos)].copy()

    comp_fundq = result.get("comp_fundq")
    ccm_link = result.get("ccm_link")
    if comp_fundq is not None and not comp_fundq.empty:
        comp = comp_fundq.copy()
        comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
        if ccm_link is not None and not ccm_link.empty and "gvkey" in ccm_link.columns:
            gvkeys = set(ccm_link["gvkey"].astype(str).str.zfill(6))
            comp = comp.loc[comp["gvkey"].astype(str).str.zfill(6).isin(gvkeys)]
        result["comp_fundq"] = comp.loc[comp["datadate"].between(min_date, max_date, inclusive="both")].copy()

    ibes_link = result.get("ibes_link")
    if ibes_link is not None and not ibes_link.empty and "permno" in ibes_link.columns:
        ibes_permnos = pd.to_numeric(ibes_link["permno"], errors="coerce")
        result["ibes_link"] = ibes_link.loc[ibes_permnos.isin(panel_permnos)].copy()

    ibes_link = result.get("ibes_link")
    ibes_tickers: set[str] = set()
    if ibes_link is not None and not ibes_link.empty and "ticker" in ibes_link.columns:
        ibes_tickers = set(ibes_link["ticker"].astype(str).str.strip().str.upper())
    for name in ("ibes_summary", "ibes_target"):
        frame = result.get(name)
        if frame is None or frame.empty:
            continue
        narrowed = frame.copy()
        narrowed["statpers"] = pd.to_datetime(narrowed["statpers"], errors="coerce")
        if ibes_tickers:
            narrowed = narrowed.loc[narrowed["ticker"].astype(str).str.strip().str.upper().isin(ibes_tickers)]
        result[name] = narrowed.loc[narrowed["statpers"].between(min_date, max_date, inclusive="both")].copy()

    return result


def run_features(args: argparse.Namespace) -> dict[str, Any]:
    config = load_simple_yaml(args.config)
    features_config = config.get("features", {}) if isinstance(config.get("features", {}), dict) else {}
    families = enabled_feature_families(features_config)

    panel_path = PROJECT_ROOT / "experiments" / "prepared" / "panel" / "monthly_labels.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Monthly label panel is required before feature build: {panel_path}")
    panel = pd.read_parquet(panel_path)
    if args.max_rows > 0:
        panel = panel.sort_values(["formation_date", "permno"]).head(args.max_rows).copy()

    resolved = resolve_data_paths(args.data_dir, include_date_ranges=False)
    required_by_family = {
        "macro": {"macro_features"},
        "compustat": {"comp_fundq", "ccm_link"},
        "ibes": {"ibes_summary", "ibes_link"},
        "crsp": set(),
    }
    missing_by_family: dict[str, list[str]] = {}
    active_families: list[str] = []
    for family in families:
        missing = sorted(required_by_family.get(family, set()).difference(resolved))
        if missing:
            missing_by_family[family] = missing
        else:
            active_families.append(family)

    # Load artifacts incrementally to keep peak memory manageable.
    # Each family's required data is loaded just before building and
    # released after merging, so we never hold all raw tables at once.
    import gc

    from autoquant_lab.eqr.features.feature_registry import _registry

    registry = _registry()
    result = panel.copy()
    all_metadata: list[dict[str, str]] = []

    for family in active_families:
        family_required = required_by_family.get(family, set())
        family_inputs: dict[str, pd.DataFrame] = {}
        for artifact_name in sorted(family_required):
            frame = _artifact_frame(resolved, artifact_name)
            if frame is not None:
                family_inputs[artifact_name] = frame
        # IBES target is optional and only needed for the ibes family.
        if family == "ibes":
            target = _artifact_frame(resolved, "ibes_target")
            if target is not None:
                family_inputs["ibes_target"] = target
        family_inputs = _filter_feature_inputs_to_panel(family_inputs, panel)

        built = registry[family](panel=panel, **family_inputs)
        feature_columns = [col for col in built.frame.columns if col not in {"permno", "formation_date"}]
        if feature_columns:
            result = result.merge(built.frame, on=["permno", "formation_date"], how="left")
            all_metadata.extend(built.metadata)

        del family_inputs
        gc.collect()

    # Validate metadata completeness.
    from autoquant_lab.eqr.features.result import FeatureBuildResult

    built_result = FeatureBuildResult(family="all", frame=result, metadata=all_metadata)
    metadata = feature_metadata_records(built_result)

    output_dir = DEFAULT_FEATURE_OUTPUT_DIR if args.output_dir == DEFAULT_OUTPUT_DIR else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "monthly_features.parquet"
    metadata_path = output_dir / "monthly_features_metadata.json"
    built_result.frame.to_parquet(feature_path, index=False)
    payload = {
        "output_panel": str(feature_path),
        "rows": int(len(built_result.frame)),
        "enabled_families": active_families,
        "skipped_families": missing_by_family,
        "feature_count": int(len(metadata)),
        "features": metadata,
        "inputs": {name: resolved[name].to_dict() for name in resolved if name in required_by_family},
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return {"feature_path": str(feature_path), "metadata_path": str(metadata_path), "rows": int(len(built_result.frame)), "feature_count": int(len(metadata))}


def main() -> int:
    args = parse_args()
    if args.stage == "labels":
        result = run_labels(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.stage == "features":
        result = run_features(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
