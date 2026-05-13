#!/usr/bin/env python3
"""Run the DDQM2 macro-to-factor-return research track."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportUnusedCallResult=false, reportAssignmentType=false, reportOperatorIssue=false, reportIndexIssue=false

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from autoquant_lab.eqr.factors import (  # noqa: E402
    backtest_factor_allocations,
    build_factor_allocations,
    build_factor_long_short_returns,
    build_factor_scores,
    train_factor_return_models,
)
from autoquant_lab.eqr.models.registry import available_models  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "server_full.yaml"
DEFAULT_PANEL = PROJECT_ROOT / "experiments" / "prepared" / "panel" / "monthly_labels.parquet"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "experiments" / "prepared" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ddqm2"
PERIOD_COLUMN = "formation_date"
ID_COLUMN = "permno"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDQM2 factor-return modeling from local prepared artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config with optional ddqm2/model settings.")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL, help="Prepared monthly labels parquet.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR, help="Prepared feature parquet directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="DDQM2 runs root directory.")
    parser.add_argument("--model", choices=available_models(), default=None, help="Registered CPU model name; overrides config.")
    parser.add_argument("--run-id", default=None, help="Optional safe run id; defaults to UTC timestamp and model name.")
    parser.add_argument("--return-column", default="ret_1m_fwd", help="Stock forward-return column used to form factor labels.")
    parser.add_argument("--quantile", type=float, default=None, help="Long/short tail quantile for factor returns.")
    parser.add_argument("--max-rows", type=int, default=0, help="Chronological row cap for smoke runs; <=0 uses all rows.")
    parser.add_argument("--validation-fraction", type=float, default=None, help="Recent fraction used for validation.")
    parser.add_argument("--holdout-fraction", type=float, default=None, help="Most recent fraction used for holdout.")
    parser.add_argument("--min-observations", type=int, default=None, help="Minimum date observations per factor model.")
    parser.add_argument("--min-weight", type=float, default=None, help="Optional minimum non-negative factor weight.")
    return parser.parse_args()


def load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_prepared_features(feature_dir: Path) -> pd.DataFrame:
    if not feature_dir.exists():
        raise FileNotFoundError(f"Prepared feature directory does not exist: {feature_dir}")
    frames: list[pd.DataFrame] = []
    for path in sorted(feature_dir.glob("*.parquet")):
        if path.name.startswith("."):
            continue
        frame = pd.read_parquet(path)
        if {ID_COLUMN, PERIOD_COLUMN}.issubset(frame.columns):
            if frame.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
                raise ValueError(f"Prepared feature file has duplicate (permno, formation_date) keys: {path}")
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prepared feature parquet files with {ID_COLUMN}/{PERIOD_COLUMN} keys found in {feature_dir}")
    merged = frames[0]
    for frame in frames[1:]:
        new_columns = [col for col in frame.columns if col not in merged.columns or col in {ID_COLUMN, PERIOD_COLUMN}]
        merged = merged.merge(frame[new_columns], on=[ID_COLUMN, PERIOD_COLUMN], how="left")
        if merged.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
            raise ValueError("Prepared feature merge produced duplicate (permno, formation_date) keys")
    return merged


def load_research_frame(panel_path: Path, feature_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not panel_path.exists():
        raise FileNotFoundError(f"Prepared panel does not exist: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel[PERIOD_COLUMN] = pd.to_datetime(panel[PERIOD_COLUMN], errors="coerce")
    if panel.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
        raise ValueError("Prepared panel has duplicate (permno, formation_date) keys")
    features = _read_prepared_features(feature_dir)
    features[PERIOD_COLUMN] = pd.to_datetime(features[PERIOD_COLUMN], errors="coerce")
    return panel, features


def _smoke_cap(panel: pd.DataFrame, features: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_rows <= 0 or len(panel) <= max_rows:
        return panel, features
    sorted_panel = panel.sort_values([PERIOD_COLUMN, ID_COLUMN]).reset_index(drop=True)
    period_counts = sorted_panel.groupby(PERIOD_COLUMN, sort=True).size()
    chosen: list[pd.Timestamp] = []
    running = 0
    for period, count_value in period_counts.items():
        count = int(count_value)
        if running > 0 and running + count > max_rows and len(chosen) >= 6:
            break
        chosen.append(pd.Timestamp(period))
        running += count
    if len(chosen) < 6:
        capped_panel = sorted_panel.head(max_rows).copy()
        chosen = [pd.Timestamp(date) for date in sorted(capped_panel[PERIOD_COLUMN].dropna().unique()) if not pd.isna(date)]
    else:
        capped_panel = sorted_panel.loc[sorted_panel[PERIOD_COLUMN].isin(chosen)].copy()
    capped_features = features.loc[features[PERIOD_COLUMN].isin(chosen)].copy()
    return capped_panel, capped_features


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, underscore, dash, and dot")
    if run_id in {".", ".."}:
        raise ValueError("run_id must name a child run directory")
    return run_id


def _safe_run_dir(output_dir: Path, run_id: str) -> Path:
    output_root = output_dir.resolve()
    run_dir = (output_root / _safe_run_id(run_id)).resolve()
    if output_root != run_dir and output_root not in run_dir.parents:
        raise ValueError("run_id must stay within output_dir")
    return run_dir


def _model_params(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = config.get("models", {})
    if isinstance(models, dict):
        raw = models.get(model_name, {})
        if isinstance(raw, dict):
            return dict(raw.get("params", raw))
    model = config.get("model", {})
    if isinstance(model, dict) and model.get("name") == model_name:
        params = model.get("hyperparameters", {})
        if isinstance(params, dict):
            return dict(params)
    return {}


def _ddqm2_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("ddqm2", {})
    return dict(section) if isinstance(section, dict) else {}


def _portfolio_summary(portfolio: pd.DataFrame) -> dict[str, float | int]:
    if portfolio.empty:
        return {"periods": 0, "mean_monthly_return": 0.0, "volatility_monthly": 0.0, "cumulative_return": 0.0, "max_drawdown": 0.0}
    returns = pd.to_numeric(portfolio["portfolio_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "periods": int(len(portfolio)),
        "mean_monthly_return": float(returns.mean()),
        "volatility_monthly": float(returns.std(ddof=0)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def _write_report(run_dir: Path, manifest: dict[str, Any], metrics: pd.DataFrame, portfolio_summary: dict[str, float | int]) -> None:
    top = metrics.sort_values("holdout_correlation", ascending=False, na_position="last").head(10) if not metrics.empty else pd.DataFrame()
    lines = [
        "# DDQM2 Factor-Return Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Model: `{manifest['model']}`",
        f"- Factor definitions used: {manifest['factor_definition_count']}",
        f"- Factor return rows: {manifest['factor_return_rows']}",
        f"- Portfolio periods: {portfolio_summary['periods']}",
        f"- Cumulative return: {portfolio_summary['cumulative_return']:.6f}",
        f"- Max drawdown: {portfolio_summary['max_drawdown']:.6f}",
        "",
        "## Top holdout factor models",
        "",
    ]
    if top.empty:
        lines.append("No factor models met the observation requirements.")
    else:
        lines.append("| factor_id | validation_corr | holdout_corr | holdout_mae |")
        lines.append("|---|---:|---:|---:|")
        for row in top.to_dict("records"):
            lines.append(f"| {row['factor_id']} | {row.get('validation_correlation', np.nan):.6f} | {row.get('holdout_correlation', np.nan):.6f} | {row.get('holdout_mae', np.nan):.6f} |")
    lines.extend(
        [
            "",
            "## Data boundary",
            "",
            "This run reads local prepared market and macro artifacts only. It does not contact remote services during modeling.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_simple_yaml(args.config)
    ddqm2 = _ddqm2_config(config)
    model_name = args.model or str(ddqm2.get("model", config.get("model", {}).get("name", "ridge") if isinstance(config.get("model"), dict) else "ridge"))
    quantile = float(args.quantile if args.quantile is not None else ddqm2.get("quantile", 0.2))
    validation_fraction = float(args.validation_fraction if args.validation_fraction is not None else ddqm2.get("validation_fraction", 0.15))
    holdout_fraction = float(args.holdout_fraction if args.holdout_fraction is not None else ddqm2.get("holdout_fraction", 0.15))
    min_observations = int(args.min_observations if args.min_observations is not None else ddqm2.get("min_observations", 24))
    min_weight = float(args.min_weight if args.min_weight is not None else ddqm2.get("min_weight", 0.0))

    panel, features = load_research_frame(args.panel, args.feature_dir)
    panel, features = _smoke_cap(panel, features, args.max_rows)
    factor_scores, metadata = build_factor_scores(features)
    factor_returns = build_factor_long_short_returns(factor_scores, panel, return_column=args.return_column, quantile=quantile)
    result = train_factor_return_models(
        factor_returns,
        features,
        model_name=model_name,
        model_params=_model_params(config, model_name),
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
        min_observations=min_observations,
    )
    allocations = build_factor_allocations(result.predictions, min_weight=min_weight)
    portfolio = backtest_factor_allocations(allocations, factor_returns)
    portfolio_summary = _portfolio_summary(portfolio)

    run_id = _safe_run_id(args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_ddqm2_{model_name}")
    run_dir = _safe_run_dir(args.output_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    factor_scores.to_parquet(run_dir / "factor_scores.parquet", index=False)
    metadata.to_csv(run_dir / "factor_metadata.csv", index=False)
    factor_returns.to_parquet(run_dir / "factor_returns.parquet", index=False)
    result.predictions.to_parquet(run_dir / "factor_predictions.parquet", index=False)
    result.metrics.to_csv(run_dir / "factor_model_metrics.csv", index=False)
    allocations.to_parquet(run_dir / "factor_allocations.parquet", index=False)
    portfolio.to_parquet(run_dir / "portfolio_returns.parquet", index=False)
    (run_dir / "portfolio_summary.json").write_text(json.dumps(portfolio_summary, indent=2, default=_json_default), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "panel": str(args.panel),
        "feature_dir": str(args.feature_dir),
        "model": model_name,
        "return_column": args.return_column,
        "quantile": quantile,
        "validation_fraction": validation_fraction,
        "holdout_fraction": holdout_fraction,
        "min_observations": min_observations,
        "min_weight": min_weight,
        "panel_rows": int(len(panel)),
        "feature_rows": int(len(features)),
        "factor_score_rows": int(len(factor_scores)),
        "factor_definition_count": int(len(metadata)),
        "factor_return_rows": int(len(factor_returns)),
        "prediction_rows": int(len(result.predictions)),
        "portfolio_summary": portfolio_summary,
        "artifacts": [
            "factor_scores.parquet",
            "factor_metadata.csv",
            "factor_returns.parquet",
            "factor_predictions.parquet",
            "factor_model_metrics.csv",
            "factor_allocations.parquet",
            "portfolio_returns.parquet",
            "portfolio_summary.json",
            "report.md",
            "manifest.json",
        ],
        "data_boundary": "offline_local_artifacts_only_no_wrds_login_no_runtime_external_api",
    }
    _write_report(run_dir, manifest, result.metrics, portfolio_summary)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "portfolio_summary": portfolio_summary}, default=_json_default))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
