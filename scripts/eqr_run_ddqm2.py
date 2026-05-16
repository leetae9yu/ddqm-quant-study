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
    backtest_stock_score_qspread,
    build_factor_allocations,
    build_factor_long_short_returns,
    build_factor_scores,
    macro_design_columns,
    select_factor_universe,
    train_factor_return_models,
)
from autoquant_lab.eqr.factors.definitions import implemented_factor_definitions  # noqa: E402
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
    parser.add_argument("--max-factor-score-rows", type=int, default=None, help="Safety cap on estimated long factor-score rows.")
    parser.add_argument("--factor-score-chunk-dates", type=int, default=None, help="Build factor scores/returns in formation-date chunks; <=0 disables chunking.")
    parser.add_argument("--factor-universe", choices=("all_implemented_current", "selected_13_global_local", "selected_13_plus_us_overrides"), default=None, help="Eligible factor universe for modeling/allocation.")
    parser.add_argument("--factor-universe-target-count", type=int, default=None, help="Target factor count for selected universes; defaults to 13.")
    parser.add_argument("--macro-feature-design", choices=("current_macro_family", "ddqm2_25x3_us_macro", "expanded_us_macro"), default=None, help="Macro feature design used by factor models.")
    parser.add_argument("--portfolio-surface", choices=("weighted_factor_return_current", "stock_score_qspread_ddqm2"), default=None, help="Portfolio surface to backtest.")
    parser.add_argument("--evaluation-mode", choices=("single_holdout", "walk_forward"), default=None, help="DDQM2 evaluation protocol; walk_forward gives expanding-window OOS predictions.")
    parser.add_argument("--walk-forward-test-periods", type=int, default=None, help="Number of monthly periods per walk-forward OOS fold.")
    parser.add_argument("--walk-forward-validation-periods", type=int, default=None, help="Number of recent periods held out for validation inside each walk-forward fold.")
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
    panel = load_prepared_panel(panel_path)
    features = _read_prepared_features(feature_dir)
    features[PERIOD_COLUMN] = pd.to_datetime(features[PERIOD_COLUMN], errors="coerce")
    return panel, features


def load_prepared_panel(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"Prepared panel does not exist: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel[PERIOD_COLUMN] = pd.to_datetime(panel[PERIOD_COLUMN], errors="coerce")
    if panel.duplicated([ID_COLUMN, PERIOD_COLUMN]).any():
        raise ValueError("Prepared panel has duplicate (permno, formation_date) keys")
    return panel


def _cap_panel(panel: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(panel) <= max_rows:
        return panel
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
        return sorted_panel.head(max_rows).copy()
    return sorted_panel.loc[sorted_panel[PERIOD_COLUMN].isin(chosen)].copy()


def _filter_features_to_panel_dates(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features[PERIOD_COLUMN] = pd.to_datetime(features[PERIOD_COLUMN], errors="coerce")
    chosen = [pd.Timestamp(date) for date in sorted(panel[PERIOD_COLUMN].dropna().unique()) if not pd.isna(date)]
    return features.loc[features[PERIOD_COLUMN].isin(chosen)].copy()


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
    summary: dict[str, float | int] = {
        "periods": int(len(portfolio)),
        "mean_monthly_return": float(returns.mean()),
        "volatility_monthly": float(returns.std(ddof=0)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }
    for column in ("long_count", "short_count", "turnover", "long_turnover", "short_turnover", "max_factor_weight", "mean_herfindahl_weight"):
        if column in portfolio.columns:
            summary[column] = float(pd.to_numeric(portfolio[column], errors="coerce").mean())
    return summary


def _portfolio_summaries_by_split(portfolio: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    return {split: _portfolio_summary(group.sort_values("formation_date")) for split, group in portfolio.groupby("split", sort=True)}


def _headline_portfolio_summary(portfolio: pd.DataFrame) -> dict[str, float | int]:
    if "split" in portfolio.columns and (portfolio["split"] == "holdout").any():
        return _portfolio_summary(portfolio.loc[portfolio["split"] == "holdout"].sort_values("formation_date"))
    return _portfolio_summary(portfolio.sort_values("formation_date"))


def _date_chunks(panel: pd.DataFrame, chunk_dates: int) -> list[list[pd.Timestamp]]:
    dates = [pd.Timestamp(date) for date in sorted(panel[PERIOD_COLUMN].dropna().unique()) if not pd.isna(date)]
    if chunk_dates <= 0:
        return [dates]
    return [dates[index : index + chunk_dates] for index in range(0, len(dates), chunk_dates)]


def _estimated_factor_score_rows(panel: pd.DataFrame, chunk_dates: int) -> int:
    factor_count = len(implemented_factor_definitions())
    if chunk_dates <= 0:
        return int(len(panel)) * factor_count
    max_panel_rows = 0
    for dates in _date_chunks(panel, chunk_dates):
        rows = int(panel[PERIOD_COLUMN].isin(dates).sum())
        max_panel_rows = max(max_panel_rows, rows)
    return max_panel_rows * factor_count


def _enforce_factor_score_budget(panel: pd.DataFrame, config: dict[str, Any], cli_limit: int | None, chunk_dates: int) -> None:
    ddqm2 = _ddqm2_config(config)
    limit = int(cli_limit if cli_limit is not None else ddqm2.get("max_factor_score_rows", 30_000_000))
    if limit <= 0:
        return
    estimated_rows = _estimated_factor_score_rows(panel, chunk_dates)
    if estimated_rows > limit:
        scope = "chunk" if chunk_dates > 0 else "total"
        message = (
            f"Estimated factor-score rows exceed safety cap: estimated={estimated_rows} limit={limit} scope={scope}. "
            "Lower --max-rows, lower --factor-score-chunk-dates, or raise --max-factor-score-rows only after checking memory."
        )
        raise ValueError(message)


def _build_factor_artifacts(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    return_column: str,
    quantile: float,
    run_dir: Path,
    chunk_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, list[str], pd.DataFrame]:
    if chunk_dates <= 0:
        factor_scores, metadata = build_factor_scores(features)
        factor_returns = build_factor_long_short_returns(factor_scores, panel, return_column=return_column, quantile=quantile)
        factor_scores.to_parquet(run_dir / "factor_scores.parquet", index=False)
        return metadata, factor_returns, int(len(factor_scores)), ["factor_scores.parquet"], factor_scores

    metadata = pd.DataFrame([definition.to_dict() for definition in implemented_factor_definitions()])
    score_dir = run_dir / "factor_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    factor_return_frames: list[pd.DataFrame] = []
    factor_score_rows = 0
    for chunk_index, dates in enumerate(_date_chunks(panel, chunk_dates)):
        chunk_panel = panel.loc[panel[PERIOD_COLUMN].isin(dates)].copy()
        chunk_features = features.loc[features[PERIOD_COLUMN].isin(dates)].copy()
        factor_scores, _ = build_factor_scores(chunk_features)
        factor_score_rows += int(len(factor_scores))
        factor_scores.to_parquet(score_dir / f"part-{chunk_index:04d}.parquet", index=False)
        factor_returns = build_factor_long_short_returns(factor_scores, chunk_panel, return_column=return_column, quantile=quantile)
        if not factor_returns.empty:
            factor_return_frames.append(factor_returns)

    if factor_return_frames:
        factor_returns = pd.concat(factor_return_frames, ignore_index=True).sort_values([PERIOD_COLUMN, "factor_id"]).reset_index(drop=True)
    else:
        factor_returns = pd.DataFrame()
    return metadata, factor_returns, factor_score_rows, ["factor_scores/"], pd.DataFrame()


def _load_selected_factor_scores(run_dir: Path, factor_score_artifacts: list[str], selected_factor_ids: set[str]) -> pd.DataFrame:
    if "factor_scores/" in factor_score_artifacts:
        score_paths = sorted((run_dir / "factor_scores").glob("part-*.parquet"))
    else:
        score_path = run_dir / "factor_scores.parquet"
        score_paths = [score_path] if score_path.exists() else []
    frames: list[pd.DataFrame] = []
    for path in score_paths:
        frame = pd.read_parquet(path)
        if "factor_id" not in frame.columns:
            continue
        selected = frame.loc[frame["factor_id"].isin(selected_factor_ids)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def _write_report(run_dir: Path, manifest: dict[str, Any], metrics: pd.DataFrame, portfolio_summary: dict[str, float | int]) -> None:
    top = metrics.sort_values("holdout_correlation", ascending=False, na_position="last").head(10) if not metrics.empty else pd.DataFrame()
    lines = [
        "# DDQM2 Factor-Return Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Model: `{manifest['model']}`",
        f"- Factor definitions used: {manifest['factor_definition_count']}",
        f"- Factor return rows: {manifest['factor_return_rows']}",
        "- Headline split: `holdout` when available; train/validation are diagnostics only.",
        f"- Portfolio periods: {portfolio_summary['periods']}",
        f"- Holdout cumulative return: {portfolio_summary['cumulative_return']:.6f}",
        f"- Holdout max drawdown: {portfolio_summary['max_drawdown']:.6f}",
        f"- Holdout turnover: {portfolio_summary.get('turnover', 0.0):.6f}",
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
            lines.append(
                f"| {row['factor_id']} | {_format_metric(row.get('validation_correlation'))} | "
                f"{_format_metric(row.get('holdout_correlation'))} | {_format_metric(row.get('holdout_mae'))} |"
            )
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
    factor_score_chunk_dates = int(args.factor_score_chunk_dates if args.factor_score_chunk_dates is not None else ddqm2.get("factor_score_chunk_dates", 0))
    factor_universe = str(args.factor_universe if args.factor_universe is not None else ddqm2.get("factor_universe", "all_implemented_current"))
    factor_universe_target_count = int(args.factor_universe_target_count if args.factor_universe_target_count is not None else ddqm2.get("factor_universe_target_count", 13))
    macro_feature_design = str(args.macro_feature_design if args.macro_feature_design is not None else ddqm2.get("macro_feature_design", "current_macro_family"))
    portfolio_surface = str(args.portfolio_surface if args.portfolio_surface is not None else ddqm2.get("portfolio_surface", "weighted_factor_return_current"))
    evaluation_mode = str(args.evaluation_mode if args.evaluation_mode is not None else ddqm2.get("evaluation_mode", "single_holdout"))
    walk_forward_test_periods = int(args.walk_forward_test_periods if args.walk_forward_test_periods is not None else ddqm2.get("walk_forward_test_periods", 12))
    walk_forward_validation_periods = int(args.walk_forward_validation_periods if args.walk_forward_validation_periods is not None else ddqm2.get("walk_forward_validation_periods", 12))

    panel = _cap_panel(load_prepared_panel(args.panel), args.max_rows)
    _enforce_factor_score_budget(panel, config, args.max_factor_score_rows, factor_score_chunk_dates)
    run_id = _safe_run_id(args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_ddqm2_{model_name}")
    run_dir = _safe_run_dir(args.output_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    features = _filter_features_to_panel_dates(_read_prepared_features(args.feature_dir), panel)
    metadata, factor_returns, factor_score_rows, factor_score_artifacts, factor_scores = _build_factor_artifacts(
        features,
        panel,
        return_column=args.return_column,
        quantile=quantile,
        run_dir=run_dir,
        chunk_dates=factor_score_chunk_dates,
    )
    all_definitions = implemented_factor_definitions()
    selected_definitions, selected_metadata = select_factor_universe(
        factor_returns,
        features,
        all_definitions,
        universe=factor_universe,
        target_count=factor_universe_target_count,
        overrides=ddqm2.get("factor_universe_overrides", []) if isinstance(ddqm2.get("factor_universe_overrides", []), list) else [],
        macro_feature_design=macro_feature_design,
    )
    selected_factor_ids = {definition.factor_id for definition in selected_definitions}
    if not selected_factor_ids:
        raise ValueError(f"Factor universe produced no runnable factors: {factor_universe}")
    metadata = selected_metadata if not selected_metadata.empty else metadata.loc[metadata["factor_id"].isin(selected_factor_ids)].copy()
    factor_returns = factor_returns.loc[factor_returns["factor_id"].isin(selected_factor_ids)].copy()
    if portfolio_surface == "stock_score_qspread_ddqm2":
        factor_scores = _load_selected_factor_scores(run_dir, factor_score_artifacts, selected_factor_ids)
    elif not factor_scores.empty:
        factor_scores = factor_scores.loc[factor_scores["factor_id"].isin(selected_factor_ids)].copy()
    result = train_factor_return_models(
        factor_returns,
        features,
        model_name=model_name,
        model_params=_model_params(config, model_name),
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
        min_observations=min_observations,
        evaluation_mode=evaluation_mode,
        walk_forward_test_periods=walk_forward_test_periods,
        walk_forward_validation_periods=walk_forward_validation_periods,
        macro_feature_design=macro_feature_design,
    )
    allocations = build_factor_allocations(result.predictions, min_weight=min_weight)
    if portfolio_surface == "weighted_factor_return_current":
        portfolio = backtest_factor_allocations(allocations, factor_returns)
        qspread_legs = pd.DataFrame()
    elif portfolio_surface == "stock_score_qspread_ddqm2":
        portfolio, qspread_legs = backtest_stock_score_qspread(allocations, factor_scores, panel, return_column=args.return_column, quantile=0.10)
    else:
        raise ValueError(f"Unsupported portfolio surface: {portfolio_surface}")
    portfolio_summary = _headline_portfolio_summary(portfolio)
    split_portfolio_summary = _portfolio_summaries_by_split(portfolio)
    macro_columns, macro_missing_columns = macro_design_columns(features, macro_feature_design)

    metadata.to_csv(run_dir / "factor_metadata.csv", index=False)
    factor_returns.to_parquet(run_dir / "factor_returns.parquet", index=False)
    result.predictions.to_parquet(run_dir / "factor_predictions.parquet", index=False)
    result.metrics.to_csv(run_dir / "factor_model_metrics.csv", index=False)
    allocations.to_parquet(run_dir / "factor_allocations.parquet", index=False)
    portfolio.to_parquet(run_dir / "portfolio_returns.parquet", index=False)
    if not qspread_legs.empty:
        qspread_legs.to_parquet(run_dir / "qspread_legs.parquet", index=False)
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
        "factor_universe": factor_universe,
        "factor_universe_target_count": factor_universe_target_count,
        "selected_factor_ids": sorted(selected_factor_ids),
        "macro_feature_design": macro_feature_design,
        "macro_feature_count": int(len(macro_columns)),
        "macro_feature_missing": macro_missing_columns,
        "portfolio_surface": portfolio_surface,
        "evaluation_mode": evaluation_mode,
        "walk_forward_test_periods": walk_forward_test_periods,
        "walk_forward_validation_periods": walk_forward_validation_periods,
        "factor_score_chunk_dates": factor_score_chunk_dates,
        "panel_rows": int(len(panel)),
        "feature_rows": int(len(features)),
        "factor_score_rows": factor_score_rows,
        "factor_definition_count": int(len(metadata)),
        "factor_return_rows": int(len(factor_returns)),
        "prediction_rows": int(len(result.predictions)),
        "portfolio_summary": portfolio_summary,
        "split_portfolio_summary": split_portfolio_summary,
        "artifacts": [
            *factor_score_artifacts,
            "factor_metadata.csv",
            "factor_returns.parquet",
            "factor_predictions.parquet",
            "factor_model_metrics.csv",
            "factor_allocations.parquet",
            "portfolio_returns.parquet",
            *([] if qspread_legs.empty else ["qspread_legs.parquet"]),
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
