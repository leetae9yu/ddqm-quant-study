#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportExplicitAny=false, reportMissingImports=false, reportMissingTypeStubs=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Read-only Streamlit dashboard for DDQM2-lite smoke artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "experiments" / "smoke_lgbm"
DEFAULT_FACTOR_RETURNS = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "factor_long_short_returns_smoke.parquet"
DEFAULT_MODEL_DATASET = PROJECT_ROOT / "prototypes" / "yfinance_sp500" / "macro_factor_model_ready_smoke.parquet"
REQUIRED_EXPERIMENT_FILES = (
    "metrics.json",
    "manifest.json",
    "predictions.parquet",
    "feature_importance.csv",
)
PROTOTYPE_CAVEAT = (
    "DDQM2-lite prototype 전용 화면입니다. 이 대시보드는 public yfinance/current-membership smoke artifact만 읽습니다. "
    "생존편향이 제거된 연구용 성과, 실제 매매 가능 성과, WRDS 기반 최종 결과로 해석하면 안 됩니다."
)
SMOKE_COMMANDS = (
    "PYTHONPATH=src python scripts/build_yfinance_price_panel.py --tickers AAPL MSFT SPY --start-date 2020-01-01 "
    "--end-date 2020-06-30 --output prototypes/yfinance_sp500/canonical_price_panel_smoke.parquet\n"
    "PYTHONPATH=src python scripts/validate_yfinance_price_panel.py prototypes/yfinance_sp500/canonical_price_panel_smoke.parquet\n"
    "PYTHONPATH=src python scripts/build_yfinance_factor_scores.py --price-panel prototypes/yfinance_sp500/canonical_price_panel_smoke.parquet "
    "--output prototypes/yfinance_sp500/factor_scores_smoke.parquet --smoke\n"
    "PYTHONPATH=src python scripts/validate_factor_scores.py prototypes/yfinance_sp500/factor_scores_smoke.parquet --smoke\n"
    "PYTHONPATH=src python scripts/build_factor_long_short_returns.py --factor-scores prototypes/yfinance_sp500/factor_scores_smoke.parquet "
    "--price-panel prototypes/yfinance_sp500/canonical_price_panel_smoke.parquet --output prototypes/yfinance_sp500/factor_long_short_returns_smoke.parquet --smoke\n"
    "PYTHONPATH=src python scripts/validate_factor_long_short_returns.py prototypes/yfinance_sp500/factor_long_short_returns_smoke.parquet --smoke\n"
    "PYTHONPATH=src python scripts/assemble_macro_factor_dataset.py --factor-returns prototypes/yfinance_sp500/factor_long_short_returns_smoke.parquet "
    "--macro-workbook expanded_macro_market_features.xlsx --output prototypes/yfinance_sp500/macro_factor_model_ready_smoke.parquet\n"
    "PYTHONPATH=src python scripts/validate_macro_factor_dataset.py prototypes/yfinance_sp500/macro_factor_model_ready_smoke.parquet\n"
    "PYTHONPATH=src python scripts/train_macro_factor_lgbm_baseline.py --input prototypes/yfinance_sp500/macro_factor_model_ready_smoke.parquet "
    "--output-dir prototypes/yfinance_sp500/experiments/smoke_lgbm --smoke\n"
    "PYTHONPATH=src python scripts/validate_macro_factor_experiment.py prototypes/yfinance_sp500/experiments/smoke_lgbm\n"
    "PYTHONPATH=src streamlit run apps/ddqm2_lite_dashboard.py"
)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"Missing JSON artifact: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON artifact {path}: {exc}"
    except OSError as exc:
        return None, f"Could not read JSON artifact {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"JSON artifact must contain an object: {path}"
    return payload, None


def read_table(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path), None
        if path.suffix == ".csv":
            return pd.read_csv(path), None
    except FileNotFoundError:
        return None, f"Missing table artifact: {path}"
    except (OSError, ValueError, ImportError) as exc:
        return None, f"Could not read table artifact {path}: {exc}"
    return None, f"Unsupported table extension for {path}; expected .parquet or .csv"


def as_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def show_missing_artifacts(experiment_dir: Path, factor_returns_path: Path, model_dataset_path: Path) -> None:
    expected = [experiment_dir / filename for filename in REQUIRED_EXPERIMENT_FILES]
    expected.extend([factor_returns_path, model_dataset_path])
    st.error("필요한 DDQM2-lite artifact가 없거나 읽을 수 없습니다. 이 dashboard는 read-only라서 파일을 대신 생성하지 않습니다.")
    st.markdown("**필요한 파일**")
    st.code("\n".join(relative_label(path) for path in expected), language="text")
    st.markdown("**dashboard 밖에서 artifact를 생성하는 smoke command**")
    st.code(SMOKE_COMMANDS, language="bash")


def metrics_to_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation"):
        split_metrics = metrics.get(split_name, {})
        if not isinstance(split_metrics, dict):
            continue
        for model_name, model_metrics in split_metrics.items():
            if isinstance(model_metrics, dict):
                row = {"split": split_name, "model": model_name}
                row.update(model_metrics)
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    metric_columns = [column for column in frame.columns if column not in {"split", "model"}]
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def summarize_manifest(manifest: dict[str, Any]) -> pd.DataFrame:
    row_counts = manifest.get("row_counts", {}) if isinstance(manifest.get("row_counts"), dict) else {}
    fields = {
        "run_id": manifest.get("run_id"),
        "created_at_utc": manifest.get("created_at_utc"),
        "split_method": manifest.get("split_method"),
        "train_dates": f"{manifest.get('train_start_date', 'n/a')} to {manifest.get('train_end_date', 'n/a')}",
        "validation_dates": f"{manifest.get('validation_start_date', 'n/a')} to {manifest.get('validation_end_date', 'n/a')}",
        "rows_train": row_counts.get("train"),
        "rows_validation": row_counts.get("validation"),
        "factor_count": manifest.get("factor_count"),
        "feature_count": len(manifest.get("feature_columns", [])) if isinstance(manifest.get("feature_columns"), list) else None,
        "input_artifact_path": manifest.get("input_artifact_path"),
    }
    return pd.DataFrame([fields])


def validation_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    if "formation_date" in data.columns:
        data["formation_date"] = pd.to_datetime(data["formation_date"], errors="coerce")
    if "split" in data.columns:
        valid = data.loc[data["split"].astype(str).eq("validation"), :].copy()
        return valid if not valid.empty else data
    return data


def regression_metrics(group: pd.DataFrame) -> pd.Series:
    actual = pd.to_numeric(group["target_long_short_return"], errors="coerce")
    predicted = pd.to_numeric(group["prediction"], errors="coerce")
    usable = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if usable.empty:
        return pd.Series({"rows": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_ic": np.nan})
    residual = usable["actual"] - usable["predicted"]
    total_sum_squares = float(((usable["actual"] - usable["actual"].mean()) ** 2).sum())
    residual_sum_squares = float((residual**2).sum())
    r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan
    ic = usable["actual"].corr(usable["predicted"]) if usable["actual"].nunique() > 1 and usable["predicted"].nunique() > 1 else np.nan
    return pd.Series(
        {
            "rows": int(len(usable)),
            "mae": float(residual.abs().mean()),
            "rmse": float(np.sqrt((residual**2).mean())),
            "r2": r2,
            "pearson_ic": ic,
        }
    )


def factor_metric_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"factor_name", "target_long_short_return", "prediction"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    valid = validation_predictions(predictions)
    return valid.groupby("factor_name", dropna=False, observed=False).apply(regression_metrics, include_groups=False).reset_index()


def date_summary(df: pd.DataFrame, column: str) -> tuple[str, str]:
    if column not in df.columns:
        return "n/a", "n/a"
    date_series = pd.to_datetime(df[column], errors="coerce")
    if not bool(date_series.notna().any()):
        return "n/a", "n/a"
    return date_series.min().strftime("%Y-%m-%d"), date_series.max().strftime("%Y-%m-%d")


def numeric_min(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce")
    finite_values = values.dropna().to_numpy(dtype=float)
    return float(np.min(finite_values)) if finite_values.size else float("nan")


def coverage_table(model_dataset: pd.DataFrame | None, factor_returns: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if model_dataset is not None and not model_dataset.empty:
        start_date, end_date = date_summary(model_dataset, "formation_date")
        macro_columns = [column for column in model_dataset.columns if str(column).startswith("macro__")]
        rows.append(
            {
                "artifact": "macro_factor_model_ready",
                "rows": len(model_dataset),
                "columns": model_dataset.shape[1],
                "factors": model_dataset["factor_name"].nunique() if "factor_name" in model_dataset.columns else np.nan,
                "start_date": start_date,
                "end_date": end_date,
                "missing_values": int(model_dataset.isna().sum().sum()),
                "macro_features": len(macro_columns),
                "min_long_count": np.nan,
                "min_short_count": np.nan,
            }
        )
    if factor_returns is not None and not factor_returns.empty:
        start_date, end_date = date_summary(factor_returns, "formation_date")
        rows.append(
            {
                "artifact": "factor_long_short_returns",
                "rows": len(factor_returns),
                "columns": factor_returns.shape[1],
                "factors": factor_returns["factor_name"].nunique() if "factor_name" in factor_returns.columns else np.nan,
                "start_date": start_date,
                "end_date": end_date,
                "missing_values": int(factor_returns.isna().sum().sum()),
                "macro_features": np.nan,
                "min_long_count": numeric_min(factor_returns, "long_count"),
                "min_short_count": numeric_min(factor_returns, "short_count"),
            }
        )
    return pd.DataFrame(rows)


def missingness_by_column(df: pd.DataFrame | None, artifact_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    missing = df.isna().sum().sort_values(ascending=False)
    return pd.DataFrame({"artifact": artifact_name, "column": missing.index, "missing_values": missing.values})


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at top left, #edf7f3 0, #f7f3e8 34%, #fbfaf6 72%); }
        .block-container { padding-top: 2.25rem; }
        [data-testid="stMetricValue"] { color: #173b36; }
        div[data-testid="stAlert"] { border-radius: 1rem; }
        h1, h2, h3 { letter-spacing: -0.035em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_usage_guide() -> None:
    st.markdown(
        """
        ### 보는 순서

        이 화면은 **DDQM2-lite smoke 실험 결과를 읽기 위한 안내판**입니다. 새 실험을 돌리거나 데이터를 다운로드하지 않고,
        이미 만들어진 artifact만 읽어서 요약합니다.

        1. **실험 요약**에서 run id, 학습/검증 기간, factor 수, feature 수를 먼저 확인합니다.
        2. **LightGBM vs naive baseline**에서 모델이 단순 기준선보다 나은지 대략 봅니다. smoke test라 성과 결론은 내리지 않습니다.
        3. **Factor별 검증 진단**에서 어떤 factor에서 오차가 큰지 확인합니다.
        4. **Long-short factor return**에서 factor label 자체가 어떻게 움직였는지 봅니다.
        5. **Prediction 진단**에서 예측값과 실제 factor return의 관계, residual 분포를 확인합니다.
        6. **Feature importance**에서 어떤 macro feature가 LightGBM split/gain에 많이 쓰였는지 봅니다.
        7. **데이터 커버리지**에서 결측, basket count, artifact row 수를 확인해 실험 신뢰도를 점검합니다.

        #### 해석할 때 주의할 점

        - 지금 결과는 **pipeline 연결 확인용 smoke 결과**입니다.
        - yfinance/current-membership 기반이라 S&P 500 생존편향을 제거하지 못합니다.
        - WRDS/CRSP/Compustat/IBES 기반 연구용 pipeline이 완성되기 전까지는 성과 수치보다 **흐름과 artifact 품질**을 보는 것이 목적입니다.
        - dashboard는 read-only입니다. 경로를 바꿔 다른 artifact를 읽을 수는 있지만, 파일을 생성하거나 수정하지 않습니다.
        """
    )


def render_pipeline_map() -> None:
    st.markdown(
        """
        ### 파이프라인 지도

        ```text
        yfinance canonical price panel
        → price/volume factor score
        → factor long-short return label
        → FRED/ALFRED macro-factor dataset
        → LightGBM baseline experiment artifact
        → 이 read-only dashboard
        ```
        """
    )


def first_metric_value(metrics_frame: pd.DataFrame, model_name: str, metric_name: str) -> float | None:
    if metrics_frame.empty or metric_name not in metrics_frame.columns:
        return None
    matching = metrics_frame.loc[
        metrics_frame["split"].astype(str).eq("validation") & metrics_frame["model"].astype(str).str.contains(model_name, case=False, na=False),
        metric_name,
    ].dropna()
    if matching.empty:
        return None
    return float(matching.iloc[0])


def format_optional_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "확인 불가"
    return f"{value:.{digits}f}"


def render_plain_language_explanation(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    factor_returns: pd.DataFrame | None,
    model_dataset: pd.DataFrame | None,
) -> None:
    st.markdown("### 한 줄 요약")
    st.info(
        "이 화면은 '매크로 환경을 보고 어떤 factor long-short return이 좋아질지 LightGBM으로 맞춰보는' "
        "DDQM2-lite smoke 실험 결과를 쉽게 읽기 위한 화면입니다. 숫자는 성과 결론이 아니라 파이프라인 연결 확인용입니다.",
        icon="💡",
    )

    row_counts = manifest.get("row_counts", {}) if isinstance(manifest.get("row_counts"), dict) else {}
    feature_count = len(manifest.get("feature_columns", [])) if isinstance(manifest.get("feature_columns"), list) else None
    plain_cols = st.columns(4)
    plain_cols[0].metric("무엇을 예측?", "Factor 수익률")
    plain_cols[1].metric("검증 샘플", str(row_counts.get("validation", "n/a")))
    plain_cols[2].metric("Factor 개수", str(manifest.get("factor_count", "n/a")))
    plain_cols[3].metric("Macro feature", str(feature_count if feature_count is not None else "n/a"))

    metrics_frame = metrics_to_frame(metrics)
    lightgbm_rmse = first_metric_value(metrics_frame, "lightgbm", "rmse")
    zero_rmse = first_metric_value(metrics_frame, "zero", "rmse")
    mean_rmse = first_metric_value(metrics_frame, "mean", "rmse")
    rmse_text = format_optional_float(lightgbm_rmse)
    baseline_text = ", ".join(
        item for item in (f"zero={format_optional_float(zero_rmse)}" if zero_rmse is not None else "", f"mean={format_optional_float(mean_rmse)}" if mean_rmse is not None else "") if item
    )
    st.markdown(
        f"""
        ### 결과를 아주 쉽게 읽으면

        - **모델이 하려는 일**: 매월 macro feature를 보고 factor별 long-short return을 예측합니다.
        - **LightGBM 검증 RMSE**: `{rmse_text}`입니다. RMSE는 낮을수록 좋습니다.
        - **단순 baseline**: {baseline_text or "artifact에서 확인 불가"}. LightGBM이 이 값보다 낮아야 최소한 의미 있는 비교가 됩니다.
        - **중요한 caveat**: 지금은 yfinance 기반 smoke라서 실제 투자 성과가 아니라, 데이터 → factor → label → model → dashboard 연결 확인입니다.
        """
    )

    if factor_returns is not None and not factor_returns.empty and "long_short_return" in factor_returns.columns:
        target = pd.to_numeric(factor_returns["long_short_return"], errors="coerce").dropna()
        if not target.empty:
            st.markdown(
                f"- **맞히려는 target의 평균**은 `{target.mean():.4f}`, 변동성은 `{target.std():.4f}`입니다. "
                "target 자체가 작고 출렁이면 모델 성능도 불안정하게 보일 수 있습니다."
            )

    if not importance.empty and {"feature", "importance", "importance_type"}.issubset(importance.columns):
        gain_importance = importance.loc[importance["importance_type"].astype(str).eq("gain"), ["feature", "importance"]].copy()
        if not gain_importance.empty:
            gain_importance["importance"] = pd.to_numeric(gain_importance["importance"], errors="coerce")
            top_features = gain_importance.sort_values("importance", ascending=False).head(5)["feature"].astype(str).tolist()
            st.markdown("- **모델이 많이 본 macro feature Top 5**: " + ", ".join(top_features))

    if model_dataset is not None and not model_dataset.empty:
        total_missing = int(model_dataset.isna().sum().sum())
        st.markdown(f"- **model dataset 결측치 총합**: `{total_missing}`개입니다. 결측이 많으면 먼저 데이터 품질을 의심해야 합니다.")

    with st.expander("용어를 더 쉽게 풀어보기", expanded=False):
        st.markdown(
            """
            - **Factor**: momentum, volatility, liquidity 같은 주식 특성 점수입니다.
            - **Long-short return**: factor 점수가 높은 종목 묶음을 사고 낮은 종목 묶음을 판 것처럼 계산한 수익률입니다.
            - **Macro feature**: 금리, 신용스프레드, 지수 가격처럼 시장 상태를 설명하는 변수입니다.
            - **RMSE/MAE**: 예측 오차입니다. 낮을수록 좋습니다.
            - **Feature importance**: LightGBM이 어떤 macro feature를 자주/강하게 사용했는지 보여주는 힌트입니다. 인과관계 증명은 아닙니다.
            """
        )


def render_detail_tab_guide() -> None:
    st.markdown(
        """
        ### 아래 상세 지표를 보는 방법

        이 탭은 아래에 이어지는 1~7번 상세 섹션의 해설판입니다.

        - **1번 실험 요약**: 이 실험 artifact가 어떤 기간/샘플/feature로 만들어졌는지 확인합니다.
        - **2번 성능 비교**: LightGBM이 zero/mean baseline보다 나은지 봅니다. 낮은 RMSE가 더 좋습니다.
        - **3번 factor별 진단**: 어떤 factor에서 모델이 특히 못 맞히는지 찾습니다.
        - **4번 long-short return**: 모델 target 자체가 안정적인지 봅니다.
        - **5번 prediction 진단**: 예측값과 실제값이 같은 방향으로 움직이는지 봅니다.
        - **6번 feature importance**: 모델이 어떤 macro feature에 반응했는지 힌트를 봅니다.
        - **7번 coverage**: 결측과 basket 수를 확인해 결과를 믿을 수 있는지 점검합니다.

        어렵게 느껴지면 **2번 RMSE → 4번 target 흐름 → 7번 데이터 품질**만 먼저 보면 됩니다.
        """
    )


def render_dashboard() -> None:
    st.set_page_config(page_title="DDQM2-lite 결과 대시보드", page_icon="📊", layout="wide")
    apply_theme()
    st.title("DDQM2-lite 결과 대시보드")
    st.caption("저장된 smoke artifact를 읽어서 보여주는 read-only Streamlit 화면입니다. 다운로드, 재생성, 재학습, 파일 수정은 하지 않습니다.")
    st.warning(PROTOTYPE_CAVEAT, icon="⚠️")
    render_pipeline_map()
    with st.expander("처음 보는 사람을 위한 읽는 법", expanded=True):
        render_usage_guide()

    with st.sidebar:
        st.header("Artifact 경로")
        st.caption("기본값은 방금 돌린 smoke experiment 결과입니다. 다른 실험을 보면 여기 경로만 바꾸면 됩니다.")
        experiment_dir = as_path(st.text_input("실험 디렉터리", value=relative_label(DEFAULT_EXPERIMENT_DIR)))
        factor_returns_path = as_path(st.text_input("Factor return artifact", value=relative_label(DEFAULT_FACTOR_RETURNS)))
        model_dataset_path = as_path(st.text_input("Model dataset artifact", value=relative_label(DEFAULT_MODEL_DATASET)))
        top_n = int(st.slider("Feature importance Top-N", min_value=5, max_value=50, value=20, step=5))
        st.divider()
        st.caption("이 dashboard는 pandas/json/pathlib로 위 경로를 읽기만 합니다.")

    missing_files = [experiment_dir / filename for filename in REQUIRED_EXPERIMENT_FILES if not (experiment_dir / filename).exists()]
    metrics, metrics_error = read_json(experiment_dir / "metrics.json")
    manifest, manifest_error = read_json(experiment_dir / "manifest.json")
    predictions, predictions_error = read_table(experiment_dir / "predictions.parquet")
    importance, importance_error = read_table(experiment_dir / "feature_importance.csv")
    factor_returns, factor_returns_error = read_table(factor_returns_path) if factor_returns_path.exists() else (None, f"Missing table artifact: {factor_returns_path}")
    model_dataset, model_dataset_error = read_table(model_dataset_path) if model_dataset_path.exists() else (None, f"Missing table artifact: {model_dataset_path}")

    errors = [error for error in (metrics_error, manifest_error, predictions_error, importance_error, factor_returns_error, model_dataset_error) if error]
    if missing_files or errors:
        show_missing_artifacts(experiment_dir, factor_returns_path, model_dataset_path)
        if errors:
            st.markdown("**읽기 상태**")
            st.code("\n".join(errors), language="text")
        return

    assert metrics is not None
    assert manifest is not None
    assert predictions is not None
    assert importance is not None

    easy_tab, detail_guide_tab = st.tabs(["쉽게 풀어쓴 설명", "상세 지표 보는 법"])
    with easy_tab:
        render_plain_language_explanation(manifest, metrics, predictions, importance, factor_returns, model_dataset)
    with detail_guide_tab:
        render_detail_tab_guide()

    st.subheader("1. 실험 요약")
    warning_text = str(manifest.get("prototype_warning") or PROTOTYPE_CAVEAT)
    st.info(warning_text, icon="🧪")
    run_summary = summarize_manifest(manifest)
    counts = manifest.get("row_counts", {}) if isinstance(manifest.get("row_counts"), dict) else {}
    metric_cols = st.columns(4)
    metric_cols[0].metric("Run ID", str(manifest.get("run_id", "n/a")))
    metric_cols[1].metric("Factor 수", str(manifest.get("factor_count", "n/a")))
    metric_cols[2].metric("검증 rows", str(counts.get("validation", "n/a")))
    metric_cols[3].metric("Macro feature 수", str(len(manifest.get("feature_columns", [])) if isinstance(manifest.get("feature_columns"), list) else "n/a"))
    st.dataframe(run_summary, width="stretch", hide_index=True)

    st.subheader("2. LightGBM vs 단순 baseline 성능")
    st.caption("검증 구간에서 LightGBM과 zero_return/train_mean_return 같은 단순 기준선을 비교합니다. smoke 결과라 방향성 확인용입니다.")
    metrics_frame = metrics_to_frame(metrics)
    if metrics_frame.empty:
        st.warning("metrics.json 안에 train/validation metric 객체가 없습니다.")
    else:
        validation_metrics = metrics_frame.loc[metrics_frame["split"].eq("validation"), :].copy()
        st.dataframe(validation_metrics, width="stretch", hide_index=True)
        chart_metrics = [column for column in ("rmse", "mae", "r2", "pearson_ic", "mean_date_ic", "mean_factor_ic") if column in validation_metrics.columns]
        for metric_name in chart_metrics:
            chart_data = validation_metrics.loc[:, ["model", metric_name]].dropna().set_index("model")
            if not chart_data.empty:
                st.bar_chart(chart_data, height=240)

    st.subheader("3. Factor별 검증 진단")
    st.caption("각 factor target에서 예측 오차가 어느 정도인지 봅니다. 특정 factor만 유독 나쁘면 label 또는 feature 구성을 점검해야 합니다.")
    factor_breakdown = factor_metric_breakdown(predictions)
    if factor_breakdown.empty:
        st.warning("predictions.parquet에서 factor별 metric을 계산할 수 없습니다.")
    else:
        st.dataframe(factor_breakdown, width="stretch", hide_index=True)
        if "rmse" in factor_breakdown.columns:
            st.bar_chart(factor_breakdown.set_index("factor_name")[["rmse"]], height=280)

    st.subheader("4. Long-short factor return label")
    st.caption("모델이 맞히려는 target입니다. 위쪽 chart는 월별 factor return, 아래 chart는 단순 누적 흐름입니다.")
    if factor_returns is not None and not factor_returns.empty and {"formation_date", "factor_name", "long_short_return"}.issubset(factor_returns.columns):
        returns = factor_returns.copy()
        returns["formation_date"] = pd.to_datetime(returns["formation_date"], errors="coerce")
        pivot = returns.pivot_table(index="formation_date", columns="factor_name", values="long_short_return", aggfunc="mean").sort_index()
        st.line_chart(pivot, height=280)
        cumulative = (1.0 + pivot.fillna(0.0)).cumprod() - 1.0
        st.line_chart(cumulative, height=280)
    else:
        st.warning("Factor return artifact가 비어 있거나 formation_date/factor_name/long_short_return 컬럼이 없습니다.")

    st.subheader("5. Prediction 진단")
    st.caption("예측값과 실제 long-short return의 관계를 봅니다. 점들이 대각선에 가까울수록 예측 방향이 맞는 편입니다.")
    valid_predictions = validation_predictions(predictions)
    if {"target_long_short_return", "prediction"}.issubset(valid_predictions.columns):
        scatter = valid_predictions.loc[:, ["target_long_short_return", "prediction", "factor_name"]].copy()
        scatter["target_long_short_return"] = pd.to_numeric(scatter["target_long_short_return"], errors="coerce")
        scatter["prediction"] = pd.to_numeric(scatter["prediction"], errors="coerce")
        st.scatter_chart(scatter.dropna(), x="prediction", y="target_long_short_return", color="factor_name", height=360)
        residual = (scatter["target_long_short_return"] - scatter["prediction"]).dropna()
        if not residual.empty:
            bins = min(30, max(5, int(np.sqrt(len(residual)))))
            counts_hist, edges = np.histogram(residual.to_numpy(dtype=float), bins=bins)
            histogram = pd.DataFrame({"residual_bin": [f"{edges[i]:.4f} to {edges[i + 1]:.4f}" for i in range(len(counts_hist))], "rows": counts_hist})
            st.bar_chart(histogram.set_index("residual_bin"), height=260)
    else:
        st.warning("Predictions artifact에 target/prediction 컬럼이 없습니다.")

    st.subheader("6. Macro feature importance")
    st.caption("LightGBM이 어떤 macro feature를 split/gain 기준으로 많이 사용했는지 봅니다. 인과 해석이 아니라 모델 사용 빈도/기여도 힌트입니다.")
    if not importance.empty and {"feature", "importance", "importance_type"}.issubset(importance.columns):
        importance_type = st.radio("Importance 기준", sorted(importance["importance_type"].astype(str).unique()), horizontal=True)
        top_importance = importance.loc[importance["importance_type"].astype(str).eq(importance_type), ["feature", "importance"]].copy()
        top_importance["importance"] = pd.to_numeric(top_importance["importance"], errors="coerce")
        top_importance = top_importance.sort_values("importance", ascending=False).head(top_n)
        st.bar_chart(top_importance.set_index("feature"), height=420)
        st.dataframe(top_importance, width="stretch", hide_index=True)
    else:
        st.warning("Feature importance artifact가 비어 있거나 필요한 컬럼이 없습니다.")

    st.subheader("7. 데이터 커버리지, 결측, basket count")
    st.caption("artifact row 수, 결측, long/short basket 크기를 확인합니다. smoke 실험에서는 basket 수가 작아서 성과보다 구조 검증에 초점을 둡니다.")
    coverage = coverage_table(model_dataset, factor_returns)
    if coverage.empty:
        st.warning("Model dataset 또는 factor return coverage를 표시할 수 없습니다.")
    else:
        st.dataframe(coverage, width="stretch", hide_index=True)
    missingness = pd.concat(
        [
            missingness_by_column(model_dataset, "macro_factor_model_ready"),
            missingness_by_column(factor_returns, "factor_long_short_returns"),
        ],
        ignore_index=True,
    )
    if not missingness.empty:
        st.dataframe(missingness.loc[missingness["missing_values"].gt(0), :], width="stretch", hide_index=True)
    if factor_returns is not None and {"factor_name", "long_count", "short_count"}.issubset(factor_returns.columns):
        basket = factor_returns.groupby("factor_name", dropna=False).agg(
            rows=("factor_name", "size"),
            min_long_count=("long_count", "min"),
            min_short_count=("short_count", "min"),
            median_long_count=("long_count", "median"),
            median_short_count=("short_count", "median"),
        )
        st.dataframe(basket.reset_index(), width="stretch", hide_index=True)


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
