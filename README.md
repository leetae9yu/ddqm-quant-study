# autoquant-lab

Credential-safe research scaffold for adapting the Samsung Securities DDQM2 LightGBM workflow to a U.S. S&P 500 setting.

This repository currently focuses on two tracks:

1. Prepare the WRDS/CRSP/Compustat/IBES data map for the final research path.
2. Provide a WRDS-free prototype path using yfinance and the existing macro workbook, clearly marked as prototype-only.

## What is in this repo

| Area | Files |
|---|---|
| Config and secrets | `.env.example`, `src/autoquant_lab/config.py` |
| Macro feature pipeline | `scripts/build_macro_features.py`, `scripts/validate_macro_workbook.py`, `scripts/diagnose_macro_feature_quality.py` |
| WRDS readiness | `scripts/probe_wrds.py`, `docs/wrds_manual_export_guide.md` |
| Data requirements | `docs/crsp_requirements.md`, `docs/compustat_requirements.md`, `docs/ibes_requirements.md`, `docs/ddqm2_sp500_data_inventory.md` |
| Prototype data path | `scripts/build_yfinance_sp500_labels.py`, `scripts/assemble_yfinance_macro_dataset.py` |
| Prototype validation | `scripts/validate_yfinance_sp500_labels.py`, `scripts/validate_yfinance_macro_dataset.py` |
| Prototype modeling | `scripts/train_yfinance_macro_lgbm_baseline.py` |
| Safety check | `scripts/scan_secrets.py` |

## Important caveats

- WRDS/CRSP/Compustat/IBES should be treated as the final research data path.
- yfinance uses public, current-membership data and is only for pipeline testing.
- Prototype outputs include `prototype_only=True` and should not be described as research-grade or survivorship-bias-free.
- Keep `.env` local. Do not commit API keys, WRDS credentials, raw licensed data, or generated prototype outputs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally:

```bash
FRED_API_KEY=your_fred_key_here
WRDS_USERNAME=your_wrds_username_here
```

Use `PYTHONPATH=src` when running scripts that import project modules.

## Safety check before commits or pushes

Run this before sharing or pushing changes:

```bash
python scripts/scan_secrets.py
```

Expected safe output:

```text
No potential secrets found in .py/.txt files under ...
```

## Macro workbook checks

Validate the existing macro feature workbook:

```bash
python scripts/validate_macro_workbook.py expanded_macro_market_features.xlsx
```

Run deeper quality diagnostics:

```bash
python scripts/diagnose_macro_feature_quality.py --top-n 5
```

This prints coverage, missing values, constant columns, duplicate columns, high correlations, robust outliers, and largest one-day changes to stdout. It does not create report files by default.

## Rebuilding macro features

If you need to rebuild the macro/market feature workbook from APIs:

```bash
PYTHONPATH=src python scripts/build_macro_features.py --output expanded_macro_market_features.xlsx
```

Requirements:

- `.env` must contain `FRED_API_KEY`.
- The script may call FRED/ALFRED and yfinance.
- Do not print or commit the API key.

## WRDS readiness

WRDS is not required for the prototype path, but it is needed for the final research-grade path.

Check WRDS access when the `wrds` package and credentials are available:

```bash
PYTHONPATH=src python scripts/probe_wrds.py
```

If Python WRDS access is blocked, follow:

```text
docs/wrds_manual_export_guide.md
```

The main data inventory is here:

```text
docs/ddqm2_sp500_data_inventory.md
```

## WRDS-free prototype path

Use this path only to verify that the pipeline works end to end before WRDS is available.

### 1. Build yfinance S&P 500 labels

Small smoke test:

```bash
PYTHONPATH=src python scripts/build_yfinance_sp500_labels.py \
  --start-date 2020-01-01 \
  --end-date 2020-03-31 \
  --max-tickers 10 \
  --horizon-days 5 \
  --output prototypes/yfinance_sp500/sp500_yfinance_labels_sample.csv
```

Validate the labels:

```bash
PYTHONPATH=src python scripts/validate_yfinance_sp500_labels.py \
  prototypes/yfinance_sp500/sp500_yfinance_labels_sample.csv \
  --label-column forward_return_5d
```

### 2. Join labels with macro features

```bash
PYTHONPATH=src python scripts/assemble_yfinance_macro_dataset.py \
  --labels prototypes/yfinance_sp500/sp500_yfinance_labels_sample.csv \
  --macro-workbook expanded_macro_market_features.xlsx \
  --output prototypes/yfinance_sp500/sp500_yfinance_macro_model_ready_sample.csv
```

Validate the assembled model-ready table:

```bash
PYTHONPATH=src python scripts/validate_yfinance_macro_dataset.py \
  prototypes/yfinance_sp500/sp500_yfinance_macro_model_ready_sample.csv \
  --label-column forward_return_5d
```

### 3. Run the LightGBM prototype baseline

```bash
PYTHONPATH=src python scripts/train_yfinance_macro_lgbm_baseline.py \
  --input prototypes/yfinance_sp500/sp500_yfinance_macro_model_ready_sample.csv \
  --label-column forward_return_5d \
  --valid-fraction 0.25 \
  --min-train-rows 20 \
  --min-valid-rows 5 \
  --n-estimators 50 \
  --early-stopping-rounds 10
```

The script prints LightGBM metrics and two simple comparisons:

- `zero_return`: always predicts 0% return.
- `train_mean_return`: always predicts the average return from the train split.

These metrics are for smoke testing only, not for research conclusions.

## Generated files and git

Generated prototype CSV/Parquet files under `prototypes/yfinance_sp500/` are ignored by git.

Before pushing:

```bash
git status --short
python scripts/scan_secrets.py
```

Do not commit:

- `.env`
- raw WRDS data
- API keys or credentials
- generated prototype datasets
- large PDFs/XLSX files unless intentionally approved

---

# 한국어 사용 가이드

이 레포는 삼성증권 DDQM2 방식의 LightGBM 아이디어를 미국 S&P 500 기준으로 옮기기 위한 작업 공간입니다.

현재 목적은 두 가지입니다.

1. WRDS, CRSP, Compustat, IBES 같은 최종 연구용 데이터 경로를 정리합니다.
2. WRDS 접속 전까지 yfinance와 기존 매크로 엑셀로 임시 실험 파이프라인을 돌립니다.

## 먼저 알아둘 점

- 최종 연구용 데이터는 WRDS/CRSP/Compustat/IBES 쪽입니다.
- yfinance 경로는 임시 테스트용입니다.
- yfinance는 현재 S&P 500 구성종목 기반이라 생존편향 문제가 있습니다.
- 그래서 prototype 결과를 최종 성과나 논문용 결과처럼 해석하면 안 됩니다.
- API 키는 `.env`에만 두고 커밋하지 않습니다.

## 설치 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` 파일을 열어서 아래처럼 채웁니다.

```bash
FRED_API_KEY=본인_FRED_KEY
WRDS_USERNAME=본인_WRDS_ID
```

## 커밋/푸시 전 보안 확인

```bash
python scripts/scan_secrets.py
```

정상이라면 `.py` 또는 `.txt` 파일 안에 키가 없다고 나옵니다.

## 매크로 엑셀 확인

기본 검증:

```bash
python scripts/validate_macro_workbook.py expanded_macro_market_features.xlsx
```

품질 진단:

```bash
python scripts/diagnose_macro_feature_quality.py --top-n 5
```

여기서는 결측치, 중복 컬럼, 상관관계가 너무 높은 컬럼, 이상치, 하루 변화가 큰 구간 등을 확인합니다.

## 매크로 피처 다시 만들기

FRED 키가 `.env`에 들어있다면 아래 명령으로 매크로 피처를 다시 만들 수 있습니다.

```bash
PYTHONPATH=src python scripts/build_macro_features.py --output expanded_macro_market_features.xlsx
```

## WRDS 확인

WRDS 패키지와 계정이 준비되면 아래 명령으로 접근 가능 여부를 확인합니다.

```bash
PYTHONPATH=src python scripts/probe_wrds.py
```

WRDS 접속이 안 되면 아래 문서를 보면 됩니다.

```text
docs/wrds_manual_export_guide.md
```

전체 데이터 설명은 아래 문서에 있습니다.

```text
docs/ddqm2_sp500_data_inventory.md
```

## WRDS 없이 임시 파이프라인 돌리기

### 1. yfinance로 S&P 500 라벨 만들기

```bash
PYTHONPATH=src python scripts/build_yfinance_sp500_labels.py \
  --start-date 2020-01-01 \
  --end-date 2020-03-31 \
  --max-tickers 10 \
  --horizon-days 5 \
  --output prototypes/yfinance_sp500/sp500_yfinance_labels_sample.csv
```

검증:

```bash
PYTHONPATH=src python scripts/validate_yfinance_sp500_labels.py \
  prototypes/yfinance_sp500/sp500_yfinance_labels_sample.csv \
  --label-column forward_return_5d
```

### 2. yfinance 라벨에 매크로 피처 붙이기

```bash
PYTHONPATH=src python scripts/assemble_yfinance_macro_dataset.py \
  --labels prototypes/yfinance_sp500/sp500_yfinance_labels_sample.csv \
  --macro-workbook expanded_macro_market_features.xlsx \
  --output prototypes/yfinance_sp500/sp500_yfinance_macro_model_ready_sample.csv
```

검증:

```bash
PYTHONPATH=src python scripts/validate_yfinance_macro_dataset.py \
  prototypes/yfinance_sp500/sp500_yfinance_macro_model_ready_sample.csv \
  --label-column forward_return_5d
```

### 3. LightGBM baseline 돌리기

```bash
PYTHONPATH=src python scripts/train_yfinance_macro_lgbm_baseline.py \
  --input prototypes/yfinance_sp500/sp500_yfinance_macro_model_ready_sample.csv \
  --label-column forward_return_5d \
  --valid-fraction 0.25 \
  --min-train-rows 20 \
  --min-valid-rows 5 \
  --n-estimators 50 \
  --early-stopping-rounds 10
```

이 스크립트는 LightGBM 결과를 두 가지 단순 기준과 비교합니다.

- `zero_return`: 항상 0% 수익률이라고 예측
- `train_mean_return`: 학습 구간 평균 수익률로 예측

이 결과는 모델 성능 결론이 아니라, 전체 파이프라인이 제대로 연결됐는지 확인하는 용도입니다.

## git 주의사항

커밋 전에 항상 확인합니다.

```bash
git status --short
python scripts/scan_secrets.py
```

커밋하지 말아야 할 것:

- `.env`
- API 키
- WRDS 원천 데이터
- 자동 생성된 prototype CSV/Parquet
- 승인되지 않은 큰 PDF/XLSX 파일
