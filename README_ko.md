# autoquant-lab

`autoquant-lab`는 EQR(Equity Quant Research) 기반의 **오프라인 자동 리서치 하네스**입니다. 이미 내려받아 둔 WRDS/FRED 계열 로컬 데이터를 읽어서, 시점 기준(point-in-time) 패널을 만들고, CPU 기반 모델 실험을 반복 실행하며, 결과를 정적 HTML 사이트로 렌더링하는 구조입니다.

중요한 원칙은 다음과 같습니다.

- `data/`는 사용자 로컬 데이터로 취급하며 커밋하지 않습니다.
- WRDS 로그인, credential prompt, 네트워크 다운로드 로직은 금지합니다.
- Codex 같은 에이전트는 실험 config만 바꿀 수 있고, 하네스 코드는 고정합니다.
- 실험 결과는 `experiments/`, `reports/`, `site/` 아래 생성되며 기본적으로 git에서 제외됩니다.
- LSTM/Transformer 같은 GPU 필요 모델은 v1 범위에서 제외합니다.

## 연구 배경

`autoquant-lab`는 삼성증권의 Data-driven Quant Model(DDQM) 아이디어를 미국 주식 데이터 기반의 오프라인 리서치 플랫폼으로 옮겨보는 프로젝트입니다. 로컬 기획 문서인 `EQR.md`는 이 아이디어를 EQR(Equity Quant Research) 관점으로 재정리한 문서입니다. 목표는 WRDS 스타일의 CRSP/Compustat/IBES 데이터와 FRED 스타일 매크로 데이터를 이용해 시점 기준 월별 `(날짜 x 종목)` 패널을 만들고, 반복 가능한 CPU 기반 실험으로 종목 또는 팩터 성과를 예측하는 것입니다.

작업 디렉토리에 있는 삼성증권 DDQM PDF들은 다음 연구 흐름을 설명합니다.

- `Data-Driven Quant Model.pdf`: DDQM 원형입니다. 거시/시장 변수로 feature를 만들고, 과거 팩터 수익률로 스타일 국면을 라벨링한 뒤, Random Forest 같은 분류 모델로 현재 국면을 예측하고, 예측된 국면에 맞춰 팩터 포트폴리오를 회전시키는 방식입니다.
- `Data-Driven Quant Model2.pdf`: DDQM2입니다. discrete regime classification을 줄이고, LightGBM 계열 회귀 모델로 각 알파 팩터의 다음 1개월 long-short 수익률을 직접 예측한 뒤, 예측 수익률에 따라 동적으로 팩터 비중을 배분하는 방식입니다.

이 레포는 삼성증권의 한국 시장 실전 모델을 그대로 복제하는 것이 아닙니다. DDQM/DDQM2의 연구 패턴을 참고하되, 미국 로컬 데이터, 명시적인 point-in-time 조인, SQLite 실험 ledger, config-only 에이전트 자동화, 정적 실험 히스토리 사이트를 갖춘 재현 가능한 리서치 하네스로 다시 설계합니다. PDF와 `EQR.md`는 로컬 연구 참고자료이며, 코드 스캐폴드를 실행하기 위해 커밋될 필요는 없습니다.

## 활성 레이아웃

| 영역 | 경로 | 역할 |
|---|---|---|
| EQR 패키지 | `src/autoquant_lab/eqr/` | 데이터 계약, PIT 조인, 패널/피처/모델/하네스 구현 |
| 설정 | `configs/` | 소스 관리되는 실험 및 하네스 설정 |
| 실험 산출물 | `experiments/` | 로컬 실행 결과. `.gitkeep` 외에는 무시 |
| 리포트 | `reports/` | 생성 리포트. `.gitkeep` 외에는 무시 |
| 정적 사이트 | `site/` | 생성된 HTML 실험 히스토리. `.gitkeep` 외에는 무시 |
| Codex Skill | `skills/eqr-autoresearch/` | 에이전트용 운영 지침 |
| 테스트 | `tests/` | 데이터 계약, 파이프라인, 모델, ledger, 리포팅 테스트 |
| 스크립트 | `scripts/eqr_*.py` | 활성 EQR 경로를 실행하는 얇은 CLI 래퍼 |

## 빠른 시작: 골든 패스

전체 오프라인 EQR autoresearch smoke를 실행합니다.

```bash
python scripts/eqr_autoresearch.py golden-path \
  --config configs/golden_path.yaml \
  --max-trials 3
```

이 명령은 다음 단계를 순서대로 실행합니다.

1. `scripts/eqr_validate_raw_data.py`로 로컬 원시 데이터 계약을 검증합니다.
2. `scripts/eqr_build_links.py`로 CRSP/Compustat/IBES point-in-time 링크를 만듭니다.
3. `scripts/eqr_prepare_panel.py`로 월별 라벨과 피처 패밀리를 준비합니다.
4. `configs/golden_path.yaml`을 검증합니다.
5. SQLite ledger에 기반해 모델/config 실험 3개를 큐에 넣고 실행합니다.
6. metrics, predictions, model, artifact, ledger state를 `experiments/` 아래 저장합니다.
7. `scripts/eqr_render_site.py`로 리포트와 정적 사이트를 생성합니다.
8. `python scripts/eqr_ci.py --smoke`를 실행합니다.

기본 골든 패스는 smoke 실행을 위해 `--max-rows 50000`을 사용합니다. 전체 오프라인 패널을 사용하려면 충분한 메모리/시간이 있는 머신에서 `--max-rows 0`을 사용하세요.

실행 후 결과 확인 예시:

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

ledger = Path("experiments/ledger.sqlite")
with sqlite3.connect(ledger) as conn:
    terminal = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE state IN ('SUCCEEDED', 'FAILED', 'REJECTED', 'DEAD_LETTER')"
    ).fetchone()[0]
print(f"terminal_runs={terminal}")
print(Path("site/index.html").resolve())
PY
```

`site/index.html`에는 실행 ID, 주요 지표, promotion 상태가 표시됩니다. 이 결과는 리서치 하네스 검증용 증거이며 투자 조언이나 실거래 가능성 주장이 아닙니다.

## 기본 스캐폴드 체크

```bash
python -c "import autoquant_lab.eqr"
python scripts/eqr_validate_import.py
python scripts/eqr_scan_secrets.py
```

## 아키텍처

```text
                          configs/golden_path.yaml
                                    |
                                    v
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Golden-path stage sequence                                         │
  │                                                                     │
  │  1. validate_raw_data ──> reports/eqr_raw_data_validation.json      │
  │  2. build_links ─────────> experiments/prepared/links/              │
  │  3. prepare_labels ──────> experiments/prepared/panel/              │
  │  4. prepare_features ────> experiments/prepared/features/           │
  │  5. validate_config ────> config hash check                        │
  │  6. execute trials ──────> experiments/ledger.sqlite + runs/       │
  │  7. render_site ────────> site/index.html + reports/               │
  │  8. ci_smoke ────────────> reports/eqr_ci_report.json              │
  └─────────────────────────────────────────────────────────────────────┘
                                    |
                                    v
                    SQLite ledger finite-state machine
              PROPOSED -> QUEUED -> CLAIMED -> RUNNING
                -> EVALUATING -> PERSISTING -> RENDERING -> SUCCEEDED
                                    |
                                    v
               experiments/runs/<run_id>/{metrics,predictions,model}
                                    |
                                    v
                  reports/eqr_experiment_history.* + site/index.html
```

### 데이터 흐름

```text
data/ (read-only)
  ├── CRSP monthly / names
  ├── Compustat company / fundq
  ├── IBES link / summary / detail / actual / target
  └── macro_features.parquet
          |
          v
eqr_validate_raw_data.py ──> raw data contract validation + offline guard
          |
          v
eqr_build_links.py ───────> PIT link tables
          |
          v
eqr_prepare_panel.py --stage labels ───> monthly_labels.parquet
          |
          v
eqr_prepare_panel.py --stage features ─> feature-family parquets
          |
          v
eqr_autoresearch.py golden-path ───────> ledgered model trials
          |
          v
eqr_render_site.py ───────────────────> static HTML experiment history
```

## 주요 구성 요소

### 1. 오프라인 데이터 계약

`src/autoquant_lab/eqr/data_contracts.py`와 `path_resolver.py`는 현재 로컬 `data/` 레이아웃을 해석하고 필요한 컬럼/날짜/중복 키를 검증합니다. manifest에 적힌 경로가 실제 파일 위치와 달라도 실제 parquet 파일을 기준으로 해석합니다.

### 2. Point-in-time 링크

`src/autoquant_lab/eqr/pit.py`는 CRSP names, CCM, IBES link를 날짜 유효 구간으로 조인합니다. ticker만으로 조인하지 않으며, 유효 기간을 벗어난 식별자 매칭은 차단합니다.

### 3. 월별 패널과 라벨

`src/autoquant_lab/eqr/panel.py`는 CRSP 월별 데이터를 사용해 `(formation_date, permno)` 패널을 만들고, 1개월/3개월/6개월 forward return 라벨을 생성합니다.

### 4. 피처 패밀리

`src/autoquant_lab/eqr/features/`는 다음 피처 그룹을 제공합니다.

- `macro`: 매크로/시장 변수의 월말 as-of 스냅샷
- `crsp`: 가격 모멘텀, 반전, 사이즈
- `compustat`: 밸류에이션, 퀄리티, 성장 지표
- `ibes`: 컨센서스, 리비전, 서프라이즈, 목표주가

### 5. CPU 모델 레지스트리

`src/autoquant_lab/eqr/models/`는 CPU 기반 모델만 지원합니다.

- LightGBM
- Ridge / ElasticNet
- RandomForest / ExtraTrees
- 평균/중앙값/랜덤 베이스라인

### 6. 다중 지표 평가

`src/autoquant_lab/eqr/metrics.py`는 다음 지표를 계산합니다.

- Rank IC
- Pearson IC
- Decile long-short return
- Hit rate
- MSE / MAE
- Turnover proxy
- Max drawdown
- Stability
- Feature coverage
- Runtime

### 7. SQLite FSM Ledger

`src/autoquant_lab/eqr/ledger.py`는 실험 상태를 SQLite finite-state machine으로 관리합니다.

```text
PROPOSED -> QUEUED -> CLAIMED -> RUNNING -> EVALUATING
  -> PERSISTING -> RENDERING -> SUCCEEDED
```

실패한 작업은 retry 후 `DEAD_LETTER`로 이동할 수 있습니다. 모든 상태 전환은 append-only event log로 기록됩니다.

### 8. Scheduler / Autoresearch Loop

`src/autoquant_lab/eqr/scheduler.py`와 `scripts/eqr_autoresearch.py`는 config-only 실험을 제안, 큐잉, 실행, 평가, 저장합니다. 에이전트는 YAML config만 바꿀 수 있고 하네스 코드는 변경하지 않는 것이 원칙입니다.

### 9. Static HTML Site

`src/autoquant_lab/eqr/reporting/`와 `scripts/eqr_render_site.py`는 실험 히스토리를 HTML로 렌더링합니다.

생성 페이지 예시:

- `site/index.html`
- `site/leaderboard.html`
- `site/dead_letter.html`
- `site/coverage.html`
- `site/about.html`
- `site/run_<run_id>.html`

## CI

전체 로컬 CI 계약을 실행합니다.

```bash
python scripts/eqr_ci.py
```

빠른 smoke 검증:

```bash
python scripts/eqr_ci.py --smoke
```

CI는 다음을 확인합니다.

1. pytest
2. 원시 데이터 계약 검증
3. config 검증
4. prepared panel 존재 확인
5. ledger FSM 테스트
6. Codex Skill 문서 검증
7. static site 검증
8. secret scan
9. offline-only guard

CI 리포트는 `reports/eqr_ci_report.json`에 생성됩니다.

## 데이터 정책

`data/` 아래 파일은 사용자가 별도로 보유한 로컬 WRDS/FRED 스타일 데이터입니다. 이 레포에서는 다음 원칙을 지킵니다.

- `data/`는 git에 올리지 않습니다.
- 파이프라인은 `data/`를 읽기만 합니다.
- WRDS 로그인이나 credential 입력을 요구하지 않습니다.
- 네트워크 다운로드나 외부 API 호출을 하지 않습니다.
- 생성된 metrics는 하네스 검증용이며 투자 조언이 아닙니다.

## Codex Skill 문서

에이전트 운영자는 실험을 제안하기 전에 반드시 다음 문서를 읽어야 합니다.

```text
skills/eqr-autoresearch/SKILL.md
```

이 문서는 다음을 정의합니다.

- inspection 순서
- 허용된 수정 경로
- 금지된 경로와 행동
- 실험 제안/큐잉/실행 방법
- metric 해석
- promotion gate
- 실패 복구 절차
- stop condition

## 레거시 프로토타입

기존 yfinance/DDQM2-lite 프로토타입은 활성 경로가 아닙니다. 새 CI, quickstart, command wrapper는 EQR 오프라인 경로만 사용합니다.
