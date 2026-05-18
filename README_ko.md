# autoquant-lab

[English](README.md)

`autoquant-lab`는 DDQM2 아이디어를 미국 주식시장에 맞게 이식하기 위한 오프라인 리서치 하네스입니다. 로컬 연구 데이터에서 point-in-time 월별 주식 패널을 만들고, macro/market feature로 factor long-short return을 예측한 뒤, 예측값을 동적 factor allocation으로 바꿔 walk-forward OOS 포트폴리오를 평가합니다.

이 저장소는 **코드와 문서 스캐폴드만** 공개합니다. Raw WRDS-style dataset, credential, private research note, generated experiment artifact, vendor/reference PDF는 의도적으로 제외합니다.

## 무엇을 하는가

현재 하네스는 USA-version DDQM2 연구 루프를 지원합니다.

1. 로컬 offline artifact에서 월별 `(date, security)` 패널을 준비합니다.
2. point-in-time lag control이 적용된 feature family를 만듭니다.
3. EQR factor registry에서 stock-level factor score를 계산합니다.
4. factor score를 다음 달 factor long-short return으로 변환합니다.
5. factor return 예측을 위해 factor별 CPU-friendly model을 학습합니다.
6. 예측 factor return을 non-negative factor weight로 변환합니다.
7. 두 가지 surface를 평가합니다.
   - weighted factor-return portfolio
   - DDQM2-style stock-level weighted factor score QSpread portfolio
8. manifest, metric, report, reproducible run metadata를 로컬에 저장합니다.

이 프로젝트는 삼성증권 DDQM/DDQM2에서 영감을 받았지만, 한국시장 production setup의 직접 복제는 아닙니다. Universe, data source, macro feature, factor definition, evaluation protocol이 모두 미국시장 adaptation에 맞게 다릅니다.

## 현재 연구 축

현재 구현은 USA-DDQM2 축을 실제 실행 가능한 run으로 올려두었습니다.

- `selected_13_global_local`: 사용 가능한 factor registry에서 DDQM2-inspired 13-factor selection을 구성합니다.
- `ddqm2_25x3_us_macro`: current, short-direction, medium-direction style feature를 포함한 DDQM2 macro design의 U.S. adaptation입니다.
- `expanded_us_macro`: 로컬 artifact가 지원하는 추가 U.S. macro/market variable 확장 축입니다.
- `stock_score_qspread_ddqm2`: stock-level weighted factor score QSpread portfolio surface입니다.

Quantile `q`는 고정 default가 아니라 연구 축으로 둡니다.

- q=0.10은 DDQM2-reference decile construction입니다.
- q=0.20과 q=0.30은 더 넓고 분산된 leg를 보기 위한 U.S. adaptation 설정입니다.

최종 matrix report:

- [`reports/usa_ddqm2_matrix_report.md`](reports/usa_ddqm2_matrix_report.md)

## Walk-forward timing과 리밸런싱

Portfolio surface는 월별입니다. 각 `formation_date`는 월별 리밸런싱 날짜이고, label은 `ret_1m_fwd`, 즉 다음 1개월 forward return입니다.

기본 full-run 설정에서는 다음처럼 동작합니다.

- portfolio weight는 매월 다시 계산합니다.
- long/short stock basket도 매월 다시 구성합니다.
- 각 월별 portfolio는 다음 1개월 수익률로 평가합니다.
- forecasting model은 12개월 walk-forward test fold마다 한 번 재학습합니다.

예를 들어 어떤 fold가 `2023-01`부터 `2023-12`까지를 test한다면, 모델은 그 test block 이전 날짜만 사용해 학습하고, 직전 validation block은 training에서 제외합니다. 즉 2023년 label로 학습한 뒤 2023년을 예측하는 구조가 아닙니다.

기본 구조는 다음과 같습니다.

```text
월별 포트폴리오 리밸런싱
+ 1개월 holding horizon
+ 12개월 단위 모델 재학습 cadence
```

더 엄격한 monthly-refit 실험을 원하면 test fold를 1개월로 설정합니다.

```bash
PYTHONPATH=src:. python scripts/eqr_run_ddqm2.py \
  --config configs/server_full.yaml \
  --evaluation-mode walk_forward \
  --walk-forward-test-periods 1 \
  --walk-forward-validation-periods 0
```

## Repository layout

```text
configs/                     안전한 YAML config와 ablation plan
scripts/                     preparation, validation, DDQM2 run, planning CLI
src/autoquant_lab/eqr/       메인 EQR package code
src/autoquant_lab/eqr/factors/DDQM2-style factor scoring, selection, allocation, backtest
tests/                       config, panel, factor, model, reporting 중심 테스트
reports/                     source-controlled final report only
```

Runtime/private directory는 ignore합니다.

```text
data/                        raw/local research data, WRDS/FRED-style export 포함, never committed
experiments/                 generated prepared panel과 run artifact
site/                        generated static report
.env                         credential/local setting
*.pdf, *.xlsx                local reference/data file
```

## 의도적으로 포함하지 않는 것

이 저장소에는 다음을 포함하지 않습니다.

- WRDS/CRSP/Compustat/IBES data
- FRED/macro source export 또는 vendor macro file
- `.env` file 또는 credential
- `EQR.md` 같은 private research note
- DDQM/DDQM2 PDF reference
- generated parquet experiment artifact
- generated static site output

이 파일들은 private local/server environment에서만 필요하며 `.gitignore`로 제외합니다.

## 주요 CLI 예시

DDQM2 ablation command를 실행하지 않고 렌더링만 합니다.

```bash
PYTHONPATH=src:. python scripts/eqr_plan_ddqm2_ablations.py --format commands --limit 8
```

이미 준비된 로컬 artifact에서 DDQM2-style experiment를 실행합니다.

```bash
PYTHONPATH=src:. python scripts/eqr_run_ddqm2.py \
  --config configs/server_full.yaml \
  --run-id example_usa_ddqm2_q20 \
  --quantile 0.20 \
  --model lightgbm \
  --factor-universe selected_13_global_local \
  --macro-feature-design ddqm2_25x3_us_macro \
  --portfolio-surface stock_score_qspread_ddqm2 \
  --evaluation-mode walk_forward \
  --min-weight 0.03
```

Focused test:

```bash
PYTHONPATH=src:. python -m pytest tests/test_ddqm2_ablation_plan.py tests/test_factors.py -q
```

## Research status

현재 matrix는 다음을 완료했습니다.

- capped panel smoke run 1개
- q=0.10, q=0.20, q=0.30에 대한 full-data LightGBM walk-forward OOS run 6개
- LightGBM, ridge, elasticnet, random forest, extra trees, baseline mean 등 CPU-friendly model 비교

최종 report의 headline 해석은 다음과 같습니다.

- q=0.20 stock-score QSpread가 가장 높은 cumulative return을 보였습니다.
- q=0.30 stock-score QSpread는 stock-score full run 중 mean/vol profile과 turnover 측면에서 더 균형 잡힌 후보였습니다.
- q=0.10은 DDQM2-reference benchmark이지 강제 default가 아닙니다.

모든 결과는 gross research backtest입니다. Transaction cost, borrow cost, slippage, market impact, capacity limit, final tradability review는 아직 반영하지 않았습니다.

## Verification

DDQM2-USA 구현 과정에서 사용한 focused check:

```text
PYTHONPATH=src:. python -m pytest tests/test_ddqm2_ablation_plan.py tests/test_factors.py -q
16 passed
```

Full-data run에서는 매우 큰 factor-score table을 다시 한 번에 materialize하지 않도록 chunked factor-score generation을 사용합니다.

## Safety and publication policy

공개 전에는 private data가 tracked file에 들어가지 않았는지 확인합니다.

```bash
git ls-files data experiments site .env EQR.md '*.pdf' '*.xlsx'
```

예상 출력은 명시적으로 안전한 placeholder를 제외하면 비어 있어야 합니다.

## License / use

이 저장소는 research scaffold입니다. 투자 조언이 아니며, 바로 배포 가능한 trading strategy도 아닙니다.
