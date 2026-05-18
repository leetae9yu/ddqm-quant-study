# autoquant-lab 실험 종합 보고서

작성일: 2026-05-16

## 초록

본 프로젝트는 삼성증권 DDQM/DDQM2의 핵심 아이디어를 미국 주식시장 데이터에 맞게 이식하기 위한 오프라인 자동실험 하네스 구축 및 초기 실험 결과를 정리한 것이다. 원문의 핵심 구조인 “macro/market variable을 이용한 factor return forecast → dynamic factor allocation → long-short/QSpread 평가”를 유지하되, CRSP/Compustat/IBES/FRED-style 로컬 데이터를 사용하는 U.S. equity adaptation으로 재구성했다. 실험은 CPU-only 환경에서 실행되도록 설계했으며, raw data, credentials, local notes, generated artifacts는 GitHub 공개 대상에서 제외했다.

최종 구현은 selected 13-factor universe, DDQM2-style U.S. macro feature design, CPU-friendly factor-return forecasting, stock-level weighted factor score QSpread, expanding walk-forward OOS evaluation, ablation planner를 포함한다. 1,250,000개는 raw 전체 데이터가 아니라 date-balanced로 cap된 prepared panel row이며, 그 패널과 383개월 OOS 구간에서 q=0.10, q=0.20, q=0.30에 대한 LightGBM 중심 matrix를 수행했고, 별도 chunked sweep으로 ridge, elasticnet, random forest, extra trees, baseline mean도 비교했다. q=0.20 stock-score QSpread는 가장 높은 cumulative return을 보였고, q=0.30 stock-score QSpread는 mean/vol, turnover, drawdown 측면에서 더 균형 잡힌 후보로 해석되었다. 모델 비교에서는 LightGBM이 1.0M q=0.10 sweep에서 가장 높았지만, ridge/elasticnet도 강한 후보였고 1.25M fixed-holdout 계열에서는 ridge/elasticnet이 LightGBM보다 높은 누적 성과를 보인 구간도 있었다. 다만 모든 결과는 gross research backtest이며 transaction cost, liquidity, borrow, market impact, capacity 검증은 아직 반영되지 않았다.

## 1. 연구 배경과 문제 정의

DDQM2는 개별 종목의 미래 수익률을 직접 예측하기보다, 시장/거시 상태를 이용해 각 alpha factor의 다음 기간 long-short return을 예측하고, 그 예측값에 따라 factor weight를 동적으로 조정하는 접근이다. 이 프로젝트의 목표는 해당 구조를 미국 주식시장 데이터에 맞게 구현하고, 장시간 서버 실험을 반복할 수 있는 하네스를 만드는 것이었다.

초기 문제는 다음과 같이 정의했다.

1. 로컬에 준비된 WRDS/FRED-style artifact만 사용한다.
2. 로그인, 런타임 다운로드, credential prompt 없이 실행한다.
3. CRSP/Compustat/IBES/FRED 원천 데이터와 private notes는 외부에 공개하지 않는다.
4. LSTM/Transformer 같은 GPU-heavy model은 제외하고 CPU-friendly model만 사용한다.
5. 단순 수익률 ranking이 아니라 walk-forward OOS 기반의 반복 가능한 연구 trail을 만든다.

따라서 본 프로젝트의 산출물은 “실전 투자전략”이라기보다, DDQM2를 미국시장에 적용하기 위한 재현 가능한 연구 하네스와 초기 empirical 결과에 가깝다.

## 2. DDQM/DDQM2에서 따른 부분과 바꾼 부분

### 2.1 DDQM/DDQM2를 따른 부분

방법론의 큰 뼈대는 DDQM2에서 가져왔다.

- 개별 종목 수익률을 직접 예측하지 않고 factor return을 예측한다.
- factor return은 top/bottom bucket long-short 방식으로 구성한다.
- macro/market variable을 이용해 각 factor의 다음 기간 수익률을 예측한다.
- factor별 별도 regression model을 사용한다.
- LightGBM을 중심 모델로 사용하되, ridge/elasticnet/tree ensemble/baseline도 비교한다.
- 예측 factor return을 non-negative factor weight로 변환한다.
- 최종적으로 factor weight를 이용한 long-short/QSpread 성과를 본다.
- q=0.10 decile construction을 DDQM2-reference setting으로 유지한다.

### 2.2 U.S. adaptation으로 바꾼 부분

원문 DDQM2를 그대로 복제하지는 않았다. 주요 차이는 다음과 같다.

- 시장 universe: KOSPI200이 아니라 U.S. equity monthly panel을 사용했다.
- 데이터: 한국시장 원천 데이터가 아니라 CRSP/Compustat/IBES/FRED-style 로컬 artifact를 사용했다.
- factor universe: 원문의 13개 factor를 그대로 복제하기보다, EQR factor registry에서 global/local alpha 기준으로 selected 13-factor universe를 구성했다.
- macro feature: 원문의 25개 base variable x current/20d/60d direction 구조를 참고해 U.S. macro design으로 adaptation했다.
- evaluation: fixed holdout보다 expanding walk-forward OOS를 중심 기준으로 삼았다.
- q: q=0.10을 reference로 두되, q=0.20/q=0.30을 U.S. adaptation ablation으로 유지했다.

### 2.3 AI agent 관점에서 새로 바꾼 부분

AI agent는 실험 엔진 그 자체라기보다, 연구 개발과 검증을 조율하는 역할을 했다. 실제 실험은 deterministic CLI harness가 수행했다. 다만 AI agent 관점에서 다음 구조가 추가되었다.

- factor universe, macro design, q, model, allocation rule, portfolio surface를 ablation axis로 분리했다.
- `eqr_plan_ddqm2_ablations.py`를 통해 실행 가능한 조합과 아직 미구현인 backlog를 구분했다.
- manifest 기반 결과 parsing과 보고서 작성 흐름을 만들었다.
- secret scan, git ignore check, raw data exclusion check를 publish workflow에 포함했다.
- 실험 중 memory bottleneck을 발견하고 chunked factor score를 다시 full concat하지 않도록 수정했다.

즉 DDQM2의 방법론적 아이디어는 유지하되, 이를 장시간 자동 실험과 안전한 공개가 가능한 engineering harness로 재구성했다.

## 3. 데이터와 보안 범위

본 보고서에는 원천 데이터 값이나 private file 내용이 포함되지 않는다. 사용한 데이터 범주는 다음과 같이 추상적으로만 표현한다.

- CRSP-style monthly equity data
- Compustat-style accounting features
- IBES-style estimate/revision features
- FRED-style macro/market features

공개하지 않은 항목은 다음과 같다.

- WRDS/CRSP/Compustat/IBES 원천 parquet 또는 raw export
- FRED/macro source export
- `.env` 및 credential
- `EQR.md` private note
- DDQM/DDQM2 PDF reference files
- generated experiment parquet/csv artifacts
- generated static site output

GitHub에는 code scaffold, configuration, tests, README, 최종 matrix report만 공개했다.

## 4. 하네스 구조

프로젝트 구조는 EQR-first layout으로 정리했다.

```text
configs/                     safe YAML configs and ablation plans
scripts/                     CLI entrypoints for preparation, validation, DDQM2 runs, planning
src/autoquant_lab/eqr/       main EQR package code
tests/                       focused tests for config, panel, factors, models, reporting
reports/                     final report only; generated reports remain ignored
experiments/                 local/server runtime artifacts, ignored by git
site/                        generated static reports, ignored by git
data/                        raw/local data, ignored by git
```

핵심 실행 흐름은 다음과 같다.

```text
prepared panel/features
→ factor score generation
→ factor long-short return labeling
→ per-factor LightGBM regression
→ predicted factor return
→ factor allocation
→ weighted factor-return or stock-score QSpread backtest
→ manifest/report
```

## 5. 구현 단계와 시행착오

### 5.1 초기 DDQM2 factor-return pipeline

초기 구현은 factor score를 만들고, 각 factor의 top/bottom long-short return을 label로 만든 뒤, macro feature로 다음 기간 factor return을 예측하는 구조였다. 이 단계에서는 baseline mean, ridge, elasticnet, random forest, extra trees, LightGBM 같은 CPU-friendly model을 비교할 수 있도록 만들었다. 이 비교는 단순히 모델 종류를 늘리기 위한 것이 아니라, DDQM2 구조의 성과가 특정 boosting implementation 하나에 과도하게 의존하는지 확인하기 위한 robustness check였다.

### 5.2 패널 준비와 메모리 안정화

초기 row cap은 시간순 head 방식이어서 오래된 기간에 데이터가 몰리는 문제가 있었다. 이를 date-balanced cap으로 바꾸면서 1.25M row artifact가 1990년부터 2024년까지 전체 기간을 포괄하게 되었다.

또한 factor score generation은 row 수가 커질수록 메모리 병목이 생겼다. 이를 해결하기 위해 다음을 추가했다.

- factor score null filtering 및 compact dtype 적용
- `factor_score_chunk_dates` 기반 월 단위 chunking
- factor score row budget guard
- chunked factor score를 다시 full concat하지 않는 memory fix

### 5.3 OOS 평가 방식 정리

초기에는 fixed holdout을 사용했지만, 최종 기준은 expanding walk-forward OOS로 전환했다.

현재 메인 평가 방식은 다음과 같다.

```text
expanding train window
+ 12개월 validation
+ 다음 12개월 OOS test fold
→ 반복
```

최종 full matrix는 383개월 OOS period를 사용했다.

### 5.4 DDQM2-USA 확장

DDQM2에 더 가까운 U.S. adaptation을 위해 다음 축을 구현했다.

- `selected_13_global_local`: global alpha와 local alpha idea를 반영한 selected 13-factor universe
- `ddqm2_25x3_us_macro`: DDQM2 25x3 macro design의 U.S. adaptation
- `expanded_us_macro`: 확장 macro feature axis
- `stock_score_qspread_ddqm2`: stock-level weighted factor score 기반 QSpread surface

이후 ablation planner를 통해 q/model/macro/surface/factor universe 조합을 명시적으로 관리했다.

## 6. AI 자율성에 대한 관찰

이 프로젝트에서 어려웠던 부분 중 하나는 AI에게 실험 설계를 어디까지 자율적으로 맡길 것인가였다. 실험 실행, 코드 작성, manifest parsing, report 작성은 자동화하기 좋았다. 하지만 탐색 공간과 제약을 느슨하게 주면 AI가 수천 개 후보 중 우연히 좋아 보이는 top-k 조합을 새로운 가설처럼 확장하려는 경향이 있었다.

실제로 필요한 것은 “무엇이든 찾아보라”가 아니라, DDQM2 구조 안에서 허용 가능한 factor universe, macro design, q, portfolio surface를 명확히 정의하는 것이었다. 따라서 사람의 역할은 모든 실험을 직접 수행하는 것이 아니라, 탐색 공간과 검증 기준을 제한하고 data-mining artifact를 걸러내는 쪽에 가까웠다.

또 하나의 문제는 결과 attribution이다. 단순 hyperparameter tuning만 자동화했다면 AI가 어느 정도 성능 개선에 기여했는지 비교적 쉽게 기록할 수 있었을 것이다. 그러나 autoquant-lab에서는 AI가 panel construction, factor selection, walk-forward split, memory chunking, portfolio surface 같은 실험 장치 자체도 수정했다. 따라서 최종 성과가 모델의 예측력 때문인지, data representation 때문인지, evaluation protocol 때문인지, 또는 AI-driven harness engineering 때문인지 정량적으로 분해하기 어렵다.

## 7. 최종 USA-DDQM2 matrix 실험

### 7.1 공통 설정

- Model: LightGBM
- Evaluation: expanding walk-forward OOS
- Prepared panel rows: 1,250,000 (date-balanced cap, not full raw universe)
- OOS periods: 383
- Factor universe: selected 13 global/local factors
- Factor score rows per full run: 52,261,398
- q values: 0.10, 0.20, 0.30

비교한 surface는 두 가지다.

1. `weighted_factor_return_current`
   - current macro family
   - factor-return portfolio surface
2. `stock_score_qspread_ddqm2`
   - DDQM2-style U.S. macro design
   - stock-level weighted factor score QSpread surface

### 7.2 결과표

| Run | q | Macro | Surface | Periods | Cum. Return | Max DD | Mean Monthly | Vol Monthly | Turnover |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `usa_ddqm2_lightgbm_q010_selected13_currentmacro_factorret` | 0.10 | current | factor-return | 383 | 386.8638 | -0.2708 | 0.0165 | 0.0408 |  |
| `usa_ddqm2_lightgbm_q010_selected13_ddqm2macro_stockscore` | 0.10 | DDQM2 25x3 | stock-score QSpread | 383 | 5094.4375 | -0.3076 | 0.0243 | 0.0627 | 0.7319 |
| `usa_ddqm2_lightgbm_q020_selected13_currentmacro_factorret` | 0.20 | current | factor-return | 383 | 53.1742 | -0.2198 | 0.0109 | 0.0304 |  |
| `usa_ddqm2_lightgbm_q020_selected13_ddqm2macro_stockscore` | 0.20 | DDQM2 25x3 | stock-score QSpread | 383 | 5202.0665 | -0.3602 | 0.0241 | 0.0572 | 0.7193 |
| `usa_ddqm2_lightgbm_q030_selected13_currentmacro_factorret` | 0.30 | current | factor-return | 383 | 30.4420 | -0.1826 | 0.0093 | 0.0228 |  |
| `usa_ddqm2_lightgbm_q030_selected13_ddqm2macro_stockscore` | 0.30 | DDQM2 25x3 | stock-score QSpread | 383 | 4366.4377 | -0.3571 | 0.0235 | 0.0540 | 0.7139 |

### 7.3 해석

stock-score QSpread surface는 factor-return surface보다 훨씬 큰 cumulative return을 보였다. 이는 factor weight를 이용해 stock-level score를 재구성하고, 그 score로 top/bottom long-short portfolio를 다시 만들기 때문에 DDQM2의 최종 portfolio construction에 더 가깝다.

q=0.10은 DDQM2-reference setting으로 의미가 있다. 다만 U.S. adaptation 관점에서는 q=0.20과 q=0.30도 중요한 비교 대상이다.

- q=0.20 stock-score는 가장 높은 cumulative return을 기록했다.
- q=0.30 stock-score는 가장 좋은 mean/vol ratio와 가장 낮은 turnover를 보였다.
- q=0.30은 q=0.20보다 cumulative return은 낮지만 practical candidate로 더 균형 잡혀 있다.

따라서 현재 기준 aggressive candidate와 balanced candidate는 다음과 같이 정리할 수 있다.

```text
Aggressive return candidate:
usa_ddqm2_lightgbm_q020_selected13_ddqm2macro_stockscore

Balanced practical candidate:
usa_ddqm2_lightgbm_q030_selected13_ddqm2macro_stockscore
```

### 7.4 다른 CPU-friendly 모델 비교

같은 패널 계열에서 다른 CPU-friendly model도 함께 돌려서, DDQM2 구조가 LightGBM에만 의존하는지 확인했다. 이 sweep은 최종 selected13/stock-score matrix와 완전히 같은 설정은 아니다. 더 넓은 all-implemented factor-return surface 계열에서 수행한 chunked diagnostic run이며, 모델 family별 상대적인 행동을 보기 위한 비교다. 따라서 최종 headline candidate는 7.2의 USA-DDQM2 matrix로 두고, 이 절의 모델 sweep은 robustness와 model-selection evidence로 해석한다.

아래 표는 1,000,000-row chunked run에서 q=0.10 기준의 대표 결과다.

| Run | Model | q | Periods | Cum. Return | Max DD |
|---|---|---:|---:|---:|---:|
| `chunked_1000000_lightgbm_q10` | lightgbm | 0.10 | 197 | 178.9471 | -0.2281 |
| `chunked_1000000_ridge_q10` | ridge | 0.10 | 197 | 130.4248 | -0.2206 |
| `chunked_1000000_elasticnet_q10` | elasticnet | 0.10 | 197 | 130.4764 | -0.2423 |
| `chunked_1000000_random_forest_q10` | random_forest | 0.10 | 197 | 76.6173 | -0.2528 |
| `chunked_1000000_extra_trees_q10` | extra_trees | 0.10 | 197 | 64.2307 | -0.2498 |
| `chunked_1000000_baseline_mean_q10` | baseline_mean | 0.10 | 197 | 22.7204 | -0.2683 |

동일 q=0.10을 1,250,000-row date-balanced prepared panel의 fixed-holdout 계열로 확장했을 때는 순위가 달라졌다.

| Run | Model | q | Periods | Cum. Return | Max DD |
|---|---|---:|---:|---:|---:|
| `chunked_1250000_ridge_q10` | ridge | 0.10 | 257 | 5565.9987 | -0.4179 |
| `chunked_1250000_elasticnet_q10` | elasticnet | 0.10 | 257 | 4213.1459 | -0.4732 |
| `chunked_1250000_lightgbm_q10` | lightgbm | 0.10 | 257 | 1112.2701 | -0.6871 |
| `chunked_1250000_lightgbm_q10_minw01` | lightgbm | 0.10 | 257 | 195.3615 | -0.6949 |
| `chunked_1250000_baseline_mean_q10` | baseline_mean | 0.10 | 257 | 13.5862 | -0.7570 |

이 비교에서 보이는 핵심은 다음과 같다.

- 1.0M q=0.10 sweep에서는 LightGBM이 가장 높은 cumulative return을 보였고, ridge/elasticnet이 매우 가까운 2위권이었다.
- ridge는 1.0M sweep에서 max drawdown이 가장 낮았고, 1.25M 계열에서는 cumulative return도 가장 높았다. 이는 단순 선형 shrinkage model이 factor-return forecast에서 충분히 강한 baseline이 될 수 있음을 보여준다.
- elasticnet은 ridge와 거의 같은 계열의 결과를 냈다. L1/L2 regularization을 함께 쓰기 때문에 feature selection 성격이 일부 들어가지만, 현재 결과만으로 ridge 대비 명확한 우위가 있다고 보기는 어렵다.
- random forest와 extra trees는 baseline mean보다 낫지만 LightGBM/ridge/elasticnet보다는 약했다. 월별 macro feature와 factor-return label의 표본 수가 제한된 상황에서는 high-variance tree ensemble이 안정적으로 우위를 확보하지 못한 것으로 해석할 수 있다.
- baseline mean도 완전히 무의미하지는 않다. 단순 평균 예측만으로도 양의 결과가 나온다는 점은 factor-return label 자체에 persistent component가 있음을 시사한다. 그러나 모델 기반 접근 대비 성과와 drawdown이 모두 약했다.
- 1.25M-row date-balanced prepared panel의 fixed-holdout 계열에서 ridge/elasticnet이 LightGBM보다 크게 높게 나온 것은 중요한 신호지만, 이 결과를 최종 headline으로 올리지는 않았다. 해당 sweep은 최종 selected13/stock-score/walk-forward matrix와 surface 및 평가 protocol이 다르기 때문이다.

따라서 현재 결론은 “LightGBM만 답”이 아니라 다음에 가깝다.

```text
최종 USA-DDQM2 matrix: LightGBM selected13 stock-score surface를 중심으로 해석
모델 robustness: ridge/elasticnet을 반드시 후속 selected13 stock-score matrix에 포함
tree ensemble: baseline은 넘지만 현재 evidence에서는 1차 후보보다 약함
```

후속 실험에서는 q=0.20/q=0.30 stock-score QSpread surface에서도 ridge와 elasticnet을 LightGBM과 같은 조건으로 다시 비교해야 한다. 특히 monthly-refit 설정(`walk_forward_test_periods=1`)에서는 regularized linear model이 LightGBM보다 더 안정적일 가능성이 있다.

## 8. 극단 tail 결과에 대한 처리

초기 실험에서는 q=0.0025 같은 극단 tail bucket이 매우 큰 cumulative return을 보였다. 그러나 leg size가 작고, reversal 계열 factor가 중복되며, 일부 spike month에 성과가 집중되는 현상이 확인되었다. 따라서 해당 결과는 최종 headline이 아니라 diagnostic으로 분리했다.

이 경험은 AI-guided search에서 중요한 교훈을 제공한다. 탐색 공간을 과도하게 열어두면 AI는 통계적으로 좋아 보이는 극단 조합을 빠르게 찾을 수 있지만, 그 결과가 해석 가능하거나 실전적인 것은 아니다. 본 프로젝트에서는 q=0.10/q=0.20/q=0.30처럼 해석 가능한 bucket을 중심으로 다시 matrix를 구성했다.

## 9. 검증

구현과 실험 과정에서 수행한 검증은 다음과 같다.

- focused pytest for factor/DDQM2 planner path
- DDQM2 runner CLI smoke
- matrix manifest parsing
- secret scan
- git ignore check
- tracked risky file check
- GitHub publish verification

대표 검증 결과:

```text
PYTHONPATH=src:. python -m pytest tests/test_ddqm2_ablation_plan.py tests/test_factors.py -q
16 passed
```

공개 전 확인한 항목:

- WRDS/CRSP/Compustat/IBES raw data not tracked
- FRED/macro source export not tracked
- `.env` not tracked
- `EQR.md` not tracked
- PDFs/xlsx not tracked
- generated parquet/csv/site artifacts not tracked
- final report only included under `reports/`

## 10. 한계

본 결과는 아직 실전 투자전략 claim으로 보기 어렵다. 주요 한계는 다음과 같다.

- transaction cost 미반영
- borrow cost 및 shorting constraint 미반영
- slippage, market impact, capacity 미반영
- turnover가 약 0.71-0.73 수준으로 높음
- selected 13-factor universe의 경제적 중복성 검증 필요
- DDQM2 25x3 macro design은 U.S. proxy adaptation이며 원문과 일대일 대응이 아님
- ridge/elasticnet 모델 sweep은 유망하지만 최종 selected13 stock-score matrix와 동일 조건으로 아직 재실행하지 않음
- regime/year breakdown과 statistical significance 검증이 부족함

따라서 본 프로젝트는 “수익 전략 발견”이라기보다 “DDQM2 방법론의 U.S. adaptation을 위한 자동실험 하네스와 초기 OOS 결과”로 해석하는 것이 적절하다.

## 11. 결론

autoquant-lab은 DDQM2의 핵심 아이디어를 미국 주식시장 데이터에 맞게 재구성한 offline CPU-only research harness이다. 최종 구현은 selected 13-factor universe, DDQM2-style macro design, CPU-friendly factor-return forecasting, dynamic factor allocation, stock-level QSpread, walk-forward OOS matrix를 포함한다.

실험 결과는 stock-score QSpread surface가 기존 factor-return surface보다 DDQM2에 더 가까우며, q=0.20과 q=0.30이 U.S. adaptation 후보로 유의미하다는 것을 보여준다. 특히 q=0.30 stock-score QSpread는 cumulative return은 q=0.20보다 낮지만, mean/vol, turnover, drawdown 측면에서 더 균형 잡힌 후보로 해석된다. 모델 관점에서는 LightGBM이 최종 matrix의 중심이지만, ridge/elasticnet 결과가 충분히 강해 후속 matrix의 핵심 비교군으로 남겨야 한다.

동시에 이 프로젝트는 AI agent를 연구 하네스 개발에 활용할 때의 한계도 보여준다. AI는 코드 작성, 실험 실행, 결과 parsing, 보고서 작성에 유용했지만, 탐색 공간을 제한하지 않으면 data-mining artifact를 강화할 수 있다. 또한 AI가 실험 장치 자체를 바꾸기 때문에, 최종 성과를 모델 성능과 harness engineering 효과로 분해하기 어렵다. 따라서 AI-assisted quant research에서는 자동화보다 더 중요한 것이 실험 경계 설정, attribution 관리, 그리고 해석 가능한 검증 기준이다.

## 12. 다음 과제

1. Transaction cost 및 turnover stress test
2. Liquidity/microcap filter sensitivity
3. Annual/regime breakdown
4. Selected 13-factor overlap 및 economic interpretation audit
5. q=0.20/q=0.30 stock-score 후보의 robustness comparison
6. Ridge/elasticnet selected13 stock-score walk-forward 재실험
7. Monthly-refit(`walk_forward_test_periods=1`)에서 LightGBM vs ridge/elasticnet 비교
8. Macro feature ablation의 세부 분해
9. Final public report와 generated artifact 관리 정책 고도화
