# DDQM2 S&P 500 적용 스터디 Heavy Prototype 실험 보고서

작성일: 2026-05-04  
프로젝트: `autoquant-lab`  
실험 ID: `ddqm2_like_150_2015_2025_lgbm`

## 1. 한눈에 보는 결론

이 스터디의 원래 목적은 **KOSPI 기준 DDQM2 방법론을 미국 S&P 500 시장에 적용해보는 것**입니다. 즉, 원 논문의 시장과 데이터 소스는 한국 시장이지만, 우리는 같은 문제의식을 미국 대형주 universe에 맞게 옮겨서 “S&P 500 버전 DDQM2”를 만들 수 있는지 확인하고 있습니다.

이번 실험은 그 목표를 향한 **S&P 500 DDQM2-like heavy prototype**입니다. “원 DDQM2를 그대로 복제한 최종 연구 결과”라기보다는, DDQM2의 핵심 흐름을 미국 시장 데이터 구조로 옮겨서 yfinance 공개 데이터로 크게 한번 돌려본 단계입니다. 이전에는 3개 종목, 6개월 데이터로 아주 작은 smoke test만 했는데, 이번에는 150개 종목과 약 11년치 데이터를 사용했습니다.

결론부터 말하면 다음과 같습니다.

- 파이프라인은 큰 규모에서도 끝까지 정상적으로 돌았습니다.
- LightGBM은 단순 baseline보다 RMSE와 MAE에서 **아주 조금 더 좋았습니다**.
- 다만 개선 폭이 작고 `best_iteration=1`이라, “강한 alpha를 찾았다”고 말하기에는 아직 부족합니다.
- 따라서 이번 결과는 **모델 성과 자랑용**이 아니라, “KOSPI DDQM2 아이디어를 S&P 500용 파이프라인으로 옮겼을 때 구조가 더 큰 데이터에서도 작동한다”는 확인에 가깝습니다.

쉽게 표현하면 이렇습니다.

> 이번 실험은 한국 시장용 DDQM2 엔진을 미국 S&P 500 도로에 올려서 시동과 주행을 확인한 단계입니다. 차가 엄청 빠르다는 결론은 아직 아닙니다.

## 2. 이번 실험이 하려는 일

원 DDQM2는 KOSPI를 기준으로 macro regime과 factor return의 관계를 모델링하는 연구입니다. 우리 스터디는 이 아이디어를 그대로 “한국 시장 복제”로 끝내는 것이 아니라, **미국 S&P 500 universe에 맞게 source adapter, factor set, label 생성 방식을 바꾸는 적용 연구**로 보고 있습니다.

DDQM2 스타일의 핵심 아이디어는 대략 다음과 같습니다.

1. 시장의 macro 상태를 본다.
2. 여러 factor가 앞으로 좋아질지 나빠질지 예측한다.
3. 그 예측을 바탕으로 factor long-short return을 모델링한다.

여기서 factor long-short return은 “특정 factor 점수가 높은 종목 묶음은 사고, 낮은 종목 묶음은 판 것처럼 계산한 수익률”입니다. 예를 들어 momentum factor라면, momentum이 높은 종목군과 낮은 종목군의 차이를 보는 식입니다.

이번 실험에서 확인한 질문은 이것입니다.

> KOSPI용 DDQM2 방법론을 S&P 500에 적용하는 방향이 맞는가? 그리고 FRED/ALFRED macro feature를 사용해서 factor long-short return을 예측하는 S&P 500 DDQM2-like pipeline이, smoke test보다 훨씬 큰 데이터에서도 제대로 작동할까?

구체적으로는 아래 흐름을 모두 확인했습니다.

```text
S&P 500 yfinance 가격/거래량 데이터
→ canonical price panel
→ factor score
→ factor long-short return label
→ macro-factor model dataset
→ LightGBM 학습/검증
→ 보고서와 dashboard에서 해석
```

## 3. 사용한 데이터 범위

이번 실험은 WRDS 없이 공개 데이터로 만든 S&P 500 prototype입니다. 그래서 시장 방향은 스터디 목적과 맞습니다. 다만 데이터 규모와 데이터 품질은 아직 최종 연구급 DDQM2 미국판에 필요한 수준과 다릅니다.

중요한 구분은 다음과 같습니다.

| 구분 | 현재 상태 |
|---|---|
| 연구 방향 | DDQM2를 S&P 500에 적용하는 방향이 맞음 |
| 시장 universe | 미국 S&P 500으로 전환됨 |
| 파이프라인 구조 | DDQM2식 macro → factor return 예측 흐름을 구현함 |
| 데이터 원천 | 아직 yfinance/FRED 중심의 공개 데이터 prototype |
| 최종 연구 데이터 | WRDS/CRSP/Compustat/IBES 기반으로 교체 필요 |

따라서 현재 한계는 “DDQM2 방향이 틀렸다”가 아니라, **S&P 500판 DDQM2를 연구급으로 만들 만큼 데이터의 깊이와 폭이 아직 부족하다**에 가깝습니다.

| 항목 | 설정 |
|---|---:|
| Universe | Wikipedia S&P 500 current-membership 중 앞쪽 150개 종목 |
| 가격 데이터 | yfinance |
| Macro feature | 기존 `expanded_macro_market_features.xlsx`의 FRED/ALFRED/yfinance macro-market features |
| 기간 | 2015-01-01 ~ 2025-12-31 |
| Formation date | 월말 기준 |
| 예측 horizon | 21 trading days |
| Long basket | factor 상위 20% |
| Short basket | factor 하위 20% |
| 최소 basket 크기 | long/short 각각 25개 이상 |
| 검증 방식 | 시간순 holdout, random split 없음 |

중요한 점은 **random split을 쓰지 않았다**는 것입니다. 금융 시계열에서 random split을 쓰면 미래 정보가 과거 학습에 섞일 수 있기 때문에, 이번에는 과거 구간으로 학습하고 더 최근 구간으로 검증했습니다.

## 4. 생성된 artifact 규모

이번 실험은 smoke test와 비교하면 꽤 큰 artifact를 만들었습니다.

| Artifact | Rows | Columns | 쉽게 말하면 |
|---|---:|---:|---|
| Price panel | 402,384 | 21 | 150개 종목의 일별 가격/거래량 정리표 |
| Factor scores | 5,889,549 | 16 | 종목별 factor 점수 대량 계산 결과 |
| Factor long-short labels | 1,919 | 21 | 월별 factor 수익률 target |
| Macro-factor model dataset | 1,919 | 96 | LightGBM에 넣을 최종 학습 테이블 |
| LightGBM experiment | 1,919 predictions | - | metric, prediction, feature importance, manifest |

여기서 가장 큰 테이블은 factor score입니다. 일별·종목별·factor별 점수를 모두 만들기 때문에 589만 rows까지 늘어났습니다. 하지만 최종 모델 학습은 월말 factor label 기준이라 1,919 rows로 압축됩니다.

## 5. 어떤 factor를 만들었나

이번 run에서는 총 15개 factor를 만들었습니다.

| Factor family | Factors |
|---|---|
| Momentum | `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m` |
| Reversal | `rev_1w`, `rev_1m` |
| Volatility | `vol_1m`, `vol_3m`, `vol_6m` |
| Drawdown | `max_dd_1m`, `max_dd_3m`, `max_dd_6m` |
| Liquidity / volume | `dollar_volume_1m`, `volume_z_1m`, `amihud_illiq_1m` |

주의할 점이 하나 있습니다. 이번 universe에는 `SPY`가 포함되지 않았습니다. 그래서 `beta_spy_6m`, `corr_spy_6m`처럼 SPY 대비 민감도를 계산하는 factor는 빠졌습니다.

다음 실험에서는 custom universe CSV에 `SPY`를 따로 넣으면 market-sensitivity factor까지 포함할 수 있습니다.

## 6. LightGBM 설정

이번에는 smoke mode가 아니라 non-smoke 설정으로 돌렸습니다.

| Parameter | Value |
|---|---:|
| Mode | non-smoke |
| Train rows | 1,514 |
| Validation rows | 405 |
| Train period | 2015-01-30 ~ 2023-08-31 |
| Validation period | 2023-09-29 ~ 2025-11-28 |
| Target | `target_long_short_return` |
| `n_estimators` | 2,000 |
| `learning_rate` | 0.02 |
| `num_leaves` | 31 |
| `early_stopping_rounds` | 100 |
| `best_iteration` | 1 |
| Validation split | 최신 20% formation dates |

설정 자체는 smoke보다 훨씬 강하게 잡았습니다. `n_estimators=2000`으로 충분히 많은 boosting round를 허용했고, `learning_rate=0.02`로 천천히 학습하게 했습니다. 또 early stopping을 100으로 두어 validation 성능이 좋아지지 않으면 멈추게 했습니다.

다만 결과적으로 `best_iteration=1`이 나왔습니다. 이 말은 “나무를 많이 쌓아도 validation 성능이 좋아지지 않았고, 사실상 초반 1번째 iteration이 가장 좋았다”는 뜻입니다. 모델이 복잡해질수록 검증 구간에서는 별 도움이 안 됐다는 신호입니다.

## 7. 성능 결과

아래 표는 validation 구간에서의 결과입니다. RMSE와 MAE는 낮을수록 좋습니다. IC는 예측값과 실제값의 상관 방향을 보는 지표로, 높을수록 좋습니다.

| Model | RMSE | MAE | R2 | Pearson IC |
|---|---:|---:|---:|---:|
| LightGBM | **0.0586885** | **0.0463509** | -0.0095 | 0.0941 |
| train_mean_return | 0.0587224 | 0.0463885 | -0.0107 | ~0.0000 |
| zero_return | 0.0592602 | 0.0468195 | -0.0293 | n/a |
| last_factor_return | 0.0769387 | 0.0599520 | -0.7350 | 0.1376 |

### 쉽게 해석하면

LightGBM이 제일 좋은 RMSE와 MAE를 냈습니다. 그래서 “모델이 아예 아무것도 못 한 것은 아니다”라고 볼 수 있습니다.

하지만 개선 폭은 매우 작습니다.

- LightGBM vs train mean RMSE 개선: 약 `0.0000339`
- LightGBM vs zero return RMSE 개선: 약 `0.0005717`
- LightGBM validation IC: `0.0941`

즉, LightGBM이 baseline보다 근소하게 앞서기는 했지만, 압도적인 차이는 아닙니다. 특히 `train_mean_return`과 거의 비슷하기 때문에, 현재 상태에서는 “macro feature가 factor return을 강하게 예측한다”고 말하기 어렵습니다.

좋게 말하면:

> 대형 prototype pipeline은 정상 작동했고, LightGBM도 baseline보다 아주 조금 나았다.

조심스럽게 말하면:

> 아직 성과라고 부르기에는 약하다. 다음 실험에서 factor 확장, 모델 regularization, walk-forward validation이 필요하다.

## 8. Smoke test와 비교하면 얼마나 커졌나

이번 run이 “빡센 run”인지 확인하려면 기존 smoke test와 비교하면 됩니다.

| 항목 | 기존 smoke | 이번 heavy prototype |
|---|---:|---:|
| Assets | 3 | 150 |
| 기간 | 2020-01 ~ 2020-06 | 2015-01 ~ 2025-12 |
| Price panel rows | 372 | 402,384 |
| Factor score rows | 3,447 | 5,889,549 |
| Label rows | 46 | 1,919 |
| Train rows | 33 | 1,514 |
| Validation rows | 13 | 405 |
| `n_estimators` | 25 | 2,000 |
| `early_stopping_rounds` | 5 | 100 |

이 비교만 보면 이번 실험은 smoke가 아닙니다. 훨씬 더 큰 prototype run입니다. 다만 “더 크다”와 “연구급이다”는 다른 말입니다. 데이터가 여전히 yfinance/current-membership 기반이기 때문입니다.

## 9. 해석할 때 조심해야 할 부분

이 섹션은 결과를 부정하려는 부분이 아니라, **결과를 어디까지 믿어도 되는지 경계선을 그어두는 부분**입니다. 숫자가 나왔다는 사실보다, 그 숫자가 어떤 데이터와 어떤 가정에서 나왔는지가 더 중요합니다.

### 9.1 데이터 쪽 한계

- **현재 S&P 500 구성종목 기준입니다.** 과거 특정 시점의 실제 S&P 500 universe가 아니라서 survivorship bias가 남아 있습니다.
- **WRDS를 쓰지 않았습니다.** CRSP delisting return, PERMNO/PERMCO, Compustat GVKEY, IBES link가 아직 없습니다.
- **yfinance는 prototype용 데이터입니다.** 가격/거래량 pipeline을 점검하기에는 유용하지만, 연구용 source of truth로 보기에는 부족합니다.
- **원 DDQM2의 KOSPI 데이터 구조를 미국 시장판으로 완전히 대체한 상태는 아닙니다.** 지금은 가격/거래량 factor 중심이고, 미국 시장용 회계·밸류에이션·수익성·투자·애널리스트 revision/dispersion feature는 아직 충분히 붙지 않았습니다.

왜 중요하냐면, 실제 과거에 존재했다가 사라진 종목이나 delisting return이 빠지면 factor return이 실제보다 좋아 보이거나 다르게 보일 수 있기 때문입니다. 그리고 원 DDQM2가 KOSPI에서 포착하려던 정보량을 미국 시장에서 재현하려면, 단순 가격 데이터만으로는 부족하고 CRSP/Compustat/IBES 계열의 point-in-time feature가 필요합니다.

### 9.2 Feature와 target 쪽 한계

- **Macro feature의 완전한 point-in-time 보장이 부족합니다.** 일부 feature는 실제 발표 시점 기준으로 완벽하게 재현되지 않았을 수 있습니다.
- **월별 target sample이 아주 큰 편은 아닙니다.** 11년치라고 해도 factor-month 기준으로는 1,919 rows입니다.

이 말은 모델이 학습할 수 있는 “정답 샘플”이 생각보다 많지 않다는 뜻입니다. 그래서 결과가 특정 기간이나 특정 factor 조합에 민감할 수 있습니다.

### 9.3 모델 쪽 한계

- **`best_iteration=1`이 나왔습니다.** 모델을 더 깊게 학습해도 validation 성능이 좋아지지 않았다는 신호입니다.
- LightGBM이 baseline보다 조금 좋았지만, 차이가 작습니다.

따라서 이번 결과는 “모델이 전혀 작동하지 않는다”는 뜻은 아닙니다. 오히려 pipeline은 잘 연결됐고 baseline보다도 약간 나았습니다. 다만 아직은 **성과 주장보다 다음 실험 설계를 위한 출발점**으로 보는 것이 더 안전합니다.

## 10. 다음 실험 제안

이번 결과를 바탕으로 다음은 아래 순서가 좋습니다.

### 10.1 Universe에 SPY 추가

`SPY`를 custom universe에 넣어서 `beta_spy_6m`, `corr_spy_6m` factor를 살립니다. 시장 민감도 factor가 빠져 있으면 DDQM2-like factor set이 조금 빈약해집니다.

기대 효과는 간단합니다. 시장 전체 움직임에 대한 민감도를 factor로 넣으면, 단순 가격/거래량 factor만 있을 때보다 macro 환경과 factor return의 연결을 더 잘 볼 수 있습니다.

### 10.2 LightGBM regularization 옵션 추가

현재 training script는 CLI에서 `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`를 직접 받지 않습니다. 다음 코드 개선에서는 이 옵션들을 추가하는 것이 좋습니다.

추천 후보는 다음과 같습니다.

| Parameter | Suggested Values |
|---|---|
| `num_leaves` | 15, 31 |
| `max_depth` | 3, 5 |
| `min_child_samples` | 30, 50 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.0, 5.0 |
| `learning_rate` | 0.01, 0.02 |
| `n_estimators` | 2,000 ~ 3,000 with early stopping |

기대 효과는 과적합을 줄이는 것입니다. 이번 run에서 `best_iteration=1`이 나온 만큼, 모델을 더 크게 만드는 것보다 **더 보수적으로 제한하는 방향**이 먼저입니다.

### 10.3 Walk-forward validation

이번에는 단일 time holdout만 했습니다. 다음에는 expanding-window 또는 rolling-window validation을 추가해서 결과가 특정 기간에만 우연히 나온 것인지 확인해야 합니다.

기대 효과는 성능의 안정성 확인입니다. 한 번의 holdout에서 조금 좋아 보이는 결과가 여러 기간에서도 반복되는지 확인할 수 있습니다.

### 10.4 WRDS adapter로 교체

최종 목표는 yfinance를 CRSP/Compustat/IBES 기반 source adapter로 바꾸는 것입니다. 이때 downstream factor, label, model, dashboard layer는 지금 만든 canonical schema를 최대한 재사용하면 됩니다.

기대 효과는 연구 신뢰도 개선입니다. WRDS 기반으로 바뀌면 survivorship bias, delisting return, point-in-time link 문제를 훨씬 더 정확하게 다룰 수 있습니다.

## 11. 최종 요약

이번 실험은 **DDQM2를 S&P 500에 적용하는 스터디의 heavy prototype으로는 성공**입니다.

성공이라고 보는 이유는 다음과 같습니다.

- KOSPI 기준 DDQM2 아이디어를 미국 S&P 500 universe로 옮기는 방향을 명확히 잡았습니다.
- 150개 S&P 500 종목과 11년치 데이터로 price panel을 만들었습니다.
- 589만 rows 규모의 factor score를 생성했습니다.
- 월별 factor long-short label을 만들었습니다.
- macro feature와 factor label을 결합했습니다.
- non-smoke LightGBM을 돌리고 baseline과 비교했습니다.
- 모든 주요 artifact validation을 통과했습니다.

하지만 모델 성능은 아직 조심스럽게 봐야 합니다.

> S&P 500판 DDQM2 파이프라인은 더 큰 규모에서도 작동한다. 다만 alpha라고 부르기에는 아직 약하다. 병목은 모델을 덜 돌린 것이 아니라, 미국 시장 DDQM2에 필요한 데이터 깊이와 폭이 아직 부족한 점이다. 다음 단계는 SPY 포함 factor 확장, regularized LightGBM 옵션 추가, walk-forward validation, 그리고 WRDS/CRSP/Compustat/IBES 기반 데이터 교체다.

이 보고서의 결론은 “좋은 성과를 냈다”가 아니라, **KOSPI DDQM2를 미국 S&P 500 버전으로 옮기기 위한 좋은 실험 기반을 만들었다**입니다.
