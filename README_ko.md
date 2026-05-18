# autoquant-lab

`autoquant-lab`는 DDQM2 스타일의 미국 주식 오프라인 리서치 하네스입니다. 로컬에 준비된 WRDS/FRED 계열 데이터를 읽어 시점 기준(point-in-time) 패널을 만들고, CPU-friendly model로 팩터 수익률과 포트폴리오 성과를 반복 실험합니다.

## 공개 범위

이 저장소는 **코드와 문서 스캐폴드만** 공개합니다.

- `data/`, `EQR.md`, `.env`는 커밋하지 않습니다.
- WRDS/CRSP/Compustat/IBES/FRED 원본 데이터는 저장소에 올리지 않습니다.
- `experiments/`, `site/`의 생성 산출물은 기본적으로 git에서 제외합니다.
- PDF, xlsx, csv, parquet 같은 로컬 연구 산출물도 공개 대상이 아닙니다.

## 무엇을 하는가

1. 로컬 패널과 feature를 point-in-time으로 준비합니다.
2. EQR factor registry를 이용해 종목별 factor score를 계산합니다.
3. factor score를 다음 1개월 long-short factor return으로 변환합니다.
4. macro/market feature로 CPU-friendly model을 학습합니다.
5. 예측 factor return을 factor weight로 바꿉니다.
6. weighted factor-return surface와 DDQM2-style stock-score QSpread surface를 평가합니다.

## 연구 배경

이 프로젝트는 삼성증권 DDQM/DDQM2의 연구 패턴을 참고하되, 미국 주식 데이터에 맞춰 다시 구성한 것입니다.

- DDQM: 거시/시장 변수로 상태를 만들고 팩터 포트폴리오를 회전
- DDQM2: 각 알파 팩터의 다음 1개월 long-short return을 직접 예측하고 동적으로 팩터 비중을 조정

이 레포는 이를 그대로 복제하는 것이 아니라, 미국 로컬 데이터와 CPU-only 실험 흐름으로 재현 가능한 연구 하네스를 만드는 데 목적이 있습니다.

## 현재 실험 축

- `selected_13_global_local`
- `ddqm2_25x3_us_macro`
- `expanded_us_macro`
- `stock_score_qspread_ddqm2`

대표적으로 LightGBM, ridge, elasticnet, random forest, extra trees, baseline mean 같은 CPU-friendly model을 비교했습니다.

## 디렉토리

| 경로 | 역할 |
|---|---|
| `src/autoquant_lab/eqr/` | 데이터 계약, 패널/피처/모델/백테스트 구현 |
| `configs/` | 실험 설정과 ablation plan |
| `scripts/` | 준비, 검증, 실행 CLI |
| `tests/` | 핵심 동작 테스트 |
| `reports/` | 최종 요약 보고서 |
| `experiments/` | 로컬 실행 결과 |
| `site/` | 생성된 정적 HTML |

## 사용 메모

- 로그인이나 외부 다운로드 없이 실행하는 것을 원칙으로 합니다.
- LSTM/Transformer 같은 GPU-heavy model은 범위 밖입니다.
- 공개 GitHub에는 코드, 설정, 테스트, README, 최종 보고서만 남깁니다.
