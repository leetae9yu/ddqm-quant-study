# autoquant-lab

`autoquant-lab` is an offline research harness for adapting the DDQM2 idea to a U.S. equity setting. It builds point-in-time monthly equity panels from local research data, forecasts factor long-short returns from macro/market features, converts predictions into dynamic factor allocations, and evaluates walk-forward out-of-sample portfolios.

This repository is published as a **code and documentation scaffold only**. Raw WRDS-style datasets, credentials, local research notes, generated experiment artifacts, and vendor/reference PDFs are intentionally excluded.

## What this project does

The current harness supports a USA-version DDQM2 research loop:

1. Prepare a monthly `(date, security)` panel from local offline artifacts, without storing public/vendor source data in git.
2. Build feature families with point-in-time lag controls.
3. Compute stock-level factor scores from an EQR factor registry.
4. Convert factor scores into next-month factor long-short returns.
5. Train one CPU-friendly model per factor to forecast factor returns.
6. Convert predicted factor returns into non-negative factor weights.
7. Evaluate both:
   - weighted factor-return portfolios, and
   - DDQM2-style stock-level weighted factor score QSpread portfolios.
8. Record manifests, metrics, reports, and reproducible run metadata locally.

The project is inspired by Samsung Securities DDQM/DDQM2, but it is not a direct reproduction of their Korean-market production setup. It is a U.S. market adaptation with explicit differences in universe, data sources, macro features, factor definitions, and evaluation protocol.

## Current research track

The latest implementation promotes the previously planned USA-DDQM2 axes into executable runs:

- `selected_13_global_local`: DDQM2-inspired 13-factor selection from the available factor registry.
- `ddqm2_25x3_us_macro`: U.S. adaptation of the DDQM2 macro design with current, short-direction, and medium-direction style features where available.
- `expanded_us_macro`: an open macro expansion axis for additional U.S. market/macro variables supported by local artifacts.
- `stock_score_qspread_ddqm2`: stock-level weighted factor score QSpread portfolio surface.

Quantile `q` is intentionally left as a research axis:

- q=0.10 is the DDQM2-reference decile construction.
- q=0.20 and q=0.30 are U.S. adaptation settings for wider, more diversified legs.

See the final matrix report:

- [`reports/usa_ddqm2_matrix_report.md`](reports/usa_ddqm2_matrix_report.md)

## Repository layout

```text
configs/                     Safe YAML configs and ablation plans
scripts/                     CLI entrypoints for preparation, validation, DDQM2 runs, and planning
src/autoquant_lab/eqr/       Main EQR package code
src/autoquant_lab/eqr/factors/DDQM2-style factor scoring, selection, allocation, and backtests
tests/                       Focused tests for config, panel, factors, models, and reporting
reports/                     Source-controlled final report only; generated local reports remain ignored
```

Runtime and private directories are ignored:

```text
data/                        raw/local research data, including WRDS/FRED-style exports, never committed
experiments/                 generated prepared panels and run artifacts
site/                        generated static reports
.env                         credentials/local settings
*.pdf, *.xlsx                local reference/data files
```

## What is intentionally not included

This repository does **not** include:

- WRDS/CRSP/Compustat/IBES data
- FRED/macro source exports or vendor macro files
- `.env` files or credentials
- private research notes such as `EQR.md`
- DDQM/DDQM2 PDF references
- generated parquet experiment artifacts
- generated static site output

Those files are required only in the private local/server environment and are excluded by `.gitignore`.

## Main CLI examples

Render runnable DDQM2 ablation commands without executing them:

```bash
PYTHONPATH=src:. python scripts/eqr_plan_ddqm2_ablations.py --format commands --limit 8
```

Run a DDQM2-style experiment from already-prepared local artifacts:

```bash
PYTHONPATH=src:. python scripts/eqr_run_ddqm2.py \
  --config configs/server_full.yaml \
  --run-id example_usa_ddqm2_q20 \
  --quantile 0.20 \
  --model lightgbm \
  --factor-universe selected_13_global_local \
  --macro-feature-design ddqm2_25x3_us_macro \
  --portfolio-surface stock_score_qspread_ddqm2 \
  --min-weight 0.03
```

Run focused tests:

```bash
PYTHONPATH=src:. python -m pytest tests/test_ddqm2_ablation_plan.py tests/test_factors.py -q
```

## Research status

The current matrix completed:

- one smoke run on a capped panel, and
- six full-data LightGBM walk-forward OOS runs across q=0.10, q=0.20, and q=0.30.

Headline interpretation from the final report:

- q=0.20 stock-score QSpread produced the highest cumulative return.
- q=0.30 stock-score QSpread had the best mean/vol profile and lower turnover among the stock-score full runs.
- q=0.10 remains the DDQM2-reference benchmark, not a forced default.

These are gross research backtest outputs. They do not include transaction costs, borrow costs, slippage, market impact, capacity limits, or final tradability review.

## Verification

Recent focused checks used during the DDQM2-USA implementation:

```text
PYTHONPATH=src:. python -m pytest tests/test_ddqm2_ablation_plan.py tests/test_factors.py -q
16 passed
```

The runner also uses chunked factor-score generation to avoid rematerializing very large score tables during full-data runs.

## Safety and publication policy

Before publishing, verify that no private data is tracked:

```bash
git ls-files data experiments site .env EQR.md '*.pdf' '*.xlsx'
```

The expected output should be empty except for explicitly safe placeholders, if any.

## License / use

This is a research scaffold. It is not investment advice and not a deployable trading strategy.
