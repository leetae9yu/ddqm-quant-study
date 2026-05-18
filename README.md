# autoquant-lab

[한국어](README_ko.md)

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

## Walk-forward timing and rebalancing

The portfolio surface is monthly. Each `formation_date` represents one monthly rebalance date, and labels use `ret_1m_fwd`, the next-month forward return.

In the default full-run configuration:

- portfolio weights are recomputed every month;
- the long/short stock baskets are rebuilt every month;
- each monthly portfolio is evaluated on the next one-month return;
- the forecasting model is refit once per 12-month walk-forward test fold.

For example, if a fold tests `2023-01` through `2023-12`, the model is fit only on dates before that test block, with the immediately preceding validation block kept out of training. It does **not** train on 2023 labels and then predict 2023.

The default setup is therefore:

```text
monthly portfolio rebalance
+ 1-month holding horizon
+ 12-month model refit cadence
```

For a stricter monthly-refit experiment, set the test fold to one month:

```bash
PYTHONPATH=src:. python scripts/eqr_run_ddqm2.py \
  --config configs/server_full.yaml \
  --evaluation-mode walk_forward \
  --walk-forward-test-periods 1 \
  --walk-forward-validation-periods 0
```

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
  --evaluation-mode walk_forward \
  --min-weight 0.03
```

Run focused tests:

```bash
PYTHONPATH=src:. python -m pytest tests/test_ddqm2_ablation_plan.py tests/test_factors.py -q
```

## Research status

The current matrix completed:

- one smoke run on a capped panel, and
- six full-panel LightGBM walk-forward OOS runs across q=0.10, q=0.20, and q=0.30 on the 1.25M date-balanced prepared panel.
- CPU-friendly model sweeps covering LightGBM, ridge, elasticnet, random forest, extra trees, and baseline mean.

Headline interpretation from the final report:

- q=0.20 stock-score QSpread produced the highest cumulative return.
- q=0.30 stock-score QSpread had the best mean/vol profile and lower turnover among the stock-score full runs.
- q=0.10 remains the DDQM2-reference benchmark, not a forced default.

Model-sweep interpretation:

- On the 1.0M q=0.10 chunked sweep, LightGBM had the highest cumulative return, while ridge and elasticnet were close second-tier candidates with competitive drawdowns.
- On the 1.25M-row date-balanced prepared panel, ridge and elasticnet outperformed LightGBM on cumulative return in the q=0.10 fixed-holdout family, so they should remain core follow-up models rather than mere baselines.
- Random forest and extra trees improved on the baseline mean model but were weaker than LightGBM/ridge/elasticnet in the available sweeps.
- The final headline matrix still uses the selected13 stock-score LightGBM setup because the model sweep used a different broader factor-return surface and evaluation setup.

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
