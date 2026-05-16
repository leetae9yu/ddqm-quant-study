# USA-DDQM2 Matrix Report

Generated: 2026-05-16

## 1. Summary

This report summarizes the first completed USA-version DDQM2 matrix run from the Oracle server. The run promoted the previously planned DDQM2-style axes into executable experiments:

- selected 13-factor universe via `selected_13_global_local`
- DDQM2-style U.S. macro design via `ddqm2_25x3_us_macro`
- stock-level weighted factor score QSpread via `stock_score_qspread_ddqm2`
- q remains a research axis: q=0.10 is DDQM2-reference, while q=0.20 and q=0.30 are U.S. adaptation settings

The matrix completed 1 smoke run and 6 full-data LightGBM runs. All full runs used 1,250,000 prepared panel rows and 383 walk-forward OOS months.

The strongest cumulative-return run was q=0.20 DDQM2-macro stock-score QSpread. The more balanced practical candidate is q=0.30 DDQM2-macro stock-score QSpread because it kept similar mean return with lower volatility, lower turnover, and slightly lower drawdown than q=0.20.

## 2. Protocol

### Server and artifacts

- Server: Oracle/main server, `~/autoquant-lab`
- Manifest source: `experiments/ddqm2/<run-id>/manifest.json`
- Local report inputs fetched into `/tmp/opencode/usa_ddqm2_results/`
- Generated experiments remain ignored and are not committed.

### Common full-run setup

- Model: `lightgbm`
- Evaluation mode: `walk_forward`
- OOS periods: `383`
- Prepared panel rows: `1,250,000`
- Feature rows: `1,250,000`
- Factor universe: `selected_13_global_local`
- Factor score chunking: enabled through server config
- Generated factor-score rows per full run: `52,261,398`

### Compared surfaces

Two surfaces were compared:

1. **Current weighted factor-return surface**
   - `macro_feature_design=current_macro_family`
   - `portfolio_surface=weighted_factor_return_current`
   - This keeps the previous factor-portfolio backtest surface, but uses the selected 13-factor universe.

2. **DDQM2-style stock-score QSpread surface**
   - `macro_feature_design=ddqm2_25x3_us_macro`
   - `portfolio_surface=stock_score_qspread_ddqm2`
   - This computes stock-level weighted factor scores, forms top/bottom quantile long-short portfolios, and records leg counts, turnover, and concentration diagnostics.

## 3. Completed runs

| Run | q | Macro | Surface | Periods | Cum. Return | Max DD | Mean Monthly | Vol Monthly | Turnover |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `usa_ddqm2_lightgbm_q010_selected13_currentmacro_factorret` | 0.10 | current | factor-return | 383 | 386.8638 | -0.2708 | 0.0165 | 0.0408 |  |
| `usa_ddqm2_lightgbm_q010_selected13_ddqm2macro_stockscore` | 0.10 | DDQM2 25x3 | stock-score QSpread | 383 | 5094.4375 | -0.3076 | 0.0243 | 0.0627 | 0.7319 |
| `usa_ddqm2_lightgbm_q020_selected13_currentmacro_factorret` | 0.20 | current | factor-return | 383 | 53.1742 | -0.2198 | 0.0109 | 0.0304 |  |
| `usa_ddqm2_lightgbm_q020_selected13_ddqm2macro_stockscore` | 0.20 | DDQM2 25x3 | stock-score QSpread | 383 | 5202.0665 | -0.3602 | 0.0241 | 0.0572 | 0.7193 |
| `usa_ddqm2_lightgbm_q030_selected13_currentmacro_factorret` | 0.30 | current | factor-return | 383 | 30.4420 | -0.1826 | 0.0093 | 0.0228 |  |
| `usa_ddqm2_lightgbm_q030_selected13_ddqm2macro_stockscore` | 0.30 | DDQM2 25x3 | stock-score QSpread | 383 | 4366.4377 | -0.3571 | 0.0235 | 0.0540 | 0.7139 |

Smoke run:

| Run | q | Model | Rows | Periods | Cum. Return | Max DD | Mean Monthly | Turnover |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `smoke_usa_ddqm2_q20_selected13_ddqm2macro_stockscore` | 0.20 | baseline_mean | 199,459 | 31 | 1.5887 | -0.0159 | 0.0317 | 0.7447 |

## 4. Main observations

### 4.1 Stock-score QSpread is materially different from factor-return backtest

The stock-score QSpread surface produced much larger cumulative returns than the weighted factor-return surface in this first matrix.

For the same selected 13-factor universe:

- q=0.10: stock-score QSpread `5094.44` vs factor-return `386.86`
- q=0.20: stock-score QSpread `5202.07` vs factor-return `53.17`
- q=0.30: stock-score QSpread `4366.44` vs factor-return `30.44`

This is expected directionally because the stock-score surface re-ranks securities using the dynamically weighted factor model, which is closer to the DDQM2 final portfolio construction than directly compounding weighted factor returns.

### 4.2 q=0.10 is the DDQM2-reference setting, but not automatically the best U.S. adaptation

q=0.10 should remain the reference point because DDQM2 uses decile long-short construction. In this run, however, q=0.20 and q=0.30 are still competitive as U.S. adaptation settings.

For stock-score QSpread:

| q | Cum. Return | Max DD | Mean Monthly | Vol Monthly | Mean/Vol | Turnover |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 5094.4375 | -0.3076 | 0.0243 | 0.0627 | 0.3878 | 0.7319 |
| 0.20 | 5202.0665 | -0.3602 | 0.0241 | 0.0572 | 0.4220 | 0.7193 |
| 0.30 | 4366.4377 | -0.3571 | 0.0235 | 0.0540 | 0.4352 | 0.7139 |

Interpretation:

- q=0.20 has the highest cumulative return.
- q=0.30 has the best mean/vol ratio and the lowest turnover among the stock-score runs.
- q=0.10 is useful as the DDQM2-reference benchmark, but should not be forced as the only U.S. setting.

### 4.3 Current practical candidate

The current practical candidate is:

```text
usa_ddqm2_lightgbm_q030_selected13_ddqm2macro_stockscore
```

Reason:

- It uses the more DDQM2-like surface: selected 13 factors, DDQM2-style macro design, and stock-level QSpread.
- It avoids treating q=0.10 as forced.
- It has slightly lower drawdown than q=0.20 stock-score.
- It has lower turnover than q=0.10 and q=0.20 stock-score.
- Its mean/vol ratio is the best among the three stock-score full runs.

The aggressive return candidate is:

```text
usa_ddqm2_lightgbm_q020_selected13_ddqm2macro_stockscore
```

This has the highest cumulative return, but worse drawdown than q=0.30.

## 5. Engineering notes

A full-data run exposed a memory issue after chunked factor-score generation: the runner was storing chunked score artifacts, then rematerializing all score chunks into memory. Codex patched this so chunked DDQM2 scores are not fully concatenated again.

Commit on the Oracle server:

```text
bbb5688 Avoid rematerializing chunked DDQM2 scores
```

This preserves factor-score chunking while allowing the stock-score surface to load only the selected factor scores it needs.

Recent relevant remote commits:

```text
bbb5688 Avoid rematerializing chunked DDQM2 scores
872d7bd chore(goals): add USA DDQM2 run restart script
dab4f02 docs(goals): allow verified incremental commits
c384abf docs(goals): add USA DDQM2 implementation goal
dc4c67c feat(factors): add DDQM2 ablation planner
fb98885 feat(factors): add USA DDQM2 portfolio surfaces
77dbd85 fix(factors): reduce scoring and model memory pressure
029c446 fix(panel): balance prepared row caps by date
```

## 6. Caveats

These results are still research backtests, not deployment claims.

Important limitations:

- Returns are gross research outputs.
- No transaction costs, borrow costs, market impact, slippage, or shorting constraints are applied.
- Turnover is high: roughly `0.71-0.73` for the full stock-score runs.
- The DDQM2 macro design is a U.S. adaptation, not a verified one-to-one copy of the original 25 Korean/global variables.
- The selected 13-factor universe is an implemented DDQM2-inspired selection path, but should still be audited for factor overlap and economic interpretation.
- The stock-score QSpread surface now resembles DDQM2 more closely, but a final USA-DDQM2 claim still needs cost/liquidity/regime robustness checks.

## 7. Next steps

Recommended next checks:

1. Add transaction-cost stress by turnover bucket.
2. Add annual and regime breakdown for the q20/q30 stock-score candidates.
3. Audit selected 13 factors: selected IDs, families, overlap, and duplicate exposure.
4. Compare current macro vs DDQM2 25x3 macro on the same stock-score surface if not already covered broadly enough.
5. Add liquidity and microcap sensitivity filters.
6. Re-render the local static report after deciding whether q20 or q30 should be the headline practical candidate.
