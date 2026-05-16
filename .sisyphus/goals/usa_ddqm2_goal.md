# USA-version DDQM2 goal

You are working in `~/autoquant-lab` on the Oracle/main server. Continue autonomously until the DDQM2-inspired U.S. harness can support and run the planned post-stabilization axes from `configs/ddqm2_ablation_plan.yaml`.

## Hard constraints

- Do not expose, print, upload, or commit private data, `.env`, `EQR.md`, PDFs, raw artifacts, or credentials.
- Do not run any WRDS login, browser login, external data download, or credential prompt.
- Keep the public repo as code/docs scaffold; generated `data/`, `experiments/`, `reports/`, `site/` artifacts remain ignored unless explicitly asked.
- Do not force `q=0.10`; keep it as the DDQM2-reference decile setting while preserving q=0.15/q=0.20/q=0.30 as U.S. adaptation ablations.
- Do not claim faithful DDQM2 replication until stock-level QSpread, 13-factor selection, DDQM2-like macro design, and turnover/diagnostics are implemented and verified.
- Do not commit or push unless explicitly asked.

## Current scaffold

- `configs/ddqm2_ablation_plan.yaml` enumerates runnable and planned DDQM2-style axes.
- `scripts/eqr_plan_ddqm2_ablations.py` renders runnable commands/backlog.
- Current runnable axes are q/model/min-weight against current all-factor/current-macro/weighted-factor-return surface.
- Planned axes that must become runnable after implementation:
  1. `selected_13_global_local`
  2. `ddqm2_25x3_us_macro`
  3. `expanded_us_macro`
  4. `stock_score_qspread_ddqm2`

## Goal

Make this feel like a real USA-version DDQM2, while preserving research degrees of freedom for Codex/autoresearch.

Implement in phases, verifying after each phase:

### Phase 0: stabilize current scaffold

- Re-run focused tests for the ablation planner and DDQM2 factor path.
- Confirm `scripts/eqr_plan_ddqm2_ablations.py --format commands --limit 4` works.
- Only proceed if this baseline is stable.

### Phase 1: selected_13_global_local

- Add a reproducible factor-universe selection path inspired by DDQM2:
  - global alpha: long-run factor L/S return strength over available training history;
  - local alpha: macro-state-conditioned best-state factor potential;
  - combine/deduplicate into a configurable target count, default 13.
- Add CLI/config support so DDQM2 runner can use either all implemented factors or selected factor subsets.
- Keep a U.S. adaptation option for substitutes/overrides; document what is exact DDQM2-like and what is U.S. adaptation.
- Promote `factor_universe.selected_13_global_local` to runnable in the ablation plan only when it actually executes.

### Phase 2: ddqm2_25x3_us_macro and expanded_us_macro

- Add macro feature-design variants without removing existing macro features:
  - current macro family;
  - U.S. DDQM2-style 25 base variables with current/20-period direction/60-period direction where available;
  - expanded U.S. macro set for rates, credit, inflation, dollar, oil, volatility, and activity variables where current local artifacts support them.
- Treat missing raw fields explicitly; use available local artifacts and documented proxies rather than downloads.
- Add runner/config support to select macro feature design for factor models.
- Promote `macro_feature_design.ddqm2_25x3_us_macro` and `macro_feature_design.expanded_us_macro` to runnable only when implemented and smoke-tested.

### Phase 3: stock_score_qspread_ddqm2

- Add a DDQM2-like portfolio surface:
  - use predicted factor weights;
  - compute stock-level weighted factor score per formation date;
  - form top/bottom decile long-short QSpread;
  - record leg counts, turnover, cumulative return, drawdown, mean monthly return, volatility, and any allocation concentration diagnostics.
- Keep current weighted factor-return backtest as a comparison surface.
- Add CLI/config support to choose portfolio surface.
- Promote `portfolio_surface.stock_score_qspread_ddqm2` to runnable only after tests and smoke run pass.

### Phase 4: run safe ablations

- Generate candidate commands from the planner.
- Start with small/safe smoke caps to verify behavior.
- Then run a limited practical matrix prioritizing:
  - LightGBM q10/q20/q30;
  - selected 13 vs all implemented;
  - current macro vs DDQM2 25x3 macro;
  - weighted factor-return vs stock QSpread surface;
  - min-weight 0 and DDQM2-like 3% floor.
- Avoid extreme tiny-tail headline chasing. Keep q001/q0025/q005 as diagnostics only if revisited.

## Verification requirements

- LSP/type diagnostics clean for changed files where possible.
- Focused tests pass.
- CLI smoke passes for each new surface/axis.
- Manual surface QA: run the planner and at least one smoke DDQM2 command using the new runnable axis.
- Update `reports/eqr_harness_report.md` with what became runnable and what remains backlog.
- Summarize results in a concise final note in the tmux session/log.
