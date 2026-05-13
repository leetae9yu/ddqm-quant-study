# autoquant-lab

[한국어 README](README_ko.md)

Canonical scaffold for the EQR autoresearch platform.

This repository is being reset around an EQR-first layout. Active package code lives under
`src/autoquant_lab/eqr/`, active wrappers are prefixed with `scripts/eqr_`, and runtime outputs
are isolated from source-controlled configuration and harness code.

## Research background

`autoquant-lab` is a research harness for adapting the Samsung Securities Data-driven Quant Model
idea to an offline U.S. equity research setting. The local planning note `EQR.md` translates the
DDQM/DDQM2 concept into an EQR (Equity Quant Research) build target: construct a point-in-time
monthly `(date x security)` panel from WRDS-style CRSP/Compustat/IBES data plus FRED-style macro
features, then run repeatable CPU-friendly experiments that forecast stock or factor performance.

The reference PDFs in the working copy are Samsung Securities DDQM materials:

- `Data-Driven Quant Model.pdf`: describes the original DDQM approach. It engineers macro/market
  features, labels style regimes from factor returns, trains a classifier such as Random Forest to
  identify regimes, and rotates factor portfolios by predicted regime.
- `Data-Driven Quant Model2.pdf`: describes DDQM2. It drops discrete regime classification and uses
  LightGBM-style regressors to predict each alpha factor's next 1-month long-short return directly
  from macro/market features, then allocates factor weights from the predicted returns.

This repository does **not** copy Samsung's Korean-market production setup directly. It uses the same
research pattern as inspiration, while rebuilding the pipeline around local U.S. data, explicit
point-in-time joins, a SQLite experiment ledger, config-only agent autonomy, and static experiment
history. The PDFs and `EQR.md` are local research references and are not required to be committed for
the code scaffold to run.

## Active layout

| Area | Path | Purpose |
|---|---|---|
| EQR package | `src/autoquant_lab/eqr/` | Canonical schemas, validation helpers, and future research modules. |
| Configs | `configs/` | Source-controlled experiment and harness configuration. |
| Experiments | `experiments/` | Local experiment runtime outputs; ignored except `.gitkeep`. |
| Reports | `reports/` | Generated research reports; ignored except `.gitkeep`. |
| Static site | `site/` | Generated static presentation output; ignored except `.gitkeep`. |
| Skill | `skills/eqr-autoresearch/` | EQR autoresearch skill assets and instructions. |
| Tests | `tests/` | Automated checks for the EQR platform. |
| Scripts | `scripts/eqr_*.py` | Thin command wrappers for the canonical EQR path. |

## Golden-path quickstart

Run the full offline EQR autoresearch smoke from raw data checks through site rendering and CI:

```bash
python scripts/eqr_autoresearch.py golden-path \
  --config configs/golden_path.yaml \
  --max-trials 3
```

The command executes the canonical demo path:

1. Validate local raw data contracts with `scripts/eqr_validate_raw_data.py`.
2. Build point-in-time CRSP/Compustat/IBES link artifacts with `scripts/eqr_build_links.py`.
3. Prepare monthly labels and feature families with `scripts/eqr_prepare_panel.py`.
4. Validate `configs/golden_path.yaml`.
5. Queue and execute three fresh ledger-backed model/config trials.
6. Persist metrics, predictions, models, artifacts, and ledger state under `experiments/`.
7. Render reports and the static site with `scripts/eqr_render_site.py`.
8. Run `python scripts/eqr_ci.py --smoke`.

By default the golden path uses `--max-rows 50000` for smoke-sized prepared panel, feature, and trial artifacts. Pass `--max-rows 0` only on machines sized for the full offline panel.

After a successful run, verify the demo artifacts:

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

`site/index.html` shows run IDs, metrics, and promotion status for the rendered experiment history. Promotion status is research evidence only; it is not a claim that any result is economically tradable.

## Basic scaffold checks

```bash
python -c "import autoquant_lab.eqr"
python scripts/eqr_validate_import.py
python scripts/eqr_scan_secrets.py
```

## Architecture

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

### Data flow

```text
data/ (read-only)
  ├── crsp_monthly.parquet ──┐
  ├── compustat_fundq.parquet ┼──> eqr_validate_raw_data.py ──> validation report
  ├── ibes_summary.parquet ───┤         |
  └── macro_monthly.parquet ──┘         v
                                  eqr_build_links.py ──> PIT link tables
                                         |
                                         v
                               eqr_prepare_panel.py --stage labels ──> monthly_labels.parquet
                                         |
                                         v
                               eqr_prepare_panel.py --stage features ──> feature family parquets
                                         |
                                         v
                                  eqr_validate_config.py ──> config hash
                                         |
                                         v
                            eqr_autoresearch.py golden-path (trials)
                                         |
                                         v
                              experiments/ledger.sqlite + runs/<run_id>/
                                         |
                                         v
                                  eqr_render_site.py ──> site/index.html
                                         |
                                         v
                                  eqr_ci.py --smoke ──> CI report
```

## Directory structure

| Path | Contents |
|---|---|
| `configs/` | Source-controlled experiment configs and report templates. |
| `configs/golden_path.yaml` | Canonical config template with all allowed keys and defaults. |
| `data/` | User-provided offline WRDS-style raw data; read-only for this workflow. |
| `experiments/prepared/links/` | Generated point-in-time link evidence. |
| `experiments/prepared/panel/` | Generated monthly panel and label artifacts. |
| `experiments/prepared/features/` | Generated feature-family parquet files and metadata. |
| `experiments/ledger.sqlite` | SQLite finite-state ledger for jobs, runs, artifacts, metrics, and events. |
| `experiments/runs/<run_id>/` | Per-run metrics, predictions, model artifact, and config evidence. |
| `reports/` | Generated JSON/Markdown validation, CI, and experiment history reports. |
| `site/` | Generated static HTML experiment history. |
| `skills/eqr-autoresearch/` | Agent-facing EQR autoresearch operating instructions. |
| `src/autoquant_lab/eqr/` | Canonical package implementation. |
| `scripts/eqr_*.py` | Thin offline command wrappers for the active EQR path. |
| `tests/` | Pytest coverage for contracts, pipeline pieces, CLI wiring, ledger, and reporting. |

## CI

Run the full local CI contract (pytest, validators, offline guard, secret scan):

```bash
python scripts/eqr_ci.py
```

For a fast smoke check that skips expensive data scans:

```bash
python scripts/eqr_ci.py --smoke
```

The CI report is written to `reports/eqr_ci_report.json`.

## Data policy

Do not modify or regenerate files under `data/` as part of scaffold work. Existing local WRDS-style
data is treated as user data and should only be read by path resolvers or metadata checks.

The golden path is offline-only: no WRDS login prompts, no credential collection, no network downloads, and no external APIs. Generated performance metrics are engineering smoke evidence for the research harness, not investment advice or a tradability claim.

## Skill documentation

Agent operators should read `skills/eqr-autoresearch/SKILL.md` before proposing new config-only experiments. The skill documents inspection order, allowed mutable paths, promotion gates, recovery steps, and the golden-path example command.

## Legacy prototypes

Old yfinance and DDQM2-lite prototype entrypoints are not part of the active EQR path. Historical
prototype code, when retained for reference, is quarantined under `prototypes/legacy/` and should not
be used by new CI, quickstarts, or command wrappers.
