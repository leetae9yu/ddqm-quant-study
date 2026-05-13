from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false

import json
from pathlib import Path
import sqlite3

import pandas as pd

from autoquant_lab.eqr.ledger import SQLiteJobLedger
from autoquant_lab.eqr.reporting import render_site
from scripts.eqr_validate_site import validate_site


def _write_run(run_root: Path, run_id: str, rank_ic: float) -> Path:
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    metrics = {
        "run_id": run_id,
        "model": "baseline_mean",
        "target": "ret_1m_fwd",
        "feature_columns": ["crsp_momentum", "macro_rate"],
        "rank_ic": rank_ic,
        "pearson_ic": rank_ic / 2,
        "hit_rate": 0.55,
        "mse": 0.01,
        "mae": 0.08,
    }
    manifest = {
        "run_id": run_id,
        "input_artifact_path": "experiments/prepared/panel/monthly_labels.parquet",
        "output_dir": f"experiments/runs/{run_id}",
        "split_method": "chronological_train_validation_holdout",
        "train_start_date": "2020-01-31",
        "train_end_date": "2020-02-29",
        "validation_start_date": "2020-03-31",
        "validation_end_date": "2020-04-30",
        "target_column": "ret_1m_fwd",
        "feature_columns": metrics["feature_columns"],
        "metrics_artifact": "metrics.json",
        "predictions_artifact": "predictions.parquet",
        "feature_importance_artifact": "feature_importance.csv",
        "config_artifact": "config.json",
        "created_at": "2026-05-13T00:00:00+00:00",
        "config": "configs/golden_path.yaml",
        "panel": "experiments/prepared/panel/monthly_labels.parquet",
        "feature_dir": "experiments/prepared/features",
        "row_count": 4,
        "period_count": 2,
        "artifacts": ["metrics.json", "predictions.parquet", "feature_importance.csv"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({"config_path": "configs/golden_path.yaml", "model_params": {"constant": 0.0}}), encoding="utf-8")
    pd.DataFrame(
        {
            "run_id": [run_id, run_id, run_id, run_id],
            "date": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-04-30", "2020-04-30"]),
            "asset_id": [1, 2, 1, 2],
            "prediction": [0.1, 0.2, 0.15, 0.25],
            "target_return": [0.11, 0.18, 0.12, 0.22],
        }
    ).to_parquet(run_dir / "predictions.parquet", index=False)
    pd.DataFrame(
        {
            "feature": ["crsp_momentum", "macro_rate"],
            "importance": [0.7, 0.3],
            "importance_type": ["model", "model"],
            "run_id": [run_id, run_id],
        }
    ).to_csv(run_dir / "feature_importance.csv", index=False)
    return run_dir


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ledger_path = tmp_path / "experiments" / "ledger.sqlite"
    run_root = tmp_path / "experiments" / "runs"
    output = tmp_path / "site"
    reports = tmp_path / "reports"
    ledger = SQLiteJobLedger(ledger_path)
    good_job = ledger.create_job("experiment", {"candidate": "good"}, job_id="job-good")
    better_job = ledger.create_job("experiment", {"candidate": "better"}, job_id="job-better")
    dead_job = ledger.create_job("experiment", {"candidate": "dead"}, job_id="job-dead")
    run_good = _write_run(run_root, "run_good", 0.02)
    run_better = _write_run(run_root, "run_better", 0.07)
    with sqlite3.connect(ledger_path) as conn:
        conn.execute("UPDATE jobs SET state = 'SUCCEEDED', current_run_id = ? WHERE job_id = ?", ("run_good", good_job))
        conn.execute("UPDATE jobs SET state = 'SUCCEEDED', current_run_id = ? WHERE job_id = ?", ("run_better", better_job))
        conn.execute("UPDATE jobs SET state = 'DEAD_LETTER' WHERE job_id = ?", (dead_job,))
        conn.execute(
            "INSERT INTO runs (run_id, job_id, worker_id, attempt, status, started_at_utc, finished_at_utc, metadata_json) VALUES (?, ?, 'worker', 1, 'SUCCEEDED', '2026-05-13T00:00:00+00:00', '2026-05-13T00:01:00+00:00', '{}')",
            ("run_good", good_job),
        )
        conn.execute(
            "INSERT INTO runs (run_id, job_id, worker_id, attempt, status, started_at_utc, finished_at_utc, metadata_json) VALUES (?, ?, 'worker', 1, 'SUCCEEDED', '2026-05-13T00:02:00+00:00', '2026-05-13T00:03:00+00:00', '{}')",
            ("run_better", better_job),
        )
        conn.execute(
            "INSERT INTO dead_letter (job_id, reason, retry_count, payload_json, metadata_json, created_at_utc) VALUES (?, 'retry budget exhausted', 3, '{}', '{}', '2026-05-13T00:04:00+00:00')",
            (dead_job,),
        )
    ledger.record_artifact(good_job, name="metrics", uri=str(run_good / "metrics.json"), run_id="run_good", artifact_type="json")
    ledger.record_artifact(better_job, name="metrics", uri=str(run_better / "metrics.json"), run_id="run_better", artifact_type="json")
    ledger.record_metric(good_job, name="rank_ic", value=0.02, run_id="run_good")
    ledger.record_metric(better_job, name="rank_ic", value=0.07, run_id="run_better")
    return ledger_path, run_root, output, reports


def test_index_page_generation(tmp_path: Path) -> None:
    ledger_path, run_root, output, reports = _fixture(tmp_path)
    result = render_site(ledger_path=ledger_path, output_dir=output, run_root=run_root, reports_dir=reports)

    index = (output / "index.html").read_text(encoding="utf-8")
    assert result.run_count == 2
    assert "Experiment history" in index
    assert "run_good" in index
    assert (reports / "eqr_experiment_history.md").exists()
    assert (reports / "eqr_experiment_history.json").exists()


def test_run_detail_page(tmp_path: Path) -> None:
    ledger_path, run_root, output, reports = _fixture(tmp_path)
    render_site(ledger_path=ledger_path, output_dir=output, run_root=run_root, reports_dir=reports)

    detail = (output / "run_run_better.html").read_text(encoding="utf-8")
    assert "Run run_better" in detail
    assert "Predictions chart" in detail
    assert "crsp_momentum" in detail
    assert "Config diff" in detail


def test_leaderboard_sorting(tmp_path: Path) -> None:
    ledger_path, run_root, output, reports = _fixture(tmp_path)
    render_site(ledger_path=ledger_path, output_dir=output, run_root=run_root, reports_dir=reports)

    leaderboard = (output / "leaderboard.html").read_text(encoding="utf-8")
    assert leaderboard.index("run_better") < leaderboard.index("run_good")


def test_link_validation(tmp_path: Path) -> None:
    ledger_path, run_root, output, reports = _fixture(tmp_path)
    render_site(ledger_path=ledger_path, output_dir=output, run_root=run_root, reports_dir=reports)

    assert validate_site(output) == []


def test_secret_scan(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        "<!doctype html><html><head><title>x</title></head><body>api_key = 'supersecretvalue'</body></html>",
        encoding="utf-8",
    )

    errors = validate_site(site)
    assert any("Secret-like token" in error for error in errors)
