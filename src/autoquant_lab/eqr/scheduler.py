"""Scheduler, adaptive budget policy, and configs-only autonomous loop for EQR experiments."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false, reportUnusedCallResult=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .config import ExperimentConfig, load_experiment_config
from .evaluation import evaluate_model
from .ledger import ClaimedJob, JobState, SQLiteJobLedger
from .metrics import REQUIRED_METRIC_KEYS
from .models.registry import create_model


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER_PATH = REPO_ROOT / "experiments" / "ledger.sqlite"
DEFAULT_PANEL_PATH = REPO_ROOT / "experiments" / "prepared" / "panel" / "monthly_labels.parquet"
DEFAULT_FEATURE_DIR = REPO_ROOT / "experiments" / "prepared" / "features"
DEFAULT_RUN_DIR = REPO_ROOT / "experiments" / "runs"

EXCLUDED_FEATURE_COLUMNS = {
    "permno",
    "permco",
    "formation_date",
    "forward_return_start",
    "forward_return_end",
    "forward_return_end_3m",
    "forward_return_end_6m",
    "ret_1m_fwd",
    "ret_3m_fwd",
    "ret_6m_fwd",
    "universe_source",
    "label_source",
}


@dataclass(frozen=True)
class BudgetPolicyResult:
    """Result of adaptive budget computation."""

    trial_count: int
    hard_cap: int
    queue_pressure_factor: float
    failure_rate_factor: float
    plateau_factor: float
    reason: str


@dataclass(frozen=True)
class ProposalResult:
    """Result of proposing a job."""

    job_id: str
    config_hash: str
    idempotency_key: str | None


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a single job through the full pipeline."""

    job_id: str
    run_id: str
    run_dir: str
    metrics: dict[str, Any]
    succeeded: bool
    error: str | None = None


def adaptive_budget(
    *,
    config: ExperimentConfig,
    ledger: SQLiteJobLedger | None = None,
    queue_pressure_weight: float = 0.3,
    failure_rate_weight: float = 0.4,
    plateau_weight: float = 0.3,
    previous_best_metrics: dict[str, float] | None = None,
    plateau_threshold: float = 0.005,
) -> BudgetPolicyResult:
    """Compute adaptive trial count from config hard cap, queue pressure, failure rate, and metric plateau.

    The budget is computed as:
        effective_trials = hard_cap * min(1.0, adjustment_factor)
    where adjustment_factor is a weighted combination of:
    - queue_pressure: reduces trials when many jobs are queued (1 - queued/total)
    - failure_rate: reduces trials when recent failures are high (1 - failure_rate)
    - plateau: reduces trials when best metric hasn't improved (0 if plateaued, 1 if improving)

    The result is always at least 1 and at most config.budget.max_trials.
    """

    hard_cap = config.budget.max_trials

    # Queue pressure: if ledger provided, compute fraction of queued jobs
    queue_pressure_factor = 1.0
    if ledger is not None:
        all_jobs = ledger.list_jobs()
        total = len(all_jobs)
        if total > 0:
            queued_count = sum(1 for job in all_jobs if job["state"] == JobState.QUEUED.value)
            queue_pressure_factor = 1.0 - (queued_count / total)
        else:
            queue_pressure_factor = 1.0

    # Failure rate: fraction of recently failed or dead-lettered jobs
    failure_rate_factor = 1.0
    if ledger is not None:
        all_jobs = ledger.list_jobs()
        terminal = [job for job in all_jobs if job["state"] in (
            JobState.SUCCEEDED.value, JobState.FAILED.value, JobState.DEAD_LETTER.value
        )]
        if terminal:
            failed = sum(1 for job in terminal if job["state"] in (
                JobState.FAILED.value, JobState.DEAD_LETTER.value
            ))
            failure_rate = failed / len(terminal)
            failure_rate_factor = 1.0 - failure_rate

    # Plateau detection: if previous best metrics exist, check if they've improved
    plateau_factor = 1.0
    if previous_best_metrics is not None and config.promotion.metric_thresholds:
        improved_count = 0
        total_metrics = 0
        for metric_name, threshold in config.promotion.metric_thresholds.items():
            if metric_name in previous_best_metrics:
                total_metrics += 1
                if previous_best_metrics[metric_name] > threshold + plateau_threshold:
                    improved_count += 1
        if total_metrics > 0:
            plateau_factor = improved_count / total_metrics
        else:
            plateau_factor = 1.0

    adjustment = (
        queue_pressure_weight * queue_pressure_factor
        + failure_rate_weight * failure_rate_factor
        + plateau_weight * plateau_factor
    )

    trial_count = max(1, min(hard_cap, max(1, int(round(hard_cap * adjustment)))))

    return BudgetPolicyResult(
        trial_count=trial_count,
        hard_cap=hard_cap,
        queue_pressure_factor=queue_pressure_factor,
        failure_rate_factor=failure_rate_factor,
        plateau_factor=plateau_factor,
        reason=(
            f"hard_cap={hard_cap}, adjustment={adjustment:.3f}, "
            f"queue_pressure={queue_pressure_factor:.3f}, "
            f"failure_rate={failure_rate_factor:.3f}, "
            f"plateau={plateau_factor:.3f}"
        ),
    )


def propose_job(
    *,
    config_path: str | Path,
    ledger: SQLiteJobLedger,
    job_type: str = "experiment",
    idempotency_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProposalResult:
    """Create a proposed job from a config file, validating the config first.

    The config is validated and its stable hash is stored in the job payload.
    No code mutation occurs - only config variants are proposed.
    """

    config = load_experiment_config(config_path)
    config_hash = config.stable_hash()

    payload: dict[str, Any] = {
        "config_path": str(config_path),
        "config_hash": config_hash,
        "model_name": config.model.name,
        "target_column": config.model.target_column,
        "budget_max_trials": config.budget.max_trials,
        "budget_retry_limit": config.budget.retry_limit,
    }

    job_metadata = {
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }

    job_id = ledger.create_job(
        job_type=job_type,
        payload=payload,
        idempotency_key=idempotency_key or config_hash,
        metadata=job_metadata,
        max_retries=config.budget.retry_limit,
    )

    return ProposalResult(
        job_id=job_id,
        config_hash=config_hash,
        idempotency_key=idempotency_key or config_hash,
    )


def claim_job(
    *,
    ledger: SQLiteJobLedger,
    worker_id: str | None = None,
    lease_seconds: int = 300,
) -> ClaimedJob | None:
    """Worker claims the next queued job with an optimistic lease."""

    effective_worker = worker_id or f"worker-{uuid4().hex[:8]}"
    return ledger.claim_next_job(effective_worker, lease_seconds=lease_seconds)


def execute_job(
    *,
    claimed: ClaimedJob,
    ledger: SQLiteJobLedger,
    config: ExperimentConfig,
    panel_path: Path | None = None,
    feature_dir: Path | None = None,
    output_dir: Path | None = None,
    max_rows: int = 0,
) -> ExecutionResult:
    """Run the full pipeline for a claimed job: prepare features, train, evaluate, persist.

    Transitions the job through: CLAIMED -> RUNNING -> EVALUATING -> PERSISTING -> RENDERING -> SUCCEEDED
    On failure, transitions to FAILED.
    """

    job_id = claimed.job_id
    run_id = claimed.run_id
    worker_id = claimed.worker_id

    try:
        # Transition: CLAIMED -> RUNNING
        ledger.start_run(job_id, worker_id=worker_id)

        # Load panel data
        effective_panel = panel_path or DEFAULT_PANEL_PATH
        if not effective_panel.exists():
            raise FileNotFoundError(f"Panel file not found: {effective_panel}")
        frame: pd.DataFrame = pd.read_parquet(effective_panel)
        frame["formation_date"] = pd.to_datetime(frame["formation_date"], errors="coerce")

        # Load or build features
        effective_feature_dir = feature_dir or DEFAULT_FEATURE_DIR
        if effective_feature_dir.exists():
            feature_frames: list[pd.DataFrame] = []
            for path in sorted(effective_feature_dir.glob("*.parquet")):
                if path.name.startswith("."):
                    continue
                ff = pd.read_parquet(path)
                if {"permno", "formation_date"}.issubset(ff.columns):
                    feature_frames.append(ff)
            if feature_frames:
                merged_features = feature_frames[0]
                for ff in feature_frames[1:]:
                    new_cols = [col for col in ff.columns if col not in merged_features.columns or col in {"permno", "formation_date"}]
                    merged_features = merged_features.merge(ff[new_cols], on=["permno", "formation_date"], how="left")
                frame = frame.merge(merged_features, on=["permno", "formation_date"], how="left", suffixes=("", "_feat"))

        # Drop rows without target
        target_column = config.model.target_column
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' not found in frame")
        frame = frame.dropna(subset=[target_column, "formation_date"]).sort_values(["formation_date", "permno"]).reset_index(drop=True)

        # Smoke cap
        if max_rows > 0 and len(frame) > max_rows:
            period_counts = frame.groupby("formation_date", sort=True).size()
            chosen: list[pd.Timestamp] = []
            running = 0
            for period, count_val in period_counts.items():
                count = int(count_val)
                if running > 0 and running + count > max_rows and len(chosen) >= 3:
                    break
                chosen.append(pd.Timestamp(str(period)))
                running += count
            capped = frame[frame["formation_date"].isin(chosen)].copy()
            frame = capped if capped["formation_date"].nunique() >= 3 else frame.head(max_rows).copy()

        # Select feature columns
        excluded = EXCLUDED_FEATURE_COLUMNS | {target_column}
        feature_columns = [
            col for col in frame.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
        ]
        if not feature_columns:
            raise ValueError("No numeric feature columns available for training")

        # Transition: RUNNING -> EVALUATING
        ledger.transition_job(job_id, JobState.EVALUATING, worker_id=worker_id)

        # Build model and evaluate
        model = create_model(config.model.name, config.model.hyperparameters)
        result = evaluate_model(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            target_column=target_column,
            period_column="formation_date",
            id_column="permno",
            validation_fraction=config.splits.validation_fraction or 0.2,
            holdout_fraction=config.splits.holdout_fraction or 0.2,
        )

        # Transition: EVALUATING -> PERSISTING
        ledger.transition_job(job_id, JobState.PERSISTING, worker_id=worker_id)

        # Persist artifacts
        effective_output = output_dir or DEFAULT_RUN_DIR
        run_dir = effective_output / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "run_id": run_id,
            "job_id": job_id,
            "model": config.model.name,
            "target": target_column,
            "feature_columns": feature_columns,
            **result.metrics,
        }

        import joblib
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
        predictions = result.predictions.rename(
            columns={"formation_date": "date", "permno": "asset_id", "actual": "target_return"}
        )
        predictions["run_id"] = run_id
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        joblib.dump(model, run_dir / "model.joblib")

        # Record artifacts in ledger
        ledger.record_artifact(
            job_id, name="metrics", uri=str(run_dir / "metrics.json"),
            run_id=run_id, artifact_type="json",
        )
        ledger.record_artifact(
            job_id, name="predictions", uri=str(run_dir / "predictions.parquet"),
            run_id=run_id, artifact_type="parquet",
        )
        ledger.record_artifact(
            job_id, name="model", uri=str(run_dir / "model.joblib"),
            run_id=run_id, artifact_type="joblib",
        )

        # Record metrics in ledger
        for metric_name in REQUIRED_METRIC_KEYS:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    ledger.record_metric(job_id, name=metric_name, value=value, run_id=run_id)

        # Transition: PERSISTING -> RENDERING
        ledger.transition_job(job_id, JobState.RENDERING, worker_id=worker_id)

        # Transition: RENDERING -> SUCCEEDED
        ledger.transition_job(job_id, JobState.SUCCEEDED, worker_id=worker_id)

        return ExecutionResult(
            job_id=job_id,
            run_id=run_id,
            run_dir=str(run_dir),
            metrics=metrics,
            succeeded=True,
        )

    except Exception as exc:
        try:
            ledger.fail_job(job_id, reason=str(exc), worker_id=worker_id)
        except Exception:
            pass
        return ExecutionResult(
            job_id=job_id,
            run_id=run_id,
            run_dir="",
            metrics={},
            succeeded=False,
            error=str(exc),
        )


def retry_job(
    *,
    job_id: str,
    ledger: SQLiteJobLedger,
    worker_id: str | None = None,
) -> dict[str, Any]:
    """Requeue a failed job if retries remain; otherwise move to dead letter."""

    job = ledger.get_job(job_id)
    if job is None:
        raise KeyError(f"Unknown job_id: {job_id}")

    if job["state"] != JobState.FAILED.value:
        raise ValueError(f"Job {job_id} is in state {job['state']}, expected FAILED")

    retry_count = int(job.get("retry_count", 0))
    max_retries = int(job.get("max_retries", 0))

    if retry_count >= max_retries:
        return ledger.dead_letter_job(
            job_id,
            reason=f"retry budget exhausted ({retry_count}/{max_retries})",
            worker_id=worker_id,
        )

    return ledger.retry_job(job_id, reason=f"retry {retry_count + 1}/{max_retries}", worker_id=worker_id)


def dead_letter_job(
    *,
    job_id: str,
    ledger: SQLiteJobLedger,
    reason: str,
    worker_id: str | None = None,
) -> dict[str, Any]:
    """Move an exhausted job to dead letter with reason."""

    return ledger.dead_letter_job(job_id, reason=reason, worker_id=worker_id)


def batch_propose(
    *,
    config_paths: Sequence[str | Path],
    ledger: SQLiteJobLedger,
    job_type: str = "experiment",
    metadata: dict[str, Any] | None = None,
) -> list[ProposalResult]:
    """Propose multiple config variants as jobs."""

    results: list[ProposalResult] = []
    for config_path in config_paths:
        result = propose_job(
            config_path=config_path,
            ledger=ledger,
            job_type=job_type,
            metadata=metadata,
        )
        results.append(result)
    return results


def run_batch(
    *,
    config_path: str | Path,
    ledger: SQLiteJobLedger,
    max_trials: int | None = None,
    worker_id: str | None = None,
    panel_path: Path | None = None,
    feature_dir: Path | None = None,
    output_dir: Path | None = None,
    max_rows: int = 0,
) -> list[ExecutionResult]:
    """Propose and execute a batch of trials from a single config.

    Uses adaptive budget to determine trial count unless max_trials is explicitly provided.
    Each trial is a separate job in the ledger with full lifecycle tracking.
    """

    config = load_experiment_config(config_path)

    # Determine trial count
    if max_trials is not None:
        trial_count = max_trials
    else:
        budget_result = adaptive_budget(config=config, ledger=ledger)
        trial_count = budget_result.trial_count

    # Propose and queue jobs
    results: list[ExecutionResult] = []
    for trial_index in range(trial_count):
        idem_key = f"{config.stable_hash()}-trial-{trial_index}"
        proposal = propose_job(
            config_path=config_path,
            ledger=ledger,
            job_type="experiment",
            idempotency_key=idem_key,
            metadata={"trial_index": trial_index, "batch_size": trial_count},
        )
        ledger.queue_job(proposal.job_id)

    # Claim and execute each job
    while True:
        claimed = claim_job(ledger=ledger, worker_id=worker_id)
        if claimed is None:
            break

        execution = execute_job(
            claimed=claimed,
            ledger=ledger,
            config=config,
            panel_path=panel_path,
            feature_dir=feature_dir,
            output_dir=output_dir,
            max_rows=max_rows,
        )
        results.append(execution)

        # If failed, attempt retry
        if not execution.succeeded:
            try:
                retry_job(job_id=execution.job_id, ledger=ledger, worker_id=worker_id)
            except (ValueError, KeyError):
                pass

    return results


def export_ledger(
    *,
    ledger: SQLiteJobLedger,
    output_path: str | Path,
    format: str = "json",
) -> Path:
    """Export ledger data to JSON or CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    jobs = ledger.list_jobs()
    events = ledger.get_events()

    if format.lower() == "csv":
        pd.DataFrame(jobs).to_csv(output, index=False)
    else:
        export_data = {
            "jobs": jobs,
            "events": events,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        output.write_text(json.dumps(export_data, indent=2, default=str), encoding="utf-8")

    return output


def list_dead_letter_jobs(
    *,
    ledger: SQLiteJobLedger,
) -> list[dict[str, Any]]:
    """List all dead letter jobs with their reasons."""

    dead_letter_jobs: list[dict[str, Any]] = []
    for job in ledger.list_jobs(state=JobState.DEAD_LETTER):
        dl_record = ledger.get_dead_letter(job["job_id"])
        entry = {**job}
        if dl_record is not None:
            entry["dead_letter_reason"] = dl_record.get("reason", "")
            entry["dead_letter_retry_count"] = dl_record.get("retry_count", 0)
        dead_letter_jobs.append(entry)
    return dead_letter_jobs