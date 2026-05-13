#!/usr/bin/env python3
"""EQR autonomous research loop: propose, claim, run-batch, retry, dead-letter, export."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from autoquant_lab.eqr.config import load_experiment_config  # noqa: E402
from autoquant_lab.eqr.ledger import JobState, LedgerConfig, SQLiteJobLedger  # noqa: E402
from autoquant_lab.eqr.scheduler import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    adaptive_budget,
    claim_job,
    execute_job,
    export_ledger,
    list_dead_letter_jobs,
    propose_job,
    retry_job,
    run_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_CONFIG = PROJECT_ROOT / "configs" / "golden_path.yaml"
DEFAULT_PANEL_PATH = PROJECT_ROOT / "experiments" / "prepared" / "panel" / "monthly_labels.parquet"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "experiments" / "prepared" / "features"
DEFAULT_RUN_DIR = PROJECT_ROOT / "experiments" / "runs"


@dataclass(frozen=True)
class GoldenPathStage:
    """One subprocess-backed golden-path preparation or validation stage."""

    name: str
    command: tuple[str, ...]


def golden_path_stage_sequence(config_path: Path, *, max_rows: int = 0) -> tuple[GoldenPathStage, ...]:
    """Return the canonical ordered subprocess stages for the EQR golden path."""

    config = str(config_path)
    row_limit_args = ("--max-rows", str(max_rows)) if max_rows > 0 else ()
    return (
        GoldenPathStage("validate_raw_data", (sys.executable, "scripts/eqr_validate_raw_data.py")),
        GoldenPathStage("build_links", (sys.executable, "scripts/eqr_build_links.py")),
        GoldenPathStage("prepare_labels", (sys.executable, "scripts/eqr_prepare_panel.py", "--stage", "labels", "--config", config, *row_limit_args)),
        GoldenPathStage("prepare_features", (sys.executable, "scripts/eqr_prepare_panel.py", "--stage", "features", "--config", config, *row_limit_args)),
        GoldenPathStage("validate_config", (sys.executable, "scripts/eqr_validate_config.py", config)),
    )


def _ledger_from_args(args: argparse.Namespace) -> SQLiteJobLedger:
    ledger_path = Path(cast(str, args.ledger))
    lease_seconds = int(getattr(args, "lease_seconds", 300))
    max_retries = int(getattr(args, "max_retries", 3))
    return SQLiteJobLedger(
        ledger_path,
        config=LedgerConfig(lease_seconds=lease_seconds, max_retries=max_retries),
    )


def cmd_propose(args: argparse.Namespace) -> int:
    """Propose a job from a config file."""
    ledger = _ledger_from_args(args)
    config_path = Path(args.config)

    result = propose_job(
        config_path=config_path,
        ledger=ledger,
        job_type=getattr(args, "job_type", "experiment"),
        idempotency_key=getattr(args, "idempotency_key", None),
    )

    # Queue the job immediately
    ledger.queue_job(result.job_id)

    output = {
        "job_id": result.job_id,
        "config_hash": result.config_hash,
        "idempotency_key": result.idempotency_key,
        "state": "QUEUED",
    }
    print(json.dumps(output, indent=2))
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    """Worker claims and executes one job."""
    ledger = _ledger_from_args(args)
    config_path = Path(args.config)
    config = load_experiment_config(config_path)

    claimed = claim_job(ledger=ledger, worker_id=getattr(args, "worker_id", None))
    if claimed is None:
        print(json.dumps({"status": "no_jobs", "message": "No queued jobs available"}))
        return 0

    execution = execute_job(
        claimed=claimed,
        ledger=ledger,
        config=config,
        panel_path=Path(args.panel) if args.panel else None,
        feature_dir=Path(args.feature_dir) if args.feature_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_rows=getattr(args, "max_rows", 0),
    )

    output = {
        "job_id": execution.job_id,
        "run_id": execution.run_id,
        "succeeded": execution.succeeded,
        "run_dir": execution.run_dir,
        "error": execution.error,
    }
    if execution.metrics:
        output["metrics"] = {
            k: v for k, v in execution.metrics.items()
            if k in {"run_id", "model", "rank_ic", "decile_long_short_return", "feature_coverage", "mse"}
        }

    print(json.dumps(output, indent=2, default=str))
    return 0 if execution.succeeded else 1


def cmd_run_batch(args: argparse.Namespace) -> int:
    """Propose and execute a batch of trials."""
    ledger = _ledger_from_args(args)
    config_path = Path(args.config)

    results = run_batch(
        config_path=config_path,
        ledger=ledger,
        max_trials=args.max_trials,
        worker_id=getattr(args, "worker_id", None),
        panel_path=Path(args.panel) if args.panel else None,
        feature_dir=Path(args.feature_dir) if args.feature_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_rows=getattr(args, "max_rows", 0),
    )

    succeeded = sum(1 for r in results if r.succeeded)
    failed = sum(1 for r in results if not r.succeeded)

    output = {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": [
            {
                "job_id": r.job_id,
                "run_id": r.run_id,
                "succeeded": r.succeeded,
                "run_dir": r.run_dir,
                "error": r.error,
            }
            for r in results
        ],
    }
    print(json.dumps(output, indent=2, default=str))
    return 0 if failed == 0 else 1


def _run_subprocess_stage(stage: GoldenPathStage) -> dict[str, Any]:
    """Run one golden-path subprocess stage and return a compact evidence record."""

    print(f"[golden-path] starting {stage.name}: {' '.join(stage.command)}")
    completed = subprocess.run(
        stage.command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    evidence = {
        "name": stage.name,
        "command": list(stage.command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"golden-path stage failed: {stage.name} (exit {completed.returncode})")
    print(f"[golden-path] completed {stage.name}")
    return evidence


def _promotion_status(config: Any, metrics: dict[str, Any], succeeded: bool) -> str:
    """Evaluate the config promotion gate for a run summary."""

    if not succeeded:
        return "not promoted"
    required = config.promotion.required_metrics
    thresholds = config.promotion.metric_thresholds
    for metric_name in required:
        value = metrics.get(metric_name)
        threshold = thresholds.get(metric_name)
        if value is None or threshold is None:
            return "pending"
        try:
            if float(value) < float(threshold):
                return "not promoted"
        except (TypeError, ValueError):
            return "pending"
    return "promoted"


def _selected_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("model", "rank_ic", "decile_long_short_return", "feature_coverage", "mse", "mae", "hit_rate")
    return {key: metrics[key] for key in keys if key in metrics}


def _execute_golden_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Propose, queue, execute, evaluate, and persist fresh golden-path trials."""

    ledger = _ledger_from_args(args)
    config_path = Path(args.config)
    config = load_experiment_config(config_path)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    max_trials = int(args.max_trials)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RUN_DIR
    panel_path = Path(args.panel) if args.panel else DEFAULT_PANEL_PATH
    feature_dir = Path(args.feature_dir) if args.feature_dir else DEFAULT_FEATURE_DIR

    for trial_index in range(max_trials):
        proposal = propose_job(
            config_path=config_path,
            ledger=ledger,
            job_type="experiment",
            idempotency_key=f"{config.stable_hash()}-golden-{batch_id}-trial-{trial_index}",
            metadata={"golden_path": True, "batch_id": batch_id, "trial_index": trial_index, "batch_size": max_trials},
        )
        ledger.queue_job(proposal.job_id, reason="golden path batch")

    summaries: list[dict[str, Any]] = []
    while len(summaries) < max_trials:
        claimed = claim_job(ledger=ledger, worker_id=getattr(args, "worker_id", None))
        if claimed is None:
            break
        execution = execute_job(
            claimed=claimed,
            ledger=ledger,
            config=config,
            panel_path=panel_path,
            feature_dir=feature_dir,
            output_dir=output_dir,
            max_rows=getattr(args, "max_rows", 0),
        )
        summaries.append(
            {
                "job_id": execution.job_id,
                "run_id": execution.run_id,
                "succeeded": execution.succeeded,
                "run_dir": execution.run_dir,
                "metrics": _selected_metrics(execution.metrics),
                "promotion_status": _promotion_status(config, execution.metrics, execution.succeeded),
                "error": execution.error,
            }
        )

    return summaries


def cmd_golden_path(args: argparse.Namespace) -> int:
    """Run the full reproducible EQR golden path from raw checks to CI smoke."""

    stage_records: list[dict[str, Any]] = []
    try:
        for stage in golden_path_stage_sequence(Path(args.config), max_rows=int(args.max_rows)):
            stage_records.append(_run_subprocess_stage(stage))

        runs = _execute_golden_trials(args)
        failed_runs = [run for run in runs if not run["succeeded"]]
        if len(runs) != int(args.max_trials):
            raise RuntimeError(f"expected {args.max_trials} golden-path runs, executed {len(runs)}")
        if failed_runs:
            raise RuntimeError(f"{len(failed_runs)} golden-path trial(s) failed")

        render_record = _run_subprocess_stage(GoldenPathStage("render_site", (sys.executable, "scripts/eqr_render_site.py", "--ledger", str(args.ledger))))
        stage_records.append(render_record)
        ci_record = _run_subprocess_stage(GoldenPathStage("ci_smoke", (sys.executable, "scripts/eqr_ci.py", "--smoke")))
        stage_records.append(ci_record)
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "stages": stage_records}, indent=2, default=str), file=sys.stderr)
        return 1

    summary = {
        "status": "succeeded",
        "config": str(Path(args.config)),
        "ledger": str(Path(args.ledger)),
        "run_count": len(runs),
        "runs": runs,
        "site": str(PROJECT_ROOT / "site" / "index.html"),
        "ci": "smoke passed",
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    """Retry failed jobs."""
    ledger = _ledger_from_args(args)

    if args.job_id:
        try:
            result = retry_job(job_id=args.job_id, ledger=ledger, worker_id=getattr(args, "worker_id", None))
            print(json.dumps({"job_id": args.job_id, "new_state": result["state"]}, indent=2))
            return 0
        except (ValueError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1
    else:
        # Retry all failed jobs
        failed_jobs = ledger.list_jobs(state=JobState.FAILED)
        retried = 0
        dead_lettered = 0
        for job in failed_jobs:
            try:
                result = retry_job(job_id=job["job_id"], ledger=ledger, worker_id=getattr(args, "worker_id", None))
                if result["state"] == JobState.QUEUED.value:
                    retried += 1
                elif result["state"] == JobState.DEAD_LETTER.value:
                    dead_lettered += 1
            except (ValueError, KeyError):
                dead_lettered += 1

        print(json.dumps({"retried": retried, "dead_lettered": dead_lettered}, indent=2))
        return 0


def cmd_dead_letter(args: argparse.Namespace) -> int:
    """List dead letter jobs."""
    ledger = _ledger_from_args(args)

    if args.job_id:
        dl = ledger.get_dead_letter(args.job_id)
        if dl is None:
            print(json.dumps({"error": f"No dead letter record for job {args.job_id}"}, indent=2))
            return 1
        print(json.dumps(dl, indent=2, default=str))
        return 0

    dead_letter_jobs = list_dead_letter_jobs(ledger=ledger)
    print(json.dumps(dead_letter_jobs, indent=2, default=str))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export ledger to JSON or CSV."""
    ledger = _ledger_from_args(args)
    output_path = export_ledger(
        ledger=ledger,
        output_path=args.output,
        format=getattr(args, "format", "json"),
    )
    print(json.dumps({"exported_to": str(output_path)}, indent=2))
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    """Compute adaptive budget for a config."""
    config = load_experiment_config(args.config)
    ledger = _ledger_from_args(args)

    result = adaptive_budget(
        config=config,
        ledger=ledger,
        queue_pressure_weight=args.queue_weight,
        failure_rate_weight=args.failure_weight,
        plateau_weight=args.plateau_weight,
    )

    print(json.dumps({
        "trial_count": result.trial_count,
        "hard_cap": result.hard_cap,
        "queue_pressure_factor": result.queue_pressure_factor,
        "failure_rate_factor": result.failure_rate_factor,
        "plateau_factor": result.plateau_factor,
        "reason": result.reason,
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EQR autonomous research loop: propose, claim, run-batch, retry, dead-letter, export.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shared arguments
    def add_common_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Path to SQLite ledger database.")
        sub.add_argument("--lease-seconds", type=int, default=300, help="Lease duration in seconds.")
        sub.add_argument("--max-retries", type=int, default=3, help="Maximum retries per job.")

    # propose
    propose_parser = subparsers.add_parser("propose", help="Propose a job from config.")
    add_common_args(propose_parser)
    propose_parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    propose_parser.add_argument("--job-type", default="experiment", help="Job type label.")
    propose_parser.add_argument("--idempotency-key", default=None, help="Idempotency key for dedup.")

    # claim
    claim_parser = subparsers.add_parser("claim", help="Worker claims and executes one job.")
    add_common_args(claim_parser)
    claim_parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    claim_parser.add_argument("--worker-id", default=None, help="Worker identifier.")
    claim_parser.add_argument("--panel", default=None, help="Path to prepared panel parquet.")
    claim_parser.add_argument("--feature-dir", default=None, help="Path to prepared features directory.")
    claim_parser.add_argument("--output-dir", default=None, help="Path to output directory.")
    claim_parser.add_argument("--max-rows", type=int, default=0, help="Max rows for smoke test (0=all).")

    # run-batch
    batch_parser = subparsers.add_parser("run-batch", help="Propose and execute a batch of trials.")
    add_common_args(batch_parser)
    batch_parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    batch_parser.add_argument("--max-trials", type=int, default=None, help="Override adaptive budget trial count.")
    batch_parser.add_argument("--worker-id", default=None, help="Worker identifier.")
    batch_parser.add_argument("--panel", default=None, help="Path to prepared panel parquet.")
    batch_parser.add_argument("--feature-dir", default=None, help="Path to prepared features directory.")
    batch_parser.add_argument("--output-dir", default=None, help="Path to output directory.")
    batch_parser.add_argument("--max-rows", type=int, default=0, help="Max rows for smoke test (0=all).")

    # golden-path
    golden_parser = subparsers.add_parser("golden-path", help="Run the full EQR golden-path smoke from raw data through CI.")
    add_common_args(golden_parser)
    golden_parser.add_argument("--config", default=str(DEFAULT_GOLDEN_CONFIG), help="Path to golden-path experiment config YAML.")
    golden_parser.add_argument("--max-trials", type=int, default=3, help="Number of fresh golden-path trials to queue and run.")
    golden_parser.add_argument("--worker-id", default="golden-path", help="Worker identifier.")
    golden_parser.add_argument("--panel", default=str(DEFAULT_PANEL_PATH), help="Path to prepared monthly panel parquet.")
    golden_parser.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR), help="Path to prepared features directory.")
    golden_parser.add_argument("--output-dir", default=str(DEFAULT_RUN_DIR), help="Path to run artifact output directory.")
    golden_parser.add_argument("--max-rows", type=int, default=50000, help="Max rows for smoke-sized panel, feature, and trial artifacts (0=all).")

    # retry
    retry_parser = subparsers.add_parser("retry", help="Retry failed jobs.")
    add_common_args(retry_parser)
    retry_parser.add_argument("--job-id", default=None, help="Specific job ID to retry (default: all failed).")
    retry_parser.add_argument("--worker-id", default=None, help="Worker identifier.")

    # dead-letter
    dl_parser = subparsers.add_parser("dead-letter", help="List dead letter jobs.")
    add_common_args(dl_parser)
    dl_parser.add_argument("--job-id", default=None, help="Specific job ID to inspect.")

    # export
    export_parser = subparsers.add_parser("export", help="Export ledger to JSON or CSV.")
    add_common_args(export_parser)
    export_parser.add_argument("--output", required=True, help="Output file path.")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format.")

    # budget
    budget_parser = subparsers.add_parser("budget", help="Compute adaptive budget for a config.")
    add_common_args(budget_parser)
    budget_parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    budget_parser.add_argument("--queue-weight", type=float, default=0.3, help="Weight for queue pressure factor.")
    budget_parser.add_argument("--failure-weight", type=float, default=0.4, help="Weight for failure rate factor.")
    budget_parser.add_argument("--plateau-weight", type=float, default=0.3, help="Weight for plateau factor.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "propose": cmd_propose,
        "claim": cmd_claim,
        "run-batch": cmd_run_batch,
        "golden-path": cmd_golden_path,
        "retry": cmd_retry,
        "dead-letter": cmd_dead_letter,
        "export": cmd_export,
        "budget": cmd_budget,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
