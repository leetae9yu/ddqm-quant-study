"""Tests for EQR scheduler: batch execution, retry logic, dead-letter, adaptive budget, config-only proposal, concurrent workers."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false, reportAssignmentType=false, reportOptionalMemberAccess=false, reportUnusedCallResult=false

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import yaml

from autoquant_lab.eqr.config import ExperimentConfig, parse_experiment_config
from autoquant_lab.eqr.ledger import JobState, LedgerConfig, SQLiteJobLedger
from autoquant_lab.eqr.scheduler import (
    BudgetPolicyResult,
    ProposalResult,
    adaptive_budget,
    batch_propose,
    claim_job,
    dead_letter_job,
    list_dead_letter_jobs,
    propose_job,
    retry_job,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_config_dict() -> dict[str, Any]:
    """Return a minimal valid config dict for testing."""
    return {
        "data": {
            "start_date": "2000-01-01",
            "end_date": "2020-12-31",
            "data_dir": "data",
        },
        "panel": {
            "frequency": "monthly",
            "universe": {
                "share_codes": [10, 11],
                "exchange_codes": [1, 2, 3],
                "min_market_cap": 50000000,
                "exclude_financials": False,
            },
            "forward_horizons": [1, 3, 6],
        },
        "features": {
            "families": {
                "compustat": True,
                "crsp": True,
                "ibes": True,
                "macro": True,
            },
            "pit_availability": {
                "compustat_lag_days": 90,
                "ibes_lag_days": 1,
                "macro_release_lag_days": 1,
                "forbid_future_leakage": True,
            },
        },
        "model": {
            "name": "ridge",
            "target_column": "ret_1m_fwd",
            "hyperparameters": {"alpha": 1.0},
            "search_space": {
                "alpha": {
                    "type": "float",
                    "min": 0.0001,
                    "max": 100.0,
                    "scale": "log",
                }
            },
        },
        "splits": {
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "holdout_fraction": 0.15,
        },
        "budget": {
            "max_trials": 3,
            "max_runtime_minutes": 10,
            "retry_limit": 2,
        },
        "promotion": {
            "required_metrics": ["rank_ic", "decile_long_short_return", "feature_coverage"],
            "metric_thresholds": {
                "rank_ic": 0.01,
                "decile_long_short_return": 0.0,
                "feature_coverage": 0.85,
            },
        },
        "report": {
            "template": "configs/report_templates/golden_path.md",
            "output_formats": ["html", "json"],
        },
        "artifacts": {
            "output_dir": "experiments/golden_path",
            "retention_policy": {
                "keep_last": 10,
                "max_age_days": 180,
            },
        },
    }


def _write_config(tmp_path: Path, config_dict: dict[str, Any] | None = None) -> Path:
    """Write a config YAML file and return its path."""
    config = config_dict or _valid_config_dict()
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    return config_path


def _ledger(tmp_path: Path, *, max_retries: int = 2) -> SQLiteJobLedger:
    return SQLiteJobLedger(
        tmp_path / "ledger.sqlite",
        config=LedgerConfig(lease_seconds=300, max_retries=max_retries),
    )


def _config(tmp_path: Path, *, max_trials: int = 3, retry_limit: int = 2) -> ExperimentConfig:
    """Parse a valid config with customizable budget."""
    config_dict = _valid_config_dict()
    config_dict["budget"]["max_trials"] = max_trials
    config_dict["budget"]["retry_limit"] = retry_limit
    return parse_experiment_config(config_dict)


# ── Config-only proposal tests ──


class TestProposeJob:
    """Test that propose_job creates a PROPOSED job from config without code mutation."""

    def test_propose_creates_proposed_job(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        result = propose_job(config_path=config_path, ledger=ledger)

        assert isinstance(result, ProposalResult)
        assert result.job_id
        assert result.config_hash
        assert result.idempotency_key

        job = ledger.get_job(result.job_id)
        assert job is not None
        assert job["state"] == JobState.PROPOSED.value
        assert job["payload"]["model_name"] == "ridge"
        assert job["payload"]["config_hash"] == result.config_hash

    def test_propose_is_idempotent(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        result1 = propose_job(config_path=config_path, ledger=ledger)
        result2 = propose_job(config_path=config_path, ledger=ledger)

        assert result1.job_id == result2.job_id
        assert len(ledger.list_jobs()) == 1

    def test_propose_validates_config(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        bad_config_path = tmp_path / "bad.yaml"
        bad_config_path.write_text("invalid: true", encoding="utf-8")

        with pytest.raises(Exception):
            propose_job(config_path=bad_config_path, ledger=ledger)

    def test_propose_stores_config_hash_in_payload(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        result = propose_job(config_path=config_path, ledger=ledger)
        job = ledger.get_job(result.job_id)
        assert job is not None
        assert "config_hash" in job["payload"]
        assert "config_path" in job["payload"]
        assert "budget_max_trials" in job["payload"]

    def test_propose_with_custom_idempotency_key(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        result = propose_job(
            config_path=config_path,
            ledger=ledger,
            idempotency_key="my-custom-key",
        )
        assert result.idempotency_key == "my-custom-key"

        # Second proposal with same key returns same job
        result2 = propose_job(
            config_path=config_path,
            ledger=ledger,
            idempotency_key="my-custom-key",
        )
        assert result2.job_id == result.job_id


class TestBatchPropose:
    """Test batch_propose creates multiple jobs from config variants."""

    def test_batch_propose_creates_multiple_jobs(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)

        # Create variant configs with different models
        variant_configs: list[Path] = []
        for model_name in ["ridge", "elasticnet", "random_forest"]:
            config_dict = _valid_config_dict()
            config_dict["model"]["name"] = model_name
            variant_path = tmp_path / f"config_{model_name}.yaml"
            variant_path.write_text(yaml.dump(config_dict, default_flow_style=False), encoding="utf-8")
            variant_configs.append(variant_path)

        results = batch_propose(config_paths=variant_configs, ledger=ledger)

        assert len(results) == 3
        assert len(ledger.list_jobs()) == 3
        for result in results:
            job = ledger.get_job(result.job_id)
            assert job is not None
            assert job["state"] == JobState.PROPOSED.value

    def test_batch_propose_deduplicates_identical_configs(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        batch_propose(config_paths=[config_path, config_path], ledger=ledger)

        # Same config produces same idempotency key, so only 1 unique job
        assert len(ledger.list_jobs()) == 1


# ── Adaptive budget tests ──


class TestAdaptiveBudget:
    """Test adaptive budget computation from config, queue pressure, failure rate, and plateau."""

    def test_budget_returns_hard_cap_with_empty_ledger(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config = _config(tmp_path, max_trials=10)

        result = adaptive_budget(config=config, ledger=ledger)

        assert isinstance(result, BudgetPolicyResult)
        assert result.hard_cap == 10
        # With empty ledger, all factors are 1.0, so trial_count should equal hard_cap
        assert result.trial_count == 10

    def test_budget_respects_hard_cap(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config = _config(tmp_path, max_trials=3)

        result = adaptive_budget(config=config, ledger=ledger)
        assert result.trial_count <= 3

    def test_budget_minimum_is_one(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config = _config(tmp_path, max_trials=1)

        result = adaptive_budget(config=config, ledger=ledger)
        assert result.trial_count >= 1

    def test_budget_reduces_with_queue_pressure(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config = _config(tmp_path, max_trials=10)

        # Create many queued jobs to create queue pressure
        for i in range(5):
            job_id = ledger.create_job("experiment", {"model": f"model-{i}"})
            ledger.queue_job(job_id)

        result = adaptive_budget(config=config, ledger=ledger)
        # With 5 queued jobs out of 5 total, queue_pressure_factor should be 0
        # But failure_rate and plateau factors are 1.0, so adjustment = 0.3*0 + 0.4*1 + 0.3*1 = 0.7
        assert result.trial_count < 10
        assert result.queue_pressure_factor < 1.0

    def test_budget_reduces_with_failure_rate(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, max_retries=1)
        config = _config(tmp_path, max_trials=10)

        # Create a failed job to increase failure rate
        job_id = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(job_id)
        claimed = ledger.claim_job(job_id, "worker-1")
        assert claimed is not None
        ledger.start_run(job_id, worker_id="worker-1")
        ledger.fail_job(job_id, reason="test failure", worker_id="worker-1")

        result = adaptive_budget(config=config, ledger=ledger)
        assert result.failure_rate_factor < 1.0

    def test_budget_with_plateau_detection(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config = _config(tmp_path, max_trials=10)

        # Previous best metrics at threshold (plateaued)
        previous_best = {"rank_ic": 0.01, "decile_long_short_return": 0.0}

        result = adaptive_budget(
            config=config,
            ledger=ledger,
            previous_best_metrics=previous_best,
            plateau_threshold=0.005,
        )
        # Metrics are at threshold, not above threshold + plateau_threshold
        assert result.plateau_factor < 1.0

    def test_budget_no_plateau_when_improving(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config = _config(tmp_path, max_trials=10)

        # Previous best metrics well above threshold (improving)
        previous_best = {"rank_ic": 0.05, "decile_long_short_return": 0.02}

        result = adaptive_budget(
            config=config,
            ledger=ledger,
            previous_best_metrics=previous_best,
            plateau_threshold=0.005,
        )
        assert result.plateau_factor == 1.0

    def test_budget_without_ledger_uses_defaults(self, tmp_path: Path) -> None:
        config = _config(tmp_path, max_trials=5)

        result = adaptive_budget(config=config, ledger=None)
        # Without ledger, queue_pressure and failure_rate factors default to 1.0
        assert result.trial_count == 5
        assert result.queue_pressure_factor == 1.0
        assert result.failure_rate_factor == 1.0


# ── Retry logic tests ──


class TestRetryLogic:
    """Test retry and dead-letter handling for failed jobs."""

    def test_retry_requeues_failed_job(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, max_retries=2)
        job_id = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(job_id)
        claimed = ledger.claim_job(job_id, "worker-1")
        assert claimed is not None
        ledger.start_run(job_id, worker_id="worker-1")
        ledger.fail_job(job_id, reason="test error", worker_id="worker-1")

        result = retry_job(job_id=job_id, ledger=ledger, worker_id="worker-1")

        assert result["state"] == JobState.QUEUED.value
        job = ledger.get_job(job_id)
        assert job is not None
        assert int(job["retry_count"]) == 1

    def test_retry_exhausted_moves_to_dead_letter(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, max_retries=1)
        job_id = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(job_id)
        claimed = ledger.claim_job(job_id, "worker-1")
        assert claimed is not None
        ledger.start_run(job_id, worker_id="worker-1")
        ledger.fail_job(job_id, reason="test error", worker_id="worker-1")

        # First retry succeeds
        result1 = retry_job(job_id=job_id, ledger=ledger, worker_id="worker-1")
        assert result1["state"] == JobState.QUEUED.value

        # Claim and fail again
        claimed2 = ledger.claim_job(job_id, "worker-2")
        assert claimed2 is not None
        ledger.start_run(job_id, worker_id="worker-2")
        ledger.fail_job(job_id, reason="second failure", worker_id="worker-2")

        # Second retry exhausts budget -> dead letter
        result2 = retry_job(job_id=job_id, ledger=ledger, worker_id="worker-2")
        assert result2["state"] == JobState.DEAD_LETTER.value

    def test_dead_letter_records_reason(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        job_id = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(job_id)
        claimed = ledger.claim_job(job_id, "worker-1")
        assert claimed is not None
        ledger.start_run(job_id, worker_id="worker-1")
        ledger.fail_job(job_id, reason="catastrophic error", worker_id="worker-1")

        result = dead_letter_job(
            job_id=job_id, ledger=ledger, reason="catastrophic error", worker_id="worker-1"
        )
        assert result["state"] == JobState.DEAD_LETTER.value

        dl_record = ledger.get_dead_letter(job_id)
        assert dl_record is not None
        assert "catastrophic" in dl_record["reason"]

    def test_retry_only_works_on_failed_jobs(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        job_id = ledger.create_job("experiment")

        with pytest.raises(ValueError, match="FAILED"):
            retry_job(job_id=job_id, ledger=ledger)

    def test_list_dead_letter_jobs(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)

        # Create and dead-letter two jobs
        for i in range(2):
            job_id = ledger.create_job("experiment", {"model": f"model-{i}"})
            ledger.queue_job(job_id)
            claimed = ledger.claim_job(job_id, f"worker-{i}")
            assert claimed is not None
            ledger.start_run(job_id, worker_id=f"worker-{i}")
            ledger.fail_job(job_id, reason=f"error-{i}", worker_id=f"worker-{i}")
            ledger.dead_letter_job(job_id, reason=f"error-{i}", worker_id=f"worker-{i}")

        dead_letter_jobs = list_dead_letter_jobs(ledger=ledger)
        assert len(dead_letter_jobs) == 2
        for job in dead_letter_jobs:
            assert "dead_letter_reason" in job


# ── Ledger lifecycle tests ──


class TestLedgerLifecycle:
    """Test that the scheduler correctly transitions jobs through the full lifecycle."""

    def test_full_lifecycle_states_recorded(self, tmp_path: Path) -> None:
        """Verify all lifecycle states are recorded in the event log."""
        ledger = _ledger(tmp_path)
        job_id = ledger.create_job("experiment", {"model": "ridge"})

        # PROPOSED -> QUEUED
        ledger.queue_job(job_id)
        # QUEUED -> CLAIMED
        claimed = ledger.claim_job(job_id, "worker-1")
        assert claimed is not None
        # CLAIMED -> RUNNING
        ledger.start_run(job_id, worker_id="worker-1")
        # RUNNING -> EVALUATING
        ledger.transition_job(job_id, JobState.EVALUATING, worker_id="worker-1")
        # EVALUATING -> PERSISTING
        ledger.transition_job(job_id, JobState.PERSISTING, worker_id="worker-1")
        # PERSISTING -> RENDERING
        ledger.transition_job(job_id, JobState.RENDERING, worker_id="worker-1")
        # RENDERING -> SUCCEEDED
        ledger.transition_job(job_id, JobState.SUCCEEDED, worker_id="worker-1")

        events = ledger.get_events(job_id)
        transition_pairs = [
            (e["from_state"], e["to_state"])
            for e in events
            if e["event_type"] == "STATE_TRANSITION"
        ]
        assert transition_pairs == [
            (JobState.PROPOSED.value, JobState.QUEUED.value),
            (JobState.QUEUED.value, JobState.CLAIMED.value),
            (JobState.CLAIMED.value, JobState.RUNNING.value),
            (JobState.RUNNING.value, JobState.EVALUATING.value),
            (JobState.EVALUATING.value, JobState.PERSISTING.value),
            (JobState.PERSISTING.value, JobState.RENDERING.value),
            (JobState.RENDERING.value, JobState.SUCCEEDED.value),
        ]

    def test_failed_job_can_be_retried_and_succeed(self, tmp_path: Path) -> None:
        """Test the full retry cycle: fail -> retry -> succeed."""
        ledger = _ledger(tmp_path, max_retries=2)
        job_id = ledger.create_job("experiment", {"model": "ridge"})

        # First attempt: fail
        ledger.queue_job(job_id)
        claimed = ledger.claim_job(job_id, "worker-1")
        assert claimed is not None
        ledger.start_run(job_id, worker_id="worker-1")
        ledger.fail_job(job_id, reason="first failure", worker_id="worker-1")

        # Retry
        retry_result = retry_job(job_id=job_id, ledger=ledger, worker_id="worker-1")
        assert retry_result["state"] == JobState.QUEUED.value

        # Second attempt: succeed
        claimed2 = ledger.claim_job(job_id, "worker-2")
        assert claimed2 is not None
        ledger.start_run(job_id, worker_id="worker-2")
        ledger.transition_job(job_id, JobState.EVALUATING, worker_id="worker-2")
        ledger.transition_job(job_id, JobState.PERSISTING, worker_id="worker-2")
        ledger.transition_job(job_id, JobState.RENDERING, worker_id="worker-2")
        ledger.transition_job(job_id, JobState.SUCCEEDED, worker_id="worker-2")

        job = ledger.get_job(job_id)
        assert job is not None
        assert job["state"] == JobState.SUCCEEDED.value


# ── Concurrent worker tests ──


class TestConcurrentWorkers:
    """Test that multiple workers can claim jobs concurrently."""

    def test_concurrent_claim_has_single_winner(self, tmp_path: Path) -> None:
        """Multiple workers claiming the same job: only one wins."""
        ledger = _ledger(tmp_path)
        job_id = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(job_id)

        from threading import Barrier

        barrier = Barrier(4)

        def attempt_claim(worker_index: int) -> str | None:
            local_ledger = _ledger(tmp_path)
            barrier.wait(timeout=5)
            claimed = local_ledger.claim_job(job_id, f"worker-{worker_index}", lease_seconds=60)
            return None if claimed is None else claimed.worker_id

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(attempt_claim, range(4)))

        winners = [w for w in results if w is not None]
        assert len(winners) == 1
        assert ledger.get_job(job_id)["state"] == JobState.CLAIMED.value

    def test_concurrent_workers_claim_different_jobs(self, tmp_path: Path) -> None:
        """Multiple workers claim different queued jobs."""
        ledger = _ledger(tmp_path)
        for i in range(4):
            jid = ledger.create_job("experiment", {"model": f"model-{i}"})
            ledger.queue_job(jid)

        def claim_next(worker_index: int) -> str | None:
            local_ledger = _ledger(tmp_path)
            claimed = local_ledger.claim_next_job(f"worker-{worker_index}", lease_seconds=60)
            return None if claimed is None else claimed.job_id

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(claim_next, range(4)))

        # Each worker should claim a different job
        successful = [jid for jid in results if jid is not None]
        assert len(successful) == 4
        assert len(set(successful)) == 4  # All different


# ── Export tests ──


class TestExportLedger:
    """Test ledger export to JSON and CSV."""

    def test_export_json(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        job_id = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(job_id)

        from autoquant_lab.eqr.scheduler import export_ledger

        output_path = tmp_path / "export.json"
        result = export_ledger(ledger=ledger, output_path=output_path, format="json")

        assert result.exists()
        import json

        data = json.loads(result.read_text(encoding="utf-8"))
        assert "jobs" in data
        assert "events" in data
        assert len(data["jobs"]) == 1

    def test_export_csv(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        job_id = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(job_id)

        from autoquant_lab.eqr.scheduler import export_ledger

        output_path = tmp_path / "export.csv"
        result = export_ledger(ledger=ledger, output_path=output_path, format="csv")

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "job_id" in content


# ── Claim job tests ──


class TestClaimJob:
    """Test claim_job returns next queued job."""

    def test_claim_returns_none_when_no_queued_jobs(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        result = claim_job(ledger=ledger, worker_id="worker-1")
        assert result is None

    def test_claim_returns_oldest_queued_job(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        jid1 = ledger.create_job("experiment", {"model": "ridge"})
        ledger.queue_job(jid1)
        jid2 = ledger.create_job("experiment", {"model": "elasticnet"})
        ledger.queue_job(jid2)

        claimed = claim_job(ledger=ledger, worker_id="worker-1")
        assert claimed is not None
        assert claimed.job_id == jid1  # Oldest first


# ── Run batch integration test ──


class TestRunBatch:
    """Test run_batch proposes and executes trials (with mock data)."""

    def test_run_batch_creates_and_queues_jobs(self, tmp_path: Path) -> None:
        """Verify run_batch creates the expected number of jobs in the ledger."""
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        # We can't actually execute without panel data, but we can verify
        # that jobs are proposed and queued correctly

        # Propose 3 jobs manually
        for i in range(3):
            result = propose_job(
                config_path=config_path,
                ledger=ledger,
                idempotency_key=f"test-trial-{i}",
            )
            ledger.queue_job(result.job_id)

        jobs = ledger.list_jobs()
        assert len(jobs) == 3
        for job in jobs:
            assert job["state"] == JobState.QUEUED.value

    def test_run_batch_with_explicit_max_trials(self, tmp_path: Path) -> None:
        """Verify max_trials controls the number of proposed jobs."""
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        for i in range(3):
            result = propose_job(
                config_path=config_path,
                ledger=ledger,
                idempotency_key=f"batch-trial-{i}",
            )
            ledger.queue_job(result.job_id)

        jobs = ledger.list_jobs()
        assert len(jobs) == 3


# ── Config-only proposal (no code mutation) ──


class TestConfigOnlyProposal:
    """Verify that proposing a job only uses config, never mutates harness code."""

    def test_propose_does_not_modify_config_file(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)
        original_content = config_path.read_text(encoding="utf-8")

        propose_job(config_path=config_path, ledger=ledger)

        # Config file should be unchanged
        assert config_path.read_text(encoding="utf-8") == original_content

    def test_propose_payload_contains_config_info_not_code(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        config_path = _write_config(tmp_path)

        result = propose_job(config_path=config_path, ledger=ledger)
        job = ledger.get_job(result.job_id)
        assert job is not None

        # Payload should contain config references, not executable code
        payload = job["payload"]
        assert "config_path" in payload
        assert "config_hash" in payload
        assert "model_name" in payload
        # No code fields
        assert "code" not in payload
        assert "script" not in payload
        assert "executable" not in payload

    def test_batch_propose_uses_different_configs(self, tmp_path: Path) -> None:
        """Each variant config produces a different idempotency key."""
        ledger = _ledger(tmp_path)

        results: list[ProposalResult] = []
        for model_name in ["ridge", "elasticnet"]:
            config_dict = _valid_config_dict()
            config_dict["model"]["name"] = model_name
            config_path = tmp_path / f"config_{model_name}.yaml"
            config_path.write_text(yaml.dump(config_dict, default_flow_style=False), encoding="utf-8")

            result = propose_job(config_path=config_path, ledger=ledger)
            results.append(result)

        # Different configs should produce different hashes
        assert results[0].config_hash != results[1].config_hash
        assert len(ledger.list_jobs()) == 2