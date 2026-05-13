from __future__ import annotations
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
import json

import pytest

from autoquant_lab.eqr.ledger import LEGAL_TRANSITIONS, IllegalStateTransition, JobState, LedgerConfig, SQLiteJobLedger


def _ledger(path: Path, *, lease_seconds: int = 300, max_retries: int = 1) -> SQLiteJobLedger:
    return SQLiteJobLedger(path, config=LedgerConfig(lease_seconds=lease_seconds, max_retries=max_retries))


def test_concurrent_claim_has_single_winner(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    ledger = _ledger(db_path)
    job_id = ledger.create_job("experiment", {"candidate": "ridge"})
    ledger.queue_job(job_id)
    barrier = Barrier(8)

    def attempt_claim(worker_index: int) -> str | None:
        local_ledger = _ledger(db_path)
        barrier.wait(timeout=5)
        claimed = local_ledger.claim_job(job_id, f"worker-{worker_index}", lease_seconds=60)
        return None if claimed is None else claimed.worker_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        winners = list(executor.map(attempt_claim, range(8)))

    assert sum(worker_id is not None for worker_id in winners) == 1
    assert ledger.get_job(job_id)["state"] == JobState.CLAIMED.value  # type: ignore[index]
    assert len([event for event in ledger.get_events(job_id) if event["to_state"] == JobState.CLAIMED.value]) == 1


def test_lease_expiry_requeues_claimed_and_running_jobs(tmp_path: Path) -> None:
    assert JobState.QUEUED in LEGAL_TRANSITIONS[JobState.CLAIMED]
    assert JobState.QUEUED in LEGAL_TRANSITIONS[JobState.RUNNING]

    ledger = _ledger(tmp_path / "ledger.sqlite", lease_seconds=1)
    job_id = ledger.create_job("experiment")
    ledger.queue_job(job_id)

    assert ledger.claim_job(job_id, "worker-a", lease_seconds=1) is not None
    expired = ledger.expire_leases(now=datetime.now(timezone.utc) + timedelta(seconds=2))

    assert expired == [job_id]
    assert ledger.get_job(job_id)["state"] == JobState.QUEUED.value  # type: ignore[index]
    assert ledger.get_lease(job_id) is None

    assert ledger.claim_job(job_id, "worker-b", lease_seconds=1) is not None
    ledger.start_run(job_id, worker_id="worker-b")
    expired_again = ledger.expire_leases(now=datetime.now(timezone.utc) + timedelta(seconds=2))

    assert expired_again == [job_id]
    assert ledger.get_job(job_id)["state"] == JobState.QUEUED.value  # type: ignore[index]
    lease_events = [event for event in ledger.get_events(job_id) if event["event_type"] == "LEASE_EXPIRED"]
    assert [event["from_state"] for event in lease_events] == [JobState.CLAIMED.value, JobState.RUNNING.value]


def test_illegal_transition_is_rejected_without_event(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite")
    job_id = ledger.create_job("experiment")

    with pytest.raises(IllegalStateTransition, match="PROPOSED -> RUNNING"):
        ledger.transition_job(job_id, JobState.RUNNING)

    assert ledger.get_job(job_id)["state"] == JobState.PROPOSED.value  # type: ignore[index]
    assert [(event["from_state"], event["to_state"]) for event in ledger.get_events(job_id)] == [
        (None, JobState.PROPOSED.value)
    ]


def test_every_state_transition_writes_event_row(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite")
    job_id = ledger.create_job("experiment")
    ledger.queue_job(job_id)
    assert ledger.claim_job(job_id, "worker-a") is not None
    ledger.start_run(job_id, worker_id="worker-a")
    ledger.transition_job(job_id, JobState.EVALUATING, worker_id="worker-a")
    ledger.transition_job(job_id, JobState.PERSISTING, worker_id="worker-a")
    ledger.transition_job(job_id, JobState.RENDERING, worker_id="worker-a")
    ledger.transition_job(job_id, JobState.SUCCEEDED, worker_id="worker-a")

    transition_pairs = [
        (event["from_state"], event["to_state"])
        for event in ledger.get_events(job_id)
        if event["event_type"] == "STATE_TRANSITION"
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


def test_idempotency_key_prevents_duplicate_job_creation(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite")

    first_job_id = ledger.create_job("experiment", {"model": "ridge"}, idempotency_key="proposal-1")
    second_job_id = ledger.create_job("experiment", {"model": "lasso"}, idempotency_key="proposal-1")

    assert second_job_id == first_job_id
    assert len(ledger.list_jobs()) == 1
    assert ledger.get_job(first_job_id)["payload"] == {"model": "ridge"}  # type: ignore[index]
    assert [event["event_type"] for event in ledger.get_events(first_job_id)] == ["JOB_CREATED"]


def test_dead_letter_stores_exhausted_job_reason(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite", max_retries=0)
    job_id = ledger.create_job("experiment", {"trial": 1})
    ledger.queue_job(job_id)
    assert ledger.claim_job(job_id, "worker-a") is not None
    ledger.fail_job(job_id, reason="evaluation crashed", worker_id="worker-a")

    with pytest.raises(IllegalStateTransition, match="Retry budget exhausted"):
        ledger.transition_job(job_id, JobState.QUEUED, worker_id="worker-a")

    job = ledger.retry_job(job_id, reason="retry budget exhausted", worker_id="worker-a")
    dead_letter = ledger.get_dead_letter(job_id)

    assert job["state"] == JobState.DEAD_LETTER.value
    assert dead_letter is not None
    assert dead_letter["reason"] == "retry budget exhausted"
    assert dead_letter["payload"] == {"trial": 1}


def test_replay_export_is_deterministic_event_stream(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite")
    job_id = ledger.create_job("experiment", {"model": "ridge"}, idempotency_key="replay-1")
    ledger.queue_job(job_id)
    assert ledger.claim_job(job_id, "worker-a") is not None
    ledger.fail_job(job_id, reason="bad metric", worker_id="worker-a")
    ledger.dead_letter_job(job_id, reason="exhausted", worker_id="worker-a")

    first_export = ledger.export_replay()
    second_export = ledger.export_replay()
    replay_json = ledger.export_replay_json()

    assert first_export == second_export
    assert json.loads(replay_json) == first_export
    assert [event["event_id"] for event in first_export] == sorted(event["event_id"] for event in first_export)
    assert [event["to_state"] for event in first_export] == [
        JobState.PROPOSED.value,
        JobState.QUEUED.value,
        JobState.CLAIMED.value,
        JobState.FAILED.value,
        JobState.DEAD_LETTER.value,
    ]
