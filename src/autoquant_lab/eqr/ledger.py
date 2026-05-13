"""SQLite-backed finite-state job ledger for EQR autonomous runs."""
# pyright: reportExplicitAny=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4


class JobState(str, Enum):
    """Legal job states for the EQR coordination ledger."""

    PROPOSED = "PROPOSED"
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    PERSISTING = "PERSISTING"
    RENDERING = "RENDERING"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


ALL_JOB_STATES: tuple[str, ...] = tuple(state.value for state in JobState)

LEGAL_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PROPOSED: frozenset({JobState.QUEUED, JobState.FAILED, JobState.REJECTED}),
    JobState.QUEUED: frozenset({JobState.CLAIMED, JobState.FAILED, JobState.REJECTED}),
    JobState.CLAIMED: frozenset({JobState.QUEUED, JobState.RUNNING, JobState.FAILED, JobState.REJECTED}),
    JobState.RUNNING: frozenset({JobState.QUEUED, JobState.EVALUATING, JobState.FAILED, JobState.REJECTED}),
    JobState.EVALUATING: frozenset({JobState.PERSISTING, JobState.FAILED, JobState.REJECTED}),
    JobState.PERSISTING: frozenset({JobState.RENDERING, JobState.FAILED, JobState.REJECTED}),
    JobState.RENDERING: frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.REJECTED}),
    JobState.SUCCEEDED: frozenset({JobState.FAILED, JobState.REJECTED}),
    JobState.REJECTED: frozenset({JobState.FAILED}),
    JobState.FAILED: frozenset({JobState.QUEUED, JobState.DEAD_LETTER, JobState.REJECTED}),
    JobState.DEAD_LETTER: frozenset({JobState.FAILED, JobState.REJECTED}),
}

LEASE_REQUEUE_STATES: tuple[JobState, ...] = (JobState.CLAIMED, JobState.RUNNING)


class LedgerError(RuntimeError):
    """Base error for ledger failures."""


class IllegalStateTransition(LedgerError, ValueError):
    """Raised when a requested transition is not in the finite-state machine."""


@dataclass(frozen=True)
class LedgerConfig:
    """Runtime settings for lease and retry behavior."""

    lease_seconds: int = 300
    max_retries: int = 3
    sqlite_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class ClaimedJob:
    """Job returned by an optimistic claim operation."""

    job_id: str
    run_id: str
    worker_id: str
    state: JobState
    lease_expires_at_utc: str
    retry_count: int
    payload: dict[str, Any]
    metadata: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "state": self.state.value,
            "lease_expires_at_utc": self.lease_expires_at_utc,
            "retry_count": self.retry_count,
            "payload": self.payload,
            "metadata": self.metadata,
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(value: datetime | None) -> datetime:
    if value is None:
        return _utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime | None = None) -> str:
    timestamp = _to_utc(value)
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_dumps(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    if isinstance(loaded, dict):
        return loaded
    return {"value": loaded}


def _coerce_state(state: JobState | str) -> JobState:
    if isinstance(state, JobState):
        return state
    return JobState(str(state))


def _state_check(column: str) -> str:
    quoted_states = ", ".join(f"'{state}'" for state in ALL_JOB_STATES)
    return f"{column} IN ({quoted_states})"


class SQLiteJobLedger:
    """SQLite implementation of a transactional job FSM ledger."""

    def __init__(self, database_path: str | Path, config: LedgerConfig | None = None) -> None:
        self.database_path = str(database_path)
        self.config = config or LedgerConfig()
        self._memory_connection: sqlite3.Connection | None = None

        if self.database_path == ":memory:":
            self._memory_connection = sqlite3.connect(
                self.database_path,
                timeout=self.config.sqlite_timeout_seconds,
                isolation_level=None,
                check_same_thread=False,
            )
            self._configure_connection(self._memory_connection, enable_wal=False)
        else:
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            self._initialize_schema(conn)

    @contextmanager
    def _connect(self):
        if self._memory_connection is not None:
            yield self._memory_connection
            return

        use_uri = self.database_path.startswith("file:")
        conn = sqlite3.connect(
            self.database_path,
            timeout=self.config.sqlite_timeout_seconds,
            isolation_level=None,
            uri=use_uri,
        )
        try:
            self._configure_connection(conn, enable_wal=True)
            yield conn
        finally:
            conn.close()

    def _configure_connection(self, conn: sqlite3.Connection, *, enable_wal: bool) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {int(self.config.sqlite_timeout_seconds * 1000)}")
        if enable_wal:
            conn.execute("PRAGMA journal_mode = WAL")

    @contextmanager
    def _transaction(self):
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        state_check = _state_check("state")
        from_state_check = _state_check("from_state")
        to_state_check = _state_check("to_state")
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                state TEXT NOT NULL CHECK ({state_check}),
                payload_json TEXT NOT NULL DEFAULT '{{}}',
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
                max_retries INTEGER NOT NULL DEFAULT 0 CHECK (max_retries >= 0),
                current_run_id TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_state_created
                ON jobs(state, created_at_utc, job_id);

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                worker_id TEXT NOT NULL,
                attempt INTEGER NOT NULL CHECK (attempt >= 1),
                status TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                finished_at_utc TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{{}}'
            );

            CREATE INDEX IF NOT EXISTS idx_runs_job
                ON runs(job_id, started_at_utc);

            CREATE TABLE IF NOT EXISTS leases (
                lease_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id) ON DELETE CASCADE,
                run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                worker_id TEXT NOT NULL,
                acquired_at_utc TEXT NOT NULL,
                renewed_at_utc TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_leases_expiry
                ON leases(expires_at_utc, job_id);

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                from_state TEXT CHECK (from_state IS NULL OR {from_state_check}),
                to_state TEXT CHECK (to_state IS NULL OR {to_state_check}),
                worker_id TEXT,
                reason TEXT,
                payload_json TEXT NOT NULL DEFAULT '{{}}',
                occurred_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_job_order
                ON events(job_id, event_id);

            CREATE TRIGGER IF NOT EXISTS prevent_events_update
            BEFORE UPDATE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_events_delete
            BEFORE DELETE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are append-only');
            END;

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                name TEXT NOT NULL,
                uri TEXT NOT NULL,
                artifact_type TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                name TEXT NOT NULL,
                value_real REAL,
                value_json TEXT NOT NULL DEFAULT '{{}}',
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                name TEXT NOT NULL,
                uri TEXT NOT NULL,
                report_format TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dead_letter (
                dead_letter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id) ON DELETE CASCADE,
                reason TEXT NOT NULL,
                retry_count INTEGER NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{{}}',
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                created_at_utc TEXT NOT NULL
            );
            """
        )
        conn.execute("PRAGMA user_version = 1")

    def create_job(
        self,
        job_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        max_retries: int | None = None,
        job_id: str | None = None,
    ) -> str:
        """Create a proposed job, returning the existing job for duplicate idempotency keys."""

        if not job_type:
            raise ValueError("job_type is required")
        retries = self.config.max_retries if max_retries is None else max_retries
        if retries < 0:
            raise ValueError("max_retries must be non-negative")

        with self._transaction() as conn:
            if idempotency_key is not None:
                existing = conn.execute(
                    "SELECT job_id FROM jobs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    return str(existing["job_id"])

            created_job_id = job_id or uuid4().hex
            now = _format_timestamp()
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, job_type, idempotency_key, state, payload_json,
                        metadata_json, retry_count, max_retries, created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        created_job_id,
                        job_type,
                        idempotency_key,
                        JobState.PROPOSED.value,
                        _json_dumps(payload),
                        _json_dumps(metadata),
                        retries,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                if idempotency_key is None:
                    raise
                existing = conn.execute(
                    "SELECT job_id FROM jobs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is None:
                    raise
                return str(existing["job_id"])

            self._append_event(
                conn,
                job_id=created_job_id,
                run_id=None,
                event_type="JOB_CREATED",
                from_state=None,
                to_state=JobState.PROPOSED,
                worker_id=None,
                reason=None,
                payload={"job_type": job_type, "idempotency_key": idempotency_key},
                occurred_at_utc=now,
            )
            return created_job_id

    def queue_job(self, job_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Move a proposed job into the queue."""

        return self.transition_job(job_id, JobState.QUEUED, reason=reason)

    def enqueue_job(self, job_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Alias for queue_job."""

        return self.queue_job(job_id, reason=reason)

    def transition(
        self,
        job_id: str,
        to_state: JobState | str,
        *,
        worker_id: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Alias for transition_job."""

        return self.transition_job(job_id, to_state, worker_id=worker_id, reason=reason, metadata=metadata)

    def transition_job(
        self,
        job_id: str,
        to_state: JobState | str,
        *,
        worker_id: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply a legal state transition and append the matching event atomically."""

        target_state = _coerce_state(to_state)
        with self._transaction() as conn:
            self._transition_job_locked(
                conn,
                job_id=job_id,
                to_state=target_state,
                worker_id=worker_id,
                reason=reason,
                metadata=metadata,
                event_type="STATE_TRANSITION",
            )
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        return job

    def claim_next_job(self, worker_id: str, *, lease_seconds: int | None = None) -> ClaimedJob | None:
        """Claim the oldest queued job with an optimistic UPDATE guard."""

        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT job_id
                FROM jobs
                WHERE state = ?
                ORDER BY created_at_utc, job_id
                LIMIT 1
                """,
                (JobState.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            return self._claim_job_locked(conn, str(row["job_id"]), worker_id, lease_seconds=lease_seconds)

    def claim_job(self, job_id: str, worker_id: str, *, lease_seconds: int | None = None) -> ClaimedJob | None:
        """Claim a specific queued job, returning None if another worker won."""

        with self._transaction() as conn:
            return self._claim_job_locked(conn, job_id, worker_id, lease_seconds=lease_seconds)

    def claim_job_rowcount(self, job_id: str, worker_id: str, *, lease_seconds: int | None = None) -> int:
        """Return 1 when a claim succeeds and 0 when the optimistic guard loses."""

        return 1 if self.claim_job(job_id, worker_id, lease_seconds=lease_seconds) is not None else 0

    def _claim_job_locked(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int | None,
    ) -> ClaimedJob | None:
        if not worker_id:
            raise ValueError("worker_id is required")
        lease_duration = self.config.lease_seconds if lease_seconds is None else lease_seconds
        if lease_duration <= 0:
            raise ValueError("lease_seconds must be positive")

        before = conn.execute(
            "SELECT retry_count, payload_json, metadata_json FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if before is None:
            raise KeyError(f"Unknown job_id: {job_id}")

        now_dt = _utcnow()
        now = _format_timestamp(now_dt)
        expires_at = _format_timestamp(now_dt + timedelta(seconds=lease_duration))
        run_id = uuid4().hex
        cursor = conn.execute(
            """
            UPDATE jobs
            SET state = ?, current_run_id = ?, updated_at_utc = ?
            WHERE job_id = ? AND state = ?
            """,
            (JobState.CLAIMED.value, run_id, now, job_id, JobState.QUEUED.value),
        )
        if cursor.rowcount != 1:
            return None

        attempt = int(before["retry_count"]) + 1
        conn.execute(
            """
            INSERT INTO runs (run_id, job_id, worker_id, attempt, status, started_at_utc, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, job_id, worker_id, attempt, JobState.CLAIMED.value, now, _json_dumps(None)),
        )
        conn.execute(
            """
            INSERT INTO leases (lease_id, job_id, run_id, worker_id, acquired_at_utc, renewed_at_utc, expires_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                lease_id = excluded.lease_id,
                run_id = excluded.run_id,
                worker_id = excluded.worker_id,
                acquired_at_utc = excluded.acquired_at_utc,
                renewed_at_utc = excluded.renewed_at_utc,
                expires_at_utc = excluded.expires_at_utc
            """,
            (uuid4().hex, job_id, run_id, worker_id, now, now, expires_at),
        )
        self._append_event(
            conn,
            job_id=job_id,
            run_id=run_id,
            event_type="STATE_TRANSITION",
            from_state=JobState.QUEUED,
            to_state=JobState.CLAIMED,
            worker_id=worker_id,
            reason="claimed",
            payload={"lease_expires_at_utc": expires_at},
            occurred_at_utc=now,
        )
        return ClaimedJob(
            job_id=job_id,
            run_id=run_id,
            worker_id=worker_id,
            state=JobState.CLAIMED,
            lease_expires_at_utc=expires_at,
            retry_count=int(before["retry_count"]),
            payload=_json_loads(str(before["payload_json"])),
            metadata=_json_loads(str(before["metadata_json"])),
        )

    def start_run(self, job_id: str, *, worker_id: str | None = None) -> dict[str, Any]:
        """Move a claimed job to RUNNING."""

        return self.transition_job(job_id, JobState.RUNNING, worker_id=worker_id)

    def fail_job(self, job_id: str, *, reason: str, worker_id: str | None = None) -> dict[str, Any]:
        """Mark a job failed."""

        return self.transition_job(job_id, JobState.FAILED, worker_id=worker_id, reason=reason)

    def retry_job(self, job_id: str, *, reason: str | None = None, worker_id: str | None = None) -> dict[str, Any]:
        """Requeue a failed job if its retry budget is not exhausted."""

        with self._transaction() as conn:
            row = self._require_job(conn, job_id)
            if JobState(str(row["state"])) != JobState.FAILED:
                raise IllegalStateTransition("Only FAILED jobs can be retried")
            if int(row["retry_count"]) >= int(row["max_retries"]):
                self._transition_job_locked(
                    conn,
                    job_id=job_id,
                    to_state=JobState.DEAD_LETTER,
                    worker_id=worker_id,
                    reason=reason or "retry budget exhausted",
                    metadata=None,
                    event_type="STATE_TRANSITION",
                )
            else:
                self._transition_job_locked(
                    conn,
                    job_id=job_id,
                    to_state=JobState.QUEUED,
                    worker_id=worker_id,
                    reason=reason or "retry",
                    metadata=None,
                    event_type="STATE_TRANSITION",
                )
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        return job

    def dead_letter_job(self, job_id: str, *, reason: str, worker_id: str | None = None) -> dict[str, Any]:
        """Move an exhausted FAILED job into the dead-letter table."""

        return self.transition_job(job_id, JobState.DEAD_LETTER, worker_id=worker_id, reason=reason)

    def reject_job(self, job_id: str, *, reason: str, worker_id: str | None = None) -> dict[str, Any]:
        """Reject a job from any state."""

        return self.transition_job(job_id, JobState.REJECTED, worker_id=worker_id, reason=reason)

    def expire_leases(self, *, now: datetime | None = None) -> list[str]:
        """Requeue jobs in configured lease states whose leases have expired."""

        now_text = _format_timestamp(now)
        requeue_state_values = tuple(state.value for state in LEASE_REQUEUE_STATES)
        with self._transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT l.job_id, l.run_id, l.worker_id, j.state
                FROM leases AS l
                JOIN jobs AS j ON j.job_id = l.job_id
                WHERE l.expires_at_utc <= ?
                  AND j.state IN ({", ".join("?" for _ in requeue_state_values)})
                ORDER BY l.expires_at_utc, l.job_id
                """,
                (now_text, *requeue_state_values),
            ).fetchall()
            expired: list[str] = []
            for row in rows:
                job_id = str(row["job_id"])
                from_state = JobState(str(row["state"]))
                run_id = str(row["run_id"]) if row["run_id"] is not None else None
                worker_id = str(row["worker_id"])
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = ?, current_run_id = NULL, updated_at_utc = ?
                    WHERE job_id = ? AND state = ?
                    """,
                    (JobState.QUEUED.value, now_text, job_id, from_state.value),
                )
                if run_id is not None:
                    conn.execute(
                        "UPDATE runs SET status = ?, finished_at_utc = ? WHERE run_id = ?",
                        ("LEASE_EXPIRED", now_text, run_id),
                    )
                conn.execute("DELETE FROM leases WHERE job_id = ?", (job_id,))
                self._append_event(
                    conn,
                    job_id=job_id,
                    run_id=run_id,
                    event_type="LEASE_EXPIRED",
                    from_state=from_state,
                    to_state=JobState.QUEUED,
                    worker_id=worker_id,
                    reason="lease expired",
                    payload={},
                    occurred_at_utc=now_text,
                )
                expired.append(job_id)
            return expired

    def renew_lease(self, job_id: str, worker_id: str, *, lease_seconds: int | None = None) -> bool:
        """Extend an active lease owned by worker_id."""

        lease_duration = self.config.lease_seconds if lease_seconds is None else lease_seconds
        if lease_duration <= 0:
            raise ValueError("lease_seconds must be positive")
        now_dt = _utcnow()
        now = _format_timestamp(now_dt)
        expires_at = _format_timestamp(now_dt + timedelta(seconds=lease_duration))
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE leases
                SET renewed_at_utc = ?, expires_at_utc = ?
                WHERE job_id = ? AND worker_id = ?
                """,
                (now, expires_at, job_id, worker_id),
            )
            return cursor.rowcount == 1

    def record_artifact(
        self,
        job_id: str,
        *,
        name: str,
        uri: str,
        run_id: str | None = None,
        artifact_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Record an artifact produced by a job or run."""

        with self._transaction() as conn:
            self._require_job(conn, job_id)
            cursor = conn.execute(
                """
                INSERT INTO artifacts (job_id, run_id, name, uri, artifact_type, metadata_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, run_id, name, uri, artifact_type, _json_dumps(metadata), _format_timestamp()),
            )
            artifact_id = cursor.lastrowid
            if artifact_id is None:
                raise LedgerError("SQLite did not return an artifact id")
            return artifact_id

    def record_metric(
        self,
        job_id: str,
        *,
        name: str,
        value: float | int | Mapping[str, Any],
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Record a numeric or structured metric for a job."""

        if isinstance(value, int | float):
            value_real = float(value)
            value_json = _json_dumps({"value": value})
        else:
            value_real = None
            value_json = _json_dumps(value)
        with self._transaction() as conn:
            self._require_job(conn, job_id)
            cursor = conn.execute(
                """
                INSERT INTO metrics (job_id, run_id, name, value_real, value_json, metadata_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, run_id, name, value_real, value_json, _json_dumps(metadata), _format_timestamp()),
            )
            metric_id = cursor.lastrowid
            if metric_id is None:
                raise LedgerError("SQLite did not return a metric id")
            return metric_id

    def record_report(
        self,
        job_id: str,
        *,
        name: str,
        uri: str,
        report_format: str,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Record a rendered report for a job."""

        with self._transaction() as conn:
            self._require_job(conn, job_id)
            cursor = conn.execute(
                """
                INSERT INTO reports (job_id, run_id, name, uri, report_format, metadata_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, run_id, name, uri, report_format, _json_dumps(metadata), _format_timestamp()),
            )
            report_id = cursor.lastrowid
            if report_id is None:
                raise LedgerError("SQLite did not return a report id")
            return report_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Fetch one job with decoded JSON fields."""

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._job_row_to_dict(row)

    def list_jobs(self, *, state: JobState | str | None = None) -> list[dict[str, Any]]:
        """List jobs in deterministic creation order."""

        with self._connect() as conn:
            if state is None:
                rows = conn.execute("SELECT * FROM jobs ORDER BY created_at_utc, job_id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE state = ? ORDER BY created_at_utc, job_id",
                    (_coerce_state(state).value,),
                ).fetchall()
        return [self._job_row_to_dict(row) for row in rows]

    def get_events(self, job_id: str | None = None) -> list[dict[str, Any]]:
        """Return append-only events in event_id order."""

        with self._connect() as conn:
            if job_id is None:
                rows = conn.execute("SELECT * FROM events ORDER BY event_id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events WHERE job_id = ? ORDER BY event_id",
                    (job_id,),
                ).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    def get_lease(self, job_id: str) -> dict[str, Any] | None:
        """Fetch the active lease for a job, if any."""

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM leases WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_dead_letter(self, job_id: str) -> dict[str, Any] | None:
        """Fetch a dead-letter record for a job, if any."""

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM dead_letter WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["payload"] = _json_loads(record.pop("payload_json"))
        record["metadata"] = _json_loads(record.pop("metadata_json"))
        return record

    def export_replay(self) -> list[dict[str, Any]]:
        """Export the event stream deterministically for replay."""

        return self.get_events()

    def export_replay_json(self) -> str:
        """Export the replay stream as stable JSON."""

        return json.dumps(self.export_replay(), sort_keys=True, separators=(",", ":"))

    def _transition_job_locked(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        to_state: JobState,
        worker_id: str | None,
        reason: str | None,
        metadata: Mapping[str, Any] | None,
        event_type: str,
    ) -> None:
        row = self._require_job(conn, job_id)
        from_state = JobState(str(row["state"]))
        if not self._is_legal_transition(from_state, to_state):
            raise IllegalStateTransition(f"Illegal transition: {from_state.value} -> {to_state.value}")
        if from_state == JobState.FAILED and to_state == JobState.QUEUED:
            if int(row["retry_count"]) >= int(row["max_retries"]):
                raise IllegalStateTransition("Retry budget exhausted; move job to DEAD_LETTER")

        now = _format_timestamp()
        run_id = str(row["current_run_id"]) if row["current_run_id"] is not None else None
        retry_increment = 1 if from_state == JobState.FAILED and to_state == JobState.QUEUED else 0
        clear_run = to_state in {
            JobState.QUEUED,
            JobState.SUCCEEDED,
            JobState.REJECTED,
            JobState.FAILED,
            JobState.DEAD_LETTER,
        }
        conn.execute(
            """
            UPDATE jobs
            SET state = ?,
                retry_count = retry_count + ?,
                current_run_id = CASE WHEN ? THEN NULL ELSE current_run_id END,
                updated_at_utc = ?
            WHERE job_id = ?
            """,
            (to_state.value, retry_increment, 1 if clear_run else 0, now, job_id),
        )
        if run_id is not None and to_state in {JobState.SUCCEEDED, JobState.REJECTED, JobState.FAILED, JobState.DEAD_LETTER}:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at_utc = ? WHERE run_id = ?",
                (to_state.value, now, run_id),
            )
        if to_state in {JobState.QUEUED, JobState.SUCCEEDED, JobState.REJECTED, JobState.FAILED, JobState.DEAD_LETTER}:
            conn.execute("DELETE FROM leases WHERE job_id = ?", (job_id,))
        if run_id is not None and to_state in {JobState.RUNNING, JobState.EVALUATING, JobState.PERSISTING, JobState.RENDERING}:
            conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (to_state.value, run_id))

        if to_state == JobState.DEAD_LETTER:
            conn.execute(
                """
                INSERT INTO dead_letter (job_id, reason, retry_count, payload_json, metadata_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    reason = excluded.reason,
                    retry_count = excluded.retry_count,
                    payload_json = excluded.payload_json,
                    metadata_json = excluded.metadata_json,
                    created_at_utc = excluded.created_at_utc
                """,
                (
                    job_id,
                    reason or "dead letter",
                    int(row["retry_count"]),
                    str(row["payload_json"]),
                    _json_dumps(metadata),
                    now,
                ),
            )

        self._append_event(
            conn,
            job_id=job_id,
            run_id=run_id,
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            worker_id=worker_id,
            reason=reason,
            payload=dict(metadata or {}),
            occurred_at_utc=now,
        )

    def _append_event(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        run_id: str | None,
        event_type: str,
        from_state: JobState | None,
        to_state: JobState | None,
        worker_id: str | None,
        reason: str | None,
        payload: Mapping[str, Any] | None,
        occurred_at_utc: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO events (
                job_id, run_id, event_type, from_state, to_state,
                worker_id, reason, payload_json, occurred_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                run_id,
                event_type,
                from_state.value if from_state is not None else None,
                to_state.value if to_state is not None else None,
                worker_id,
                reason,
                _json_dumps(payload),
                occurred_at_utc,
            ),
        )

    def _require_job(self, conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        return row

    def _is_legal_transition(self, from_state: JobState, to_state: JobState) -> bool:
        if from_state == to_state:
            return False
        return to_state in LEGAL_TRANSITIONS[from_state]

    def _job_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        job = dict(row)
        job["payload"] = _json_loads(job.pop("payload_json"))
        job["metadata"] = _json_loads(job.pop("metadata_json"))
        return job

    def _event_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        event = dict(row)
        event["payload"] = _json_loads(event.pop("payload_json"))
        return event


JobLedger = SQLiteJobLedger

__all__ = [
    "ALL_JOB_STATES",
    "ClaimedJob",
    "IllegalStateTransition",
    "JobLedger",
    "JobState",
    "LEGAL_TRANSITIONS",
    "LedgerConfig",
    "LedgerError",
    "SQLiteJobLedger",
]
