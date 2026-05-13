"""Render ledger-backed EQR experiment reports and a static HTML site."""
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from . import templates


VALIDATION_METRIC = "rank_ic"
LOWER_IS_BETTER = {"mse", "mae", "max_drawdown", "runtime", "turnover_proxy"}
PROMOTED_STATES = {"SUCCEEDED"}
DEAD_STATES = {"FAILED", "REJECTED", "DEAD_LETTER"}


@dataclass(frozen=True)
class RenderResult:
    """Summary of generated report and site files."""

    output_dir: Path
    reports_dir: Path
    html_files: tuple[Path, ...]
    report_files: tuple[Path, ...]
    run_count: int


def render_site(
    *,
    ledger_path: Path,
    output_dir: Path,
    run_root: Path | None = None,
    reports_dir: Path | None = None,
) -> RenderResult:
    """Render markdown/json source reports and a self-contained static HTML site."""

    project_root = _project_root_from_ledger(ledger_path)
    effective_run_root = run_root or project_root / "experiments" / "runs"
    effective_reports_dir = reports_dir or project_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_reports_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _load_snapshot(ledger_path, effective_run_root)
    runs = snapshot["runs"]
    jobs = snapshot["jobs"]
    dead_letters = snapshot["dead_letters"]

    report_files = _write_source_reports(effective_reports_dir, snapshot)
    html_files: list[Path] = []
    pages = {
        "index.html": _index_page(runs, jobs),
        "leaderboard.html": _leaderboard_page(runs),
        "dead_letter.html": _dead_letter_page(dead_letters, jobs),
        "coverage.html": _coverage_page(runs),
        "about.html": _about_page(snapshot, report_files),
    }
    for filename, content in pages.items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        html_files.append(path)

    for run in runs:
        path = output_dir / f"run_{_safe_name(str(run['run_id']))}.html"
        path.write_text(_run_page(run), encoding="utf-8")
        html_files.append(path)

    return RenderResult(
        output_dir=output_dir,
        reports_dir=effective_reports_dir,
        html_files=tuple(sorted(html_files)),
        report_files=tuple(sorted(report_files)),
        run_count=len(runs),
    )


def _project_root_from_ledger(ledger_path: Path) -> Path:
    resolved = ledger_path.resolve()
    if resolved.parent.name == "experiments":
        return resolved.parent.parent
    return Path.cwd()


def _load_snapshot(ledger_path: Path, run_root: Path) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    dead_letters: list[dict[str, Any]] = []
    ledger_hash = _short_hash(ledger_path) if ledger_path.exists() else "missing"

    if ledger_path.exists():
        conn = sqlite3.connect(ledger_path)
        conn.row_factory = sqlite3.Row
        try:
            jobs = [_decode_row(row) for row in conn.execute("SELECT * FROM jobs ORDER BY created_at_utc, job_id")]
            if _table_exists(conn, "runs"):
                ledger_runs = [_decode_row(row) for row in conn.execute("SELECT * FROM runs ORDER BY started_at_utc, run_id")]
            else:
                ledger_runs = []
            if _table_exists(conn, "artifacts"):
                artifacts = [_decode_row(row) for row in conn.execute("SELECT * FROM artifacts ORDER BY artifact_id")]
            if _table_exists(conn, "metrics"):
                metrics = [_decode_row(row) for row in conn.execute("SELECT * FROM metrics ORDER BY metric_id")]
            if _table_exists(conn, "dead_letter"):
                dead_letters = [_decode_row(row) for row in conn.execute("SELECT * FROM dead_letter ORDER BY created_at_utc, job_id")]
        finally:
            conn.close()
    else:
        ledger_runs = []

    artifact_runs = _runs_from_artifacts(run_root)
    runs = _merge_runs(ledger_runs, artifact_runs, jobs, metrics, artifacts, run_root)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_path": str(ledger_path),
        "ledger_hash": ledger_hash,
        "run_root": str(run_root),
        "jobs": jobs,
        "runs": runs,
        "artifacts": artifacts,
        "metrics": metrics,
        "dead_letters": dead_letters,
        "reproducibility": _reproducibility(runs, ledger_hash),
    }


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone()
    return row is not None


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    for key in list(record):
        if key.endswith("_json"):
            target = key[:-5]
            record[target] = _json_loads(record.pop(key))
    return record


def _json_loads(value: object) -> Any:
    if value in (None, ""):
        return {}
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {"raw": str(value)}


def _runs_from_artifacts(run_root: Path) -> list[dict[str, Any]]:
    if not run_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics.json")
        run_id = str(manifest.get("run_id") or metrics.get("run_id") or run_dir.name)
        rows.append(
            {
                "run_id": run_id,
                "job_id": manifest.get("job_id", ""),
                "status": manifest.get("status", "ARTIFACT_ONLY"),
                "started_at_utc": manifest.get("created_at", ""),
                "finished_at_utc": manifest.get("created_at", ""),
                "metadata": {},
                "run_dir": str(run_dir),
            }
        )
    return rows


def _merge_runs(
    ledger_runs: list[dict[str, Any]],
    artifact_runs: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    ledger_metrics: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    run_root: Path,
) -> list[dict[str, Any]]:
    by_run_id: dict[str, dict[str, Any]] = {str(run["run_id"]): dict(run) for run in artifact_runs}
    for run in ledger_runs:
        merged = by_run_id.get(str(run["run_id"]), {})
        merged.update(run)
        by_run_id[str(run["run_id"])] = merged

    job_by_id = {str(job["job_id"]): job for job in jobs}
    metrics_by_run = _ledger_metrics_by_run(ledger_metrics)
    artifact_uri_by_run = _artifact_uris_by_run(artifacts)
    enriched: list[dict[str, Any]] = []
    for run_id, run in by_run_id.items():
        run_dir = _discover_run_dir(run_id, run, run_root, artifact_uri_by_run.get(run_id, {}))
        manifest = _read_json(run_dir / "manifest.json") if run_dir else {}
        file_metrics = _read_json(run_dir / "metrics.json") if run_dir else {}
        config = _read_json(run_dir / "config.json") if run_dir else {}
        features = _read_features(run_dir, manifest, file_metrics) if run_dir else []
        predictions = _prediction_summary(run_dir / "predictions.parquet") if run_dir else {"rows": 0, "periods": []}
        metric_values = {**metrics_by_run.get(run_id, {}), **{k: v for k, v in file_metrics.items() if _is_scalar(v)}}
        job = job_by_id.get(str(run.get("job_id", "")), {})
        status = str(job.get("state") or run.get("status") or manifest.get("status") or "UNKNOWN")
        enriched.append(
            {
                **run,
                "run_id": run_id,
                "job": job,
                "status": status,
                "run_dir": str(run_dir) if run_dir else "",
                "manifest": manifest,
                "metrics": metric_values,
                "config": config,
                "features": features,
                "predictions": predictions,
                "promotion_status": _promotion_status(status, metric_values),
                "repro_hash": _run_hash(run_dir, manifest, metric_values) if run_dir else _short_text_hash(run_id),
            }
        )
    return sorted(enriched, key=lambda item: str(item.get("started_at_utc") or item.get("run_id") or ""), reverse=True)


def _ledger_metrics_by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if not run_id:
            continue
        value = row.get("value_real") if row.get("value_real") is not None else row.get("value")
        grouped.setdefault(str(run_id), {})[str(row.get("name"))] = value
    return grouped


def _artifact_uris_by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if not run_id:
            continue
        grouped.setdefault(str(run_id), {})[str(row.get("name"))] = str(row.get("uri") or "")
    return grouped


def _discover_run_dir(run_id: str, run: dict[str, Any], run_root: Path, artifact_uris: dict[str, str]) -> Path | None:
    candidates = [Path(str(run.get("run_dir", ""))), run_root / run_id]
    candidates.extend(Path(uri).parent for uri in artifact_uris.values() if uri)
    for candidate in candidates:
        if str(candidate) and candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _read_features(run_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    manifest_features = manifest.get("feature_columns")
    metric_features = metrics.get("feature_columns")
    features = manifest_features if isinstance(manifest_features, list) else metric_features
    if isinstance(features, list):
        return [str(item) for item in features]
    feature_path = run_dir / "feature_importance.csv"
    if feature_path.exists():
        try:
            frame = pd.read_csv(feature_path, usecols=["feature"])
            return [str(value) for value in frame["feature"].head(100).tolist()]
        except Exception:
            return []
    return []


def _prediction_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rows": 0, "periods": [], "chart_points": []}
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return {"rows": 0, "periods": [], "chart_points": [], "error": str(exc)}
    summary: dict[str, Any] = {"rows": int(len(frame)), "periods": [], "chart_points": []}
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        summary["periods"] = [str(value.date()) for value in sorted(dates.dropna().unique())]
        if "prediction" in frame.columns:
            grouped = frame.assign(_date=dates).dropna(subset=["_date"]).groupby("_date")["prediction"].mean().reset_index()
            summary["chart_points"] = [
                {"date": str(row["_date"].date()), "prediction": _round(row["prediction"])} for _, row in grouped.tail(24).iterrows()
            ]
    if "target_return" in frame.columns:
        summary["target_mean"] = _round(pd.to_numeric(frame["target_return"], errors="coerce").mean())
    if "prediction" in frame.columns:
        summary["prediction_mean"] = _round(pd.to_numeric(frame["prediction"], errors="coerce").mean())
    return summary


def _round(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return round(number, 6)


def _promotion_status(status: str, metrics: dict[str, Any]) -> str:
    if status in DEAD_STATES:
        return "not promoted"
    if status in PROMOTED_STATES and metrics.get(VALIDATION_METRIC) is not None:
        return "promoted"
    return "pending"


def _write_source_reports(reports_dir: Path, snapshot: dict[str, Any]) -> list[Path]:
    json_path = reports_dir / "eqr_experiment_history.json"
    md_path = reports_dir / "eqr_experiment_history.md"
    json_path.write_text(json.dumps(_redacted_snapshot(snapshot), indent=2, default=str), encoding="utf-8")
    md_path.write_text(_markdown_report(snapshot), encoding="utf-8")
    return [json_path, md_path]


def _redacted_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    safe = _sanitize_for_report(snapshot)
    safe["ledger_path"] = Path(str(snapshot.get("ledger_path", ""))).name
    safe["run_root"] = "experiments/runs"
    return safe


def _sanitize_for_report(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"run_dir", "uri", "output_dir", "input_artifact_path", "config", "panel", "feature_dir", "ledger_path", "run_root"}:
                sanitized[key] = _relative_hint(str(item))
            else:
                sanitized[key] = _sanitize_for_report(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_report(item) for item in value]
    return value


def _markdown_report(snapshot: dict[str, Any]) -> str:
    runs = snapshot["runs"]
    lines = ["# EQR Experiment History", "", f"Generated at: {snapshot['generated_at']}", "", "## Runs", ""]
    for run in runs:
        metrics = run.get("metrics", {})
        lines.append(f"- `{run['run_id']}`: {run.get('status', 'UNKNOWN')} / {VALIDATION_METRIC}={metrics.get(VALIDATION_METRIC, 'n/a')}")
    lines.extend(["", "## Reproducibility", ""])
    for key, value in snapshot["reproducibility"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def _index_page(runs: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> str:
    leaderboard = _leaderboard_rows(runs)[:8]
    rows = [
        [templates.link(f"run_{_safe_name(str(run['run_id']))}.html", str(run["run_id"])), templates.text(run.get("status")), _metric_cell(run, VALIDATION_METRIC), templates.text(run.get("promotion_status"))]
        for run in runs[:10]
    ]
    body = templates.hero("Experiment history", "Offline reports for EQR autonomous research runs, with metrics, artifacts, lineage, and promotion decisions.")
    body += templates.stat_cards(
        [
            ("Runs", len(runs)),
            ("Jobs", len(jobs)),
            ("Promoted", sum(1 for run in runs if run.get("promotion_status") == "promoted")),
            ("Dead letters", sum(1 for job in jobs if job.get("state") == "DEAD_LETTER")),
        ]
    )
    body += templates.section("Recent runs", templates.table(["Run", "State", VALIDATION_METRIC, "Promotion"], rows))
    body += templates.section("Leaderboard preview", templates.table(["Rank", "Run", VALIDATION_METRIC, "Status"], leaderboard))
    return templates.page("Index", body)


def _leaderboard_page(runs: list[dict[str, Any]]) -> str:
    body = templates.hero("Leaderboard", f"Runs sorted by validation metric: {VALIDATION_METRIC}.")
    body += templates.section("Ranked runs", templates.table(["Rank", "Run", VALIDATION_METRIC, "Promotion", "State"], _leaderboard_rows(runs, include_promotion=True)))
    body += templates.section("Metric trends", templates.table(["Run", "Started", "Metric", "Value"], _trend_rows(runs)))
    return templates.page("Leaderboard", body)


def _leaderboard_rows(runs: list[dict[str, Any]], *, include_promotion: bool = False) -> list[list[object]]:
    sorted_runs = sorted(runs, key=lambda run: _score(run, VALIDATION_METRIC), reverse=VALIDATION_METRIC not in LOWER_IS_BETTER)
    rows: list[list[object]] = []
    for index, run in enumerate(sorted_runs, start=1):
        base = [index, templates.link(f"run_{_safe_name(str(run['run_id']))}.html", str(run["run_id"])), _metric_cell(run, VALIDATION_METRIC)]
        if include_promotion:
            base.extend([_promotion_badge(str(run.get("promotion_status", "pending"))), templates.text(run.get("status"))])
        else:
            base.append(templates.text(run.get("status")))
        rows.append(base)
    return rows


def _score(run: dict[str, Any], metric_name: str) -> float:
    value = run.get("metrics", {}).get(metric_name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf") if metric_name not in LOWER_IS_BETTER else float("inf")


def _trend_rows(runs: list[dict[str, Any]]) -> list[list[object]]:
    rows: list[list[object]] = []
    for run in sorted(runs, key=lambda item: str(item.get("started_at_utc") or item.get("run_id") or "")):
        for metric_name in ("rank_ic", "pearson_ic", "hit_rate", "mse", "mae"):
            if metric_name in run.get("metrics", {}):
                rows.append([templates.link(f"run_{_safe_name(str(run['run_id']))}.html", str(run["run_id"])), templates.text(run.get("started_at_utc")), metric_name, _metric_cell(run, metric_name)])
    return rows


def _dead_letter_page(dead_letters: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> str:
    job_by_id = {str(job.get("job_id")): job for job in jobs}
    rows = []
    for item in dead_letters:
        job = job_by_id.get(str(item.get("job_id")), {})
        rows.append([templates.text(item.get("job_id")), templates.text(item.get("reason")), templates.text(item.get("retry_count")), templates.text(job.get("updated_at_utc"))])
    body = templates.hero("Dead letters", "Failed or exhausted jobs with retry counts and terminal reasons.")
    body += templates.section("Terminal failures", templates.table(["Job", "Reason", "Retries", "Updated"], rows))
    return templates.page("Dead letter", body)


def _coverage_page(runs: list[dict[str, Any]]) -> str:
    period_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for run in runs:
        for period in run.get("predictions", {}).get("periods", []):
            period_counts[str(period)[:7]] = period_counts.get(str(period)[:7], 0) + 1
        for feature in run.get("features", []):
            family = str(feature).split("_", 1)[0] if "_" in str(feature) else "other"
            family_counts[family] = family_counts.get(family, 0) + 1
    period_rows = [[templates.text(key), value] for key, value in sorted(period_counts.items())]
    family_rows = [[templates.text(key), value] for key, value in sorted(family_counts.items())]
    body = templates.hero("Data coverage", "Coverage summarized by prediction period and feature-family prefix across generated run artifacts.")
    body += '<div class="split">'
    body += templates.section("Periods", templates.table(["Period", "Run observations"], period_rows))
    body += templates.section("Feature families", templates.table(["Family", "Feature uses"], family_rows))
    body += "</div>"
    return templates.page("Coverage", body)


def _about_page(snapshot: dict[str, Any], report_files: list[Path]) -> str:
    report_links = "".join(f"<li>{templates.text(path.name)}</li>" for path in report_files)
    lineage = {
        "ledger": Path(str(snapshot.get("ledger_path", ""))).name,
        "run_root": "experiments/runs",
        "reports": [path.name for path in report_files],
    }
    body = templates.hero("About", "Static, reproducible EQR experiment reports rendered from ledger rows and local artifacts.")
    body += templates.section("Reproducibility hashes", templates.code_block(snapshot["reproducibility"]))
    body += templates.section("Data lineage", templates.code_block(lineage))
    body += templates.section("Source reports", f"<ul>{report_links}</ul>")
    return templates.page("About", body)


def _run_page(run: dict[str, Any]) -> str:
    metrics = run.get("metrics", {})
    metric_rows = [[templates.text(key), templates.text(value)] for key, value in sorted(metrics.items()) if _is_scalar(value)]
    feature_rows = [[templates.text(feature)] for feature in run.get("features", [])[:100]]
    prediction_summary = run.get("predictions", {})
    chart = _svg_chart(prediction_summary.get("chart_points", []))
    body = templates.hero(f"Run {run['run_id']}", f"State {run.get('status', 'UNKNOWN')} with promotion status {run.get('promotion_status', 'pending')}.")
    body += templates.stat_cards(
        [
            (VALIDATION_METRIC, metrics.get(VALIDATION_METRIC, "n/a")),
            ("Predictions", prediction_summary.get("rows", 0)),
            ("Features", len(run.get("features", []))),
            ("Hash", run.get("repro_hash", "n/a")),
        ]
    )
    body += templates.section("Metrics", templates.table(["Metric", "Value"], metric_rows))
    body += '<div class="split">'
    body += templates.section("Config", templates.code_block(_safe_config(run.get("config", {}))))
    body += templates.section("Config diff", templates.code_block(_config_diff(run.get("config", {}), run.get("manifest", {}))))
    body += "</div>"
    body += templates.section("Feature list", templates.table(["Feature"], feature_rows))
    body += templates.section("Predictions chart", chart)
    body += templates.section("Manifest", templates.code_block(_safe_manifest(run.get("manifest", {}))))
    return templates.page(f"Run {run['run_id']}", body)


def _metric_cell(run: dict[str, Any], metric_name: str) -> str:
    value = run.get("metrics", {}).get(metric_name)
    if isinstance(value, float):
        return templates.text(f"{value:.6g}")
    return templates.text(value if value is not None else "n/a")


def _promotion_badge(status: str) -> str:
    tone = "success" if status == "promoted" else "danger" if status == "not promoted" else "warning"
    return templates.badge(status, tone)


def _svg_chart(points: list[dict[str, Any]]) -> str:
    if not points:
        return '<p class="muted">No prediction series available.</p>'
    values = [float(point["prediction"]) for point in points if point.get("prediction") is not None]
    if not values:
        return '<p class="muted">Prediction series has no numeric values.</p>'
    width = 720
    height = 180
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    step = width / max(len(values) - 1, 1)
    coords = [f"{index * step:.2f},{height - ((value - low) / span * (height - 24) + 12):.2f}" for index, value in enumerate(values)]
    labels = f"min {low:.4g} · max {high:.4g}"
    return f'<svg class="sparkline" viewBox="0 0 {width} {height}" role="img" aria-label="Mean prediction trend"><polyline points="{" ".join(coords)}" fill="none" stroke="var(--color-primary)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><text x="16" y="28" fill="var(--color-muted)" font-size="14">{templates.text(labels)}</text></svg>'


def _config_diff(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": config.get("config_path") or manifest.get("config"),
        "target_column": manifest.get("target_column"),
        "row_count": manifest.get("row_count"),
        "period_count": manifest.get("period_count"),
        "model_params": config.get("model_params", {}),
    }


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    safe = dict(config)
    if "config_path" in safe:
        safe["config_path"] = Path(str(safe["config_path"])).name
    return safe


def _safe_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    safe = dict(manifest)
    for key in ("input_artifact_path", "output_dir", "config", "panel", "feature_dir"):
        if key in safe:
            safe[key] = _relative_hint(str(safe[key]))
    return safe


def _relative_hint(value: str) -> str:
    for marker in ("experiments/", "configs/", "data/"):
        if marker in value:
            return value[value.index(marker) :]
    return Path(value).name


def _reproducibility(runs: list[dict[str, Any]], ledger_hash: str) -> dict[str, str]:
    joined = "|".join(sorted(str(run.get("repro_hash", "")) for run in runs))
    return {
        "ledger_hash16": ledger_hash,
        "run_set_hash16": _short_text_hash(joined),
        "renderer": "autoquant_lab.eqr.reporting.renderer",
    }


def _run_hash(run_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8"))
    digest.update(json.dumps(metrics, sort_keys=True, default=str).encode("utf-8"))
    for filename in ("config.json", "feature_importance.csv"):
        path = run_dir / filename
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _short_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "missing"
    return digest.hexdigest()[:16]


def _short_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)
