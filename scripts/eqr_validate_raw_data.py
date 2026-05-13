#!/usr/bin/env python3
"""Validate offline EQR raw data artifacts under a local data directory."""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
from pandas.util import hash_pandas_object


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from autoquant_lab.eqr.data_contracts import ARTIFACT_CONTRACTS, CONTRACTS_BY_NAME, ArtifactContract  # noqa: E402
from autoquant_lab.eqr.path_resolver import ResolvedArtifact, resolve_data_paths  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "eqr_raw_data_validation.json"
SCAN_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}
SCAN_DIRS = ("src/autoquant_lab/eqr", "configs")

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("wrds_connection", re.compile(r"\bwrds\s*\.\s*" + r"Connection\s*\(", re.IGNORECASE)),
    ("wrds_credentials", re.compile(r"\bwrds\b.*\b(?:user|username|pass|password|credential|login|prompt)\b", re.IGNORECASE)),
    (
        "credential_prompt",
        re.compile(
            r"\b(?:get" + r"pass\s*\(|in" + r"put\s*\().{0,80}\b(?:w" + r"rds|user|username|pass|password|credential|login)\b",
            re.IGNORECASE,
        ),
    ),
    ("requests_download", re.compile(r"\brequests\s*\.\s*(?:get|post|put|request)\s*\(", re.IGNORECASE)),
    ("urllib_download", re.compile(r"\burllib\.request\s*\.\s*(?:urlopen|urlretrieve)\s*\(", re.IGNORECASE)),
    ("subprocess_network_download", re.compile(r"\b(?:" + "cu" + r"rl|" + "wg" + r"et)\b", re.IGNORECASE)),
    ("fred_api_download", re.compile(r"\b(?:" + "fred" + r"api|pandas_data" + r"reader)\b", re.IGNORECASE)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local offline EQR raw Parquet artifacts.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Local data directory to validate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root for offline guard scanning.")
    parser.add_argument("--check-offline-only", action="store_true", help="Only run the offline active-pipeline guard.")
    return parser.parse_args()


def _candidate_scan_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_dir in SCAN_DIRS:
        root = project_root / relative_dir
        if not root.exists():
            continue
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in {"__pycache__", ".pytest_cache", ".mypy_cache"}]
            for filename in filenames:
                path = Path(current_root) / filename
                if path.suffix in SCAN_SUFFIXES:
                    files.append(path)

    scripts_dir = project_root / "scripts"
    if scripts_dir.exists():
        files.extend(sorted(scripts_dir.glob("eqr_*.py")))
    return sorted(set(files))


def _is_offline_guard_enforcement_line(path: Path, stripped: str) -> bool:
    """Return True for literals that define or document guard rules, not active behavior."""

    if path.name in {"eqr_validate_raw_data.py", "eqr_validate_skill.py"}:
        return True
    return "PATTERN" in stripped or "ConfigValidationError" in stripped


def scan_offline_guard(project_root: str | Path) -> list[dict[str, Any]]:
    """Return active EQR files containing disallowed WRDS or network-download code."""

    root = Path(project_root).expanduser().resolve()
    findings: list[dict[str, Any]] = []
    for path in _candidate_scan_files(root):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            findings.append({"path": str(path), "line": 0, "pattern": "read_error", "text": str(exc)})
            continue

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _is_offline_guard_enforcement_line(path, stripped):
                continue
            for pattern_name, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    findings.append({"path": str(path), "line": line_number, "pattern": pattern_name, "text": stripped})
    return findings


def _validate_required_columns(contract: ArtifactContract, artifact: ResolvedArtifact) -> dict[str, Any]:
    present = set(artifact.columns)
    missing = [column for column in contract.required_columns if column not in present]
    return {
        "required_columns": list(contract.required_columns),
        "missing_columns": missing,
        "ok": not missing,
    }


def _validate_dates(contract: ArtifactContract, artifact: ResolvedArtifact) -> dict[str, Any]:
    results: dict[str, Any] = {}
    available = set(artifact.columns)
    for spec in contract.date_columns:
        if spec.name not in available:
            results[spec.name] = {"ok": False, "parse_failures": None, "min": None, "max": None, "error": "missing date column"}
            continue

        parse_failures = 0
        non_null_count = 0
        minimum: pd.Timestamp | None = None
        maximum: pd.Timestamp | None = None
        for file_path in artifact.files:
            series = pd.read_parquet(file_path, columns=[spec.name])[spec.name]
            parsed = pd.to_datetime(series, errors="coerce")
            non_null = series.notna()
            non_null_count += int(non_null.sum())
            parse_failures += int((parsed.isna() & non_null).sum())
            if parsed.notna().any():
                file_min = parsed.min()
                file_max = parsed.max()
                minimum = file_min if minimum is None or file_min < minimum else minimum
                maximum = file_max if maximum is None or file_max > maximum else maximum

        results[spec.name] = {
            "ok": parse_failures == 0,
            "nullable": spec.nullable,
            "non_null_count": non_null_count,
            "parse_failures": parse_failures,
            "min": minimum.date().isoformat() if minimum is not None else None,
            "max": maximum.date().isoformat() if maximum is not None else None,
        }
    return results


def _duplicate_key_summary(contract: ArtifactContract, artifact: ResolvedArtifact) -> dict[str, Any]:
    missing = [column for column in contract.key_columns if column not in artifact.columns]
    if missing:
        return {"checked": False, "key_columns": list(contract.key_columns), "duplicate_count": None, "missing_key_columns": missing}

    seen: set[int] = set()
    duplicate_count = 0
    rows_checked = 0
    for file_path in artifact.files:
        frame = pd.read_parquet(file_path, columns=list(contract.key_columns))
        rows_checked += len(frame)
        hashes = hash_pandas_object(frame, index=False).to_numpy("uint64")  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        local_seen: set[int] = set()
        for value in hashes:
            key_hash = int(value)
            if key_hash in local_seen or key_hash in seen:
                duplicate_count += 1
            local_seen.add(key_hash)
        seen.update(local_seen)

    return {
        "checked": True,
        "key_columns": list(contract.key_columns),
        "rows_checked": rows_checked,
        "duplicate_count": duplicate_count,
        "ok": True,
        "severity": "warning" if duplicate_count else "ok",
    }


def validate_artifact(contract: ArtifactContract, artifact: ResolvedArtifact) -> dict[str, Any]:
    required = _validate_required_columns(contract, artifact)
    dates = _validate_dates(contract, artifact)
    duplicates = _duplicate_key_summary(contract, artifact) if required["ok"] else {
        "checked": False,
        "key_columns": list(contract.key_columns),
        "duplicate_count": None,
        "reason": "required columns missing",
    }
    date_ok = all(result.get("ok", False) for result in dates.values())
    ok = bool(required["ok"] and date_ok)
    return {
        "artifact_name": artifact.artifact_name,
        "artifact_type": artifact.artifact_type,
        "path": str(artifact.path),
        "canonical_location": artifact.canonical_location,
        "current_location": artifact.current_location,
        "row_count": artifact.row_count,
        "columns": list(artifact.columns),
        "column_count": len(artifact.columns),
        "required_column_status": required,
        "date_status": dates,
        "duplicate_key_status": duplicates,
        "ok": ok,
    }


def validate_raw_data(data_dir: str | Path, project_root: str | Path) -> dict[str, Any]:
    resolved = resolve_data_paths(data_dir, include_date_ranges=True)
    artifacts: dict[str, Any] = {}
    missing_artifacts: list[str] = []

    for contract in ARTIFACT_CONTRACTS:
        artifact = resolved.get(contract.artifact_name)
        if artifact is None:
            missing_artifacts.append(contract.artifact_name)
            continue
        artifacts[contract.artifact_name] = validate_artifact(contract, artifact)

    offline_findings = scan_offline_guard(project_root)
    failed_artifacts = [name for name, result in artifacts.items() if not result["ok"]]
    ok = not missing_artifacts and not failed_artifacts and not offline_findings
    return {
        "ok": ok,
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "artifact_count": len(artifacts),
        "expected_artifact_count": len(CONTRACTS_BY_NAME),
        "missing_artifacts": missing_artifacts,
        "failed_artifacts": failed_artifacts,
        "offline_guard": {"ok": not offline_findings, "findings": offline_findings},
        "artifacts": artifacts,
    }


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    status = "OK" if report["ok"] else "FAILED"
    print(f"EQR raw data validation: {status}")
    print(f"Data directory: {report['data_dir']}")
    print(f"Artifacts: {report['artifact_count']}/{report['expected_artifact_count']}")
    for name, artifact in report["artifacts"].items():
        required = artifact["required_column_status"]
        date_parts = []
        for column, date_status in artifact["date_status"].items():
            date_parts.append(f"{column}={date_status['min']}..{date_status['max']} parse_failures={date_status['parse_failures']}")
        duplicate_status = artifact["duplicate_key_status"]
        duplicate_count = duplicate_status.get("duplicate_count")
        print(
            f"- {name}: rows={artifact['row_count']} columns={artifact['column_count']} "
            f"required_ok={required['ok']} missing={required['missing_columns']} "
            f"duplicates={duplicate_count} dates=[{'; '.join(date_parts)}]"
        )
    if report["missing_artifacts"]:
        print(f"Missing artifacts: {', '.join(report['missing_artifacts'])}")
    if report["offline_guard"]["findings"]:
        print("Offline guard findings:")
        for finding in report["offline_guard"]["findings"]:
            print(f"- {finding['path']}:{finding['line']} {finding['pattern']}: {finding['text']}")


def main() -> None:
    args = parse_args()
    if args.check_offline_only:
        findings = scan_offline_guard(args.project_root)
        if findings:
            print("Offline guard findings:")
            for finding in findings:
                print(f"- {finding['path']}:{finding['line']} {finding['pattern']}: {finding['text']}")
            raise SystemExit(1)
        print("Offline-only guard passed")
        raise SystemExit(0)

    report = validate_raw_data(args.data_dir, args.project_root)
    _write_report(report, args.output)
    _print_summary(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
