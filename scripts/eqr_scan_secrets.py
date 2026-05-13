#!/usr/bin/env python3
"""Scan working-tree or staged text files for likely hardcoded secrets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
from typing import cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDED_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "prototypes", "reports", "tests"}
DEFAULT_EXCLUDED_FILES = {".env"}
SCANNED_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "assignment_secret",
        re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd)\b\s*=\s*['\"]([^'\"]{8,})['\"]"),
    ),
    ("hex_32_plus", re.compile(r"\b[a-fA-F0-9]{32,}\b")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan text files for likely hardcoded secrets.")
    _ = parser.add_argument("root", nargs="?", type=Path, default=PROJECT_ROOT, help="Workspace root to scan.")
    _ = parser.add_argument("--staged", action="store_true", help="Scan staged git content instead of the working tree.")
    return parser.parse_args()


def should_scan_file(path: Path) -> bool:
    return path.suffix in SCANNED_SUFFIXES and path.name not in DEFAULT_EXCLUDED_FILES


def iter_candidate_files(root: Path):
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in DEFAULT_EXCLUDED_DIRS]
        for filename in filenames:
            path = Path(current_root) / filename
            if should_scan_file(path):
                yield path


def iter_staged_files(root: Path):
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    for rel_path in result.stdout.splitlines():
        path = root / rel_path
        if should_scan_file(path):
            yield rel_path, path


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        print(f"Warning: could not read {path}: {exc}")
        return findings

    for line_number, line in enumerate(lines, start=1):
        for pattern_name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((line_number, pattern_name, line.strip()))
    return findings


def scan_text(lines: list[str]) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        for pattern_name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((line_number, pattern_name, line.strip()))
    return findings


def scan_staged_file(root: Path, rel_path: str) -> list[tuple[int, str, str]]:
    result = subprocess.run(
        ["git", "show", f":{rel_path}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        errors="ignore",
    )
    return scan_text(result.stdout.splitlines())


def redact(line: str) -> str:
    redacted = re.sub(r"(['\"])([^'\"]{4})[^'\"]{4,}([^'\"]{4})(['\"])", r"\1\2...\3\4", line)
    return re.sub(r"\b([a-fA-F0-9]{8})[a-fA-F0-9]{16,}([a-fA-F0-9]{8})\b", r"\1...\2", redacted)


def main() -> None:
    args = parse_args()
    root = cast(Path, args.root).resolve()
    staged = cast(bool, args.staged)
    all_findings: list[tuple[str, int, str, str]] = []

    if staged:
        for rel_path, _path in iter_staged_files(root):
            for line_number, pattern_name, line in scan_staged_file(root, rel_path):
                all_findings.append((rel_path, line_number, pattern_name, line))
    else:
        for path in iter_candidate_files(root):
            for line_number, pattern_name, line in scan_file(path):
                all_findings.append((str(path.relative_to(root)), line_number, pattern_name, line))

    if not all_findings:
        scope = "staged files" if staged else f"text files under {root}"
        print(f"No potential secrets found in {scope}")
        return

    print(f"Potential secrets found: {len(all_findings)}")
    for rel_path, line_number, pattern_name, line in all_findings:
        print(f"{rel_path}:{line_number}: {pattern_name}: {redact(line)}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
