#!/usr/bin/env python3
"""Run the full local EQR CI contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "reports" / "eqr_ci_report.json"
PANEL_PATH = REPO_ROOT / "experiments" / "prepared" / "panel" / "monthly_labels.parquet"

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eqr_validate_raw_data import scan_offline_guard  # noqa: E402


@dataclass(frozen=True)
class Stage:
    """One CI stage in execution order."""

    name: str
    description: str
    command: tuple[str, ...] | None = None
    check: Callable[[bool], tuple[int, str]] | None = None
    smoke_command: tuple[str, ...] | None = None

    def command_for(self, smoke: bool) -> tuple[str, ...] | None:
        if smoke and self.smoke_command is not None:
            return self.smoke_command
        return self.command


def _python(*parts: str) -> tuple[str, ...]:
    return (sys.executable, *parts)


def check_panel_exists(_smoke: bool) -> tuple[int, str]:
    if PANEL_PATH.exists():
        return 0, f"Found prepared panel: {PANEL_PATH.relative_to(REPO_ROOT)}"
    return 1, f"Missing prepared panel: {PANEL_PATH.relative_to(REPO_ROOT)}"


def check_offline_only(_smoke: bool) -> tuple[int, str]:
    findings = cast(list[dict[str, object]], scan_offline_guard(REPO_ROOT))
    if not findings:
        return 0, "Offline guard found no active credential prompt or network download code."
    lines = ["Offline guard findings:"]
    for finding in findings:
        path = Path(str(finding["path"]))
        try:
            display_path = path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = path
        line = str(finding.get("line", "?"))
        pattern = str(finding.get("pattern", "unknown"))
        text = str(finding.get("text", ""))
        lines.append(f"{display_path}:{line} {pattern}: {text}")
    return 1, "\n".join(lines)


STAGES: tuple[Stage, ...] = (
    Stage(
        name="pytest",
        description="Run the project pytest suite.",
        command=_python("-m", "pytest", "tests/", "-q"),
        smoke_command=_python("-m", "pytest", "tests/test_ci.py", "tests/test_config.py", "tests/test_ledger_fsm.py", "-q"),
    ),
    Stage(
        name="raw_data",
        description="Validate local offline raw data contracts.",
        command=_python("scripts/eqr_validate_raw_data.py", "--data-dir", "data", "--project-root", "."),
        smoke_command=_python("scripts/eqr_validate_raw_data.py", "--check-offline-only", "--project-root", "."),
    ),
    Stage(
        name="config",
        description="Validate the golden path experiment config.",
        command=_python("scripts/eqr_validate_config.py", "configs/golden_path.yaml"),
    ),
    Stage(
        name="panel",
        description="Verify the prepared monthly labels panel exists.",
        check=check_panel_exists,
    ),
    Stage(
        name="ledger",
        description="Run ledger FSM tests.",
        command=_python("-m", "pytest", "tests/test_ledger_fsm.py", "-q"),
    ),
    Stage(
        name="skill",
        description="Validate the EQR autoresearch skill runbook.",
        command=_python("scripts/eqr_validate_skill.py", "skills/eqr-autoresearch/SKILL.md"),
    ),
    Stage(
        name="site",
        description="Validate the generated static site.",
        command=_python("scripts/eqr_validate_site.py", "site"),
    ),
    Stage(
        name="secrets",
        description="Scan the working tree for likely hardcoded secrets.",
        command=_python("scripts/eqr_scan_secrets.py"),
    ),
    Stage(
        name="offline",
        description="Ensure the active EQR pipeline is offline-only.",
        check=check_offline_only,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EQR CI contract.")
    _ = parser.add_argument("--smoke", action="store_true", help="Use fast checks where a full stage has an approved smoke substitute.")
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Path to write the CI JSON report.")
    return parser.parse_args()


def run_command(command: tuple[str, ...]) -> tuple[int, str, str]:
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.returncode, completed.stdout, completed.stderr


def run_stage(stage: Stage, smoke: bool) -> dict[str, object]:
    print(f"==> {stage.name}: {stage.description}")
    started = time.monotonic()
    command = stage.command_for(smoke)
    stdout = ""
    stderr = ""

    if command is not None:
        returncode, stdout, stderr = run_command(command)
        detail = " ".join(command)
    elif stage.check is not None:
        returncode, stdout = stage.check(smoke)
        detail = "internal check"
    else:
        returncode = 1
        stderr = "Stage has neither command nor check."
        detail = "invalid stage"

    duration_seconds = round(time.monotonic() - started, 3)
    status = "pass" if returncode == 0 else "fail"
    print(f"<== {stage.name}: {status.upper()} ({duration_seconds:.3f}s)")
    if stdout:
        print(stdout.rstrip())
    if stderr:
        print(stderr.rstrip(), file=sys.stderr)

    return {
        "name": stage.name,
        "description": stage.description,
        "status": status,
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "command": list(command) if command is not None else None,
        "detail": detail,
        "stdout": stdout,
        "stderr": stderr,
    }


def write_report(report_path: Path, report: Mapping[str, object]) -> None:
    resolved_report_path = report_path if report_path.is_absolute() else REPO_ROOT / report_path
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = resolved_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    smoke = bool(cast(bool, args.smoke))
    results = [run_stage(stage, smoke) for stage in STAGES]
    ok = all(result["returncode"] == 0 for result in results)
    report = {
        "ok": ok,
        "smoke": smoke,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage_count": len(STAGES),
        "stages": results,
    }
    report_path = cast(Path, args.report)
    write_report(report_path, report)
    display_report_path = report_path if report_path.is_absolute() else REPO_ROOT / report_path
    print(f"CI report written to {display_report_path}")
    print("EQR CI PASSED" if ok else "EQR CI FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
