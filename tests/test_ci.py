"""Tests for the EQR CI contract wrapper."""

from __future__ import annotations

from pathlib import Path

from scripts import eqr_ci


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_STAGE_NAMES: tuple[str, ...] = (
    "pytest",
    "raw_data",
    "config",
    "panel",
    "ledger",
    "skill",
    "site",
    "secrets",
    "offline",
)


def test_ci_stage_enumeration_matches_contract() -> None:
    assert tuple(stage.name for stage in eqr_ci.STAGES) == EXPECTED_STAGE_NAMES


def test_ci_stages_are_executable_or_checked() -> None:
    for stage in eqr_ci.STAGES:
        assert stage.command is not None or stage.check is not None, stage.name


def test_ci_script_writes_default_report_under_reports() -> None:
    assert eqr_ci.DEFAULT_REPORT == REPO_ROOT / "reports" / "eqr_ci_report.json"


def test_smoke_mode_uses_fast_raw_data_offline_check() -> None:
    raw_data = next(stage for stage in eqr_ci.STAGES if stage.name == "raw_data")
    command = raw_data.command_for(smoke=True)
    assert command is not None
    assert "--check-offline-only" in command
    assert "--data-dir" not in command


def test_required_validators_are_wired() -> None:
    commands = {
        stage.name: stage.command_for(smoke=False)
        for stage in eqr_ci.STAGES
        if stage.command_for(smoke=False) is not None
    }
    assert commands["pytest"] is not None and "tests/" in commands["pytest"]
    assert commands["raw_data"] is not None and "scripts/eqr_validate_raw_data.py" in commands["raw_data"]
    assert commands["config"] is not None and "scripts/eqr_validate_config.py" in commands["config"]
    assert commands["ledger"] is not None and "tests/test_ledger_fsm.py" in commands["ledger"]
    assert commands["skill"] is not None and "scripts/eqr_validate_skill.py" in commands["skill"]
    assert commands["site"] is not None and "scripts/eqr_validate_site.py" in commands["site"]
    assert commands["secrets"] is not None and "scripts/eqr_scan_secrets.py" in commands["secrets"]
