"""Tests for the EQR golden-path autoresearch command."""
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false, reportPrivateUsage=false

from __future__ import annotations

import json
from pathlib import Path

from scripts import eqr_autoresearch


def test_golden_path_command_structure() -> None:
    parser = eqr_autoresearch.build_parser()
    args = parser.parse_args(["golden-path", "--config", "configs/golden_path.yaml", "--max-trials", "3"])

    assert args.command == "golden-path"
    assert args.config == "configs/golden_path.yaml"
    assert args.max_trials == 3
    assert args.panel.endswith("experiments/prepared/panel/monthly_labels.parquet")
    assert args.feature_dir.endswith("experiments/prepared/features")


def test_golden_path_command_defaults() -> None:
    parser = eqr_autoresearch.build_parser()
    args = parser.parse_args(["golden-path"])

    assert args.command == "golden-path"
    assert args.max_trials == 3
    assert args.max_rows == 50000
    assert args.worker_id == "golden-path"


def test_golden_path_stage_sequence() -> None:
    stages = eqr_autoresearch.golden_path_stage_sequence(Path("configs/golden_path.yaml"))

    assert tuple(stage.name for stage in stages) == (
        "validate_raw_data",
        "build_links",
        "prepare_labels",
        "prepare_features",
        "validate_config",
    )
    commands = {stage.name: stage.command for stage in stages}
    assert "scripts/eqr_validate_raw_data.py" in commands["validate_raw_data"]
    assert "scripts/eqr_build_links.py" in commands["build_links"]
    assert commands["prepare_labels"][-4:] == ("--stage", "labels", "--config", "configs/golden_path.yaml")
    assert commands["prepare_features"][-4:] == ("--stage", "features", "--config", "configs/golden_path.yaml")
    assert commands["validate_config"][-2:] == ("scripts/eqr_validate_config.py", "configs/golden_path.yaml")


def test_golden_path_stage_sequence_with_max_rows() -> None:
    stages = eqr_autoresearch.golden_path_stage_sequence(Path("configs/golden_path.yaml"), max_rows=50000)

    commands = {stage.name: stage.command for stage in stages}
    assert "--max-rows" in commands["prepare_labels"]
    assert "50000" in commands["prepare_labels"]
    assert "--max-rows" in commands["prepare_features"]
    assert "50000" in commands["prepare_features"]
    assert "--max-rows" not in commands["validate_raw_data"]
    assert "--max-rows" not in commands["build_links"]


def test_golden_path_stage_sequence_no_max_rows() -> None:
    stages = eqr_autoresearch.golden_path_stage_sequence(Path("configs/golden_path.yaml"), max_rows=0)

    commands = {stage.name: stage.command for stage in stages}
    for name, cmd in commands.items():
        assert "--max-rows" not in cmd, f"stage {name} should not have --max-rows when max_rows=0"


def test_golden_path_runs_end_to_end_stage_order(monkeypatch, tmp_path: Path) -> None:
    observed_stages: list[str] = []

    def fake_stage(stage: eqr_autoresearch.GoldenPathStage) -> dict[str, object]:
        observed_stages.append(stage.name)
        return {"name": stage.name, "returncode": 0}

    def fake_trials(args: object) -> list[dict[str, object]]:
        return [
            {
                "job_id": f"job-{index}",
                "run_id": f"run-{index}",
                "succeeded": True,
                "run_dir": str(tmp_path / f"run-{index}"),
                "metrics": {"rank_ic": 0.02, "decile_long_short_return": 0.01, "feature_coverage": 0.9},
                "promotion_status": "promoted",
                "error": None,
            }
            for index in range(3)
        ]

    monkeypatch.setattr(eqr_autoresearch, "_run_subprocess_stage", fake_stage)
    monkeypatch.setattr(eqr_autoresearch, "_execute_golden_trials", fake_trials)

    parser = eqr_autoresearch.build_parser()
    args = parser.parse_args(
        [
            "golden-path",
            "--config",
            "configs/golden_path.yaml",
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
            "--max-trials",
            "3",
        ]
    )

    assert eqr_autoresearch.cmd_golden_path(args) == 0
    assert observed_stages == [
        "validate_raw_data",
        "build_links",
        "prepare_labels",
        "prepare_features",
        "validate_config",
        "render_site",
        "ci_smoke",
    ]


def test_golden_path_summary_output(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_stage(stage: eqr_autoresearch.GoldenPathStage) -> dict[str, object]:
        return {"name": stage.name, "returncode": 0}

    def fake_trials(args: object) -> list[dict[str, object]]:
        return [
            {
                "job_id": f"job-{i}",
                "run_id": f"run-{i}",
                "succeeded": True,
                "run_dir": str(tmp_path / f"run-{i}"),
                "metrics": {"rank_ic": 0.02, "decile_long_short_return": 0.01, "feature_coverage": 0.9},
                "promotion_status": "promoted",
                "error": None,
            }
            for i in range(3)
        ]

    monkeypatch.setattr(eqr_autoresearch, "_run_subprocess_stage", fake_stage)
    monkeypatch.setattr(eqr_autoresearch, "_execute_golden_trials", fake_trials)

    parser = eqr_autoresearch.build_parser()
    args = parser.parse_args(
        [
            "golden-path",
            "--config",
            "configs/golden_path.yaml",
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
            "--max-trials",
            "3",
        ]
    )

    result = eqr_autoresearch.cmd_golden_path(args)
    assert result == 0

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "succeeded"
    assert summary["run_count"] == 3
    assert "runs" in summary
    assert summary["ci"] == "smoke passed"
    assert len(summary["runs"]) == 3
    for run in summary["runs"]:
        assert run["promotion_status"] == "promoted"
        assert run["succeeded"] is True


def test_golden_path_fails_on_stage_error(monkeypatch, tmp_path: Path) -> None:
    def failing_stage(stage: eqr_autoresearch.GoldenPathStage) -> dict[str, object]:
        if stage.name == "build_links":
            raise RuntimeError("golden-path stage failed: build_links (exit 1)")
        return {"name": stage.name, "returncode": 0}

    monkeypatch.setattr(eqr_autoresearch, "_run_subprocess_stage", failing_stage)
    monkeypatch.setattr(eqr_autoresearch, "_execute_golden_trials", lambda args: [])

    parser = eqr_autoresearch.build_parser()
    args = parser.parse_args(
        [
            "golden-path",
            "--config",
            "configs/golden_path.yaml",
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
            "--max-trials",
            "3",
        ]
    )

    result = eqr_autoresearch.cmd_golden_path(args)
    assert result == 1


def test_golden_path_fails_on_trial_mismatch(monkeypatch, tmp_path: Path) -> None:
    def fake_stage(stage: eqr_autoresearch.GoldenPathStage) -> dict[str, object]:
        return {"name": stage.name, "returncode": 0}

    def short_trials(args: object) -> list[dict[str, object]]:
        return [
            {
                "job_id": "job-0",
                "run_id": "run-0",
                "succeeded": True,
                "run_dir": str(tmp_path / "run-0"),
                "metrics": {"rank_ic": 0.02},
                "promotion_status": "promoted",
                "error": None,
            }
        ]

    monkeypatch.setattr(eqr_autoresearch, "_run_subprocess_stage", fake_stage)
    monkeypatch.setattr(eqr_autoresearch, "_execute_golden_trials", short_trials)

    parser = eqr_autoresearch.build_parser()
    args = parser.parse_args(
        [
            "golden-path",
            "--config",
            "configs/golden_path.yaml",
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
            "--max-trials",
            "3",
        ]
    )

    result = eqr_autoresearch.cmd_golden_path(args)
    assert result == 1


def test_promotion_status_promoted() -> None:
    from autoquant_lab.eqr.config import load_experiment_config

    config = load_experiment_config(Path("configs/golden_path.yaml"))
    metrics = {"rank_ic": 0.05, "decile_long_short_return": 0.02, "feature_coverage": 0.95}
    assert eqr_autoresearch._promotion_status(config, metrics, succeeded=True) == "promoted"


def test_promotion_status_not_promoted() -> None:
    from autoquant_lab.eqr.config import load_experiment_config

    config = load_experiment_config(Path("configs/golden_path.yaml"))
    metrics = {"rank_ic": 0.001, "decile_long_short_return": -0.01, "feature_coverage": 0.5}
    assert eqr_autoresearch._promotion_status(config, metrics, succeeded=True) == "not promoted"


def test_promotion_status_failed_run() -> None:
    from autoquant_lab.eqr.config import load_experiment_config

    config = load_experiment_config(Path("configs/golden_path.yaml"))
    metrics = {"rank_ic": 0.05, "decile_long_short_return": 0.02, "feature_coverage": 0.95}
    assert eqr_autoresearch._promotion_status(config, metrics, succeeded=False) == "not promoted"


def test_promotion_status_pending_when_missing_metric() -> None:
    from autoquant_lab.eqr.config import load_experiment_config

    config = load_experiment_config(Path("configs/golden_path.yaml"))
    metrics = {"rank_ic": 0.05}
    assert eqr_autoresearch._promotion_status(config, metrics, succeeded=True) == "pending"


def test_selected_metrics_filters_keys() -> None:
    metrics = {
        "model": "ridge",
        "rank_ic": 0.05,
        "decile_long_short_return": 0.02,
        "feature_coverage": 0.95,
        "mse": 0.04,
        "mae": 0.15,
        "hit_rate": 0.52,
        "extra_key": "should_be_filtered",
    }
    selected = eqr_autoresearch._selected_metrics(metrics)
    assert "model" in selected
    assert "rank_ic" in selected
    assert "decile_long_short_return" in selected
    assert "feature_coverage" in selected
    assert "mse" in selected
    assert "mae" in selected
    assert "hit_rate" in selected
    assert "extra_key" not in selected


def test_selected_metrics_omits_absent_keys() -> None:
    metrics = {"rank_ic": 0.05}
    selected = eqr_autoresearch._selected_metrics(metrics)
    assert "rank_ic" in selected
    assert "model" not in selected
    assert "mse" not in selected
