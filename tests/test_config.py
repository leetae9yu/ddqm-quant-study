from __future__ import annotations
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import pytest

from autoquant_lab.eqr.config import ConfigValidationError, load_experiment_config, parse_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_config() -> dict[str, Any]:
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
            "max_trials": 24,
            "max_runtime_minutes": 120,
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


def test_valid_golden_path_config_passes_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/eqr_validate_config.py", "configs/golden_path.yaml"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("config_hash=")
    assert load_experiment_config(REPO_ROOT / "configs" / "golden_path.yaml").model.name == "ridge"


def test_negative_cli_rejects_temp_config() -> None:
    golden = (REPO_ROOT / "configs" / "golden_path.yaml").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_config = Path(tmpdir) / "bad_config.yaml"
        bad_config.write_text(golden.replace('data_dir: "data"', 'data_dir: "../data"'), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/eqr_validate_config.py", str(bad_config)],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

    assert result.returncode != 0
    assert "Config validation failed" in result.stderr


def test_shell_command_rejection() -> None:
    config = _valid_config()
    config["artifacts"]["output_dir"] = "experiments/golden_path && rm -rf data"

    with pytest.raises(ConfigValidationError, match="Shell command"):
        parse_experiment_config(config)


def test_path_traversal_rejection() -> None:
    config = _valid_config()
    config["data"]["data_dir"] = "../data"

    with pytest.raises(ConfigValidationError, match="Path traversal"):
        parse_experiment_config(config)


def test_invalid_model_name_rejection() -> None:
    config = _valid_config()
    config["model"]["name"] = "unregistered_model"

    with pytest.raises(ConfigValidationError, match="Unknown model"):
        parse_experiment_config(config)


def test_invalid_feature_family_rejection() -> None:
    config = _valid_config()
    config["features"]["families"] = deepcopy(config["features"]["families"])
    config["features"]["families"]["future_prices"] = True

    with pytest.raises(ConfigValidationError, match="Unknown feature families"):
        parse_experiment_config(config)


def test_wrds_login_pattern_rejection() -> None:
    config = _valid_config()
    config["model"]["hyperparameters"]["credentials"] = "wrds.Connection(username='agent')"

    with pytest.raises(ConfigValidationError, match="WRDS login"):
        parse_experiment_config(config)


def test_invalid_promotion_metric_rejection() -> None:
    config = _valid_config()
    config["promotion"]["required_metrics"] = ["rank_ic", "made_up_metric"]
    config["promotion"]["metric_thresholds"]["made_up_metric"] = 0.0

    with pytest.raises(ConfigValidationError, match="Unknown required promotion metrics"):
        parse_experiment_config(config)


def test_unsupported_yaml_type_rejection() -> None:
    config = _valid_config()
    config["model"]["search_space"]["alpha"]["bad_values"] = {"set_values"}

    with pytest.raises(ConfigValidationError, match="Unsupported YAML value type"):
        parse_experiment_config(config)


def test_invalid_date_range_rejection() -> None:
    config = _valid_config()
    config["data"]["start_date"] = "2021-01-01"
    config["data"]["end_date"] = "2020-01-01"

    with pytest.raises(ConfigValidationError, match="start_date must be before"):
        parse_experiment_config(config)


def test_split_fraction_sum_rejection() -> None:
    config = _valid_config()
    config["splits"]["holdout_fraction"] = 0.20

    with pytest.raises(ConfigValidationError, match="sum to 1.0"):
        parse_experiment_config(config)


def test_mixed_split_policy_rejection() -> None:
    config = _valid_config()
    config["splits"]["train"] = {"start": "2000-01-01", "end": "2010-01-01"}

    with pytest.raises(ConfigValidationError, match="either fractions or explicit"):
        parse_experiment_config(config)


def test_explicit_split_out_of_range_rejection() -> None:
    config = _valid_config()
    config["splits"] = {
        "train": {"start": "1999-01-01", "end": "2010-01-01"},
        "validation": {"start": "2010-02-01", "end": "2015-01-01"},
        "holdout": {"start": "2015-02-01", "end": "2020-01-01"},
    }

    with pytest.raises(ConfigValidationError, match="within data date range"):
        parse_experiment_config(config)


def test_explicit_split_overlap_rejection() -> None:
    config = _valid_config()
    config["splits"] = {
        "train": {"start": "2000-01-01", "end": "2010-01-01"},
        "validation": {"start": "2009-12-31", "end": "2015-01-01"},
        "holdout": {"start": "2015-02-01", "end": "2020-01-01"},
    }

    with pytest.raises(ConfigValidationError, match="ordered and non-overlapping"):
        parse_experiment_config(config)
