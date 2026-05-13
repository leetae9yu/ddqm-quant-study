from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from autoquant_lab.eqr.schemas import (
    EXPERIMENT_ARTIFACT_REQUIRED_FILES,
    EXPERIMENT_FEATURE_IMPORTANCE_REQUIRED_COLUMNS,
    EXPERIMENT_MANIFEST_REQUIRED_FIELDS,
    EXPERIMENT_PREDICTIONS_REQUIRED_COLUMNS,
)


def _panel(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for month in range(1, 7):
        for permno in range(1, 5):
            rows.append(
                {
                    "permno": permno,
                    "permco": permno,
                    "formation_date": pd.Timestamp(2020, month, 29 if month == 2 else 28),
                    "price": float(10 + permno),
                    "ret_1m": float(month * 0.001),
                    "market_cap": float(1000 + permno),
                    "ret_1m_fwd": float((permno - 2) * 0.01 + month * 0.001),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_eqr_train_writes_canonical_artifacts(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    output_dir = tmp_path / "runs"
    _panel(panel_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eqr_train.py",
            "--config",
            "configs/golden_path.yaml",
            "--model",
            "baseline_mean",
            "--panel",
            str(panel_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "unit_smoke",
            "--max-rows",
            "0",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    run_dir = Path(payload["run_dir"])

    for artifact in EXPERIMENT_ARTIFACT_REQUIRED_FILES:
        assert (run_dir / artifact).exists()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for field in EXPERIMENT_MANIFEST_REQUIRED_FIELDS:
        assert field in manifest

    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    assert tuple(predictions.columns) == EXPERIMENT_PREDICTIONS_REQUIRED_COLUMNS
    feature_importance = pd.read_csv(run_dir / "feature_importance.csv")
    assert tuple(feature_importance.columns) == EXPERIMENT_FEATURE_IMPORTANCE_REQUIRED_COLUMNS


def test_eqr_train_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    _panel(panel_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eqr_train.py",
            "--config",
            "configs/golden_path.yaml",
            "--model",
            "baseline_mean",
            "--panel",
            str(panel_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "../escape",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "run_id" in completed.stderr
