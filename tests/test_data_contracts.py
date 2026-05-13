from __future__ import annotations
# pyright: reportMissingImports=false

import importlib.util
from pathlib import Path

import pandas as pd

from autoquant_lab.eqr.data_contracts import CONTRACTS_BY_NAME
from autoquant_lab.eqr.path_resolver import resolve_data_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "eqr_validate_raw_data.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("eqr_validate_raw_data", VALIDATOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contracts_define_required_dates_and_keys_for_all_artifacts() -> None:
    expected = {
        "crsp_monthly",
        "crsp_names",
        "ccm_link",
        "comp_company",
        "comp_fundq",
        "ibes_link",
        "ibes_summary",
        "ibes_detail",
        "ibes_actual",
        "ibes_target",
        "macro_features",
    }

    assert set(CONTRACTS_BY_NAME) == expected
    for contract in CONTRACTS_BY_NAME.values():
        assert contract.required_columns
        assert contract.date_columns
        assert contract.key_columns


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"date": pd.to_datetime(["2024-01-31"]), "ret": [0.01]}).to_parquet(data_dir / "monthly_crsp_msf.parquet")

    validator = _load_validator_module()
    resolved = resolve_data_paths(data_dir, include_date_ranges=False)
    result = validator.validate_artifact(CONTRACTS_BY_NAME["crsp_monthly"], resolved["crsp_monthly"])

    assert not result["ok"]
    assert "permno" in result["required_column_status"]["missing_columns"]


def test_offline_guard_rejects_wrds_connection(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bad_dir = project_root / "src" / "autoquant_lab" / "eqr"
    bad_dir.mkdir(parents=True)
    (project_root / "scripts").mkdir()
    (project_root / "configs").mkdir()
    (bad_dir / "bad.py").write_text("import wrds\nconn = wrds.Connection()\n", encoding="utf-8")

    validator = _load_validator_module()
    findings = validator.scan_offline_guard(project_root)

    assert findings
    assert findings[0]["pattern"] == "wrds_connection"
