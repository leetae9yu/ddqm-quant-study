from __future__ import annotations
# pyright: reportMissingImports=false

from pathlib import Path

import pytest

from autoquant_lab.eqr.path_resolver import resolve_data_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_path_resolution_on_current_data_layout() -> None:
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        pytest.skip("local data directory is not available")

    resolved = resolve_data_paths(data_dir, include_date_ranges=False)

    assert resolved["crsp_monthly"].path.name == "monthly_crsp_msf_by_year"
    assert resolved["crsp_monthly"].is_partitioned
    assert resolved["crsp_monthly"].row_count > 0
    assert resolved["crsp_names"].path.name == "crsp_msenames.parquet"
    assert resolved["ccm_link"].path.name == "ccm_linktable.parquet"
    assert resolved["comp_company"].path.name == "comp_company.parquet"
    assert resolved["comp_fundq"].path.name == "comp_fundq_by_year"
    assert resolved["ibes_link"].path.name == "ibes_link.parquet"
    assert resolved["ibes_summary"].path.name == "ibes_statsum_epsus_by_year"
    assert resolved["ibes_detail"].path.name == "ibes_det_epsus_by_year"
    assert resolved["ibes_actual"].path.name == "ibes_act_epsus_by_year"
    assert resolved["ibes_target"].path.name == "ibes_ptgsum_by_year"
    assert resolved["macro_features"].path.name == "macro_features.parquet"


def test_resolver_ignores_stale_manifest_paths() -> None:
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        pytest.skip("local data directory is not available")

    resolved = resolve_data_paths(data_dir, include_date_ranges=False)

    for artifact in resolved.values():
        assert artifact.path.exists()
        assert "data/raw/eqr_real" not in str(artifact.path)
