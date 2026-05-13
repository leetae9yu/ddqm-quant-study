#!/usr/bin/env python3
"""Validate that the canonical EQR package imports from the active path."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EQR_DIR = REPO_ROOT / "src" / "autoquant_lab" / "eqr"
repo_root_path = str(REPO_ROOT)
if repo_root_path not in sys.path:
    sys.path.insert(0, repo_root_path)


def main() -> None:
    eqr = importlib.import_module("autoquant_lab.eqr")

    exported = set(getattr(eqr, "__all__", ()))
    required = {"EQR_PANEL_REQUIRED_COLUMNS", "read_dataset", "require_columns"}
    missing = sorted(required.difference(exported))
    if missing:
        raise SystemExit(f"autoquant_lab.eqr is missing expected exports: {missing}")
    module_file = Path(getattr(eqr, "__file__", "")).resolve()
    if module_file.parent != EXPECTED_EQR_DIR:
        raise SystemExit(f"autoquant_lab.eqr resolved from unexpected path: {module_file}")
    print("autoquant_lab.eqr import OK")


if __name__ == "__main__":
    main()
