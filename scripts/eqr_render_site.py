#!/usr/bin/env python3
"""Render the EQR experiment history static site."""
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from autoquant_lab.eqr.reporting import render_site  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = PROJECT_ROOT / "experiments" / "ledger.sqlite"
DEFAULT_OUTPUT = PROJECT_ROOT / "site"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render EQR reports and static experiment history site.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="SQLite experiment ledger path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Static HTML output directory.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="Experiment run artifact directory.")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="Markdown/JSON source report directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = render_site(ledger_path=args.ledger, output_dir=args.output, run_root=args.run_root, reports_dir=args.reports)
    print(
        {
            "output_dir": str(result.output_dir),
            "reports_dir": str(result.reports_dir),
            "html_files": len(result.html_files),
            "report_files": len(result.report_files),
            "run_count": result.run_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
