#!/usr/bin/env python3
"""Validate an EQR experiment YAML config against the safe grammar."""
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
src_path = str(SRC_DIR)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from autoquant_lab.eqr.config import ConfigValidationError, load_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an EQR experiment YAML config.")
    parser.add_argument("config", type=Path, help="Path to YAML experiment config.")
    parser.add_argument("--json", action="store_true", help="Print normalized config JSON in addition to the hash.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(args.config)
    except (ConfigValidationError, OSError) as exc:
        print(f"Config validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"config_hash={config.stable_hash()}")
    if args.json:
        print(json.dumps(config.normalized_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
