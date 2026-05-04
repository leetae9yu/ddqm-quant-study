#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Probe WRDS connectivity and print table schemas plus row counts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import re
from typing import Any


DEFAULT_TARGETS = ("crsp.dsf", "comp.funda")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely connect to WRDS and print schemas plus row counts for library.table targets."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        metavar="library.table",
        help="WRDS tables to probe, such as crsp.dsf or comp.funda. Defaults to common CRSP/Compustat tables.",
    )
    return parser.parse_args()


def parse_target(target: str) -> tuple[str, str]:
    parts = target.split(".")
    if len(parts) != 2 or not all(IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise ValueError(f"Invalid target {target!r}; expected library.table with simple SQL identifiers.")
    return parts[0].lower(), parts[1].lower()


def connect_wrds() -> Any | None:
    try:
        import wrds
    except ImportError:
        print("WRDS package is not installed. Install dependencies with: pip install -r requirements.txt")
        return None

    try:
        return wrds.Connection()
    except Exception as exc:  # noqa: BLE001 - authentication errors vary by local WRDS setup.
        exc_name = exc.__class__.__name__
        print(
            "Could not open a WRDS connection. "
            "Check your WRDS credentials, .pgpass, or interactive login setup. "
            f"Failure type: {exc_name}."
        )
        return None


def query_row_count(connection: Any, library: str, table: str) -> int | None:
    result = connection.raw_sql(f"select count(*) as row_count from {library}.{table}")
    if result.empty:
        return None
    return int(result.iloc[0]["row_count"])


def print_schema(schema: Any) -> None:
    if hasattr(schema, "to_string"):
        print(schema.to_string(index=False))
    else:
        print(schema)


def probe_table(connection: Any, library: str, table: str) -> None:
    print(f"\n== {library}.{table} ==")
    try:
        schema = connection.describe_table(library=library, table=table)
        print("Schema:")
        print_schema(schema)

        row_count = query_row_count(connection, library, table)
        print(f"Row count: {row_count if row_count is not None else 'unknown'}")
    except Exception as exc:  # noqa: BLE001 - keep probing other requested tables.
        print(f"Probe failed for {library}.{table}; failure type: {exc.__class__.__name__}.")


def probe_targets(connection: Any, targets: Iterable[str]) -> int:
    exit_code = 0
    for target in targets:
        try:
            library, table = parse_target(target)
        except ValueError as exc:
            print(exc)
            exit_code = 2
            continue
        probe_table(connection, library, table)
    return exit_code


def main() -> None:
    args = parse_args()
    connection = connect_wrds()
    if connection is None:
        return

    try:
        exit_code = probe_targets(connection, args.targets)
    finally:
        connection.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
