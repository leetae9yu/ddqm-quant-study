"""Resolve local EQR raw artifact paths without trusting stale manifests."""
# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .data_contracts import ARTIFACT_CONTRACTS, ArtifactContract


@dataclass(frozen=True)
class ResolvedArtifact:
    """Resolved local location and metadata for one logical artifact."""

    artifact_name: str
    artifact_type: str
    path: Path
    is_partitioned: bool
    files: tuple[Path, ...]
    row_count: int
    columns: tuple[str, ...]
    date_ranges: dict[str, dict[str, str | None]] = field(default_factory=dict)
    canonical_location: str = ""
    current_location: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_type": self.artifact_type,
            "path": str(self.path),
            "is_partitioned": self.is_partitioned,
            "files": [str(path) for path in self.files],
            "row_count": self.row_count,
            "columns": list(self.columns),
            "date_ranges": self.date_ranges,
            "canonical_location": self.canonical_location,
            "current_location": self.current_location,
        }


def _parquet_files(path: Path) -> tuple[Path, ...]:
    if path.is_file() and path.suffix == ".parquet":
        return (path,)
    if path.is_dir():
        return tuple(sorted(child for child in path.iterdir() if child.is_file() and child.suffix == ".parquet"))
    return ()


def _metadata_for_files(files: tuple[Path, ...]) -> tuple[int, tuple[str, ...]]:
    row_count = 0
    columns: tuple[str, ...] = ()
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        row_count += parquet_file.metadata.num_rows
        file_columns = tuple(parquet_file.schema.names)
        if not columns:
            columns = file_columns
    return row_count, columns


def _date_range_for_files(files: tuple[Path, ...], column: str) -> dict[str, str | None]:
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    non_null_count = 0
    parse_failures = 0

    for file_path in files:
        series = pd.read_parquet(file_path, columns=[column])[column]
        parsed = pd.to_datetime(series, errors="coerce")
        non_null = series.notna()
        non_null_count += int(non_null.sum())
        parse_failures += int((parsed.isna() & non_null).sum())
        if parsed.notna().any():
            file_min = parsed.min()
            file_max = parsed.max()
            minimum = file_min if minimum is None or file_min < minimum else minimum
            maximum = file_max if maximum is None or file_max > maximum else maximum

    return {
        "min": minimum.date().isoformat() if minimum is not None else None,
        "max": maximum.date().isoformat() if maximum is not None else None,
        "non_null_count": str(non_null_count),
        "parse_failures": str(parse_failures),
    }


def _resolve_contract(data_dir: Path, contract: ArtifactContract, include_date_ranges: bool) -> ResolvedArtifact | None:
    for relative_location in contract.current_locations:
        candidate = data_dir / relative_location
        files = _parquet_files(candidate)
        if not files:
            continue

        row_count, columns = _metadata_for_files(files)
        available = set(columns)
        date_ranges = {
            spec.name: _date_range_for_files(files, spec.name)
            for spec in contract.date_columns
            if include_date_ranges and spec.name in available
        }
        return ResolvedArtifact(
            artifact_name=contract.artifact_name,
            artifact_type=contract.artifact_type,
            path=candidate,
            is_partitioned=candidate.is_dir(),
            files=files,
            row_count=row_count,
            columns=columns,
            date_ranges=date_ranges,
            canonical_location=contract.canonical_location,
            current_location=relative_location,
        )
    return None


def resolve_data_paths(data_dir: str | Path, include_date_ranges: bool = True) -> dict[str, ResolvedArtifact]:
    """Map the actual local data layout to logical EQR artifact names."""

    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {root}")

    resolved: dict[str, ResolvedArtifact] = {}
    for contract in ARTIFACT_CONTRACTS:
        artifact = _resolve_contract(root, contract, include_date_ranges=include_date_ranges)
        if artifact is not None:
            resolved[contract.artifact_name] = artifact
    return resolved


__all__ = ["ResolvedArtifact", "resolve_data_paths"]
