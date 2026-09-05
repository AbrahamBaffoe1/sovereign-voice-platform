"""Schema-tolerant Hugging Face Parquet reader for governed speech sources.

Some upstream dataset repositories contain valid Parquet shards whose physical schema has extra
columns that are absent from the repository-level `datasets.Features` metadata. In that situation,
`datasets.load_dataset` can reject the shard before yielding a single record. This module reads only
the catalog-mapped columns from immutable, revision-pinned Parquet files so unrelated physical
columns cannot invalidate otherwise correct speech data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from training.data.catalog import DataSource


def required_parquet_columns(source: DataSource) -> tuple[str, ...]:
    """Return unique physical columns required by the source's semantic field mapping."""
    return tuple(dict.fromkeys(value for value in source.fields.values() if value))


def load_pinned_parquet_rows(
    source: DataSource,
    *,
    revision: str,
    token: str | None,
    batch_size: int = 256,
) -> Iterable[dict[str, Any]]:
    """Yield mapped rows from revision-pinned Hub Parquet shards without Dataset feature casting."""
    if source.provider != "huggingface" or not source.repo_id:
        raise ValueError(f"source {source.source_id} is not a Hugging Face source")
    if not source.data_files_glob:
        raise ValueError(f"source {source.source_id} has no data_files_glob configured")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise RuntimeError("install the 'data' extra for revision-pinned Parquet acquisition") from exc

    filesystem = HfFileSystem(token=token)
    pattern = f"datasets/{source.repo_id}@{revision}/{source.data_files_glob.lstrip('/')}"
    paths = sorted(str(path) for path in filesystem.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"source {source.source_id} matched no Parquet files at revision {revision}: {source.data_files_glob}"
        )

    columns = required_parquet_columns(source)
    if not columns:
        raise ValueError(f"source {source.source_id} has no mapped fields")

    for path in paths:
        with filesystem.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            available = set(parquet.schema_arrow.names)
            missing = sorted(set(columns) - available)
            if missing:
                raise ValueError(
                    f"source {source.source_id} Parquet shard {path} is missing mapped columns: {missing}"
                )
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
                yield from batch.to_pylist()
