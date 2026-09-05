"""Process-isolated entrypoint for governed Hugging Face corpus acquisition.

The normal acquisition code uses Hugging Face `datasets` streaming. A governed source may instead
specify ``data_files_glob`` when its physical Parquet shards are valid but repository-level feature
metadata is incompatible. Those sources are dispatched to the revision-pinned raw Parquet reader.

Provider work stays process-isolated because `datasets`/PyArrow readers have exhibited native
interpreter-finalization crashes after short-lived streams already produced valid output. We flush
all Python-managed output and only then bypass third-party finalizers with ``os._exit``.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Iterable
from typing import Any

import training.data.acquire as acquire
from training.data.catalog import DataSource
from training.data.hf_parquet import load_pinned_parquet_rows

_DEFAULT_HF_ROW_LOADER = acquire._load_hf_rows


def _governed_hf_row_loader(
    source: DataSource,
    *,
    revision: str,
    token: str | None,
) -> Iterable[dict[str, Any]]:
    """Route catalog-declared raw Parquet sources without hardcoding a language or dataset name."""
    if source.data_files_glob:
        return load_pinned_parquet_rows(source, revision=revision, token=token)
    return _DEFAULT_HF_ROW_LOADER(source, revision=revision, token=token)


def _install_provider_dispatch() -> None:
    """Install the governed loader at the process boundary used for all real corpus acquisition."""
    acquire._load_hf_rows = _governed_hf_row_loader


def _finish(code: int) -> None:
    """Flush Python-managed streams, then bypass unsafe third-party native finalizers."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


def run() -> None:
    """Run acquisition with governed provider dispatch and preserve the resulting process status."""
    try:
        _install_provider_dispatch()
        acquire.main()
    except SystemExit as exc:
        raw_code = exc.code
        code = raw_code if isinstance(raw_code, int) else (0 if raw_code is None else 1)
        _finish(code)
    except BaseException:
        traceback.print_exc()
        _finish(1)
    else:
        _finish(0)


if __name__ == "__main__":
    run()
