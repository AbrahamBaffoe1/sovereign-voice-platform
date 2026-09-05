"""Process-isolated entrypoint for Hugging Face streaming acquisition.

`datasets`/PyArrow streaming readers have exhibited native interpreter-finalization crashes after a
short-lived iterator has already produced valid output. Acquisition is therefore isolated in a child
process and exits only after all corpus files/receipts have been closed and stdout/stderr flushed.
The hard process exit intentionally avoids third-party Python finalizers; it does not bypass our own
file/context-manager cleanup or error reporting.
"""

from __future__ import annotations

import os
import sys
import traceback

from training.data.acquire import main


def _finish(code: int) -> None:
    """Flush Python-managed streams, then bypass unsafe third-party native finalizers."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


def run() -> None:
    """Run the normal acquisition CLI and preserve its success/error status across the hard exit."""
    try:
        main()
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
