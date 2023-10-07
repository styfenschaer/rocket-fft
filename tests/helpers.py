import os
from pathlib import Path

import pytest


def set_numba_capture_errors_new_style():
    # See: https://numba.readthedocs.io/en/latest/reference/
    # deprecation.html#deprecation-of-old-style-numba-captured-errors
    os.environ["NUMBA_CAPTURED_ERRORS"] = "new_style"


@pytest.fixture(autouse=True)
def numba_cache_cleanup():
    cache_dir = Path(__file__).parent / "__pycache__"

    for file in os.listdir(cache_dir):
        path = cache_dir / file
        if path.suffix in (".nbc", ".nbi"):
            try:
                os.remove(path)
            except (FileNotFoundError, PermissionError):
                pass
