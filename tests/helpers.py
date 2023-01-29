import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def numba_cache_cleanup():
    cache_dir = Path(__file__).parent / '__pycache__'

    for file in os.listdir(cache_dir):
        path = cache_dir / file
        if path.suffix in ('.nbc', '.nbi'):
            try:
                os.remove(path)
            except (FileNotFoundError, PermissionError):
                pass
