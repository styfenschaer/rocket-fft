import os
from pathlib import Path

import pytest


# @pytest.fixture(autouse=True)
# def numba_cache_cleanup():
#     yield
#     root = Path(__file__).parent
#     cache_dir = root / '__pycache__'
#     for file in os.listdir(cache_dir):
#         path = cache_dir / file
#         if path.suffix in ['.nbc', '.nbi']:
#             os.remove(path)


