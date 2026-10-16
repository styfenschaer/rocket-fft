import os
import shutil
import tempfile

import numba
import pytest

_CACHE_DIR = pytest.StashKey[str]()


def pytest_configure(config):
    # Functions compiled with cache=True write to a throwaway directory instead of
    # __pycache__ next to the tests, so no run ever loads a stale cache. Subprocesses
    # inherit it through the environment.
    cache_dir = tempfile.mkdtemp(prefix="rocket-fft-numba-cache-")
    config.stash[_CACHE_DIR] = cache_dir
    numba.config.CACHE_DIR = cache_dir
    os.environ["NUMBA_CACHE_DIR"] = cache_dir


def pytest_unconfigure(config):
    cache_dir = config.stash.get(_CACHE_DIR, None)
    if cache_dir is not None:
        shutil.rmtree(cache_dir, ignore_errors=True)
