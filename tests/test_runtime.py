"""Caching, multiprocessing, multithreading and worker settings."""

import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numba as nb
import numpy as np
import scipy.fft
from common import ScipyFFT, random_array

import rocket_fft

CACHING_SCRIPT = textwrap.dedent("""
    import numba as nb
    import numpy as np
    import scipy.fft

    njit = nb.njit(cache=True, nogil=True)
    funcs = [
        njit(lambda a: scipy.fft.fft(a)),
        njit(lambda a: scipy.fft.dct(a)),
        njit(lambda a: scipy.fft.fht(a, 1.0, 1.0)),
        njit(lambda a: np.fft.rfftn(a)),
    ]
    for f in funcs:
        f(np.ones(42))
    print(sum(sum(f.stats.cache_hits.values()) for f in funcs))
""")


def test_caching(tmp_path):
    script = tmp_path / "cached.py"
    script.write_text(CACHING_SCRIPT)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "cache")}

    def run():
        res = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        return int(res.stdout.split()[-1])

    assert run() == 0  # compiles and writes the cache
    assert run() == 4  # loads every function from the cache


def _compare_fft(a):
    return np.allclose(ScipyFFT.fft(a), scipy.fft.fft(a))


def _compare_dct(a):
    return np.allclose(ScipyFFT.dct(a), scipy.fft.dct(a))


def _compare_fht(a):
    return np.allclose(ScipyFFT.fht(a, 1.0, 1.0), scipy.fft.fht(a, 1.0, 1.0))


COMPARISONS = [_compare_fft, _compare_dct, _compare_fht]


def test_multiprocessing():
    inputs = [random_array(2**16, seed=i) for i in range(16)]
    for compare in COMPARISONS:
        assert all(map(compare, inputs[:1]))  # compile in the parent first
        with Pool(processes=4) as pool:
            assert all(pool.map(compare, inputs))


def test_multithreading():
    inputs = [random_array(2**16, seed=i) for i in range(32)]
    for compare in COMPARISONS:
        with ThreadPoolExecutor(max_workers=8) as executor:
            assert all(executor.map(compare, inputs))


@nb.njit
def _jit_get_workers():
    return rocket_fft.get_workers()


def test_set_workers_affects_compiled_functions():
    original = rocket_fft.get_workers()
    try:
        assert _jit_get_workers() == original  # compiled with the original default
        new = 2 if original != 2 else 1
        rocket_fft.set_workers(new)
        assert rocket_fft.get_workers() == new
        assert _jit_get_workers() == new
    finally:
        rocket_fft.set_workers(original)


def test_workers_give_same_result():
    x = random_array((64, 256), np.complex128)
    ref = scipy.fft.fft(x)
    for workers in [None, 1, 2, 4, -1]:
        np.testing.assert_allclose(ScipyFFT.fft(x, workers=workers), ref, rtol=1e-12, atol=1e-12)
