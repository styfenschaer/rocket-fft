from multiprocessing import Pool

import numba as nb
import numpy as np
import scipy.fft
from helpers import (
    numba_cache_cleanup,
    set_numba_capture_errors_new_style,
    ScipyFFT,
)

set_numba_capture_errors_new_style()


def func1(a):
    return np.allclose(scipy.fft.fft(a), ScipyFFT.fft(a))


def func2(a):
    return np.allclose(scipy.fft.dct(a), ScipyFFT.dct(a))


def func3(a):
    return np.allclose(scipy.fft.fht(a, 1.0, 1.0), ScipyFFT.fht(a, 1.0, 1.0))


def test_all():
    inputs = [np.random.rand(2**20) for _ in range(42)]
    for func in (func1, func2, func3):
        # Run once to cache functions
        list(map(func, inputs))

        with Pool(processes=8) as pool:
            ret = pool.map(func, inputs)
            assert all(ret)


if __name__ == "__main__":
    test_all()
