from concurrent.futures import ThreadPoolExecutor

import numba as nb
import numpy as np
import scipy.fft
from helpers import (
    numba_cache_cleanup,
    set_numba_capture_errors_new_style,
    ScipyFFT,
)

set_numba_capture_errors_new_style()


def make_compare(f1, f2):
    def impl(arg):
        assert np.allclose(f1(arg), f2(arg))

    return impl


def test_all():
    inputs = [np.random.rand(2**20) for _ in range(42)]

    comp1 = make_compare(ScipyFFT.fft, scipy.fft.fft)
    comp2 = make_compare(ScipyFFT.dct, scipy.fft.dct)
    comp3 = make_compare(ScipyFFT.fht, lambda a: scipy.fft.fht(a, 1.0, 1.0))

    for func in (comp1, comp2, comp3):
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(func, inputs)
