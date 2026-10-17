"""The fast path for single 1-D transforms gives exactly the results of the general path.

The native library runs single 1-D transforms into a contiguous output through
DUCC0's 1-D plans directly (see run_c2c_1d and friends). With a strided output,
the same transform takes DUCC0's general n-dimensional path, so comparing both
checks that the fast path is bit-identical.
"""

import numba as nb
import numpy as np
import pytest
from common import NumpyFFT, random_array

from rocket_fft import c2c, c2c_sym, c2r, r2c

FAST_PATH_MAX_LENGTH = 2**16  # kFastPathMaxLength in _rocket_fft_numba.cpp
LENGTHS = list(range(1, 70)) + [97, 127, 128, 1009, 1024, 4096]
AXES = np.array([0])


@nb.njit
def jit_c2c(ain, aout, forward, fct):
    c2c(ain, aout, AXES, forward, fct, 1)


@nb.njit
def jit_c2c_sym(ain, aout, forward, fct):
    c2c_sym(ain, aout, AXES, forward, fct, 1)


@nb.njit
def jit_r2c(ain, aout, forward, fct):
    r2c(ain, aout, AXES, forward, fct, 1)


@nb.njit
def jit_c2r(ain, aout, forward, fct):
    c2r(ain, aout, AXES, forward, fct, 1)


def _both_paths(fn, ain, out_len, out_dtype, forward, fct):
    """Results with a contiguous output (fast path) and a strided one (general path)."""
    fast = np.empty(out_len, out_dtype)
    general = np.empty(2 * out_len, out_dtype)[::2]
    fn(ain, fast, forward, fct)
    fn(ain, general, forward, fct)
    return fast, general


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("real", [np.float64, np.float32])
def test_bit_identical(real, forward):
    cplx = np.result_type(real, np.complex64)
    fct = real(0.5)
    for n in LENGTHS:
        x = random_array(n, real, seed=n)
        z = random_array(n, cplx, seed=n)
        h = random_array(n // 2 + 1, cplx, seed=n)
        cases = [
            ("c2c", jit_c2c, z, n, cplx),
            ("c2c_sym", jit_c2c_sym, x, n, cplx),
            ("r2c", jit_r2c, x, n // 2 + 1, cplx),
            ("c2r", jit_c2r, h, n, real),
        ]
        for name, fn, ain, out_len, out_dtype in cases:
            fast, general = _both_paths(fn, ain, out_len, out_dtype, forward, fct)
            np.testing.assert_array_equal(fast, general, err_msg=f"{name}, n={n}")


@pytest.mark.parametrize("n", [FAST_PATH_MAX_LENGTH - 1, FAST_PATH_MAX_LENGTH, FAST_PATH_MAX_LENGTH + 1])
def test_lengths_around_the_limit(n):
    # Below the limit 1-D transforms take the fast path, above it the general path.
    x = random_array(n)
    z = random_array(n, np.complex128)
    tol = dict(rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(NumpyFFT.fft(z), np.fft.fft(z), **tol)
    np.testing.assert_allclose(NumpyFFT.fft(x), np.fft.fft(x), **tol)
    np.testing.assert_allclose(NumpyFFT.rfft(x), np.fft.rfft(x), **tol)
    y = np.fft.rfft(x)
    np.testing.assert_allclose(NumpyFFT.irfft(y, n), np.fft.irfft(y, n), **tol)
