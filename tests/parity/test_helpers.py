"""Frequency, shift and fast-length helpers in jitted code match NumPy and SciPy."""

import numpy as np
import pytest
import scipy.fft
from common import NumpyFFT, ScipyFFT, assert_new_array, assert_parity, random_array

import rocket_fft

LIBS = [(NumpyFFT, np.fft), (ScipyFFT, scipy.fft)]
HAS_PREV_FAST_LEN = hasattr(scipy.fft, "prev_fast_len")


@pytest.mark.parametrize("jit, lib", LIBS)
@pytest.mark.parametrize("name", ["fftfreq", "rfftfreq"])
def test_freq(jit, lib, name):
    for n in [1, 2, 8, 9, 16, 31, 100]:
        for d in [1.0, 0.1, np.pi, 3]:
            assert_parity(getattr(jit, name)(n, d), getattr(lib, name)(n, d))


@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("name", ["fftfreq", "rfftfreq"])
def test_freq_device(name, device):
    got = getattr(NumpyFFT, name)(8, 0.5, device)
    assert_parity(got, getattr(np.fft, name)(8, 0.5, device=device))


@pytest.mark.parametrize("jit, lib", LIBS)
@pytest.mark.parametrize("name", ["fftshift", "ifftshift"])
@pytest.mark.parametrize("axes", [None, 0, -1, (0,), (0, -1), (-1, 0)])
def test_shift(jit, lib, name, axes):
    for shape in [(8,), (9,), (4, 6), (3, 5), (3, 4, 5)]:
        if axes is not None and np.ndim(axes) and len(shape) < 2:
            continue
        x = random_array(shape)
        before = x.copy()
        got = getattr(jit, name)(x, axes)
        assert_parity(got, getattr(lib, name)(x, axes))
        assert_new_array(got, x, before)


@pytest.mark.parametrize("name", ["fftshift", "ifftshift"])
def test_shift_all_2d_shapes(name):
    # Includes empty and odd-sized dimensions.
    for i in range(16):
        for j in range(16):
            x = random_array((i, j))
            for axes in [0, 1, None, (0,), (0, 1)]:
                assert_parity(getattr(NumpyFFT, name)(x, axes), getattr(np.fft, name)(x, axes))


def test_shift_roundtrip():
    for n in [1, 4, 9, 100, 211]:
        x = random_array(n)
        np.testing.assert_array_equal(NumpyFFT.ifftshift(NumpyFFT.fftshift(x)), x)


@pytest.mark.parametrize("real", [False, True])
def test_next_fast_len(real):
    for n in list(range(1, 200)) + [1009, 10007, 2**20 + 1]:
        assert ScipyFFT.next_fast_len(n, real) == scipy.fft.next_fast_len(n, real)
        assert rocket_fft.next_fast_len(n, real) == scipy.fft.next_fast_len(n, real)


@pytest.mark.skipif(not HAS_PREV_FAST_LEN, reason="scipy.fft.prev_fast_len requires SciPy >= 1.15")
@pytest.mark.parametrize("real", [False, True])
def test_prev_fast_len(real):
    for n in list(range(0, 200)) + [1009, 10007, 2**20 + 1]:
        assert ScipyFFT.prev_fast_len(n, real) == scipy.fft.prev_fast_len(n, real)
        assert rocket_fft.prev_fast_len(n, real) == scipy.fft.prev_fast_len(n, real)
