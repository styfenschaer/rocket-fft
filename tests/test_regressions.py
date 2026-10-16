"""Regression tests for behaviour that previously diverged from NumPy/SciPy or crashed."""

import gc

import numba as nb
import numpy as np
import pytest
import scipy.fft
from common import NumpyFFT, ScipyFFT
from numba.core.runtime import _nrt_python, rtsys

import rocket_fft


@pytest.mark.parametrize("n", [0, 1])
def test_irfft_too_short(n):
    x = np.ones(n, dtype=np.complex128)
    with pytest.raises(ValueError):
        np.fft.irfft(x)
    with pytest.raises(ValueError):
        NumpyFFT.irfft(x)


def test_lowlevel_axis_out_of_range():
    @nb.njit
    def c2c(a, b, axes):
        rocket_fft.c2c(a, b, axes, True, 1.0, 1)

    a = np.random.rand(4, 8) + 0j
    with pytest.raises(ValueError):
        c2c(a, np.zeros_like(a), np.array([5]))


@pytest.mark.parametrize("fn", [rocket_fft.good_size, rocket_fft.prev_good_size])
def test_fast_len_negative_target(fn):
    with pytest.raises(ValueError):
        fn(-5)
    with pytest.raises(ValueError):
        nb.njit(lambda n: fn(n))(-5)


def test_fast_len_python_and_jit_agree():
    for n in [0, 1, 7, 13, 97, 1000, 1023]:
        for real in (False, True):
            assert rocket_fft.good_size(n, real) == nb.njit(lambda n, r: rocket_fft.good_size(n, r))(n, real)
            assert rocket_fft.next_fast_len(n, real) == scipy.fft.next_fast_len(n, real)


def test_minus_one_selects_default_length():
    x = np.random.rand(4, 6) + 0j
    assert np.allclose(ScipyFFT.fftn(x, s=(-1, 8), axes=(0, 1)), scipy.fft.fftn(x, s=(-1, 8), axes=(0, 1)))
    assert np.allclose(NumpyFFT.fftn(x, s=(8, -1), axes=(0, 1)), np.fft.fftn(x, s=(8, -1), axes=(0, 1)))


def test_numpy_duplicate_axes_with_shape():
    x = np.random.rand(4, 6) + 0j
    s, axes = (6, 3), (0, 0)
    assert np.allclose(NumpyFFT.fftn(x, s=s, axes=axes), np.fft.fftn(x, s=s, axes=axes))


def test_overwrite_readonly_input():
    x = np.random.rand(8) + 0j
    x.flags.writeable = False
    y = ScipyFFT.fft(x, overwrite_x=True)
    assert y is not x
    assert np.allclose(y, scipy.fft.fft(x))


def _live_allocations():
    gc.collect()
    stats = rtsys.get_allocation_stats()
    return stats.alloc - stats.free


def test_no_leaks():
    _nrt_python.memsys_enable_stats()
    x = np.random.rand(8, 8)
    calls = [
        lambda: NumpyFFT.fft(x),
        lambda: ScipyFFT.rfftn(x),
        lambda: ScipyFFT.dctn(x),
        lambda: ScipyFFT.fht(x, 0.1, 0.5),
        lambda: NumpyFFT.fftfreq(8, 0.1),
        lambda: NumpyFFT.fftshift(x),
    ]
    for call in calls:
        call()  # compile outside the measurement
    before = _live_allocations()
    for _ in range(10):
        for call in calls:
            call()
        with pytest.raises(ValueError):
            NumpyFFT.fftshift(x, axes=5)
    assert _live_allocations() == before
