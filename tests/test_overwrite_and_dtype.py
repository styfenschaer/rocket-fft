import numba as nb
import numpy as np
import scipy.fft
from helpers import (
    numba_cache_cleanup,
    set_numba_capture_errors_new_style,
    ScipyFFT,
)
from numba.core.errors import TypingError
from pytest import raises as assert_raises

set_numba_capture_errors_new_style()


def test_r2r():
    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.dct(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.dct(x, overwrite_x=True)
    assert x is y

    x = np.random.rand(42).astype(np.byte)
    y = ScipyFFT.dct(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.float32)
    y = ScipyFFT.dct(x, overwrite_x=True)
    assert x is y
    assert y.dtype == np.float32

    x = np.random.rand(42).astype(np.complex128)
    y = ScipyFFT.dct(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.complex64)
    y = ScipyFFT.dct(x, overwrite_x=True)
    assert x is y
    assert y.dtype == np.complex64


def test_c2r():
    x = np.random.rand(42).astype(np.complex128)
    y = ScipyFFT.irfft(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.complex128)
    y = ScipyFFT.irfft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.complex64)
    y = ScipyFFT.irfft(x)
    assert x is not y
    assert y.dtype == np.float32

    x = np.random.rand(42).astype(np.complex64)
    y = ScipyFFT.irfft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.irfft(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.irfft(x, overwrite_x=True)
    assert x is not y


def test_r2c():
    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.rfft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.rfft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.int16)
    y = ScipyFFT.rfft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.int16)
    y = ScipyFFT.rfft(x, overwrite_x=True)
    assert x is not y

    with assert_raises(TypingError):
        x = np.random.rand(42).astype(np.complex128)
        y = ScipyFFT.rfft(x, overwrite_x=True)


def test_c2c():
    x = np.random.rand(42).astype(np.complex128)
    y = ScipyFFT.fft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.complex128)
    y = ScipyFFT.fft(x, overwrite_x=True)
    assert x is y

    x = np.random.rand(42).astype(np.complex64)
    y = ScipyFFT.fft(x)
    assert x is not y
    assert y.dtype == np.complex64

    x = np.random.rand(42).astype(np.complex64)
    y = ScipyFFT.fft(x, overwrite_x=True)
    assert x is y


def test_c2c_sym():
    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.fft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.float64)
    y = ScipyFFT.fft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.bool_)
    y = ScipyFFT.fft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.bool_)
    y = ScipyFFT.fft(x, overwrite_x=True)
    assert x is not y
