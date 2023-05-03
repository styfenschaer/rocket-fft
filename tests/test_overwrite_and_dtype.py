import numba as nb
import numpy as np
import scipy.fft
from helpers import numba_cache_cleanup
from numba import TypingError
from pytest import raises as assert_raises


@nb.njit
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fft(x, n, axis, norm, overwrite_x, workers)


@nb.njit
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.rfft(x, n, axis, norm, overwrite_x, workers)


@nb.njit
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.irfft(x, n, axis, norm, overwrite_x, workers)


@nb.njit
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    return scipy.fft.dct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


def test_r2r():
    x = np.random.rand(42).astype(np.float64)
    y = dct(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.float64)
    y = dct(x, overwrite_x=True)
    assert x is y

    x = np.random.rand(42).astype(np.byte)
    y = dct(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.float32)
    y = dct(x, overwrite_x=True)
    assert x is y
    assert y.dtype == np.float32
    
    x = np.random.rand(42).astype(np.complex128)
    y = dct(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.complex64)
    y = dct(x, overwrite_x=True)
    assert x is y
    assert y.dtype == np.complex64


def test_c2r():
    x = np.random.rand(42).astype(np.complex128)
    y = irfft(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.complex128)
    y = irfft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.complex64)
    y = irfft(x)
    assert x is not y
    assert y.dtype == np.float32

    x = np.random.rand(42).astype(np.complex64)
    y = irfft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.float64)
    y = irfft(x)
    assert x is not y
    assert y.dtype == np.float64

    x = np.random.rand(42).astype(np.float64)
    y = irfft(x, overwrite_x=True)
    assert x is not y


def test_r2c():
    x = np.random.rand(42).astype(np.float64)
    y = rfft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.float64)
    y = rfft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.int16)
    y = rfft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.int16)
    y = rfft(x, overwrite_x=True)
    assert x is not y

    with assert_raises(TypingError):
        x = np.random.rand(42).astype(np.complex128)
        y = rfft(x, overwrite_x=True)


def test_c2c():
    x = np.random.rand(42).astype(np.complex128)
    y = fft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.complex128)
    y = fft(x, overwrite_x=True)
    assert x is y

    x = np.random.rand(42).astype(np.complex64)
    y = fft(x)
    assert x is not y
    assert y.dtype == np.complex64

    x = np.random.rand(42).astype(np.complex64)
    y = fft(x, overwrite_x=True)
    assert x is y


def test_c2c_sym():
    x = np.random.rand(42).astype(np.float64)
    y = fft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.float64)
    y = fft(x, overwrite_x=True)
    assert x is not y

    x = np.random.rand(42).astype(np.bool_)
    y = fft(x)
    assert x is not y
    assert y.dtype == np.complex128

    x = np.random.rand(42).astype(np.bool_)
    y = fft(x, overwrite_x=True)
    assert x is not y
