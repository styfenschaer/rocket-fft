import numba as nb
import numpy as np
from numba.np.numpy_support import as_dtype

from rocket_fft import numpy_like
from rocket_fft.overloads import _numpy_cmplx_lut

numpy_like()
    
    
@nb.njit
def fft(a, n=None, axis=-1, norm=None):
    return np.fft.fft(a, n, axis, norm)


@nb.njit
def fft2(x, s=None, axes=(-2, -1), norm=None):
    return np.fft.fft2(x, s, axes, norm)


@nb.njit
def fftn(x, s=None, axes=None, norm=None):
    return np.fft.fftn(x, s, axes, norm)


@nb.njit
def ifft2(x, s=None, axes=(-2, -1), norm=None):
    return np.fft.ifft2(x, s, axes, norm)


@nb.njit
def ifftn(x, s=None, axes=None, norm=None):
    return np.fft.ifftn(x, s, axes, norm)


def test_numpy_like_dtypes():
    x = np.random.rand(42)

    for ty in _numpy_cmplx_lut.keys():
        ty = as_dtype(ty)

        dty1 = np.fft.fft(x.astype(ty)).dtype
        dty2 = fft(x.astype(ty)).dtype
        assert dty1 == dty2


def test_numpy_like_axes():
    x = np.random.rand(3, 3, 3, 3).astype(np.complex128)
    
    for fn in (fft2, fftn, ifft2, ifftn): 
        for axes in [(0, 0), (0, 2, 2), (0, 2, 1, 0), (3, 1), (2, 1, 0), (0, 1, 2, 3)]:
            numpy_fn = getattr(np.fft, fn.__name__)
            assert np.allclose(fn(x, axes=axes), numpy_fn(x, axes=axes))