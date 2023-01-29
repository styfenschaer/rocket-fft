import numba as nb
import numpy as np
import scipy.fft
from numba.np.numpy_support import as_dtype

from rocket_fft import scipy_like
from rocket_fft.overloads import _scipy_cmplx_lut


@nb.njit
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fft(x, n, axis, norm, overwrite_x, workers)


@nb.njit
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    return scipy.fft.dct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


def test_scipy_like():
    x = np.random.rand(42)

    scipy_like()

    for ty in _scipy_cmplx_lut.keys():
        ty = as_dtype(ty)
        
        dty1 = scipy.fft.fft(x.astype(ty)).dtype
        dty2 = fft(x.astype(ty)).dtype
        assert dty1 == dty2

        dty1 = scipy.fft.dct(x.astype(ty)).dtype
        dty2 = dct(x.astype(ty)).dtype
        assert dty1 == dty2
