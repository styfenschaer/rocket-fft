import numba as nb
import numpy as np
import scipy.fft
from numba.np.numpy_support import as_dtype
from pytest import raises as assert_raises

from rocket_fft import scipy_like
from rocket_fft.overloads import _scipy_cmplx_lut

scipy_like()


@nb.njit
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fft(x, n, axis, norm, overwrite_x, workers)


@nb.njit
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fft2(x, s, axes, norm, overwrite_x, workers)


@nb.njit
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fftn(x, s, axes, norm, overwrite_x, workers)


@nb.njit
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ifft2(x, s, axes, norm, overwrite_x, workers)


@nb.njit
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ifftn(x, s, axes, norm, overwrite_x, workers)


@nb.njit
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    return scipy.fft.dct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


@nb.njit
def fht(a, dln, mu, offset=0.0, bias=0.0):
    return scipy.fft.fht(a, dln, mu, offset, bias)


@nb.njit
def ifht(A, dln, mu, offset=0.0, bias=0.0):
    return scipy.fft.ifht(A, dln, mu, offset, bias)


@nb.njit
def fhtoffset(dln, mu, initial=0.0, bias=0.0):
    return scipy.fft.fhtoffset(dln, mu, initial, bias)


def test_scipy_like_dtypes():
    x = np.random.rand(42)

    for ty in _scipy_cmplx_lut.keys():
        ty = as_dtype(ty)
        
        dty1 = scipy.fft.fft(x.astype(ty)).dtype
        dty2 = fft(x.astype(ty)).dtype
        assert dty1 == dty2

        dty1 = scipy.fft.dct(x.astype(ty)).dtype
        dty2 = dct(x.astype(ty)).dtype
        assert dty1 == dty2
        
        if not isinstance(ty.type(0), np.complexfloating):
            one = ty.type(1.0) if isinstance(ty.type(0), np.number) else 1.0

            dty1 = scipy.fft.fht(x.astype(ty), one, one).dtype
            dty2 = fht(x.astype(ty), one, one).dtype
            assert dty1 == dty2
            
            dty1 = scipy.fft.ifht(x.astype(ty), one, one).dtype
            dty2 = ifht(x.astype(ty), one, one).dtype
            assert dty1 == dty2
            
            dty1 = scipy.fft.fhtoffset(one, one)
            dty2 = fhtoffset(one, one)
            assert type(dty1.item()) == type(dty2)
         
         
def test_scipy_like_axes():
    x = np.random.rand(3, 3, 3, 3).astype(np.complex128)
    
    for fn in (fft2, fftn, ifft2, ifftn):  
        for axes in [(0, 0), (0, 2, 2), (0, 2, 1, 0)]:
            with assert_raises(ValueError):
                fn(x, axes=axes)
            
    for fn in (fft2, fftn, ifft2, ifftn):  
        for axes in [(3, 1), (2, 1, 0), (0, 1, 2, 3)]:
            scipy_fn = getattr(scipy.fft, fn.__name__)
            assert np.allclose(fn(x, axes=axes), scipy_fn(x, axes=axes))
            