import numpy as np
import scipy.fft
from numba.core.errors import NumbaValueError
from numba.np.numpy_support import as_dtype
from pytest import raises as assert_raises

from helpers import set_numba_capture_errors_new_style, ScipyFFT
from rocket_fft.overloads import _as_cmplx_lut, scipy_like

set_numba_capture_errors_new_style()

scipy_like()


def test_scipy_like_dtypes():
    x = np.random.rand(42)

    for ty in _as_cmplx_lut.keys():
        ty = as_dtype(ty)

        dty1 = scipy.fft.fft(x.astype(ty)).dtype
        dty2 = ScipyFFT.fft(x.astype(ty)).dtype
        assert dty1 == dty2

        dty1 = scipy.fft.dct(x.astype(ty)).dtype
        dty2 = ScipyFFT.dct(x.astype(ty)).dtype
        assert dty1 == dty2

        if not isinstance(ty.type(0), np.complexfloating):
            one = ty.type(1.0) if isinstance(ty.type(0), np.number) else 1.0

            dty1 = scipy.fft.fht(x.astype(ty), one, one).dtype
            dty2 = ScipyFFT.fht(x.astype(ty), one, one).dtype
            assert dty1 == dty2

            dty1 = scipy.fft.ifht(x.astype(ty), one, one).dtype
            dty2 = ScipyFFT.ifht(x.astype(ty), one, one).dtype
            assert dty1 == dty2

            dty1 = scipy.fft.fhtoffset(one, one)
            dty2 = ScipyFFT.fhtoffset(one, one)
            assert type(dty1.item()) == type(dty2)


def test_scipy_like_axes():
    x = np.random.rand(3, 3, 3, 3).astype(np.complex128)

    for fn in (ScipyFFT.fft2, ScipyFFT.fftn, ScipyFFT.ifft2, ScipyFFT.ifftn):
        for axes in [(0, 0), (0, 2, 2), (0, 2, 1, 0)]:
            with assert_raises(NumbaValueError):
                fn(x, axes=axes)

    for fn in (ScipyFFT.fft2, ScipyFFT.fftn, ScipyFFT.ifft2, ScipyFFT.ifftn):
        for axes in [(3, 1), (2, 1, 0), (0, 1, 2, 3)]:
            scipy_fn = getattr(scipy.fft, fn.__name__)
            assert np.allclose(fn(x, axes=axes), scipy_fn(x, axes=axes))

    for fn in (ScipyFFT.fft2, ScipyFFT.fftn, ScipyFFT.ifft2, ScipyFFT.ifftn):
        for axes in [(0, 0), (0, 2, 2), (0, 2, 1, 0), (3, 1), (2, 1, 0), (0, 1, 2, 3)]:
            scipy_fn = getattr(scipy.fft, fn.__name__)

            try:
                res = fn(x, axes=axes)
                raises = False
            except NumbaValueError:
                res = None
                raises = True

            try:
                scipy_res = scipy_fn(x, axes=axes)
                scipy_raises = False
            except Exception:
                scipy_res = None
                scipy_raises = True

            assert raises == scipy_raises
            if res is not None and scipy_res is not None:
                assert np.allclose(res, scipy_res)
