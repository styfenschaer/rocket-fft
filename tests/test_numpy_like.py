import numpy as np
from numba.core.errors import NumbaValueError
from numba.np.numpy_support import as_dtype
from pytest import raises as assert_raises

from helpers import set_numba_capture_errors_new_style, NumpyFFT
from rocket_fft.overloads import _as_cmplx_lut, numpy_like

set_numba_capture_errors_new_style()

numpy_like()


def test_numpy_like_dtypes():
    x = np.random.rand(42)

    for ty in _as_cmplx_lut.keys():
        ty = as_dtype(ty)

        dty1 = np.fft.fft(x.astype(ty)).dtype
        dty2 = NumpyFFT.fft(x.astype(ty)).dtype
        assert dty1 == dty2


def test_numpy_like_axes():
    x = np.random.rand(3, 3, 3, 3).astype(np.complex128)

    for fn in (NumpyFFT.fft2, NumpyFFT.fftn, NumpyFFT.ifft2, NumpyFFT.ifftn):
        for axes in [(0, 0), (0, 2, 2), (0, 2, 1, 0), (3, 1), (2, 1, 0), (0, 1, 2, 3)]:
            numpy_fn = getattr(np.fft, fn.__name__)

            try:
                res = fn(x, axes=axes)
                raises = False
            except NumbaValueError:
                res = None
                raises = True

            try:
                numpy_res = numpy_fn(x, axes=axes)
                numpy_raises = False
            except Exception:
                numpy_res = None
                numpy_raises = True

            assert raises == numpy_raises
            if res is not None and numpy_res is not None:
                assert np.allclose(res, numpy_res)
