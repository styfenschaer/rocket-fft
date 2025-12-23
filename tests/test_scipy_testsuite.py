"""
These tests are borrowed from Scipy.
Thanks to all who contributed to these tests.
https://github.com/scipy/scipy/blob/main/scipy/fft/_pocketfft/tests/test_basic.py
https://github.com/scipy/scipy/blob/main/scipy/fft/_pocketfft/tests/test_real_transforms.py
Whenever I changed a test, I left a note.
"""

from functools import partial
from os.path import join
from typing import Callable, Dict, Tuple, Type, Union

import numba as nb
import numpy as np
import numpy.fft
import pytest
import scipy.fft
from helpers import numba_cache_cleanup, set_numba_capture_errors_new_style
from numba.core.errors import NumbaValueError, TypingError
from numpy import add, arange, array, asarray, cdouble, dot, exp, pi, swapaxes, zeros
from numpy.random import rand
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_almost_equal_nulp,
    assert_array_less,
    assert_equal,
)
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)
from scipy.fft._pocketfft.tests.test_real_transforms import fftpack_test_dir

set_numba_capture_errors_new_style()

# At maximum double precision is supported
np.longcomplex = np.complex128
np.longdouble = np.float64
np.longfloat = np.float64

# All functions should be cacheable and run without the GIL
njit = partial(nb.njit, cache=True, nogil=True)


@njit
def dct(
    x,
    type=2,
    n=None,
    axis=-1,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.dct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


@njit
def idct(
    x,
    type=2,
    n=None,
    axis=-1,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.idct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


@njit
def dst(
    x,
    type=2,
    n=None,
    axis=-1,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.dst(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


@njit
def idst(
    x,
    type=2,
    n=None,
    axis=-1,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.idst(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)


@njit
def dctn(
    x,
    type=2,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.dctn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)


@njit
def idctn(
    x,
    type=2,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.idctn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)


@njit
def dstn(
    x,
    type=2,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.dstn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)


@njit
def idstn(
    x,
    type=2,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    orthogonalize=None,
):
    return scipy.fft.idstn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)


@njit
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fft(x, n, axis, norm, overwrite_x, workers)


@njit
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fft2(x, s, axes, norm, overwrite_x, workers)


@njit
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.fftn(x, s, axes, norm, overwrite_x, workers)


@njit
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ifft(x, n, axis, norm, overwrite_x, workers)


@njit
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ifft2(x, s, axes, norm, overwrite_x, workers)


@njit
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ifftn(x, s, axes, norm, overwrite_x, workers)


@njit
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.rfft(x, n, axis, norm, overwrite_x, workers)


@njit
def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.rfft2(x, s, axes, norm, overwrite_x, workers)


@njit
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.rfftn(x, s, axes, norm, overwrite_x, workers)


@njit
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.irfft(x, n, axis, norm, overwrite_x, workers)


@njit
def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.irfft2(x, s, axes, norm, overwrite_x, workers)


@njit
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.irfftn(x, s, axes, norm, overwrite_x, workers)


@njit
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.hfft(x, n, axis, norm, overwrite_x, workers)


@njit
def hfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.hfft2(x, s, axes, norm, overwrite_x, workers)


@njit
def hfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.hfftn(x, s, axes, norm, overwrite_x, workers)


@njit
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ihfft(x, n, axis, norm, overwrite_x, workers)


@njit
def ihfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ihfft2(x, s, axes, norm, overwrite_x, workers)


@njit
def ihfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    return scipy.fft.ihfftn(x, s, axes, norm, overwrite_x, workers)


# "large" composite numbers supported by FFT._PYPOCKETFFT
LARGE_COMPOSITE_SIZES = [
    2**13,
    2**5 * 3**5,
    2**3 * 3**3 * 5**2,
]
SMALL_COMPOSITE_SIZES = [
    2,
    2 * 3 * 5,
    2 * 2 * 3 * 3,
]
# prime
LARGE_PRIME_SIZES = [2011]
SMALL_PRIME_SIZES = [29]


def _assert_close_in_norm(x, y, rtol, size, rdt):
    # helper function for testing
    err_msg = "size: %s  rdt: %s" % (size, rdt)
    assert_array_less(np.linalg.norm(x - y), rtol * np.linalg.norm(x), err_msg)


def random(size):
    return rand(*size)


def swap_byteorder(arr):
    """Returns the same array with swapped byteorder"""
    dtype = arr.dtype.newbyteorder("S")
    return arr.astype(dtype)


def get_mat(n):
    data = arange(n)
    data = add.outer(data, data)
    return data


def direct_dft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = -arange(n) * (2j * pi / n)
    for i in range(n):
        y[i] = dot(exp(i * w), x)
    return y


def direct_idft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = arange(n) * (2j * pi / n)
    for i in range(n):
        y[i] = dot(exp(i * w), x) / n
    return y


def direct_dftn(x):
    x = asarray(x)
    for axis in range(x.ndim):
        x = fft(x, axis=axis)
    return x


def direct_idftn(x):
    x = asarray(x)
    for axis in range(x.ndim):
        x = ifft(x, axis=axis)
    return x


def direct_rdft(x):
    x = asarray(x)
    n = len(x)
    w = -arange(n) * (2j * pi / n)
    y = zeros(n // 2 + 1, dtype=cdouble)
    for i in range(n // 2 + 1):
        y[i] = dot(exp(i * w), x)
    return y


def direct_irdft(x, n):
    x = asarray(x)
    x1 = zeros(n, dtype=cdouble)
    for i in range(n // 2 + 1):
        x1[i] = x[i]
        if i > 0 and 2 * i < n:
            x1[n - i] = np.conj(x[i])
    return direct_idft(x1).real


def direct_rdftn(x):
    return fftn(rfft(x), axes=range(x.ndim - 1))


class _TestFFTBase:
    def setup_method(self):
        self.cdt = None
        self.rdt = None
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([1, 2, 3, 4 + 1j, 1, 2, 3, 4 + 2j], dtype=self.cdt)
        y = fft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_dft(x)
        assert_array_almost_equal(y, y1)
        x = np.array([1, 2, 3, 4 + 0j, 5], dtype=self.cdt)
        assert_array_almost_equal(fft(x), direct_dft(x))

    def test_n_argument_real(self):
        x1 = np.array([1, 2, 3, 4], dtype=self.rdt)
        x2 = np.array([1, 2, 3, 4], dtype=self.rdt)
        y = fft(np.array([x1, x2]), n=4)
        assert_equal(y.dtype, self.cdt)
        assert_equal(y.shape, (2, 4))
        assert_array_almost_equal(y[0], direct_dft(x1))
        assert_array_almost_equal(y[1], direct_dft(x2))

    def _test_n_argument_complex(self):
        x1 = np.array([1, 2, 3, 4 + 1j], dtype=self.cdt)
        x2 = np.array([1, 2, 3, 4 + 1j], dtype=self.cdt)
        y = fft(np.array([x1, x2]), n=4)
        assert_equal(y.dtype, self.cdt)
        assert_equal(y.shape, (2, 4))
        assert_array_almost_equal(y[0], direct_dft(x1))
        assert_array_almost_equal(y[1], direct_dft(x2))

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2**i
            # Int64 to avoid numerical problems
            x = np.arange(n).astype(np.int64)
            y = fft(x.astype(complex))
            y2 = numpy.fft.fft(x)
            assert_array_almost_equal(y, y2, decimal=6)
            y = fft(x)
            assert_array_almost_equal(y, y2, decimal=6)

    def test_invalid_sizes(self):
        # NOTE: Tests modified.
        assert_raises(ValueError, fft, [])
        assert_raises(NumbaValueError, fft, np.array([[1, 1], [2, 2]]), -5)


class TestLongDoubleFFT(_TestFFTBase):
    def setup_method(self):
        self.cdt = np.longcomplex
        self.rdt = np.longdouble


class TestDoubleFFT(_TestFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double


class TestSingleFFT(_TestFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32


# NOTE: Numba has no support for float16.
# class TestFloat16FFT:
#     def test_1_argument_real(self):
#         x1 = np.array([1, 2, 3, 4], dtype=np.float16)
#         y = fft(x1, n=4)
#         assert_equal(y.dtype, np.complex64)
#         assert_equal(y.shape, (4, ))
#         assert_array_almost_equal(y, direct_dft(x1.astype(np.float32)))

#     def test_n_argument_real(self):
#         x1 = np.array([1, 2, 3, 4], dtype=np.float16)
#         x2 = np.array([1, 2, 3, 4], dtype=np.float16)
#         y = fft([x1, x2], n=4)
#         assert_equal(y.dtype, np.complex64)
#         assert_equal(y.shape, (2, 4))
#         assert_array_almost_equal(y[0], direct_dft(x1.astype(np.float32)))
#         assert_array_almost_equal(y[1], direct_dft(x2.astype(np.float32)))


class _TestIFFTBase:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([1, 2, 3, 4 + 1j, 1, 2, 3, 4 + 2j], self.cdt)
        y = ifft(x)
        y1 = direct_idft(x)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(y, y1)

        x = np.array([1, 2, 3, 4 + 0j, 5], self.cdt)
        assert_array_almost_equal(ifft(x), direct_idft(x))

    def test_definition_real(self):
        x = np.array([1, 2, 3, 4, 1, 2, 3, 4], self.rdt)
        y = ifft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_idft(x)
        assert_array_almost_equal(y, y1)

        x = np.array([1, 2, 3, 4, 5], dtype=self.rdt)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(ifft(x), direct_idft(x))

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2**i
            # Int64 to avoid numerical problems
            x = np.arange(n).astype(np.int64)
            y = ifft(x.astype(self.cdt))
            # We test agains scipy because of numerical reasons.
            # Numpy has different backend.
            y2 = scipy.fft.ifft(x)
            assert_allclose(y, y2, rtol=1e-4, atol=1e-4)
            y = ifft(x)
            assert_allclose(y, y2, rtol=1e-4, atol=1e-4)

    def test_random_complex(self):
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            x = random([size]).astype(self.cdt)
            x = random([size]).astype(self.cdt) + 1j * x
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)

    def test_random_real(self):
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            x = random([size]).astype(self.rdt)
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)

    def test_size_accuracy(self):
        # Sanity check for the accuracy for prime and non-prime sized inputs
        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            np.random.seed(1234)
            x = np.random.rand(size).astype(self.rdt)
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)

            x = (x + 1j * np.random.rand(size)).astype(self.cdt)
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)

    def test_invalid_sizes(self):
        # NOTE: Test modified.
        assert_raises(ValueError, ifft, [])
        assert_raises(NumbaValueError, ifft, np.array([[1, 1], [2, 2]]), -5)


@pytest.mark.skipif(
    np.longdouble is np.float64, reason="Long double is aliased to double"
)
class TestLongDoubleIFFT(_TestIFFTBase):
    def setup_method(self):
        self.cdt = np.longcomplex
        self.rdt = np.longdouble
        self.rtol = 1e-10
        self.atol = 1e-10


class TestDoubleIFFT(_TestIFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double
        self.rtol = 1e-10
        self.atol = 1e-10


class TestSingleIFFT(_TestIFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32
        self.rtol = 1e-5
        self.atol = 1e-4


class Testfft2:
    def setup_method(self):
        np.random.seed(1234)

    def test_regression_244(self):
        """FFT returns wrong result with axes parameter."""
        # fftn (and hence fft2) used to break when both axes and shape were
        # used
        x = numpy.ones((4, 4, 2))
        y = fft2(x, s=(8, 8), axes=(-3, -2))
        y_r = numpy.fft.fftn(x, s=(8, 8), axes=(-3, -2))
        assert_array_almost_equal(y, y_r)

    def test_invalid_sizes(self):
        # Tests modified.
        assert_raises(ValueError, fft2, [[]])
        assert_raises(NumbaValueError, fft2, np.array([[1, 1], [2, 2]]), (4, -3))


class TestFftnSingle:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = fftn(np.array(x, np.float32))
        assert_(
            y.dtype == np.complex64, msg="double precision output with single precision"
        )

        y_r = np.array(fftn(x), np.complex64)
        assert_array_almost_equal_nulp(y, y_r)

    @pytest.mark.parametrize("size", SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    def test_size_accuracy_small(self, size):
        x = np.random.rand(size, size) + 1j * np.random.rand(size, size)
        y1 = fftn(x.real.astype(np.float32))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2000)

    @pytest.mark.parametrize("size", LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    def test_size_accuracy_large(self, size):
        x = np.random.rand(size, 3) + 1j * np.random.rand(size, 3)
        y1 = fftn(x.real.astype(np.float32))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2000)

    # NOTE: Numba has no support for float16
    # def test_definition_float16(self):
    #     x = np.array([[1, 2, 3],
    #                   [4, 5, 6],
    #                   [7, 8, 9]])
    #     y = fftn(np.array(x, np.float16))
    #     assert_equal(y.dtype, np.complex64)
    #     y_r = np.array(fftn(x), np.complex64)
    #     assert_array_almost_equal_nulp(y, y_r)

    # NOTE: Numba has no support for float16
    # @pytest.mark.parametrize("size", SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    # def test_float16_input_small(self, size):
    #     x = np.random.rand(size, size) + 1j*np.random.rand(size, size)
    #     y1 = fftn(x.real.astype(np.float16))
    #     y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

    #     assert_equal(y1.dtype, np.complex64)
    #     assert_array_almost_equal_nulp(y1, y2, 5e5)

    # NOTE: Numba has no support for float16
    # @pytest.mark.parametrize("size", LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    # def test_float16_input_large(self, size):
    #     x = np.random.rand(size, 3) + 1j*np.random.rand(size, 3)
    #     y1 = fftn(x.real.astype(np.float16))
    #     y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

    #     assert_equal(y1.dtype, np.complex64)
    #     assert_array_almost_equal_nulp(y1, y2, 2e6)


class TestFftn:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = fftn(x)
        assert_array_almost_equal(y, direct_dftn(x))

        x = random((20, 26))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

        x = random((5, 4, 3, 20))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

    def test_axes_argument(self):
        # plane == ji_plane, x== kji_space
        plane1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        plane2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        plane3 = np.array([[19, 20, 21], [22, 23, 24], [25, 26, 27]])
        ki_plane1 = np.array([[1, 2, 3], [10, 11, 12], [19, 20, 21]])
        ki_plane2 = np.array([[4, 5, 6], [13, 14, 15], [22, 23, 24]])
        ki_plane3 = np.array([[7, 8, 9], [16, 17, 18], [25, 26, 27]])
        jk_plane1 = np.array([[1, 10, 19], [4, 13, 22], [7, 16, 25]])
        jk_plane2 = np.array([[2, 11, 20], [5, 14, 23], [8, 17, 26]])
        jk_plane3 = np.array([[3, 12, 21], [6, 15, 24], [9, 18, 27]])
        kj_plane1 = np.array([[1, 4, 7], [10, 13, 16], [19, 22, 25]])
        kj_plane2 = np.array([[2, 5, 8], [11, 14, 17], [20, 23, 26]])
        kj_plane3 = np.array([[3, 6, 9], [12, 15, 18], [21, 24, 27]])
        ij_plane1 = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        ij_plane2 = np.array([[10, 13, 16], [11, 14, 17], [12, 15, 18]])
        ij_plane3 = np.array([[19, 22, 25], [20, 23, 26], [21, 24, 27]])
        ik_plane1 = np.array([[1, 10, 19], [2, 11, 20], [3, 12, 21]])
        ik_plane2 = np.array([[4, 13, 22], [5, 14, 23], [6, 15, 24]])
        ik_plane3 = np.array([[7, 16, 25], [8, 17, 26], [9, 18, 27]])
        ijk_space = np.array([jk_plane1, jk_plane2, jk_plane3])
        ikj_space = np.array([kj_plane1, kj_plane2, kj_plane3])
        jik_space = np.array([ik_plane1, ik_plane2, ik_plane3])
        jki_space = np.array([ki_plane1, ki_plane2, ki_plane3])
        kij_space = np.array([ij_plane1, ij_plane2, ij_plane3])
        x = np.array([plane1, plane2, plane3])

        assert_array_almost_equal(fftn(x), fftn(x, axes=(-3, -2, -1)))  # kji_space
        assert_array_almost_equal(fftn(x), fftn(x, axes=(0, 1, 2)))
        assert_array_almost_equal(fftn(x, axes=(0, 2)), fftn(x, axes=(0, -1)))
        y = fftn(x, axes=(2, 1, 0))  # ijk_space
        assert_array_almost_equal(swapaxes(y, -1, -3), fftn(ijk_space))
        y = fftn(x, axes=(2, 0, 1))  # ikj_space
        assert_array_almost_equal(
            swapaxes(swapaxes(y, -1, -3), -1, -2), fftn(ikj_space)
        )
        y = fftn(x, axes=(1, 2, 0))  # jik_space
        assert_array_almost_equal(
            swapaxes(swapaxes(y, -1, -3), -3, -2), fftn(jik_space)
        )
        y = fftn(x, axes=(1, 0, 2))  # jki_space
        assert_array_almost_equal(swapaxes(y, -2, -3), fftn(jki_space))
        y = fftn(x, axes=(0, 2, 1))  # kij_space
        assert_array_almost_equal(swapaxes(y, -2, -1), fftn(kij_space))

        y = fftn(x, axes=(-2, -1))  # ji_plane
        assert_array_almost_equal(fftn(plane1), y[0])
        assert_array_almost_equal(fftn(plane2), y[1])
        assert_array_almost_equal(fftn(plane3), y[2])

        y = fftn(x, axes=(1, 2))  # ji_plane
        assert_array_almost_equal(fftn(plane1), y[0])
        assert_array_almost_equal(fftn(plane2), y[1])
        assert_array_almost_equal(fftn(plane3), y[2])

        y = fftn(x, axes=(-3, -2))  # kj_plane
        assert_array_almost_equal(fftn(x[:, :, 0]), y[:, :, 0])
        assert_array_almost_equal(fftn(x[:, :, 1]), y[:, :, 1])
        assert_array_almost_equal(fftn(x[:, :, 2]), y[:, :, 2])

        y = fftn(x, axes=(-3, -1))  # ki_plane
        assert_array_almost_equal(fftn(x[:, 0, :]), y[:, 0, :])
        assert_array_almost_equal(fftn(x[:, 1, :]), y[:, 1, :])
        assert_array_almost_equal(fftn(x[:, 2, :]), y[:, 2, :])

        y = fftn(x, axes=(-1, -2))  # ij_plane
        assert_array_almost_equal(fftn(ij_plane1), swapaxes(y[0], -2, -1))
        assert_array_almost_equal(fftn(ij_plane2), swapaxes(y[1], -2, -1))
        assert_array_almost_equal(fftn(ij_plane3), swapaxes(y[2], -2, -1))

        y = fftn(x, axes=(-1, -3))  # ik_plane
        assert_array_almost_equal(fftn(ik_plane1), swapaxes(y[:, 0, :], -1, -2))
        assert_array_almost_equal(fftn(ik_plane2), swapaxes(y[:, 1, :], -1, -2))
        assert_array_almost_equal(fftn(ik_plane3), swapaxes(y[:, 2, :], -1, -2))

        y = fftn(x, axes=(-2, -3))  # jk_plane
        assert_array_almost_equal(fftn(jk_plane1), swapaxes(y[:, :, 0], -1, -2))
        assert_array_almost_equal(fftn(jk_plane2), swapaxes(y[:, :, 1], -1, -2))
        assert_array_almost_equal(fftn(jk_plane3), swapaxes(y[:, :, 2], -1, -2))

        y = fftn(x, axes=(-1,))  # i_line
        for i in range(3):
            for j in range(3):
                assert_array_almost_equal(fft(x[i, j, :]), y[i, j, :])
        y = fftn(x, axes=(-2,))  # j_line
        for i in range(3):
            for j in range(3):
                assert_array_almost_equal(fft(x[i, :, j]), y[i, :, j])
        y = fftn(x, axes=(0,))  # k_line
        for i in range(3):
            for j in range(3):
                assert_array_almost_equal(fft(x[:, i, j]), y[:, i, j])

        # NOTE: We don't allows this case
        # y = fftn(x, axes=())  # point
        # assert_array_almost_equal(y, x)
        with assert_raises(TypingError):
            y = fftn(x, axes=())

    def test_shape_argument(self):
        small_x = np.array([[1, 2, 3], [4, 5, 6]])
        large_x1 = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        y = fftn(small_x, s=(4, 4))
        assert_array_almost_equal(y, fftn(large_x1))

        y = fftn(small_x, s=(3, 4))
        assert_array_almost_equal(y, fftn(large_x1[:-1]))

    def test_shape_axes_argument(self):
        small_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        large_x1 = array(
            np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0]])
        )
        y = fftn(small_x, s=(4, 4), axes=(-2, -1))
        assert_array_almost_equal(y, fftn(large_x1))
        y = fftn(small_x, s=(4, 4), axes=(-1, -2))

        assert_array_almost_equal(y, swapaxes(fftn(swapaxes(large_x1, -1, -2)), -1, -2))

    def test_shape_axes_argument2(self):
        # Change shape of the last axis
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-1,), s=(8,))
        assert_array_almost_equal(y, fft(x, axis=-1, n=8))

        # Change shape of an arbitrary axis which is not the last one
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-2,), s=(8,))
        assert_array_almost_equal(y, fft(x, axis=-2, n=8))

        # Change shape of axes: cf #244, where shape and axes were mixed up
        x = numpy.random.random((4, 4, 2))
        y = fftn(x, axes=(-3, -2), s=(8, 8))
        assert_array_almost_equal(y, numpy.fft.fftn(x, axes=(-3, -2), s=(8, 8)))

    def test_shape_argument_more(self):
        x = zeros((4, 4, 2))
        # NOTE: Not specific error message
        with assert_raises(NumbaValueError):
            fftn(x, s=(8, 8, 2, 1))

    def test_invalid_sizes(self):
        # NOTE: We don't raise an error in this case but return an empty array.
        # with assert_raises(ValueError,
        #                    match="invalid number of data points"
        #                    r" \(\[1, 0\]\) specified"):
        #     fftn(np.array([[]]))
        x = np.array([[]])
        assert_array_almost_equal(x, fftn(x))

        # NOTE: No specific error message
        with assert_raises(NumbaValueError):
            fftn(np.array([[1, 1], [2, 2]]), (4, -3))

    def test_no_axes(self):
        # NOTE: We don't allow this case
        x = numpy.random.random((2, 2, 2))
        # assert_allclose(fftn(x, axes=[]), x, atol=1e-7)
        with assert_raises(TypingError):
            y = fftn(x, axes=())


class TestIfftn:
    dtype = None
    cdtype = None

    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize(
        "dtype,cdtype,maxnlp",
        [(np.float64, np.complex128, 2000), (np.float32, np.complex64, 3500)],
    )
    def test_definition(self, dtype, cdtype, maxnlp):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        y = ifftn(x)
        assert_equal(y.dtype, cdtype)
        assert_array_almost_equal_nulp(y, direct_idftn(x), maxnlp)

        x = random((20, 26))
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

        x = random((5, 4, 3, 20))
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

    @pytest.mark.parametrize("maxnlp", [2000, 3500])
    @pytest.mark.parametrize("size", [1, 2, 51, 32, 64, 92])
    def test_random_complex(self, maxnlp, size):
        x = random([size, size]) + 1j * random([size, size])
        assert_array_almost_equal_nulp(ifftn(fftn(x)), x, maxnlp)
        assert_array_almost_equal_nulp(fftn(ifftn(x)), x, maxnlp)

    def test_invalid_sizes(self):
        # NOTE: Test modified
        with assert_raises(ValueError):
            ifftn([[]])

        # NOTE: Note specific error message
        with assert_raises(NumbaValueError):
            ifftn(np.array([[1, 1], [2, 2]]), (4, -3))

    def test_no_axes(self):
        # NOTE: We don't allow this case
        x = numpy.random.random((2, 2, 2))
        # assert_allclose(ifftn(x, axes=[]), x, atol=1e-7)
        with assert_raises(ValueError):
            ifftn(x, axes=[])


# NOTE: We don't test this because almost all of it is not supported by Numba
# class FakeArray:
#     def __init__(self, data):
#         self._data = data
#         self.__array_interface__ = data.__array_interface__


# class FakeArray2:
#     def __init__(self, data):
#         self._data = data

#     def __array__(self):
#         return self._data

# NOTE: We don't consider this one.
# TODO: Is this test actually valuable? The behavior it's testing shouldn't be
# relied upon by users except for overwrite_x = False
# class TestOverwrite:
#     """Check input overwrite behavior of the FFT functions."""

#     real_dtypes = [np.float32, np.float64, np.longfloat]
#     dtypes = real_dtypes + [np.complex64, np.complex128, np.longcomplex]
#     fftsizes = [8, 16, 32]

#     def _check(self, x, routine, fftsize, axis, overwrite_x, should_overwrite):
#         x2 = x.copy()
#         for fake in [lambda x: x, FakeArray, FakeArray2]:
#             routine(fake(x2), fftsize, axis, overwrite_x=overwrite_x)

#             sig = "%s(%s%r, %r, axis=%r, overwrite_x=%r)" % (
#                 routine.__name__, x.dtype, x.shape, fftsize, axis, overwrite_x)
#             if not should_overwrite:
#                 assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)

#     def _check_1d(self, routine, dtype, shape, axis, overwritable_dtypes,
#                   fftsize, overwrite_x):
#         np.random.seed(1234)
#         if np.issubdtype(dtype, np.complexfloating):
#             data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
#         else:
#             data = np.random.randn(*shape)
#         data = data.astype(dtype)

#         should_overwrite = (overwrite_x
#                             and dtype in overwritable_dtypes
#                             and fftsize <= shape[axis])
#         self._check(data, routine, fftsize, axis,
#                     overwrite_x=overwrite_x,
#                     should_overwrite=should_overwrite)

#     @pytest.mark.parametrize("dtype", dtypes)
#     @pytest.mark.parametrize("fftsize", fftsizes)
#     @pytest.mark.parametrize("overwrite_x", [True, False])
#     @pytest.mark.parametrize("shape,axes", [((16,), -1),
#                                             ((16, 2), 0),
#                                             ((2, 16), 1)])
#     def test_fft_ifft(self, dtype, fftsize, overwrite_x, shape, axes):
#         overwritable = (np.longcomplex, np.complex128, np.complex64)
#         self._check_1d(fft, dtype, shape, axes, overwritable,
#                        fftsize, overwrite_x)
#         self._check_1d(ifft, dtype, shape, axes, overwritable,
#                        fftsize, overwrite_x)

#     @pytest.mark.parametrize("dtype", real_dtypes)
#     @pytest.mark.parametrize("fftsize", fftsizes)
#     @pytest.mark.parametrize("overwrite_x", [True, False])
#     @pytest.mark.parametrize("shape,axes", [((16,), -1),
#                                             ((16, 2), 0),
#                                             ((2, 16), 1)])
#     def test_rfft_irfft(self, dtype, fftsize, overwrite_x, shape, axes):
#         overwritable = self.real_dtypes
#         self._check_1d(irfft, dtype, shape, axes, overwritable,
#                        fftsize, overwrite_x)
#         self._check_1d(rfft, dtype, shape, axes, overwritable,
#                        fftsize, overwrite_x)

#     def _check_nd_one(self, routine, dtype, shape, axes, overwritable_dtypes,
#                       overwrite_x):
#         np.random.seed(1234)
#         if np.issubdtype(dtype, np.complexfloating):
#             data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
#         else:
#             data = np.random.randn(*shape)
#         data = data.astype(dtype)

#         def fftshape_iter(shp):
#             if len(shp) <= 0:
#                 yield ()
#             else:
#                 for j in (shp[0]//2, shp[0], shp[0]*2):
#                     for rest in fftshape_iter(shp[1:]):
#                         yield (j,) + rest

#         def part_shape(shape, axes):
#             if axes is None:
#                 return shape
#             else:
#                 return tuple(np.take(shape, axes))

#         def should_overwrite(data, shape, axes):
#             s = part_shape(data.shape, axes)
#             return (overwrite_x and
#                     np.prod(shape) <= np.prod(s)
#                     and dtype in overwritable_dtypes)

#         for fftshape in fftshape_iter(part_shape(shape, axes)):
#             self._check(data, routine, fftshape, axes,
#                         overwrite_x=overwrite_x,
#                         should_overwrite=should_overwrite(data, fftshape, axes))
#             if data.ndim > 1:
#                 # check fortran order
#                 self._check(data.T, routine, fftshape, axes,
#                             overwrite_x=overwrite_x,
#                             should_overwrite=should_overwrite(
#                                 data.T, fftshape, axes))

#     @pytest.mark.parametrize("dtype", dtypes)
#     @pytest.mark.parametrize("overwrite_x", [True, False])
#     @pytest.mark.parametrize("shape,axes", [((16,), None),
#                                             ((16,), (0,)),
#                                             ((16, 2), (0,)),
#                                             ((2, 16), (1,)),
#                                             ((8, 16), None),
#                                             ((8, 16), (0, 1)),
#                                             ((8, 16, 2), (0, 1)),
#                                             ((8, 16, 2), (1, 2)),
#                                             ((8, 16, 2), (0,)),
#                                             ((8, 16, 2), (1,)),
#                                             ((8, 16, 2), (2,)),
#                                             ((8, 16, 2), None),
#                                             ((8, 16, 2), (0, 1, 2))])
#     def test_fftn_ifftn(self, dtype, overwrite_x, shape, axes):
#         overwritable = (np.longcomplex, np.complex128, np.complex64)
#         self._check_nd_one(fftn, dtype, shape, axes, overwritable,
#                            overwrite_x)
#         self._check_nd_one(ifftn, dtype, shape, axes, overwritable,
#                            overwrite_x)


@pytest.mark.parametrize("func", [fft, ifft, fftn, ifftn, rfft, irfft, rfftn, irfftn])
def test_invalid_norm(func):
    x = np.arange(10, dtype=float)
    # NOTE: Test modified; not providing explicit error message.
    with assert_raises(NumbaValueError):
        func(x, norm="o")


# NOTE: Not supported by Numba
# @pytest.mark.parametrize("func", [fft, ifft, fftn, ifftn,
#                                  rfft, irfft, rfftn, irfftn])
# def test_swapped_byte_order_complex(func):
#     rng = np.random.RandomState(1234)
#     x = rng.rand(10) + 1j * rng.rand(10)
# assert_allclose(func(swap_byteorder(x)), func(x))

# NOTE: Not supported by Numba
# @pytest.mark.parametrize("func", [ihfft, ihfftn, rfft, rfftn])
# def test_swapped_byte_order_real(func):
#     rng = np.random.RandomState(1234)
#     x = rng.rand(10)
#     assert_allclose(func(swap_byteorder(x)), func(x))


class _TestRFFTBase:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        for t in [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4, 5]]:
            x = np.array(t, dtype=self.rdt)
            y = rfft(x)
            y1 = direct_rdft(x)
            assert_array_almost_equal(y, y1)
            assert_equal(y.dtype, self.cdt)

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2**i
            # NOTE: We test int64 and against scipy because
            # of numerical discrepancy otherwise
            x = np.arange(n).astype(np.int64)
            y1 = scipy.fft.rfft(x)
            y = rfft(x)
            assert_array_almost_equal(y, y1, decimal=6)

    def test_invalid_sizes(self):
        # NOTE: Tests modified.
        assert_raises(ValueError, rfft, [])
        assert_raises(NumbaValueError, rfft, np.array([[1, 1], [2, 2]]), -5)

    def test_complex_input(self):
        x = np.zeros(10, dtype=self.cdt)
        # NOTE: Only test Error not specific message
        with assert_raises(TypingError):
            rfft(x)

    # NOTE: Numba does not support this
    # See gh-5790
    # class MockSeries:
    #     def __init__(self, data):
    #         self.data = np.asarray(data)

    #     def __getattr__(self, item):
    #         try:
    #             return getattr(self.data, item)
    #         except AttributeError as e:
    #             raise AttributeError((""MockSeries" object "
    #                                   "has no attribute "{attr}"".
    #                                   format(attr=item))) from e

    # def test_non_ndarray_with_dtype(self):
    #     x = np.array([1., 2., 3., 4., 5.])
    #     xs = _TestRFFTBase.MockSeries(x)

    #     expected = [1, 2, 3, 4, 5]
    #     rfft(xs)

    #     # Data should not have been overwritten
    #     assert_equal(x, expected)
    #     assert_equal(xs.data, expected)


@pytest.mark.skipif(
    np.longfloat is np.float64, reason="Long double is aliased to double"
)
class TestRFFTLongDouble(_TestRFFTBase):
    def setup_method(self):
        self.cdt = np.longcomplex
        self.rdt = np.longfloat


class TestRFFTDouble(_TestRFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double


class TestRFFTSingle(_TestRFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32


class _TestIRFFTBase:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x1 = np.array([1, 2 + 3j, 4 + 1j, 1 + 2j, 3 + 4j])
        x1_1 = np.array([1, 2 + 3j, 4 + 1j, 2 + 3j, 4, 2 - 3j, 4 - 1j, 2 - 3j])
        x1 = x1_1[:5]
        x2_1 = np.array(
            [1, 2 + 3j, 4 + 1j, 2 + 3j, 4 + 5j, 4 - 5j, 2 - 3j, 4 - 1j, 2 - 3j]
        )
        x2 = x2_1[:5]

        def _test(x, xr):
            y = irfft(np.array(x, dtype=self.cdt), n=len(xr))
            y1 = direct_irdft(x, len(xr))
            assert_equal(y.dtype, self.rdt)
            assert_array_almost_equal(y, y1, decimal=self.ndec)
            assert_array_almost_equal(y, ifft(xr), decimal=self.ndec)

        _test(x1, x1_1)
        _test(x2, x2_1)

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2**i
            x = np.arange(-1, n, 2) + 1j * np.arange(0, n + 1, 2)
            x[0] = 0
            if n % 2 == 0:
                x[-1] = np.real(x[-1])
            y1 = np.fft.irfft(x)
            y = irfft(x)
            assert_array_almost_equal(y, y1)

    def test_random_real(self):
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            x = random([size]).astype(self.rdt)
            y1 = irfft(rfft(x), n=size)
            y2 = rfft(irfft(x, n=(size * 2 - 1)))
            assert_equal(y1.dtype, self.rdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(
                y1, x, decimal=self.ndec, err_msg="size=%d" % size
            )
            assert_array_almost_equal(
                y2, x, decimal=self.ndec, err_msg="size=%d" % size
            )

    def test_size_accuracy(self):
        # Sanity check for the accuracy for prime and non-prime sized inputs
        if self.rdt == np.float32:
            rtol = 1e-5
        elif self.rdt == np.float64:
            rtol = 1e-10

        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            np.random.seed(1234)
            x = np.random.rand(size).astype(self.rdt)
            y = irfft(rfft(x), len(x))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            y = rfft(irfft(x, 2 * len(x) - 1))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)

    def test_invalid_sizes(self):
        # NOTE: Modified test
        assert_raises(ValueError, irfft, [])
        assert_raises(NumbaValueError, irfft, np.array([[1, 1], [2, 2]]), -5)


# self.ndec is bogus; we should have a assert_array_approx_equal for number of
# significant digits
@pytest.mark.skipif(
    np.longfloat is np.float64, reason="Long double is aliased to double"
)
class TestIRFFTLongDouble(_TestIRFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double
        self.ndec = 14


class TestIRFFTDouble(_TestIRFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double
        self.ndec = 14


class TestIRFFTSingle(_TestIRFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32
        self.ndec = 5


MDATA_COUNT = 8
FFTWDATA_COUNT = 14


# NOTE: No longdouble for us
def is_longdouble_binary_compatible():
    return False
    # try:
    #     one = np.frombuffer(
    #         b"\x00\x00\x00\x00\x00\x00\x00\x80\xff\x3f\x00\x00\x00\x00\x00\x00",
    #         dtype="<f16")
    #     return one == np.longfloat(1.)
    # except TypeError:
    #     return False


def get_reference_data():
    ref = getattr(globals(), "__reference_data", None)
    if ref is not None:
        return ref

    # Matlab reference data
    MDATA = np.load(join(fftpack_test_dir, "test.npz"))
    X = [MDATA["x%d" % i] for i in range(MDATA_COUNT)]
    Y = [MDATA["y%d" % i] for i in range(MDATA_COUNT)]

    # FFTW reference data: the data are organized as follows:
    #    * SIZES is an array containing all available sizes
    #    * for every type (1, 2, 3, 4) and every size, the array dct_type_size
    #    contains the output of the DCT applied to the input np.linspace(0, size-1,
    #    size)
    FFTWDATA_DOUBLE = np.load(join(fftpack_test_dir, "fftw_double_ref.npz"))
    FFTWDATA_SINGLE = np.load(join(fftpack_test_dir, "fftw_single_ref.npz"))
    FFTWDATA_SIZES = FFTWDATA_DOUBLE["sizes"]
    assert len(FFTWDATA_SIZES) == FFTWDATA_COUNT

    if is_longdouble_binary_compatible():
        FFTWDATA_LONGDOUBLE = np.load(join(fftpack_test_dir, "fftw_longdouble_ref.npz"))
    else:
        FFTWDATA_LONGDOUBLE = {
            k: v.astype(np.longfloat) for k, v in FFTWDATA_DOUBLE.items()
        }

    ref = {
        "FFTWDATA_LONGDOUBLE": FFTWDATA_LONGDOUBLE,
        "FFTWDATA_DOUBLE": FFTWDATA_DOUBLE,
        "FFTWDATA_SINGLE": FFTWDATA_SINGLE,
        "FFTWDATA_SIZES": FFTWDATA_SIZES,
        "X": X,
        "Y": Y,
    }

    globals()["__reference_data"] = ref
    return ref


@pytest.fixture(params=range(FFTWDATA_COUNT))
def fftwdata_size(request):
    return get_reference_data()["FFTWDATA_SIZES"][request.param]


@pytest.fixture(params=range(MDATA_COUNT))
def mdata_x(request):
    return get_reference_data()["X"][request.param]


@pytest.fixture(params=range(MDATA_COUNT))
def mdata_xy(request):
    ref = get_reference_data()
    y = ref["Y"][request.param]
    x = ref["X"][request.param]
    return x, y


def fftw_dct_ref(type, size, dt):
    x = np.linspace(0, size - 1, size).astype(dt)
    dt = np.result_type(np.float32, dt)
    if dt == np.double:
        data = get_reference_data()["FFTWDATA_DOUBLE"]
    elif dt == np.float32:
        data = get_reference_data()["FFTWDATA_SINGLE"]
    elif dt == np.longfloat:
        data = get_reference_data()["FFTWDATA_LONGDOUBLE"]
    else:
        raise ValueError()
    y = (data["dct_%d_%d" % (type, size)]).astype(dt)
    return x, y, dt


def fftw_dst_ref(type, size, dt):
    x = np.linspace(0, size - 1, size).astype(dt)
    dt = np.result_type(np.float32, dt)
    if dt == np.double:
        data = get_reference_data()["FFTWDATA_DOUBLE"]
    elif dt == np.float32:
        data = get_reference_data()["FFTWDATA_SINGLE"]
    elif dt == np.longfloat:
        data = get_reference_data()["FFTWDATA_LONGDOUBLE"]
    else:
        raise ValueError()
    y = (data["dst_%d_%d" % (type, size)]).astype(dt)
    return x, y, dt


def ref_2d(func, x, **kwargs):
    """Calculate 2-D reference data from a 1d transform"""
    x = np.array(x, copy=True)
    for row in range(x.shape[0]):
        x[row, :] = func(x[row, :], **kwargs)
    for col in range(x.shape[1]):
        x[:, col] = func(x[:, col], **kwargs)
    return x


def naive_dct1(x, norm=None):
    """Calculate textbook definition version of DCT-I."""
    x = np.array(x, copy=True)
    N = len(x)
    M = N - 1
    y = np.zeros(N)
    m0, m = 1, 2
    if norm == "ortho":
        m0 = np.sqrt(1.0 / M)
        m = np.sqrt(2.0 / M)
    for k in range(N):
        for n in range(1, N - 1):
            y[k] += m * x[n] * np.cos(np.pi * n * k / M)
        y[k] += m0 * x[0]
        y[k] += m0 * x[N - 1] * (1 if k % 2 == 0 else -1)
    if norm == "ortho":
        y[0] *= 1 / np.sqrt(2)
        y[N - 1] *= 1 / np.sqrt(2)
    return y


def naive_dst1(x, norm=None):
    """Calculate textbook definition version of DST-I."""
    x = np.array(x, copy=True)
    N = len(x)
    M = N + 1
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += 2 * x[n] * np.sin(np.pi * (n + 1.0) * (k + 1.0) / M)
    if norm == "ortho":
        y *= np.sqrt(0.5 / M)
    return y


def naive_dct4(x, norm=None):
    """Calculate textbook definition version of DCT-IV."""
    x = np.array(x, copy=True)
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n] * np.cos(np.pi * (n + 0.5) * (k + 0.5) / (N))
    if norm == "ortho":
        y *= np.sqrt(2.0 / N)
    else:
        y *= 2
    return y


def naive_dst4(x, norm=None):
    """Calculate textbook definition version of DST-IV."""
    x = np.array(x, copy=True)
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n] * np.sin(np.pi * (n + 0.5) * (k + 0.5) / (N))
    if norm == "ortho":
        y *= np.sqrt(2.0 / N)
    else:
        y *= 2
    return y


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.longcomplex])
@pytest.mark.parametrize("transform", [dct, dst, idct, idst])
def test_complex(transform, dtype):
    y = transform(1j * np.arange(5, dtype=dtype))
    x = 1j * transform(np.arange(5))
    assert_array_almost_equal(x, y)


DecMapType = Dict[
    Tuple[Callable[..., np.ndarray], Union[Type[np.floating], Type[int]], int],
    int,
]


# map (tranform, dtype, type) -> decimal
dec_map: DecMapType = {
    # DCT
    (dct, np.double, 1): 13,
    (dct, np.float32, 1): 6,
    (dct, np.double, 2): 14,
    (dct, np.float32, 2): 5,
    (dct, np.double, 3): 14,
    (dct, np.float32, 3): 5,
    (dct, np.double, 4): 13,
    (dct, np.float32, 4): 6,
    # IDCT
    (idct, np.double, 1): 14,
    (idct, np.float32, 1): 6,
    (idct, np.double, 2): 14,
    (idct, np.float32, 2): 5,
    (idct, np.double, 3): 14,
    (idct, np.float32, 3): 5,
    (idct, np.double, 4): 14,
    (idct, np.float32, 4): 6,
    # DST
    (dst, np.double, 1): 13,
    (dst, np.float32, 1): 6,
    (dst, np.double, 2): 14,
    (dst, np.float32, 2): 6,
    (dst, np.double, 3): 14,
    (dst, np.float32, 3): 7,
    (dst, np.double, 4): 13,
    (dst, np.float32, 4): 6,
    # IDST
    (idst, np.double, 1): 14,
    (idst, np.float32, 1): 6,
    (idst, np.double, 2): 14,
    (idst, np.float32, 2): 6,
    (idst, np.double, 3): 14,
    (idst, np.float32, 3): 6,
    (idst, np.double, 4): 14,
    (idst, np.float32, 4): 6,
}

for k, v in dec_map.copy().items():
    if k[1] == np.double:
        dec_map[(k[0], np.longdouble, k[2])] = v
    elif k[1] == np.float32:
        dec_map[(k[0], int, k[2])] = v


@pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
class TestDCT:
    def test_definition(self, rdt, type, fftwdata_size):
        x, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt)
        y = dct(x, type=type)
        # NOTE: We use different casting rule
        # assert_equal(y.dtype, dt)
        dec = dec_map[(dct, rdt, type)]
        assert_allclose(y, yr, rtol=0.0, atol=np.max(yr) * 10 ** (-dec))

    @pytest.mark.parametrize("size", [7, 8, 9, 16, 32, 64])
    def test_axis(self, rdt, type, size):
        nt = 2
        dec = dec_map[(dct, rdt, type)]
        x = np.random.randn(nt, size)
        y = dct(x, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[j], dct(x[j], type=type), decimal=dec)

        x = x.T
        y = dct(x, axis=0, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[:, j], dct(x[:, j], type=type), decimal=dec)


# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# def test_dct1_definition_ortho(rdt, mdata_x):
#     dct = scipy.fft.dct
#     # Test orthornomal mode.
#     dec = dec_map[(dct, rdt, 1)]
#     x = np.array(mdata_x, dtype=rdt)
#     dt = np.result_type(np.float32, rdt)
#     y = dct(x, norm="ortho", type=1)
#     y2 = naive_dct1(x, norm="ortho")
#     assert_equal(y.dtype, dt)
#     assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))


# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# def test_dct2_definition_matlab(mdata_xy, rdt):
#     dct = scipy.fft.dct
#     # Test correspondence with matlab (orthornomal mode).
#     dt = np.result_type(np.float32, rdt)
#     x = np.array(mdata_xy[0], dtype=dt)
#     yr = mdata_xy[1]
#     y = dct(x, norm="ortho", type=2)
#     dec = dec_map[(dct, rdt, 2)]
#     assert_equal(y.dtype, dt)
#     assert_array_almost_equal(y, yr, decimal=dec)


# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# def test_dct3_definition_ortho(mdata_x, rdt):
#     dct = scipy.fft.dct
#     # Test orthornomal mode.
#     x = np.array(mdata_x, dtype=rdt)
#     dt = np.result_type(np.float32, rdt)
#     y = dct(x, norm="ortho", type=2)
#     xi = dct(y, norm="ortho", type=3)
#     dec = dec_map[(dct, rdt, 3)]
#     assert_equal(xi.dtype, dt)
#     assert_array_almost_equal(xi, x, decimal=dec)

# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# def test_dct4_definition_ortho(mdata_x, rdt):
#     dct = scipy.fft.dct
#     # Test orthornomal mode.
#     x = np.array(mdata_x, dtype=rdt)
#     dt = np.result_type(np.float32, rdt)
#     y = dct(x, norm="ortho", type=4)
#     y2 = naive_dct4(x, norm="ortho")
#     dec = dec_map[(dct, rdt, 4)]
#     assert_equal(y.dtype, dt)
#     assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))

# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# @pytest.mark.parametrize("type", [1, 2, 3, 4])
# def test_idct_definition(fftwdata_size, rdt, type):
#     idct = scipy.fft.idct
#     xr, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt)
#     x = idct(yr, type=type)
#     dec = dec_map[(idct, rdt, type)]
#     assert_equal(x.dtype, dt)
#     assert_allclose(x, xr, rtol=0., atol=np.max(xr)*10**(-dec))


# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# @pytest.mark.parametrize("type", [1, 2, 3, 4])
# def test_definition(fftwdata_size, rdt, type):
#     dst = scipy.fft.dst
#     xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt)
#     y = dst(xr, type=type)
#     dec = dec_map[(dst, rdt, type)]
#     assert_equal(y.dtype, dt)
#     assert_allclose(y, yr, rtol=0., atol=np.max(yr)*10**(-dec))


# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# def test_dst1_definition_ortho(rdt, mdata_x):
#     dst = scipy.fft.dst
#     # Test orthornomal mode.
#     dec = dec_map[(dst, rdt, 1)]
#     x = np.array(mdata_x, dtype=rdt)
#     dt = np.result_type(np.float32, rdt)
#     y = dst(x, norm="ortho", type=1)
#     y2 = naive_dst1(x, norm="ortho")
#     assert_equal(y.dtype, dt)
#     assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))


# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# def test_dst4_definition_ortho(rdt, mdata_x):
#     dst = scipy.fft.dst
#     # Test orthornomal mode.
#     dec = dec_map[(dst, rdt, 4)]
#     x = np.array(mdata_x, dtype=rdt)
#     dt = np.result_type(np.float32, rdt)
#     y = dst(x, norm="ortho", type=4)
#     y2 = naive_dst4(x, norm="ortho")
#     assert_equal(y.dtype, dt)
#     assert_array_almost_equal(y, y2, decimal=dec)


# TODO: scipy.fft.dct does also not pass these tests!?
# @pytest.mark.parametrize("rdt", [np.longfloat, np.double, np.float32, int])
# @pytest.mark.parametrize("type", [1, 2, 3, 4])
# def test_idst_definition(fftwdata_size, rdt, type):
#     idst = scipy.fft.idct
#     xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt)
#     x = idst(yr, type=type)
#     dec = dec_map[(idst, rdt, type)]
#     assert_equal(x.dtype, dt)
#     assert_allclose(x, xr, rtol=0., atol=np.max(xr)*10**(-dec))


@pytest.mark.parametrize("routine", [dct, dst, idct, idst])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longfloat])
@pytest.mark.parametrize("shape, axis", [((16,), -1), ((16, 2), 0), ((2, 16), 1)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("overwrite_x", [True, False])
@pytest.mark.parametrize("norm", [None, "ortho"])
def test_overwrite(routine, dtype, shape, axis, type, norm, overwrite_x):
    # Check input overwrite behavior
    np.random.seed(1234)
    if np.issubdtype(dtype, np.complexfloating):
        x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    else:
        x = np.random.randn(*shape)
    x = x.astype(dtype)
    x2 = x.copy()
    routine(x2, type, None, axis, norm, overwrite_x=overwrite_x)

    sig = "%s(%s%r, %r, axis=%r, overwrite_x=%r)" % (
        routine.__name__,
        x.dtype,
        x.shape,
        None,
        axis,
        overwrite_x,
    )
    if not overwrite_x:
        assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)


class Test_DCTN_IDCTN:
    dec = 14
    dct_type = [1, 2, 3, 4]
    norms = [None, "backward", "ortho", "forward"]
    rstate = np.random.RandomState(1234)
    shape = (32, 16)
    data = rstate.randn(*shape)

    @pytest.mark.parametrize("fforward,finverse", [(dctn, idctn), (dstn, idstn)])
    @pytest.mark.parametrize(
        "axes",
        [
            None,
            1,
            (1,),
            np.array([1]),
            0,
            (0,),
            np.array([0]),
            (0, 1),
            np.array([0, 1]),
            (-2, -1),
            np.array([-2, -1]),
        ],
    )
    @pytest.mark.parametrize("dct_type", dct_type)
    @pytest.mark.parametrize("norm", ["ortho"])
    def test_axes_round_trip(self, fforward, finverse, axes, dct_type, norm):
        tmp = fforward(self.data, type=dct_type, axes=axes, norm=norm)
        tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)
        assert_array_almost_equal(self.data, tmp, decimal=12)

    @pytest.mark.parametrize("funcn,func", [(dctn, dct), (dstn, dst)])
    @pytest.mark.parametrize("dct_type", dct_type)
    @pytest.mark.parametrize("norm", norms)
    def test_dctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        y1 = funcn(self.data, type=dct_type, axes=None, norm=norm)
        y2 = ref_2d(func, self.data, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize("funcn,func", [(idctn, idct), (idstn, idst)])
    @pytest.mark.parametrize("dct_type", dct_type)
    @pytest.mark.parametrize("norm", norms)
    def test_idctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        fdata = dctn(self.data, type=dct_type, norm=norm)
        y1 = funcn(fdata, type=dct_type, norm=norm)
        y2 = ref_2d(func, fdata, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize("fforward,finverse", [(dctn, idctn), (dstn, idstn)])
    def test_axes_and_shape(self, fforward, finverse):
        # NOTE: Test without explicit error message
        # s is passed as tuple instead of inteter
        with assert_raises(NumbaValueError):
            fforward(self.data, s=(self.data.shape[0],), axes=(0, 1))
        # NOTE: Test without explicit error message
        # axes is passed as tuple instead of inteter
        with assert_raises(NumbaValueError):
            fforward(self.data, s=self.data.shape, axes=(0,))

    @pytest.mark.parametrize("fforward", [dctn, dstn])
    def test_shape(self, fforward):
        tmp = fforward(self.data, s=(128, 128), axes=None)
        assert_equal(tmp.shape, (128, 128))

    # NOTE: Removed tests where axes is a list and where axis is an inteter
    @pytest.mark.parametrize("fforward,finverse", [(dctn, idctn), (dstn, idstn)])
    @pytest.mark.parametrize(
        "axes",
        [
            (1,),
            (0,),
        ],
    )
    def test_shape_is_none_with_axes(self, fforward, finverse, axes):
        tmp = fforward(self.data, s=None, axes=axes, norm="ortho")
        tmp = finverse(tmp, s=None, axes=axes, norm="ortho")
        assert_array_almost_equal(self.data, tmp, decimal=self.dec)


# NOTE: Not supported by Numba
# @pytest.mark.parametrize("func", [dct, dctn, idct, idctn,
#                                   dst, dstn, idst, idstn])
# def test_swapped_byte_order(func):
#     rng = np.random.RandomState(1234)
#     x = rng.rand(10)
#     swapped_dt = x.dtype.newbyteorder("S")
#     assert_allclose(func(x.astype(swapped_dt)), func(x))
