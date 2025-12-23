"""
These tests are borrowed from Numpy.
Thanks to all who contributed to these tests.
https://github.com/numpy/numpy/blob/main/numpy/fft/tests/test_helper.py
https://github.com/numpy/numpy/blob/main/numpy/fft/tests/test_pocketfft.py
Whenever I changed a test, I left a note.
"""

from functools import partial

import numba as nb
import numpy as np
import pytest
from helpers import numba_cache_cleanup, set_numba_capture_errors_new_style
from numba.core.errors import TypingError
from numpy import pi
from numpy.random import random
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_raises

set_numba_capture_errors_new_style()

# At maximum double precision is supported
np.longcomplex = np.complex128
np.longdouble = np.float64
np.longfloat = np.float64

# All functions should be cacheable and run without the GIL
njit = partial(nb.njit, cache=True, nogil=True)


@njit
def fft(a, n=None, axis=-1, norm=None):
    return np.fft.fft(a, n, axis, norm)


@njit
def fft2(a, s=None, axes=(-2, -1), norm=None):
    return np.fft.fft2(a, s, axes, norm)


@njit
def fftn(a, s=None, axes=None, norm=None):
    return np.fft.fftn(a, s, axes, norm)


@njit
def ifft(a, n=None, axis=-1, norm=None):
    return np.fft.ifft(a, n, axis, norm)


@njit
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    return np.fft.ifft2(a, s, axes, norm)


@njit
def ifftn(a, s=None, axes=None, norm=None):
    return np.fft.ifftn(a, s, axes, norm)


@njit
def rfft(a, n=None, axis=-1, norm=None):
    return np.fft.rfft(a, n, axis, norm)


@njit
def rfft2(a, s=None, axes=(-2, -1), norm=None):
    return np.fft.rfft2(a, s, axes, norm)


@njit
def rfftn(a, s=None, axes=None, norm=None):
    return np.fft.rfftn(a, s, axes, norm)


@njit
def irfft(a, n=None, axis=-1, norm=None):
    return np.fft.irfft(a, n, axis, norm)


@njit
def irfft2(a, s=None, axes=(-2, -1), norm=None):
    return np.fft.irfft2(a, s, axes, norm)


@njit
def irfftn(a, s=None, axes=None, norm=None):
    return np.fft.irfftn(a, s, axes, norm)


@njit
def hfft(a, n=None, axis=-1, norm=None):
    return np.fft.hfft(a, n, axis, norm)


@njit
def ihfft(a, n=None, axis=-1, norm=None):
    return np.fft.ihfft(a, n, axis, norm)


@njit
def fftshift(x, axes=None):
    return np.fft.fftshift(x, axes)


@njit
def ifftshift(x, axes=None):
    return np.fft.ifftshift(x, axes)


@njit
def fftfreq(n, d=1.0):
    return np.fft.fftfreq(n, d)


@njit
def rfftfreq(n, d=1.0):
    return np.fft.rfftfreq(n, d)


def fft1(x):
    L = len(x)
    phase = -2j * np.pi * (np.arange(L) / L)
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x * np.exp(phase), axis=1)


class TestFFTShift:

    def test_fft_n(self):
        # NOTE: Test modified
        assert_raises(TypingError, fft, [1, 2, 3], 0)


class TestFFT1D:

    def test_identity(self):
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        xr = random(maxlen)
        for i in range(1, maxlen):
            assert_allclose(ifft(fft(x[0:i])), x[0:i], atol=1e-12)
            assert_allclose(irfft(rfft(xr[0:i]), i), xr[0:i], atol=1e-12)

    def test_fft(self):
        x = random(30) + 1j * random(30)
        assert_allclose(fft1(x), fft(x), atol=1e-6)
        assert_allclose(fft1(x), fft(x, norm="backward"), atol=1e-6)
        assert_allclose(fft1(x) / np.sqrt(30), fft(x, norm="ortho"), atol=1e-6)
        assert_allclose(fft1(x) / 30.0, fft(x, norm="forward"), atol=1e-6)

    @pytest.mark.parametrize("norm", (None, "backward", "ortho", "forward"))
    def test_ifft(self, norm):
        x = random(30) + 1j * random(30)
        assert_allclose(x, ifft(fft(x, norm=norm), norm=norm), atol=1e-6)
        # NOTE: We test without explicit error message
        # Ensure we get the correct error message
        with pytest.raises(ValueError):
            ifft([], norm=norm)

    def test_fft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(fft(fft(x, axis=1), axis=0), fft2(x), atol=1e-6)
        assert_allclose(fft2(x), fft2(x, norm="backward"), atol=1e-6)
        assert_allclose(fft2(x) / np.sqrt(30 * 20), fft2(x, norm="ortho"), atol=1e-6)
        assert_allclose(fft2(x) / (30.0 * 20.0), fft2(x, norm="forward"), atol=1e-6)

    def test_ifft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(ifft(ifft(x, axis=1), axis=0), ifft2(x), atol=1e-6)
        assert_allclose(ifft2(x), ifft2(x, norm="backward"), atol=1e-6)
        assert_allclose(ifft2(x) * np.sqrt(30 * 20), ifft2(x, norm="ortho"), atol=1e-6)
        assert_allclose(ifft2(x) * (30.0 * 20.0), ifft2(x, norm="forward"), atol=1e-6)

    def test_fftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(fft(fft(fft(x, axis=2), axis=1), axis=0), fftn(x), atol=1e-6)
        assert_allclose(fftn(x), fftn(x, norm="backward"), atol=1e-6)
        assert_allclose(
            fftn(x) / np.sqrt(30 * 20 * 10), fftn(x, norm="ortho"), atol=1e-6
        )
        assert_allclose(
            fftn(x) / (30.0 * 20.0 * 10.0), fftn(x, norm="forward"), atol=1e-6
        )

    def test_ifftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(
            ifft(ifft(ifft(x, axis=2), axis=1), axis=0), ifftn(x), atol=1e-6
        )
        assert_allclose(ifftn(x), ifftn(x, norm="backward"), atol=1e-6)
        assert_allclose(
            ifftn(x) * np.sqrt(30 * 20 * 10), ifftn(x, norm="ortho"), atol=1e-6
        )
        assert_allclose(
            ifftn(x) * (30.0 * 20.0 * 10.0), ifftn(x, norm="forward"), atol=1e-6
        )

    def test_rfft(self):
        x = random(30)
        for n in [x.size, 2 * x.size]:
            for norm in [None, "backward", "ortho", "forward"]:
                assert_allclose(
                    fft(x, n=n, norm=norm)[: (n // 2 + 1)],
                    rfft(x, n=n, norm=norm),
                    atol=1e-6,
                )
            assert_allclose(rfft(x, n=n), rfft(x, n=n, norm="backward"), atol=1e-6)
            assert_allclose(
                rfft(x, n=n) / np.sqrt(n), rfft(x, n=n, norm="ortho"), atol=1e-6
            )
            assert_allclose(rfft(x, n=n) / n, rfft(x, n=n, norm="forward"), atol=1e-6)

    def test_irfft(self):
        x = random(30)
        assert_allclose(x, irfft(rfft(x)), atol=1e-6)
        assert_allclose(x, irfft(rfft(x, norm="backward"), norm="backward"), atol=1e-6)
        assert_allclose(x, irfft(rfft(x, norm="ortho"), norm="ortho"), atol=1e-6)
        assert_allclose(x, irfft(rfft(x, norm="forward"), norm="forward"), atol=1e-6)

    def test_rfft2(self):
        x = random((30, 20))
        assert_allclose(fft2(x)[:, :11], rfft2(x), atol=1e-6)
        assert_allclose(rfft2(x), rfft2(x, norm="backward"), atol=1e-6)
        assert_allclose(rfft2(x) / np.sqrt(30 * 20), rfft2(x, norm="ortho"), atol=1e-6)
        assert_allclose(rfft2(x) / (30.0 * 20.0), rfft2(x, norm="forward"), atol=1e-6)

    def test_irfft2(self):
        x = random((30, 20))
        assert_allclose(x, irfft2(rfft2(x)), atol=1e-6)
        assert_allclose(
            x, irfft2(rfft2(x, norm="backward"), norm="backward"), atol=1e-6
        )
        assert_allclose(x, irfft2(rfft2(x, norm="ortho"), norm="ortho"), atol=1e-6)
        assert_allclose(x, irfft2(rfft2(x, norm="forward"), norm="forward"), atol=1e-6)

    def test_rfftn(self):
        x = random((30, 20, 10))
        assert_allclose(fftn(x)[:, :, :6], rfftn(x), atol=1e-6)
        assert_allclose(rfftn(x), rfftn(x, norm="backward"), atol=1e-6)
        assert_allclose(
            rfftn(x) / np.sqrt(30 * 20 * 10), rfftn(x, norm="ortho"), atol=1e-6
        )
        assert_allclose(
            rfftn(x) / (30.0 * 20.0 * 10.0), rfftn(x, norm="forward"), atol=1e-6
        )

    def test_irfftn(self):
        x = random((30, 20, 10))
        assert_allclose(x, irfftn(rfftn(x)), atol=1e-6)
        assert_allclose(
            x, irfftn(rfftn(x, norm="backward"), norm="backward"), atol=1e-6
        )
        assert_allclose(x, irfftn(rfftn(x, norm="ortho"), norm="ortho"), atol=1e-6)
        assert_allclose(x, irfftn(rfftn(x, norm="forward"), norm="forward"), atol=1e-6)

    def test_hfft(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        assert_allclose(fft(x), hfft(x_herm), atol=1e-6)
        assert_allclose(hfft(x_herm), hfft(x_herm, norm="backward"), atol=1e-6)
        assert_allclose(
            hfft(x_herm) / np.sqrt(30), hfft(x_herm, norm="ortho"), atol=1e-6
        )
        assert_allclose(hfft(x_herm) / 30.0, hfft(x_herm, norm="forward"), atol=1e-6)

    def test_ihfft(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        assert_allclose(x_herm, ihfft(hfft(x_herm)), atol=1e-6)
        assert_allclose(
            x_herm, ihfft(hfft(x_herm, norm="backward"), norm="backward"), atol=1e-6
        )
        assert_allclose(
            x_herm, ihfft(hfft(x_herm, norm="ortho"), norm="ortho"), atol=1e-6
        )
        assert_allclose(
            x_herm, ihfft(hfft(x_herm, norm="forward"), norm="forward"), atol=1e-6
        )

    # TODO: Fails
    @pytest.mark.parametrize("op", [fftn, ifftn, rfftn, irfftn])
    def test_axes(self, op):
        x = random((30, 20, 10))
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for a in axes:
            op_tr = op(np.transpose(x, a))
            tr_op = np.transpose(op(x, axes=a), a)
            assert_allclose(op_tr, tr_op, atol=1e-6)

    def test_all_1d_norm_preserving(self):
        # verify that round-trip transforms are norm-preserving
        x = random(30)
        x_norm = np.linalg.norm(x)
        n = x.size * 2
        func_pairs = [
            (fft, ifft),
            (rfft, irfft),
            # hfft: order so the first function takes x.size samples
            #       (necessary for comparison to x_norm above)
            (ihfft, hfft),
        ]
        for forw, back in func_pairs:
            for n in [x.size, 2 * x.size]:
                for norm in [None, "backward", "ortho", "forward"]:
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    assert_allclose(x_norm, np.linalg.norm(tmp), atol=1e-6)

    # NOTE: No support for np.half -> removed from list
    @pytest.mark.parametrize("dtype", [np.single, np.double, np.longdouble])
    def test_dtypes(self, dtype):
        # make sure that all input precisions are accepted and internally
        # converted to 64bit
        x = random(30).astype(dtype)
        assert_allclose(ifft(fft(x)), x, atol=1e-6)
        assert_allclose(irfft(rfft(x)), x, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("order", ["F", "non-contiguous"])
@pytest.mark.parametrize("fft", [fft, fft2, fftn, ifft, ifft2, ifftn])
def test_fft_with_order(dtype, order, fft):
    # Check that FFT/IFFT produces identical results for C, Fortran and
    # non contiguous arrays
    rng = np.random.RandomState(42)
    X = rng.rand(8, 7, 13).astype(dtype, copy=False)
    # See discussion in pull/14178
    _tol = 8.0 * np.sqrt(np.log2(X.size)) * np.finfo(X.dtype).eps
    if order == "F":
        Y = np.asfortranarray(X)
    else:
        # Make a non contiguous array
        Y = X[::-1]
        X = np.ascontiguousarray(X[::-1])

    if fft.__name__.endswith("fft"):
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    elif fft.__name__.endswith(("fft2", "fftn")):
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith("fftn"):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    else:
        raise ValueError()


# NOTE: We skip this
# @pytest.mark.skipif(IS_WASM, reason="Cannot start thread")
# class TestFFTThreadSafe:
#     threads = 16
#     input_shape = (800, 200)

#     def _test_mtsame(self, func, *args):
#         def worker(args, q):
#             q.put(func(*args))

#         q = queue.Queue()
#         expected = func(*args)

#         # Spin off a bunch of threads to call the same function simultaneously
#         t = [threading.Thread(target=worker, args=(args, q))
#              for i in range(self.threads)]
#         [x.start() for x in t]

#         [x.join() for x in t]
#         # Make sure all threads returned the correct value
#         for i in range(self.threads):
#             assert_array_equal(q.get(timeout=5), expected,
#                                "Function returned wrong value in multithreaded context")

#     def test_fft(self):
#         a = np.ones(self.input_shape) * 1+0j
#         self._test_mtsame(fft, a)

#     def test_ifft(self):
#         a = np.ones(self.input_shape) * 1+0j
#         self._test_mtsame(ifft, a)

#     def test_rfft(self):
#         a = np.ones(self.input_shape)
#         self._test_mtsame(rfft, a)

#     def test_irfft(self):
#         a = np.ones(self.input_shape) * 1+0j
#         self._test_mtsame(irfft, a)


class TestFFTShift:

    def test_definition(self):
        x = np.array([0, 1, 2, 3, 4, -4, -3, -2, -1])
        y = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        assert_array_almost_equal(fftshift(x), y)
        assert_array_almost_equal(ifftshift(y), x)
        x = np.array([0, 1, 2, 3, 4, -5, -4, -3, -2, -1])
        y = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
        assert_array_almost_equal(fftshift(x), y)
        assert_array_almost_equal(ifftshift(y), x)

    def test_inverse(self):
        for n in [1, 4, 9, 100, 211]:
            x = np.random.random((n,))
            assert_array_almost_equal(ifftshift(fftshift(x)), x)

    def test_axes_keyword(self):
        freqs = np.array([[0, 1, 2], [3, 4, -4], [-3, -2, -1]])
        shifted = np.array([[-1, -3, -2], [2, 0, 1], [-4, 3, 4]])
        assert_array_almost_equal(fftshift(freqs, axes=(0, 1)), shifted)
        assert_array_almost_equal(fftshift(freqs, axes=0), fftshift(freqs, axes=(0,)))
        assert_array_almost_equal(ifftshift(shifted, axes=(0, 1)), freqs)
        assert_array_almost_equal(
            ifftshift(shifted, axes=0), ifftshift(shifted, axes=(0,))
        )

        assert_array_almost_equal(fftshift(freqs), shifted)
        assert_array_almost_equal(ifftshift(shifted), freqs)

    # NOTE: When axes were passes as list it has been changed to tuple
    def test_uneven_dims(self):
        """Test 2D input, which has uneven dimension sizes"""
        freqs = np.array([[0, 1], [2, 3], [4, 5]])

        # shift in dimension 0
        shift_dim0 = np.array([[4, 5], [0, 1], [2, 3]])
        assert_array_almost_equal(fftshift(freqs, axes=0), shift_dim0)
        assert_array_almost_equal(ifftshift(shift_dim0, axes=0), freqs)
        assert_array_almost_equal(fftshift(freqs, axes=(0,)), shift_dim0)
        assert_array_almost_equal(ifftshift(shift_dim0, axes=(0,)), freqs)

        # shift in dimension 1
        shift_dim1 = np.array([[1, 0], [3, 2], [5, 4]])
        assert_array_almost_equal(fftshift(freqs, axes=1), shift_dim1)
        assert_array_almost_equal(ifftshift(shift_dim1, axes=1), freqs)

        # shift in both dimensions
        shift_dim_both = np.array([[5, 4], [1, 0], [3, 2]])
        assert_array_almost_equal(fftshift(freqs, axes=(0, 1)), shift_dim_both)
        assert_array_almost_equal(ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        assert_array_almost_equal(fftshift(freqs, axes=(0, 1)), shift_dim_both)
        assert_array_almost_equal(ifftshift(shift_dim_both, axes=(0, 1)), freqs)

        # axes=None (default) shift in all dimensions
        assert_array_almost_equal(fftshift(freqs, axes=None), shift_dim_both)
        assert_array_almost_equal(ifftshift(shift_dim_both, axes=None), freqs)
        assert_array_almost_equal(fftshift(freqs), shift_dim_both)
        assert_array_almost_equal(ifftshift(shift_dim_both), freqs)

    def test_equal_to_original(self):
        """Test that the new (>=v1.15) implementation (see #10073) is equal to the original (<=v1.14)"""
        from numpy._core import arange, asarray, concatenate, take

        def original_fftshift(x, axes=None):
            """How fftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        def original_ifftshift(x, axes=None):
            """How ifftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = n - (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        # create possible 2d array combinations and try all possible keywords
        # compare output to original functions
        for i in range(16):
            for j in range(16):
                for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                    inp = np.random.rand(i, j)
                    assert_array_almost_equal(
                        fftshift(inp, axes_keyword),
                        original_fftshift(inp, axes_keyword),
                    )

                    assert_array_almost_equal(
                        ifftshift(inp, axes_keyword),
                        original_ifftshift(inp, axes_keyword),
                    )


class TestFFTFreq:

    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        assert_array_almost_equal(9 * fftfreq(9), x)
        assert_array_almost_equal(9 * pi * fftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        assert_array_almost_equal(10 * fftfreq(10), x)
        assert_array_almost_equal(10 * pi * fftfreq(10, pi), x)


class TestRFFTFreq:

    def test_definition(self):
        x = [0, 1, 2, 3, 4]
        assert_array_almost_equal(9 * rfftfreq(9), x)
        assert_array_almost_equal(9 * pi * rfftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, 5]
        assert_array_almost_equal(10 * rfftfreq(10), x)
        assert_array_almost_equal(10 * pi * rfftfreq(10, pi), x)


class TestIRFFTN:

    def test_not_last_axis_success(self):
        ar, ai = np.random.random((2, 16, 8, 32))
        a = ar + 1j * ai

        axes = (-2,)

        # Should not raise error
        irfftn(a, axes=axes)
