import numba as nb
import numpy as np
import scipy.fft
import pytest
from numba.core.errors import TypingError, NumbaValueError
from pytest import raises as assert_raises
from helpers import set_numba_capture_errors_new_style, NumpyFFT, ScipyFFT

set_numba_capture_errors_new_style()


def mk_match(pos, name):
    lut = {
        0: "1st",
        1: "2nd",
        2: "3rd",
        3: "4th",
        4: "5th",
        5: "6th",
        6: "7th",
        7: "8th",
    }
    pos = lut.get(pos)
    return rf".* {pos} .* '{name}' .*"


class TestFFTShift:
    x1 = np.random.rand(42)
    x2 = np.random.rand(42, 42)

    def test_target(self):
        with assert_raises(TypingError, match=mk_match(0, "x")):
            NumpyFFT.fftshift(list(self.x1), 0)
        with assert_raises(TypingError, match=mk_match(0, "x")):
            NumpyFFT.fftshift(tuple(self.x2), 1)
        with assert_raises(TypingError, match=mk_match(0, "x")):
            NumpyFFT.fftshift(1, 0)
        with assert_raises(TypingError, match=mk_match(0, "x")):
            NumpyFFT.fftshift(None, 0)
        NumpyFFT.fftshift(self.x2, 1)
        NumpyFFT.fftshift(self.x2, (1,))
        NumpyFFT.fftshift(self.x2, (0, 1))

    def test_axes(self):
        with assert_raises(TypingError, match=mk_match(1, "axes")):
            NumpyFFT.fftshift(self.x1, 0.0)
        with assert_raises(TypingError, match=mk_match(1, "axes")):
            NumpyFFT.fftshift(self.x2, False)
        NumpyFFT.fftshift(self.x1, 0)
        NumpyFFT.fftshift(self.x2, (0,))
        NumpyFFT.fftshift(self.x2, (0, 1))
        NumpyFFT.fftshift(self.x2, None)


class TestFFTFreq:
    def test_target(self):
        with assert_raises(TypingError, match=mk_match(0, "n")):
            NumpyFFT.fftfreq(1.0, 2.0)
        with assert_raises(TypingError, match=mk_match(0, "n")):
            NumpyFFT.fftfreq(None, 2.0)
        with assert_raises(TypingError, match=mk_match(0, "n")):
            NumpyFFT.fftfreq((1,), 2.0)
        NumpyFFT.fftfreq(1, 1.0)

    def test_real(self):
        with assert_raises(TypingError, match=mk_match(1, "d")):
            NumpyFFT.fftfreq(2, None)
        with assert_raises(TypingError, match=mk_match(1, "d")):
            NumpyFFT.fftfreq(2, True)
        with assert_raises(TypingError, match=mk_match(1, "d")):
            NumpyFFT.fftfreq(2, (1,))
        NumpyFFT.fftfreq(1, 1)
        NumpyFFT.fftfreq(1, 1.0)
        NumpyFFT.fftfreq(1, 1j)

    def test_device(self):
        with assert_raises(ValueError):
            NumpyFFT.fftfreq(1, device="gpu")
        NumpyFFT.rfftfreq(1, device=None)
        NumpyFFT.rfftfreq(1, device="cpu")

    def test_xp(self):
        with assert_raises(TypingError):
            ScipyFFT.fftfreq(1, xp=nb)
            ScipyFFT.rfftfreq(1, xp=np)
        ScipyFFT.rfftfreq(1, xp=None)

        # Within jitted code passing numpy should work
        with assert_raises(TypingError):

            @(lambda f: f())
            @nb.njit
            def _():
                return scipy.fft.fftfreq(1, xp=nb)

        @(lambda f: f())
        @nb.njit
        def _():
            return scipy.fft.fftfreq(1, xp=np)


class TestNextFastLen:
    def test_target(self):
        with assert_raises(TypingError, match=mk_match(0, "target")):
            ScipyFFT.next_fast_len(1.0, real=True)
        with assert_raises(TypingError, match=mk_match(0, "target")):
            ScipyFFT.next_fast_len(None, real=True)
        with assert_raises(TypingError, match=mk_match(0, "target")):
            ScipyFFT.next_fast_len((1,), real=True)
        ScipyFFT.next_fast_len(1, real=True)

    def test_real(self):
        with assert_raises(TypingError, match=mk_match(1, "real")):
            ScipyFFT.next_fast_len(1, real=None)
        with assert_raises(TypingError, match=mk_match(1, "real")):
            ScipyFFT.next_fast_len(1, real=(True,))
        with assert_raises(TypingError, match=mk_match(1, "real")):
            ScipyFFT.next_fast_len(1, real=1)
        ScipyFFT.next_fast_len(1, real=False)


class Test1D:
    x = np.random.rand(42)

    @pytest.mark.parametrize("func", [ScipyFFT.fft, ScipyFFT.dct])
    def test_x(self, func):
        with assert_raises(TypingError, match=mk_match(0, "x")):
            func(list(self.x))
        with assert_raises(TypingError, match=mk_match(0, "x")):
            func(tuple(self.x))
        with assert_raises(TypeError):
            func(a=self.x)
        func(self.x)

    @pytest.mark.parametrize("func", [NumpyFFT.fft])
    def test_a(self, func):
        with assert_raises(TypingError, match=mk_match(0, "a")):
            func(list(self.x))
        with assert_raises(TypingError, match=mk_match(0, "a")):
            func(tuple(self.x))
        with assert_raises(TypeError):
            func(x=self.x)
        func(self.x)

    @pytest.mark.parametrize("func", [NumpyFFT.fft, ScipyFFT.fft])
    def test_n_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(1, "n")):
            func(self.x, n=(1.0,))
        with assert_raises(TypingError, match=mk_match(1, "n")):
            func(self.x, n=[1.0])
        with assert_raises(TypingError, match=mk_match(1, "n")):
            func(self.x, n=1.0)
        func(self.x, n=1)
        func(self.x, n=(1,))
        func(self.x, n=None)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_type(self, func):
        with assert_raises(TypingError, match=mk_match(1, "type")):
            func(self.x, type=(1.0,))
        with assert_raises(TypingError, match=mk_match(1, "type")):
            func(self.x, type=[1.0])
        with assert_raises(TypingError, match=mk_match(1, "type")):
            func(self.x, type=1.0)
        func(self.x, type=1)
        func(self.x, type=2)
        func(self.x, type=3)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_n_real(self, func):
        with assert_raises(TypingError, match=mk_match(2, "n")):
            func(self.x, n=(1.0,))
        with assert_raises(TypingError, match=mk_match(2, "n")):
            func(self.x, n=[1.0])
        with assert_raises(TypingError, match=mk_match(2, "n")):
            func(self.x, n=1.0)
        func(self.x, n=1)
        func(self.x, n=(1,))
        func(self.x, n=None)

    @pytest.mark.parametrize("func", [NumpyFFT.fft, ScipyFFT.fft])
    def test_axis_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(2, "axis")):
            func(self.x, axis=(-1.0,))
        with assert_raises(TypingError, match=mk_match(2, "axis")):
            func(self.x, axis=[-1.0])
        with assert_raises(TypingError, match=mk_match(2, "axis")):
            func(self.x, axis=-1.0)
        func(self.x, axis=-1)
        func(self.x, axis=(-1,))
        func(self.x, axis=None)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_axis_real(self, func):
        with assert_raises(TypingError, match=mk_match(3, "axis")):
            func(self.x, axis=(-1.0,))
        with assert_raises(TypingError, match=mk_match(3, "axis")):
            func(self.x, axis=[-1.0])
        with assert_raises(TypingError, match=mk_match(3, "axis")):
            func(self.x, axis=-1.0)
        func(self.x, axis=-1)
        func(self.x, axis=(-1,))
        func(self.x, axis=None)

    @pytest.mark.parametrize("func", [NumpyFFT.fft, ScipyFFT.fft])
    def test_norm_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(3, "norm")):
            func(self.x, norm=0)
        func(self.x, norm=None)
        func(self.x, norm="ortho")

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_norm_real(self, func):
        with assert_raises(TypingError, match=mk_match(4, "norm")):
            func(self.x, norm=0)
        func(self.x, norm=None)
        func(self.x, norm="ortho")

    @pytest.mark.parametrize("func", [ScipyFFT.fft])
    def test_overwrite_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, overwrite_x=0)
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, overwrite_x=None)
        func(self.x, overwrite_x=True)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_overwrite_real(self, func):
        with assert_raises(TypingError, match=mk_match(5, "overwrite_x")):
            func(self.x, overwrite_x=0)
        with assert_raises(TypingError, match=mk_match(5, "overwrite_x")):
            func(self.x, overwrite_x=None)
        func(self.x, overwrite_x=True)

    @pytest.mark.parametrize("func", [ScipyFFT.fft])
    def test_workers_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(5, "workers")):
            func(self.x, workers=1.0)
        func(self.x, workers=None)
        func(self.x, workers=1)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_workers_real(self, func):
        with assert_raises(TypingError, match=mk_match(6, "workers")):
            func(self.x, workers=1.0)
        func(self.x, workers=None)
        func(self.x, workers=1)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_orthogonalize(self, func):
        with assert_raises(TypingError, match=mk_match(7, "orthogonalize")):
            func(self.x, orthogonalize=1.0)
        with assert_raises(TypingError, match=mk_match(7, "orthogonalize")):
            func(self.x, orthogonalize=(False,))
        func(self.x, orthogonalize=None)
        func(self.x, orthogonalize=True)

    @pytest.mark.parametrize("func", [NumpyFFT.fft])
    def test_out(self, func):
        with assert_raises(TypingError, match=mk_match(4, "out")):
            func(self.x, out=self.x)
        func(self.x, out=None)


class Test2D:
    x = np.random.rand(42, 42)

    @pytest.mark.parametrize("func", [ScipyFFT.fft2])
    def test_x(self, func):
        with assert_raises(TypingError, match=mk_match(0, "x")):
            func(list(self.x))
        with assert_raises(TypingError, match=mk_match(0, "x")):
            func(tuple(self.x))
        with assert_raises(TypeError):
            func(a=self.x)
        func(self.x)

    @pytest.mark.parametrize("func", [NumpyFFT.fft2])
    def test_a(self, func):
        with assert_raises(TypingError, match=mk_match(0, "a")):
            func(list(self.x))
        with assert_raises(TypingError, match=mk_match(0, "a")):
            func(tuple(self.x))
        func(self.x)

    @pytest.mark.parametrize("func", [NumpyFFT.fft2, ScipyFFT.fft2])
    def test_s(self, func):
        with assert_raises(TypingError, match=mk_match(1, "s")):
            func(self.x, s=(1.0,))
        with assert_raises(TypingError, match=mk_match(1, "s")):
            func(self.x, s=[1.0])
        with assert_raises(TypingError, match=mk_match(1, "s")):
            func(self.x, s=1.0)
        with assert_raises(NumbaValueError):
            func(self.x, 1)
        with assert_raises(NumbaValueError):
            func(self.x, s=(1,))
        func(self.x, s=(2, 3))
        func(self.x, None)

    @pytest.mark.parametrize("func", [NumpyFFT.fft2, ScipyFFT.fft2])
    def test_axes(self, func):
        with assert_raises(TypingError, match=mk_match(2, "axes")):
            func(self.x, axes=(-1.0,))
        with assert_raises(TypingError, match=mk_match(2, "axes")):
            func(self.x, axes=[-1.0])
        with assert_raises(TypingError, match=mk_match(2, "axes")):
            func(self.x, axes=-1.0)
        func(self.x, axes=-1)
        func(self.x, axes=(-1,))
        func(self.x, None, None)

    @pytest.mark.parametrize("func", [NumpyFFT.fft2, ScipyFFT.fft2])
    def test_norm(self, func):
        with assert_raises(TypingError, match=mk_match(3, "norm")):
            func(self.x, norm=0)
        with assert_raises(TypingError, match=mk_match(3, "norm")):
            func(self.x, norm=0)
        with assert_raises(TypingError, match=mk_match(3, "norm")):
            func(self.x, None, (-2, -1), norm=0)
        func(self.x, None, (-2, -1), None)
        func(self.x, norm="ortho")

    @pytest.mark.parametrize("func", [ScipyFFT.fft2])
    def test_overwrite(self, func):
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, overwrite_x=0)
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, overwrite_x=None)
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, None, (-2, -1), None, None)
        func(self.x, overwrite_x=True)
        func(self.x, None, (-2, -1), None, True)

    @pytest.mark.parametrize("func", [ScipyFFT.fft2])
    def test_workers(self, func):
        with assert_raises(TypingError, match=mk_match(5, "workers")):
            func(self.x, workers=1.0)
        with assert_raises(TypingError, match=mk_match(5, "workers")):
            func(self.x, None, (-2, -1), None, True, 1.0)
        func(self.x, workers=None)
        func(self.x, workers=1)
        func(self.x, None, (-2, -1), None, True, 4)


class TestND:
    x = np.random.rand(7, 7, 7)

    @pytest.mark.parametrize("func", [ScipyFFT.fftn, ScipyFFT.dctn])
    def test_x(self, func):
        with assert_raises(TypingError, match=mk_match(0, "x")):
            func(list(self.x))
        with assert_raises(TypingError, match=mk_match(0, "x")):
            func(tuple(self.x))
        with assert_raises(TypeError):
            func(a=self.x)
        func(self.x)

    @pytest.mark.parametrize("func", [NumpyFFT.fftn])
    def test_a(self, func):
        with assert_raises(TypingError, match=mk_match(0, "a")):
            func(list(self.x))
        with assert_raises(TypingError, match=mk_match(0, "a")):
            func(tuple(self.x))
        func(self.x)

    @pytest.mark.parametrize("func", [NumpyFFT.fftn, ScipyFFT.fftn])
    def test_s_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(1, "s")):
            func(self.x, s=(1.0,))
        with assert_raises(TypingError, match=mk_match(1, "s")):
            func(self.x, s=[1.0])
        with assert_raises(TypingError, match=mk_match(1, "s")):
            func(self.x, s=1.0)
        with assert_raises(NumbaValueError):
            func(self.x, 0)
        with assert_raises(NumbaValueError):
            func(self.x, s=(0,))
        func(self.x, s=(2, 3, 7))
        func(self.x, None)

    @pytest.mark.parametrize("func", [ScipyFFT.dctn])
    def test_type(self, func):
        with assert_raises(TypingError, match=mk_match(1, "type")):
            func(self.x, type=(1.0,))
        with assert_raises(TypingError, match=mk_match(1, "type")):
            func(self.x, [1.0])
        with assert_raises(TypingError, match=mk_match(1, "type")):
            func(self.x, type=1.0)
        func(self.x, type=1)
        func(self.x, 2)
        func(self.x, type=3)

    @pytest.mark.parametrize("func", [ScipyFFT.dctn])
    def test_s_real(self, func):
        with assert_raises(TypingError, match=mk_match(2, "s")):
            func(self.x, s=(1.0,))
        with assert_raises(TypingError, match=mk_match(2, "s")):
            func(self.x, s=[1.0])
        with assert_raises(TypingError, match=mk_match(2, "s")):
            func(self.x, s=1.0)
        with assert_raises(NumbaValueError):
            func(self.x, 1, 0)
        with assert_raises(NumbaValueError):
            func(self.x, s=(0,))
        func(self.x, s=(2, 3, 7))
        func(self.x, 1, None)

    @pytest.mark.parametrize("func", [NumpyFFT.fftn, ScipyFFT.fftn])
    def test_axes_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(2, "axes")):
            func(self.x, axes=(-1.0,))
        with assert_raises(TypingError, match=mk_match(2, "axes")):
            func(self.x, axes=[-1.0])
        with assert_raises(TypingError, match=mk_match(2, "axes")):
            func(self.x, None, -1.0)
        func(self.x, axes=-1)
        func(self.x, axes=(0, 1, 2))
        func(self.x, None, None)

    @pytest.mark.parametrize("func", [ScipyFFT.dctn])
    def test_axes_real(self, func):
        with assert_raises(TypingError, match=mk_match(3, "axes")):
            func(self.x, axes=(-1.0,))
        with assert_raises(TypingError, match=mk_match(3, "axes")):
            func(self.x, axes=[-1.0])
        with assert_raises(TypingError, match=mk_match(3, "axes")):
            func(self.x, 2, None, axes=-1.0)
        func(self.x, axes=-1)
        func(self.x, axes=(0, 1, 2))
        func(self.x, 2, None, None)

    @pytest.mark.parametrize("func", [NumpyFFT.fftn, ScipyFFT.fftn])
    def test_norm_real(self, func):
        with assert_raises(TypingError, match=mk_match(3, "norm")):
            func(self.x, norm=0)
        with assert_raises(TypingError, match=mk_match(3, "norm")):
            func(self.x, norm=0)
        with assert_raises(TypingError, match=mk_match(3, "norm")):
            func(self.x, None, (-2, -1), norm=0)
        with assert_raises(NumbaValueError):
            func(self.x, None, (0, 1, 2, 3), norm="ortho")
        func(self.x, None, (0, 1, 2), None)
        func(self.x, norm="ortho")

    @pytest.mark.parametrize("func", [ScipyFFT.dctn])
    def test_norm_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(4, "norm")):
            func(self.x, norm=0)
        with assert_raises(TypingError, match=mk_match(4, "norm")):
            func(self.x, norm=0)
        with assert_raises(TypingError, match=mk_match(4, "norm")):
            func(self.x, 2, None, (-2, -1), norm=0)
        with assert_raises(NumbaValueError):
            func(self.x, 2, None, (0, 1, 2, 3), norm="ortho")
        func(self.x, 2, None, (0, 1, 2), None)
        func(self.x, norm="ortho")

    @pytest.mark.parametrize("func", [ScipyFFT.fftn])
    def test_overwrite_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, overwrite_x=0)
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, overwrite_x=None)
        with assert_raises(TypingError, match=mk_match(4, "overwrite_x")):
            func(self.x, None, (-2, -1), None, None)
        with assert_raises(NumbaValueError):
            func(self.x, None, (-4, -2, -1), None, False)
        func(self.x, overwrite_x=True)
        func(self.x, None, (-3, -2, -1), None, True)

    @pytest.mark.parametrize("func", [ScipyFFT.dctn])
    def test_overwrite_real(self, func):
        with assert_raises(TypingError, match=mk_match(5, "overwrite_x")):
            func(self.x, overwrite_x=0)
        with assert_raises(TypingError, match=mk_match(5, "overwrite_x")):
            func(self.x, overwrite_x=None)
        with assert_raises(TypingError, match=mk_match(5, "overwrite_x")):
            func(self.x, 2, None, (-2, -1), None, None)
        with assert_raises(NumbaValueError):
            func(self.x, 2, (-4, -2, -1), None, None)
        func(self.x, overwrite_x=True)
        func(self.x, 2, None, (-3, -2, -1), None, True)

    @pytest.mark.parametrize("func", [ScipyFFT.fftn])
    def test_workers_real(self, func):
        with assert_raises(TypingError, match=mk_match(5, "workers")):
            func(self.x, workers=1.0)
        with assert_raises(TypingError, match=mk_match(5, "workers")):
            func(self.x, None, (-2, -1), None, True, 1.0)
        func(self.x, workers=None)
        func(self.x, workers=1)
        func(self.x, None, (0, 1, 2), None, True, 4)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_workers_cmplx(self, func):
        with assert_raises(TypingError, match=mk_match(6, "workers")):
            func(self.x, workers=1.0)
        with assert_raises(TypingError, match=mk_match(6, "workers")):
            func(self.x, 2, None, (-2, -1), None, True, 1.0)
        func(self.x, workers=None)
        func(self.x, workers=1)
        func(self.x, 1, None, (0, 1, 2), None, True, 4)

    @pytest.mark.parametrize("func", [ScipyFFT.dct])
    def test_orthogonalize(self, func):
        with assert_raises(TypingError, match=mk_match(7, "orthogonalize")):
            func(self.x, orthogonalize=1.0)
        with assert_raises(TypingError, match=mk_match(7, "orthogonalize")):
            func(self.x, orthogonalize=[False])
        with assert_raises(TypingError, match=mk_match(7, "orthogonalize")):
            func(self.x, 1, None, (0, 1, 2), None, True, 4, (False,))
        func(self.x, orthogonalize=None)
        func(self.x, 1, None, (0, 1, 2), None, True, 4, orthogonalize=True)


class TestFht:
    x = np.random.rand(7)

    def test_a(self):
        with assert_raises(TypingError, match=mk_match(0, "a")):
            ScipyFFT.fht(list(self.x), 1.0, 1.0)
        with assert_raises(TypingError, match=mk_match(0, "a")):
            ScipyFFT.fht(tuple(self.x), 1.0, 1.0)
        with assert_raises(TypeError):
            ScipyFFT.fht(A=self.x, dln=1.0, mu=1.0)
        with assert_raises(TypingError):
            ScipyFFT.fht(self.x.astype(np.complex128), 1.0, 1.0)
        ScipyFFT.fht(self.x, 1.0, 1.0)

    def test_A(self):
        with assert_raises(TypingError, match=mk_match(0, "A")):
            ScipyFFT.ifht(list(self.x), 1.0, 1.0)
        with assert_raises(TypingError, match=mk_match(0, "A")):
            ScipyFFT.ifht(tuple(self.x), 1.0, 1.0)
        with assert_raises(TypeError):
            ScipyFFT.ifht(a=self.x, dln=1.0, mu=1.0)
        with assert_raises(TypingError):
            ScipyFFT.ifht(self.x.astype(np.complex128), 1.0, 1.0)

    def test_dln(self):
        for func in (ScipyFFT.fht, ScipyFFT.ifht):
            with assert_raises(TypingError, match=mk_match(1, "dln")):
                func(self.x, [1.0], 1.0)
            with assert_raises(TypingError, match=mk_match(1, "dln")):
                func(self.x, "1.0", 1.0)
            with assert_raises(TypingError, match=mk_match(1, "dln")):
                func(self.x, True, 1.0)

    def test_mu(self):
        for func in (ScipyFFT.fht, ScipyFFT.ifht):
            with assert_raises(TypingError, match=mk_match(2, "mu")):
                func(self.x, 1.0, [1.0])
            with assert_raises(TypingError, match=mk_match(2, "mu")):
                func(self.x, 1.0, "1.0")
            with assert_raises(TypingError, match=mk_match(2, "mu")):
                func(self.x, 1.0, True)

    def test_offset(self):
        for func in (ScipyFFT.fht, ScipyFFT.ifht):
            with assert_raises(TypingError, match=mk_match(3, "offset")):
                func(self.x, 1.0, 1.0, [1.0])
            with assert_raises(TypingError, match=mk_match(3, "offset")):
                func(self.x, 1.0, 1.0, "1.0")
            with assert_raises(TypingError, match=mk_match(3, "offset")):
                func(self.x, 1.0, 1.0, True)

    def test_bias(self):
        for func in (ScipyFFT.fht, ScipyFFT.ifht):
            with assert_raises(TypingError, match=mk_match(4, "bias")):
                func(self.x, 1.0, 1.0, 1.0, [1.0])
            with assert_raises(TypingError, match=mk_match(4, "bias")):
                func(self.x, 1.0, 1.0, 1.0, "1.0")
            with assert_raises(TypingError, match=mk_match(4, "bias")):
                func(self.x, 1.0, 1.0, 1.0, True)


class TestFhtoffset:
    def test_dln(self):
        with assert_raises(TypingError, match=mk_match(0, "dln")):
            ScipyFFT.fhtoffset([1.0], 1.0)
        with assert_raises(TypingError, match=mk_match(0, "dln")):
            ScipyFFT.fhtoffset("1.0", 1.0)
        with assert_raises(TypingError, match=mk_match(0, "dln")):
            ScipyFFT.fhtoffset(True, 1.0)

    def test_mu(self):
        with assert_raises(TypingError, match=mk_match(1, "mu")):
            ScipyFFT.fhtoffset(1.0, [1.0])
        with assert_raises(TypingError, match=mk_match(1, "mu")):
            ScipyFFT.fhtoffset(1.0, "1.0")
        with assert_raises(TypingError, match=mk_match(1, "mu")):
            ScipyFFT.fhtoffset(1.0, True)

    def test_initial(self):
        with assert_raises(TypingError, match=mk_match(2, "initial")):
            ScipyFFT.fhtoffset(1.0, 1.0, [1.0])
        with assert_raises(TypingError, match=mk_match(2, "initial")):
            ScipyFFT.fhtoffset(1.0, 1.0, "1.0")
        with assert_raises(TypingError, match=mk_match(2, "initial")):
            ScipyFFT.fhtoffset(1.0, 1.0, True)

    def test_bias(self):
        with assert_raises(TypingError, match=mk_match(3, "bias")):
            ScipyFFT.fhtoffset(1.0, 1.0, 1.0, [1.0])
        with assert_raises(TypingError, match=mk_match(3, "bias")):
            ScipyFFT.fhtoffset(1.0, 1.0, 1.0, "1.0")
        with assert_raises(TypingError, match=mk_match(3, "bias")):
            ScipyFFT.fhtoffset(1.0, 1.0, 1.0, True)
