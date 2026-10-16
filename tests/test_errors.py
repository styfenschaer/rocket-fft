"""Invalid arguments are rejected like NumPy and SciPy reject them.

Values that are only known at run time raise the same exception as the library
(mostly ValueError). Arguments of the wrong type are rejected at compile time
with a TypingError that names the offending argument.
"""

import numba as nb
import numpy as np
import pytest
import scipy.fft
from common import NumpyFFT, ScipyFFT, random_array
from numba.core.errors import TypingError
from pytest import raises as assert_raises

import rocket_fft


def assert_both_raise(exc, name, *args, lib="scipy", **kwargs):
    jit, ref = (ScipyFFT, scipy.fft) if lib == "scipy" else (NumpyFFT, np.fft)
    with pytest.raises(exc):
        getattr(ref, name)(*args, **kwargs)
    with pytest.raises(exc):
        getattr(jit, name)(*args, **kwargs)


# -----------------------------------------------------------------------------
# Invalid values
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("lib", ["numpy", "scipy"])
@pytest.mark.parametrize("n", [0, -1, -5])
@pytest.mark.parametrize("name", ["fft", "ifft", "rfft", "irfft"])
def test_invalid_n(name, lib, n):
    if (name, lib, n) == ("irfft", "scipy", 0):
        pytest.xfail("known difference: scipy.fft.irfft(x, n=0) returns one element")
    assert_both_raise(ValueError, name, random_array(16), n, lib=lib)


@pytest.mark.xfail(strict=True, reason="known difference, kept deliberately: rocket-fft rejects n=0")
def test_scipy_irfft_zero_length():
    x = random_array(16, np.complex128)
    assert ScipyFFT.irfft(x, 0).shape == scipy.fft.irfft(x, 0).shape


@pytest.mark.parametrize("lib", ["numpy", "scipy"])
@pytest.mark.parametrize("name", ["fft", "ifft", "fftn", "ifftn", "rfft", "irfft", "rfftn", "irfftn"])
def test_invalid_norm(lib, name):
    assert_both_raise(ValueError, name, random_array(10), norm="o", lib=lib)


@pytest.mark.parametrize("name", ["dct", "dst", "idct", "idst", "dctn", "dstn"])
@pytest.mark.parametrize("type", [0, 5, -1])
def test_invalid_r2r_type(name, type):
    assert_both_raise(ValueError, name, random_array(16), type=type)


@pytest.mark.parametrize("axis", [1, 5, -2, -5])
def test_axis_out_of_bounds(axis):
    # NumPy and SciPy happen to raise IndexError; rocket-fft raises ValueError,
    # like for every other invalid value only known at run time.
    x = random_array(16)
    for jit, ref in [(NumpyFFT, np.fft), (ScipyFFT, scipy.fft)]:
        with pytest.raises(IndexError):
            ref.fft(x, axis=axis)
        with pytest.raises(ValueError):
            jit.fft(x, axis=axis)


# NumPy deprecates s without axes, but this case is still a ValueError either way.
@pytest.mark.filterwarnings("ignore:`axes` should not be `None`:DeprecationWarning")
@pytest.mark.parametrize("lib", ["numpy", "scipy"])
@pytest.mark.parametrize("name", ["fft2", "fftn", "ifftn", "rfftn"])
def test_invalid_s(lib, name):
    x = random_array((2, 2))
    assert_both_raise(ValueError, name, x, (4, -3), (0, 1), lib=lib)
    assert_both_raise(ValueError, name, x, (4, 0), (0, 1), lib=lib)
    if name != "fft2":  # more lengths than dimensions
        assert_both_raise(ValueError, name, random_array((4, 4, 2)), s=(8, 8, 2, 1), lib=lib)


@pytest.mark.parametrize("name", ["dctn", "dstn"])
def test_r2r_s_and_axes_mismatch(name):
    x = random_array((32, 16))
    assert_both_raise(ValueError, name, x, s=(32,), axes=(0, 1))
    assert_both_raise(ValueError, name, x, s=(32, 16), axes=(0,))


@pytest.mark.parametrize("workers", [0, -(10**6)])
def test_invalid_workers(workers):
    assert_both_raise(ValueError, "fft", random_array(16), workers=workers)
    assert_both_raise(ValueError, "dct", random_array(16), workers=workers)


@pytest.mark.parametrize("workers", [0, -1, 10**6])
def test_invalid_set_workers(workers):
    with pytest.raises(ValueError):
        rocket_fft.set_workers(workers)


@pytest.mark.parametrize("name", ["next_fast_len", "prev_fast_len"])
def test_invalid_fast_len(name):
    if not hasattr(scipy.fft, name):
        pytest.skip(f"scipy.fft.{name} not available")
    assert_both_raise(ValueError, name, -1)


@pytest.mark.parametrize("lib", ["numpy", "scipy"])
def test_empty_transform_axis(lib):
    # A zero-length transform axis is an error, an empty batch dimension is not.
    assert_both_raise(ValueError, "fft", np.array([], dtype=np.complex128), lib=lib)
    assert_both_raise(ValueError, "ifft", np.array([], dtype=np.complex128), lib=lib)
    assert_both_raise(ValueError, "rfft", np.array([], dtype=np.float64), lib=lib)
    assert_both_raise(ValueError, "irfft", np.array([], dtype=np.complex128), lib=lib)
    assert_both_raise(ValueError, "fft2", np.empty((0, 5), dtype=np.complex128), lib=lib)
    assert_both_raise(ValueError, "fftn", np.empty((1, 0), dtype=np.complex128), lib=lib)
    assert_both_raise(ValueError, "rfft2", np.empty((4, 0), dtype=np.float64), lib=lib)


def test_empty_batch():
    x = np.empty((0, 8), dtype=np.complex128)
    assert NumpyFFT.fft(x).shape == np.fft.fft(x).shape
    assert ScipyFFT.fft(x).shape == scipy.fft.fft(x).shape
    assert ScipyFFT.dct(x.real).shape == scipy.fft.dct(x.real).shape
    assert NumpyFFT.rfft(x.real).shape == np.fft.rfft(x.real).shape


@pytest.mark.parametrize("name", ["fft", "rfft"])
def test_out_mismatch(name):
    x = random_array(16, np.complex128 if name == "fft" else np.float64)
    ref_shape = np.fft.fft(x).shape if name == "fft" else (9,)
    for out in [np.zeros(ref_shape[0] - 1, np.complex128), np.zeros(ref_shape, np.float64)]:
        with pytest.raises((ValueError, TypeError)):
            getattr(np.fft, name)(x, out=out)
        with pytest.raises((ValueError, TypingError)):
            getattr(NumpyFFT, name)(x, out=out)


# -----------------------------------------------------------------------------
# Invalid types
# -----------------------------------------------------------------------------


def test_fft_list_input():
    with assert_raises(TypingError):
        NumpyFFT.fft([1, 2, 3], 0)


@pytest.mark.parametrize("name", ["rfft", "ihfft", "rfftn", "fht", "ifht"])
def test_complex_input_to_real_transform(name):
    x = random_array(16, np.complex128)
    args = (0.1, 0.5) if name in ("fht", "ifht") else ()
    with assert_raises(TypingError):
        getattr(ScipyFFT, name)(x, *args)
    if name in ("rfft", "rfftn"):
        with assert_raises(TypingError):
            getattr(ScipyFFT, name)(x, overwrite_x=True)


def test_too_many_axes():
    # NumPy allows repeated axes, but per-call bookkeeping is capped at 64 entries.
    x = random_array((2, 2), np.complex128)
    with pytest.raises(ValueError, match="at most 64"):
        NumpyFFT.fftn(x, axes=(0,) * 65)
    np.testing.assert_allclose(NumpyFFT.fftn(x, axes=(0, 1) * 32), np.fft.fftn(x, axes=(0, 1) * 32))


@pytest.mark.parametrize("name", ["fftn", "ifftn", "dctn"])
def test_empty_axes(name):
    # SciPy treats axes=() as the identity; rocket-fft rejects it at compile time.
    with assert_raises(TypingError):
        getattr(ScipyFFT, name)(random_array((2, 2, 2)), axes=())


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


@pytest.mark.skipif(
    not hasattr(scipy.fft, "prev_fast_len"),
    reason="scipy.fft.prev_fast_len requires SciPy >= 1.15",
)
class TestPrevFastLen:
    def test_target(self):
        with assert_raises(TypingError, match=mk_match(0, "target")):
            ScipyFFT.prev_fast_len(1.0, real=True)
        with assert_raises(TypingError, match=mk_match(0, "target")):
            ScipyFFT.prev_fast_len(None, real=True)
        with assert_raises(TypingError, match=mk_match(0, "target")):
            ScipyFFT.prev_fast_len((1,), real=True)
        ScipyFFT.prev_fast_len(1, real=True)

    def test_real(self):
        with assert_raises(TypingError, match=mk_match(1, "real")):
            ScipyFFT.prev_fast_len(1, real=None)
        with assert_raises(TypingError, match=mk_match(1, "real")):
            ScipyFFT.prev_fast_len(1, real=(True,))
        with assert_raises(TypingError, match=mk_match(1, "real")):
            ScipyFFT.prev_fast_len(1, real=1)
        ScipyFFT.prev_fast_len(1, real=False)


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

    @pytest.mark.parametrize(
        "func",
        [
            NumpyFFT.fft,
            NumpyFFT.ifft,
            NumpyFFT.rfft,
            NumpyFFT.irfft,
            NumpyFFT.hfft,
            NumpyFFT.ihfft,
        ],
    )
    def test_out(self, func):
        with assert_raises(TypingError, match=mk_match(4, "out")):
            func(self.x, out=1.0)
        with assert_raises(TypingError, match=mk_match(4, "out")):
            func(self.x, out=(1, 2))
        func(self.x, out=None)
        if func in (NumpyFFT.irfft, NumpyFFT.hfft):
            inp = self.x[:22] + 1j * self.x[:22]
            out = np.empty(42, dtype=np.float64)
            func(inp, out=out)
        elif func in (NumpyFFT.rfft, NumpyFFT.ihfft):
            inp = self.x
            out = np.empty(22, dtype=np.complex128)
            func(inp, out=out)
        else:
            inp = self.x.astype(np.complex128)
            out = np.empty_like(inp)
            func(inp, out=out)


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
        with assert_raises(ValueError):
            func(self.x, 1)
        with assert_raises(ValueError):
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

    @pytest.mark.parametrize(
        "func",
        [
            NumpyFFT.fft2,
            NumpyFFT.ifft2,
            NumpyFFT.rfft2,
            NumpyFFT.irfft2,
        ],
    )
    def test_out(self, func):
        with assert_raises(TypingError, match=mk_match(4, "out")):
            func(self.x, out=1.0)
        with assert_raises(TypingError, match=mk_match(4, "out")):
            func(self.x, out=(1, 2))
        func(self.x, out=None)
        if func is NumpyFFT.irfft2:
            inp = (self.x[:, :22] + 1j * self.x[:, :22]).copy()
            out = np.empty((42, 42), dtype=np.float64)
            func(inp, out=out)
        elif func is NumpyFFT.rfft2:
            inp = self.x
            out = np.empty((42, 22), dtype=np.complex128)
            func(inp, out=out)
        else:
            inp = self.x.astype(np.complex128)
            out = np.empty_like(inp)
            func(inp, out=out)


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
        with assert_raises(ValueError):
            func(self.x, 0)
        with assert_raises(ValueError):
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
        with assert_raises(ValueError):
            func(self.x, 1, 0)
        with assert_raises(ValueError):
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
        with assert_raises(ValueError):
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
        with assert_raises(ValueError):
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
        with assert_raises(ValueError):
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
        with assert_raises(ValueError):
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

    @pytest.mark.parametrize(
        "func",
        [
            NumpyFFT.fftn,
            NumpyFFT.ifftn,
            NumpyFFT.rfftn,
            NumpyFFT.irfftn,
        ],
    )
    def test_out(self, func):
        with assert_raises(TypingError, match=mk_match(4, "out")):
            func(self.x, out=1.0)
        with assert_raises(TypingError, match=mk_match(4, "out")):
            func(self.x, out=(1, 2))
        func(self.x, out=None)
        if func is NumpyFFT.irfftn:
            inp = (self.x[:, :, :4] + 1j * self.x[:, :, :4]).copy()
            out = np.empty((7, 7, 6), dtype=np.float64)
            func(inp, out=out)
        elif func is NumpyFFT.rfftn:
            inp = self.x
            out = np.empty((7, 7, 4), dtype=np.complex128)
            func(inp, out=out)
        else:
            inp = self.x.astype(np.complex128)
            out = np.empty_like(inp)
            func(inp, out=out)


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
