"""numpy.fft transforms in jitted code match NumPy."""

import itertools

import numpy as np
import pytest
from common import NORMS, NumpyFFT, assert_new_array, assert_parity, random_array

PAIRS_1D = [("fft", "ifft"), ("rfft", "irfft"), ("hfft", "ihfft")]
PAIRS_ND = [("fft2", "ifft2"), ("fftn", "ifftn"), ("rfft2", "irfft2"), ("rfftn", "irfftn")]


def _check(name, x, *args, **kwargs):
    """Compare one call with NumPy, including that a fresh array is returned."""
    before = x.copy()
    got = getattr(NumpyFFT, name)(x, *args, **kwargs)
    ref = getattr(np.fft, name)(x, *args, **kwargs)
    assert_parity(got, ref)
    assert_new_array(got, x, before)
    return got


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("n", [None, 8, 16, 40])
@pytest.mark.parametrize("forward, inverse", PAIRS_1D)
def test_1d(forward, inverse, n, norm):
    dtype = np.complex128 if forward == "hfft" else np.float64
    for axis in (0, -1):
        y = _check(forward, random_array((32, 32), dtype), n, axis, norm)
        _check(inverse, y, n, axis, norm)
    if forward == "fft":
        _check("fft", random_array(32, np.complex128), n, -1, norm)


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("axes", [None, (0, 1), (-2, -1), (-1, 0), (0, 2), (-1,), (-2, -1, 0)])
@pytest.mark.parametrize("forward, inverse", PAIRS_ND)
def test_nd(forward, inverse, axes, norm):
    if forward.endswith("2") and (axes is None or len(axes) != 2):
        pytest.skip("fft2 family needs exactly two axes")
    y = _check(forward, random_array((4, 5, 6)), axes=axes, norm=norm)
    _check(inverse, y, axes=axes, norm=norm)
    if forward.startswith("fft"):
        _check(forward, random_array((4, 5, 6), np.complex128), axes=axes, norm=norm)


@pytest.mark.parametrize("s", [(8, 6), (3, 10), (-1, 7)])
@pytest.mark.parametrize("forward, inverse", PAIRS_ND)
def test_nd_shape(forward, inverse, s):
    x = random_array((4, 5, 6))
    y = _check(forward, x, s=s, axes=(0, 2))
    _check(inverse, y, s=s, axes=(0, 2))


# Every length from 1 to 512 exercises all of DUCC0's radix kernels and Bluestein.
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_all_lengths(dtype):
    x = random_array(512, dtype)
    z = random_array(512, np.complex128)
    rtol = 1e-12 if dtype == np.float64 else 1e-4
    for n in range(1, 513):
        assert_parity(NumpyFFT.fft(z[:n]), np.fft.fft(z[:n]), rtol, rtol * n)
        assert_parity(NumpyFFT.ifft(z[:n]), np.fft.ifft(z[:n]), rtol, rtol)
        assert_parity(NumpyFFT.rfft(x[:n]), np.fft.rfft(x[:n]), rtol, rtol * n)
        r = np.fft.rfft(x[:n])
        assert_parity(NumpyFFT.irfft(r, n), np.fft.irfft(r, n), rtol, rtol)


@pytest.mark.parametrize("name", ["fftn", "ifftn", "rfftn", "irfftn"])
def test_axes_order(name):
    x = random_array((30, 20, 10))
    if name == "irfftn":
        x = np.fft.rfftn(x)
    for axes in itertools.permutations(range(3)):
        _check(name, x, axes=axes)


def test_irfftn_not_last_axis():
    x = random_array((2, 16, 8, 32), np.complex128)
    _check("irfftn", x, axes=(-2,))


@pytest.mark.parametrize("name", ["fft2", "fftn", "ifft2", "ifftn"])
@pytest.mark.parametrize("axes", [(0, 0), (0, 2, 2), (0, 2, 1, 0), (3, 1), (2, 1, 0), (0, 1, 2, 3)])
def test_duplicate_axes(name, axes):
    # NumPy transforms a repeated axis once per occurrence.
    x = random_array((3, 3, 3, 3), np.complex128)
    _check(name, x, axes=axes)


@pytest.mark.parametrize("name", ["rfftn", "irfftn"])
@pytest.mark.parametrize("axes", [(0, 0), (2, 0, 2), (1, 2, 1)])
def test_duplicate_axes_real(name, axes):
    x = random_array((4, 6, 8))
    if name == "irfftn":
        x = np.fft.rfftn(x)
    _check(name, x, axes=axes)


OUT_CASES = [
    ("fft", (16,), np.complex128), ("ifft", (16,), np.complex128),
    ("fft2", (8, 6), np.complex128), ("ifft2", (8, 6), np.complex128),
    ("fftn", (4, 6, 8), np.complex128), ("ifftn", (4, 6, 8), np.complex128),
    ("rfft", (16,), np.float64), ("irfft", (16,), np.complex128),
    ("rfft2", (8, 6), np.float64), ("irfft2", (8, 6), np.complex128),
    ("rfftn", (4, 6, 8), np.float64), ("irfftn", (4, 6, 8), np.complex128),
    ("hfft", (16,), np.complex128), ("ihfft", (16,), np.float64),
]  # fmt: skip


@pytest.mark.parametrize("name, shape, dtype", OUT_CASES)
def test_out(name, shape, dtype):
    x = random_array(shape, dtype)
    ref = getattr(np.fft, name)(x)
    out = np.zeros_like(ref)
    got = getattr(NumpyFFT, name)(x, out=out)
    assert got is out
    assert_parity(got, ref)


def test_out_is_input():
    x = random_array(16, np.complex128)
    ref = np.fft.fft(x)
    got = NumpyFFT.fft(x, out=x)
    assert got is x
    assert_parity(got, ref)
