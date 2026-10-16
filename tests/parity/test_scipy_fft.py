"""scipy.fft transforms in jitted code match SciPy."""

import numpy as np
import pytest
import scipy.fft
from common import NORMS, ScipyFFT, assert_parity, random_array

PAIRS_1D = [("fft", "ifft"), ("rfft", "irfft"), ("hfft", "ihfft")]
PAIRS_ND = [("fft2", "ifft2"), ("fftn", "ifftn"), ("rfft2", "irfft2"), ("rfftn", "irfftn")]
R2R_1D = ["dct", "idct", "dst", "idst"]
R2R_ND = ["dctn", "idctn", "dstn", "idstn"]

# Sizes that take DUCC0's large-radix and Bluestein code paths.
LARGE_SIZES = [2**13, 2**5 * 3**5, 2**3 * 3**3 * 5**2, 2011]


def _check(name, x, *args, overwrite_x=False, rtol=1e-12, atol=1e-12, **kwargs):
    """Compare one call with SciPy. Without overwrite_x the input must stay intact."""
    x_jit, x_ref = x.copy(), x.copy()
    got = getattr(ScipyFFT, name)(x_jit, *args, overwrite_x=overwrite_x, **kwargs)
    ref = getattr(scipy.fft, name)(x_ref, *args, overwrite_x=overwrite_x, **kwargs)
    assert_parity(got, ref, rtol, atol)
    if not overwrite_x:
        np.testing.assert_array_equal(x_jit, x)
        assert not np.shares_memory(got, x_jit)
    return got


@pytest.mark.parametrize("overwrite_x", [False, True])
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("n", [None, 8, 16, 40])
@pytest.mark.parametrize("forward, inverse", PAIRS_1D)
def test_1d(forward, inverse, n, norm, overwrite_x):
    dtype = np.complex128 if forward == "hfft" else np.float64
    for axis in (0, -1):
        y = _check(forward, random_array((32, 32), dtype), n, axis, norm, overwrite_x=overwrite_x)
        _check(inverse, y, n, axis, norm, overwrite_x=overwrite_x)
    if forward == "fft":
        _check("fft", random_array(32, np.complex128), n, -1, norm, overwrite_x=overwrite_x)


@pytest.mark.parametrize("overwrite_x", [False, True])
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("axes", [None, (0, 1), (-2, -1), (-1, 0), (0, 2), (-1,), (-2, -1, 0)])
@pytest.mark.parametrize("forward, inverse", PAIRS_ND)
def test_nd(forward, inverse, axes, norm, overwrite_x):
    if forward.endswith("2") and (axes is None or len(axes) != 2):
        pytest.skip("fft2 family needs exactly two axes")
    y = _check(forward, random_array((4, 5, 6)), axes=axes, norm=norm, overwrite_x=overwrite_x)
    _check(inverse, y, axes=axes, norm=norm, overwrite_x=overwrite_x)
    if forward.startswith("fft"):
        x = random_array((4, 5, 6), np.complex128)
        _check(forward, x, axes=axes, norm=norm, overwrite_x=overwrite_x)


@pytest.mark.parametrize("s", [(8, 6), (3, 10), (-1, 7)])
@pytest.mark.parametrize("forward, inverse", PAIRS_ND + [("hfft2", "ihfft2"), ("hfftn", "ihfftn")])
def test_nd_shape(forward, inverse, s):
    x = random_array((4, 5, 6), np.complex128 if forward.startswith("h") else np.float64)
    y = _check(forward, x, s=s, axes=(0, 2))
    _check(inverse, y, s=s, axes=(0, 2))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("size", LARGE_SIZES)
def test_large_sizes(size, dtype):
    rtol = 1e-10 if dtype == np.float64 else 1e-4
    x = random_array(size, dtype)
    z = random_array(size, np.complex128).astype(np.result_type(dtype, np.complex64))
    atol = rtol * np.sqrt(size)
    _check("fft", z, rtol=rtol, atol=atol)
    _check("ifft", z, rtol=rtol, atol=atol / size)
    r = _check("rfft", x, rtol=rtol, atol=atol)
    _check("irfft", r, size, rtol=rtol, atol=atol / size)
    _check("fftn", random_array((size, 3), dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("overwrite_x", [False, True])
@pytest.mark.parametrize("orthogonalize", [None, False, True])
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("name", R2R_1D)
def test_r2r_1d(name, type, norm, orthogonalize, overwrite_x):
    x = random_array((32, 32))
    for n in (None, 8, 16, 40):
        for axis in (0, -1):
            kwargs = dict(type=type, n=n, axis=axis, norm=norm, orthogonalize=orthogonalize)
            _check(name, x, overwrite_x=overwrite_x, **kwargs)


# Odd and prime lengths take separate DCT/DST code paths (DCT-IV in particular).
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("name", R2R_1D)
def test_r2r_sizes(name, type, dtype):
    rtol = 1e-10 if dtype == np.float64 else 1e-4
    x = random_array(512, dtype)
    for n in list(range(2, 64)) + [97, 127, 128, 255, 256, 509, 512]:
        _check(name, x[:n], type=type, rtol=rtol, atol=rtol * n)


@pytest.mark.parametrize("overwrite_x", [False, True])
@pytest.mark.parametrize("orthogonalize", [None, False, True])
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("name", R2R_ND)
def test_r2r_nd(name, type, norm, orthogonalize, overwrite_x):
    for shape in ((4, 5), (3, 4, 5)):
        x = random_array(shape)
        for axes in (None, (0,), (-1,), (0, 1), (1, 0)):
            kwargs = dict(type=type, axes=axes, norm=norm, orthogonalize=orthogonalize)
            _check(name, x, overwrite_x=overwrite_x, **kwargs)


@pytest.mark.parametrize("name", R2R_ND)
def test_r2r_nd_shape(name):
    x = random_array((32, 16))
    _check(name, x, s=(128, 128))
    _check(name, x, s=(5, 7), axes=(1, 0))
    _check(name, x, s=(-1, 7), axes=(0, 1))


@pytest.mark.parametrize("name", R2R_ND + ["fftn", "ifftn"])
@pytest.mark.parametrize("axes", [np.array([1]), np.array([0, 1]), np.array([-2, -1])])
def test_array_axes(name, axes):
    _check(name, random_array((8, 6)), axes=axes)


@pytest.mark.parametrize("name", R2R_1D + R2R_ND)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_r2r_complex_input(name, dtype):
    # Real and imaginary parts are transformed independently.
    rtol = 1e-12 if dtype == np.complex128 else 1e-5
    _check(name, random_array((6, 9), dtype), rtol=rtol, atol=rtol)


@pytest.mark.parametrize("name", ["fft2", "fftn", "ifft2", "ifftn"])
@pytest.mark.parametrize("axes", [(0, 0), (0, 2, 2), (0, 2, 1, 0), (3, 1), (2, 1, 0), (0, 1, 2, 3)])
def test_duplicate_axes(name, axes):
    # Unlike NumPy, SciPy rejects repeated axes.
    x = random_array((3, 3, 3, 3), np.complex128)
    if len(set(axes)) != len(axes):
        with pytest.raises(ValueError):
            getattr(scipy.fft, name)(x, axes=axes)
        with pytest.raises(ValueError):
            getattr(ScipyFFT, name)(x, axes=axes)
    else:
        _check(name, x, axes=axes)
