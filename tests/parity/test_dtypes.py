"""Result dtypes, precision and overwrite_x semantics match NumPy and SciPy."""

import numpy as np
import pytest
import scipy.fft
from common import NORMS, NumpyFFT, ScipyFFT, assert_parity, random_array

DTYPES = [
    np.bool_,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float32, np.float64,
    np.complex64, np.complex128,
]  # fmt: skip
REAL_DTYPES = [dt for dt in DTYPES if not np.issubdtype(dt, np.complexfloating)]


def _input(dtype, shape=42):
    if dtype is np.bool_:
        return np.arange(np.prod(shape)).reshape(shape) % 2 == 0
    if np.issubdtype(dtype, np.integer):
        return np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    return random_array(shape, dtype)


def _tol(*arrays):
    single = any(a.dtype in (np.float32, np.complex64) for a in arrays)
    return dict(rtol=1e-4, atol=1e-3) if single else dict(rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("dtype", DTYPES)
def test_numpy(dtype):
    x = _input(dtype)
    for name in ["fft", "ifft", "fftn", "hfft"] + (["rfft", "ihfft"] if dtype in REAL_DTYPES else []):
        got, ref = getattr(NumpyFFT, name)(x), getattr(np.fft, name)(x)
        assert_parity(got, ref, **_tol(got, ref))


@pytest.mark.parametrize("dtype", DTYPES)
def test_scipy(dtype):
    x = _input(dtype)
    names = ["fft", "ifft", "fftn", "hfft", "dct", "idst", "dctn"]
    if dtype in REAL_DTYPES:
        names += ["rfft", "ihfft", "rfftn"]
    for name in names:
        got, ref = getattr(ScipyFFT, name)(x), getattr(scipy.fft, name)(x)
        assert_parity(got, ref, **_tol(got, ref))


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fht(dtype):
    x = _input(dtype)
    one = 1.0 if dtype is np.bool_ else dtype(1)
    for name in ["fht", "ifht"]:
        got, ref = getattr(ScipyFFT, name)(x, one, one), getattr(scipy.fft, name)(x, one, one)
        assert_parity(got, ref, **_tol(got, ref))
    got, ref = ScipyFFT.fhtoffset(one, one), scipy.fft.fhtoffset(one, one)
    assert type(got) is type(ref.item())
    assert got == pytest.approx(ref)


@pytest.mark.parametrize("norm", NORMS)
def test_single_precision_values(norm):
    # The transforms run in single precision, so the results only match SciPy's
    # (also single precision) to float32 accuracy.
    x = random_array((8, 6, 4), np.float32)
    z = random_array((8, 6, 4), np.complex64)
    tol = dict(rtol=1e-4, atol=1e-4)
    for name in ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "hfft"]:
        assert_parity(getattr(ScipyFFT, name)(z, norm=norm), getattr(scipy.fft, name)(z, norm=norm), **tol)
        assert_parity(getattr(NumpyFFT, name)(z, norm=norm), getattr(np.fft, name)(z, norm=norm), **tol)
    for name in ["rfft", "rfft2", "rfftn", "ihfft", "dct", "dst", "dctn", "dstn"]:
        assert_parity(getattr(ScipyFFT, name)(x, norm=norm), getattr(scipy.fft, name)(x, norm=norm), **tol)
    for name in ["irfft", "irfft2", "irfftn"]:
        y = getattr(np.fft, name[1:])(x)
        assert_parity(getattr(ScipyFFT, name)(y, norm=norm), getattr(scipy.fft, name)(y, norm=norm), **tol)
        assert_parity(getattr(NumpyFFT, name)(y, norm=norm), getattr(np.fft, name)(y, norm=norm), **tol)


def test_single_precision_fht():
    x = random_array(32, np.float32)
    dln, mu = np.float32(0.1), np.float32(0.5)
    tol = dict(rtol=1e-4, atol=1e-4)
    y = ScipyFFT.fht(x, dln, mu)
    assert_parity(y, scipy.fft.fht(x, dln, mu), **tol)
    assert_parity(ScipyFFT.ifht(y, dln, mu), scipy.fft.ifht(y, dln, mu), **tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["fft", "ifft", "fftn", "rfft", "irfft", "dct", "idctn", "dst"])
def test_overwrite_x(name, dtype):
    # Whether the result is the input itself, or reuses its memory, matches SciPy.
    if name == "rfft" and dtype not in REAL_DTYPES:
        pytest.skip("rfft takes real input only")
    x_jit, x_ref = _input(dtype), _input(dtype)
    got = getattr(ScipyFFT, name)(x_jit, overwrite_x=True)
    ref = getattr(scipy.fft, name)(x_ref, overwrite_x=True)
    assert_parity(got, ref, **_tol(got, ref))
    assert (got is x_jit) == (ref is x_ref)
    assert np.shares_memory(got, x_jit) == np.shares_memory(ref, x_ref)
