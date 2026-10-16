"""Output memory layout parity with NumPy and SciPy.

For every function and a variety of input layouts, the jitted result must have
the same strides (hence the same contiguity) as the reference result. For
SciPy, whether the result is the input itself or shares memory with it
(``overwrite_x``, truncating ``n``/``s``) must match as well.
"""

import functools

import numba as nb
import numpy as np
import pytest
import scipy.fft

import rocket_fft  # noqa: F401

SHAPE = (4, 6, 8)

# Each layout builds a fresh array of SHAPE (except where noted) from a base array.
LAYOUTS = {
    "C": lambda f: f(SHAPE),
    "F": lambda f: np.asfortranarray(f(SHAPE)),
    "inner-strided": lambda f: f((4, 6, 16))[..., ::2],
    "outer-strided": lambda f: f((8, 6, 8))[::2],
    "F-strided": lambda f: np.asfortranarray(f((4, 6, 16)))[..., ::2],
    "permuted": lambda f: f((8, 4, 6)).transpose(1, 2, 0),
    "negative": lambda f: f(SHAPE)[::-1, :, ::-1],
    "size-1 F": lambda f: np.asfortranarray(f((1, 6, 8))),
    "1-D strided": lambda f: f((16,))[::2],
}


def _generator(dtype):
    def make(shape):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(shape)
        if np.issubdtype(dtype, np.complexfloating):
            x = x + 1j * rng.standard_normal(shape)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-5, 5, shape)
        return x.astype(dtype)

    return make


def _make_input(layout, dtype):
    return LAYOUTS[layout](_generator(dtype))


@functools.lru_cache(maxsize=None)
def _jitted(module, name, kwargs):
    ns = {"np": np, "scipy": scipy}
    args = "".join(f", {k}={v!r}" for k, v in kwargs)
    exec(f"def f(x, overwrite_x): return {module}.{name}(x{args}{', overwrite_x=overwrite_x' if module == 'scipy.fft' else ''})", ns)
    return nb.njit(ns["f"])


def _applicable(x, kwargs):
    ndim_needed = 0
    for key, value in kwargs:
        if key in ("axis",):
            ndim_needed = max(ndim_needed, value + 1 if value >= 0 else -value)
        if key in ("axes",) and value is not None:
            ndim_needed = max(ndim_needed, max(a + 1 if a >= 0 else -a for a in value))
        if key == "s" and value is not None:
            ndim_needed = max(ndim_needed, len(value))
    return x.ndim >= max(ndim_needed, 1 if not any(k in ("axes", "s") for k, _ in kwargs) else 2)


def _check(module, name, kwargs, layout, dtype, overwrite_x=False):
    ref_in = _make_input(layout, dtype)
    if not _applicable(ref_in, kwargs) or (name in ("fft2", "ifft2", "rfft2", "irfft2", "hfft2", "ihfft2") and ref_in.ndim < 2):
        pytest.skip("not applicable to this input")
    jit_in = _make_input(layout, dtype)
    ref_fn = functools.reduce(getattr, module.split(".")[1:] + [name], np if module.startswith("np") else scipy)
    kw = dict(kwargs)
    if module == "scipy.fft":
        kw["overwrite_x"] = overwrite_x
    try:
        ref = ref_fn(ref_in, **kw)
    except (ValueError, TypeError):
        pytest.skip("rejected by the reference")
    got = _jitted(module, name, kwargs)(jit_in, overwrite_x)

    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert got.strides == ref.strides
    assert got.flags.c_contiguous == ref.flags.c_contiguous
    assert got.flags.f_contiguous == ref.flags.f_contiguous
    assert (got is jit_in) == (ref is ref_in)
    assert np.shares_memory(got, jit_in) == np.shares_memory(ref, ref_in)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


FFT_CASES = [
    ("fft", ()), ("fft", (("axis", 0),)), ("fft", (("n", 12),)), ("fft", (("n", 5),)),
    ("ifft", (("axis", 1),)),
    ("rfft", ()), ("rfft", (("axis", 1),)), ("rfft", (("n", 5), ("axis", 0))),
    ("irfft", ()), ("irfft", (("n", 7),)), ("irfft", (("axis", 0),)),
    ("hfft", ()), ("ihfft", (("axis", 1),)),
    ("fft2", ()), ("fft2", (("axes", (0, 2)),)), ("fft2", (("s", (3, 10)), ("axes", (0, 2)))),
    ("ifft2", ()), ("rfft2", ()), ("irfft2", (("axes", (1, 0)),)),
    ("fftn", ()), ("fftn", (("axes", (2, 0)),)), ("fftn", (("s", (5, 7)), ("axes", (1, 2)))),
    ("ifftn", (("axes", (0, 1)),)),
    ("rfftn", ()), ("rfftn", (("axes", (0, 1)),)), ("rfftn", (("s", (3, 6)), ("axes", (2, 0)))),
    ("irfftn", ()), ("irfftn", (("s", (4, 6, 10)), ("axes", (0, 1, 2)))),
]
SCIPY_ONLY_CASES = [
    ("hfft2", ()), ("ihfft2", (("axes", (0, 2)),)), ("hfftn", (("axes", (1, 2)),)), ("ihfftn", ()),
]
R2R_CASES = [
    (name, kwargs)
    for name in ("dct", "idct", "dst", "idst")
    for kwargs in ((), (("type", 1),), (("axis", 0),), (("n", 12),), (("n", 5),))
] + [
    (name, kwargs)
    for name in ("dctn", "idctn", "dstn", "idstn")
    for kwargs in ((), (("axes", (0, 2)),), (("s", (5, 3)), ("axes", (1, 0))), (("s", (3, 3)), ("axes", (0, 1))))
]


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.int64])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name, kwargs", FFT_CASES)
def test_numpy_fft(name, kwargs, layout, dtype):
    if name in ("rfft", "ihfft", "rfft2", "rfftn") and dtype is np.complex128:
        pytest.skip("real input only")
    _check("np.fft", name, kwargs, layout, dtype)


@pytest.mark.parametrize("overwrite_x", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.int64, np.float32])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name, kwargs", FFT_CASES + SCIPY_ONLY_CASES + R2R_CASES)
def test_scipy_fft(name, kwargs, layout, dtype, overwrite_x):
    if name in ("rfft", "ihfft", "rfft2", "rfftn", "ihfft2", "ihfftn") and dtype is np.complex128:
        pytest.skip("real input only")
    _check("scipy.fft", name, kwargs, layout, dtype, overwrite_x)


@functools.lru_cache(maxsize=None)
def _jitted_simple(src):
    ns = {"np": np, "scipy": scipy}
    exec(f"def f(x): return {src}", ns)
    return nb.njit(ns["f"])


EXTRA_CASES = [
    ("np.fft.fftshift(x)", np.fft.fftshift),
    ("np.fft.ifftshift(x, axes=0)", lambda x: np.fft.ifftshift(x, axes=0)),
    ("scipy.fft.fftshift(x, axes=(0, -1))", lambda x: scipy.fft.fftshift(x, axes=(0, -1))),
    ("np.roll(x, 3)", lambda x: np.roll(x, 3)),
    ("np.roll(x, 3, axis=None)", lambda x: np.roll(x, 3, axis=None)),
    ("np.roll(x, (1, -2), axis=(0, -1))", lambda x: np.roll(x, (1, -2), axis=(0, -1))),
    ("scipy.fft.fht(x, 0.1, 0.5)", lambda x: scipy.fft.fht(x, 0.1, 0.5)),
    ("scipy.fft.ifht(x, 0.1, 0.5, offset=0.2)", lambda x: scipy.fft.ifht(x, 0.1, 0.5, offset=0.2)),
]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("src, ref_fn", EXTRA_CASES, ids=[c[0] for c in EXTRA_CASES])
def test_extras(src, ref_fn, layout, dtype):
    x = _make_input(layout, dtype)
    if "axis=(0, -1)" in src and x.ndim < 2:
        pytest.skip("not applicable to this input")
    ref = ref_fn(x)
    got = _jitted_simple(src)(x)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert got.strides == ref.strides
    assert got.flags.c_contiguous == ref.flags.c_contiguous
    assert got.flags.f_contiguous == ref.flags.f_contiguous
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_fht_bias_dtype():
    """SciPy promotes float32 input to float64 iff bias != 0.

    That depends on the run-time value of bias, but a compiled function needs
    a static result dtype, so rocket-fft keeps single precision. Only float32
    input with a nonzero bias therefore differs from SciPy (values agree).
    """
    x = np.random.rand(16).astype(np.float32)
    fht = nb.njit(lambda x, bias: scipy.fft.fht(x, 0.1, 0.5, bias=bias))
    assert fht(x, 0.0).dtype == scipy.fft.fht(x, 0.1, 0.5, bias=0.0).dtype == np.float32
    assert fht(x, 0.3).dtype == np.float32  # SciPy: float64 (known, unavoidable difference)
    np.testing.assert_allclose(fht(x, 0.3), scipy.fft.fht(x, 0.1, 0.5, bias=0.3), rtol=1e-5)
