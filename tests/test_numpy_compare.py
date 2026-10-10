import numpy as np
import pytest

from helpers import (
    numba_cache_cleanup,
    set_numba_capture_errors_new_style,
    NumpyFFT,
)

set_numba_capture_errors_new_style()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def assert_same_result(a, b, rtol=1e-12, atol=1e-12):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def assert_numpy_equivalent_behavior(inp, out_nb, out_np, inp_before):
    """
    Ensure aliasing, identity, and mutation behavior matches NumPy exactly.
    """
    # Identity equivalence
    assert (out_nb is inp) == (out_np is inp)

    # Memory sharing equivalence
    assert np.shares_memory(out_nb, inp) == np.shares_memory(out_np, inp)

    # Input mutation equivalence
    mutated_np = not np.array_equal(inp, inp_before)
    mutated_nb = not np.array_equal(inp, inp_before)
    assert mutated_nb == mutated_np


def random_array(rng, shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return rng.standard_normal(shape).astype(dtype) + 1j * rng.standard_normal(
            shape
        ).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


# -----------------------------------------------------------------------------
# FFT / IFFT (1D)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("n", [None, 8])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_fft_ifft(dtype, n, axis, norm):
    rng = np.random.default_rng(0)
    a = random_array(rng, (16,), dtype)
    a_before = a.copy()

    out_nb = NumpyFFT.fft(a, n, axis, norm)
    out_np = np.fft.fft(a, n, axis, norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(a, out_nb, out_np, a_before)

    a_before = a.copy()

    out_nb = NumpyFFT.ifft(a, n, axis, norm)
    out_np = np.fft.ifft(a, n, axis, norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(a, out_nb, out_np, a_before)


# -----------------------------------------------------------------------------
# FFT2 / IFFT2
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("shape", [(8, 6)])
@pytest.mark.parametrize("axes", [(0, 1), (-2, -1)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_fft2_ifft2(dtype, shape, axes, norm):
    rng = np.random.default_rng(1)
    a = random_array(rng, shape, dtype)
    a_before = a.copy()

    out_nb = NumpyFFT.fft2(a, axes=axes, norm=norm)
    out_np = np.fft.fft2(a, axes=axes, norm=norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(a, out_nb, out_np, a_before)

    a_before = a.copy()

    out_nb = NumpyFFT.ifft2(a, axes=axes, norm=norm)
    out_np = np.fft.ifft2(a, axes=axes, norm=norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(a, out_nb, out_np, a_before)


# -----------------------------------------------------------------------------
# FFTN / IFFTN
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("shape", [(4, 5, 6)])
@pytest.mark.parametrize("axes", [None, (-1,), (-2, -1), (-2, -1, 0)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_fftn_ifftn(dtype, shape, axes, norm):
    rng = np.random.default_rng(2)
    a = random_array(rng, shape, dtype)
    a_before = a.copy()

    out_nb = NumpyFFT.fftn(a, axes=axes, norm=norm)
    out_np = np.fft.fftn(a, axes=axes, norm=norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(a, out_nb, out_np, a_before)

    a_before = a.copy()

    out_nb = NumpyFFT.ifftn(a, axes=axes, norm=norm)
    out_np = np.fft.ifftn(a, axes=axes, norm=norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(a, out_nb, out_np, a_before)


# -----------------------------------------------------------------------------
# RFFT / IRFFT (1D)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [None, 16])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_rfft_irfft(n, axis, norm):
    rng = np.random.default_rng(3)
    a = rng.standard_normal(32)
    a_before = a.copy()

    r_nb = NumpyFFT.rfft(a, n, axis, norm)
    r_np = np.fft.rfft(a, n, axis, norm)

    assert_same_result(r_nb, r_np)
    assert_numpy_equivalent_behavior(a, r_nb, r_np, a_before)

    r_before = r_nb.copy()

    out_nb = NumpyFFT.irfft(r_nb, n, axis, norm)
    out_np = np.fft.irfft(r_np, n, axis, norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(r_nb, out_nb, out_np, r_before)


# -----------------------------------------------------------------------------
# RFFT2 / IRFFT2
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(8, 6)])
@pytest.mark.parametrize("axes", [(0, 1), (-2, -1)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_rfft2_irfft2(shape, axes, norm):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(shape)
    a_before = a.copy()

    r_nb = NumpyFFT.rfft2(a, axes=axes, norm=norm)
    r_np = np.fft.rfft2(a, axes=axes, norm=norm)

    assert_same_result(r_nb, r_np)
    assert_numpy_equivalent_behavior(a, r_nb, r_np, a_before)

    r_before = r_nb.copy()

    out_nb = NumpyFFT.irfft2(r_nb, axes=axes, norm=norm)
    out_np = np.fft.irfft2(r_np, axes=axes, norm=norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(r_nb, out_nb, out_np, r_before)


# -----------------------------------------------------------------------------
# RFFTN / IRFFTN
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 5, 6)])
@pytest.mark.parametrize("axes", [None, (-1,), (-2, -1), (-2, -1, 0)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_rfftn_irfftn(shape, axes, norm):
    rng = np.random.default_rng(5)
    a = rng.standard_normal(shape)
    a_before = a.copy()

    r_nb = NumpyFFT.rfftn(a, axes=axes, norm=norm)
    r_np = np.fft.rfftn(a, axes=axes, norm=norm)

    assert_same_result(r_nb, r_np)
    assert_numpy_equivalent_behavior(a, r_nb, r_np, a_before)

    r_before = r_nb.copy()

    out_nb = NumpyFFT.irfftn(r_nb, axes=axes, norm=norm)
    out_np = np.fft.irfftn(r_np, axes=axes, norm=norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(r_nb, out_nb, out_np, r_before)


# -----------------------------------------------------------------------------
# HFFT / IHFFT
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [None, 16])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_hfft_ihfft(n, axis, norm):
    rng = np.random.default_rng(6)
    a = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    a_before = a.copy()

    r_nb = NumpyFFT.hfft(a, n, axis, norm)
    r_np = np.fft.hfft(a, n, axis, norm)

    assert_same_result(r_nb, r_np)
    assert_numpy_equivalent_behavior(a, r_nb, r_np, a_before)

    r_before = r_nb.copy()

    out_nb = NumpyFFT.ihfft(r_nb, n, axis, norm)
    out_np = np.fft.ihfft(r_np, n, axis, norm)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(r_nb, out_nb, out_np, r_before)


# -----------------------------------------------------------------------------
# Frequencies
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 8, 16])
@pytest.mark.parametrize("d", [1.0, 0.1])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_fftfreq(n, d, device):
    out_nb = NumpyFFT.fftfreq(n, d, device)
    out_np = np.fft.fftfreq(n, d, device)
    assert_same_result(out_nb, out_np)


@pytest.mark.parametrize("n", [1, 8, 16])
@pytest.mark.parametrize("d", [1.0, 0.1])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_rfftfreq(n, d, device):
    out_nb = NumpyFFT.rfftfreq(n, d, device)
    out_np = np.fft.rfftfreq(n, d, device)
    assert_same_result(out_nb, out_np)


# -----------------------------------------------------------------------------
# FFTSHIFT / IFFTSHIFT
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(8,), (4, 6), (3, 4, 5)])
@pytest.mark.parametrize("axes", [None, 0, -1, (0, -1)])
def test_fftshift_ifftshift(shape, axes):
    rng = np.random.default_rng(7)
    a = rng.standard_normal(shape)
    a_before = a.copy()

    s_nb = NumpyFFT.fftshift(a, axes)
    s_np = np.fft.fftshift(a, axes)

    assert_same_result(s_nb, s_np)
    assert_numpy_equivalent_behavior(a, s_nb, s_np, a_before)

    s_before = s_nb.copy()

    out_nb = NumpyFFT.ifftshift(s_nb, axes)
    out_np = np.fft.ifftshift(s_np, axes)

    assert_same_result(out_nb, out_np)
    assert_numpy_equivalent_behavior(s_nb, out_nb, out_np, s_before)


# -----------------------------------------------------------------------------
# OUT ARGUMENT
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func_name,shape,is_complex",
    [
        ("fft", (16,), True),
        ("ifft", (16,), True),
        ("fft2", (8, 6), True),
        ("ifft2", (8, 6), True),
        ("fftn", (4, 6, 8), True),
        ("ifftn", (4, 6, 8), True),
        ("rfft", (16,), False),
        ("irfft", (16,), True),
        ("rfft2", (8, 6), False),
        ("irfft2", (8, 6), True),
        ("rfftn", (4, 6, 8), False),
        ("irfftn", (4, 6, 8), True),
        ("hfft", (16,), True),
        ("ihfft", (16,), False),
    ],
)
def test_out_argument(func_name, shape, is_complex):
    rng = np.random.default_rng(42)
    nb_func = getattr(NumpyFFT, func_name)
    np_func = getattr(np.fft, func_name)

    if is_complex:
        if func_name in ("irfft", "irfft2", "irfftn", "hfft"):
            real_shape = list(shape)
            real_shape[-1] = (shape[-1] - 1) * 2
            r_func = getattr(np.fft, "rfft" if func_name == "hfft" else func_name[1:])
            inp = r_func(rng.standard_normal(real_shape))
        else:
            inp = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    else:
        inp = rng.standard_normal(shape)

    expected_np = np_func(inp)
    out_nb = np.zeros_like(expected_np)
    res_nb = nb_func(inp, out=out_nb)

    assert res_nb is out_nb
    assert_same_result(res_nb, expected_np)


def test_out_inplace():
    rng = np.random.default_rng(123)
    a = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    a_np = a.copy()
    a_nb = a.copy()

    res_np = np.fft.fft(a_np, out=a_np)
    res_nb = NumpyFFT.fft(a_nb, out=a_nb)

    assert res_np is a_np
    assert res_nb is a_nb
    assert_same_result(res_nb, res_np)


def test_out_shape_mismatch():
    a = np.ones(16, dtype=np.complex128)
    out = np.zeros(8, dtype=np.complex128)
    with pytest.raises(Exception):
        NumpyFFT.fft(a, out=out)


def test_out_dtype_mismatch():
    a = np.ones(16, dtype=np.complex128)
    out = np.zeros(16, dtype=np.float64)
    with pytest.raises(Exception):
        NumpyFFT.fft(a, out=out)
