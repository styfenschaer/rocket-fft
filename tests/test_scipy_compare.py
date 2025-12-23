import numpy as np
import pytest
import scipy.fft
from pytest import raises

from helpers import (
    numba_cache_cleanup,
    set_numba_capture_errors_new_style,
    ScipyFFT,
)

set_numba_capture_errors_new_style()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def assert_same_result(a, b, rtol=1e-12, atol=1e-12):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def random_array(rng, shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return rng.standard_normal(shape).astype(dtype) + 1j * rng.standard_normal(
            shape
        ).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


def assert_not_same_array(out, inp):
    assert out is not inp
    assert not np.shares_memory(out, inp)


def assert_input_unchanged(inp, before):
    assert np.array_equal(inp, before)


# -----------------------------------------------------------------------------
# FFT / IFFT (1D)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("n", [None, 8])
@pytest.mark.parametrize("axis", [-1])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_fft_ifft(dtype, n, axis, norm, overwrite_x):
    rng = np.random.default_rng(0)
    a = random_array(rng, (16,), dtype)

    assert_same_result(
        ScipyFFT.fft(a, n, axis, norm, overwrite_x),
        scipy.fft.fft(a, n, axis, norm, overwrite_x),
    )

    assert_same_result(
        ScipyFFT.ifft(a, n, axis, norm, overwrite_x),
        scipy.fft.ifft(a, n, axis, norm, overwrite_x),
    )


# -----------------------------------------------------------------------------
# FFT2 / IFFT2
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("shape", [(8, 6)])
@pytest.mark.parametrize("axes", [(-2, -1)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_fft2_ifft2(dtype, shape, axes, norm, overwrite_x):
    rng = np.random.default_rng(1)
    a = random_array(rng, shape, dtype)

    assert_same_result(
        ScipyFFT.fft2(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
        scipy.fft.fft2(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
    )

    assert_same_result(
        ScipyFFT.ifft2(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
        scipy.fft.ifft2(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
    )


# -----------------------------------------------------------------------------
# FFTN / IFFTN
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("shape", [(4, 5, 6)])
@pytest.mark.parametrize("axes", [None, (-1,), (-2, -1)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_fftn_ifftn(dtype, shape, axes, norm, overwrite_x):
    rng = np.random.default_rng(2)
    a = random_array(rng, shape, dtype)

    assert_same_result(
        ScipyFFT.fftn(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
        scipy.fft.fftn(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
    )

    assert_same_result(
        ScipyFFT.ifftn(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
        scipy.fft.ifftn(a, axes=axes, norm=norm, overwrite_x=overwrite_x),
    )


# -----------------------------------------------------------------------------
# RFFT family
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [None, 16])
@pytest.mark.parametrize("axis", [-1])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_rfft_irfft(n, axis, norm, overwrite_x):
    rng = np.random.default_rng(3)
    a = rng.standard_normal(32)

    r_nb = ScipyFFT.rfft(a, n, axis, norm, overwrite_x)
    r_np = scipy.fft.rfft(a, n, axis, norm, overwrite_x)
    assert_same_result(r_nb, r_np)

    assert_same_result(
        ScipyFFT.irfft(r_nb, n, axis, norm, overwrite_x),
        scipy.fft.irfft(r_np, n, axis, norm, overwrite_x),
    )


@pytest.mark.parametrize("shape", [(8, 6)])
@pytest.mark.parametrize("axes", [(-2, -1)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_rfft2_irfft2(shape, axes, norm, overwrite_x):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(shape)

    r_nb = ScipyFFT.rfft2(a, axes=axes, norm=norm, overwrite_x=overwrite_x)
    r_np = scipy.fft.rfft2(a, axes=axes, norm=norm, overwrite_x=overwrite_x)
    assert_same_result(r_nb, r_np)

    assert_same_result(
        ScipyFFT.irfft2(r_nb, axes=axes, norm=norm, overwrite_x=overwrite_x),
        scipy.fft.irfft2(r_np, axes=axes, norm=norm, overwrite_x=overwrite_x),
    )


@pytest.mark.parametrize("shape", [(4, 5, 6)])
@pytest.mark.parametrize("axes", [None, (-2, -1)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_rfftn_irfftn(shape, axes, norm, overwrite_x):
    rng = np.random.default_rng(5)
    a = rng.standard_normal(shape)

    r_nb = ScipyFFT.rfftn(a, axes=axes, norm=norm, overwrite_x=overwrite_x)
    r_np = scipy.fft.rfftn(a, axes=axes, norm=norm, overwrite_x=overwrite_x)
    assert_same_result(r_nb, r_np)

    assert_same_result(
        ScipyFFT.irfftn(r_nb, axes=axes, norm=norm, overwrite_x=overwrite_x),
        scipy.fft.irfftn(r_np, axes=axes, norm=norm, overwrite_x=overwrite_x),
    )


# -----------------------------------------------------------------------------
# HFFT / IHFFT
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [None, 16])
@pytest.mark.parametrize("axis", [-1])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_hfft_ihfft(n, axis, norm, overwrite_x):
    rng = np.random.default_rng(6)
    a = rng.standard_normal(16) + 1j * rng.standard_normal(16)

    r_nb = ScipyFFT.hfft(a, n, axis, norm, overwrite_x)
    r_np = scipy.fft.hfft(a, n, axis, norm, overwrite_x)
    assert_same_result(r_nb, r_np)

    assert_same_result(
        ScipyFFT.ihfft(r_nb, n, axis, norm, overwrite_x),
        scipy.fft.ihfft(r_np, n, axis, norm, overwrite_x),
    )


# -----------------------------------------------------------------------------
# Frequencies
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 8, 16])
@pytest.mark.parametrize("d", [1.0, 0.1])
@pytest.mark.parametrize("xp", [None])
@pytest.mark.parametrize("device", [None])
def test_fftfreq(n, d, xp, device):
    assert_same_result(
        ScipyFFT.fftfreq(n, d, xp=xp, device=device),
        scipy.fft.fftfreq(n, d, xp=xp, device=device),
    )


@pytest.mark.parametrize("n", [1, 8, 16])
@pytest.mark.parametrize("d", [1.0, 0.1])
@pytest.mark.parametrize("xp", [None])
@pytest.mark.parametrize("device", [None])
def test_rfftfreq(n, d, xp, device):
    assert_same_result(
        ScipyFFT.rfftfreq(n, d, xp=xp, device=device),
        scipy.fft.rfftfreq(n, d, xp=xp, device=device),
    )


# -----------------------------------------------------------------------------
# FFTSHIFT / IFFTSHIFT
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(8,), (4, 6), (3, 4, 5)])
@pytest.mark.parametrize("axes", [None, 0, -1, (0, -1)])
def test_fftshift_ifftshift(shape, axes):
    rng = np.random.default_rng(7)
    a = rng.standard_normal(shape)

    s_nb = ScipyFFT.fftshift(a, axes)
    s_np = scipy.fft.fftshift(a, axes)
    assert_same_result(s_nb, s_np)

    assert_same_result(
        ScipyFFT.ifftshift(s_nb, axes),
        scipy.fft.ifftshift(s_np, axes),
    )


# -----------------------------------------------------------------------------
# next_fast_len
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 8, 16, 31])
@pytest.mark.parametrize("real", [False, True])
def test_next_fast_len(n, real):
    assert ScipyFFT.next_fast_len(n, real) == scipy.fft.next_fast_len(n, real)


# -----------------------------------------------------------------------------
# DCT / DST families
# -----------------------------------------------------------------------------
DCT_DST_TYPES = [1, 2, 3, 4]
NORMS = [None, "backward", "ortho", "forward"]
ORTHO = [None, False, True]
OVERWRITE = [False, True]

AXES_1D = [0, -1]
AXES_ND = [None, (0,), (-1,), (0, 1)]

SHAPES_1D = [8, 16]
SHAPES_ND = [(4, 5), (3, 4, 5)]


@pytest.mark.parametrize("n", SHAPES_1D + [None])
@pytest.mark.parametrize("axis", AXES_1D)
@pytest.mark.parametrize("type", DCT_DST_TYPES)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("orthogonalize", ORTHO)
@pytest.mark.parametrize("overwrite_x", OVERWRITE)
def test_dct_idct(n, axis, type, norm, orthogonalize, overwrite_x):
    rng = np.random.default_rng(8)
    a = rng.standard_normal(32)
    a0 = a.copy()

    r_nb = ScipyFFT.dct(
        a,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )
    r_np = scipy.fft.dct(
        a0,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    assert_same_result(r_nb, r_np)

    out_nb = ScipyFFT.idct(
        r_nb,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )
    out_np = scipy.fft.idct(
        r_np,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    assert_same_result(out_nb, out_np)


@pytest.mark.parametrize("n", SHAPES_1D + [None])
@pytest.mark.parametrize("axis", AXES_1D)
@pytest.mark.parametrize("type", DCT_DST_TYPES)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("orthogonalize", ORTHO)
@pytest.mark.parametrize("overwrite_x", OVERWRITE)
def test_dst_idst(n, axis, type, norm, orthogonalize, overwrite_x):
    rng = np.random.default_rng(9)
    a = rng.standard_normal(32)
    a0 = a.copy()

    r_nb = ScipyFFT.dst(
        a,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    r_np = scipy.fft.dst(
        a0,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    # TODO: Remove this if the bug is fixed
    if norm is None and orthogonalize is None:
        assert_same_result(r_nb, r_np)

    out_nb = ScipyFFT.idst(
        r_nb,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    out_np = scipy.fft.idst(
        r_np,
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    # TODO: Remove this if the bug is fixed
    if norm is None and orthogonalize is None:
        assert_same_result(out_nb, out_np)


@pytest.mark.parametrize("shape", SHAPES_ND)
@pytest.mark.parametrize("axes", AXES_ND)
@pytest.mark.parametrize("s", [None])
@pytest.mark.parametrize("type", DCT_DST_TYPES)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("orthogonalize", ORTHO)
@pytest.mark.parametrize("overwrite_x", OVERWRITE)
def test_dctn_idctn(shape, axes, s, type, norm, orthogonalize, overwrite_x):
    rng = np.random.default_rng(10)
    a = rng.standard_normal(shape)
    a0 = a.copy()

    r_nb = ScipyFFT.dctn(
        a,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )
    r_np = scipy.fft.dctn(
        a0,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    assert_same_result(r_nb, r_np)

    out_nb = ScipyFFT.idctn(
        r_nb,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )
    out_np = scipy.fft.idctn(
        r_np,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    assert_same_result(out_nb, out_np)


@pytest.mark.parametrize("shape", SHAPES_ND)
@pytest.mark.parametrize("axes", AXES_ND)
@pytest.mark.parametrize("s", [None])
@pytest.mark.parametrize("type", DCT_DST_TYPES)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("orthogonalize", ORTHO)
@pytest.mark.parametrize("overwrite_x", OVERWRITE)
def test_dstn_idstn(shape, axes, s, type, norm, orthogonalize, overwrite_x):
    rng = np.random.default_rng(11)
    a = rng.standard_normal(shape)
    a0 = a.copy()

    r_nb = ScipyFFT.dstn(
        a,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    r_np = scipy.fft.dstn(
        a0,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    # TODO: Remove this if the bug is fixed
    if norm is None and orthogonalize is None:
        assert_same_result(r_nb, r_np)

    out_nb = ScipyFFT.idstn(
        r_nb,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    out_np = scipy.fft.idstn(
        r_np,
        type=type,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=1,
        orthogonalize=orthogonalize,
    )

    # TODO: Remove this if the bug is fixed
    if norm is None and orthogonalize is None:
        assert_same_result(out_nb, out_np)


# -----------------------------------------------------------------------------
# FHT
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [8, 16, 32, 64])
@pytest.mark.parametrize("dln", [0.05, 0.1])
@pytest.mark.parametrize("mu", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("offset", [0.0, 0.3])
@pytest.mark.parametrize("bias", [0.0, 0.7])
def test_fht_ifht(n, dln, mu, offset, bias):
    rng = np.random.default_rng(12)
    a = rng.standard_normal(n)

    r_nb = ScipyFFT.fht(a, dln, mu, offset=offset, bias=bias)
    r_np = scipy.fft.fht(a, dln, mu, offset=offset, bias=bias)

    assert_same_result(r_nb, r_np)

    out_nb = ScipyFFT.ifht(r_nb, dln, mu, offset=offset, bias=bias)
    out_np = scipy.fft.ifht(r_np, dln, mu, offset=offset, bias=bias)

    assert_same_result(out_nb, out_np)


@pytest.mark.parametrize("dln", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("mu", [0.0, 0.5, 1.0, 2.5])
@pytest.mark.parametrize("initial", [0.0, 0.3])
@pytest.mark.parametrize("bias", [0.0, 0.8])
def test_fhtoffset(dln, mu, initial, bias):
    out_nb = ScipyFFT.fhtoffset(dln, mu, initial=initial, bias=bias)
    out_np = scipy.fft.fhtoffset(dln, mu, initial=initial, bias=bias)

    assert out_nb == out_np
