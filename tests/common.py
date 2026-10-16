"""Jitted wrappers around the numpy.fft and scipy.fft APIs, and shared assertions.

Each wrapper mirrors its library's signature and defaults, so a test can call
``NumpyFFT.fft(...)`` exactly like ``np.fft.fft(...)`` and compare the results.
"""

import numba as nb
import numpy as np
import scipy.fft

NORMS = [None, "backward", "ortho", "forward"]


def random_array(shape, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def assert_parity(got, ref, rtol=1e-12, atol=1e-12):
    """Same dtype, shape, memory layout and values (within tolerance) as the reference result."""
    assert got.dtype == ref.dtype, f"dtype: {got.dtype} != {ref.dtype}"
    assert got.shape == ref.shape, f"shape: {got.shape} != {ref.shape}"
    if got.size:  # strides of an empty array are arbitrary
        assert got.strides == ref.strides, f"strides: {got.strides} != {ref.strides}"
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)


def assert_new_array(out, inp, inp_before):
    """The result is a fresh array and the input was left untouched."""
    assert out is not inp
    assert not np.shares_memory(out, inp)
    np.testing.assert_array_equal(inp, inp_before)


class NumpyFFT:
    # fmt: off
    fft = staticmethod(nb.njit(lambda a, n=None, axis=-1, norm=None, out=None: np.fft.fft(a, n, axis, norm, out)))
    ifft = staticmethod(nb.njit(lambda a, n=None, axis=-1, norm=None, out=None: np.fft.ifft(a, n, axis, norm, out)))
    fft2 = staticmethod(nb.njit(lambda a, s=None, axes=(-2, -1), norm=None, out=None: np.fft.fft2(a, s, axes, norm, out)))
    ifft2 = staticmethod(nb.njit(lambda a, s=None, axes=(-2, -1), norm=None, out=None: np.fft.ifft2(a, s, axes, norm, out)))
    fftn = staticmethod(nb.njit(lambda a, s=None, axes=None, norm=None, out=None: np.fft.fftn(a, s, axes, norm, out)))
    ifftn = staticmethod(nb.njit(lambda a, s=None, axes=None, norm=None, out=None: np.fft.ifftn(a, s, axes, norm, out)))
    rfft = staticmethod(nb.njit(lambda a, n=None, axis=-1, norm=None, out=None: np.fft.rfft(a, n, axis, norm, out)))
    irfft = staticmethod(nb.njit(lambda a, n=None, axis=-1, norm=None, out=None: np.fft.irfft(a, n, axis, norm, out)))
    rfft2 = staticmethod(nb.njit(lambda a, s=None, axes=(-2, -1), norm=None, out=None: np.fft.rfft2(a, s, axes, norm, out)))
    irfft2 = staticmethod(nb.njit(lambda a, s=None, axes=(-2, -1), norm=None, out=None: np.fft.irfft2(a, s, axes, norm, out)))
    rfftn = staticmethod(nb.njit(lambda a, s=None, axes=None, norm=None, out=None: np.fft.rfftn(a, s, axes, norm, out)))
    irfftn = staticmethod(nb.njit(lambda a, s=None, axes=None, norm=None, out=None: np.fft.irfftn(a, s, axes, norm, out)))
    hfft = staticmethod(nb.njit(lambda a, n=None, axis=-1, norm=None, out=None: np.fft.hfft(a, n, axis, norm, out)))
    ihfft = staticmethod(nb.njit(lambda a, n=None, axis=-1, norm=None, out=None: np.fft.ihfft(a, n, axis, norm, out)))
    fftfreq = staticmethod(nb.njit(lambda n, d=1.0, device=None: np.fft.fftfreq(n, d, device)))
    rfftfreq = staticmethod(nb.njit(lambda n, d=1.0, device=None: np.fft.rfftfreq(n, d, device)))
    fftshift = staticmethod(nb.njit(lambda x, axes=None: np.fft.fftshift(x, axes)))
    ifftshift = staticmethod(nb.njit(lambda x, axes=None: np.fft.ifftshift(x, axes)))
    # fmt: on


class ScipyFFT:
    # fmt: off
    fft = staticmethod(nb.njit(lambda x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.fft(x, n, axis, norm, overwrite_x, workers, plan=plan)))
    ifft = staticmethod(nb.njit(lambda x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.ifft(x, n, axis, norm, overwrite_x, workers, plan=plan)))
    fft2 = staticmethod(nb.njit(lambda x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.fft2(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    ifft2 = staticmethod(nb.njit(lambda x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.ifft2(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    fftn = staticmethod(nb.njit(lambda x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.fftn(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    ifftn = staticmethod(nb.njit(lambda x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.ifftn(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    rfft = staticmethod(nb.njit(lambda x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.rfft(x, n, axis, norm, overwrite_x, workers, plan=plan)))
    irfft = staticmethod(nb.njit(lambda x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.irfft(x, n, axis, norm, overwrite_x, workers, plan=plan)))
    rfft2 = staticmethod(nb.njit(lambda x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.rfft2(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    irfft2 = staticmethod(nb.njit(lambda x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.irfft2(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    rfftn = staticmethod(nb.njit(lambda x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.rfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    irfftn = staticmethod(nb.njit(lambda x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.irfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    hfft = staticmethod(nb.njit(lambda x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.hfft(x, n, axis, norm, overwrite_x, workers, plan=plan)))
    ihfft = staticmethod(nb.njit(lambda x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.ihfft(x, n, axis, norm, overwrite_x, workers, plan=plan)))
    hfft2 = staticmethod(nb.njit(lambda x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.hfft2(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    ihfft2 = staticmethod(nb.njit(lambda x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.ihfft2(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    hfftn = staticmethod(nb.njit(lambda x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.hfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    ihfftn = staticmethod(nb.njit(lambda x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None: scipy.fft.ihfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)))
    dct = staticmethod(nb.njit(lambda x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.dct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)))
    idct = staticmethod(nb.njit(lambda x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.idct(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)))
    dst = staticmethod(nb.njit(lambda x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.dst(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)))
    idst = staticmethod(nb.njit(lambda x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.idst(x, type, n, axis, norm, overwrite_x, workers, orthogonalize)))
    dctn = staticmethod(nb.njit(lambda x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.dctn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)))
    idctn = staticmethod(nb.njit(lambda x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.idctn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)))
    dstn = staticmethod(nb.njit(lambda x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.dstn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)))
    idstn = staticmethod(nb.njit(lambda x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None: scipy.fft.idstn(x, type, s, axes, norm, overwrite_x, workers, orthogonalize)))
    fht = staticmethod(nb.njit(lambda a, dln, mu, offset=0.0, bias=0.0: scipy.fft.fht(a, dln, mu, offset, bias)))
    ifht = staticmethod(nb.njit(lambda A, dln, mu, offset=0.0, bias=0.0: scipy.fft.ifht(A, dln, mu, offset, bias)))
    fhtoffset = staticmethod(nb.njit(lambda dln, mu, initial=0.0, bias=0.0: scipy.fft.fhtoffset(dln, mu, initial, bias)))
    fftfreq = staticmethod(nb.njit(lambda n, d=1.0, xp=None, device=None: scipy.fft.fftfreq(n, d, xp=xp, device=device)))
    rfftfreq = staticmethod(nb.njit(lambda n, d=1.0, xp=None, device=None: scipy.fft.rfftfreq(n, d, xp=xp, device=device)))
    fftshift = staticmethod(nb.njit(lambda x, axes=None: scipy.fft.fftshift(x, axes)))
    ifftshift = staticmethod(nb.njit(lambda x, axes=None: scipy.fft.ifftshift(x, axes)))
    next_fast_len = staticmethod(nb.njit(lambda target, real=False: scipy.fft.next_fast_len(target, real)))
    if hasattr(scipy.fft, "prev_fast_len"):  # SciPy >= 1.15
        prev_fast_len = staticmethod(nb.njit(lambda target, real=False: scipy.fft.prev_fast_len(target, real)))
    # fmt: on
