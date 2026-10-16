"""Implementations of ``scipy.fft`` (SciPy semantics: duplicate axes are rejected)."""

from __future__ import annotations

import numpy.fft
import scipy.fft

from . import _kernels
from . import _signatures as sig
from ._dispatch import implement
from ._native import Api, Flag, Kind
from ._numpy_fft import fftshift_stub

# Stubs defining how arguments bind (names, order and defaults).
def fft_1d_stub(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None): ...
def fft_2d_stub(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, plan=None): ...
def fft_nd_stub(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None): ...
def r2r_1d_stub(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None): ...
def r2r_nd_stub(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None): ...
def fht_stub(a, dln, mu, offset=0.0, bias=0.0): ...
def ifht_stub(A, dln, mu, offset=0.0, bias=0.0): ...
def fhtoffset_stub(dln, mu, initial=0.0, bias=0.0): ...
def fftfreq_stub(n, d=1.0, xp=None, device=None): ...
def fast_len_stub(target, real=False): ...


_FFT_STUBS = {"1d": fft_1d_stub, "2d": fft_2d_stub, "nd": fft_nd_stub}


def _flags(forward: bool, one_dim: bool) -> Flag:
    flags = Flag.SCIPY | Flag.UNIQUE_AXES | (Flag.FORWARD if forward else Flag.NONE)
    return flags | Flag.DEFAULT_LAST_AXIS if one_dim else flags


def _fft(func, dims: str, kind: Kind, forward: bool, real_input: bool = False) -> None:
    n, axis = ("n", "axis") if dims == "1d" else ("s", "axes")
    args = ("x", n, axis, "norm", None, "overwrite_x", "workers", None, None)
    signature = sig.scipy_fft(nd=dims != "1d", real_input=real_input)
    kernel = _kernels.transform(kind, _flags(forward, one_dim=dims == "1d"), Api.SCIPY)
    implement(func, _FFT_STUBS[dims], kernel, args, signature.validate)


def _r2r(func, nd: bool, kind: Kind, forward: bool) -> None:
    n, axis = ("s", "axes") if nd else ("n", "axis")
    args = ("x", n, axis, "norm", None, "overwrite_x", "workers", "type", "orthogonalize")
    kernel = _kernels.transform(kind, _flags(forward, one_dim=not nd), Api.SCIPY)
    implement(func, r2r_nd_stub if nd else r2r_1d_stub, kernel, args, sig.scipy_r2r(nd).validate)


_fft(scipy.fft.fft, "1d", Kind.C2C, forward=True)
_fft(scipy.fft.ifft, "1d", Kind.C2C, forward=False)
_fft(scipy.fft.rfft, "1d", Kind.R2C, forward=True, real_input=True)
_fft(scipy.fft.irfft, "1d", Kind.C2R, forward=False)
_fft(scipy.fft.hfft, "1d", Kind.C2R, forward=True)
_fft(scipy.fft.ihfft, "1d", Kind.R2C, forward=False, real_input=True)

_fft(scipy.fft.fft2, "2d", Kind.C2C, forward=True)
_fft(scipy.fft.ifft2, "2d", Kind.C2C, forward=False)
_fft(scipy.fft.rfft2, "2d", Kind.R2C, forward=True, real_input=True)
_fft(scipy.fft.irfft2, "2d", Kind.C2R, forward=False)
_fft(scipy.fft.hfft2, "2d", Kind.C2R, forward=True)
_fft(scipy.fft.ihfft2, "2d", Kind.R2C, forward=False, real_input=True)

_fft(scipy.fft.fftn, "nd", Kind.C2C, forward=True)
_fft(scipy.fft.ifftn, "nd", Kind.C2C, forward=False)
_fft(scipy.fft.rfftn, "nd", Kind.R2C, forward=True, real_input=True)
_fft(scipy.fft.irfftn, "nd", Kind.C2R, forward=False)
_fft(scipy.fft.hfftn, "nd", Kind.C2R, forward=True)
_fft(scipy.fft.ihfftn, "nd", Kind.R2C, forward=False, real_input=True)

_r2r(scipy.fft.dct, nd=False, kind=Kind.DCT, forward=True)
_r2r(scipy.fft.idct, nd=False, kind=Kind.DCT, forward=False)
_r2r(scipy.fft.dst, nd=False, kind=Kind.DST, forward=True)
_r2r(scipy.fft.idst, nd=False, kind=Kind.DST, forward=False)
_r2r(scipy.fft.dctn, nd=True, kind=Kind.DCT, forward=True)
_r2r(scipy.fft.idctn, nd=True, kind=Kind.DCT, forward=False)
_r2r(scipy.fft.dstn, nd=True, kind=Kind.DST, forward=True)
_r2r(scipy.fft.idstn, nd=True, kind=Kind.DST, forward=False)

implement(scipy.fft.fht, fht_stub, _kernels.fht(inverse=False), validate=sig.FHT.validate)
implement(scipy.fft.ifht, ifht_stub, _kernels.fht(inverse=True), validate=sig.IFHT.validate)
implement(scipy.fft.fhtoffset, fhtoffset_stub, _kernels.fhtoffset, validate=sig.FHTOFFSET.validate)

# Older SciPy versions re-export these helpers from numpy.fft; the NumPy
# implementations then already cover them.
_FREQ_ARGS = ("n", "d", "device")
if scipy.fft.fftfreq is not numpy.fft.fftfreq:
    implement(scipy.fft.fftfreq, fftfreq_stub, _kernels.freq(rfft=False), _FREQ_ARGS, sig.SCIPY_FFTFREQ.validate)
if scipy.fft.rfftfreq is not numpy.fft.rfftfreq:
    implement(scipy.fft.rfftfreq, fftfreq_stub, _kernels.freq(rfft=True), _FREQ_ARGS, sig.SCIPY_FFTFREQ.validate)
if scipy.fft.fftshift is not numpy.fft.fftshift:
    implement(scipy.fft.fftshift, fftshift_stub, _kernels.fftshift(inverse=False), validate=sig.FFTSHIFT.validate)
if scipy.fft.ifftshift is not numpy.fft.ifftshift:
    implement(scipy.fft.ifftshift, fftshift_stub, _kernels.fftshift(inverse=True), validate=sig.FFTSHIFT.validate)

implement(scipy.fft.next_fast_len, fast_len_stub, _kernels.fast_len("rf_good_size"), validate=sig.FAST_LEN.validate)
if hasattr(scipy.fft, "prev_fast_len"):  # SciPy >= 1.15
    implement(scipy.fft.prev_fast_len, fast_len_stub, _kernels.fast_len("rf_prev_good_size"),
              validate=sig.FAST_LEN.validate)
