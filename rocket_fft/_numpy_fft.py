"""Implementations of ``numpy.fft`` (NumPy semantics: duplicate axes are allowed)."""

from __future__ import annotations

import numpy.fft

from . import _kernels
from . import _signatures as sig
from ._dispatch import implement
from ._native import Api, Flag, Kind

# Stubs defining how arguments bind (names, order and defaults).
def fft_1d_stub(a, n=None, axis=-1, norm=None, out=None): ...
def fft_2d_stub(a, s=None, axes=(-2, -1), norm=None, out=None): ...
def fft_nd_stub(a, s=None, axes=None, norm=None, out=None): ...
def fftfreq_stub(n, d=1.0, device=None): ...
def fftshift_stub(x, axes=None): ...


_FFT_STUBS = {"1d": fft_1d_stub, "2d": fft_2d_stub, "nd": fft_nd_stub}


def _fft(func, dims: str, kind: Kind, forward: bool, real_input: bool = False) -> None:
    flags = Flag.FORWARD if forward else Flag.NONE
    if dims == "1d":
        flags |= Flag.DEFAULT_LAST_AXIS
        args = ("a", "n", "axis", "norm", "out", None, None, None, None)
    else:
        args = ("a", "s", "axes", "norm", "out", None, None, None, None)
    signature = sig.numpy_fft(nd=dims != "1d", real_input=real_input)
    implement(func, _FFT_STUBS[dims], _kernels.transform(kind, flags, Api.NUMPY), args, signature.validate)


_fft(numpy.fft.fft, "1d", Kind.C2C, forward=True)
_fft(numpy.fft.ifft, "1d", Kind.C2C, forward=False)
_fft(numpy.fft.rfft, "1d", Kind.R2C, forward=True, real_input=True)
_fft(numpy.fft.irfft, "1d", Kind.C2R, forward=False)
_fft(numpy.fft.hfft, "1d", Kind.C2R, forward=True)
_fft(numpy.fft.ihfft, "1d", Kind.R2C, forward=False, real_input=True)

_fft(numpy.fft.fft2, "2d", Kind.C2C, forward=True)
_fft(numpy.fft.ifft2, "2d", Kind.C2C, forward=False)
_fft(numpy.fft.rfft2, "2d", Kind.R2C, forward=True, real_input=True)
_fft(numpy.fft.irfft2, "2d", Kind.C2R, forward=False)

_fft(numpy.fft.fftn, "nd", Kind.C2C, forward=True)
_fft(numpy.fft.ifftn, "nd", Kind.C2C, forward=False)
_fft(numpy.fft.rfftn, "nd", Kind.R2C, forward=True, real_input=True)
_fft(numpy.fft.irfftn, "nd", Kind.C2R, forward=False)

implement(numpy.fft.fftfreq, fftfreq_stub, _kernels.freq(rfft=False), validate=sig.NUMPY_FFTFREQ.validate)
implement(numpy.fft.rfftfreq, fftfreq_stub, _kernels.freq(rfft=True), validate=sig.NUMPY_FFTFREQ.validate)
implement(numpy.fft.fftshift, fftshift_stub, _kernels.fftshift(inverse=False), validate=sig.FFTSHIFT.validate)
implement(numpy.fft.ifftshift, fftshift_stub, _kernels.fftshift(inverse=True), validate=sig.FFTSHIFT.validate)
