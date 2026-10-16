"""Compile-time argument validation for the ``numpy.fft``/``scipy.fft`` overloads.

Every public function declares a :class:`Signature`; validating it raises a
``TypingError`` that names the offending argument and its position.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import typeof
from numba.core import types
from numba.core.errors import TypingError

from ._codegen import is_none
from ._kernels import complex_dtype


def _unliteral(ty) -> types.Type:
    """Normalize an argument type; omitted arguments may arrive as their raw default value."""
    if isinstance(ty, types.Omitted):
        ty = ty.value
    if not isinstance(ty, types.Type):
        ty = types.none if ty is None else typeof(ty)
    return types.unliteral(ty)


def _ordinal(pos: int) -> str:
    suffix = "th" if 11 <= pos % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(pos % 10, "th")
    return f"{pos}{suffix}"


def _is_int(ty: types.Type, allow_bool: bool = True) -> bool:
    return isinstance(ty, types.Integer) or (allow_bool and isinstance(ty, types.Boolean))


class Param:
    """Base class: ``check`` returns ``None`` if valid, otherwise a description of what is expected."""

    def check(self, ty: types.Type) -> str | None:
        raise NotImplementedError


class Array(Param):
    def __init__(self, real: bool = False):
        self.real = real

    def check(self, ty):
        if not isinstance(ty, types.Array):
            return "an array"
        complex_dtype(ty.dtype)  # raises for unsupported dtypes
        if self.real and isinstance(ty.dtype, types.Complex):
            return "a real array"
        return None


class Out(Param):
    def check(self, ty):
        return None if is_none(ty) or isinstance(ty, types.Array) else "an array"


class Int(Param):
    def check(self, ty):
        return None if _is_int(ty, allow_bool=False) else "an integer"


class Bool(Param):
    def __init__(self, allow_none: bool = False):
        self.allow_none = allow_none

    def check(self, ty):
        if self.allow_none and is_none(ty):
            return None
        return None if isinstance(ty, types.Boolean) else "a boolean"


class Scalar(Param):
    def check(self, ty):
        ok = isinstance(ty, types.Number) and not isinstance(ty, types.Boolean)
        return None if ok else "a scalar"


class IntSeq(Param):
    """``None``, an integer, or a non-empty tuple, list or 1-D array of integers."""

    def __init__(self, allow_bool: bool = True):
        self.allow_bool = allow_bool

    def check(self, ty):
        if is_none(ty) or _is_int(ty, self.allow_bool):
            return None
        if isinstance(ty, types.BaseTuple):
            if len(ty) > 0 and all(_is_int(t, self.allow_bool) for t in ty):
                return None
        elif isinstance(ty, (types.Array, types.List)):
            if _is_int(ty.dtype, self.allow_bool) and getattr(ty, "ndim", 1) == 1:
                return None
        return "an integer or a sequence of integers"


class Norm(Param):
    def check(self, ty):
        return None if is_none(ty) or isinstance(ty, types.UnicodeType) else "a string"


class Workers(Param):
    def check(self, ty):
        return None if is_none(ty) or _is_int(ty) else "an integer"


class NoneOnly(Param):
    def check(self, ty):
        return None if is_none(ty) else "None"


class Device(Param):
    def check(self, ty):
        if is_none(ty) or isinstance(ty, types.UnicodeType):
            return None
        return "'cpu'"


class Namespace(Param):
    def check(self, ty):
        if is_none(ty):
            return None
        if isinstance(ty, types.Module) and ty.pymod is np:
            return None
        return "None or the NumPy module"


class Signature:
    def __init__(self, **params: Param):
        self.params = list(params.items())

    def validate(self, *args: types.Type) -> None:
        for pos, ((name, param), ty) in enumerate(zip(self.params, args), start=1):
            expected = param.check(_unliteral(ty))
            if expected is not None:
                raise TypingError(f"The {_ordinal(pos)} argument '{name}' must be {expected}.")


# -----------------------------------------------------------------------------
# Signatures of the public functions
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def numpy_fft(nd: bool, real_input: bool = False) -> Signature:
    n, axis = ("s", "axes") if nd else ("n", "axis")
    return Signature(a=Array(real_input), **{n: IntSeq(), axis: IntSeq()}, norm=Norm(), out=Out())


@lru_cache(maxsize=None)
def scipy_fft(nd: bool, real_input: bool = False) -> Signature:
    n, axis = ("s", "axes") if nd else ("n", "axis")
    return Signature(
        x=Array(real_input),
        **{n: IntSeq(), axis: IntSeq()},
        norm=Norm(),
        overwrite_x=Bool(),
        workers=Workers(),
        plan=NoneOnly(),
    )


@lru_cache(maxsize=None)
def scipy_r2r(nd: bool) -> Signature:
    n, axis = ("s", "axes") if nd else ("n", "axis")
    return Signature(
        x=Array(),
        type=Int(),
        **{n: IntSeq(), axis: IntSeq()},
        norm=Norm(),
        overwrite_x=Bool(),
        workers=Workers(),
        orthogonalize=Bool(allow_none=True),
    )


FHT = Signature(a=Array(real=True), dln=Scalar(), mu=Scalar(), offset=Scalar(), bias=Scalar())
IFHT = Signature(A=Array(real=True), dln=Scalar(), mu=Scalar(), offset=Scalar(), bias=Scalar())
FHTOFFSET = Signature(dln=Scalar(), mu=Scalar(), initial=Scalar(), bias=Scalar())
NUMPY_FFTFREQ = Signature(n=Int(), d=Scalar(), device=Device())
SCIPY_FFTFREQ = Signature(n=Int(), d=Scalar(), xp=Namespace(), device=Device())
FFTSHIFT = Signature(x=Array(), axes=IntSeq(allow_bool=False))
FAST_LEN = Signature(target=Int(), real=Bool())
