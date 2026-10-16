"""Functionality beyond ``numpy.fft``/``scipy.fft`` kept from earlier releases.

* ``numpy.roll`` with NumPy's full signature ``roll(a, shift, axis=None)``.
  Numba itself only supports ``roll(a, shift)`` with an integer shift.
* ``loggamma`` and ``poch`` (exposed as ``rocket_fft.special``): ufuncs that
  work in Python and jitted code, on scalars and arrays.

All computation happens in the native library (see the "Extras" section of
``_rocket_fft_numba.cpp``); this module only lowers the calls.
"""

from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import vectorize
from numba.core import cgutils, types
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload

from . import _codegen as cg
from ._dispatch import implement
from ._signatures import IntSeq, Param, Signature

# -----------------------------------------------------------------------------
# numpy.roll
# -----------------------------------------------------------------------------


class _ArrayLike(Param):
    def check(self, ty):
        ok = isinstance(ty, (types.Array, types.Number, types.Boolean, types.BaseTuple, types.List, types.ListType))
        return None if ok else "array-like"


class _Shift(IntSeq):
    def check(self, ty):
        return "an integer or a sequence of integers" if cg.is_none(ty) else super().check(ty)


ROLL = Signature(a=_ArrayLike(), shift=_Shift(), axis=IntSeq())


def _roll_stub(a, shift, axis=None): ...


def _roll_kernel(typingctx, a, shift, axis):
    if not isinstance(a, types.Array):
        raise TypingError("array-like inputs are converted by the fallback overload")
    # Rolling the flattened array (axis=None) always yields C order.
    layout = "C" if cg.is_none(axis) else cg.like_input_layout(a)
    ret = types.Array(a.dtype, a.ndim, layout)
    sig = ret(a, shift, axis)

    def codegen(context, builder, sig, args):
        (a_t, shift_t, axis_t), (a_v, shift_v, axis_v) = sig.args, args
        return cg.call_returning_array(context, builder, "rf_roll", ret, [
            cg.array_ptr(context, builder, a_t, a_v),
            cg.const(a_t.ndim),
            cg.int_seq(context, builder, shift_t, shift_v),
            cg.int_seq(context, builder, axis_t, axis_v),
        ])

    return sig, codegen


implement(np.roll, _roll_stub, _roll_kernel, validate=ROLL.validate)


@overload(np.roll)
def _roll_array_like(a, shift, axis=None):
    """Fallback for scalars, tuples and lists: convert to an array first."""
    if isinstance(a, types.Array):
        return None
    ROLL.validate(a, shift, axis)
    return lambda a, shift, axis=None: np.roll(np.asarray(a), shift, axis)


# -----------------------------------------------------------------------------
# Special functions
# -----------------------------------------------------------------------------

def _check_real(name: str, ty: types.Type) -> None:
    if not isinstance(ty, (types.Float, types.Integer, types.Boolean)):
        raise TypingError(f"Argument '{name}' must be a real number.")


@intrinsic
def _loggamma(typingctx, z):
    if isinstance(z, types.Complex):
        def codegen(context, builder, sig, args):
            re, im = cg.to_c128(context, builder, sig.args[0], args[0])
            out = cgutils.alloca_once(builder, cg.f64, size=2)
            cg.call(builder, "rf_loggamma_complex", ir.VoidType(), [re, im, out])
            result = context.make_complex(builder, types.complex128)
            result.real = builder.load(out)
            result.imag = builder.load(builder.gep(out, [cg.const(1)]))
            return result._getvalue()

        return types.complex128(z), codegen

    _check_real("z", z)

    def codegen(context, builder, sig, args):
        return cg.call(builder, "rf_loggamma", cg.f64, [cg.to_f64(builder, sig.args[0], args[0])])

    return types.float64(z), codegen


@intrinsic
def _poch(typingctx, z, m):
    _check_real("z", z)
    _check_real("m", m)

    def codegen(context, builder, sig, args):
        scalars = [cg.to_f64(builder, t, v) for t, v in zip(sig.args, args)]
        return cg.call(builder, "rf_poch", cg.f64, scalars)

    return types.float64(z, m), codegen


# Signatures are compiled on demand, for whatever input types are used.
@vectorize
def loggamma(z):
    """As ``scipy.special.loggamma``."""
    return _loggamma(z)


@vectorize
def poch(z, m):
    """As ``scipy.special.poch``."""
    return _poch(z, m)
