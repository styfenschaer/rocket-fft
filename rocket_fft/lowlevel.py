"""Low-level interface to DUCC0's transforms.

These functions can only be called from jitted code. They write into a
caller-provided output array and apply exactly the scale factor ``fct``,
without any normalization logic. Transform lengths are taken from the output
array: where input and output differ in size along an axis, the input is
zero-padded or truncated. Duplicate axes are rejected.
"""

from __future__ import annotations

from numba.core import types
from numba.core.errors import TypingError
from numba.extending import intrinsic

from . import _codegen as cg
from ._native import Flag, Kind, Norm

_REAL = (types.float32, types.float64)
_COMPLEX = (types.complex64, types.complex128)


def _check_arrays(ain, aout, axes, in_dtypes, out_dtypes) -> None:
    if not isinstance(ain, types.Array) or ain.dtype not in in_dtypes:
        raise TypingError(f"'ain' must be an array with dtype {' or '.join(map(str, in_dtypes))}.")
    if not isinstance(aout, types.Array) or aout.dtype not in out_dtypes:
        raise TypingError(f"'aout' must be an array with dtype {' or '.join(map(str, out_dtypes))}.")
    if ain.ndim != aout.ndim:
        raise TypingError("'ain' and 'aout' must have the same number of dimensions.")
    if not aout.mutable:
        raise TypingError("'aout' must be writeable.")
    if not (isinstance(axes, types.Array) and axes.ndim == 1
            and isinstance(axes.dtype, (types.Integer, types.Float))):
        raise TypingError("'axes' must be a one-dimensional array of integers.")


def _precision(dtype: types.Type) -> int:
    return 32 if dtype in (types.float32, types.complex64) else 64


def _check_scalar(value, name: str, kind: type, exclude: type | None = None) -> None:
    if not isinstance(value, kind) or (exclude is not None and isinstance(value, exclude)):
        raise TypingError(f"Invalid type {value} for argument '{name}'.")


def _flag(builder, flag: Flag, ty, value):
    """``flag`` if the boolean argument is true, else 0."""
    return builder.mul(cg.to_i64(builder, ty, value), cg.const(flag))


def _execute(context, builder, kind: Kind, sig, args, *, fct, nthreads,
             flags=Flag.NONE, runtime_flags=None, r2r_type=None, ortho=None):
    """Emit a call to ``rf_execute``. ``fct``, ``nthreads``, ``r2r_type`` and ``ortho`` are ``(type, value)`` pairs."""
    ain_t, aout_t, axes_t = sig.args[:3]
    ain, aout, axes = args[:3]
    flag_value = cg.const(flags | Flag.LENGTHS_FROM_OUTPUT | Flag.UNIQUE_AXES)
    if runtime_flags is not None:
        flag_value = builder.or_(flag_value, runtime_flags)
    opts = cg.options(
        builder,
        kind=int(kind),
        flags=flag_value,
        in_dtype=cg.dtype_code(ain_t.dtype),
        out_dtype=cg.dtype_code(aout_t.dtype),
        norm=int(Norm.EXPLICIT),
        fct=cg.to_f64(builder, *fct),
        r2r_type=cg.to_i64(builder, *r2r_type) if r2r_type else cg.const(0),
        ortho=cg.to_i64(builder, *ortho) if ortho else cg.const(0),
        workers=cg.to_i64(builder, *nthreads),
        workers_given=1,
    )
    status = cg.call(builder, "rf_execute", cg.i64, [
        opts,
        cg.array_ptr(context, builder, ain_t, ain),
        cg.array_ptr(context, builder, aout_t, aout),
        cg.const(ain_t.ndim),
        cg.int_seq(context, builder, types.none, None),
        cg.int_seq(context, builder, axes_t, axes),
    ])
    cg.check_status(context, builder, status)
    return context.get_dummy_value()


def _make_complex_transform(name: str, kind: Kind, in_dtypes, out_dtypes, doc: str):
    def impl(typingctx, ain, aout, axes, forward, fct, nthreads):
        _check_arrays(ain, aout, axes, in_dtypes, out_dtypes)
        if _precision(ain.dtype) != _precision(aout.dtype):
            raise TypingError("'ain' and 'aout' must have the same precision.")
        _check_scalar(forward, "forward", types.Boolean)
        _check_scalar(fct, "fct", types.Float)
        _check_scalar(nthreads, "nthreads", types.Integer, exclude=types.Boolean)
        sig = types.none(ain, aout, axes, forward, fct, nthreads)

        def codegen(context, builder, sig, args):
            return _execute(context, builder, kind, sig, args,
                            runtime_flags=_flag(builder, Flag.FORWARD, sig.args[3], args[3]),
                            fct=(sig.args[4], args[4]), nthreads=(sig.args[5], args[5]))

        return sig, codegen

    impl.__name__ = name
    impl.__doc__ = doc
    return intrinsic(impl)


def _make_r2r_transform(name: str, kind: Kind, doc: str):
    def impl(typingctx, ain, aout, axes, type, fct, ortho, nthreads):
        _check_arrays(ain, aout, axes, _REAL, (ain.dtype,) if isinstance(ain, types.Array) else _REAL)
        _check_scalar(type, "type", types.Integer, exclude=types.Boolean)
        _check_scalar(fct, "fct", types.Float)
        _check_scalar(ortho, "ortho", types.Boolean)
        _check_scalar(nthreads, "nthreads", types.Integer, exclude=types.Boolean)
        sig = types.none(ain, aout, axes, type, fct, ortho, nthreads)

        def codegen(context, builder, sig, args):
            return _execute(context, builder, kind, sig, args, flags=Flag.FORWARD,
                            r2r_type=(sig.args[3], args[3]), fct=(sig.args[4], args[4]),
                            ortho=(sig.args[5], args[5]), nthreads=(sig.args[6], args[6]))

        return sig, codegen

    impl.__name__ = name
    impl.__doc__ = doc
    return intrinsic(impl)


def _make_hartley(name: str, kind: Kind, doc: str):
    def impl(typingctx, ain, aout, axes, fct, nthreads):
        _check_arrays(ain, aout, axes, _REAL, (ain.dtype,) if isinstance(ain, types.Array) else _REAL)
        _check_scalar(fct, "fct", types.Float)
        _check_scalar(nthreads, "nthreads", types.Integer, exclude=types.Boolean)
        sig = types.none(ain, aout, axes, fct, nthreads)

        def codegen(context, builder, sig, args):
            return _execute(context, builder, kind, sig, args,
                            fct=(sig.args[3], args[3]), nthreads=(sig.args[4], args[4]))

        return sig, codegen

    impl.__name__ = name
    impl.__doc__ = doc
    return intrinsic(impl)


@intrinsic
def r2r_fftpack(typingctx, ain, aout, axes, real2hermitian, forward, fct, nthreads):
    """FFTPACK-style real-to-real transform (``r2r_fftpack(ain, aout, axes, real2hermitian, forward, fct, nthreads)``)."""
    _check_arrays(ain, aout, axes, _REAL, (ain.dtype,) if isinstance(ain, types.Array) else _REAL)
    _check_scalar(real2hermitian, "real2hermitian", types.Boolean)
    _check_scalar(forward, "forward", types.Boolean)
    _check_scalar(fct, "fct", types.Float)
    _check_scalar(nthreads, "nthreads", types.Integer, exclude=types.Boolean)
    sig = types.none(ain, aout, axes, real2hermitian, forward, fct, nthreads)

    def codegen(context, builder, sig, args):
        runtime_flags = builder.or_(
            _flag(builder, Flag.REAL2HERMITIAN, sig.args[3], args[3]),
            _flag(builder, Flag.FORWARD, sig.args[4], args[4]),
        )
        return _execute(context, builder, Kind.FFTPACK, sig, args, runtime_flags=runtime_flags,
                        fct=(sig.args[5], args[5]), nthreads=(sig.args[6], args[6]))

    return sig, codegen


c2c = _make_complex_transform(
    "c2c", Kind.C2C, _COMPLEX, _COMPLEX,
    "Complex-to-complex FFT (``c2c(ain, aout, axes, forward, fct, nthreads)``).",
)
c2c_sym = _make_complex_transform(
    "c2c_sym", Kind.C2C, _REAL, _COMPLEX,
    "Complex FFT of real input, exploiting Hermitian symmetry (``c2c_sym(ain, aout, axes, forward, fct, nthreads)``).",
)
r2c = _make_complex_transform(
    "r2c", Kind.R2C, _REAL, _COMPLEX,
    "Real-to-complex FFT (``r2c(ain, aout, axes, forward, fct, nthreads)``).",
)
c2r = _make_complex_transform(
    "c2r", Kind.C2R, _COMPLEX, _REAL,
    "Complex-to-real FFT (``c2r(ain, aout, axes, forward, fct, nthreads)``).",
)
dct = _make_r2r_transform(
    "dct", Kind.DCT, "Discrete cosine transform (``dct(ain, aout, axes, type, fct, ortho, nthreads)``).",
)
dst = _make_r2r_transform(
    "dst", Kind.DST, "Discrete sine transform (``dst(ain, aout, axes, type, fct, ortho, nthreads)``).",
)
r2r_separable_hartley = _make_hartley(
    "r2r_separable_hartley", Kind.SEPARABLE_HARTLEY,
    "Separable Hartley transform (``r2r_separable_hartley(ain, aout, axes, fct, nthreads)``).",
)
r2r_genuine_hartley = _make_hartley(
    "r2r_genuine_hartley", Kind.GENUINE_HARTLEY,
    "Genuine multidimensional Hartley transform (``r2r_genuine_hartley(ain, aout, axes, fct, nthreads)``).",
)

separable_hartley = r2r_separable_hartley
genuine_hartley = r2r_genuine_hartley
fftpack = r2r_fftpack
