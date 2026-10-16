"""Kernels behind the ``numpy.fft`` and ``scipy.fft`` implementations.

A kernel is a function ``kernel(typingctx, *arg_types) -> (signature, codegen)``
(see ``_dispatch.implement``). Each kernel lowers to a single call into the
native frontend, which validates the arguments, allocates the result and
computes it. The generated LLVM IR is therefore tiny and compiles quickly,
whatever arguments a function is called with.
"""

from __future__ import annotations

from numba.core import types
from numba.core.errors import TypingError

from . import _codegen as cg
from ._codegen import is_none
from ._native import Api, Flag, Kind

# -----------------------------------------------------------------------------
# Dtype rules
# -----------------------------------------------------------------------------

_SUPPORTED_DTYPES = (
    types.boolean,
    types.int8, types.int16, types.int32, types.int64,
    types.uint8, types.uint16, types.uint32, types.uint64,
    types.float32, types.float64,
    types.complex64, types.complex128,
)


def complex_dtype(dtype: types.Type) -> types.Type:
    """Complex dtype a transform of ``dtype`` computes in (single precision is preserved)."""
    if dtype not in _SUPPORTED_DTYPES:
        raise TypingError(f"Unsupported dtype {dtype}.")
    if dtype in (types.float32, types.complex64):
        return types.complex64
    return types.complex128


def real_dtype(dtype: types.Type) -> types.Type:
    return complex_dtype(dtype).underlying_float


def result_dtype(kind: Kind, dtype: types.Type) -> types.Type:
    if kind in (Kind.C2C, Kind.R2C):
        return complex_dtype(dtype)
    if kind is Kind.C2R:
        return real_dtype(dtype)
    # Real-to-real transforms keep complex inputs complex.
    return complex_dtype(dtype) if isinstance(dtype, types.Complex) else real_dtype(dtype)


# -----------------------------------------------------------------------------
# FFT, DCT and DST
# -----------------------------------------------------------------------------

def _overwrite_states(overwrite_x: types.Type) -> set:
    """Possible run-time values of ``overwrite_x``."""
    if isinstance(overwrite_x, types.Omitted):
        return {bool(overwrite_x.value)}
    if isinstance(overwrite_x, types.BooleanLiteral):
        return {bool(overwrite_x.literal_value)}
    return {False, True} if isinstance(overwrite_x, types.Boolean) else {False}


def _can_reuse_input(kind: Kind, x: types.Array, out_dtype: types.Type) -> bool:
    """Whether SciPy's ``overwrite_x`` may return (a view of) the input itself."""
    transforms_in_place = kind in (Kind.DCT, Kind.DST) or (kind is Kind.C2C and isinstance(x.dtype, types.Complex))
    return transforms_in_place and x.mutable and x.dtype == out_dtype


def _scipy_layout(kind: Kind, x: types.Array, s: types.Type, reuse_states: set) -> str:
    """Numba layout of a scipy.fft result; mirrors ``scipy_result`` in the native library."""
    complex_in = isinstance(x.dtype, types.Complex)
    r2r = kind in (Kind.DCT, Kind.DST)
    if not ((kind is Kind.C2C and complex_in) or r2r):
        return "C"
    converted = not isinstance(x.dtype, (types.Float, types.Complex))
    truncated = not cg.is_none(s)
    layouts = {"C"} if truncated else set()  # a zero-padded copy
    for reuse in ({True} if converted else reuse_states):
        if reuse and converted:
            layouts.add("A" if truncated else cg.like_input_layout(x))
        elif reuse:
            layouts.add("A" if truncated else x.layout)
        else:
            layouts.add(cg.like_input_layout(x) if r2r and complex_in else "C")
    return cg.combine_layouts(layouts)


def _numpy_layout(x: types.Array, s: types.Type) -> str:
    """Numba layout of a numpy.fft result; mirrors ``numpy_result`` in the native library."""
    layout = cg.like_input_layout(x)
    # Without new lengths, a Fortran-contiguous input stays Fortran-contiguous at every step.
    return layout if layout == "C" or (layout == "F" and cg.is_none(s)) else "A"


def transform(kind: Kind, flags: Flag, api: Api):
    """Kernel ``(x, s, axes, norm, out, overwrite_x, workers, type, orthogonalize) -> array``.

    Arguments that do not apply to a function are passed as ``None``. The
    result has exactly the memory layout NumPy's or SciPy's function returns.
    """

    def kernel(typingctx, x, s, axes, norm, out, overwrite_x, workers, type, orthogonalize):
        out_dtype = result_dtype(kind, x.dtype)
        has_out = not is_none(out)
        reuse = not has_out and _can_reuse_input(kind, x, out_dtype) and True in _overwrite_states(overwrite_x)
        if has_out:
            if out.dtype != out_dtype:
                raise TypingError(f"Cannot cast output from dtype '{out_dtype}' to dtype '{out.dtype}'.")
            if out.ndim != x.ndim:
                raise TypingError("The output array must have the same number of dimensions as the input.")
            if not out.mutable:
                raise TypingError("The output array must be writeable.")
            ret = out
        elif api is Api.NUMPY:
            ret = types.Array(out_dtype, x.ndim, _numpy_layout(x, s))
        else:
            reuse_states = _overwrite_states(overwrite_x) if reuse else {False}
            ret = types.Array(out_dtype, x.ndim, _scipy_layout(kind, x, s, reuse_states))
        sig = ret(x, s, axes, norm, out, overwrite_x, workers, type, orthogonalize)

        def codegen(context, builder, sig, args):
            x_t, s_t, axes_t, norm_t, out_t, ow_t, workers_t, type_t, ortho_t = sig.args
            x_v, s_v, axes_v, norm_v, out_v, ow_v, workers_v, type_v, ortho_v = args
            n_workers, workers_given = cg.workers(builder, workers_t, workers_v)
            opts = cg.options(
                builder,
                kind=int(kind),
                flags=int(flags),
                in_dtype=cg.dtype_code(x_t.dtype),
                out_dtype=cg.dtype_code(out_dtype),
                norm=cg.norm(context, builder, norm_t, norm_v),
                r2r_type=cg.const(0) if is_none(type_t) else cg.to_i64(builder, type_t, type_v),
                ortho=cg.optional_flag(builder, ortho_t, ortho_v),
                workers=n_workers,
                workers_given=workers_given,
            )
            return cg.call_returning_array(context, builder, "rf_transform", ret, [
                opts,
                cg.array_ptr(context, builder, x_t, x_v),
                cg.const(x_t.ndim),
                cg.int_seq(context, builder, s_t, s_v),
                cg.int_seq(context, builder, axes_t, axes_v),
                cg.array_ptr(context, builder, out_t, out_v) if has_out else cg.null(),
                cg.to_i64(builder, ow_t, ow_v) if reuse else cg.const(0),
                cg.const(api),
            ])

        return sig, codegen

    return kernel


# -----------------------------------------------------------------------------
# Fast Hankel transform
# -----------------------------------------------------------------------------

def fht(inverse: bool):
    """Kernel ``(a, dln, mu, offset, bias) -> array`` for ``fht``/``ifht``."""

    def kernel(typingctx, a, dln, mu, offset, bias):
        # Known difference from SciPy: SciPy returns float64 for float32 input whenever 
        # bias != 0, i.e. depending on a run-time value. A compiled function needs a 
        # static result dtype, so float32 is kept but the values agree.
        out_dtype = types.float32 if a.dtype == types.float32 else types.float64
        # Like SciPy, the result is a view reversed along the last axis.
        ret = types.Array(out_dtype, a.ndim, "A")
        sig = ret(a, dln, mu, offset, bias)

        def codegen(context, builder, sig, args):
            a_t, *scalar_types = sig.args
            a_v, *scalar_values = args
            return cg.call_returning_array(context, builder, "rf_fht", ret, [
                cg.array_ptr(context, builder, a_t, a_v),
                cg.const(a_t.ndim),
                cg.const(cg.dtype_code(a_t.dtype)),
                cg.const(cg.dtype_code(out_dtype)),
                *(cg.to_f64(builder, t, v) for t, v in zip(scalar_types, scalar_values)),
                cg.const(inverse),
            ])

        return sig, codegen

    return kernel


def fhtoffset(typingctx, dln, mu, initial, bias):
    """Kernel ``(dln, mu, initial, bias) -> float``."""
    sig = types.float64(dln, mu, initial, bias)

    def codegen(context, builder, sig, args):
        scalars = [cg.to_f64(builder, t, v) for t, v in zip(sig.args, args)]
        return cg.call(builder, "rf_fhtoffset", cg.f64, scalars)

    return sig, codegen


# -----------------------------------------------------------------------------
# Helpers: fftfreq, rfftfreq, fftshift, ifftshift
# -----------------------------------------------------------------------------

def freq(rfft: bool):
    """Kernel ``(n, d, device) -> array`` for ``fftfreq``/``rfftfreq``."""

    def kernel(typingctx, n, d, device):
        is_complex = isinstance(d, types.Complex) or (
            isinstance(d, types.Omitted) and isinstance(d.value, complex)
        )
        ret = types.Array(types.complex128 if is_complex else types.float64, 1, "C")
        sig = ret(n, d, device)

        def codegen(context, builder, sig, args):
            (n_t, d_t, device_t), (n_v, d_v, device_v) = sig.args, args
            if isinstance(device_t, types.UnicodeType):
                cg.check_device(context, builder, device_v)
            d_re, d_im = cg.to_c128(context, builder, d_t, d_v)
            return cg.call_returning_array(context, builder, "rf_freq", ret, [
                cg.to_i64(builder, n_t, n_v), d_re, d_im, cg.const(is_complex), cg.const(rfft),
            ])

        return sig, codegen

    return kernel


def fftshift(inverse: bool):
    """Kernel ``(x, axes) -> array`` for ``fftshift``/``ifftshift``."""

    def kernel(typingctx, x, axes):
        ret = types.Array(x.dtype, x.ndim, cg.like_input_layout(x))
        sig = ret(x, axes)

        def codegen(context, builder, sig, args):
            (x_t, axes_t), (x_v, axes_v) = sig.args, args
            return cg.call_returning_array(context, builder, "rf_fftshift", ret, [
                cg.array_ptr(context, builder, x_t, x_v),
                cg.const(x_t.ndim),
                cg.int_seq(context, builder, axes_t, axes_v),
                cg.const(inverse),
            ])

        return sig, codegen

    return kernel


# -----------------------------------------------------------------------------
# Optimal lengths and worker threads
# -----------------------------------------------------------------------------

def fast_len(native_name: str):
    """Kernel ``(target, real) -> int`` calling ``rf_good_size`` or ``rf_prev_good_size``."""

    def kernel(typingctx, target, real):
        if not cg.is_int_like(target) or not cg.is_int_like(real):
            raise TypingError("'target' must be an integer and 'real' a boolean.")
        sig = types.int64(target, real)

        def codegen(context, builder, sig, args):
            (target_t, real_t), (target_v, real_v) = sig.args, args
            result = cg.call(builder, native_name, cg.i64, [
                cg.to_i64(builder, target_t, target_v),
                cg.to_i64(builder, real_t, real_v),
            ])
            # Negative results encode a status code.
            failed = builder.icmp_signed("<", result, cg.const(0))
            cg.check_status(context, builder, builder.select(failed, builder.neg(result), cg.const(0)))
            return result

        return sig, codegen

    return kernel


def default_workers(typingctx):
    """Kernel ``() -> int`` returning the default number of workers."""

    def codegen(context, builder, sig, args):
        return cg.call(builder, "rf_get_default_workers", cg.i64, [])

    return types.int64(), codegen
