import inspect
from functools import partial, wraps
from os import cpu_count

import numpy as np
import numpy.fft
from numba.core import types
from numba.core.errors import NumbaValueError, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import overload, register_jitable
from numba.np.numpy_support import is_nonelike

from . import pocketfft
from . import typutils as tu
from .imputils import (
    implements_jit,
    implements_overload,
    otherwise,
)
from .typutils import (
    is_contiguous_array,
    is_integer,
    is_integer_2tuple,
    is_unicode,
    is_literal_bool,
    is_literal_integer,
    is_literal_string,
    is_nonelike,
    is_not_nonelike,
    is_scalar,
    typing_check,
)

# Unlike NumPy, SciPy is an optional runtime dependency
try:
    import scipy.fft
    from . import special

    _scipy_installed_ = True
except ImportError:
    _scipy_installed_ = False

# -----------------------------------------------------------------------------
# Type conversion
# -----------------------------------------------------------------------------

_cmplx_lut = {
    types.complex64: types.complex64,
    types.complex128: types.complex128,
    types.float32: types.complex64,
    types.float64: types.complex128,
    types.int8: types.complex128,
    types.int16: types.complex128,
    types.int32: types.complex128,
    types.int64: types.complex128,
    types.uint8: types.complex128,
    types.uint16: types.complex128,
    types.uint32: types.complex128,
    types.uint64: types.complex128,
    types.bool_: types.complex128,
    types.byte: types.complex128,
}
_real_lut = {key: val.underlying_float for key, val in _cmplx_lut.items()}


def _as_supported_dtype(dtype, lut):
    ty = lut.get(dtype)
    if ty is not None:
        return ty

    keys = tuple(lut.keys())
    raise TypingError(f"Unsupported dtype {dtype}; supported are {keys}.")


as_supported_cmplx = partial(_as_supported_dtype, lut=_cmplx_lut)
as_supported_real = partial(_as_supported_dtype, lut=_real_lut)


# -----------------------------------------------------------------------------
# Typing checkers
# -----------------------------------------------------------------------------

fft_typing = tu.TypingChecker(
    a=tu.Check(
        types.Array,
        as_one=True,
        as_seq=False,
        allow_none=False,
        msg="The {} argument '{}' must be an array.",
    ),
    x=tu.Check(
        types.Array,
        as_one=True,
        as_seq=False,
        allow_none=False,
        msg="The {} argument '{}' must be an array.",
    ),
    n=tu.Check(
        types.Integer,
        as_one=True,
        as_seq=True,
        allow_none=True,
        msg="The {} argument '{}' must be an integer.",
    ),
    s=tu.Check(
        types.Integer,
        as_one=True,
        as_seq=True,
        allow_none=True,
        msg="The {} argument '{}' must be a sequence of integers.",
    ),
    axis=tu.Check(
        types.Integer,
        as_one=True,
        as_seq=True,
        allow_none=True,
        msg="The {} argument '{}' must be an integer.",
    ),
    axes=tu.Check(
        types.Integer,
        as_one=True,
        as_seq=True,
        allow_none=True,
        msg="The {} argument '{}' must be a sequence of integers.",
    ),
    norm=tu.Check(
        types.UnicodeType,
        as_one=True,
        as_seq=False,
        allow_none=True,
        msg="The {} argument '{}' must be a string.",
    ),
    type=tu.Check(
        types.Integer,
        as_one=True,
        as_seq=False,
        allow_none=False,
        msg="The {} argument '{}' must be an integer.",
    ),
    overwrite_x=tu.Check(
        types.Boolean,
        as_one=True,
        as_seq=False,
        allow_none=False,
        msg="The {} argument '{}' must be a boolean.",
    ),
    out=tu.Check(
        types.NoneType,
        as_one=True,
        as_seq=False,
        allow_none=True,
        msg="The {} argument '{}' is not yet supported.",  # TODO
    ),
    workers=tu.Check(
        types.Integer,
        as_one=True,
        as_seq=False,
        allow_none=True,
        msg="The {} argument '{}' must be an integer.",
    ),
    orthogonalize=tu.Check(
        types.Boolean,
        as_one=True,
        as_seq=False,
        allow_none=True,
        msg="The {} argument '{}' must be a boolean.",
    ),
    plan=tu.Check(
        types.NoneType,
        as_one=True,
        as_seq=False,
        allow_none=True,
        msg="The {} argument '{}' is not supported.",
    ),
)


fftshift_typing = tu.TypingChecker(
    x=tu.Check(
        types.Array,
        msg="The {} argument '{}' must be an array.",
    ),
    axes=tu.Check(
        types.Integer,
        as_seq=True,
        allow_none=True,
        msg="The {} argument '{}' must be a sequence of integers or an integer.",
    ),
)


fftfreq_typing = tu.TypingChecker(
    n=tu.Check(
        types.Integer,
        msg="The {} argument '{}' must be an integer.",
    ),
    d=tu.Check(
        types.Number,
        msg="The {} argument '{}' must be a scalar.",
    ),
    device=tu.Check(
        (types.StringLiteral, types.UnicodeType),
        as_one=True,
        as_seq=False,
        allow_none=True,
        msg="Only the value 'cpu' is supported for the {} argument '{}'.",
    ),
    xp=tu.Check(
        (types.Module, types.NoneType),
        as_one=True,
        as_seq=False,
        allow_none=True,
        msg="The {} argument '{}' is not yet supported.",
    ),
)

fht_typing = tu.TypingChecker(
    a=tu.Check(
        types.Array,
        msg="The {} argument '{}' must be an array.",
    ),
    A=tu.Check(
        types.Array,
        msg="The {} argument '{}' must be an array.",
    ),
    dln=tu.Check(
        types.Number,
        msg="The {} argument '{}' must be a scalar.",
    ),
    mu=tu.Check(
        types.Number,
        msg="The {} argument '{}' must be a scalar.",
    ),
    initial=tu.Check(
        types.Number,
        msg="The {} argument '{}' must be a scalar.",
    ),
    bias=tu.Check(
        types.Number,
        msg="The {} argument '{}' must be a scalar.",
    ),
    offset=tu.Check(
        types.Number,
        msg="The {} argument '{}' must be a scalar.",
    ),
)

# -----------------------------------------------------------------------------
# Public functions (not jitted)
# -----------------------------------------------------------------------------


_cpu_count = cpu_count()
_num_workers = 1  # default is a single worker


def get_workers():
    return _num_workers


def set_workers(workers):
    if workers < 1:
        raise ValueError("Number of workers cannot be smaller than one.")
    if workers > _cpu_count:
        raise ValueError("Number of workers exceeds " f"CPU count of {_cpu_count}.")

    global _num_workers
    _num_workers = _num_workers


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


@register_jitable
def wraparound_axis(x, ax):
    if (ax >= x.ndim) or (ax < -x.ndim):
        raise NumbaValueError("Axis exceeds dimensionality of input.")
    if ax < 0:
        ax += x.ndim
    return ax


@register_jitable
def wraparound_axes(x, axes):
    for i, ax in enumerate(axes):
        if (ax >= x.ndim) or (ax < -x.ndim):
            raise NumbaValueError("Axes exceeds dimensionality of input.")
        if ax < 0:
            axes[i] += x.ndim


@register_jitable(locals={"slots": types.UniTuple(types.byte, 32)})
def ensure_unique_axes(axes):
    slots = (0,) * 32  # np.MAXDIMS
    for ax in axes:
        if slots[ax] != 0:
            raise NumbaValueError("All axes must be unique.")
        slots = tuple_setitem(slots, ax, 1)


@register_jitable
def assert_valid_shape(shape):
    for n in shape:
        if n < 1:
            raise NumbaValueError("Invalid number of data points specified.")


@register_jitable
def toarray(arg):
    a = np.asarray(arg, dtype=np.int64)
    return np.atleast_1d(a)


@implements_jit(prefer_literal=True)
def ndshape_and_axes(x, s, axes):
    pass


@ndshape_and_axes.impl(s=is_nonelike, axes=is_literal_integer(-1))
def _(x, s, axes):
    # Specialization for default 1D transform
    return s, np.array([x.ndim - 1])


@ndshape_and_axes.impl(s=is_nonelike, axes=is_integer)
def _(x, s, axes):
    # Specialization for 1D transform
    axes = wraparound_axis(x, axes)
    axes = np.array([axes])
    return s, axes


@ndshape_and_axes.impl(s=is_nonelike, axes=is_integer_2tuple)
def _(x, s, axes):
    # Specialization for 2D transform
    ax1, ax2 = axes
    ax1 = wraparound_axis(x, ax1)
    ax2 = wraparound_axis(x, ax2)
    ensure_unique_axes((ax1, ax2))
    axes = np.array([ax1, ax2])
    return s, axes


@ndshape_and_axes.impl(s=is_nonelike, axes=is_nonelike)
def _(x, s, axes):
    # Specialization for default ND transform
    # Axes not specified, transform all axes
    axes = np.arange(x.ndim)
    return s, axes


@ndshape_and_axes.impl(s=is_nonelike, axes=is_not_nonelike)
def _(x, s, axes):
    axes = toarray(axes)
    wraparound_axes(x, axes)
    ensure_unique_axes(axes)
    return s, axes


@ndshape_and_axes.impl(s=is_not_nonelike, axes=is_nonelike)
def _(x, s, axes):
    s = toarray(s)
    assert_valid_shape(s)
    if s.size > x.ndim:
        raise NumbaValueError("Shape requires more axes than are present.")
    # Axes not specified, transform last len(s) axes
    axes = np.arange(x.ndim - s.size, x.ndim)
    return s, axes


@ndshape_and_axes.impl(otherwise)
def _(x, s, axes):
    s = toarray(s)
    assert_valid_shape(s)
    axes = toarray(axes)
    wraparound_axes(x, axes)
    ensure_unique_axes(axes)
    if s.size != axes.size:
        raise NumbaValueError(
            "When given, axes and shape arguments" " have to be of the same length."
        )
    return s, axes


@implements_jit
def mul_axes(shape, axes, delta=None):
    pass


@mul_axes.impl(delta=is_nonelike)
def _(shape, axes, delta=None):
    n = 1.0
    for ax in axes:
        n *= shape[ax]
    return n


@mul_axes.impl(otherwise)
def _(shape, axes, delta=None):
    n = 1.0
    for ax in axes:
        n *= 2.0 * (shape[ax] + delta)
    return n


@implements_jit(prefer_literal=True)
def get_fct(x, axes, norm, forward, delta=None):
    pass


@get_fct.impl(norm=is_nonelike, forward=is_literal_bool(True))
def _(x, axes, norm, forward, delta=None):
    return 1.0


@get_fct.impl(norm=is_nonelike, forward=is_literal_bool(False))
def _(x, axes, norm, forward, delta=None):
    return 1.0 / mul_axes(x.shape, axes, delta)


@get_fct.impl(norm=is_not_nonelike, forward=is_literal_bool(True))
def _(x, axes, norm, forward, delta=None):
    if norm == "backward":
        return 1.0
    elif norm == "ortho":
        return 1.0 / np.sqrt(mul_axes(x.shape, axes, delta))
    elif norm == "forward":
        return 1.0 / mul_axes(x.shape, axes, delta)
    raise NumbaValueError(
        "Invalid norm value; should be" " 'backward', 'ortho' or 'forward'."
    )


@get_fct.impl(norm=is_not_nonelike, forward=is_literal_bool(False))
def _(x, axes, norm, forward, delta=None):
    if norm == "backward":
        return 1.0 / mul_axes(x.shape, axes, delta)
    elif norm == "ortho":
        return 1.0 / np.sqrt(mul_axes(x.shape, axes, delta))
    elif norm == "forward":
        return 1.0
    raise NumbaValueError(
        "Invalid norm value; should be" " 'backward', 'ortho' or 'forward'."
    )


@implements_jit(prefer_literal=True)
def get_nthreads(workers):
    pass


@get_nthreads.impl(workers=is_nonelike)
def _(workers):
    return _num_workers


@get_nthreads.impl(otherwise)
def _(workers):
    if workers > 0:
        return workers
    if workers == 0:
        raise NumbaValueError("Workers must not be zero.")
    if workers < -_cpu_count:
        raise NumbaValueError("Workers value out of range.")
    return workers + _cpu_count + 1


@implements_jit
def zeropad_or_crop(x, s, axes, dtype):
    pass


@zeropad_or_crop.preproc
def _(x, s, axes, dtype):
    if hasattr(dtype, "instance_type"):
        dtype = dtype.instance_type
    elif hasattr(dtype, "_dtype"):
        dtype = dtype._dtype
    return x, s, axes, dtype


@zeropad_or_crop.impl(s=is_not_nonelike)
def _(x, s, axes, dtype):
    shape = x.shape
    for n, ax in zip(s, axes):
        shape = tuple_setitem(shape, ax, n)
    out = np.zeros(shape, dtype=dtype)
    for i, (s1, s2) in enumerate(zip(x.shape, out.shape)):
        shape = tuple_setitem(shape, i, min(s1, s2))
    for index in np.ndindex(shape):
        out[index] = x[index]
    return out


@zeropad_or_crop.impl(lambda x, s, axes, dtype: x.dtype != dtype)
def _(x, s, axes, dtype):
    return x.astype(dtype)


@zeropad_or_crop.impl(otherwise)
def _(x, s, axes, dtype):
    return x


def generated_alloc_output(s, istype, reqtype):
    # We don't allocate a new array if:
    # 1. overwrite was requested (runtime check)
    # 2. array got casted (compile time check)
    # 3. the array has been zero-padded/truncated (compile time check)
    if (istype != reqtype) or not is_nonelike(s):

        @register_jitable
        def alloc_output(x, overwrite_x):
            return x

        return alloc_output

    @implements_jit(prefer_literal=True)
    def alloc_output(x, overwrite_x):
        pass

    @alloc_output.impl(overwrite_x=is_literal_bool(False))
    def _(x, overwrite_x):
        # Specialization for default case
        return np.empty(x.shape, dtype=x.dtype)

    @alloc_output.impl(otherwise)
    def _(x, overwrite_x):
        if overwrite_x:
            return x
        out = np.empty(x.shape, dtype=x.dtype)
        return out

    return alloc_output


@register_jitable
def decrease_shape(shape, axes):
    idx = axes[-1]
    n = (shape[idx] // 2) + 1
    shape = tuple_setitem(shape, idx, n)
    return shape


@register_jitable
def increase_shape(shape, axes):
    idx = axes[-1]
    n = (shape[idx] - 1) * 2
    shape = tuple_setitem(shape, idx, n)
    return shape


@implements_jit
def resize(shape, x, s, axes):
    pass


@resize.impl(s=is_nonelike)
def _(shape, x, s, axes):
    return shape


@resize.impl(otherwise)
def _(shape, x, s, axes):
    last_ax = x.ndim - 1
    for i, ax in enumerate(axes):
        if ax == last_ax:
            shape = tuple_setitem(shape, last_ax, s[i])
            break
    return shape


@implements_jit(prefer_literal=True)
def get_type(type, forward):
    pass


@get_type.impl(type=is_literal_integer(2), forward=is_literal_bool(True))
def _(type, forward):
    # Specialization for default case forward
    return 2


@get_type.impl(type=is_literal_integer(2), forward=is_literal_bool(False))
def _(type, forward):
    # Specialization for default case backward
    return 3


@get_type.impl(forward=is_literal_bool(True))
def _(type, forward):
    if type not in (1, 2, 3, 4):
        raise NumbaValueError("Invalid type; must be one of (1, 2, 3, 4).")
    return type


@get_type.impl(forward=is_literal_bool(False))
def _(type, forward):
    if type == 2:
        return 3
    if type == 3:
        return 2
    if type not in (1, 4):
        raise NumbaValueError("Invalid type; must be one of (1, 2, 3, 4).")
    return type


@implements_jit
def get_ortho(norm, ortho):
    pass


@get_ortho.impl(ortho=is_not_nonelike)
def _(norm, ortho):
    return ortho


@get_ortho.impl(otherwise)
def _(norm, ortho):
    return norm == "ortho"


@implements_jit(prefer_literal=True)
def check_device(device):
    pass


@check_device.impl(device=is_nonelike)
@check_device.impl(device=is_literal_string("cpu"))
@check_device.impl(device=is_literal_string("CPU"))
def _(device):
    return


@check_device.impl(device=is_unicode)
def _(device):
    if device != "cpu" and device != "CPU":
        raise ValueError("Only 'cpu' device is supported.")


def _get_slice_tuple(arr):
    pass


@overload(_get_slice_tuple)
def _get_slice_tuple_impl(arr):
    tup = (slice(None),) * arr.ndim
    return lambda arr: tup


@register_jitable
def _roll_core_impl(a, shift, axis):
    axis, shift = np.broadcast_arrays(axis, shift)

    shifts = {ax: 0 for ax in range(a.ndim)}
    for ax, sh in zip(axis, shift):
        if (ax >= a.ndim) or (ax < -a.ndim):
            raise NumbaValueError("axis is out of bounds")
        if ax < 0:
            ax += a.ndim
        shifts[ax] += sh

    n_sclices = a.shape
    out_slices = [(slice(None), slice(None))] * a.ndim
    arr_slices = [(slice(None), slice(None))] * a.ndim
    for ax, sh in shifts.items():
        sh %= a.shape[ax] or 1
        n_sclices = tuple_setitem(n_sclices, ax, (2 if sh else 1))
        if sh:
            out_slices[ax] = (slice(None, sh), slice(sh, None))
            arr_slices[ax] = (slice(-sh, None), slice(None, -sh))

    out = np.empty(a.shape, dtype=a.dtype)

    arr_index = _get_slice_tuple(a)
    out_index = _get_slice_tuple(a)
    for index in np.ndindex(n_sclices):
        for ax, i in enumerate(index):
            arr_index = tuple_setitem(arr_index, ax, arr_slices[ax][i])
            out_index = tuple_setitem(out_index, ax, out_slices[ax][i])

        out[out_index] = a[arr_index]

    return out


@implements_overload(numpy.roll)
def roll(a, shift, axis=None):
    typing_check((types.Number, types.Boolean), as_seq=True)(
        a,
        "The 1st argument 'a' must be array-like.",
    )
    typing_check((types.Integer, types.Boolean), as_seq=True)(
        shift,
        "The 2nd argument 'shift' must be a sequence of integers or an integer.",
    )
    typing_check((types.Integer, types.Boolean), as_seq=True, allow_none=True)(
        axis,
        "If specified, the 3rd argument 'axis' must be a sequence of integers or an integer",
    )


@roll.impl(a=is_scalar)
def _(a, shift, axis=None):
    return np.asarray(a)


@roll.impl(axis=is_nonelike)
def _(a, shift, axis=None):
    arr = np.asarray(a)
    out = np.empty(arr.shape, dtype=arr.dtype)
    arr = arr.ravel()  # much faster than arr.flat on A/F arrays

    shift = np.asarray(shift)
    if shift.ndim > 1:
        raise NumbaValueError("'shift' must be a scalar or 1D sequence")

    sh = shift.sum() % (arr.size or 1)
    inv_sh = arr.size - sh

    for i in range(inv_sh):
        out.flat[sh + i] = arr[i]
    for i in range(sh):
        out.flat[i] = arr[inv_sh + i]

    return out


@roll.impl(a=is_contiguous_array(layout="C"))
def _(a, shift, axis=None):
    return _roll_core_impl(a, shift, axis)


@register_jitable
def _transpose_axes(axis, ndim):
    axis = np.asarray(axis).ravel()
    return np.array([(ndim - ax - 1) % ndim for ax in axis])


@roll.impl(a=is_contiguous_array(layout="F"))
def _(a, shift, axis=None):
    axis = _transpose_axes(axis, a.ndim)
    return _roll_core_impl(a.T, shift, axis).T


@roll.impl(otherwise)
def _(a, shift, axis=None):
    arr = np.asarray(a)

    if arr.strides[0] >= arr.strides[-1]:
        return _roll_core_impl(arr, shift, axis)

    axis = _transpose_axes(axis, arr.ndim)
    return _roll_core_impl(arr.T, shift, axis).T


@register_jitable
def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0):
    lnkr = offset
    q = bias
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = np.linspace(0, np.pi * (n // 2) / (n * dln), n // 2 + 1)
    u = np.empty(n // 2 + 1, dtype=np.complex128)
    v = np.empty(n // 2 + 1, dtype=np.complex128)
    u[:] = xm + y * 1j
    special.loggamma(u, v)
    u.real[:] = xp
    special.loggamma(u, u)
    y *= 2 * (np.log(2) - lnkr)
    u = np.exp((u.real - v.real + np.log(2) * q) + (u.imag + v.imag + y) * 1j)
    u.imag[-1] = 0
    if not np.isfinite(u[0]):
        u[0] = 2**q * special.poch(xm, xp - xm)
    return u


@register_jitable
def _fhtq(a, u):
    if np.isinf(u[0]):
        # TODO: Any better solution for dealing with warnings in jitted code?
        print("Warning: singular transform; consider changing the bias")
        u = u.copy()
        u[0] = 0
    A = np.fft.rfft(a)
    A *= u
    A = np.fft.irfft(A, a.shape[-1])
    return A[..., ::-1]


@register_jitable
def _ifhtq(a, u):
    if u[0] == 0:
        # TODO: Any better solution for dealing with warnings in jitted code?
        print("Warning: singular inverse transform; consider changing the bias")
        u = u.copy()
        u[0] = np.inf
    A = np.fft.rfft(a)
    A /= u.conj()
    A = np.fft.irfft(A, a.shape[-1])
    return A[..., ::-1]


def next_fast_len(target, real=False):
    typing_check(types.Integer)(
        target,
        "The 1st argument 'target' must be an integer.",
    )
    typing_check(types.Boolean)(
        real,
        "The 2nd argument 'real' must be a boolean.",
    )

    def impl(target, real=False):
        if target < 0:
            raise NumbaValueError("Target cannot be negative.")
        return pocketfft.numba_good_size(target, real)

    return impl


# TODO
def prev_fast_len(target, real=False):
    raise NotImplementedError("scipy.fft.prev_fast_len is not yet supported.")


# -----------------------------------------------------------------------------
# Generic transforms
# -----------------------------------------------------------------------------


def apply_signature(src, dst):
    src_sig = inspect.signature(src)
    code = dst.__code__

    src_nargs = len(src_sig.parameters)
    dst_nargs = code.co_argcount

    # number of non-argument locals must stay identical
    src_locals = len(code.co_varnames) - dst_nargs
    dst_locals = len(code.co_varnames) - dst_nargs

    if src_locals != dst_locals:
        raise ValueError("Destination function has incompatible local variable layout")

    new_varnames = (
        *src_sig.parameters,
        *code.co_varnames[dst_nargs:],
    )

    dst.__code__ = code.replace(
        co_varnames=new_varnames,
        co_argcount=src_nargs,
    )

    dst.__defaults__ = tuple(
        p.default
        for p in src_sig.parameters.values()
        if p.default is not inspect._empty
    )

    return dst


def apply_signature_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return apply_signature(func, func(*args, **kwargs))

    return wrapper


def auto_typing_decorator(typing_checker):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            typing_checker(**bound.arguments)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def scipy_c2cn(x, s, forward):
    rettype = as_supported_cmplx(x.dtype)

    if isinstance(x.dtype, types.Complex):
        alloc_output = generated_alloc_output(s, x.dtype, rettype)

        def impl(x, s, axes, norm, overwrite_x, workers, plan):
            s, axes = ndshape_and_axes(x, s, axes)
            x = zeropad_or_crop(x, s, axes, rettype)
            out = alloc_output(x, overwrite_x)
            fct = get_fct(x, axes, norm, forward)
            nthreads = get_nthreads(workers)
            pocketfft.numba_c2c(x, out, axes, forward, fct, nthreads)
            return out

    else:
        argtype = as_supported_real(x.dtype)

        def impl(x, s, axes, norm, overwrite_x, workers, plan):
            s, axes = ndshape_and_axes(x, s, axes)
            x = zeropad_or_crop(x, s, axes, argtype)
            out = np.empty(x.shape, dtype=rettype)
            fct = get_fct(x, axes, norm, forward)
            nthreads = get_nthreads(workers)
            pocketfft.numba_c2c_sym(x, out, axes, forward, fct, nthreads)
            return out

    return impl


def scipy_c2rn(x, forward):
    argtype = as_supported_cmplx(x.dtype)
    rettype = as_supported_real(argtype)

    def impl(x, s, axes, norm, overwrite_x, workers, plan):
        s, axes = ndshape_and_axes(x, s, axes)
        xin = zeropad_or_crop(x, s, axes, argtype)
        shape = increase_shape(x.shape, axes)
        shape = resize(shape, x, s, axes)
        out = np.empty(shape, dtype=rettype)
        fct = get_fct(out, axes, norm, forward)
        nthreads = get_nthreads(workers)
        pocketfft.numba_c2r(xin, out, axes, forward, fct, nthreads)
        return out

    return impl


def scipy_r2rn(x, s, trafo, delta, forward):
    if isinstance(x.dtype, types.Complex):
        argtype = as_supported_cmplx(x.dtype)

        @register_jitable
        def transform(x, out, axes, type, fct, ortho, nthreads):
            trafo(x.real, out.real, axes, type, fct, ortho, nthreads)
            trafo(x.imag, out.imag, axes, type, fct, ortho, nthreads)

    else:
        argtype = as_supported_real(x.dtype)
        transform = trafo

    rettype = argtype
    alloc_output = generated_alloc_output(s, x.dtype, rettype)

    def impl(x, type, s, axes, norm, overwrite_x, workers, orthogonalize):
        s, axes = ndshape_and_axes(x, s, axes)
        x = zeropad_or_crop(x, s, axes, rettype)
        out = alloc_output(x, overwrite_x)
        type = get_type(type, forward)
        delta_ = delta if type == 1 else 0.0
        fct = get_fct(out, axes, norm, forward, delta_)
        ortho = get_ortho(norm, orthogonalize)
        nthreads = get_nthreads(workers)
        transform(x, out, axes, type, fct, ortho, nthreads)
        return out

    return impl


def scipy_r2cn(x, forward):
    if isinstance(x.dtype, types.Complex):
        raise TypingError(f"unsupported dtype {x.dtype}")

    argtype = as_supported_real(x.dtype)
    rettype = as_supported_cmplx(argtype)

    def impl(x, s, axes, norm, overwrite_x, workers, plan):
        s, axes = ndshape_and_axes(x, s, axes)
        x = zeropad_or_crop(x, s, axes, argtype)
        shape = decrease_shape(x.shape, axes)
        out = np.empty(shape, dtype=rettype)
        fct = get_fct(x, axes, norm, forward)
        nthreads = get_nthreads(workers)
        pocketfft.numba_r2c(x, out, axes, forward, fct, nthreads)
        return out

    return impl


def numpy_c2cn(x, s, forward):
    rettype = as_supported_cmplx(x.dtype)

    if isinstance(x.dtype, types.Complex):
        alloc_output = generated_alloc_output(s, x.dtype, rettype)

        def impl(x, s, axes, norm, _out):
            s, axes = ndshape_and_axes(x, s, axes)
            x = zeropad_or_crop(x, s, axes, rettype)
            out = alloc_output(x, False)
            fct = get_fct(x, axes, norm, forward)
            nthreads = get_nthreads(None)
            pocketfft.numba_c2c(x, out, axes, forward, fct, nthreads)
            return out

    else:
        argtype = as_supported_real(x.dtype)

        def impl(x, s, axes, norm, _out):
            s, axes = ndshape_and_axes(x, s, axes)
            x = zeropad_or_crop(x, s, axes, argtype)
            out = np.empty(x.shape, dtype=rettype)
            fct = get_fct(x, axes, norm, forward)
            nthreads = get_nthreads(None)
            pocketfft.numba_c2c_sym(x, out, axes, forward, fct, nthreads)
            return out

    return impl


def numpy_r2cn(x, forward):
    if isinstance(x.dtype, types.Complex):
        raise TypingError(f"unsupported dtype {x.dtype}")

    argtype = as_supported_real(x.dtype)
    rettype = as_supported_cmplx(argtype)

    def impl(x, s, axes, norm, _out):
        s, axes = ndshape_and_axes(x, s, axes)
        x = zeropad_or_crop(x, s, axes, argtype)
        shape = decrease_shape(x.shape, axes)
        out = np.empty(shape, dtype=rettype)
        fct = get_fct(x, axes, norm, forward)
        nthreads = get_nthreads(None)
        pocketfft.numba_r2c(x, out, axes, forward, fct, nthreads)
        return out

    return impl


def numpy_c2rn(x, forward):
    argtype = as_supported_cmplx(x.dtype)
    rettype = as_supported_real(argtype)

    def impl(x, s, axes, norm, _out):
        s, axes = ndshape_and_axes(x, s, axes)
        xin = zeropad_or_crop(x, s, axes, argtype)
        shape = increase_shape(x.shape, axes)
        shape = resize(shape, x, s, axes)
        out = np.empty(shape, dtype=rettype)
        fct = get_fct(out, axes, norm, forward)
        nthreads = get_nthreads(None)
        pocketfft.numba_c2r(xin, out, axes, forward, fct, nthreads)
        return out

    return impl


# -----------------------------------------------------------------------------
# Numpy FFT overloads
# -----------------------------------------------------------------------------


@overload(numpy.fft.fft)
@apply_signature_decorator
@auto_typing_decorator(fft_typing)
def numpy_fft(a, n=None, axis=-1, norm=None, out=None):
    return numpy_c2cn(a, n, forward=True)


@overload(numpy.fft.ifft)
@apply_signature_decorator
def numpy_ifft(a, n=None, axis=-1, norm=None, out=None):
    fft_typing(a=a, n=n, axis=axis, norm=norm, out=out)
    return numpy_c2cn(a, n, forward=False)


@overload(numpy.fft.fft2)
@apply_signature_decorator
def numpy_fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_c2cn(a, s, forward=True)


@overload(numpy.fft.ifft2)
@apply_signature_decorator
def numpy_ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_c2cn(a, s, forward=False)


@overload(numpy.fft.fftn)
@apply_signature_decorator
def numpy_fftn(a, s=None, axes=None, norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_c2cn(a, s, forward=True)


@overload(numpy.fft.ifftn)
@apply_signature_decorator
def numpy_ifftn(a, s=None, axes=None, norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_c2cn(a, s, forward=False)


@overload(numpy.fft.rfft)
@apply_signature_decorator
def numpy_rfft(a, n=None, axis=-1, norm=None, out=None):
    fft_typing(a=a, n=n, axes=axis, norm=norm, out=out)
    return numpy_r2cn(a, forward=True)


@overload(numpy.fft.irfft)
@apply_signature_decorator
def numpy_irfft(a, n=None, axis=-1, norm=None, out=None):
    fft_typing(a=a, n=n, axes=axis, norm=norm, out=out)
    return numpy_c2rn(a, forward=False)


@overload(numpy.fft.rfft2)
@apply_signature_decorator
def numpy_rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_r2cn(a, forward=True)


@overload(numpy.fft.irfft2)
@apply_signature_decorator
def numpy_irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_c2rn(a, forward=False)


@overload(numpy.fft.rfftn)
@apply_signature_decorator
def numpy_rfftn(a, s=None, axes=None, norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_r2cn(a, forward=True)


@overload(numpy.fft.irfftn)
@apply_signature_decorator
def numpy_irfftn(a, s=None, axes=None, norm=None, out=None):
    fft_typing(a=a, s=s, axes=axes, norm=norm, out=out)
    return numpy_c2rn(a, forward=False)


@overload(numpy.fft.hfft)
@apply_signature_decorator
def numpy_hfft(a, n=None, axis=-1, norm=None, out=None):
    fft_typing(a=a, n=n, axis=axis, norm=norm, out=out)
    return numpy_c2rn(a, forward=True)


@overload(numpy.fft.ihfft)
@apply_signature_decorator
def numpy_ihfft(a, n=None, axis=-1, norm=None, out=None):
    fft_typing(a=a, n=n, axis=axis, norm=norm, out=out)
    return numpy_r2cn(a, forward=False)


# -----------------------------------------------------------------------------
# Numpy helper overloads
# -----------------------------------------------------------------------------


def numpy_fftfreq_impl(n, d=1.0, device=None):
    check_device(device)
    val = 1.0 / (n * d)
    results = np.empty(n, dtype=np.int64)
    N = (n - 1) // 2 + 1
    p1 = np.arange(N)
    results[:N] = p1
    p2 = np.arange(-(n // 2), 0)
    results[N:] = p2
    return results * val


@overload(numpy.fft.fftfreq)
def numpy_fftfreq(n, d=1.0, device=None):
    fftfreq_typing(n=n, d=d, device=device)
    return numpy_fftfreq_impl


def numpy_rfftfreq_impl(n, d=1.0, device=None):
    check_device(device)
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(N)
    return results * val


@overload(numpy.fft.rfftfreq)
def numpy_rfftfreq(n, d=1.0, device=None):
    fftfreq_typing(n=n, d=d, device=device)
    return numpy_rfftfreq_impl


@implements_overload(numpy.fft.fftshift)
def fftshift(x, axes=None):
    fftshift_typing(x=x, axes=axes)


@fftshift.impl(axes=is_nonelike)
def _(x, axes=None):
    axes = x.shape
    shift = x.shape
    for i, dim in enumerate(x.shape):
        shift = tuple_setitem(shift, i, dim // 2)
        axes = tuple_setitem(axes, i, i)
    return np.roll(x, shift, axes)


@fftshift.impl(axes=is_integer)
def _(x, axes=None):
    shift = x.shape[axes] // 2
    return np.roll(x, shift, axes)


@fftshift.impl(otherwise)
def _(x, axes=None):
    shift = x.shape[: len(axes)]
    for i, ax in enumerate(axes):
        shift = tuple_setitem(shift, i, x.shape[ax] // 2)
    return np.roll(x, shift, axes)


@implements_overload(numpy.fft.ifftshift)
def ifftshift(x, axes=None):
    fftshift_typing(x=x, axes=axes)


@ifftshift.impl(axes=is_nonelike)
def _(x, axes=None):
    axes = x.shape
    shift = x.shape
    for i, dim in enumerate(x.shape):
        shift = tuple_setitem(shift, i, -(dim // 2))
        axes = tuple_setitem(axes, i, i)
    return np.roll(x, shift, axes)


@ifftshift.impl(axes=is_integer)
def _(x, axes=None):
    shift = -(x.shape[axes] // 2)
    return np.roll(x, shift, axes)


@ifftshift.impl(otherwise)
def _(x, axes=None):
    shift = x.shape[: len(axes)]
    for i, ax in enumerate(axes):
        shift = tuple_setitem(shift, i, -(x.shape[ax] // 2))
    return np.roll(x, shift, axes)


# -----------------------------------------------------------------------------
# Scipy FFT overload
# -----------------------------------------------------------------------------


if _scipy_installed_:

    @overload(scipy.fft.fft, strict=False)
    @apply_signature_decorator
    def scipy_fft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2cn(x, n, forward=True)

    @overload(scipy.fft.ifft)
    @apply_signature_decorator
    def scipy_ifft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2cn(x, n, forward=False)

    @overload(scipy.fft.fft2)
    @apply_signature_decorator
    def scipy_fft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2cn(x, s, forward=True)

    @overload(scipy.fft.ifft2)
    @apply_signature_decorator
    def scipy_ifft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2cn(x, s, forward=False)

    @overload(scipy.fft.fftn)
    @apply_signature_decorator
    def scipy_fftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2cn(x, s, forward=True)

    @overload(scipy.fft.ifftn)
    @apply_signature_decorator
    def scipy_ifftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2cn(x, s, forward=False)

    @overload(scipy.fft.rfft)
    @apply_signature_decorator
    def scipy_rfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_r2cn(x, forward=True)

    @overload(scipy.fft.irfft)
    @apply_signature_decorator
    def scipy_irfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2rn(x, forward=False)

    @overload(scipy.fft.rfft2)
    @apply_signature_decorator
    def scipy_rfft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_r2cn(x, forward=True)

    @overload(scipy.fft.irfft2)
    @apply_signature_decorator
    def scipy_irfft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_r2cn(x, forward=False)

    @overload(scipy.fft.rfftn)
    @apply_signature_decorator
    def scipy_rfftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_r2cn(x, forward=True)

    @overload(scipy.fft.irfftn)
    @apply_signature_decorator
    def scipy_irfftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2rn(x, forward=False)

    @overload(scipy.fft.hfft)
    @apply_signature_decorator
    def scipy_hfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2rn(x, forward=True)

    @overload(scipy.fft.ihfft)
    @apply_signature_decorator
    def scipy_ihfft(
        x,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_r2cn(x, forward=False)

    @overload(scipy.fft.hfft2)
    @apply_signature_decorator
    def scipy_hfft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2rn(x, forward=True)

    @overload(scipy.fft.ihfft2)
    @apply_signature_decorator
    def scipy_ihfft2(
        x,
        s=None,
        axes=(-2, -1),
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_r2cn(x, forward=False)

    @overload(scipy.fft.hfftn)
    @apply_signature_decorator
    def scipy_hfftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_c2rn(x, forward=True)

    @overload(scipy.fft.ihfftn)
    @apply_signature_decorator
    def scipy_ihfftn(
        x,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        plan=None,
    ):
        fft_typing(
            x=x,
            type=type,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            plan=plan,
        )
        return scipy_r2cn(x, forward=False)

    @overload(scipy.fft.dct)
    @apply_signature_decorator
    def scipy_dct(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            n,
            trafo=pocketfft.numba_dct,
            delta=-1,
            forward=True,
        )

    @overload(scipy.fft.idct)
    @apply_signature_decorator
    def scipy_idct(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            n,
            trafo=pocketfft.numba_dct,
            delta=-1,
            forward=False,
        )

    @overload(scipy.fft.dctn)
    @apply_signature_decorator
    def scipy_dctn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            s,
            trafo=pocketfft.numba_dct,
            delta=-1,
            forward=True,
        )

    @overload(scipy.fft.idctn)
    @apply_signature_decorator
    def scipy_idctn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            s,
            trafo=pocketfft.numba_dct,
            delta=-1,
            forward=False,
        )

    @overload(scipy.fft.dst)
    @apply_signature_decorator
    def scipy_dst(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            n,
            trafo=pocketfft.numba_dst,
            delta=1,
            forward=True,
        )

    @overload(scipy.fft.idst)
    @apply_signature_decorator
    def scipy_idst(
        x,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            n,
            trafo=pocketfft.numba_dst,
            delta=1,
            forward=False,
        )

    @overload(scipy.fft.dstn)
    @apply_signature_decorator
    def scipy_dstn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            s,
            trafo=pocketfft.numba_dst,
            delta=1,
            forward=True,
        )

    @overload(scipy.fft.idstn)
    @apply_signature_decorator
    def scipy_idstn(
        x,
        type=2,
        s=None,
        axes=None,
        norm=None,
        overwrite_x=False,
        workers=None,
        orthogonalize=None,
    ):
        fft_typing(
            x=x,
            type=type,
            s=s,
            axes=axes,
            norm=norm,
            overwrite_x=overwrite_x,
            workers=workers,
            orthogonalize=orthogonalize,
        )
        return scipy_r2rn(
            x,
            s,
            trafo=pocketfft.numba_dst,
            delta=1,
            forward=False,
        )

    @overload(scipy.fft.fht)
    def scipy_fht(a, dln, mu, offset=0.0, bias=0.0):
        fht_typing(
            a=a,
            dln=dln,
            mu=mu,
            offset=offset,
            bias=bias,
        )

        dtype = as_supported_real(a.dtype)
        if isinstance(a.dtype, types.Complex):
            raise TypingError("The 1st argument 'a' must be a real array.")

        def impl(a, dln, mu, offset=0.0, bias=0.0):
            a = np.asarray(a, dtype=dtype)
            dln = dtype(dln)
            mu = dtype(mu)
            offset = dtype(offset)
            bias = dtype(bias)

            n = a.shape[-1]
            if bias != 0:
                j_c = (n - 1) / 2
                j = np.arange(n)
                a = a * np.exp(-bias * (j - j_c) * dln)
                a = np.asarray(a, dtype=dtype)
            u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
            A = _fhtq(a, u)
            if bias != 0:
                A *= np.exp(-bias * ((j - j_c) * dln + offset))
            return A

        return impl

    @overload(scipy.fft.ifht)
    def scipy_ifht(A, dln, mu, offset=0.0, bias=0.0):
        fht_typing(
            A=A,
            dln=dln,
            mu=mu,
            offset=offset,
            bias=bias,
        )

        dtype = as_supported_real(A.dtype)
        if isinstance(A.dtype, types.Complex):
            raise TypingError("The 1st argument 'A' must be a real array.")

        def impl(A, dln, mu, offset=0.0, bias=0.0):
            A = np.asarray(A, dtype=dtype)
            dln = dtype(dln)
            mu = dtype(mu)
            offset = dtype(offset)
            bias = dtype(bias)

            n = A.shape[-1]
            if bias != 0:
                j_c = (n - 1) / 2
                j = np.arange(n)
                A = A * np.exp(bias * ((j - j_c) * dln + offset))
                A = np.asarray(A, dtype=dtype)
            u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
            a = _ifhtq(A, u)
            if bias != 0:
                a /= np.exp(-bias * (j - j_c) * dln)
            return a

        return impl

    @overload(scipy.fft.fftshift)
    def scipy_fftshift(x, axes=None):
        return fftshift(x, axes)

    @overload(scipy.fft.ifftshift)
    def scipy_ifftshift(x, axes=None):
        return ifftshift(x, axes)

    @overload(scipy.fft.fftfreq)
    def scipy_fftfreq(n, d=1.0, *, xp=None, device=None):
        fftfreq_typing(n=n, d=d, xp=xp, device=device)

        if isinstance(xp, types.Module) and xp.pymod is not numpy:
            raise TypingError("Only NumPy namespace is supported.")

        numpy_impl = register_jitable(numpy_fftfreq_impl)

        def impl(n, d=1.0, xp=None, device=None):
            return numpy_impl(n, d, device)

        return impl

    @overload(scipy.fft.rfftfreq)
    def scipy_rfftfreq(n, d=1.0, *, xp=None, device=None):
        fftfreq_typing(n=n, d=d, xp=xp, device=device)

        if isinstance(xp, types.Module) and xp.pymod is not numpy:
            raise TypingError("Only NumPy namespace is supported.")

        numpy_impl = register_jitable(numpy_rfftfreq_impl)

        def impl(n, d=1.0, xp=None, device=None):
            return numpy_impl(n, d, device)

        return impl

    @overload(scipy.fft.fhtoffset)
    def scipy_fhtoffset(dln, mu, initial=0.0, bias=0.0):
        fht_typing(dln=dln, mu=mu, initial=initial, bias=bias)

        def impl(dln, mu, initial=0.0, bias=0.0):
            lnkr = initial
            q = bias
            xp = (mu + 1 + q) / 2
            xm = (mu + 1 - q) / 2
            y = np.pi / (2 * dln)
            zp = special.loggamma(xp + 1j * y)
            zm = special.loggamma(xm + 1j * y)
            arg = (np.log(2) - lnkr) / dln + (zp.imag + zm.imag) / np.pi
            return lnkr + (arg - np.round(arg)) * dln

        return impl

    overload(scipy.fft.next_fast_len)(next_fast_len)
    overload(scipy.fft.prev_fast_len)(prev_fast_len)
