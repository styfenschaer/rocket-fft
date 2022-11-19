import ctypes
from functools import partial
from pathlib import Path

import numba as nb
import numpy as np
import scipy.fft
from llvmlite import ir
from numba import TypingError
from numba.core import cgutils, types
from numba.core.cgutils import get_or_insert_function
from numba.core.config import NUMBA_NUM_THREADS as _cpu_count
from numba.core.typing import signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import intrinsic, overload, register_jitable
from numba.np.arrayobj import make_array
from numba.np.numpy_support import is_nonelike, type_can_asarray

# TODO
# 1. Write additional tests to check if typing works correctly

_path = Path(__file__).parent / '_pocketfft'
_dll = ctypes.CDLL(str(_path))


# Casting rules lookup table
# These rules may differ to what Scipy/Numpy does
_as_complex_lut = {
    types.complex64: types.complex64,
    types.complex128: types.complex128,
    types.float32: types.complex64,
    types.float64: types.complex128,
    types.int8: types.complex64,
    types.int16: types.complex64,
    types.int32: types.complex64,
    types.int64: types.complex128,
    types.uint8: types.complex64,
    types.uint16: types.complex64,
    types.uint32: types.complex64,
    types.uint64: types.complex128,
    types.bool_: types.complex64,
    types.byte: types.complex64,
}
_as_float_lut = {key: val.underlying_float
                 for key, val in _as_complex_lut.items()}


def _as_supported_type(lut, dtype):
    ty = lut.get(dtype, None)
    if not ty is None:
        return ty
    keys = tuple(lut.keys())
    raise TypingError(f'unsupported dtype {dtype}; supported are {keys}')


as_supported_complex = partial(_as_supported_type, _as_complex_lut)
as_supported_float = partial(_as_supported_type, _as_float_lut)


ll_size_t = ir.IntType(64)
ll_int64 = ir.IntType(64)
ll_double = ir.DoubleType()
ll_bool = ir.IntType(1)
ll_voidptr = cgutils.voidptr_t
ll_void = ir.VoidType()
void = types.void
size_t = types.size_t


class _PocketfftDispatcher:
    _ll_complex = ir.FunctionType(ll_void, [ll_size_t,  # ndim
                                            ll_voidptr,  # ain
                                            ll_voidptr,  # aout
                                            ll_voidptr,  # axes
                                            ll_bool,  # forward
                                            ll_double,  # fct
                                            ll_size_t])  # nthreads
    ll_c2c_internal = _ll_complex
    ll_r2c_internal = _ll_complex
    ll_c2r_internal = _ll_complex

    _ll_real = ir.FunctionType(ll_void, [ll_size_t,  # ndim
                                         ll_voidptr,  # ain
                                         ll_voidptr,  # aout
                                         ll_voidptr,  # axes
                                         ll_int64,  # type
                                         ll_double,  # fct
                                         ll_bool,  # ortho
                                         ll_size_t])  # nthreads
    ll_dct_internal = _ll_real
    ll_dst_internal = _ll_real

    ll_good_size_internal = ir.FunctionType(ll_size_t, [ll_size_t,  # target
                                                        ll_bool])  # real

    def __call__(self, builder, fname, args):
        ftype = getattr(self, 'll_' + fname)
        fn = get_or_insert_function(builder.module, ftype, fname)
        return builder.call(fn, args)


ll_pktfft = _PocketfftDispatcher()


def array_as_voidptr(context, builder, ary_t, ary):
    ary = make_array(ary_t)(context, builder, ary)
    ptr = ary._getpointer()
    return builder.bitcast(ptr, ll_voidptr)


def build_complex_transform(fname):
    def impl(typingctx, ain, aout, axes, forward, fct, nthreads):

        def codegen(context, builder, sig, args):
            ain, aout, axes, forward, fct, nthreads = args
            ain_t, aout_t, axes_t, *_ = sig.args

            ndim = ll_size_t(ain_t.ndim)
            ain_ptr = array_as_voidptr(context, builder, ain_t,  ain)
            aout_ptr = array_as_voidptr(context, builder, aout_t, aout)
            axes_ptr = array_as_voidptr(context, builder, axes_t, axes)

            ll_pktfft(builder, fname, (ndim, ain_ptr, aout_ptr,
                      axes_ptr, forward, fct, nthreads))

        sig = signature(void, ain, aout, axes, forward, fct, nthreads)
        return sig, codegen

    impl.__name__ = fname
    return intrinsic(impl)


c2c_internal = build_complex_transform('c2c_internal')
r2c_internal = build_complex_transform('r2c_internal')
c2r_internal = build_complex_transform('c2r_internal')


def build_real_transform(fname):
    def impl(typingctx, ain, aout, axes, type, fct, ortho, nthreads):

        def codegen(context, builder, sig, args):
            ain, aout, axes, type, fct, ortho, nthreads = args
            ain_t, aout_t, axes_t, *_ = sig.args

            ndim = ll_size_t(ain_t.ndim)
            ain_ptr = array_as_voidptr(context, builder, ain_t,  ain)
            aout_ptr = array_as_voidptr(context, builder, aout_t, aout)
            axes_ptr = array_as_voidptr(context, builder, axes_t, axes)

            ll_pktfft(builder, fname, (ndim, ain_ptr, aout_ptr,
                      axes_ptr, type, fct, ortho, nthreads))

        sig = signature(void, ain, aout, axes, type, fct, ortho, nthreads)
        return sig, codegen

    impl.__name__ = fname
    return intrinsic(impl)


dst_internal = build_real_transform('dst_internal')
dct_internal = build_real_transform('dct_internal')


@intrinsic
def good_size_internal(typingctx, n, real):

    def codegen(context, builder, sig, args):
        ret = ll_pktfft(builder, 'good_size_internal', args)
        return ret

    sig = signature(size_t, n, real)
    return sig, codegen


def build_get_fct_complex(forward, norm):
    @register_jitable
    def mul_axes(x, axes):
        n = 1.0
        for ax in axes:
            n *= x.shape[ax]
        return n

    if is_nonelike(norm):
        def impl(x, axes, norm):
            return 1.0 if forward else 1.0 / mul_axes(x, axes)

    else:
        def impl(x, axes, norm):
            if norm == 'backward':
                return 1.0 if forward else 1.0 / mul_axes(x, axes)
            elif norm == 'ortho':
                return 1.0 / np.sqrt(mul_axes(x, axes))
            elif norm == 'forward':
                return 1.0 / mul_axes(x, axes) if forward else 1.0
            raise ValueError(
                "Invalid norm value; should be 'backward','ortho' or 'forward'.")

    return register_jitable(impl)


def build_get_nthreads(workers):
    if is_nonelike(workers):
        return register_jitable(lambda workers: 1)

    def impl(workers):
        if workers > 0:
            return workers
        if workers == 0:
            raise ValueError('workers must not be zero')
        if workers < 0 and workers >= -_cpu_count:
            return workers + 1 + _cpu_count
        raise ValueError('workers value out of range')

    return register_jitable(impl)


@register_jitable
def assert_unique_axes(axes):
    if axes.size != np.unique(axes).size:
        raise ValueError('all axes must be unique')


@register_jitable
def wraparound_axes(x, axes):
    for i, ax in enumerate(axes):
        if ax < 0:
            axes[i] += x.ndim
        elif ax > x.ndim:
            raise ValueError('axes exceeds dimensionality of input')


def build_get_ndshape_and_axes(s, axes):
    if is_nonelike(s) and is_nonelike(axes):
        def impl(x, s, axes):
            # `axes` not provided, transform all axes
            axes = np.arange(x.ndim)
            return s, axes

    elif is_nonelike(s) and not is_nonelike(axes):
        def impl(x, s, axes):
            axes = np.asarray(axes)
            assert_unique_axes(axes)
            wraparound_axes(x, axes)
            return s, axes

    elif not is_nonelike(s) and is_nonelike(axes):
        def impl(x, s, axes):
            s = np.asarray(s)
            if s.min() < 1:
                raise ValueError('invalid number of data points specified')
            if s.size > x.ndim:
                raise ValueError('shape requires more axes than are present')
            # `axes` not provided, transform last `len(s)` axes
            axes = np.arange(x.ndim-s.size, x.ndim)
            return s, axes

    else:
        def impl(x, s, axes):
            s = np.asarray(s)
            if s.min() < 1:
                raise ValueError('invalid number of data points specified')
            axes = np.asarray(axes)
            assert_unique_axes(axes)
            wraparound_axes(x, axes)
            if s.size != axes.size:
                raise ValueError('when given, axes and shape arguments'
                                 ' have to be of the same length')
            return s, axes

    return register_jitable(impl)


def build_zeropad_or_crop(s):
    if is_nonelike(s):
        return register_jitable(lambda x, s, axes: x)

    def impl(x, s, axes):
        shape = x.shape
        for newsize, ax in zip(s, axes):
            shape = tuple_setitem(shape, ax, newsize)
        xout = np.zeros(shape, dtype=x.dtype)
        for i, (s1, s2) in enumerate(zip(x.shape, xout.shape)):
            # smaller axis is decisive how many elements are copied
            shape = tuple_setitem(shape, i, min(s1, s2))
        for index in np.ndindex(shape):
            xout[index] = x[index]
        return xout

    return register_jitable(impl)


def build_astype(hastype, reqtype):
    # Numba does not support `x.astype(reqtype, copy=False)`
    # so we handle this at compile time.
    if hastype == reqtype:
        return register_jitable(lambda x: x)
    else:
        return register_jitable(lambda x: x.astype(reqtype))


def build_alloc_new(s, hastype, reqtype):
    # We don't allocate a new array if:
    # 1. overwrite was requested (runtime check)
    # 2. we already have a new array because we had to cast it (compile time check)
    # 3. because the array has been zero padded/truncated (compile time check)
    if hastype != reqtype or not is_nonelike(s):
        return register_jitable(lambda x, overwrite_x: x)

    def impl(x, overwrite_x):
        if overwrite_x:
            return x
        xout = np.empty_like(x)
        return xout

    return register_jitable(impl)


def c2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    get_fct = build_get_fct_complex(forward, norm)
    get_nthreads = build_get_nthreads(workers)
    get_ndshape_and_axes = build_get_ndshape_and_axes(s, axes)
    zeropad_or_crop = build_zeropad_or_crop(s)

    argtype = as_supported_complex(x.dtype)
    rettype = argtype

    astype = build_astype(x.dtype, argtype)
    alloc_new = build_alloc_new(s, x.dtype, rettype)

    # TODO: take advantage of symmetry for real valued data
    # TODO: copies data twice if `s` is provided and `x.dtype != argtype`
    def impl(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
        s, axes = get_ndshape_and_axes(x, s, axes)
        x = astype(x)
        x = zeropad_or_crop(x, s, axes)
        xout = alloc_new(x, overwrite_x)
        fct = get_fct(x, axes, norm)
        nthreads = get_nthreads(workers)
        c2c_internal(x, xout, axes, forward, fct, nthreads)
        return xout

    return impl


def check_arguments_ndim(x, s, axes):
    if not isinstance(x, types.Array):
        raise TypingError("The first argument 'x' must be an array")
    if not is_nonelike(s):
        if not (type_can_asarray(s) and isinstance(s.dtype, types.Integer)):
            raise TypingError(
                "The second argument 's' must be sequence of integers or None")
    if not is_nonelike(axes):
        if not (type_can_asarray(axes) and isinstance(axes.dtype, types.Integer)):
            raise TypingError(
                "The third argument 'axes' must be sequence of integers or None")


@overload(np.fft.fftn)
@overload(scipy.fft.fftn)
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return c2cn(True, x, s, axes, norm, overwrite_x, workers)


@overload(np.fft.ifftn)
@overload(scipy.fft.ifftn)
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return c2cn(False, x, s, axes, norm, overwrite_x, workers)


def patch_2dim(func, forward, x, n, axis, norm, overwrite_x, workers):
    func_impl = func(forward, x, n, axis, norm, overwrite_x, workers)
    func_impl = register_jitable(func_impl)

    # Typing signature must match signature of implementation. 2-dim case has
    # different default axes.
    def impl(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
        return func_impl(x, s, axes, norm, overwrite_x, workers)

    return impl


@overload(np.fft.fft2)
@overload(scipy.fft.fft2)
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return patch_2dim(c2cn, True, x, s, axes, norm, overwrite_x, workers)


@overload(np.fft.ifft2)
@overload(scipy.fft.ifft2)
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return patch_2dim(c2cn, False, x, s, axes, norm, overwrite_x, workers)


def as_integer_1tuple_or_none(val, msg):
    if isinstance(val, types.Integer):
        return types.UniTuple(val, 1)
    if isinstance(val, int):
        ty = nb.typeof(val)
        return types.UniTuple(ty, 1)
    if is_nonelike(val):
        return val
    raise TypingError(msg)


def check_1dim_arguments(x, n, axis):
    msg = "The second argument 'n' must be an integer or None"
    s = as_integer_1tuple_or_none(n, msg)
    msg = "The third argument 'axis' must be an integer or None"
    axes = as_integer_1tuple_or_none(axis, msg)
    check_arguments_ndim(x, s, axes)
    return x, s, axes


def build_as_none_or_array(arg):
    identity = register_jitable(lambda val: val)
    asarray = register_jitable(lambda val: np.array([val]))
    return identity if is_nonelike(arg) else asarray


def patch_1dim_complex(func, forward, x, n, axis, norm, overwrite_x, workers):
    func_impl = func(forward, x, n, axis, norm, overwrite_x, workers)
    func_impl = register_jitable(func_impl)

    axes_as_none_or_array = build_as_none_or_array(axis)
    s_as_none_or_array = build_as_none_or_array(n)

    def impl(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
        axes = axes_as_none_or_array(axis)
        s = s_as_none_or_array(n)
        return func_impl(x, s, axes, norm, overwrite_x, workers)

    return impl


# TODO: One-dimensional transform as special case of n-dimensional transform
# results in unnecessarily long compile times. A seperate implementation would
# most likely halfen the compile time for the default case.
# `fftn` on one-dimensional also has less overhead than `fft`!
@overload(np.fft.fft)
@overload(scipy.fft.fft)
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_complex(c2cn, True, x, n, axis, norm, overwrite_x, workers)


@overload(np.fft.ifft)
@overload(scipy.fft.ifft)
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_complex(c2cn, False, x, n, axis, norm, overwrite_x, workers)


@register_jitable
def inc_or_dec_shape(shape, axes, inc):
    idx = axes[-1]
    if inc:
        newval = (shape[idx] - 1) * 2
    else:
        newval = (shape[idx] // 2) + 1
    shape = tuple_setitem(shape, idx, newval)
    return shape


def r2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    get_fct = build_get_fct_complex(forward, norm)
    get_nthreads = build_get_nthreads(workers)
    get_ndshape_and_axes = build_get_ndshape_and_axes(s, axes)
    zeropad_or_crop = build_zeropad_or_crop(s)

    if hasattr(x.dtype, 'underlying_float'):
        raise TypingError(f'unsupported dtype {x.dtype}')
    argtype = as_supported_float(x.dtype)
    rettype = as_supported_complex(argtype)

    astype = build_astype(x.dtype, argtype)

    # TODO: copies data twice if `s` is provided
    def impl(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
        s, axes = get_ndshape_and_axes(x, s, axes)
        x = astype(x)
        x = zeropad_or_crop(x, s, axes)
        shape = inc_or_dec_shape(x.shape, axes, inc=False)
        xout = np.empty(shape, dtype=rettype)
        fct = get_fct(x, axes, norm)
        nthreads = get_nthreads(workers)
        r2c_internal(x, xout, axes, forward, fct, nthreads)
        return xout

    return impl


@overload(np.fft.rfftn)
@overload(scipy.fft.rfftn)
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return r2cn(True, x, s, axes, norm, overwrite_x, workers)


@overload(scipy.fft.ihfftn)
def ihfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return r2cn(False, x, s, axes, norm, overwrite_x, workers)


@overload(np.fft.rfft2)
@overload(scipy.fft.rfft2)
def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return patch_2dim(r2cn, True, x, s, axes, norm, overwrite_x, workers)


@overload(scipy.fft.ihfft2)
def ihfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return patch_2dim(r2cn, False, x, s, axes, norm, overwrite_x, workers)


# TODO: See comment of `fft`.
@overload(np.fft.rfft)
@overload(scipy.fft.rfft)
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_complex(r2cn, True, x, n, axis, norm, overwrite_x, workers)


# TODO: See comment of `fft`.
@overload(np.fft.ihfft)
@overload(scipy.fft.ihfft)
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_complex(r2cn, False, x, n, axis, norm, overwrite_x, workers)


def build_resize(s):
    if is_nonelike(s):
        return register_jitable(lambda shape, x, s, axes: shape)

    def impl(shape, x, s, axes):
        for i, ax in enumerate(axes):
            if ax == x.ndim-1:
                shape = tuple_setitem(shape, x.ndim-1, s[i])
        return shape

    return register_jitable(impl)


def c2rn(forward, x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    get_fct = build_get_fct_complex(forward, norm)
    get_nthreads = build_get_nthreads(workers)
    get_ndshape_and_axes = build_get_ndshape_and_axes(s, axes)
    zeropad_or_crop = build_zeropad_or_crop(s)
    # If `s` is provided we do additional resizing to
    # obtain the requested output shape.
    resize = build_resize(s)

    argtype = as_supported_complex(x.dtype)
    rettype = as_supported_float(argtype)

    astype = build_astype(x.dtype, argtype)

    # TODO: copies data twice if `s` is provided
    def impl(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
        s, axes = get_ndshape_and_axes(x, s, axes)
        x = astype(x)
        xin = zeropad_or_crop(x, s, axes)
        shape = inc_or_dec_shape(x.shape, axes, inc=True)
        shape = resize(shape, x, s, axes)
        xout = np.empty(shape, dtype=rettype)
        fct = get_fct(xout, axes, norm)
        nthreads = get_nthreads(workers)
        c2r_internal(xin, xout, axes, forward, fct, nthreads)
        return xout

    return impl


@overload(np.fft.irfftn)
@overload(scipy.fft.irfftn)
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return c2rn(False, x, s, axes, norm, overwrite_x, workers)


@overload(scipy.fft.hfftn)
def hfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return c2rn(True, x, s, axes, norm, overwrite_x, workers)


@overload(np.fft.irfft2)
@overload(scipy.fft.irfft2)
def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return patch_2dim(c2rn, False, x, s, axes, norm, overwrite_x, workers)


@overload(scipy.fft.hfft2)
def hfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    check_arguments_ndim(x, s, axes)
    return patch_2dim(c2rn, True, x, s, axes, norm, overwrite_x, workers)


# TODO: See comment of `fft`.
@overload(np.fft.irfft)
@overload(scipy.fft.irfft)
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_complex(c2rn, False, x, n, axis, norm, overwrite_x, workers)


# TODO: See comment of `fft`.
@overload(np.fft.hfft)
@overload(scipy.fft.hfft)
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_complex(c2rn, True, x, n, axis, norm, overwrite_x, workers)


def build_get_type(forward):
    if forward:
        return register_jitable(lambda type: type)

    def impl(type):
        if type == 2:
            return 3
        if type == 3:
            return 2
        return type

    return register_jitable(impl)


def build_get_ortho(orthogonalize):
    if not is_nonelike(orthogonalize):
        return register_jitable(lambda norm, ortho:  ortho)

    def impl(norm, ortho):
        if norm == 'ortho':
            return True
        return False

    return register_jitable(impl)


def build_get_fct_real(forward, norm):
    # For real transforms (DCT and DST) the norm. factor is different.
    @register_jitable
    def mul_axes(x, axes, delta):
        n = 1.0
        for ax in axes:
            n *= 2.0 * (x.shape[ax]+delta)
        return n

    if is_nonelike(norm):
        def impl(x, axes, norm, delta):
            return 1.0 if forward else 1.0 / mul_axes(x, axes, delta)

    else:
        def impl(x, axes, norm, delta):
            if norm == 'backward':
                return 1.0 if forward else 1.0 / mul_axes(x, axes, delta)
            elif norm == 'ortho':
                return 1.0 / np.sqrt(mul_axes(x, axes, delta))
            elif norm == 'forward':
                return 1.0 / mul_axes(x, axes, delta) if forward else 1.0
            raise ValueError(
                "Invalid norm value; should be 'backward','ortho' or 'forward'.")

    return register_jitable(impl)


# TODO: `r2rn` has generally quite a long compile time. Can we improve that?
def r2rn(transform, forward, x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    get_fct = build_get_fct_real(forward, norm)
    get_nthreads = build_get_nthreads(workers)
    get_ndshape_and_axes = build_get_ndshape_and_axes(s, axes)
    zeropad_or_crop = build_zeropad_or_crop(s)
    get_type = build_get_type(forward)
    get_ortho = build_get_ortho(orthogonalize)

    if hasattr(x.dtype, 'underlying_float'):
        argtype = as_supported_complex(x.dtype)

        # Transform real and imaginary part seperately if input is complex.
        @register_jitable
        def do_transform(xin, xout, axes, type, fct, ortho, nthreads):
            transform(xin.real, xout.real, axes, type, fct, ortho, nthreads)
            transform(xin.imag, xout.imag, axes, type, fct, ortho, nthreads)
    else:
        argtype = as_supported_float(x.dtype)
        do_transform = register_jitable(transform)

    rettype = argtype

    astype = build_astype(x.dtype, rettype)
    alloc_new = build_alloc_new(s, x.dtype, rettype)

    # Delta depends on whether it's DCT or DST!
    delta = 1.0 if 'dst' in transform.__name__ else -1.0

    # TODO: copies data twice if `s` is provided and `x.dtype != argtype`
    def impl(x, type=2, s=None, axes=None, norm=None,
             overwrite_x=False, workers=None, orthogonalize=None):
        s, axes = get_ndshape_and_axes(x, s, axes)
        x = astype(x)
        x = zeropad_or_crop(x, s, axes)
        xout = alloc_new(x, overwrite_x)
        type = get_type(type)
        if type == 1:
            fct = get_fct(xout, axes, norm, delta)
        else:
            fct = get_fct(xout, axes, norm, 0.0)
        ortho = get_ortho(norm, orthogonalize)
        nthreads = get_nthreads(workers)
        do_transform(x, xout, axes, type, fct, ortho, nthreads)
        return xout

    return impl


@overload(scipy.fft.dctn)
def dctn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    check_arguments_ndim(x, s, axes)
    return r2rn(dct_internal, True, x, type, s, axes, norm,
                overwrite_x, workers, orthogonalize)


@overload(scipy.fft.idctn)
def idctn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, orthogonalize=None):
    check_arguments_ndim(x, s, axes)
    return r2rn(dct_internal, False, x, type, s, axes, norm,
                overwrite_x, workers, orthogonalize)


@overload(scipy.fft.dstn)
def dstn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    check_arguments_ndim(x, s, axes)
    return r2rn(dst_internal, True, x, type, s, axes, norm,
                overwrite_x, workers, orthogonalize)


@overload(scipy.fft.idstn)
def idstn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, orthogonalize=None):
    check_arguments_ndim(x, s, axes)
    return r2rn(dst_internal, False, x, type, s, axes, norm,
                overwrite_x, workers, orthogonalize)


def patch_1dim_real(func, transform, forward, x, type, n, axis, norm,
                    overwrite_x, workers, orthogonalize):
    func_impl = func(transform, forward, x, type, n, axis, norm,
                     overwrite_x, workers, orthogonalize)
    func_impl = register_jitable(func_impl)

    axes_as_none_or_array = build_as_none_or_array(axis)
    s_as_none_or_array = build_as_none_or_array(n)

    def impl(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
             workers=None, orthogonalize=None):
        axes = axes_as_none_or_array(axis)
        s = s_as_none_or_array(n)
        return func_impl(x, type, s, axes, norm, overwrite_x,
                         workers, orthogonalize)

    return impl


# TODO: See comment of `fft`.
@overload(scipy.fft.dct)
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, orthogonalize=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_real(r2rn, dct_internal, True, x, type, n, axis, norm,
                           overwrite_x, workers, orthogonalize)


# TODO: See comment of `fft`.
@overload(scipy.fft.idct)
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_real(r2rn, dct_internal, False, x, type, n, axis, norm,
                           overwrite_x, workers, orthogonalize)


# TODO: See comment of `fft`.
@overload(scipy.fft.dst)
def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, orthogonalize=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_real(r2rn, dst_internal, True, x, type, n, axis, norm,
                           overwrite_x, workers, orthogonalize)


# TODO: See comment of `fft`.
@overload(scipy.fft.idst)
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    x, n, axis = check_1dim_arguments(x, n, axis)
    return patch_1dim_real(r2rn, dst_internal, False, x, type, n, axis, norm,
                           overwrite_x, workers, orthogonalize)


@overload(np.roll)
def roll(a, shift, axis=None):
    # TODO: The multidimensional case is extremly inefficient!
    # I only implemented a naiv approach.
    if not isinstance(a, types.Array):
        raise TypingError("The first argument 'a' must be an array")
    if not isinstance(shift, types.Integer):
        if not (type_can_asarray(shift) and isinstance(shift.dtype, types.Integer)):
            raise TypingError(
                "The second argument 'shift' must be a sequences of integers or an integer")
    if not is_nonelike(axis):
        if not isinstance(axis, types.Integer):
            if not (type_can_asarray(axis) and isinstance(axis.dtype, types.Integer)):
                raise TypingError("The second argument 'axis' must be a sequences"
                                  "of integers an integer or None")
        if type_can_asarray(axis) != type_can_asarray(shift):
            raise TypingError('If axis is provided, shift and axis must both'
                              ' be integers or integer sequences of equal length')

    if type_can_asarray(shift):
        make_shift = register_jitable(lambda shift: shift)
    else:
        make_shift = register_jitable(lambda shift: np.asarray([shift]))

    if is_nonelike(axis):
        def impl(a, shift, axis=None):
            shift = make_shift(shift)

            shape = a.shape
            a = a.ravel()
            r = np.empty_like(a)
            for sh in shift:
                if sh >= 0:
                    for i in range(r.size):
                        r[i] = a[i-sh]
                else:
                    for i in range(r.size):
                        r[i+sh] = a[i]
            r = r.reshape(shape)
            return r

    else:
        def impl(a, shift, axis=None):
            axis, shift = np.broadcast_arrays(axis, shift)
            # make copy because `axis` is readonly but we
            # eventually need to write it.
            axis = axis.copy()
            wraparound_axes(a, axis)

            shifts = {ax: 0 for ax in range(a.ndim)}
            # Sum shift along same axis; only roll each axis once
            for ax, sh in zip(axis, shift):
                shifts[ax] += sh
            for ax, sh in shifts.items():
                if sh == 0:
                    continue
                # If shift is longer than axis, we don't need to
                # roll the full axis length
                sh %= a.shape[ax]
                # This is a hack so that we later can
                # roll a positive value
                if sh <= 0:
                    shifts[ax] = sh
                else:
                    shifts[ax] = sh - a.shape[ax]
            # Remove zero shifts
            shifts = {ax: sh for ax, sh in shifts.items() if sh}

            # TODO: This part is not efficient.
            # We actually only want to copy but do a lot of work with the
            # `tuple_setitem`. Maybe we also trigger boundschecking?
            # Alternatively we can use `np.swapaxes` but this seems to move
            # data -> not good either.
            r = np.empty_like(a)
            for index, ai in np.ndenumerate(a):
                for ax, sh in shifts.items():
                    val = index[ax] + sh
                    index = tuple_setitem(index, ax, val)
                r[index] = ai

            return r

    return impl


# Both overloads only for clarity; Scipy uses Numpy internally.
@overload(np.fft.fftshift)
@overload(scipy.fft.fftshift)
def fftshift(x, axes=None):
    if not isinstance(x, types.Array):
        raise TypingError("The first argument 'x' must be an array")
    if not is_nonelike(axes) and not isinstance(axes, types.Integer):
        if not (type_can_asarray(axes) and isinstance(axes.dtype, types.Integer)):
            raise TypingError("The second argument 'axes' must be sequence of"
                              "integers an integer or None")

    if is_nonelike(axes):
        def impl(x, axes=None):
            axes = x.shape
            shift = x.shape
            for i, dim in enumerate(x.shape):
                shift = tuple_setitem(shift, i, dim//2)
                axes = tuple_setitem(axes, i, i)
            return np.roll(x, shift, axes)

    elif isinstance(axes, types.Integer):
        def impl(x, axes=None):
            shift = x.shape[axes] // 2
            return np.roll(x, shift, axes)

    else:
        def impl(x, axes=None):
            shift = x.shape[:len(axes)]
            for i, ax in enumerate(axes):
                shift = tuple_setitem(shift, i, x.shape[ax]//2)
            return np.roll(x, shift, axes)

    return impl


@overload(np.fft.ifftshift)
@overload(scipy.fft.ifftshift)
def ifftshift(x, axes=None):
    if not isinstance(x, types.Array):
        raise TypingError("The first argument 'x' must be an array")
    if not is_nonelike(axes) and not isinstance(axes, types.Integer):
        if not (type_can_asarray(axes) and isinstance(axes.dtype, types.Integer)):
            raise TypingError("The second argument 'axes' must be sequence of"
                              "integers an integer or None")

    if is_nonelike(axes):
        def impl(x, axes=None):
            axes = x.shape
            shift = x.shape
            for i, dim in enumerate(x.shape):
                shift = tuple_setitem(shift, i, -(dim // 2))
                axes = tuple_setitem(axes, i, i)
            return np.roll(x, shift, axes)

    elif isinstance(axes, types.Integer):
        def impl(x, axes=None):
            shift = -(x.shape[axes] // 2)
            return np.roll(x, shift, axes)

    else:
        def impl(x, axes=None):
            shift = x.shape[:len(axes)]
            for i, ax in enumerate(axes):
                shift = tuple_setitem(shift, i, -(x.shape[ax] // 2))
            return np.roll(x, shift, axes)

    return impl


@overload(np.fft.fftfreq)
@overload(scipy.fft.fftfreq)
def fftfreq(n, d=1.0):
    if not isinstance(n, types.Integer):
        raise TypingError(
            "The first argument 'n' must be an integer")
    if not isinstance(n, types.Number):
        raise TypingError(
            "The second argument 'd' must be a scaler")

    def impl(n, d=1.0):
        val = 1.0 / (n * d)
        results = np.empty(n, dtype=np.int64)
        N = (n-1)//2 + 1
        p1 = np.arange(N)
        results[:N] = p1
        p2 = np.arange(-(n//2), 0)
        results[N:] = p2
        return results * val

    return impl


@overload(np.fft.rfftfreq)
@overload(scipy.fft.rfftfreq)
def rfftfreq(n, d=1.0):
    if not isinstance(n, types.Integer):
        raise TypingError(
            "The first argument 'n' must be an integer")
    if not isinstance(n, types.Number):
        raise TypingError(
            "The second argument 'd' must be a scaler")

    def impl(n, d=1.0):
        val = 1.0/(n*d)
        N = n//2 + 1
        results = np.arange(N)
        return results * val

    return impl


@overload(scipy.fft.next_fast_len)
def next_fast_len(target, real):
    if not isinstance(target, types.Integer):
        raise TypingError("First argument 'target' must be an integer.")

    def impl(target, real):
        if target < 0:
            raise ValueError('Target cannot be negative')
        return good_size_internal(target, real)

    return impl
