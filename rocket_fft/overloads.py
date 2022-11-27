import numpy.fft
import scipy.fft
import inspect
import ctypes
from functools import partial, wraps
from pathlib import Path

import numba as nb
import numpy as np
from llvmlite import ir
from numba import TypingError, generated_jit
from numba.core import cgutils, types
from numba.core.cgutils import get_or_insert_function
from numba.core.config import NUMBA_NUM_THREADS as _cpu_count
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import intrinsic, overload, register_jitable
from numba.np.arrayobj import make_array
from numba.np.numpy_support import is_nonelike

# TODO:
# can optimize for literal values?

ll_size_t = ir.IntType(64)
ll_int64 = ir.IntType(64)
ll_double = ir.DoubleType()
ll_bool = ir.IntType(1)
ll_voidptr = cgutils.voidptr_t
ll_void = ir.VoidType()
void = types.void
size_t = types.size_t


def load_pocketfft():
    path = Path(__file__).parent / "_pocketfft_numba.so"
    dll = ctypes.CDLL(str(path))
    return dll


class _Pocketfft:
    def __init__(self):
        self.dll = load_pocketfft()

    @staticmethod
    def _call_cmplx(fname, builder, args):
        fntype = ir.FunctionType(ll_void, [ll_size_t,  # ndim
                                           ll_voidptr,  # ain
                                           ll_voidptr,  # aout
                                           ll_voidptr,  # axes
                                           ll_bool,  # forward
                                           ll_double,  # fct
                                           ll_size_t])  # nthreads
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

    c2c = partial(_call_cmplx.__func__, 'numba_c2c')
    r2c = partial(_call_cmplx.__func__, 'numba_r2c')
    c2r = partial(_call_cmplx.__func__, 'numba_c2r')

    @staticmethod
    def _call_real(fname, builder, args):
        fntype = ir.FunctionType(ll_void,  [ll_size_t,  # ndim
                                            ll_voidptr,  # ain
                                            ll_voidptr,  # aout
                                            ll_voidptr,  # axes
                                            ll_int64,  # type
                                            ll_double,  # fct
                                            ll_bool,  # ortho
                                            ll_size_t])  # nthreads
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

    dct = partial(_call_real.__func__, 'numba_dct')
    dst = partial(_call_real.__func__, 'numba_dst')

    @staticmethod
    def good_size(builder, args):
        fname = 'numba_good_size'
        fntype = ir.FunctionType(ll_size_t,  [ll_size_t,  # target
                                              ll_bool])  # real
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)


ll_pocketfft = _Pocketfft()


def array_as_voidptr(context, builder, ary_t, ary):
    ary = make_array(ary_t)(context, builder, ary)
    ptr = ary._getpointer()
    return builder.bitcast(ptr, ll_voidptr)


def _numba_cmplx(func, typingctx, ain, aout, axes, forward, fct, nthreads):
    def codegen(context, builder, sig, args):
        ain, aout, axes, forward, fct, nthreads = args
        ain_t, aout_t, axes_t, *_ = sig.args

        ndim = ll_size_t(ain_t.ndim)
        ain_ptr = array_as_voidptr(context, builder, ain_t, ain)
        aout_ptr = array_as_voidptr(context, builder, aout_t, aout)
        ax_ptr = array_as_voidptr(context, builder, axes_t, axes)

        args = (ndim, ain_ptr, aout_ptr, ax_ptr, forward, fct, nthreads)
        func(builder, args)

    sig = void(ain, aout, axes, forward, fct, nthreads)
    return sig, codegen


numba_c2c = intrinsic(partial(_numba_cmplx, ll_pocketfft.c2c))
numba_r2c = intrinsic(partial(_numba_cmplx, ll_pocketfft.r2c))
numba_c2r = intrinsic(partial(_numba_cmplx, ll_pocketfft.c2r))


def _numba_real(func, typingctx, ain, aout, axes, type, fct, ortho, nthreads):
    def codegen(context, builder, sig, args):
        ain, aout, axes, type, fct, ortho, nthreads = args
        ain_t, aout_t, axes_t, *_ = sig.args

        ndim = ll_size_t(ain_t.ndim)
        ain_ptr = array_as_voidptr(context, builder, ain_t, ain)
        aout_ptr = array_as_voidptr(context, builder, aout_t, aout)
        ax_ptr = array_as_voidptr(context, builder, axes_t, axes)

        args = (ndim, ain_ptr, aout_ptr, ax_ptr, type, fct, ortho, nthreads)
        func(builder, args)

    sig = void(ain, aout, axes, type, fct, ortho, nthreads)
    return sig, codegen


numba_dst = intrinsic(partial(_numba_real, ll_pocketfft.dst))
numba_dct = intrinsic(partial(_numba_real, ll_pocketfft.dct))


@intrinsic
def numba_good_size(typingctx, n, real):
    def codegen(context, builder, sig, args):
        ret = ll_pocketfft.good_size(builder, args)
        return ret

    sig = size_t(n, real)
    return sig, codegen


# Casting rules lookup table
# These rules may differ to Scipy/Numpy
_as_cmplx_lut = {
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
                 for key, val in _as_cmplx_lut.items()}


def _as_supported_type(lut, dtype):
    ty = lut.get(dtype)
    if ty is not None:
        return ty
    keys = tuple(lut.keys())
    raise TypingError(f"Unsupported dtype {dtype}; supported are {keys}.")


as_supported_cmplx = partial(_as_supported_type, _as_cmplx_lut)
as_supported_float = partial(_as_supported_type, _as_float_lut)


def is_sequence_like(arg):
    seq_like = (types.Array, types.Sequence, types.Tuple,
                types.ListType, types.Set)
    return isinstance(arg, seq_like)


class _TypingCheck:
    __slots__ = ('ty', 'as_one', 'as_seq', 'allow_none', 'msg')

    def __init__(self, ty, as_one, as_seq, allow_none, msg):
        self.ty = ty
        self.as_one = as_one
        self.as_seq = as_seq
        self.allow_none = allow_none
        self.msg = msg

    def __call__(self, val, fmt=None):
        # It's not a numba type -> make it one
        if not hasattr(val, 'cast_python_value'):
            val = nb.typeof(val)
        if self.allow_none and is_nonelike(val):
            return True
        if self.as_one and isinstance(val, self.ty):
            return True
        if self.as_seq and is_sequence_like(val):
            if isinstance(val.dtype, self.ty):
                return True
        if self.msg is not None:
            msg = self.msg.format(fmt)
            raise TypeError(msg)
        return False


# Checks typing of the arguments of the FFT functions
_typing_checkers = {
    'a': _TypingCheck(
        types.Array, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'a' must be an array."),
    'x': _TypingCheck(
        types.Array, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'x' must be an array."),
    'n': _TypingCheck(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 'n' must be an integer."),
    's': _TypingCheck(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 's' must be a sequence of integers."),
    'axis': _TypingCheck(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 'axis' must be an integer."),
    'axes': _TypingCheck(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 'axes' must be a sequence of integers."),
    'norm': _TypingCheck(
        types.UnicodeType, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument 'norm' must be a string."),
    'type': _TypingCheck(
        types.Integer, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'type' must be an integer."),
    'overwrite_x': _TypingCheck(
        types.Boolean, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'overwrite_x' must be a boolean."),
    'workers': _TypingCheck(
        types.Integer, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument 'workers' must be an integer."),
    'orthogonalize': _TypingCheck(
        types.Boolean, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument 'orthogonalize' must be a boolean."),
}


# Helper to format the typing error of the FFT functions
_pos_to_text = {
    0: '1st', 1: '2nd', 2: '3rd', 3: '4th',
    4: '5th', 5: '6th', 6: '7th', 7: '8th',
}


class _FFTBuilder:
    _registry = []
    
    def __init__(self, header, check_typing=True):
        self.header = header
        self.check_typing = check_typing
        self.registry = []

    def __call__(self, func, *args, **kwargs):
        @wraps(self.header)
        def ol_impl(*iargs, **ikwargs):
            kwd = inspect.getcallargs(self.header, *iargs, **ikwargs)
            if self.check_typing:
                self._check_typing(**kwd)
            params = tuple(kwd.values())
            impl = func(params, *args, **kwargs)
            self._patch_co(self.header, impl)
            return wraps(self.header)(impl)

        self.active_impl = ol_impl
        return self

    def register(self, func):
        entry = (func, self.active_impl)
        self.registry.append(entry)
        self._registry.append(entry)
        overload(func)(self.active_impl)

    @staticmethod
    def _patch_co(f0, f1):
        sig_f0 = inspect.signature(f0).parameters.keys()
        sig_f1 = inspect.signature(f1).parameters.keys()
        cov = list(f1.__code__.co_varnames)
        for p0, p1 in zip(tuple(sig_f0), tuple(sig_f1)):
            if p0 != p1:
                idx = cov.index(p1)
                cov[idx] = p0
        cov = tuple(cov)
        f1.__code__ = f1.__code__.replace(co_varnames=cov)

    @staticmethod
    def _check_typing(**kwargs):
        for i, (key, val) in enumerate(kwargs.items()):
            fn = _typing_checkers.get(key)
            if fn is not None:
                pos = _pos_to_text.get(i)
                fn(val, fmt=pos)


@register_jitable
def assert_unique_axes(axes):
    if len(set(axes)) != axes.size:
        raise ValueError("All axes must be unique.")


@register_jitable
def wraparound_axes(x, axes):
    for i, ax in enumerate(axes):
        if ax < 0:
            axes[i] += x.ndim
        elif ax >= x.ndim:
            raise ValueError("Axes exceeds dimensionality of input.")


@generated_jit
def ndshape_and_axes(x, s, axes):
    def asarray_or_none(arg):
        if is_nonelike(arg):
            return register_jitable(lambda val: val)

        def impl(val):
            a = np.asarray(val)
            return np.atleast_1d(a)

        return register_jitable(impl)

    handle_shape = asarray_or_none(s)
    handle_axes = asarray_or_none(axes)

    if is_nonelike(s) and is_nonelike(axes):
        def impl(x, s, axes):
            # `axes` not specified, transform all axes
            axes = np.arange(x.ndim)
            return s, axes

    elif is_nonelike(s) and not is_nonelike(axes):
        def impl(x, s, axes):
            axes = handle_axes(axes)
            assert_unique_axes(axes)
            wraparound_axes(x, axes)
            return s, axes

    elif not is_nonelike(s) and is_nonelike(axes):
        def impl(x, s, axes):
            s = handle_shape(s)
            if s.min() < 1:
                raise ValueError("Invalid number of data points specified.")
            if s.size > x.ndim:
                raise ValueError("Shape requires more axes than are present.")
            # `axes` not specified, transform last `len(s)` axes
            axes = np.arange(x.ndim - s.size, x.ndim)
            return s, axes

    else:
        def impl(x, s, axes):
            s = handle_shape(s)
            if s.min() < 1:
                raise ValueError("Invalid number of data points specified.")
            axes = handle_axes(axes)
            assert_unique_axes(axes)
            wraparound_axes(x, axes)
            if s.size != axes.size:
                raise ValueError("When given, axes and shape arguments"
                                 " have to be of the same length.")
            return s, axes

    return impl


@generated_jit
def zeropad_or_crop(x, s, axes):
    if is_nonelike(s):
        return lambda x, s, axes: x

    def impl(x, s, axes):
        shape = x.shape
        for newlen, ax in zip(s, axes):
            shape = tuple_setitem(shape, ax, newlen)
        out = np.zeros(shape, dtype=x.dtype)
        # smaller axis is decisive how many elements are copied
        for i, (s1, s2) in enumerate(zip(x.shape, out.shape)):
            shape = tuple_setitem(shape, i, min(s1, s2))
        for index in np.ndindex(shape):
            out[index] = x[index]
        return out

    return impl


@generated_jit
def fct_cmplx(x, axes, norm, forward):
    @register_jitable
    def mul_axes(x, axes):
        n = 1.0
        for ax in axes:
            n *= x.shape[ax]
        return n

    if is_nonelike(norm):
        def impl(x, axes, norm, forward):
            return 1.0 if forward else 1.0 / mul_axes(x, axes)

    else:
        def impl(x, axes, norm, forward):
            if norm == "backward":
                return 1.0 if forward else 1.0 / mul_axes(x, axes)
            elif norm == "ortho":
                return 1.0 / np.sqrt(mul_axes(x, axes))
            elif norm == "forward":
                return 1.0 / mul_axes(x, axes) if forward else 1.0
            raise ValueError("Invalid norm value; should be"
                             " backward', 'ortho' or 'forward'.")

    return impl


@generated_jit
def get_nthreads(workers):
    if is_nonelike(workers):
        return lambda workers: 1

    def impl(workers):
        if workers > 0:
            return workers
        if workers == 0:
            raise ValueError("Workers must not be zero.")
        if workers < 0 and workers >= -_cpu_count:
            return workers + 1 + _cpu_count
        raise ValueError("Workers value out of range.")

    return impl


@generated_jit
def astype(ary, dtype):
    if hasattr(dtype, 'instance_type'):
        dtype = dtype.instance_type
    elif hasattr(dtype, '_dtype'):
        dtype = dtype._dtype

    if ary.dtype != dtype:
        def impl(ary, dtype):
            return ary.astype(dtype)

    else:
        def impl(ary, dtype):
            return ary

    return impl


def generated_alloc_output(s, istype, reqtype):
    # We don't allocate a new array if:
    # 1. overwrite was requested (runtime check)
    # 2. array got casted -> we have a new one already (compile time check)
    # 3. the array has been zero-padded/truncated (compile time check)
    if istype != reqtype or not is_nonelike(s):
        return register_jitable(lambda x, overwrite_x: x)

    @generated_jit
    def alloc_output(x, overwrite_x):
        if not hasattr(overwrite_x, 'literal_value'):
            def impl(x, overwrite_x):
                if overwrite_x:
                    return x
                out = np.empty_like(x)
                return out

        elif overwrite_x.literal_value:
            def impl(x, overwrite_x):
                return x

        else:
            def impl(x, overwrite_x):
                return np.empty_like(x)

        return impl

    return alloc_output


# TODO: take advantage of symmetry for real valued data
# TODO: copies data twice if `s` is specified and `x.dtype != argtype`
def c2cn(args, forward):
    x, s, *_ = args

    rettype = as_supported_cmplx(x.dtype)
    alloc_output = generated_alloc_output(s, x.dtype, rettype)

    def impl(x, s, axes, norm, overwrite_x, workers):
        s, axes = ndshape_and_axes(x, s, axes)
        x = astype(x, dtype=rettype)
        x = zeropad_or_crop(x, s, axes)
        out = alloc_output(x, overwrite_x)
        fct = fct_cmplx(x, axes, norm, forward)
        nthreads = get_nthreads(workers)
        numba_c2c(x, out, axes, forward, fct, nthreads)
        return out

    return impl


class HeaderOnlyError(NotImplementedError):
    ...


def _numpy_c1d(a, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError('Numpy complex 1D header cannot be called!')


def _numpy_c2d(a, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError('Numpy complex 2D header cannot be called!')


def _numpy_cnd(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError('Numpy complex ND header cannot be called!')


def _scipy_c1d(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError('Scipy complex 1D header cannot be called!')


def _scipy_c2d(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError('Scipy complex 2D header cannot be called!')


def _scipy_cnd(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError('Scipy complex ND header cannot be called!')


numpy_c1d_builder = _FFTBuilder(_numpy_c1d)
numpy_c2d_builder = _FFTBuilder(_numpy_c2d)
numpy_cnd_builder = _FFTBuilder(_numpy_cnd)
scipy_c1d_builder = _FFTBuilder(_scipy_c1d)
scipy_c2d_builder = _FFTBuilder(_scipy_c2d)
scipy_cnd_builder = _FFTBuilder(_scipy_cnd)

numpy_c1d_builder(c2cn, forward=True).register(numpy.fft.fft)
numpy_c2d_builder(c2cn, forward=True).register(numpy.fft.fft2)
numpy_cnd_builder(c2cn, forward=True).register(numpy.fft.fftn)
numpy_c1d_builder(c2cn, forward=False).register(numpy.fft.ifft)
numpy_c2d_builder(c2cn, forward=False).register(numpy.fft.ifft2)
numpy_cnd_builder(c2cn, forward=False).register(numpy.fft.ifftn)

scipy_c1d_builder(c2cn, forward=True).register(scipy.fft.fft)
scipy_c2d_builder(c2cn, forward=True).register(scipy.fft.fft2)
scipy_cnd_builder(c2cn, forward=True).register(scipy.fft.fftn)
scipy_c1d_builder(c2cn, forward=False).register(scipy.fft.ifft)
scipy_c2d_builder(c2cn, forward=False).register(scipy.fft.ifft2)
scipy_cnd_builder(c2cn, forward=False).register(scipy.fft.ifftn)


@register_jitable
def inc_or_dec_shape(shape, axes, inc):
    idx = axes[-1]
    if inc:
        newval = (shape[idx] - 1) * 2
    else:
        newval = (shape[idx] // 2) + 1
    shape = tuple_setitem(shape, idx, newval)
    return shape


# TODO: copies data twice if `s` is specified
def r2cn(args, forward):
    x, *_ = args

    if hasattr(x.dtype, "underlying_float"):
        raise TypingError(f"unsupported dtype {x.dtype}")

    argtype = as_supported_float(x.dtype)
    rettype = as_supported_cmplx(argtype)

    def impl(x, s, axes, norm, overwrite_x, workers):
        s, axes = ndshape_and_axes(x, s, axes)
        x = astype(x, dtype=argtype)
        x = zeropad_or_crop(x, s, axes)
        shape = inc_or_dec_shape(x.shape, axes, inc=False)
        out = np.empty(shape, dtype=rettype)
        fct = fct_cmplx(x, axes, norm, forward)
        nthreads = get_nthreads(workers)
        numba_r2c(x, out, axes, forward, fct, nthreads)
        return out

    return impl


numpy_c1d_builder(r2cn, forward=True).register(numpy.fft.rfft)
numpy_c2d_builder(r2cn, forward=True).register(numpy.fft.rfft2)
numpy_cnd_builder(r2cn, forward=True).register(numpy.fft.rfftn)
numpy_c1d_builder(r2cn, forward=False).register(numpy.fft.ihfft)

scipy_c1d_builder(r2cn, forward=True).register(scipy.fft.rfft)
scipy_c2d_builder(r2cn, forward=True).register(scipy.fft.rfft2)
scipy_cnd_builder(r2cn, forward=True).register(scipy.fft.rfftn)
scipy_c1d_builder(r2cn, forward=False).register(scipy.fft.ihfft)
scipy_c2d_builder(r2cn, forward=False).register(scipy.fft.ihfft2)
scipy_cnd_builder(r2cn, forward=False).register(scipy.fft.ihfftn)


@generated_jit
def resize(shape, x, s, axes):
    if is_nonelike(s):
        return lambda shape, x, s, axes: shape

    def impl(shape, x, s, axes):
        last_ax = x.ndim - 1
        for i, ax in enumerate(axes):
            if ax == last_ax:
                shape = tuple_setitem(shape, last_ax, s[i])
        return shape

    return impl


# TODO: copies data twice if `s` is specified
def c2rn(args, forward):
    x, *_ = args

    argtype = as_supported_cmplx(x.dtype)
    rettype = as_supported_float(argtype)

    def impl(x, s, axes, norm, overwrite_x, workers):
        s, axes = ndshape_and_axes(x, s, axes)
        x = astype(x, dtype=argtype)
        xin = zeropad_or_crop(x, s, axes)
        shape = inc_or_dec_shape(x.shape, axes, inc=True)
        shape = resize(shape, x, s, axes)
        out = np.empty(shape, dtype=rettype)
        fct = fct_cmplx(out, axes, norm, forward)
        nthreads = get_nthreads(workers)
        numba_c2r(xin, out, axes, forward, fct, nthreads)
        return out

    return impl


numpy_c1d_builder(c2rn, forward=False).register(numpy.fft.irfft)
numpy_c2d_builder(c2rn, forward=False).register(numpy.fft.irfft2)
numpy_cnd_builder(c2rn, forward=False).register(numpy.fft.irfftn)
numpy_c1d_builder(c2rn, forward=True).register(numpy.fft.hfft)

scipy_c1d_builder(c2rn, forward=False).register(scipy.fft.irfft)
scipy_c2d_builder(c2rn, forward=False).register(scipy.fft.irfft2)
scipy_cnd_builder(c2rn, forward=False).register(scipy.fft.irfftn)
scipy_c1d_builder(c2rn, forward=True).register(scipy.fft.hfft)
scipy_c2d_builder(c2rn, forward=True).register(scipy.fft.hfft2)
scipy_cnd_builder(c2rn, forward=True).register(scipy.fft.hfftn)


@register_jitable
def get_type(type, forward):
    if forward:
        return type
    if type == 2:
        return 3
    if type == 3:
        return 2
    return type


@generated_jit
def get_ortho(norm, ortho):
    if not is_nonelike(ortho):
        return lambda norm, ortho: ortho

    def impl(norm, ortho):
        if norm == "ortho":
            return True
        return False

    return impl


@generated_jit
def fct_real(x, axes, norm, delta, forward):
    @register_jitable
    def mul_axes(x, axes, delta):
        n = 1.0
        for ax in axes:
            n *= 2.0 * (x.shape[ax] + delta)
        return n

    if is_nonelike(norm):
        def impl(x, axes, norm, delta, forward):
            return 1.0 if forward else 1.0 / mul_axes(x, axes, delta)

    else:
        def impl(x, axes, norm, delta, forward):
            if norm == "backward":
                return 1.0 if forward else 1.0 / mul_axes(x, axes, delta)
            elif norm == "ortho":
                return 1.0 / np.sqrt(mul_axes(x, axes, delta))
            elif norm == "forward":
                return 1.0 / mul_axes(x, axes, delta) if forward else 1.0
            raise ValueError("Invalid norm value; should be"
                             " 'backward', 'ortho' or 'forward'.")

    return impl


# TODO: copies data twice if `s` is specified and `x.dtype != argtype`
def r2rn(args, trafo, delta, forward):
    x, _, s, *_ = args

    if hasattr(x.dtype, "underlying_float"):
        argtype = as_supported_cmplx(x.dtype)

        # Transform real and imaginary part seperately if input is complex.
        @register_jitable
        def do_transform(x, out, axes, type, fct, ortho, nthreads):
            trafo(x.real, out.real, axes, type, fct, ortho, nthreads)
            trafo(x.imag, out.imag, axes, type, fct, ortho, nthreads)

    else:
        argtype = as_supported_float(x.dtype)
        do_transform = trafo

    rettype = argtype
    alloc_output = generated_alloc_output(s, x.dtype, rettype)

    def impl(x, type, s, axes, norm, overwrite_x, workers, orthogonalize):
        s, axes = ndshape_and_axes(x, s, axes)
        x = astype(x, dtype=rettype)
        x = zeropad_or_crop(x, s, axes)
        out = alloc_output(x, overwrite_x)
        type = get_type(type, forward)
        delta_ = delta if type == 1 else 0.0
        fct = fct_real(out, axes, norm, delta_, forward)
        ortho = get_ortho(norm, orthogonalize)
        nthreads = get_nthreads(workers)
        do_transform(x, out, axes, type, fct, ortho, nthreads)
        return out

    return impl


def _scipy_r1d(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
               workers=None, orthogonalize=None):
    raise HeaderOnlyError('Scipy real 1D header cannot be called!')


def _scipy_rnd(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
               workers=None, orthogonalize=None):
    raise HeaderOnlyError('Scipy real ND header cannot be called!')


scipy_r1d_builder = _FFTBuilder(_scipy_r1d)
scipy_rnd_builder = _FFTBuilder(_scipy_rnd)


_common_dct = dict(trafo=numba_dct, delta=-1)
scipy_r1d_builder(r2rn, **_common_dct, forward=True).register(scipy.fft.dct)
scipy_rnd_builder(r2rn, **_common_dct, forward=True).register(scipy.fft.dctn)
scipy_r1d_builder(r2rn, **_common_dct, forward=False).register(scipy.fft.idct)
scipy_rnd_builder(r2rn, **_common_dct, forward=False).register(scipy.fft.idctn)

_common_dst = dict(trafo=numba_dst, delta=1)
scipy_r1d_builder(r2rn, **_common_dst, forward=True).register(scipy.fft.dst)
scipy_rnd_builder(r2rn, **_common_dst, forward=True).register(scipy.fft.dstn)
scipy_r1d_builder(r2rn, **_common_dst, forward=False).register(scipy.fft.idst)
scipy_rnd_builder(r2rn, **_common_dst, forward=False).register(scipy.fft.idstn)


@overload(np.roll)
def roll(a, shift, axis=None):
    # TODO: The multidimensional case is extremly inefficient!
    # I only implemented a naiv approach.
    if not isinstance(a, types.Array):
        raise TypeError("The 1st argument 'a' must be an array.")

    if not (is_sequence_like(shift)
            and isinstance(shift.dtype, types.Integer)):
        if not isinstance(shift, types.Integer):
            raise TypeError("The 2nd argument 'shift' must be a"
                            " sequences of integers or an integer.")

    if not is_nonelike(axis):
        if not (is_sequence_like(axis)
                and isinstance(axis.dtype, types.Integer)):
            if not isinstance(axis, types.Integer):
                raise TypeError("The 3rd argument 'axis' must be a"
                                " sequences of integers or an integer.")

    if not is_nonelike(axis):
        if is_sequence_like(axis) != is_sequence_like(shift):
            raise TypingError("If axis is specified, shift and"
                              " axis must both be integers or"
                              " integer sequences of equal length.")

    if is_nonelike(axis):
        def impl(a, shift, axis=None):
            sh = np.asarray(shift).sum()
            r = np.empty_like(a.ravel())
            r[sh:] = a[:-sh]
            r[:sh] = a[-sh:]
            return r.reshape(a.shape)

    else:
        def impl(a, shift, axis=None):
            axis, shift = np.broadcast_arrays(axis, shift)
            # `axis` is readonly but we eventually need to write it.
            axis = axis.copy()
            wraparound_axes(a, axis)

            a_index = a.shape
            for i in range(a.ndim):
                a_index = tuple_setitem(a_index, i, 0)

            r_index = a_index
            for ax, sh in zip(axis, shift):
                r_index = tuple_setitem(r_index, ax, r_index[ax] + sh)
            for i in range(a.ndim):
                if r_index[i] > 0:
                    r_index = tuple_setitem(r_index, i, r_index[i] - a.shape[i])
                if r_index[i] != 0:
                    if a.shape[i] == 0:
                        r_index = tuple_setitem(r_index, i, 0)
                    else:
                        val = np.abs(r_index[i]) % a.shape[i]
                        r_index = tuple_setitem(r_index, i, -val)
            r_index_init = r_index

            # TODO: This part is not efficient.
            r = np.empty_like(a)

            # This is like `np.ndindex` except that we maintain two index
            # tuples in parallel; a normal one and a shifted one.
            done = r.size == 0
            while not done:
                r[r_index] = a[a_index]

                done = True
                for i in range(a.ndim):
                    r_index = tuple_setitem(r_index, i, r_index[i] + 1)
                    a_index = tuple_setitem(a_index, i, a_index[i] + 1)

                    if a_index[i] < a.shape[i]:
                        done = False
                        break

                    r_index = tuple_setitem(r_index, i, r_index_init[i])
                    a_index = tuple_setitem(a_index, i, 0)

            return r

    return impl


def _check_typing_fftshift(x, axes):
    if not isinstance(x, types.Array):
        raise TypeError("The 1st argument 'x' must be an array.")

    if not is_nonelike(axes):
        if not (is_sequence_like(axes)
                and isinstance(axes.dtype, types.Integer)):
            if not isinstance(axes, types.Integer):
                raise TypeError("The 2nd argument 'axes' must be sequence of"
                                " integers, an integer or None.")


@overload(np.fft.fftshift)
def fftshift(x, axes=None):
    _check_typing_fftshift(x, axes)

    if is_nonelike(axes):
        def impl(x, axes=None):
            axes = x.shape
            shift = x.shape
            for i, dim in enumerate(x.shape):
                shift = tuple_setitem(shift, i, dim // 2)
                axes = tuple_setitem(axes, i, i)
            return np.roll(x, shift, axes)

    elif isinstance(axes, types.Integer):
        def impl(x, axes=None):
            shift = x.shape[axes] // 2
            return np.roll(x, shift, axes)

    else:
        def impl(x, axes=None):
            shift = x.shape[: len(axes)]
            for i, ax in enumerate(axes):
                shift = tuple_setitem(shift, i, x.shape[ax] // 2)
            return np.roll(x, shift, axes)

    return impl


@overload(np.fft.ifftshift)
def ifftshift(x, axes=None):
    _check_typing_fftshift(x, axes)

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
            shift = x.shape[: len(axes)]
            for i, ax in enumerate(axes):
                shift = tuple_setitem(shift, i, -(x.shape[ax] // 2))
            return np.roll(x, shift, axes)

    return impl


def _check_typing_fftfreq(n, d):
    if not isinstance(n, types.Integer):
        raise TypeError("The 1st argument 'n' must be an integer.")

    if not isinstance(d, types.Number):
        raise TypeError("The 2nd argument 'd' must be a scaler.")


@overload(np.fft.fftfreq)
def fftfreq(n, d=1.0):
    _check_typing_fftfreq(n, d)

    def impl(n, d=1.0):
        val = 1.0 / (n * d)
        results = np.empty(n, dtype=np.int64)
        N = (n - 1) // 2 + 1
        p1 = np.arange(N)
        results[:N] = p1
        p2 = np.arange(-(n // 2), 0)
        results[N:] = p2
        return results * val

    return impl


@overload(np.fft.rfftfreq)
def rfftfreq(n, d=1.0):
    _check_typing_fftfreq(n, d)

    def impl(n, d=1.0):
        val = 1.0 / (n * d)
        N = n // 2 + 1
        results = np.arange(N)
        return results * val

    return impl


@overload(scipy.fft.next_fast_len)
def next_fast_len(target, real):
    if not isinstance(target, types.Integer):
        raise TypeError("The 1st argument 'target' must be an integer.")

    if not isinstance(real, types.Boolean):
        raise TypeError("The 2nd argument 'real' must be a boolean.")

    def impl(target, real):
        if target < 0:
            raise ValueError("Target cannot be negative.")
        return numba_good_size(target, real)

    return impl
