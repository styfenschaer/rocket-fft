import inspect
from functools import partial, wraps

import numpy as np
import numpy.fft
import scipy.fft
from numba import TypingError, generated_jit
from numba.core import types
from numba.core.config import NUMBA_NUM_THREADS as _cpu_count
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import overload, register_jitable
from numba.np.numpy_support import is_nonelike

from . import ipocketfft as pfft
from . import numba_typing as tg
from .imputils import implements_jit, implements_overload, otherwise
from .numba_typing import (is_integer, is_integer_2tuple, is_nonelike,
                           is_not_nonelike, is_sequence_like, literal_is_false,
                           literal_is_true)

# TODO:
# can optimize for literal values?
# optimize default values for 1D/2D transforms


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


fft_typing = tg.TypingChecker().register(
    a=tg.Check(
        types.Array, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'a' must be an array."),
    x=tg.Check(
        types.Array, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'x' must be an array."),
    n=tg.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 'n' must be an integer."),
    s=tg.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 's' must be a sequence of integers."),
    axis=tg.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 'axis' must be an integer."),
    axes=tg.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument 'axes' must be a sequence of integers."),
    norm=tg.Check(
        types.UnicodeType, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument 'norm' must be a string."),
    type=tg.Check(
        types.Integer, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'type' must be an integer."),
    overwrite_x=tg.Check(
        types.Boolean, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument 'overwrite_x' must be a boolean."),
    workers=tg.Check(
        types.Integer, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument 'workers' must be an integer."),
    orthogonalize=tg.Check(
        types.Boolean, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument 'orthogonalize' must be a boolean."),
)


class FFTBuilder:
    def __init__(self, header, typing=None):
        self.header = header
        self.typing = typing
        self.built = None
        self.register = []

    def __call__(self, func, *args, **kwargs):
        @wraps(self.header)
        def ol_func(*iargs, **ikwargs):
            kwd = self._get_callargs(*iargs, **ikwargs)
            if self.typing is not None:
                self.typing.reset()
                self.typing(**kwd)
            params = tuple(kwd.values())
            impl = func(params, *args, **kwargs)
            self._patch_co(impl)
            return wraps(self.header)(impl)

        self.built = ol_func
        return self

    def overload(self, func):
        entry = (func, self.built)
        self.register.append(entry)
        overload(func)(self.built)
        return self

    @property
    def signature(self):
        return inspect.signature(self.header)

    def _get_callargs(self, *args, **kwargs):
        kwd = inspect.getcallargs(self.header, *args, **kwargs)
        params = self.signature.parameters.keys()
        return {key: kwd[key] for key in params}

    def _patch_co(self, func):
        params_header = self.signature.parameters
        params_func = inspect.signature(func).parameters
        cov = list(func.__code__.co_varnames)
        for ph, pf in zip(params_header.keys(), params_func.keys()):
            if ph != pf:
                idx = cov.index(pf)
                cov[idx] = ph
        cov = tuple(cov)
        func.__code__ = func.__code__.replace(co_varnames=cov)


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


@register_jitable
def asarray(arg):
    a = np.asarray(arg)
    return np.atleast_1d(a)


@implements_jit
def ndshape_and_axes(x, s, axes):
    pass


@ndshape_and_axes.impl(s=is_nonelike, axes=is_integer)
def _(x, s, axes):
    # compile time reduction for default 1D transform
    if axes < 0:
        axes += x.ndim
    elif axes >= x.ndim:
        raise ValueError("Axes exceeds dimensionality of input.")
    axes = np.array([axes])
    return s, axes


@ndshape_and_axes.impl(s=is_nonelike, axes=is_integer_2tuple)
def _(x, s, axes):
    # compile time reduction for default 2D transform
    ax1, ax2 = axes
    if ax1 < 0:
        ax1 += x.ndim
    if ax2 < 0:
        ax2 += x.ndim
    elif ax1 >= x.ndim or ax2 >= x.ndim:
        raise ValueError("Axes exceeds dimensionality of input.")
    axes = np.array([ax1, ax2])
    return s, axes


@ndshape_and_axes.impl(s=is_nonelike, axes=is_nonelike)
def _(x, s, axes):
    # axes not specified, transform all axes
    axes = np.arange(x.ndim)
    return s, axes


@ndshape_and_axes.impl(s=is_nonelike, axes=is_not_nonelike)
def _(x, s, axes):
    axes = asarray(axes)
    assert_unique_axes(axes)
    wraparound_axes(x, axes)
    return s, axes


@ndshape_and_axes.impl(s=is_not_nonelike, axes=is_nonelike)
def _(x, s, axes):
    s = asarray(s)
    if s.min() < 1:
        raise ValueError("Invalid number of data points specified.")
    if s.size > x.ndim:
        raise ValueError("Shape requires more axes than are present.")
    # axes not specified, transform last len(s) axes
    axes = np.arange(x.ndim - s.size, x.ndim)
    return s, axes


@ndshape_and_axes.impl(s=is_not_nonelike, axes=is_not_nonelike)
def _(x, s, axes):
    s = asarray(s)
    if s.min() < 1:
        raise ValueError("Invalid number of data points specified.")
    axes = asarray(axes)
    assert_unique_axes(axes)
    wraparound_axes(x, axes)
    if s.size != axes.size:
        raise ValueError("When given, axes and shape arguments"
                         " have to be of the same length.")
    return s, axes


ndshape_and_axes = ndshape_and_axes.generate()


@implements_jit
def zeropad_or_crop(x, s, axes):
    pass


@zeropad_or_crop.impl(s=is_nonelike)
def _(x, s, axes):
    return x


@zeropad_or_crop.impl(otherwise)
def _(x, s, axes):
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


zeropad_or_crop = zeropad_or_crop.generate()


@implements_jit
def mul_axes(x, axes, delta=None):
    pass


@mul_axes.impl(delta=is_nonelike)
def _(x, axes, delta=None):
    n = 1.0
    for ax in axes:
        n *= x.shape[ax]
    return n


@mul_axes.impl(otherwise)
def _(x, axes, delta=None):
    n = 1.0
    for ax in axes:
        n *= 2.0 * (x.shape[ax] + delta)
    return n


mul_axes = mul_axes.generate()


@implements_jit
def get_fct(x, axes, norm, forward, delta=None):
    pass


@get_fct.impl(norm=is_nonelike, forward=literal_is_true)
def _(x, axes, norm, forward, delta=None):
    return 1.0


@get_fct.impl(norm=is_nonelike, forward=literal_is_false)
def _(x, axes, norm, forward, delta=None):
    return 1.0 / mul_axes(x, axes, delta)


@get_fct.impl(norm=is_not_nonelike, forward=literal_is_true)
def _(x, axes, norm, forward, delta=None):
    if norm == "backward":
        return 1.0
    elif norm == "ortho":
        return 1.0 / np.sqrt(mul_axes(x, axes, delta))
    elif norm == "forward":
        return 1.0 / mul_axes(x, axes, delta)
    raise ValueError("Invalid norm value; should be"
                     " 'backward', 'ortho' or 'forward'.")


@get_fct.impl(otherwise)
def _(x, axes, norm, forward, delta=None):
    if norm == "backward":
        return 1.0 / mul_axes(x, axes, delta)
    elif norm == "ortho":
        return 1.0 / np.sqrt(mul_axes(x, axes, delta))
    elif norm == "forward":
        return 1.0
    raise ValueError("Invalid norm value; should be"
                     " 'backward', 'ortho' or 'forward'.")


get_fct = get_fct.generate()


@implements_jit
def get_nthreads(workers):
    pass


@get_nthreads.impl(workers=is_nonelike)
def _(workers):
    return 1


@get_nthreads.impl(otherwise)
def _(workers):
    if workers > 0:
        return workers
    if workers == 0:
        raise ValueError("Workers must not be zero.")
    if workers < 0 and workers >= -_cpu_count:
        return workers + 1 + _cpu_count
    raise ValueError("Workers value out of range.")


get_nthreads = get_nthreads.generate()


@implements_jit
def astype(ary, dtype):
    pass


@astype.preproc
def _(ary, dtype):
    if hasattr(dtype, 'instance_type'):
        dtype = dtype.instance_type
    elif hasattr(dtype, '_dtype'):
        dtype = dtype._dtype
    return ary, dtype


@astype.impl(lambda ary, dtype: ary.dtype != dtype)
def _(ary, dtype):
    return ary.astype(dtype)


@astype.impl(otherwise)
def _(ary, dtype):
    return ary


astype = astype.generate()


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
        fct = get_fct(x, axes, norm, forward)
        nthreads = get_nthreads(workers)
        pfft.numba_c2c(x, out, axes, forward, fct, nthreads)
        return out

    return impl


class HeaderOnlyError(NotImplementedError):
    """Guards functions used as header to the FFTBuilder class."""


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


numpy_c1d_builder = FFTBuilder(_numpy_c1d, typing=fft_typing)
numpy_c2d_builder = FFTBuilder(_numpy_c2d, typing=fft_typing)
numpy_cnd_builder = FFTBuilder(_numpy_cnd, typing=fft_typing)
scipy_c1d_builder = FFTBuilder(_scipy_c1d, typing=fft_typing)
scipy_c2d_builder = FFTBuilder(_scipy_c2d, typing=fft_typing)
scipy_cnd_builder = FFTBuilder(_scipy_cnd, typing=fft_typing)

numpy_c1d_builder(c2cn, forward=True).overload(numpy.fft.fft)
numpy_c2d_builder(c2cn, forward=True).overload(numpy.fft.fft2)
numpy_cnd_builder(c2cn, forward=True).overload(numpy.fft.fftn)
numpy_c1d_builder(c2cn, forward=False).overload(numpy.fft.ifft)
numpy_c2d_builder(c2cn, forward=False).overload(numpy.fft.ifft2)
numpy_cnd_builder(c2cn, forward=False).overload(numpy.fft.ifftn)

scipy_c1d_builder(c2cn, forward=True).overload(scipy.fft.fft)
scipy_c2d_builder(c2cn, forward=True).overload(scipy.fft.fft2)
scipy_cnd_builder(c2cn, forward=True).overload(scipy.fft.fftn)
scipy_c1d_builder(c2cn, forward=False).overload(scipy.fft.ifft)
scipy_c2d_builder(c2cn, forward=False).overload(scipy.fft.ifft2)
scipy_cnd_builder(c2cn, forward=False).overload(scipy.fft.ifftn)


@register_jitable
def decrease_shape(shape, axes):
    idx = axes[-1]
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
        shape = decrease_shape(x.shape, axes)
        out = np.empty(shape, dtype=rettype)
        fct = get_fct(x, axes, norm, forward)
        nthreads = get_nthreads(workers)
        pfft.numba_r2c(x, out, axes, forward, fct, nthreads)
        return out

    return impl


numpy_c1d_builder(r2cn, forward=True).overload(numpy.fft.rfft)
numpy_c2d_builder(r2cn, forward=True).overload(numpy.fft.rfft2)
numpy_cnd_builder(r2cn, forward=True).overload(numpy.fft.rfftn)
numpy_c1d_builder(r2cn, forward=False).overload(numpy.fft.ihfft)

scipy_c1d_builder(r2cn, forward=True).overload(scipy.fft.rfft)
scipy_c2d_builder(r2cn, forward=True).overload(scipy.fft.rfft2)
scipy_cnd_builder(r2cn, forward=True).overload(scipy.fft.rfftn)
scipy_c1d_builder(r2cn, forward=False).overload(scipy.fft.ihfft)
scipy_c2d_builder(r2cn, forward=False).overload(scipy.fft.ihfft2)
scipy_cnd_builder(r2cn, forward=False).overload(scipy.fft.ihfftn)


@register_jitable
def increase_shape(shape, axes):
    idx = axes[-1]
    newval = (shape[idx] - 1) * 2
    shape = tuple_setitem(shape, idx, newval)
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
    return shape


resize = resize.generate()


# TODO: copies data twice if `s` is specified
def c2rn(args, forward):
    x, *_ = args

    argtype = as_supported_cmplx(x.dtype)
    rettype = as_supported_float(argtype)

    def impl(x, s, axes, norm, overwrite_x, workers):
        s, axes = ndshape_and_axes(x, s, axes)
        x = astype(x, dtype=argtype)
        xin = zeropad_or_crop(x, s, axes)
        shape = increase_shape(x.shape, axes)
        shape = resize(shape, x, s, axes)
        out = np.empty(shape, dtype=rettype)
        fct = get_fct(out, axes, norm, forward)
        nthreads = get_nthreads(workers)
        pfft.numba_c2r(xin, out, axes, forward, fct, nthreads)
        return out

    return impl


numpy_c1d_builder(c2rn, forward=False).overload(numpy.fft.irfft)
numpy_c2d_builder(c2rn, forward=False).overload(numpy.fft.irfft2)
numpy_cnd_builder(c2rn, forward=False).overload(numpy.fft.irfftn)
numpy_c1d_builder(c2rn, forward=True).overload(numpy.fft.hfft)

scipy_c1d_builder(c2rn, forward=False).overload(scipy.fft.irfft)
scipy_c2d_builder(c2rn, forward=False).overload(scipy.fft.irfft2)
scipy_cnd_builder(c2rn, forward=False).overload(scipy.fft.irfftn)
scipy_c1d_builder(c2rn, forward=True).overload(scipy.fft.hfft)
scipy_c2d_builder(c2rn, forward=True).overload(scipy.fft.hfft2)
scipy_cnd_builder(c2rn, forward=True).overload(scipy.fft.hfftn)


@implements_jit
def get_type(type, forward):
    pass


@get_type.impl(forward=literal_is_true)
def _(type, forward):
    return type


@get_type.impl(otherwise)
def _(type, forward):
    if type == 2:
        return 3
    if type == 3:
        return 2
    return type


get_type = get_type.generate()


@implements_jit
def get_ortho(norm, ortho):
    pass


@get_ortho.impl(ortho=is_not_nonelike)
def _(norm, ortho):
    return ortho


@get_ortho.impl(otherwise)
def _(norm, ortho):
    if norm == "ortho":
        return True
    return False


get_ortho = get_ortho.generate()


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
        fct = get_fct(out, axes, norm, forward, delta_)
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


scipy_r1d_builder = FFTBuilder(_scipy_r1d, typing=fft_typing)
scipy_rnd_builder = FFTBuilder(_scipy_rnd, typing=fft_typing)


_common_dct = dict(trafo=pfft.numba_dct, delta=-1)
scipy_r1d_builder(r2rn, **_common_dct, forward=True).overload(scipy.fft.dct)
scipy_rnd_builder(r2rn, **_common_dct, forward=True).overload(scipy.fft.dctn)
scipy_r1d_builder(r2rn, **_common_dct, forward=False).overload(scipy.fft.idct)
scipy_rnd_builder(r2rn, **_common_dct, forward=False).overload(scipy.fft.idctn)

_common_dst = dict(trafo=pfft.numba_dst, delta=1)
scipy_r1d_builder(r2rn, **_common_dst, forward=True).overload(scipy.fft.dst)
scipy_rnd_builder(r2rn, **_common_dst, forward=True).overload(scipy.fft.dstn)
scipy_r1d_builder(r2rn, **_common_dst, forward=False).overload(scipy.fft.idst)
scipy_rnd_builder(r2rn, **_common_dst, forward=False).overload(scipy.fft.idstn)


@implements_overload(np.roll)
def roll(a, shift, axis=None):
    # TODO: The multidimensional case is extremly inefficient!
    # I only implemented a naiv approach.
    msg = "The 1st argument 'a' must be an array."
    tg.Check(types.Array, msg=msg)
    msg = "The 2nd argument 'shift' must be a sequences of integers or an integer."
    tg.Check(types.Integer, as_seq=True, msg=msg)
    msg = "The 3rd argument 'axis' must be a sequences of integers or an integer."
    tg.Check(types.Integer, as_seq=True, allow_none=True, msg=msg)

    if is_sequence_like(axis) != is_sequence_like(shift):
        raise TypingError("If axis is specified, shift and axis must both "
                          "be integers or  integer sequences of equal length.")


@roll.impl(axis=is_nonelike)
def _(a, shift, axis=None):
    sh = np.asarray(shift).sum()
    r = np.empty_like(a.ravel())
    r[sh:] = a[:-sh]
    r[:sh] = a[-sh:]
    return r.reshape(a.shape)


@roll.impl(otherwise)
def _(a, shift, axis=None):
    axis, shift = np.broadcast_arrays(axis, shift)
    # axis is readonly but we eventually need to write it.
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

    # This is like np.ndindex except that we maintain two index
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


roll = roll.generate()


def _typing_fftshift(x, axes):
    msg = "The 1st argument 'x' must be an array."
    tg.Check(types.Array, msg=msg)(x)
    msg = "The 2nd argument 'axes' must be a sequences of integers or an integer."
    tg.Check(types.Integer, as_seq=True,
             allow_none=True, msg=msg)(axes)


@implements_overload(np.fft.fftshift)
def fftshift(x, axes=None):
    _typing_fftshift(x, axes)


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


fftshift = fftshift.generate()


@implements_overload(np.fft.ifftshift)
def ifftshift(x, axes=None):
    _typing_fftshift(x, axes)


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


ifftshift = ifftshift.generate()


def _typing_fftfreq(n, d):
    msg = "The 1st argument 'n' must be an integer."
    tg.Check(types.Integer, msg=msg)(n)
    msg = "The 2nd argument 'd' must be an scaler."
    tg.Check(types.Number, msg=msg)(d)


@overload(np.fft.fftfreq)
def fftfreq(n, d=1.0):
    _typing_fftfreq(n, d)

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
    _typing_fftfreq(n, d)

    def impl(n, d=1.0):
        val = 1.0 / (n * d)
        N = n // 2 + 1
        results = np.arange(N)
        return results * val

    return impl


@overload(scipy.fft.next_fast_len)
def next_fast_len(target, real):
    msg = "The 1st argument 'target' must be an integer."
    tg.Check(types.Integer, msg=msg)(target)
    msg = "The 2nd argument 'real' must be a boolean."
    tg.Check(types.Boolean, msg=msg)(real)

    def impl(target, real):
        if target < 0:
            raise ValueError("Target cannot be negative.")
        return pfft.numba_good_size(target, real)

    return impl
