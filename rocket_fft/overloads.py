import inspect
from functools import partial, wraps
from os import cpu_count
from types import MappingProxyType

import numpy as np
import numpy.fft
from numba import TypingError
from numba.core import types
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import overload, register_jitable
from numba.np.numpy_support import is_nonelike

from . import pocketfft
from . import typutils as tu
from .imputils import implements_jit, implements_overload, otherwise
from .typutils import (is_contiguous_array, is_integer, is_integer_2tuple,
                       is_literal_bool, is_literal_integer, is_nonelike,
                       is_not_nonelike, is_scalar, typing_check)

# Unlike NumPy, SciPy is an optional runtime dependency
try:
    import scipy.fft

    from . import special
    _scipy_installed_ = True
except ImportError:
    _scipy_installed_ = False

# SciPy and NumPy differ in when they convert types and handle duplicate axes.
# These functions mimic their behavior. They must be called before the compilation
# of our internals, otherwise they have no effect. After compilation, the changes 
# are ireversible. 

# Public API
def numpy_like():
    _set_luts(_numpy_cmplx_lut, _numpy_real_lut)
    _set_assert_unique_axes(_numpy_assert_unique_axes)
 
 
 # Public API
def scipy_like():
    _set_luts(_scipy_cmplx_lut, _scipy_real_lut)
    _set_assert_unique_axes(_scipy_assert_unique_axes)   
    
    
# Lookup tables for type conversion
_scipy_cmplx_lut = MappingProxyType({
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
})
_scipy_real_lut = MappingProxyType({
    key: val.underlying_float for key, val in _scipy_cmplx_lut.items()
})

_numpy_cmplx_lut = MappingProxyType({
    key: types.complex128 for key in _scipy_cmplx_lut.keys()
})
_numpy_real_lut = MappingProxyType({
    key: types.float64 for key in _scipy_cmplx_lut.keys()
})


_as_cmplx_lut = None
_as_real_lut = None


def _set_luts(cmplx_lut, real_lut):
    global _as_cmplx_lut, _as_real_lut
    _as_cmplx_lut = cmplx_lut.copy()
    _as_real_lut = real_lut.copy()


def _as_supported_dtype(dtype, real):
    lut = _as_real_lut if real else _as_cmplx_lut
    if lut is None:
        raise RuntimeError("Type conversion lookup table not instantiated.")

    ty = lut.get(dtype)
    if ty is not None:
        return ty

    keys = tuple(lut.keys())
    raise TypingError(f"Unsupported dtype {dtype}; supported are {keys}.")


as_supported_cmplx = partial(_as_supported_dtype, real=False)
as_supported_real = partial(_as_supported_dtype, real=True)


fft_typing = tu.TypingChecker(
    a=tu.Check(
        types.Array, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument '{}' must be an array."),
    x=tu.Check(
        types.Array, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument '{}' must be an array."),
    n=tu.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument '{}' must be an integer."),
    s=tu.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument '{}' must be a sequence of integers."),
    axis=tu.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument '{}' must be an integer."),
    axes=tu.Check(
        types.Integer, as_one=True, as_seq=True, allow_none=True,
        msg="The {} argument '{}' must be a sequence of integers."),
    norm=tu.Check(
        types.UnicodeType, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument '{}' must be a string."),
    type=tu.Check(
        types.Integer, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument '{}' must be an integer."),
    overwrite_x=tu.Check(
        types.Boolean, as_one=True, as_seq=False, allow_none=False,
        msg="The {} argument '{}' must be a boolean."),
    workers=tu.Check(
        types.Integer, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument '{}' must be an integer."),
    orthogonalize=tu.Check(
        types.Boolean, as_one=True, as_seq=False, allow_none=True,
        msg="The {} argument '{}' must be a boolean."),
)


class FFTBuilder:
    __slots__ = ("header", "typing_checker", "built")
    
    register = {}

    def __init__(self, header, typing_checker=None):
        self.header = header
        self.typing_checker = typing_checker
        self.built = None

    def __call__(self, func, *args, **kwargs):
        @wraps(self.header)
        def ol_func(*iargs, **ikwargs):
            kwd = self._get_callargs(*iargs, **ikwargs)
            if self.typing_checker is not None:
                self.typing_checker(**kwd)
            params = tuple(kwd.values())
            impl = func(params, *args, **kwargs)
            self._patch_co_varnames(impl)
            return wraps(self.header)(impl)

        self.built = ol_func
        return self

    def overload(self, func):
        entry = (self, self.built)
        FFTBuilder.register[func] = entry
        overload(func)(self.built)
        return self

    @property
    def signature(self):
        return inspect.signature(self.header)

    def _get_callargs(self, *args, **kwargs):
        kwd = inspect.getcallargs(self.header, *args, **kwargs)
        params = self.signature.parameters.keys()
        return {key: kwd[key] for key in params}

    def _patch_co_varnames(self, func):
        header_params = self.signature.parameters.keys()
        func_params = inspect.signature(func).parameters.keys()
        name_map = {old: new for old, new in zip(func_params, header_params)}
        co_varnames = func.__code__.co_varnames
        cov = tuple(name_map.get(name, name) for name in co_varnames)
        func.__code__ = func.__code__.replace(co_varnames=cov)


@register_jitable
def wraparound_axis(x, ax):
    if (ax >= x.ndim) or (ax < -x.ndim):
        raise ValueError("Axis exceeds dimensionality of input.")
    if ax < 0:
        ax += x.ndim
    return ax


@register_jitable
def wraparound_axes(x, axes):
    for i, ax in enumerate(axes):
        if (ax >= x.ndim) or (ax < -x.ndim):
            raise ValueError("Axes exceeds dimensionality of input.")
        if ax < 0:
            axes[i] += x.ndim

# NumPy allows passing duplicate axes for fft2, fftn, ifft2 and ifft
# while SciPy doesn't

@register_jitable(locals={"slots": types.UniTuple(types.byte, 32)})
def _scipy_assert_unique_axes(axes):
    slots = (0,) * 32  # np.MAXDIMS
    for ax in axes:
        if slots[ax] != 0:
            raise ValueError("All axes must be unique.")
        slots = tuple_setitem(slots, ax, 1)


@register_jitable
def _numpy_assert_unique_axes(axes):
    pass


assert_unique_axes = None


def _set_assert_unique_axes(impl):
    global assert_unique_axes
    assert_unique_axes = impl
    

@register_jitable
def assert_valid_shape(shape):
    for n in shape:
        if n < 1:
            raise ValueError("Invalid number of data points specified.")


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
    assert_unique_axes((ax1, ax2))
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
    assert_unique_axes(axes)
    return s, axes


@ndshape_and_axes.impl(s=is_not_nonelike, axes=is_nonelike)
def _(x, s, axes):
    s = toarray(s)
    assert_valid_shape(s)
    if s.size > x.ndim:
        raise ValueError("Shape requires more axes than are present.")
    # Axes not specified, transform last len(s) axes
    axes = np.arange(x.ndim - s.size, x.ndim)
    return s, axes


@ndshape_and_axes.impl(otherwise)
def _(x, s, axes):
    s = toarray(s)
    assert_valid_shape(s)
    axes = toarray(axes)
    wraparound_axes(x, axes)
    assert_unique_axes(axes)
    if s.size != axes.size:
        raise ValueError("When given, axes and shape arguments"
                         " have to be of the same length.")
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
    raise ValueError("Invalid norm value; should be"
                     " 'backward', 'ortho' or 'forward'.")


@get_fct.impl(norm=is_not_nonelike, forward=is_literal_bool(False))
def _(x, axes, norm, forward, delta=None):
    if norm == "backward":
        return 1.0 / mul_axes(x.shape, axes, delta)
    elif norm == "ortho":
        return 1.0 / np.sqrt(mul_axes(x.shape, axes, delta))
    elif norm == "forward":
        return 1.0
    raise ValueError("Invalid norm value; should be"
                     " 'backward', 'ortho' or 'forward'.")


_cpu_count = cpu_count()
_default_workers = 1


# Public API
def get_workers():
    return _default_workers


# Public API
def set_workers(workers):
    if workers < 1:
        raise ValueError("Number of workers cannot be smaller than one.")
    if workers > _cpu_count:
        raise ValueError("Number of workers exceeds "
                         f"CPU count of {_cpu_count}.")
    
    global _default_workers
    _default_workers = workers


@implements_jit(prefer_literal=True)
def get_nthreads(workers):
    pass


@get_nthreads.impl(workers=is_nonelike)
def _(workers):
    return _default_workers


@get_nthreads.impl(otherwise)
def _(workers):
    if workers > 0:
        return workers
    if workers == 0:
        raise ValueError("Workers must not be zero.")
    if workers < -_cpu_count:
        raise ValueError("Workers value out of range.")
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


def c2cn(args, forward):
    x, s, *_ = args

    rettype = as_supported_cmplx(x.dtype)

    if isinstance(x.dtype, types.Complex):
        alloc_output = generated_alloc_output(s, x.dtype, rettype)

        def impl(x, s, axes, norm, overwrite_x, workers):
            s, axes = ndshape_and_axes(x, s, axes)
            x = zeropad_or_crop(x, s, axes, rettype)
            out = alloc_output(x, overwrite_x)
            fct = get_fct(x, axes, norm, forward)
            nthreads = get_nthreads(workers)
            pocketfft.numba_c2c(x, out, axes, forward, fct, nthreads)
            return out

    else:
        argtype = as_supported_real(x.dtype)

        def impl(x, s, axes, norm, overwrite_x, workers):
            s, axes = ndshape_and_axes(x, s, axes)
            x = zeropad_or_crop(x, s, axes, argtype)
            out = np.empty(x.shape, dtype=rettype)
            fct = get_fct(x, axes, norm, forward)
            nthreads = get_nthreads(workers)
            pocketfft.numba_c2c_sym(x, out, axes, forward, fct, nthreads)
            return out

    return impl


class HeaderOnlyError(NotImplementedError):
    """This error guards headers used by the FFTBuilder"""


def _numpy_c1d(a, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError("Numpy complex 1D header cannot be called!")

def _numpy_c2d(a, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError("Numpy complex 2D header cannot be called!")

def _numpy_cnd(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError("Numpy complex ND header cannot be called!")

def _scipy_c1d(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError("Scipy complex 1D header cannot be called!")

def _scipy_c2d(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError("Scipy complex 2D header cannot be called!")

def _scipy_cnd(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None):
    raise HeaderOnlyError("Scipy complex ND header cannot be called!")


numpy_c1d_builder = FFTBuilder(_numpy_c1d, typing_checker=fft_typing)
numpy_c2d_builder = FFTBuilder(_numpy_c2d, typing_checker=fft_typing)
numpy_cnd_builder = FFTBuilder(_numpy_cnd, typing_checker=fft_typing)
scipy_c1d_builder = FFTBuilder(_scipy_c1d, typing_checker=fft_typing)
scipy_c2d_builder = FFTBuilder(_scipy_c2d, typing_checker=fft_typing)
scipy_cnd_builder = FFTBuilder(_scipy_cnd, typing_checker=fft_typing)


numpy_c1d_builder(c2cn, forward=True).overload(numpy.fft.fft)
numpy_c2d_builder(c2cn, forward=True).overload(numpy.fft.fft2)
numpy_cnd_builder(c2cn, forward=True).overload(numpy.fft.fftn)
numpy_c1d_builder(c2cn, forward=False).overload(numpy.fft.ifft)
numpy_c2d_builder(c2cn, forward=False).overload(numpy.fft.ifft2)
numpy_cnd_builder(c2cn, forward=False).overload(numpy.fft.ifftn)

if _scipy_installed_:
    scipy_c1d_builder(c2cn, forward=True).overload(scipy.fft.fft)
    scipy_c2d_builder(c2cn, forward=True).overload(scipy.fft.fft2)
    scipy_cnd_builder(c2cn, forward=True).overload(scipy.fft.fftn)
    scipy_c1d_builder(c2cn, forward=False).overload(scipy.fft.ifft)
    scipy_c2d_builder(c2cn, forward=False).overload(scipy.fft.ifft2)
    scipy_cnd_builder(c2cn, forward=False).overload(scipy.fft.ifftn)


@register_jitable
def decrease_shape(shape, axes):
    idx = axes[-1]
    n = (shape[idx] // 2) + 1
    shape = tuple_setitem(shape, idx, n)
    return shape


def r2cn(args, forward):
    x, *_ = args

    if isinstance(x.dtype, types.Complex):
        raise TypingError(f"unsupported dtype {x.dtype}")

    argtype = as_supported_real(x.dtype)
    rettype = as_supported_cmplx(argtype)

    def impl(x, s, axes, norm, overwrite_x, workers):
        s, axes = ndshape_and_axes(x, s, axes)
        x = zeropad_or_crop(x, s, axes, argtype)
        shape = decrease_shape(x.shape, axes)
        out = np.empty(shape, dtype=rettype)
        fct = get_fct(x, axes, norm, forward)
        nthreads = get_nthreads(workers)
        pocketfft.numba_r2c(x, out, axes, forward, fct, nthreads)
        return out

    return impl


numpy_c1d_builder(r2cn, forward=True).overload(numpy.fft.rfft)
numpy_c2d_builder(r2cn, forward=True).overload(numpy.fft.rfft2)
numpy_cnd_builder(r2cn, forward=True).overload(numpy.fft.rfftn)
numpy_c1d_builder(r2cn, forward=False).overload(numpy.fft.ihfft)

if _scipy_installed_:
    scipy_c1d_builder(r2cn, forward=True).overload(scipy.fft.rfft)
    scipy_c2d_builder(r2cn, forward=True).overload(scipy.fft.rfft2)
    scipy_cnd_builder(r2cn, forward=True).overload(scipy.fft.rfftn)
    scipy_c1d_builder(r2cn, forward=False).overload(scipy.fft.ihfft)
    scipy_c2d_builder(r2cn, forward=False).overload(scipy.fft.ihfft2)
    scipy_cnd_builder(r2cn, forward=False).overload(scipy.fft.ihfftn)


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


def c2rn(args, forward):
    x, *_ = args

    argtype = as_supported_cmplx(x.dtype)
    rettype = as_supported_real(argtype)

    def impl(x, s, axes, norm, overwrite_x, workers):
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


numpy_c1d_builder(c2rn, forward=False).overload(numpy.fft.irfft)
numpy_c2d_builder(c2rn, forward=False).overload(numpy.fft.irfft2)
numpy_cnd_builder(c2rn, forward=False).overload(numpy.fft.irfftn)
numpy_c1d_builder(c2rn, forward=True).overload(numpy.fft.hfft)

if _scipy_installed_:
    scipy_c1d_builder(c2rn, forward=False).overload(scipy.fft.irfft)
    scipy_c2d_builder(c2rn, forward=False).overload(scipy.fft.irfft2)
    scipy_cnd_builder(c2rn, forward=False).overload(scipy.fft.irfftn)
    scipy_c1d_builder(c2rn, forward=True).overload(scipy.fft.hfft)
    scipy_c2d_builder(c2rn, forward=True).overload(scipy.fft.hfft2)
    scipy_cnd_builder(c2rn, forward=True).overload(scipy.fft.hfftn)


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
        raise ValueError("Invalid type; must be one of (1, 2, 3, 4).")
    return type


@get_type.impl(forward=is_literal_bool(False))
def _(type, forward):
    if type == 2:
        return 3
    if type == 3:
        return 2
    if type not in (1, 4):
        raise ValueError("Invalid type; must be one of (1, 2, 3, 4).")
    return type


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


def r2rn(args, trafo, delta, forward):
    x, _, s, *_ = args

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


def _scipy_r1d(x, type=2, n=None, axis=-1, norm=None,
               overwrite_x=False, workers=None, orthogonalize=None):
    raise HeaderOnlyError("Scipy real 1D header cannot be called!")

def _scipy_rnd(x, type=2, s=None, axes=None, norm=None,
               overwrite_x=False, workers=None, orthogonalize=None):
    raise HeaderOnlyError("Scipy real ND header cannot be called!")


scipy_r1d_builder = FFTBuilder(_scipy_r1d, typing_checker=fft_typing)
scipy_rnd_builder = FFTBuilder(_scipy_rnd, typing_checker=fft_typing)


if _scipy_installed_:
    _common_dct = dict(trafo=pocketfft.numba_dct, delta=-1)
    scipy_r1d_builder(r2rn, **_common_dct, forward=True).overload(scipy.fft.dct)
    scipy_rnd_builder(r2rn, **_common_dct, forward=True).overload(scipy.fft.dctn)
    scipy_r1d_builder(r2rn, **_common_dct, forward=False).overload(scipy.fft.idct)
    scipy_rnd_builder(r2rn, **_common_dct, forward=False).overload(scipy.fft.idctn)

    _common_dst = dict(trafo=pocketfft.numba_dst, delta=1)
    scipy_r1d_builder(r2rn, **_common_dst, forward=True).overload(scipy.fft.dst)
    scipy_rnd_builder(r2rn, **_common_dst, forward=True).overload(scipy.fft.dstn)
    scipy_r1d_builder(r2rn, **_common_dst, forward=False).overload(scipy.fft.idst)
    scipy_rnd_builder(r2rn, **_common_dst, forward=False).overload(scipy.fft.idstn)


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
            raise ValueError("axis is out of bounds")
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


@implements_overload(np.roll)
def roll(a, shift, axis=None):
    typing_check((types.Number, types.Boolean), as_seq=True)(
        a, "The 1st argument 'a' must be array-like.")
    typing_check((types.Integer, types.Boolean), as_seq=True)(
        shift, ("The 2nd argument 'shift' must be a"
                " sequence of integers or an integer."))
    typing_check((types.Integer, types.Boolean), as_seq=True, allow_none=True)(
        axis, ("If specified, the 3rd argument 'axis' must be a"
               " sequence of integers or an integer."))

                     
@roll.impl(a=is_scalar)
def _(a, shift, axis=None):
    return np.asarray(a)


@roll.impl(axis=is_nonelike)
def _(a, shift, axis=None):
    arr = np.asarray(a)
    out = np.empty(arr.shape, dtype=arr.dtype)

    shift = np.asarray(shift)
    if shift.ndim > 1:
        ValueError("'shift' must be scalars or 1D sequence")
        
    sh = shift.sum() % (arr.size or 1)
    inv_sh = arr.size - sh
    
    for i in range(inv_sh):
        out.flat[sh + i] = arr.flat[i]
    for i in range(sh):
        out.flat[i] = arr.flat[inv_sh + i]
    
    return out


@roll.impl(a=is_contiguous_array(layout="C"))
def _(a, shift, axis=None):
    return _roll_core_impl(a, shift, axis)


@register_jitable
def _transpose_axes(axis, ndim):
    axis = np.atleast_1d(np.asarray(axis))
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


fftshift_typing = tu.TypingChecker(
    x=tu.Check(
        types.Array,
        msg="The {} argument '{}' must be an array."),
    axes=tu.Check(
        types.Integer, as_seq=True, allow_none=True,
        msg="The {} argument '{}' must be a sequence of integers or an integer."),
)


@implements_overload(np.fft.fftshift)
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
    shift = x.shape[:len(axes)]
    for i, ax in enumerate(axes):
        shift = tuple_setitem(shift, i, x.shape[ax] // 2)
    return np.roll(x, shift, axes)


@implements_overload(np.fft.ifftshift)
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
    shift = x.shape[:len(axes)]
    for i, ax in enumerate(axes):
        shift = tuple_setitem(shift, i, -(x.shape[ax] // 2))
    return np.roll(x, shift, axes)


fftfreq_typing = tu.TypingChecker(
    n=tu.Check(
        types.Integer,
        msg="The {} argument '{}' must be an integer."),
    d=tu.Check(
        types.Number,
        msg="The {} argument '{}' must be a scalar."),
)


@overload(np.fft.fftfreq)
def fftfreq(n, d=1.0):
    fftfreq_typing(n=n, d=d)

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
    fftfreq_typing(n=n, d=d)

    def impl(n, d=1.0):
        val = 1.0 / (n * d)
        N = n // 2 + 1
        results = np.arange(N)
        return results * val

    return impl


def next_fast_len(target, real=False):
    typing_check(types.Integer)(
        target, "The 1st argument 'target' must be an integer.")
    typing_check(types.Boolean)(
        real, "The 2nd argument 'real' must be a boolean.")

    def impl(target, real=False):
        if target < 0:
            raise ValueError("Target cannot be negative.")
        return pocketfft.numba_good_size(target, real)

    return impl


if _scipy_installed_:
    overload(scipy.fft.next_fast_len)(next_fast_len)
    
# The following functions are adopted from SciPy:
# https://github.com/scipy/scipy/blob/main/scipy/fft/_fftlog.py
# These function cannot be used without SciPy as they require
# scipy.special.cython_special to work.
    
if _scipy_installed_:
    fht_typing = tu.TypingChecker(
        a=tu.Check(
            types.Array,
            msg="The {} argument '{}' must be an array."),
        A=tu.Check(
            types.Array,
            msg="The {} argument '{}' must be an array."),
        dln=tu.Check(
            types.Number, 
            msg="The {} argument '{}' must be a scalar."),
        mu=tu.Check(
            types.Number, 
            msg="The {} argument '{}' must be a scalar."),
        initial=tu.Check(
            types.Number, 
            msg="The {} argument '{}' must be a scalar."),
        bias=tu.Check(
            types.Number, 
            msg="The {} argument '{}' must be a scalar."),
        offset=tu.Check(
            types.Number, 
            msg="The {} argument '{}' must be a scalar."),
        )


    @register_jitable
    def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0):
        lnkr = offset
        q = bias
        xp = (mu+1+q)/2
        xm = (mu+1-q)/2
        y = np.linspace(0, np.pi*(n//2)/(n*dln), n//2+1)
        u = np.empty(n//2+1, dtype=np.complex128)
        v = np.empty(n//2+1, dtype=np.complex128)
        u[:] = xm + y*1j
        special.loggamma(u, v)
        u.real[:] = xp
        special.loggamma(u, u)
        y *= 2*(np.log(2) - lnkr)
        u = np.exp((u.real - v.real + np.log(2)*q) + (u.imag + v.imag + y)*1j)
        u.imag[-1] = 0
        if not np.isfinite(u[0]):
            u[0] = 2**q * special.poch(xm, xp-xm)
        return u


    @overload(scipy.fft.fhtoffset)
    def fhtoffset(dln, mu, initial=0.0, bias=0.0):
        fht_typing(dln=dln, mu=mu, initial=initial, bias=bias)
        
        def impl(dln, mu, initial=0.0, bias=0.0):
            lnkr = initial
            q = bias
            
            xp = (mu+1+q)/2
            xm = (mu+1-q)/2
            y = np.pi/(2*dln)
            zp = special.loggamma(xp + 1j*y)
            zm = special.loggamma(xm + 1j*y)
            arg = (np.log(2) - lnkr)/dln + (zp.imag + zm.imag)/np.pi
            return lnkr + (arg - np.round(arg))*dln

        return impl
        
        
    @register_jitable
    def _fhtq(a, u):
        if np.isinf(u[0]):
            # TODO: Is there a better solution for dealing with warnings?
            print("WARNING: singular transform; consider changing the bias")
            u = u.copy()
            u[0] = 0
        A = np.fft.rfft(a) 
        A *= u
        A = np.fft.irfft(A, a.shape[-1])
        return A[..., ::-1]


    @register_jitable
    def _ifhtq(a, u):
        if u[0] == 0:
            # TODO: Is there a better solution for dealing with warnings?
            print("WARNING: singular inverse transform; consider changing the bias")
            u = u.copy()
            u[0] = np.inf
        A = np.fft.rfft(a) 
        A /= u.conj()
        A = np.fft.irfft(A, a.shape[-1])
        return A[..., ::-1]


    @overload(scipy.fft.fht)
    def fht(a, dln, mu, offset=0.0, bias=0.0):
        fht_typing(a=a, dln=dln, mu=mu, offset=offset, bias=bias)
        
        dtype = _scipy_real_lut.get(a.dtype)
        if isinstance(dtype, types.Complex):
            raise TypingError("The 1st argument 'a' must be a real array.")

        def impl(a, dln, mu, offset=0.0, bias=0.0):
            a = np.asarray(a, dtype=dtype)
            dln = dtype(dln)
            mu = dtype(mu)
            offset = dtype(offset)
            bias = dtype(bias)
            
            n = a.shape[-1]
            if bias != 0:
                j_c = (n-1)/2
                j = np.arange(n)
                a = a * np.exp(-bias*(j - j_c)*dln)
                a = np.asarray(a, dtype=dtype)
            u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
            A = _fhtq(a, u)
            if bias != 0:
                A *= np.exp(-bias*((j - j_c)*dln + offset))
            return A
        
        return impl


    @overload(scipy.fft.ifht)
    def ifht(A, dln, mu, offset=0.0, bias=0.0):
        fht_typing(A=A, dln=dln, mu=mu, offset=offset, bias=bias)

        dtype = _scipy_real_lut.get(A.dtype)
        if isinstance(dtype, types.Complex):
            raise TypingError("The 1st argument 'A' must be a real array.")

        def impl(A, dln, mu, offset=0.0, bias=0.0):
            A = np.asarray(A, dtype=dtype)
            dln = dtype(dln)
            mu = dtype(mu)
            offset = dtype(offset)
            bias = dtype(bias)
            
            n = A.shape[-1]
            if bias != 0:
                j_c = (n-1)/2
                j = np.arange(n)
                A = A * np.exp(bias*((j - j_c)*dln + offset))
                A = np.asarray(A, dtype=dtype)
            u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
            a = _ifhtq(A, u)
            if bias != 0:
                a /= np.exp(-bias*(j - j_c)*dln)
            return a
        
        return impl