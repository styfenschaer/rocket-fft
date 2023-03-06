from functools import partial

from llvmlite import binding, ir
from numba import TypingError, generated_jit, types, vectorize
from numba.core.cgutils import get_or_insert_function
from numba.extending import get_cython_function_address as _gcfa
from numba.extending import intrinsic

from .extutils import load_extension_library

load_extension_library("_special_helpers")

get_special_function_address = partial(_gcfa, "scipy.special.cython_special")

ll_longlong = ir.IntType(64)
ll_double = ir.DoubleType()
ll_double_ptr = ll_double.as_pointer()
ll_complex128 = ir.LiteralStructType([ll_double, ll_double])
ll_complex128_ptr = ll_complex128.as_pointer()


def __pyx_fuse_0loggamma_wrapped(builder, real, imag):
    fnty = ir.FunctionType(ll_double_ptr, (ll_longlong, ll_double, ll_double))
    fname = "__pyx_fuse_0loggamma_call_by_address"
    fn = get_or_insert_function(builder.module, fnty, fname)
    addr = get_special_function_address("__pyx_fuse_0loggamma")
    real = builder.fpext(real, ll_double)
    imag = builder.fpext(imag, ll_double)
    return builder.call(fn, [ll_longlong(addr), real, imag])


def complex128_from_pointer(builder, ptr):
    ptr = builder.bitcast(ptr, ll_complex128_ptr)
    return builder.load(ptr)


@intrinsic
def _intr_complex_loggamma(typingctx, z):
    if not isinstance(z, types.Complex):
        raise TypingError("Argument 'z' must be a complex")

    def codegen(context, builder, sig, args):
        real = builder.extract_value(args[0], 0)
        imag = builder.extract_value(args[0], 1)
        ret_ptr = __pyx_fuse_0loggamma_wrapped(builder, real, imag)
        ret = complex128_from_pointer(builder, ret_ptr)
        return ret

    sig = types.complex128(z)
    return sig, codegen


@intrinsic
def _intr_real_loggamma(typingctx, z):
    if not isinstance(z, types.Float):
        raise TypingError("Argument 'z' must be a float")

    def codegen(context, builder, sig, args):
        fname = "__pyx_fuse_1loggamma"
        addr = get_special_function_address(fname)
        binding.add_symbol(fname, addr)
        fnty = ir.FunctionType(ll_double, (ll_double,))
        fn = get_or_insert_function(builder.module, fnty, fname)
        z = builder.fpext(args[0], ll_double)
        return builder.call(fn, [z])

    sig = types.double(z)
    return sig, codegen


@generated_jit
def _loggamma(z):
    if isinstance(z, types.Complex):
        return lambda z: _intr_complex_loggamma(z)
    if isinstance(z, types.Float):
        return lambda z: _intr_real_loggamma(z)
    raise TypingError("Argument 'z' must be a float or complex")


@intrinsic
def _intr_poch(typingctx, z, m):
    if not isinstance(z, types.Float):
        raise TypingError("First argument 'z' must be a float")
    if not isinstance(m, types.Float):
        raise TypingError("Second argument 'm' must be a float")

    def codegen(context, builder, sig, args):
        addr = get_special_function_address("poch")
        binding.add_symbol("poch", addr)
        z = builder.fpext(args[0], ll_double)
        m = builder.fpext(args[1], ll_double)
        fnty = ir.FunctionType(ll_double, (ll_double, ll_double))
        fn = get_or_insert_function(builder.module, fnty, "poch")
        return builder.call(fn, [z, m])

    sig = types.double(z, m)
    return sig, codegen

    
@vectorize
def loggamma(z):
    return _loggamma(z)

    
@vectorize
def poch(z, m):
    return _intr_poch(z, m)