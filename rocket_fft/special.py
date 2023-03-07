from functools import partial

from llvmlite import binding, ir
from numba import TypingError, generated_jit, types, vectorize
from numba.core.cgutils import alloca_once_value, get_or_insert_function
from numba.extending import get_cython_function_address as _gcfa
from numba.extending import intrinsic

from .extutils import load_extension_library

load_extension_library("_special_helpers")

get_special_function_address = partial(_gcfa, "scipy.special.cython_special")

ll_void = ir.VoidType()
ll_int32 = ir.IntType(32)
ll_longlong = ir.IntType(64)
ll_double = ir.DoubleType()
ll_double_ptr = ll_double.as_pointer()
ll_complex128 = ir.LiteralStructType([ll_double, ll_double])


def __pyx_fuse_0loggamma_wrapped(builder, real, imag, real_out, imag_out):
    fnty = ir.FunctionType(ll_void, (ll_longlong, ll_double, ll_double,ll_double_ptr,ll_double_ptr))
    fname = "__pyx_fuse_0loggamma_call_by_address2"
    fn = get_or_insert_function(builder.module, fnty, fname)
    addr = get_special_function_address("__pyx_fuse_0loggamma")
    real = builder.fpext(real, ll_double)
    imag = builder.fpext(imag, ll_double)
    real_out = builder.bitcast(real_out, ll_double_ptr)
    imag_out = builder.bitcast(imag_out, ll_double_ptr)
    builder.call(fn, [ll_longlong(addr), real, imag, real_out, imag_out])


@intrinsic
def _intr_complex_loggamma(typingctx, z):
    if not isinstance(z, types.Complex):
        raise TypingError("Argument 'z' must be a complex")

    def codegen(context, builder, sig, args):
        [z] = args
        real = builder.extract_value(z, 0)
        imag = builder.extract_value(z, 1)
        real_out_ptr = builder.alloca(ll_double)
        imag_out_ptr = builder.alloca(ll_double)
        __pyx_fuse_0loggamma_wrapped(builder, real, imag, real_out_ptr, imag_out_ptr)
        zout = builder.alloca(ll_complex128)
        zout_real = builder.gep(zout, [ll_int32(0), ll_int32(0)])
        zout_imag = builder.gep(zout, [ll_int32(0), ll_int32(1)])
        real_out = builder.load(real_out_ptr)
        imag_out = builder.load(imag_out_ptr)
        builder.store(real_out, zout_real)
        builder.store(imag_out, zout_imag)
        return builder.load(zout)

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