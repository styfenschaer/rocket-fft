from functools import partial

from llvmlite import binding, ir
from numba import TypingError, generated_jit, types, vectorize
from numba.core.cgutils import get_or_insert_function
from numba.extending import get_cython_function_address as _gcfa
from numba.extending import intrinsic

get_special_function_address = partial(_gcfa, "scipy.special.cython_special")
get_helpers_function_address = partial(_gcfa, "rocket_fft._special_helpers")


ll_void = ir.VoidType()
ll_int32 = ir.IntType(32)
ll_longlong = ir.IntType(64)
ll_double = ir.DoubleType()
ll_double_ptr = ll_double.as_pointer()
ll_complex128 = ir.LiteralStructType([ll_double, ll_double])


def __pyx_fuse_0loggamma(builder, real, imag, real_out, imag_out):
    fname = "__pyx_fuse_0loggamma"
    addr = get_helpers_function_address(fname)
    binding.add_symbol(fname, addr)
    
    arg_types = (ll_double, ll_double, ll_double_ptr, ll_double_ptr)
    fnty = ir.FunctionType(ll_void, arg_types)
    fn = get_or_insert_function(builder.module, fnty, fname)
    
    real = builder.fpext(real, ll_double)
    imag = builder.fpext(imag, ll_double)
    real_out = builder.bitcast(real_out, ll_double_ptr)
    imag_out = builder.bitcast(imag_out, ll_double_ptr)
    builder.call(fn, [real, imag, real_out, imag_out])
    

@intrinsic
def _complex_loggamma(typingctx, z):
    if not isinstance(z, types.Complex):
        raise TypingError("Argument 'z' must be a complex")

    def codegen(context, builder, sig, args):
        real = builder.extract_value(args[0], 0)
        imag = builder.extract_value(args[0], 1)
        real_out = builder.alloca(ll_double)
        imag_out = builder.alloca(ll_double)
        __pyx_fuse_0loggamma(builder, real, imag, real_out, imag_out)
        zout = builder.alloca(ll_complex128)
        zout_real = builder.gep(zout, [ll_int32(0), ll_int32(0)])
        zout_imag = builder.gep(zout, [ll_int32(0), ll_int32(1)])
        builder.store(builder.load(real_out), zout_real)
        builder.store(builder.load(imag_out), zout_imag)
        return builder.load(zout)

    sig = types.complex128(z)
    return sig, codegen


@intrinsic
def _real_loggamma(typingctx, z):
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
        return lambda z: _complex_loggamma(z)
    if isinstance(z, types.Float):
        return lambda z: _real_loggamma(z)
    raise TypingError("Argument 'z' must be a float or complex")


@intrinsic
def _poch(typingctx, z, m):
    if not isinstance(z, types.Float):
        raise TypingError("First argument 'z' must be a float")
    if not isinstance(m, types.Float):
        raise TypingError("Second argument 'm' must be a float")

    def codegen(context, builder, sig, args):
        fname = "poch"
        addr = get_special_function_address(fname)
        binding.add_symbol(fname, addr)
        
        fnty = ir.FunctionType(ll_double, (ll_double, ll_double))
        fn = get_or_insert_function(builder.module, fnty, fname)
        
        z = builder.fpext(args[0], ll_double)
        m = builder.fpext(args[1], ll_double)
        return builder.call(fn, [z, m])

    sig = types.double(z, m)
    return sig, codegen

    
@vectorize
def loggamma(z):
    return _loggamma(z)

    
@vectorize
def poch(z, m):
    return _poch(z, m)