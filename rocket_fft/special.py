import ctypes

from llvmlite import ir
from numba import types, vectorize
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload

from .extutils import ExtensionLibrary

lib = ExtensionLibrary("_special_helpers")
lib.load_permanently()

dll = ctypes.PyDLL(lib.path)
dll.init_special_functions()

ll_void = ir.VoidType()
ll_int32 = ir.IntType(32)
ll_double = ir.DoubleType()
ll_double_ptr = ll_double.as_pointer()
ll_complex128 = ir.LiteralStructType([ll_double, ll_double])


def _call_complex_loggamma(builder, real, imag, real_out, imag_out):
    fname = "numba_complex_loggamma"
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
        raise TypingError("Argument 'z' must be complex")

    def codegen(context, builder, sig, args):
        real = builder.extract_value(args[0], 0)
        imag = builder.extract_value(args[0], 1)
        real_out = builder.alloca(ll_double)
        imag_out = builder.alloca(ll_double)
        _call_complex_loggamma(builder, real, imag, real_out, imag_out)
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
        fname = "numba_real_loggamma"
        fnty = ir.FunctionType(ll_double, (ll_double,))
        fn = get_or_insert_function(builder.module, fnty, fname)

        z = builder.fpext(args[0], ll_double)
        return builder.call(fn, [z])

    sig = types.double(z)
    return sig, codegen


def _loggamma(z):
    pass


@overload(_loggamma)
def _loggamma_impl(z):
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
        fname = "numba_poch"
        fnty = ir.FunctionType(ll_double, (ll_double, ll_double))
        fn = get_or_insert_function(builder.module, fnty, fname)

        z = builder.fpext(args[0], ll_double)
        m = builder.fpext(args[1], ll_double)
        return builder.call(fn, [z, m])

    sig = types.double(z, m)
    return sig, codegen


_loggamma_sigs = (
    "float64(float32)",
    "float64(float64)",
    "complex128(complex64)",
    "complex128(complex128)",
)


@vectorize
def loggamma(z):
    return _loggamma(z)


_poch_sigs = (
    "float64(float32, float32)",
    "float64(float64, float32)",
    "float64(float32, float64)",
    "float64(float64, float64)",
)


@vectorize
def poch(z, m):
    return _poch(z, m)


def add_signatures(loggamma_sigs=None, poch_sigs=None):
    list(map(loggamma.add, loggamma_sigs or _loggamma_sigs))
    list(map(poch.add, poch_sigs or _poch_sigs))
