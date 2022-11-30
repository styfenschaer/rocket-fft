import ctypes
from functools import partial
from pathlib import Path

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.cgutils import get_or_insert_function
from numba.extending import intrinsic
from numba.np.arrayobj import make_array


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


class Pocketfft:
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


ll_pocketfft = Pocketfft()


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
