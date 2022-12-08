import ctypes
import glob
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
    pattern = Path(__file__).parent / '*.so'
    libpath = glob.glob(str(pattern))[0]
    return ctypes.CDLL(libpath)


def spartial(func, *args, **kargs):
    func = func.__func__
    return partial(func, *args, **kargs)


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

    c2c = spartial(_call_cmplx, 'numba_c2c')
    r2c = spartial(_call_cmplx, 'numba_r2c')
    c2r = spartial(_call_cmplx, 'numba_c2r')
    c2c_sym = spartial(_call_cmplx, 'numba_c2c_sym')

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

    dct = spartial(_call_real, 'numba_dct')
    dst = spartial(_call_real, 'numba_dst')

    @staticmethod
    def _call_hartley(fname, builder, args):
        fntype = ir.FunctionType(ll_void,  [ll_size_t,  # ndim
                                            ll_voidptr,  # ain
                                            ll_voidptr,  # aout
                                            ll_voidptr,  # axes
                                            ll_double,  # fct
                                            ll_size_t])  # nthreads
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

    separable_hartley = spartial(_call_hartley, 'numba_separable_hartley')
    genuine_hartley = spartial(_call_hartley, 'numba_genuine_hartley')

    @staticmethod
    def fftpack(builder, args):
        fname = 'numba_fftpack'
        fntype = ir.FunctionType(ll_void,  [ll_size_t,  # ndim
                                            ll_voidptr,  # ain
                                            ll_voidptr,  # aout
                                            ll_voidptr,  # axes
                                            ll_bool,  # real2hermitian
                                            ll_bool,  # forward
                                            ll_double,  # fct
                                            ll_size_t])  # nthreads
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

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


def ipartial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    return intrinsic(partial_func)


numba_c2c = ipartial(_numba_cmplx, ll_pocketfft.c2c)
numba_r2c = ipartial(_numba_cmplx, ll_pocketfft.r2c)
numba_c2r = ipartial(_numba_cmplx, ll_pocketfft.c2r)
numba_c2c_sym = ipartial(_numba_cmplx, ll_pocketfft.c2c_sym)


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


numba_dst = ipartial(_numba_real, ll_pocketfft.dst)
numba_dct = ipartial(_numba_real, ll_pocketfft.dct)


def _numba_hartley(func, typingctx, ain, aout, axes, fct, nthreads):
    def codegen(context, builder, sig, args):
        ain, aout, axes, fct, nthreads = args
        ain_t, aout_t, axes_t, *_ = sig.args

        ndim = ll_size_t(ain_t.ndim)
        ain_ptr = array_as_voidptr(context, builder, ain_t, ain)
        aout_ptr = array_as_voidptr(context, builder, aout_t, aout)
        ax_ptr = array_as_voidptr(context, builder, axes_t, axes)

        args = (ndim, ain_ptr, aout_ptr, ax_ptr, fct, nthreads)
        func(builder, args)

    sig = void(ain, aout, axes, fct, nthreads)
    return sig, codegen


_ll_pfft = ll_pocketfft
numba_separable_hartley = ipartial(_numba_hartley, _ll_pfft.separable_hartley)
numba_genuine_hartley = ipartial(_numba_hartley, _ll_pfft.genuine_hartley)


@intrinsic
def numba_fftpack(typingctx, ain, aout, axes, real2hermitian,
                  forward, fct, nthreads):
    def codegen(context, builder, sig, args):
        ain, aout, axes, real2hermitian, forward, fct, nthreads = args
        ain_t, aout_t, axes_t, *_ = sig.args

        ndim = ll_size_t(ain_t.ndim)
        ain_ptr = array_as_voidptr(context, builder, ain_t, ain)
        aout_ptr = array_as_voidptr(context, builder, aout_t, aout)
        ax_ptr = array_as_voidptr(context, builder, axes_t, axes)

        args = (ndim, ain_ptr, aout_ptr, ax_ptr,
                real2hermitian, forward, fct, nthreads)
        ll_pocketfft.fftpack(builder, args)

    sig = void(ain, aout, axes, real2hermitian, forward, fct, nthreads)
    return sig, codegen


@intrinsic
def numba_good_size(typingctx, n, real):
    def codegen(context, builder, sig, args):
        ret = ll_pocketfft.good_size(builder, args)
        return ret

    sig = size_t(n, real)
    return sig, codegen