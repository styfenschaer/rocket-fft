import ctypes
from functools import partial
from pathlib import Path

from llvmlite import ir
from llvmlite.binding import load_library_permanently
from numba.core import types
from numba.core.cgutils import get_or_insert_function
from numba.extending import intrinsic
from numba.np.arrayobj import make_array

ll_size_t = ir.IntType(64)
ll_int64 = ir.IntType(64)
ll_double = ir.DoubleType()
ll_bool = ir.IntType(1)
ll_voidptr = ir.IntType(8).as_pointer()
ll_void = ir.VoidType()

void = types.void
size_t = types.int64


def load_pocketfft():
    search_path = Path(__file__).parent.parent
    matches = search_path.glob("**/_pocketfft_numba.so")
    libpath = str(next(matches))
    load_library_permanently(libpath)
    return ctypes.CDLL(libpath)


class Pocketfft:
    def __init__(self):
        self.dll = load_pocketfft()

    def _call_cmplx(fname, builder, args):
        fntype = ir.FunctionType(
            ll_void,
            (
                ll_size_t,  # ndim
                ll_voidptr,  # ain
                ll_voidptr,  # aout
                ll_voidptr,  # axes
                ll_bool,  # forward
                ll_double,  # fct
                ll_size_t,  # nthreads
            )
        )
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

    c2c = partial(_call_cmplx, "numba_c2c")
    r2c = partial(_call_cmplx, "numba_r2c")
    c2r = partial(_call_cmplx, "numba_c2r")
    c2c_sym = partial(_call_cmplx, "numba_c2c_sym")

    def _call_real(fname, builder, args):
        fntype = ir.FunctionType(
            ll_void,
            (
                ll_size_t,  # ndim
                ll_voidptr,  # ain
                ll_voidptr,  # aout
                ll_voidptr,  # axes
                ll_int64,  # type
                ll_double,  # fct
                ll_bool,  # ortho
                ll_size_t,  # nthreads
            )
        )
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

    dct = partial(_call_real, "numba_dct")
    dst = partial(_call_real, "numba_dst")

    def _call_hartley(fname, builder, args):
        fntype = ir.FunctionType(
            ll_void,
            (
                ll_size_t,  # ndim
                ll_voidptr,  # ain
                ll_voidptr,  # aout
                ll_voidptr,  # axes
                ll_double,  # fct
                ll_size_t,  # nthreads
            )
        )
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

    r2r_separable_hartley = partial(_call_hartley, "numba_r2r_separable_hartley")
    r2r_genuine_hartley = partial(_call_hartley, "numba_r2r_genuine_hartley")

    @staticmethod
    def r2r_fftpack(builder, args):
        fname = "numba_r2r_fftpack"
        fntype = ir.FunctionType(
            ll_void,
            (
                ll_size_t,  # ndim
                ll_voidptr,  # ain
                ll_voidptr,  # aout
                ll_voidptr,  # axes
                ll_bool,  # real2hermitian
                ll_bool,  # forward
                ll_double,  # fct
                ll_size_t,  # nthreads
            )
        )
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)

    @staticmethod
    def good_size(builder, args):
        fname = "numba_good_size"
        fntype = ir.FunctionType(
            ll_size_t,
            (
                ll_size_t,  # target
                ll_bool,  # real
            )
        )
        fn = get_or_insert_function(builder.module, fntype, fname)
        return builder.call(fn, args)


ll_pocketfft = Pocketfft()


def array_as_voidptr(context, builder, ary_t, ary):
    ary = make_array(ary_t)(context, builder, ary)
    ptr = ary._getpointer()
    return builder.bitcast(ptr, ll_voidptr)


_tmpl = """
def _(typingctx, ain, aout, axes, {0}):
    def codegen(context, builder, sig, args):
        ain, aout, axes, *rest = args
        ain_t, aout_t, axes_t, *_ = sig.args

        ndim = ll_size_t(ain_t.ndim)
        ain_ptr = array_as_voidptr(context, builder, ain_t, ain)
        aout_ptr = array_as_voidptr(context, builder, aout_t, aout)
        ax_ptr = array_as_voidptr(context, builder, axes_t, axes)

        args = (ndim, ain_ptr, aout_ptr, ax_ptr, *rest)
        ll_pocketfft.{1}(builder, args)
        
    sig = void(ain, aout, axes, {0})
    return sig, codegen
"""


class Builder:
    def __init__(self, *extra_args):
        self.extra_args = ", ".join(extra_args)

    def __call__(self, fname):
        src = _tmpl.format(self.extra_args, fname)
        exec(src)
        func = locals()["_"]
        func.__name__ = fname
        return intrinsic(func)


cmplx_builder = Builder("forward", "fct", "nthreads")
numba_c2c = cmplx_builder("c2c")
numba_r2c = cmplx_builder("r2c")
numba_c2r = cmplx_builder("c2r")
numba_c2c_sym = cmplx_builder("c2c_sym")

real_builder = Builder("type", "fct", "ortho", "nthreads")
numba_dst = real_builder("dst")
numba_dct = real_builder("dct")

hartley_builder = Builder("fct", "nthreads")
numba_r2r_separable_hartley = hartley_builder("r2r_separable_hartley")
numba_r2r_genuine_hartley = hartley_builder("r2r_genuine_hartley")

fftpack_builder = Builder("real2hermitian", "forward", "fct", "nthreads")
numba_r2r_fftpack = fftpack_builder("r2r_fftpack")


@intrinsic
def numba_good_size(typingctx, n, real):
    def codegen(context, builder, sig, args):
        ret = ll_pocketfft.good_size(builder, args)
        return ret

    sig = size_t(n, real)
    return sig, codegen
