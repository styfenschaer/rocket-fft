import ctypes
from pathlib import Path

from llvmlite import ir
from numba.core import types
from numba.extending import intrinsic
from numba.np.arrayobj import make_array


def load_pocketfft():
    search_path = Path(__file__).parent.parent
    matches = search_path.glob("**/_pocketfft_numba.so")
    libpath = str(next(matches))
    return ctypes.CDLL(libpath)


def ll_array_as_voidptr(context, builder, ary_t, ary):
    ary = make_array(ary_t)(context, builder, ary)
    ptr = ary._getpointer()
    voidptr = ir.IntType(8).as_pointer()
    return builder.bitcast(ptr, voidptr)


@intrinsic
def array_as_voidptr(typingctx, ary):
    def codegen(context, builder, sig, args):
        [ary] = args
        [ary_t] = sig.args
        ptr = ll_array_as_voidptr(context, builder, ary_t, ary)
        return ptr

    sig = types.voidptr(ary)
    return sig, codegen
