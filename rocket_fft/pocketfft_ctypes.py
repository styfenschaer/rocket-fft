import ctypes

from numba import generated_jit

# array_as_voidptr is used in template string
from .pocketfft_utils import array_as_voidptr, load_pocketfft


class Pocketfft:
    def __init__(self):
        self.dll = load_pocketfft()

    def _get_cmplx(self, fname):
        func = getattr(self.dll, fname)
        func.argtypes = [
            ctypes.c_size_t,  # ndim
            ctypes.c_void_p,  # ain
            ctypes.c_void_p,  # aout
            ctypes.c_void_p,  # axes
            ctypes.c_bool,  # forward
            ctypes.c_double,  # fct
            ctypes.c_size_t,  # nthreads
        ]
        return func

    c2c = property(lambda self: self._get_cmplx("numba_c2c"))
    r2c = property(lambda self: self._get_cmplx("numba_r2c"))
    c2r = property(lambda self: self._get_cmplx("numba_c2r"))
    c2c_sym = property(lambda self: self._get_cmplx("numba_c2c_sym"))

    def _get_real(self, fname):
        func = getattr(self.dll, fname)
        func.argtypes = [
            ctypes.c_size_t,  # ndim
            ctypes.c_void_p,  # ain
            ctypes.c_void_p,  # aout
            ctypes.c_void_p,  # axes
            ctypes.c_int64,  # type
            ctypes.c_double,  # fct
            ctypes.c_bool,  # ortho
            ctypes.c_size_t,  # nthreads
        ]
        return func

    dct = property(lambda self: self._get_real("numba_dct"))
    dst = property(lambda self: self._get_real("numba_dst"))

    def _get_hartley(self, fname):
        func = getattr(self.dll, fname)
        func.argtypes = [
            ctypes.c_size_t,  # ndim
            ctypes.c_void_p,  # ain
            ctypes.c_void_p,  # aout
            ctypes.c_void_p,  # axes
            ctypes.c_double,  # fct
            ctypes.c_size_t,  # nthreads
        ]
        return func

    separable_hartley = property(lambda self: self._get_hartley("numba_separable_hartley"))
    genuine_hartley = property(lambda self: self._get_hartley("numba_genuine_hartley"))

    @property
    def fftpack(self):
        fftpack = self.dll.numba_fftpack
        fftpack.argtypes = [
            ctypes.c_size_t,  # ndim
            ctypes.c_void_p,  # ain
            ctypes.c_void_p,  # aout
            ctypes.c_void_p,  # axes
            ctypes.c_bool,  # real2hermitian
            ctypes.c_bool,  # forward
            ctypes.c_double,  # fct
            ctypes.c_size_t,  # nthreads
        ]
        return fftpack

    @property
    def good_size(self):
        good_size = self.dll.numba_good_size
        good_size.argtypes = [
            ctypes.c_size_t,  # target
            ctypes.c_bool,  # real
        ]
        good_size.restype = ctypes.c_size_t
        return good_size


pocketfft = Pocketfft()


_tmpl = """
def _(ain, aout, axes, {0}):
    func = pocketfft.{1}
    ndim = ain.ndim 

    def impl(ain, aout, axes, {0}):
        ain_ptr = array_as_voidptr(ain)
        aout_ptr = array_as_voidptr(aout)
        ax_ptr = array_as_voidptr(axes)
        func(ndim, ain_ptr, aout_ptr, ax_ptr, {0})
    
    return impl
"""


class Builder:
    def __init__(self, *extra_args):
        self.extra_args = ", ".join(extra_args)

    def __call__(self, fname):
        src = _tmpl.format(self.extra_args, fname)
        exec(src)
        func = locals()["_"]
        func.__name__ = fname
        return generated_jit(func)


cmplx_builder = Builder("forward", "fct", "nthreads")
numba_c2c = cmplx_builder("c2c")
numba_r2c = cmplx_builder("r2c")
numba_c2r = cmplx_builder("c2r")
numba_c2c_sym = cmplx_builder("c2c_sym")

real_builder = Builder("type", "fct", "ortho", "nthreads")
numba_dst = real_builder("dst")
numba_dct = real_builder("dct")

hartley_builder = Builder("fct", "nthreads")
numba_separable_hartley = hartley_builder("separable_hartley")
numba_genuine_hartley = hartley_builder("genuine_hartley")

fftpack_builder = Builder("real2hermitian", "forward", "fct", "nthreads")
numba_fftpack = fftpack_builder("fftpack")

numba_good_size = pocketfft.good_size
