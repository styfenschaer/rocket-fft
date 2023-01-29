from . import pocketfft
from ._version import __version__
from .overloads import numpy_like, scipy_like

try:
    import scipy
    scipy_like()
except ImportError:
    numpy_like()
        
        
c2c = pocketfft.numba_c2c
r2c = pocketfft.numba_r2c
c2r = pocketfft.numba_c2r
c2c_sym = pocketfft.numba_c2c_sym

dst = pocketfft.numba_dst
dct = pocketfft.numba_dct

r2r_separable_hartley = pocketfft.numba_r2r_separable_hartley
r2r_genuine_hartley = pocketfft.numba_r2r_genuine_hartley
r2r_fftpack = pocketfft.numba_r2r_fftpack

separable_hartley = r2r_separable_hartley
genuine_hartley = r2r_genuine_hartley
fftpack = r2r_fftpack

good_size = pocketfft.numba_good_size


def _init_extension():
    ...
