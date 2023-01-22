from .overloads import numpy_like, pocketfft, scipy_like

scipy_like()


c2c = pocketfft.numba_c2c
r2c = pocketfft.numba_r2c
c2r = pocketfft.numba_c2r
c2c_sym = pocketfft.numba_c2c_sym
dst = pocketfft.numba_dst
dct = pocketfft.numba_dct
separable_hartley = pocketfft.numba_separable_hartley
genuine_hartley = pocketfft.numba_genuine_hartley
fftpack = pocketfft.numba_fftpack
good_size = pocketfft.numba_good_size


def _init_extension():
    pass
