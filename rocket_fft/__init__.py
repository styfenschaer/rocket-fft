from .overloads import numpy_like, scipy_like, rocketfft_like


def _init_extension():
    from . import overloads