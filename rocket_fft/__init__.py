"""Rocket-FFT: ``numpy.fft`` and ``scipy.fft`` for Numba.

Importing this package registers Numba overloads for ``numpy.fft`` and, if
SciPy is installed, ``scipy.fft``. The transforms are computed by DUCC0, the
C++ library that also powers ``scipy.fft``.
"""

from ._version import __version__
from ._utils import (
    get_workers,
    good_size,
    next_fast_len,
    prev_fast_len,
    prev_good_size,
    set_workers,
)
from .lowlevel import (
    c2c,
    c2c_sym,
    c2r,
    dct,
    dst,
    fftpack,
    genuine_hartley,
    r2c,
    r2r_fftpack,
    r2r_genuine_hartley,
    r2r_separable_hartley,
    separable_hartley,
)

from . import _numpy_fft  # noqa: F401  (registers the numpy.fft overloads)
from . import _extras  # noqa: F401  (registers numpy.roll)
from . import special

try:
    import scipy.fft  # noqa: F401
except ImportError:
    pass
else:
    from . import _scipy_fft  # noqa: F401  (registers the scipy.fft overloads)


def _init_extension() -> None:
    """Entry point for Numba's ``numba_extensions`` hook."""


def scipy_like() -> None:
    """Deprecated no-op, kept for backwards compatibility.

    Each interface now follows its own library: ``scipy.fft`` rejects
    duplicate axes, ``numpy.fft`` allows them.
    """
    _warn_axes_switch_removed("scipy_like")


def numpy_like() -> None:
    """Deprecated no-op, kept for backwards compatibility.

    Each interface now follows its own library: ``scipy.fft`` rejects
    duplicate axes, ``numpy.fft`` allows them.
    """
    _warn_axes_switch_removed("numpy_like")


def _warn_axes_switch_removed(name: str) -> None:
    import warnings

    warnings.warn(
        f"rocket_fft.{name}() is deprecated and has no effect: scipy.fft now always "
        "rejects duplicate axes and numpy.fft always allows them, like the "
        "libraries themselves. You can safely remove the call.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "__version__",
    "c2c",
    "c2c_sym",
    "c2r",
    "dct",
    "dst",
    "fftpack",
    "genuine_hartley",
    "get_workers",
    "good_size",
    "next_fast_len",
    "numpy_like",
    "prev_fast_len",
    "prev_good_size",
    "r2c",
    "r2r_fftpack",
    "r2r_genuine_hartley",
    "r2r_separable_hartley",
    "scipy_like",
    "separable_hartley",
    "set_workers",
    "special",
]
