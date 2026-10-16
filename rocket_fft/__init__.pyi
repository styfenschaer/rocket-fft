from typing import Union

from numpy import complex64, complex128, float32, float64, floating, integer
from numpy.typing import NDArray

from . import special as special

__version__: str

_Real = Union[NDArray[float32], NDArray[float64]]
_Complex = Union[NDArray[complex64], NDArray[complex128]]
_Axes = NDArray[Union[integer, floating]]

# Low-level interface (jitted code only). The output array determines the
# transform lengths; `fct` is applied as is and `nthreads` follows SciPy's
# `workers` semantics.

def c2c(ain: _Complex, aout: _Complex, axes: _Axes, forward: bool, fct: float, nthreads: int) -> None:
    """Complex-to-complex FFT."""

def c2c_sym(ain: _Real, aout: _Complex, axes: _Axes, forward: bool, fct: float, nthreads: int) -> None:
    """Complex FFT of real input, exploiting Hermitian symmetry."""

def r2c(ain: _Real, aout: _Complex, axes: _Axes, forward: bool, fct: float, nthreads: int) -> None:
    """Real-to-complex FFT; the last axis in `axes` is the halved one."""

def c2r(ain: _Complex, aout: _Real, axes: _Axes, forward: bool, fct: float, nthreads: int) -> None:
    """Complex-to-real FFT; the last axis in `axes` is the halved one."""

def dct(ain: _Real, aout: _Real, axes: _Axes, type: int, fct: float, ortho: bool, nthreads: int) -> None:
    """Discrete cosine transform of type 1-4."""

def dst(ain: _Real, aout: _Real, axes: _Axes, type: int, fct: float, ortho: bool, nthreads: int) -> None:
    """Discrete sine transform of type 1-4."""

def r2r_separable_hartley(ain: _Real, aout: _Real, axes: _Axes, fct: float, nthreads: int) -> None:
    """Separable Hartley transform."""

def r2r_genuine_hartley(ain: _Real, aout: _Real, axes: _Axes, fct: float, nthreads: int) -> None:
    """Genuine multidimensional Hartley transform."""

def r2r_fftpack(
    ain: _Real, aout: _Real, axes: _Axes, real2hermitian: bool, forward: bool, fct: float, nthreads: int
) -> None:
    """FFTPACK-style real-to-real transform."""

separable_hartley = r2r_separable_hartley
genuine_hartley = r2r_genuine_hartley
fftpack = r2r_fftpack

# Usable from both Python and jitted code.

def good_size(target: int, real: bool = False) -> int:
    """Smallest fast FFT length greater than or equal to `target`."""

def prev_good_size(target: int, real: bool = False) -> int:
    """Largest fast FFT length less than or equal to `target`."""

next_fast_len = good_size
prev_fast_len = prev_good_size

def get_workers() -> int:
    """Default number of worker threads."""

def set_workers(workers: int) -> None:
    """Set the default number of worker threads."""

# Deprecated no-ops, kept for backwards compatibility. Each interface now follows
# its own library: scipy.fft rejects duplicate axes, numpy.fft allows them.

def scipy_like() -> None:
    """Deprecated no-op."""

def numpy_like() -> None:
    """Deprecated no-op."""
