from typing import Union

from numpy import bool8, complex64, complex128, float32, float64, int64
from numpy.typing import NDArray

def c2c(
    ain: Union[NDArray[complex64], NDArray[complex128]],
    aout: Union[NDArray[complex64], NDArray[complex128]],
    axes: NDArray[int64],
    forward: bool8,
    fct: float64,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def r2c(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[complex64], NDArray[complex128]],
    axes: NDArray[int64],
    forward: bool8,
    fct: float64,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def c2r(
    ain: Union[NDArray[complex64], NDArray[complex128]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    forward: bool8,
    fct: float64,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def c2c_sym(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[complex64], NDArray[complex128]],
    axes: NDArray[int64],
    forward: bool8,
    fct: float64,
    nthreads: int64
) -> None:
    """Similar to c2c, but takes a real-valued input array. 
    It uses symmetry to speed up the calculation.
    Please refer to https://github.com/hayguen/pocketfft for documentation.
    """


def dst(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    type: int64,
    fct: float64,
    ortho: bool8,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def dct(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    type: int64,
    fct: float64,
    ortho: bool8,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def r2r_separable_hartley(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    fct: float64,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def r2r_genuine_hartley(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    fct: float64,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def r2r_fftpack(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    real2hermitian: bool8,
    forward: bool8,
    fct: float64,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/hayguen/pocketfft for documentation."""


def good_size(target: int64, real: bool8) -> int64:
    """Find the next fast size of input data to fft, for zero-padding, etc."""


def scipy_like() -> None:
    """Use SciPy-like type conversion. 

    - Conversion to real type:\\
    float32 -> float32\\
    complex64 -> float32\\
    other -> float64  

    - Conversion to complex type:\\
    float32 -> complex64\\
    complex64 -> complex64\\
    other -> complex128  
    """


def numpy_like() -> None:
    """Use NumPy-like type conversion. 

    - Conversion to real type:\\
    any -> float64  

    - Conversion to complex type:\\
    any -> complex128  
    """


separable_hartley = r2r_separable_hartley
genuine_hartley = r2r_genuine_hartley
fftpack = r2r_fftpack
