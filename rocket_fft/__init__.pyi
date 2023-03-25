from typing import Union

from numpy import bool8, complex64, complex128, float32, float64, int64
from numpy.typing import NDArray


def c2c(
    ain: Union[NDArray[complex64], NDArray[complex128]],
    aout: Union[NDArray[complex64], NDArray[complex128]],
    axes: NDArray[int64],
    forward: bool8,
    fct: Union[float32, float64],
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


def r2c(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[complex64], NDArray[complex128]],
    axes: NDArray[int64],
    forward: bool8,
    fct: Union[float32, float64],
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


def c2r(
    ain: Union[NDArray[complex64], NDArray[complex128]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    forward: bool8,
    fct: Union[float32, float64],
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


def c2c_sym(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[complex64], NDArray[complex128]],
    axes: NDArray[int64],
    forward: bool8,
    fct: Union[float32, float64],
    nthreads: int64
) -> None:
    """Similar to c2c, but takes a real-valued input array. 
    It uses symmetry to speed up the calculation.
    Please refer to https://github.com/mreineck/pocketfft for documentation.
    """


def dst(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    type: int64,
    fct: Union[float32, float64],
    ortho: bool8,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


def dct(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    type: int64,
    fct: Union[float32, float64],
    ortho: bool8,
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


def r2r_separable_hartley(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    fct: Union[float32, float64],
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


def r2r_genuine_hartley(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    fct: Union[float32, float64],
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


def r2r_fftpack(
    ain: Union[NDArray[float32], NDArray[float64]],
    aout: Union[NDArray[float32], NDArray[float64]],
    axes: NDArray[int64],
    real2hermitian: bool8,
    forward: bool8,
    fct: Union[float32, float64],
    nthreads: int64
) -> None:
    """Please refer to https://github.com/mreineck/pocketfft for documentation."""


separable_hartley = r2r_separable_hartley
genuine_hartley = r2r_genuine_hartley
fftpack = r2r_fftpack


def good_size(target: int64, real: bool8) -> int64:
    """Find the next fast size of input data to fft, for zero-padding, etc."""


def scipy_like() -> None:
    """Use SciPy-like type conversion. 
    - Conversion to real type:\\
    `float32` -> `float32`\\
    `complex64` -> `float32`\\
    `Other` -> `float64`  
    - Conversion to complex type:\\
    `float32` -> `complex64`\\
    `complex64` -> `complex64`\\
    `Other` -> `complex128`  
    
    Handle the `axes` argument in the same way as SciPy. 
    Passing duplicate axes to the `fft2`, `fftn`, `ifft2`, or `ifftn` 
    functions will result in an error.
    """


def numpy_like() -> None:
    """Use NumPy-like type conversion. 
    - Conversion to real type:\\
    `Any` -> `float64`  
    - Conversion to complex type:\\
    `Any` -> `complex128`  
    
    Handle the `axes` argument in the same way as NumPy. 
    Passing duplicate axes to the `fft2`, `fftn`, `ifft2`, or `ifftn` 
    functions is allowed
    """


def get_workers() -> int:
    """Returns the default number of workers used."""


def set_workers(workers: int) -> None:
    """Sets the default number of workers used.
    This change cannot be undone after compilation of Rocket-FFT's internals.
    """
