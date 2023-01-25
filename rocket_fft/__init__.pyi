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
) -> None: ...
def r2c(
    ain: Union[NDArray[float32], NDArray[float64]], 
    aout: Union[NDArray[complex64], NDArray[complex128]], 
    axes: NDArray[int64], 
    forward: bool8, 
    fct: float64, 
    nthreads: int64
) -> None: ...
def c2r(
    ain: Union[NDArray[complex64], NDArray[complex128]], 
    aout: Union[NDArray[float32], NDArray[float64]], 
    axes: NDArray[int64], 
    forward: bool8, 
    fct: float64, 
    nthreads: int64
) -> None: ...
def c2c_sym(
    ain: Union[NDArray[float32], NDArray[float64]], 
    aout: Union[NDArray[complex64], NDArray[complex128]], 
    axes: NDArray[int64], 
    forward: bool8, 
    fct: float64, 
    nthreads: int64
) -> None: ...
def dst(
    ain: Union[NDArray[float32], NDArray[float64]], 
    aout: Union[NDArray[float32], NDArray[float64]], 
    axes: NDArray[int64], 
    type: int64, 
    fct: float64, 
    ortho: bool8, 
    nthreads: int64
) -> None: ...
def dct(
    ain: Union[NDArray[float32], NDArray[float64]], 
    aout: Union[NDArray[float32], NDArray[float64]], 
    axes: NDArray[int64], 
    type: int64, 
    fct: float64, 
    ortho: bool8, 
    nthreads: int64
) -> None: ...
def separable_hartley(
    ain: Union[NDArray[float32], NDArray[float64]], 
    aout: Union[NDArray[float32], NDArray[float64]], 
    axes: NDArray[int64], 
    fct: float64, 
    nthreads: int64
) -> None: ...
def genuine_hartley(
    ain: Union[NDArray[float32], NDArray[float64]], 
    aout: Union[NDArray[float32], NDArray[float64]], 
    axes: NDArray[int64], 
    fct: float64, 
    nthreads: int64
) -> None: ...
def fftpack(
    ain: Union[NDArray[float32], NDArray[float64]], 
    aout: Union[NDArray[float32], NDArray[float64]], 
    axes: NDArray[int64], 
    real2hermitian: bool8, 
    forward: bool8, 
    fct: float64, 
    nthreads: int64
) -> None: ...

def good_size(target: int64, real: bool8) -> int64: ...

def scipy_like() -> None: ...
def numpy_like() -> None: ...