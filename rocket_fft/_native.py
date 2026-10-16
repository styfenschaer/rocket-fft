"""Bindings to the native frontend in ``_rocket_fft_numba.cpp``.

The enums below mirror the ABI defined at the top of the C++ file and must be
kept in sync with it.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from enum import IntEnum, IntFlag

from llvmlite.binding import load_library_permanently
from numba.core import types
from numba.core.runtime import _nrt_python


def _library_path(name: str) -> str | None:
    spec = importlib.util.find_spec(f"{__package__}.{name}")
    return None if spec is None else spec.origin


def _select_library() -> tuple[str, str]:
    """Picks the native library build: AVX2/FMA if available, else generic.

    The generic build always exists; x86-64 wheels also contain an AVX2/FMA build.
    Set ROCKET_FFT_SIMD to "generic" or "avx2" to force one of them.
    """
    generic = _library_path("_rocket_fft_numba")
    choice = os.environ.get("ROCKET_FFT_SIMD", "auto").strip().lower()
    if choice not in ("auto", "generic", "avx2"):
        raise ValueError(f"ROCKET_FFT_SIMD must be 'auto', 'generic' or 'avx2', not {choice!r}")
    if choice == "generic":
        return generic, "generic"
    avx2 = _library_path("_rocket_fft_numba_avx2")
    supported = avx2 is not None and ctypes.CDLL(generic).rf_cpu_supports_avx2() != 0
    if supported:
        return avx2, "avx2"
    if choice == "avx2":
        reason = "is not built for this platform" if avx2 is None else "needs a CPU with AVX2 and FMA"
        raise ImportError(f"ROCKET_FFT_SIMD=avx2: the AVX2 build of rocket-fft {reason}")
    return generic, "generic"


LIBRARY_PATH, SIMD = _select_library()
_lib = ctypes.CDLL(LIBRARY_PATH)

# Must match kAbiVersion in _rocket_fft_numba.cpp; bump both together.
ABI_VERSION = 1


def _check_abi_version() -> None:
    try:
        found = _lib.rf_abi_version()
    except AttributeError:  # built before the version existed
        found = None
    if found != ABI_VERSION:
        raise ImportError(
            f"The native library {LIBRARY_PATH} does not match this version of rocket-fft "
            f"(ABI version {found}, expected {ABI_VERSION}). It is probably a stale build; "
            "rebuild it, for example with `pip install -e .`."
        )


_check_abi_version()

# Make the symbols visible to LLVM so that jitted code can call them directly.
load_library_permanently(LIBRARY_PATH)

# Arrays returned by the native library are allocated through Numba's runtime.
_get_nrt_api = ctypes.CFUNCTYPE(ctypes.c_void_p)(_nrt_python.c_helpers["get_api"])
_lib.rf_set_nrt_api.argtypes = [ctypes.c_void_p]
_lib.rf_set_nrt_api(_get_nrt_api())


class Kind(IntEnum):
    C2C = 0
    R2C = 1
    C2R = 2
    DCT = 3
    DST = 4
    SEPARABLE_HARTLEY = 5
    GENUINE_HARTLEY = 6
    FFTPACK = 7


class Flag(IntFlag):
    NONE = 0
    FORWARD = 1
    UNIQUE_AXES = 2
    LENGTHS_FROM_OUTPUT = 4
    REAL2HERMITIAN = 8
    DEFAULT_LAST_AXIS = 16
    SCIPY = 32


class Api(IntEnum):
    """Whose output-layout rules a transform follows."""

    NUMPY = 0
    SCIPY = 1


class Norm(IntEnum):
    INVALID = -1
    BACKWARD = 0
    ORTHO = 1
    FORWARD = 2
    EXPLICIT = 3


DTYPE_CODES: dict[types.Type, int] = {
    types.boolean: 0,
    types.int8: 1,
    types.int16: 2,
    types.int32: 3,
    types.int64: 4,
    types.uint8: 5,
    types.uint16: 6,
    types.uint32: 7,
    types.uint64: 8,
    types.float32: 9,
    types.float64: 10,
    types.complex64: 11,
    types.complex128: 12,
}


def _load_status_table() -> dict[int, tuple[type[Exception], str]]:
    _lib.rf_status_count.restype = ctypes.c_int64
    _lib.rf_status_message.argtypes = [ctypes.c_int64]
    _lib.rf_status_message.restype = ctypes.c_char_p
    _lib.rf_status_exception.argtypes = [ctypes.c_int64]
    _lib.rf_status_exception.restype = ctypes.c_int64

    exceptions = {0: ValueError, 1: ZeroDivisionError, 2: MemoryError}
    table = {}
    for code in range(1, _lib.rf_status_count()):
        exc = exceptions[_lib.rf_status_exception(code)]
        table[code] = (exc, _lib.rf_status_message(code).decode())
    return table


# Maps every non-zero status code to the exception it raises.
STATUS_TABLE: dict[int, tuple[type[Exception], str]] = _load_status_table()


def raise_for_status(code: int) -> None:
    if code != 0:
        exc, msg = STATUS_TABLE[code]
        raise exc(msg)


def _bind(name: str, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


set_default_workers = _bind("rf_set_default_workers", None, ctypes.c_int64)
get_default_workers = _bind("rf_get_default_workers", ctypes.c_int64)
cpu_count = _bind("rf_cpu_count", ctypes.c_int64)
good_size = _bind("rf_good_size", ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
prev_good_size = _bind("rf_prev_good_size", ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
