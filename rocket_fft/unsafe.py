from contextlib import contextmanager

from numba import types

from .overloads import FFTBuilder, _as_cmplx_lut, _as_real_lut


def get_mapping_table(real):
    if real not in (True, False):
        raise TypeError("The 1st argument 'real' must be a boolean.")

    return _as_real_lut if real else _as_cmplx_lut


def maps_to(ty, real):
    if not isinstance(ty, types.Type):
        raise TypeError("The 1st argument 'ty' must be a Numba type.")
    if real not in (True, False):
        raise TypeError("The 2nd argument 'real' must be a boolean.")

    lut = get_mapping_table(real)
    return lut[ty]


def update_mapping_table(argty, retty, real=None):
    if not all(isinstance(ty, types.Type) for ty in (argty, retty)):
        raise TypeError(
            "The first two arguments 'argty' and 'retty' must be Numba types.")

    real = isinstance(retty, types.Float) if real is None else real
    if real not in (True, False):
        raise TypeError("The 3rd argument 'real' must be a boolean.")

    supported = (types.f4, types.f8) if real else (types.c8, types.c16)
    if retty not in supported:
        raise TypeError(
            f"Unsupported return type {retty}; must be one of {supported}.")

    lut = get_mapping_table(real)
    lut[argty] = retty


def _get_builder(overloaded_func):
    entry = FFTBuilder.register.get(overloaded_func)
    if entry is None:
        raise ValueError(f"No FFT builder found for {overloaded_func}")

    builder, _ = entry
    return builder


def disable_typing_check(overloaded_func):
    builder = _get_builder(overloaded_func)
    builder.typing_checker = None


