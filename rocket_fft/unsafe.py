from functools import partial

from numba import types

from .overloads import _as_cmplx_lut, _as_float_lut


def get_mapping_table(real):
    if not isinstance(real, bool):
        raise TypeError("The 1st argument 'real' must be a boolean.")

    return _as_float_lut if real else _as_cmplx_lut


def is_mapped_to(ty, real):
    if not isinstance(ty, types.Type):
        raise TypeError("The 1st argument 'ty' must be a Numba type.")
    if not isinstance(real, bool):
        raise TypeError("The 2nd argument 'real' must be a boolean.")

    lut = get_mapping_table(real)
    return lut[ty]


is_mapped_to_cmplx = partial(is_mapped_to, real=False)
is_mapped_to_real = partial(is_mapped_to, real=True)


def update_mapping_table(argty, retty, real=None):
    if not all(isinstance(ty, types.Type) for ty in (argty, retty)):
        raise TypeError(
            "The first two arguments 'argty' and 'retty' must be Numba types.")

    if real is None:
        real = isinstance(retty, types.Complex)
    if not isinstance(real, bool):
        raise TypeError("The 3rd argument 'real' must be a boolean.")

    if real:
        supported = (types.float32, types.float64)
    else:
        supported = (types.complex64, types.complex128)
    if retty not in supported:
        raise TypeError(
            f"Unsupported return type {retty}; must be one of {supported}.")

    lut = get_mapping_table(real)
    lut[argty] = retty


update_mapping_table_cmplx = partial(update_mapping_table, real=False)
update_mapping_table_real = partial(update_mapping_table, real=True)
