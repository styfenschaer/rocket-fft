from functools import partial

from numba import types

from .overloads import _as_cmplx_lut, _as_float_lut


def get_mapping_table(cmplx):
    if not isinstance(cmplx, bool):
        raise TypeError("The 1st argument 'cmplx' must be a boolean.")

    return _as_cmplx_lut if cmplx else _as_float_lut


def maps_to(ty, cmplx):
    if not isinstance(ty, types.Type):
        raise TypeError("The 1st argument must be a Numba type.")
    if not isinstance(cmplx, bool):
        raise TypeError("The 2nd argument 'cmplx' must be a boolean.")

    lut = get_mapping_table(cmplx)
    return lut[ty]


maps_to_complex = partial(maps_to, cmplx=True)
maps_to_real = partial(maps_to, cmplx=False)


def update_dtype_mapping(argty, retty, cmplx=None):
    if not all(isinstance(ty, types.Type) for ty in (argty, retty)):
        raise TypeError("The first two arguments must Numba types.")

    if cmplx is None:
        cmplx = isinstance(retty, types.Complex)
    if not isinstance(cmplx, bool):
        raise TypeError("The 3rd argument 'cmplx' must be a boolean.")

    if cmplx:
        supported = (types.complex64, types.complex128)
    else:
        supported = (types.float32, types.float64)
    if retty not in supported:
        raise TypeError(
            f"Unsupported return type {retty}; must be one of {supported}.")

    lut = get_mapping_table(cmplx)
    lut[argty] = retty


update_dtype_mapping_complex = partial(update_dtype_mapping, cmplx=True)
update_dtype_mapping_real = partial(update_dtype_mapping, cmplx=False)
