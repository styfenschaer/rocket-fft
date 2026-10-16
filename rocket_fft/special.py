"""Special functions as Numba ufuncs, usable in Python and jitted code."""

from ._extras import loggamma, poch

__all__ = ["loggamma", "poch"]
