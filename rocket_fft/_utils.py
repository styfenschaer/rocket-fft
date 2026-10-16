"""Worker-thread configuration and optimal transform lengths.

All functions work both in pure Python and in jitted code. The default number
of workers lives in the native library and is read at run time, so
different to older Rocket-FFT versions ``set_workers`` also affects functions 
that are already compiled or cached.
"""

from __future__ import annotations

import operator

from . import _kernels, _native
from ._dispatch import implement


def get_workers() -> int:
    """Return the default number of worker threads."""
    return int(_native.get_default_workers())


def set_workers(workers: int) -> None:
    """Set the default number of worker threads.

    The default applies whenever ``workers`` is not passed explicitly, also to
    functions that were compiled before the call.
    """
    workers = operator.index(workers)
    cpu_count = int(_native.cpu_count())
    if not 1 <= workers <= cpu_count:
        raise ValueError(f"workers must be between 1 and the CPU count ({cpu_count}); got {workers}.")
    _native.set_default_workers(workers)


def good_size(target: int, real: bool = False) -> int:
    """Return the smallest fast FFT length that is greater than or equal to ``target``."""
    return _fast_len(_native.good_size, target, real)


def prev_good_size(target: int, real: bool = False) -> int:
    """Return the largest fast FFT length that is less than or equal to ``target``."""
    return _fast_len(_native.prev_good_size, target, real)


def _fast_len(fn, target, real) -> int:
    result = int(fn(operator.index(target), int(bool(real))))
    if result < 0:
        _native.raise_for_status(-result)
    return result


def _fast_len_stub(target, real=False): ...


implement(get_workers, get_workers, _kernels.default_workers)
implement(good_size, _fast_len_stub, _kernels.fast_len("rf_good_size"))
implement(prev_good_size, _fast_len_stub, _kernels.fast_len("rf_prev_good_size"))

next_fast_len = good_size
prev_fast_len = prev_good_size
