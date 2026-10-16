"""Registration of Python functions as Numba builtins backed by a kernel.

``@overload`` compiles a separate implementation function for every
signature, which roughly doubles the compile time of a cheap call. The
functions of this package lower directly to a single call into the native
library, so they are registered at a lower level instead: a typing template
plus a lowering that emits the kernel's code inline.

A *kernel* has the same shape as the body of an ``@intrinsic``: a function
``kernel(typingctx, *arg_types) -> (signature, codegen)``.
"""

from __future__ import annotations

import inspect
from typing import Callable, Sequence

from numba.core import types
from numba.core.imputils import lower_builtin
from numba.core.typing.templates import AbstractTemplate, fold_arguments, infer_global, signature

Kernel = Callable[..., tuple]


def _fold(pysig: inspect.Signature, args, kws) -> tuple:
    """Bind call arguments to ``pysig``; omitted arguments become ``types.Omitted``."""
    return tuple(
        fold_arguments(
            pysig,
            args,
            kws,
            lambda index, param, value: value,
            lambda index, param, default: types.Omitted(default),
            lambda index, param, values: types.StarArgTuple(values),
        )
    )


def implement(
    func: Callable,
    stub: Callable,
    kernel: Kernel,
    kernel_args: Sequence[str | None] | None = None,
    validate: Callable[..., None] | None = None,
) -> None:
    """Make ``func`` callable from jitted code.

    Parameters
    ----------
    func
        The Python function to implement, e.g. ``numpy.fft.fft``.
    stub
        A function whose signature (parameter names and defaults) defines how
        calls to ``func`` are bound.
    kernel
        Kernel that computes the result.
    kernel_args
        For each kernel argument, the name of the stub parameter that is passed
        to it, or ``None`` to pass ``None``. Defaults to all stub parameters in
        order.
    validate
        Called with the bound argument types; raises ``TypingError`` for
        invalid arguments.
    """
    pysig = inspect.signature(stub)
    names = list(pysig.parameters)
    if kernel_args is None:
        kernel_args = names
    index = [names.index(name) if name is not None else None for name in kernel_args]

    def kernel_types(arg_types: Sequence[types.Type]) -> list[types.Type]:
        return [arg_types[i] if i is not None else types.none for i in index]

    class Template(AbstractTemplate):
        key = func

        def generic(self, args, kws):
            bound = _fold(pysig, args, kws)
            if validate is not None:
                validate(*bound)
            kernel_sig, _ = kernel(self.context, *kernel_types(bound))
            return signature(kernel_sig.return_type, *bound).replace(pysig=pysig)

    Template.__name__ = f"Template({getattr(func, '__qualname__', func)})"
    infer_global(func, types.Function(Template))

    @lower_builtin(func, types.VarArg(types.Any))
    def lower(context, builder, sig, args):
        kernel_sig, codegen = kernel(context.typing_context, *kernel_types(sig.args))
        dummy = context.get_dummy_value()
        kernel_values = [args[i] if i is not None else dummy for i in index]
        return codegen(context, builder, kernel_sig, kernel_values)
