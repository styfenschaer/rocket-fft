import inspect
from functools import wraps

from numba import TypingError
from numba.extending import overload


def otherwise(*args):
    return True


class Overloader:
    def __init__(self, header=None, overl=None, **options):
        self.header = header
        self.overl = overl
        self.checks = []
        self.impls = []
        self.preprocs = []
        self.options = options
        if self.header is not None:
            self._overload()

    def __call__(self, impl):
        if self.header is None:
            self.header = impl
            self._overload()
            return self

        self.impls.append(impl)
        return self

    def impl(self, arg=None, **kwargs):
        if arg is not None:
            self.checks.append(arg)
            return self

        entry = tuple(kwargs.items())
        self.checks.append(entry)
        return self

    def preproc(self, func):
        self.preprocs.append(func)
        return self

    @property
    def impl_func(self):
        def impl_func_(*args):
            self.header(*args)
            for fn in self.preprocs:
                args = fn(*args)

            kwd = inspect.getcallargs(self.header, *args)
            for check, impl in zip(self.checks, self.impls):
                if not isinstance(check, tuple):
                    if check(*args):
                        return impl
                else:
                    if all(fn(kwd[kw]) for kw, fn in check):
                        return impl

            raise TypingError("No implementation found for function {} with "
                              "arguments {}.".format(self.header.__name__, args))

        return impl_func_

    def _overload(self):
        overl = self if self.overl is None else self.overl
        impl_func = wraps(self.header)(self.impl_func)
        overload(overl, **self.options)(impl_func)


def implements_jit(func=None, jit_options=None, strict=True,
                   inline="never", prefer_literal=False, **kwargs):
    if jit_options is None:
        jit_options = {}

    def wrapper(func):
        overloader = Overloader(
            func, None, jit_options=jit_options, strict=strict,
            inline=inline, prefer_literal=prefer_literal, **kwargs)
        return overloader

    if func is not None:
        return wrapper(func)
    return wrapper


def implements_overload(overloaded_func, jit_options=None, strict=True,
                        inline="never", prefer_literal=False, **kwargs):
    if jit_options is None:
        jit_options = {}

    overloader = Overloader(
        None, overloaded_func, jit_options=jit_options, strict=strict,
        inline=inline, prefer_literal=prefer_literal, **kwargs)
    return overloader
