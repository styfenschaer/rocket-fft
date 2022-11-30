import inspect
from functools import wraps

from numba import TypingError, generated_jit


def otherwise(*args):
    return True


class implements_jit:
    def __init__(self, func=None, **kwargs):
        self.func = func
        self.checks = []
        self.impls = []
        self.preprocs = []
        self.options = kwargs

    def __call__(self, impl):
        if self.func is None:
            self.func = impl
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

    def generate(self, **kwargs):
        @wraps(self.func)
        def impl(*args):
            self.func(*args)
            for fn in self.preprocs:
                args = fn(*args)
            kwd = inspect.getcallargs(self.func, *args)
            for check, impl in zip(self.checks, self.impls):
                if not isinstance(check, tuple):
                    if check(*args):
                        return impl
                    continue
                if all(val(kwd[key]) for key, val in check):
                    return impl
            msg = 'No implementation found for function {}.'
            raise TypingError(msg.format(self.func.__name__))

        options = self.options.copy()
        options.update(kwargs)
        return generated_jit(**options)(impl)
