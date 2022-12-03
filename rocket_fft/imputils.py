import inspect
from enum import IntEnum, auto
from functools import wraps

from numba import TypingError
from numba.core import config
from numba.core.registry import CPUDispatcher
from numba.extending import overload


def otherwise(*args):
    return True


class Steps(IntEnum):
    ARGS_CHECK = auto()
    KWARGS_CHECK = auto()
    GENERIC_STEP = auto()


class _CustomDispatcherMixin:
    def impl(self, arg=None, **kwargs):
        if arg is not None:
            entry = (Steps.ARGS_CHECK, arg)
            self.steps.append(entry)
            return self

        items = tuple(kwargs.items())
        entry = (Steps.KWARGS_CHECK, items)
        self.steps.append(entry)
        return self

    def preproc(self, func):
        self.preprocs.append(func)
        return self

    def step(self, func):
        entry = (Steps.GENERIC_STEP, func)
        self.steps.append(entry)
        self.impls.append(None)
        return self

    @property
    def impl_func(self):
        @wraps(self.header)
        def impl_func_(*args):
            self.header(*args)
            for fn in self.preprocs:
                args = fn(*args)

            kwd = inspect.getcallargs(self.header, *args)
            for (kind, func), impl in zip(self.steps, self.impls):
                if kind == Steps.ARGS_CHECK:
                    if func(*args):
                        return impl
                elif kind == Steps.KWARGS_CHECK:
                    if all(val(kwd[key]) for key, val in func):
                        return impl
                elif kind == Steps.GENERIC_STEP:
                    args = func(*args)

            msg = 'No implementation found for function {}.'
            raise TypingError(msg.format(self.header.__name__))

        return impl_func_


class GeneratedDispatcher(_CustomDispatcherMixin, CPUDispatcher):
    def __init__(self, header, **kwargs):
        self.header = header
        self.steps = []
        self.impls = []
        self.preprocs = []
        self.options = kwargs
        super().__init__(header, **kwargs)

    def __call__(self, impl):
        self._update()
        self.impls.append(impl)
        return self

    def _update(self):
        dct = self.typingctx._globals
        for key, val in reversed(dct.items()):
            if val.dispatcher == self:
                break
        dct.pop(key)
        super().__init__(self.impl_func, **self.options)


def implements_jit(func=None, cache=False, pipeline_class=None, **options):
    dispatcher_args = {}
    if pipeline_class is not None:
        dispatcher_args['pipeline_class'] = pipeline_class

    def wrapper(func):
        if config.DISABLE_JIT:
            return func
        disp = GeneratedDispatcher(
            header=func, targetoptions=options,
            impl_kind='generated', **dispatcher_args)
        if cache:
            disp.enable_caching()
        return disp

    if func is not None:
        return wrapper(func)
    return wrapper


class OverloadDispatcher(_CustomDispatcherMixin):
    def __init__(self, overl, **kwargs):
        self.overl = overl
        self.header = None
        self.steps = []
        self.impls = []
        self.preprocs = []
        self.options = kwargs

    def __call__(self, impl):
        if self.header is None:
            self.header = impl
            return self

        self._update()
        self.impls.append(impl)
        return self

    def _update(self):
        overl = self.overl
        impl = self.impl_func
        overload(overl, **self.options)(impl)


def implements_overload(func, jit_options=None, strict=True, inline='never',
                        prefer_literal=False, **kwargs):
    if jit_options is None:
        jit_options = {}

    disp = OverloadDispatcher(
        func, jit_options=jit_options, strict=strict,
        inline=inline, prefer_literal=prefer_literal, **kwargs)
    return disp
