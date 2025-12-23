import inspect
from functools import wraps

from numba.core.errors import TypingError
from numba.extending import overload


def always_true(*args):
    return True


class ConditionalOverload:
    def __init__(self, signature=None, target=None, **options):
        self.signature = signature
        self.target = target

        self.conditions = []
        self.implementations = []
        self.preprocessors = []

        self.options = options
        if self.signature is not None:
            self._register_overload()

    def __call__(self, implementation):
        if self.signature is None:
            self.signature = implementation
            self._register_overload()
            return self

        self.implementations.append(implementation)
        return implementation

    def case(self, predicate=None, **kw_conditions):
        if predicate is not None:
            self.conditions.append(predicate)
            return self

        entry = tuple(kw_conditions.items())
        self.conditions.append(entry)
        return self

    def fallback(self, func):
        return self.case(always_true)(func)

    def preproc(self, func):
        self.preprocessors.append(func)
        return func

    @property
    def dispatcher(self):
        def dispatch(*args):
            # Typing gate
            self.signature(*args)

            # Preprocessing
            for fn in self.preprocessors:
                args = fn(*args)

            sig = inspect.signature(self.signature)
            bound = sig.bind(*args)
            bound.apply_defaults()
            arguments = bound.arguments

            for condition, implementation in zip(
                self.conditions,
                self.implementations,
            ):
                if callable(condition):
                    if condition(*args):
                        return implementation
                elif all(check(arguments[name]) for name, check in condition):
                    return implementation

            raise TypingError(
                "No implementation found for function "
                f"{self.signature.__name__} with arguments {args}."
            )

        return dispatch

    def _register_overload(self):
        target = self if self.target is None else self.target
        impl = wraps(self.signature)(self.dispatcher)
        overload(target, **self.options)(impl)


def implements_jit(
    func=None,
    jit_options=None,
    strict=True,
    inline="never",
    prefer_literal=True,
    **kwargs,
):
    if jit_options is None:
        jit_options = {}

    def wrapper(func):
        return ConditionalOverload(
            func,
            None,
            jit_options=jit_options,
            strict=strict,
            inline=inline,
            prefer_literal=prefer_literal,
            **kwargs,
        )

    if func is None:
        return wrapper

    return wrapper(func)


def implements_overload(
    func,
    jit_options=None,
    strict=True,
    inline="never",
    prefer_literal=True,
    **kwargs,
):
    if jit_options is None:
        jit_options = {}

    return ConditionalOverload(
        None,
        func,
        jit_options=jit_options,
        strict=strict,
        inline=inline,
        prefer_literal=prefer_literal,
        **kwargs,
    )
