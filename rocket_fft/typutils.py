import inspect
from functools import wraps

import numba as nb
from numba.core import types
from numba.core.errors import TypingError
from numba.np.numpy_support import is_nonelike


def is_sequence_like(arg):
    seq_like = (
        types.Tuple,
        types.ListType,
        types.Array,
        types.Sequence,
    )
    return isinstance(arg, seq_like)


def is_unicode(arg):
    return arg is types.unicode_type


def is_integer(arg):
    return isinstance(arg, (types.Integer, types.Boolean))


def is_scalar(arg):
    return isinstance(arg, (types.Number, types.Boolean))


def is_integer_2tuple(arg):
    if not isinstance(arg, types.UniTuple):
        return False
    return arg.count == 2 and is_integer(arg.dtype)


def is_literal_integer(val):
    def impl(arg):
        if not isinstance(arg, types.IntegerLiteral):
            return False
        return arg.literal_value == val

    return impl


def is_literal_bool(val):
    def impl(arg):
        if not isinstance(arg, types.BooleanLiteral):
            return False
        return arg.literal_value == val

    return impl


def is_literal_string(val):
    def impl(arg):
        if not isinstance(arg, types.StringLiteral):
            return False
        return arg.literal_value == val

    return impl


def is_contiguous_array(layout):
    def impl(arg):
        if not isinstance(arg, types.Array):
            return False
        return arg.layout == layout

    return impl


def is_not_nonelike(arg):
    return not is_nonelike(arg)


class TypeConstraint:
    def __init__(
        self,
        expected_type,
        allow_scalar=True,
        allow_sequence=False,
        allow_none=False,
        error_message=None,
    ):
        self.expected_type = expected_type
        self.allow_scalar = allow_scalar
        self.allow_sequence = allow_sequence
        self.allow_none = allow_none
        self.error_message = error_message

    def __call__(self, value, fmt=None):
        # Normalize to Numba type
        if not isinstance(value, types.Type):
            value = nb.typeof(value)

        if isinstance(value, types.Literal):
            value = value.literal_type

        if self.allow_none and is_nonelike(value):
            return True

        if self.allow_scalar and isinstance(value, self.expected_type):
            return True

        if self.allow_sequence and is_sequence_like(value):
            if isinstance(value.dtype, self.expected_type):
                return True

        if self.error_message is None:
            return False

        if fmt is None:
            raise TypingError(self.error_message)

        raise TypingError(self.error_message.format(*fmt))


class TypingValidator:
    def __init__(self, **constraints):
        self.constraints = constraints

    def __call__(self, **kwargs):
        for position, (name, value) in enumerate(kwargs.items(), start=1):
            constraint = self.constraints.get(name)
            if constraint is not None:
                ordinal = self._ordinal(position)
                constraint(value, fmt=(ordinal, name))
        return self

    def register(self, **constraints):
        self.constraints.update(constraints)
        return self

    def decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            self(**bound.arguments)
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def _ordinal(n):
        suffixes = ("th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th")
        return str(n) + suffixes[n % 10]
