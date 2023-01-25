import numba as nb
from numba import TypingError
from numba.core import types
from numba.np.numpy_support import is_nonelike


def is_sequence_like(arg):
    seq_like = (types.Tuple, types.Array, types.Set,
                types.ListType, types.Sequence)
    return isinstance(arg, seq_like)


def is_integer(arg):
    return isinstance(arg, types.Integer)


def is_integer_2tuple(arg):
    return (isinstance(arg, types.UniTuple)
            and (arg.count == 2)
            and isinstance(arg.dtype, types.Integer))


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


def is_not_nonelike(arg):
    return not is_nonelike(arg)


class Check:
    def __init__(self, ty, as_one=True, as_seq=False, allow_none=False, msg=None):
        self.ty = ty
        self.as_one = as_one
        self.as_seq = as_seq
        self.allow_none = allow_none
        self.msg = msg

    def __call__(self, arg, fmt=None):
        if not isinstance(arg, types.Type):
            arg = nb.typeof(arg)
        if self.allow_none and is_nonelike(arg):
            return True
        if self.as_one and isinstance(arg, self.ty):
            return True
        if self.as_seq and is_sequence_like(arg):
            if isinstance(arg.dtype, self.ty):
                return True
        if self.msg is None:
            return False
        raise TypingError(self.msg.format(fmt))


def typing_check(ty, as_one=True, as_seq=False, allow_none=False):
    def impl(arg, msg):
        check = Check(ty, as_one, as_seq, allow_none, msg)
        return check(arg)

    return impl


class TypingChecker:
    def __init__(self, **checks):
        self.checks = checks

    def __call__(self, **kwargs):
        items = kwargs.items()
        for i, (key, val) in enumerate(items, start=1):
            check = self.checks.get(key)
            if check is not None:
                txt = self.get_ordinal(i)
                check(val, fmt=txt)
        return self

    def register(self, **kwargs):
        self.checks.update(kwargs)
        return self

    @staticmethod
    def get_ordinal(n):
        ordinals = ("th", "st", "nd", "rd", "th",
                    "th", "th", "th", "th", "th")
        return str(n) + ordinals[n % 10]
