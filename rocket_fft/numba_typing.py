from contextlib import contextmanager

import numba as nb
from numba import TypingError
from numba.core import types
from numba.np.numpy_support import is_nonelike


def is_sequence_like(arg):
    seq_like = (types.Array, types.Set, types.Tuple,
                types.ListType, types.Sequence)
    return isinstance(arg, seq_like)


def is_integer(arg):
    return isinstance(arg, types.Integer)


def is_integer_2tuple(arg):
    if not isinstance(arg, types.UniTuple):
        return False
    if not arg.count == 2:
        return False
    if not isinstance(arg, types.Integer):
        return False
    return True


def literal_integer(val):
    def inner(arg):
        if not isinstance(arg, types.IntegerLiteral):
            return False
        return arg.literal_value == val

    return inner


def literal_bool(val):
    def inner(arg):
        if not isinstance(arg, types.BooleanLiteral):
            return False
        return arg.literal_value == val

    return inner


def is_not_nonelike(arg):
    return not is_nonelike(arg)


class Check:
    def __init__(self, ty, as_one=True, as_seq=False,
                 allow_none=False, msg=None):
        self.ty = ty
        self.as_one = as_one
        self.as_seq = as_seq
        self.allow_none = allow_none
        self.msg = msg

    def __call__(self, arg, msg=None, fmt=None):
        if not isinstance(arg, types.Type):
            arg = nb.typeof(arg)
        if self.allow_none and is_nonelike(arg):
            return True
        if self.as_one and isinstance(arg, self.ty):
            return True
        if self.as_seq and is_sequence_like(arg):
            if isinstance(arg.dtype, self.ty):
                return True
        msg = self.msg if msg is None else msg
        if msg is None:
            return False
        raise TypingError(msg.format(fmt))


class TypingChecker:
    def __init__(self, **checks):
        self.checks = checks

    def __call__(self, **kwargs):
        items = kwargs.items()
        for pos, (key, val) in enumerate(items):
            check = self.checks.get(key)
            if check is not None:
                txt = self._int_to_ordinal(pos+1)
                check(val, fmt=txt)
        return self

    def register(self, **kwargs):
        for key, check in kwargs.items():
            self.checks[key] = check
        return self

    @staticmethod
    def _int_to_ordinal(n):
        # Adopted from https://codegolf.stackexchange.com/questions/4707
        suffix = "tsnrhtdd"[(n//10 % 10 != 1)*(n % 10 < 4)*n % 10::4]
        return f"{n}{suffix}"


@contextmanager
def typing_check(ty, as_one=True, as_seq=False, allow_none=False):
    check = Check(ty, as_one, as_seq, allow_none)
    yield check
