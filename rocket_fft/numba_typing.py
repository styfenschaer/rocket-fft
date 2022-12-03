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
    if not is_integer(arg.dtype):
        return False
    return True


def literal_is_true(arg):
    if not hasattr(arg, 'literal_value'):
        raise TypingError('Argument must be literal value.')
    return arg.literal_value


def literal_is_false(arg):
    return not literal_is_true(arg)


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
        # It's not a Numba type so make it one
        # TODO: Is there a better way to check this?
        if not hasattr(arg, 'cast_python_value'):
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
        self.argpos = 0
        self.checks = checks

    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            check = self.checks.get(key)
            if check is not None:
                i = self.argpos
                pos = self._pos_to_text(i)
                check(val, fmt=pos)
            self.argpos += 1
        return self

    def reset(self):
        self.argpos = 0
        return self

    def register(self, **kwargs):
        for key, check in kwargs.items():
            self.checks[key] = check
        return self

    @staticmethod
    def _pos_to_text(pos):
        lut = {0: '1st', 1: '2nd', 2: '3rd'}
        pos = lut.get(pos)
        if pos is None:
            pos = str(pos) + 'th'
        return pos


@contextmanager
def typing_check(ty, as_one=True, as_seq=False, allow_none=False):
    check = Check(ty, as_one, as_seq, allow_none)
    yield check
