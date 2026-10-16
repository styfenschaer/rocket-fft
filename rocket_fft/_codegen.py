"""LLVM lowering helpers shared by all kernels and intrinsics.

These helpers turn Numba values into the plain C arguments expected by the
native frontend and turn its status codes back into Python exceptions.
Argument types that are known at compile time (``None``, omitted defaults,
literals) are folded into constants, so they cost nothing at run time.
"""

from __future__ import annotations

from typing import Sequence

from llvmlite import ir
from numba.core import cgutils, types
from numba.cpython.listobj import ListInstance
from numba.np.arrayobj import make_array

from ._native import DTYPE_CODES, STATUS_TABLE, Norm

i32 = ir.IntType(32)
i64 = ir.IntType(64)
f64 = ir.DoubleType()
voidptr = ir.IntType(8).as_pointer()

# struct IntSeq { int64 count; const char* data; int64 stride; int64 dtype; }
INT_SEQ = ir.LiteralStructType([i64, voidptr, i64, i64])

# struct Options (see _rocket_fft_numba.cpp)
OPTIONS_FIELDS = (
    "kind", "flags", "in_dtype", "out_dtype", "norm",
    "fct", "r2r_type", "ortho", "workers", "workers_given",
)
OPTIONS = ir.LiteralStructType([f64 if name == "fct" else i64 for name in OPTIONS_FIELDS])

_INT_TYPES = (types.Integer, types.Boolean)
_NORM_NAMES = {"backward": Norm.BACKWARD, "ortho": Norm.ORTHO, "forward": Norm.FORWARD}


def is_none(ty) -> bool:
    """``None``, or an omitted argument whose default is ``None``.

    Unlike ``numba.np.numpy_support.is_nonelike``, which treats every omitted
    argument as ``None`` regardless of its default.
    """
    if isinstance(ty, types.Omitted):
        return ty.value is None
    return ty is None or isinstance(ty, types.NoneType)


def const(value: int) -> ir.Constant:
    return ir.Constant(i64, int(value))


def dtype_code(dtype: types.Type) -> int:
    return DTYPE_CODES[dtype]


def call(builder: ir.IRBuilder, name: str, ret: ir.Type, args: Sequence[ir.Value]) -> ir.Value:
    """Call a function of the native library; its prototype is taken from the arguments."""
    fnty = ir.FunctionType(ret, [a.type for a in args])
    fn = cgutils.get_or_insert_function(builder.module, fnty, name)
    return builder.call(fn, args)


def struct_ptr(builder: ir.IRBuilder, struct: ir.LiteralStructType, values: Sequence[ir.Value]) -> ir.Value:
    """Store ``values`` in a stack-allocated struct and return a pointer to it."""
    ptr = cgutils.alloca_once(builder, struct)
    for i, value in enumerate(values):
        builder.store(value, cgutils.gep_inbounds(builder, ptr, 0, i))
    return ptr


def options(builder: ir.IRBuilder, **fields: ir.Value | int | float) -> ir.Value:
    values = []
    for name, ty in zip(OPTIONS_FIELDS, OPTIONS.elements):
        value = fields.get(name, 0)
        values.append(value if isinstance(value, ir.Value) else ir.Constant(ty, value))
    return struct_ptr(builder, OPTIONS, values)


def array_ptr(context, builder: ir.IRBuilder, ty: types.Array, value: ir.Value) -> ir.Value:
    """Pointer to the in-memory struct of a Numba array."""
    return builder.bitcast(make_array(ty)(context, builder, value)._getpointer(), voidptr)


def like_input_layout(ty: types.Array) -> str:
    """Numba layout of ``np.empty_like(x)`` for an array ``x`` of type ``ty``.

    NumPy keeps the memory order of ``x``; ``A`` means it is only known at run time.
    """
    if ty.ndim < 2:
        return "C"
    return ty.layout if ty.layout in ("C", "F") else "A"


def combine_layouts(layouts) -> str:
    """Numba layout that covers every layout a result may have at run time."""
    layouts = set(layouts)
    return layouts.pop() if len(layouts) == 1 else "A"


def null() -> ir.Value:
    return ir.Constant(voidptr, None)


def call_returning_array(context, builder: ir.IRBuilder, name: str, ty: types.Array, args: Sequence[ir.Value]) -> ir.Value:
    """Call a native function that fills an array struct (its last argument) with a new reference.

    The status code it returns is checked; on success the array is returned.
    """
    slot = cgutils.alloca_once(builder, context.get_value_type(ty))
    status = call(builder, name, i64, [*args, builder.bitcast(slot, voidptr)])
    check_status(context, builder, status)
    return builder.load(slot)


# -----------------------------------------------------------------------------
# Scalars
# -----------------------------------------------------------------------------

def _omitted_value(ty: types.Type):
    return ty.value if isinstance(ty, types.Omitted) else None


def to_i64(builder: ir.IRBuilder, ty: types.Type, value: ir.Value) -> ir.Value:
    """Convert an integer or boolean (or omitted default) to int64."""
    if isinstance(ty, types.Omitted):
        return const(ty.value)
    if isinstance(ty, types.Boolean) or not ty.signed:
        return builder.zext(value, i64) if value.type.width < 64 else value
    return builder.sext(value, i64) if value.type.width < 64 else value


def to_f64(builder: ir.IRBuilder, ty: types.Type, value: ir.Value) -> ir.Value:
    """Convert a real number (or omitted default) to double."""
    if isinstance(ty, types.Omitted):
        return ir.Constant(f64, float(ty.value))
    if isinstance(ty, types.Boolean):
        return builder.uitofp(value, f64)
    if isinstance(ty, types.Integer):
        return builder.sitofp(value, f64) if ty.signed else builder.uitofp(value, f64)
    return builder.fpext(value, f64) if value.type != f64 else value


def to_c128(context, builder: ir.IRBuilder, ty: types.Type, value: ir.Value) -> tuple[ir.Value, ir.Value]:
    """Real and imaginary part of a number (or omitted default) as doubles."""
    if isinstance(ty, types.Complex):
        z = context.make_complex(builder, ty, value)
        return to_f64(builder, ty.underlying_float, z.real), to_f64(builder, ty.underlying_float, z.imag)
    if isinstance(ty, types.Omitted) and isinstance(ty.value, complex):
        return ir.Constant(f64, ty.value.real), ir.Constant(f64, ty.value.imag)
    return to_f64(builder, ty, value), ir.Constant(f64, 0.0)


def optional_flag(builder: ir.IRBuilder, ty: types.Type, value: ir.Value) -> ir.Value:
    """Tri-state boolean: -1 for None, otherwise 0 or 1."""
    if is_none(ty):
        return const(-1)
    return to_i64(builder, ty, value)


def workers(builder: ir.IRBuilder, ty: types.Type, value: ir.Value) -> tuple[ir.Value, ir.Value]:
    """The ``(workers, workers_given)`` pair; ``None`` selects the global default."""
    if is_none(ty):
        return const(0), const(0)
    return to_i64(builder, ty, value), const(1)


def norm(context, builder: ir.IRBuilder, ty: types.Type, value: ir.Value) -> ir.Value:
    """Norm code of a ``norm`` argument; invalid strings yield ``Norm.INVALID``."""
    if is_none(ty):
        return const(Norm.BACKWARD)
    literal = _omitted_value(ty)
    if isinstance(ty, types.StringLiteral):
        literal = ty.literal_value
    if literal is not None:
        return const(_NORM_NAMES.get(literal, Norm.INVALID))
    string = context.make_helper(builder, types.unicode_type, value)
    return call(builder, "rf_parse_norm", i64, [string.data, string.length, string.kind])


def check_device(context, builder: ir.IRBuilder, value: ir.Value) -> None:
    """Raise unless the (unicode) ``device`` argument is ``'cpu'``."""
    string = context.make_helper(builder, types.unicode_type, value)
    check_status(context, builder, call(builder, "rf_check_device", i64, [string.data, string.length, string.kind]))


# -----------------------------------------------------------------------------
# Integer sequences (shape and axes arguments)
# -----------------------------------------------------------------------------

def is_int_like(ty: types.Type) -> bool:
    """Integer or boolean, including an omitted argument with such a default."""
    if isinstance(ty, types.Omitted):
        return isinstance(ty.value, (int, bool))
    return isinstance(ty, _INT_TYPES)


def _const_ints(builder: ir.IRBuilder, values: Sequence[ir.Value]) -> ir.Value:
    buf = cgutils.alloca_once(builder, i64, size=max(len(values), 1))
    for i, value in enumerate(values):
        builder.store(value, builder.gep(buf, [ir.Constant(i32, i)]))
    return buf


def int_seq(context, builder: ir.IRBuilder, ty: types.Type, value: ir.Value) -> ir.Value:
    """Lower ``None``, an integer, a tuple, a list or an array to an ``IntSeq``."""
    if is_none(ty):
        return struct_ptr(builder, INT_SEQ, [const(-1), ir.Constant(voidptr, None), const(0), const(0)])

    if isinstance(ty, types.Omitted):
        raw = ty.value if isinstance(ty.value, tuple) else (ty.value,)
        values = [const(v) for v in raw]
    elif is_int_like(ty):
        values = [to_i64(builder, ty, value)]
    elif isinstance(ty, types.BaseTuple):
        values = [to_i64(builder, t, builder.extract_value(value, i)) for i, t in enumerate(ty)]
    else:
        if isinstance(ty, types.Array):
            ary = make_array(ty)(context, builder, value)
            count = builder.extract_value(ary.shape, 0)
            data = ary.data
            stride = builder.extract_value(ary.strides, 0)
        elif isinstance(ty, types.List):
            lst = ListInstance(context, builder, ty, value)
            count = lst.size
            data = lst.data
            stride = const(context.get_abi_sizeof(context.get_data_type(ty.dtype)))
        else:
            raise TypeError(f"unsupported integer sequence type {ty}")
        dtype = const(dtype_code(ty.dtype))
        return struct_ptr(builder, INT_SEQ, [count, builder.bitcast(data, voidptr), stride, dtype])

    buf = _const_ints(builder, values)
    return struct_ptr(
        builder,
        INT_SEQ,
        [const(len(values)), builder.bitcast(buf, voidptr), const(8), const(dtype_code(types.int64))],
    )


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------

def check_status(context, builder: ir.IRBuilder, status: ir.Value) -> None:
    """Raise the exception corresponding to a non-zero status code.

    The native library has already released anything it allocated when it
    returns an error, so there is nothing to clean up here.
    """
    failed = builder.icmp_signed("!=", status, const(0))
    with cgutils.if_unlikely(builder, failed):
        unknown = builder.append_basic_block("rf.unknown_status")
        switch = builder.switch(status, unknown)
        for code, (exc, msg) in STATUS_TABLE.items():
            block = builder.append_basic_block(f"rf.status_{code}")
            switch.add_case(const(code), block)
            with builder.goto_block(block):
                context.call_conv.return_user_exc(builder, exc, (msg,))
        with builder.goto_block(unknown):
            context.call_conv.return_user_exc(builder, RuntimeError, ("unknown rocket-fft status",))
