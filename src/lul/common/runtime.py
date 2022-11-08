from __future__ import annotations

import dataclasses
import operator
import sys
import os
import inspect
import numbers
from typing_extensions import * # type: ignore
# if not TYPE_CHECKING:
#     from typing_extensions import ParamSpec, ParamSpecArgs, ParamSpecKwargs
from typing import *
import types

# if not TYPE_CHECKING:
#     List = List
#     Tuple = Tuple
# else:
#     List = list
#     Tuple = tuple

import reprlib
import re
import io
import keyword
import functools
import builtins as py
import collections.abc as std
import abc # type: ignore
from abc import *
import collections # type: ignore
import contextvars as CV
import contextlib
# from types import NoneType

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")

P2 = ParamSpec("P2")
R2 = TypeVar("R2")
T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
T6 = TypeVar("T6")
T7 = TypeVar("T7")
T8 = TypeVar("T8")
T9 = TypeVar("T9")



nil = None
t = True

# noinspection PyTypeChecker
unset: None = ["%unset"]

def unbound():
    return unset

def unboundp(x):
    return x is unset

assert unboundp(unbound()) is True
assert unbound() is unset

def either(x: T, y: U, *, unset=unbound()) -> Union[T, U]:
    return y if x is unset else y

assert either(unbound(), 21) == 21
assert either(unbound(), unset) is unset

def init(thunk: Callable[[], T], val: Optional[T], *, unset=unbound()) -> T:
    return thunk() if val is unset else val

assert init(lambda: 21, unset) == 21
assert init(lambda: 21, None) is None
# assert init(lambda: cast(Cons[int, str], 21), None) is None

@contextlib.contextmanager
def CV_let(var: CV.ContextVar[T], val: T):
    token = var.set(val)
    try:
        yield var
    finally:
        var.reset(token)

indent_level = CV.ContextVar[int]("indent_level", default=0)

def indentation() -> str:
    return " " * indent_level.get()

def with_indent(n=2):
    return CV_let(indent_level, indent_level.get() + n)

def repr_get(kvs, k, *default):
    if isinstance(kvs, std.Mapping):
        return kvs.get(k, *default)
    else:
        return getattr(kvs, k, *default)

def repr_fields(self: T, *kvs: Tuple[str, Optional[Callable[[str, T], str]]]):
    xs = []
    with with_indent():
        ind = indentation()
        for kv in kvs:
            k, v = kv
            if v is None:
                v = lambda k, self: (prrepr(it) if (it := repr_get(self, k, None)) is not None else None)
            x = v(k, self)
            if x is not None:
                xs.append(f"\n{ind}{k}={x}")
    return ",".join(xs)

def repr_self(self: T, *kvs: Tuple[str, Optional[Callable[[str, T], str]]], name=unset):
    if name is unset:
        name = nameof(py.type(self))
    return name + "(" + repr_fields(self, *kvs) + ")"

def repr_call(f, **args):
    return repr_self(args, *((k, None) for k in args.keys()), name=nameof(f))

@functools.singledispatch
def nameof(x, unknown="<unknown>"):
    if isinstance(x, str) and x.isprintable():
        return x
    return getattr(x, "__qualname__", getattr(x, "__name__", unknown)).replace(".<locals>", "").replace(".<lambda>", ".<fn>")

def stringp(x):
    return isinstance(x, str)

def vectorp(x):
    return isinstance(x, (list, tuple))

def SDATA(x):
    assert stringp(x)
    return x

class Error(Exception):
    pass

class CircularIteration(Error):
    pass

def error(msg, value=None):
    if value is not None:
        msg = msg % value
    raise Error(msg)

Lisp_Object = object
TLisp = TypeVar("TLisp", bound=Lisp_Object)
if TYPE_CHECKING:
    from . import buffer

NoneType = type(None)

# enum symbol_interned
# {
#   SYMBOL_UNINTERNED = 0,
#   SYMBOL_INTERNED = 1,
#   SYMBOL_INTERNED_IN_INITIAL_OBARRAY = 2
# };
import enum

class symbol_interned(enum.IntEnum):
    SYMBOL_UNINTERNED = 0
    SYMBOL_INTERNED = 1
    SYMBOL_INTERNED_IN_INITIAL_OBARRAY = 2

SYMBOL_UNINTERNED = symbol_interned.SYMBOL_UNINTERNED
SYMBOL_INTERNED = symbol_interned.SYMBOL_INTERNED
SYMBOL_INTERNED_IN_INITIAL_OBARRAY = symbol_interned.SYMBOL_INTERNED_IN_INITIAL_OBARRAY

class symbol_redirect(enum.IntEnum):
    SYMBOL_PLAINVAL = 4
    SYMBOL_VARALIAS = 1
    SYMBOL_LOCALIZED = 2
    SYMBOL_FORWARDED = 3

SYMBOL_PLAINVAL = symbol_redirect.SYMBOL_PLAINVAL
SYMBOL_VARALIAS = symbol_redirect.SYMBOL_VARALIAS
SYMBOL_LOCALIZED = symbol_redirect.SYMBOL_LOCALIZED
SYMBOL_FORWARDED = symbol_redirect.SYMBOL_FORWARDED

class symbol_trapped_write(enum.IntEnum):
    SYMBOL_UNTRAPPED_WRITE = 0
    SYMBOL_NOWRITE = 1
    SYMBOL_TRAPPED_WRITE = 2

SYMBOL_UNTRAPPED_WRITE = symbol_trapped_write.SYMBOL_UNTRAPPED_WRITE
SYMBOL_NOWRITE = symbol_trapped_write.SYMBOL_NOWRITE
SYMBOL_TRAPPED_WRITE = symbol_trapped_write.SYMBOL_TRAPPED_WRITE

@dataclasses.dataclass
class LispSymbol(Generic[TLisp]):
    # /* The symbol's name, as a Lisp string.  */
    name: str

    # /* Indicates where the value can be found:
    # 	 0 : it's a plain var, the value is in the `value' field.
    # 	 1 : it's a varalias, the value is really in the `alias' symbol.
    # 	 2 : it's a localized var, the value is in the `blv' object.
    # 	 3 : it's a forwarding variable, the value is in `forward'.  */
    redirect: symbol_redirect = SYMBOL_PLAINVAL

    # /* 0 : normal case, just set the value
    # 	 1 : constant, cannot set, e.g. nil, t, :keywords.
    # 	 2 : trap the write, call watcher functions.  */
    trapped_write: symbol_trapped_write = SYMBOL_UNTRAPPED_WRITE

    # /* Interned state of the symbol.  This is an enumerator from
    # 	 enum symbol_interned.  */
    interned: symbol_interned = SYMBOL_UNINTERNED

    # /* True means that this variable has been explicitly declared
    # 	 special (with `defvar' etc), and shouldn't be lexically bound.  */
    declared_special: bool = False

    # /* Value of the symbol or Q.unbound if unbound.  Which alternative of the
    # 	 union is used depends on the `redirect' field above.  */
    @dataclasses.dataclass
    class _val(Generic[TLisp]):
        value: CV.ContextVar[TLisp] = None
        alias: LispSymbol = None
        # blv: Lisp_Buffer_Local_Value[TLisp] = None
        # fwd: lispfwd = None
    val: _val[TLisp] = dataclasses.field(default_factory=_val)

    # function: Lisp_Object = dataclasses.field(default_factory=lambda: Q.nil)
    function: Lisp_Object = None

    # plist: Lisp_Object = dataclasses.field(default_factory=lambda: Q.nil)
    plist: Lisp_Object = None

    # def __init__(self, name: str, **kws):
    #     super().__init__()
    #     self.name = name
    #     if len(kws) > 1:
    #         raise ValueError(f"Expected only one keyword, got {kws!r}")
    #     self.val = self._val(**kws)

    def __repr__(self):
        return self.name

    @property
    def u(self):
        return self

    @property
    def s(self):
        return self

def SYMBOL_NAME(sym: LispSymbol):
    return sym.name

def SREF(string: str, index: int):
    return SDATA(string)[index]

class QMeta(type):
    def __getattribute__(self, id):
        if not id.startswith('_'):
            name = uncompile_id(id)
            if id not in self.__dict__:
                super().__setattr__(id, LispSymbol(name))
        return super().__getattribute__(id)

class Q(metaclass=QMeta):
    t: LispSymbol[bool] = LispSymbol("t")
    nil: LispSymbol[NoneType] = LispSymbol("nil")
    unbound: LispSymbol[NoneType] = LispSymbol("unbound")

class V:
    obarray: List[LispSymbol]

def builtin_lisp_symbols():
    for k, v in Q.__dict__.items():
        if isinstance(v, LispSymbol):
            yield v

NULL = None

def make_fixnum(i):
    return i

OBARRAY_SIZE = 15121

def make_vector(length: int, init: Lisp_Object):
    return [init] * length


# /* Placeholder for make-docfile to process.  The actual symbol
#    definition is done by lread.c's define_symbol.  */
def DEFSYM(sym: Lisp_Object, name: str):
    assert XSYMBOL(sym).name == name

def init_obarray_once():
    V.obarray = make_vector(OBARRAY_SIZE, make_fixnum(0))
    global initial_obarray
    initial_obarray = V.obarray
    # staticpro( & initial_obarray);

    #   for (int i = 0; i < ARRAYELTS (lispsym); i++)
    #     define_symbol (builtin_lisp_symbol (i), defsym_name[i]);
    for sym in builtin_lisp_symbols():
        define_symbol(sym, sym.name)

    DEFSYM(Q.unbound, "unbound")
    DEFSYM(Q.nil, "nil")
    # SET_SYMBOL_VAL(XSYMBOL(Q.nil), Q.nil)
    SET_SYMBOL_VAL(XSYMBOL(Q.nil), None)
    make_symbol_constant(Q.nil)
    XSYMBOL(Q.nil).u.s.declared_special = True

    DEFSYM(Q.t, "t");
    # SET_SYMBOL_VAL(XSYMBOL(Q.t), Q.t)
    SET_SYMBOL_VAL(XSYMBOL(Q.t), True)
    make_symbol_constant(Q.t)
    XSYMBOL(Q.t).u.s.declared_special = True

    #   /* Qt is correct even if not dumping.  loadup.el will set to nil at end.  */
    #   Vpurify_flag = Qt;
    V.purify_flag = Q.t

    DEFSYM(Q.variable_documentation, "variable-documentation")



def set_symbol_name(sym: Lisp_Object, name: str):
    XSYMBOL(sym).u.s.name = name


def set_symbol_function(sym: Lisp_Object, function: Lisp_Object):
    XSYMBOL(sym).u.s.function = function

def set_symbol_plist(sym: Lisp_Object, plist: Lisp_Object):
    XSYMBOL(sym).u.s.plist = plist

def set_symbol_next(sym: Lisp_Object, next: Lisp_Object):
    #XSYMBOL(sym).u.s.next = next
    pass

def init_symbol(val: Lisp_Object, name: str):
    p = XSYMBOL(val)
    set_symbol_name(val, name)
    set_symbol_plist(val, Q.nil)
    p.u.s.redirect = SYMBOL_PLAINVAL
    SET_SYMBOL_VAL(p, Q.unbound)
    set_symbol_function(val, Q.nil)
    set_symbol_next(val, NULL)
    p.u.s.gcmarkbit = False
    p.u.s.interned = SYMBOL_UNINTERNED
    p.u.s.trapped_write = SYMBOL_UNTRAPPED_WRITE
    p.u.s.declared_special = False
    p.u.s.pinned = False

def define_symbol(sym: LispSymbol, name: str):
    init_symbol(sym, name)
    #   /* Qunbound is uninterned, so that it's not confused with any symbol
    #      'unbound' created by a Lisp program.  */
    if not EQ(sym, Q.unbound):
        intern_sym(sym, initial_obarray)

initial_obarray: List[LispSymbol]

def EQ(x, y):
    return x is y

def XSYMBOL(sym) -> LispSymbol:
    assert isinstance(sym, LispSymbol)
    return sym

def intern_sym(sym: LispSymbol, obarray: Lisp_Object):
    sym.interned = SYMBOL_INTERNED_IN_INITIAL_OBARRAY if EQ(obarray, initial_obarray) else SYMBOL_INTERNED

    if SREF(SYMBOL_NAME(sym), 0) == ':' and EQ(obarray, initial_obarray):
        make_symbol_constant(sym)
        XSYMBOL(sym).u.s.redirect = SYMBOL_PLAINVAL
        # /* Mark keywords as special.  This makes (let ((:key 'foo)) ...)
        # 	 in lexically bound elisp signal an error, as documented.  */
        XSYMBOL(sym).u.s.declared_special = True
        SET_SYMBOL_VAL(XSYMBOL(sym), sym)

    #   ptr = aref_addr (obarray, XFIXNUM (index));
    #   set_symbol_next (sym, SYMBOLP (*ptr) ? XSYMBOL (*ptr) : NULL);
    #   *ptr = sym;
    return sym

def make_symbol_constant(sym: Lisp_Object):
    XSYMBOL(sym).u.s.trapped_write = SYMBOL_NOWRITE

def SET_SYMBOL_VAL(sym: LispSymbol, v):
    assert sym.u.s.redirect == SYMBOL_PLAINVAL
    sym.u.s.val.value = v

def intern(name: str, obarray=None):
    if obarray is None:
        obarray = initial_obarray
    for sym in obarray:
        if sym.name == name:
            return sym
    sym = LispSymbol(name=name)
    obarray.insert(0, sym)
    return sym

# Q.current_buffer = intern("current-buffer")
# Q.current_indentation = intern("current-indentation")

def lul_init():
    init_obarray_once()




def compiled_prefix():
    return "LISP_"

def compile_char(c: str):
    if c == "-":
        return "_"
    return f'_0{ord(c):02d}'

def uncompile_char(c: str):
    if m := re.fullmatch("[_]([0][0-9][0-9]+)", c):
        it = m.groups()[0]
        code = int(it)
        return chr(code)
    if c == "_":
        return "-"
    else:
        return c

def compile_id_hook(id: str, name: str):
    if not id.isidentifier():
        return ["_", id]
    if id.startswith("_") and not name.startswith("-"):
        return [compiled_prefix(), id]
    if keyword.iskeyword(id):
        return [id, "_"]

def compile_id(name: str):
    id = name
    if id.endswith("?"):
        id = id[:-1] + ("-p" if "-" in name else "p")
    id = ''.join([compile_char(c) if not (c.isalpha() or c == "-") else c for c in id])
    id = id.replace('-', '_')
    while it := compile_id_hook(id, name):
        id = ''.join(it)
    return id

def uncompile_id(id: str):
    if id.endswith("_p"):
        id = id[:-2] + "?"
    elif id.endswith("p"):
        id = id[:-1] + "?"
    if id.startswith(compiled_prefix()):
        id = id[len(compiled_prefix()):]
    def replace(m: re.Match):
        return uncompile_char(m.groups()[0])
    id = re.sub("([_](?:[0][0-9][0-9]+)?)", replace, id)
    return id

@functools.singledispatch
def functionp(x):
    return inspect.isfunction(x)

@functools.singledispatch
def null(x):
    return x is None

@functools.singledispatch
def truthy(x):
    if isinstance(x, bool):
        return x
    elif isinstance(x, numbers.Number):
        return True
    elif isinstance(x, std.Sized):
        return True
    else:
        return bool(x)

def falsep(x):
    return not truthy(x)

def ok(x):
    return not null(x)

def no(x):
    return null(x) or falsep(x)

def yes(x):
    return not no(x)

@functools.singledispatch
def iterate(l):
    return [(k, v) for k, v in py.enumerate(l)]

@iterate.register
def iterate_dict(l: std.MutableMapping):
    return [(k, v) for k, v in l.items()]

@functools.singledispatch
def keys(x):
    return [k for k, v in iterate(x)]

@functools.singledispatch
def values(x):
    return [v for k, v in iterate(x)]

# # HookFn = Callable[P, R]
# HookResult = TypeVar("HookResult")
# HookArgs = ParamSpec("HookArgs")
# HookFn = Callable[HookArgs, HookResult]
# HookFns = List[HookFn]
# HookType = List[HookFn]
#
# # def hookfns(hook: HookType) -> HookFns:
#     # if functionp(hook):
#     #     hook = [hook]
#     # return values(hook)
# # def hookfns(hook: Iterable[Callable[P2, R2]]) -> Iterator[Callable[P2, R2]]:
# #     # return [f for f in hook]
# #     return iter(hook)
# # def hookfns(fns: Sequence[Callable[P2, R2]]) -> List[Callable[P2, R2]]:
# # def hookfn
#     # for f in fns:
#     #     yield f
#     # return list(fns)
# hookfns = py.list
#
#
#
# # def call_hook_with_args(hook: HookType, *args: P.args, **kws: P.kwargs) -> R:
# def call_hook_with_args(hook: HookType, *args: HookArgs.args, **kws: HookArgs.kwargs) -> HookResult:
#     for f in hookfns(hook):
#         yield f(*args, **kws)
#
# # def run_hook_with_args(hook: HookType, *args: P.args, **kws: P.kwargs) -> R:
# # def run_hook_with_args(hook: HookType, *args: HookArgs.args, **kws: HookArgs.kwargs) -> HookResult:
# def run_hook_with_args(hook: List[Callable[P0, R0]], *args: P0.args, **kws: P0.kwargs) -> R0:
#     x: R0 = None
#     for f in hook:
#         reveal_type(f)
#         x = f(*args, **kws)
#     # for x in call_hook_with_args(hook, *args, **kws):
#     #     pass
#     return x
#
# if TYPE_CHECKING:
#     def run_hook_with_args(hook: List[Callable[P0, R0]], args: P0.args, **kws: P0.kwargs) -> R0:
#         ...
#
# # def run_hook_with_args_until_failure(hook: HookType, *args: P.args, **kws: P.kwargs) -> R:
# # if no(x := f(*args, **kws)):
# #     return x
# def run_hook_with_args_until_failure(hook: HookType, *args: P.args, **kws: P.kwargs) -> HookResult:
#     for x in call_hook_with_args(hook, *args, **kws):
#         if no(x):
#             return x
#
# # def run_hook_with_args_until_success(hook: HookType, *args: P.args, **kws: P.kwargs) -> R:
# def run_hook_with_args_until_success(hook: HookType, *args: P.args, **kws: P.kwargs) -> HookResult:
#     for x in call_hook_with_args(hook, *args, **kws):
#         if yes(x):
#             return x
#     # for f in hookfns(hook):
#     #     if yes(x := f(*args, **kws)):
#     #         return x
#
# P_prn = ParamSpec("P_prn")
#
# # def print(self, *args, sep=' ', end='\n', file=None): # known special case of print
# #     pass
#
# def prn(x: T, y, z, *args, **kws) -> T:
#     print(x, y, z, *args, **kws)
#     return x
#
# class Prn:
#     def __call__(self, *args, sep=' ', end='\n', file=None, flush=False):
#         py.print(*args, sep=sep, end=end, file=file, flush=flush)
#
# print = Prn()
# # prn: print = print
# reveal_type(print)


# if TYPE_CHECKING:
#     prn = py.print
#     # class Prn(Generic[P, R]):
#     #     def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
#     #         return print(*args, **kwargs)
#     #     # def __class_getitem__(cls, item: Callable[P, R]) -> Callable[P, R]:
#     #     #     return item
#     #     def __class_getitem__(cls, item: R) -> R:
#     #         # return Prn[item]
#     #         return item
#     #     @classmethod
#     #     def get(cls, f: Callable[P, R]) -> Callable[P, R]:
#     #         return f
#     #     # return item
#     #
#     # zz = Prn[HookArgs, HookResult]
#     # reveal_type(Prn.get(py.print))
#     # Prn[print](1,2,3, file=42)

#
#
# reveal_type(prn)
#
# def to_int(x: str) -> int:
#     return int(x)
#
# def lst(*args: T):
#     reveal_type(args)
#     # return py.list(*args)
#     return [x for x in args]
#
# # h = [prn, prn, lambda x, *args: x]
# # h = [lst, str, to_int, py.list, int]
# h = [lst]
# h = [to_int]
# h = [print]
# # h = [*h, int]
# reveal_type(h)
#
# P0 = ParamSpec("P0")
# R0 = TypeVar("R0")
# P01 = ParamSpec("P01")
# R01 = TypeVar("R01")
#
# P0_args = TypeVar("P0_args", bound=P0.args)
# P0_kws = TypeVar("P0_kws", bound=P0.kwargs)
#
# def run_hook(h: List[Callable[P0, R0]], *args: P0.args, **kws: P0.kwargs) -> R0:
#     x = None
#     for f in h:
#         x = f(*args, **kws)
#     return x
#
# # for f in hookfns(h):
# # for f in h:
# for f in hookfns(h):
#     f(1, file=io.StringIO, omgz=42)
#     f(1)
#     f("ok")
# h0 = h[0]
# reveal_type(h0)
# reveal_type(run_hook([to_int], 1, file=io.StringIO))
# run_hook([to_int], "ok", file=io.StringIO)
# h2 = [prn]
# assert run_hook_with_args([to_int], 1, 2, 3) == 1
# assert run_hook_with_args([to_int], "ok", 2, 3) == 1
# assert run_hook_with_args([prn], "ok", 2, 3) == 1
# assert run_hook([print], "ok", 2, 3, file=io.StringIO) == 1
#
# def hook2(f: Callable[P, R], f2: Callable[P0, R0]) -> Union[Callable[P, R], Callable[P0, R0]]:
#     def wrapper(*args, **kws):
#         f(*args, **kws)
#         return f2(*args, **kws)
#     return wrapper
# # reveal_type(functools.reduce)
# functools.reduce(hook2, [to_int, print])
#
# Z0 = TypeVar("Z0")
# Z1 = TypeVar("Z1")
# Z01 = TypeVar("Z01", Z0, Z1, Union[Z0, Z1])
#
# class Join(Callable[[Z0, Z01], Union[Z0, Z01]], Generic[Z0, Z01]):
#     def __class_getitem__(cls, item: Tuple[Callable[P, R], Callable[P0, R0]]) -> Union[Callable[P, R], Callable[P0, R0]]:
#         ...
#     # def __call__(self, x: Z0, ys: Z01):
#     pass
#
# qq: TypeAlias = Join[to_int, print]
# reveal_type(qq)
#
#
#
# # def join(x: T, y: U) -> List[Union[T, U]]:
# def join(x: T, y: U) -> Tuple[T, U]:
#     return x, y
#
# XY = Tuple[T, U, ...]
#
# def hd(x: Tuple[T, ...]) -> T:
#     return x[0]
#
# def at(x: Sequence[T], i: SupportsIndex) -> T:
#     return x[i]
#
# reveal_type(hd(join(type(None), operator.getitem)))
# # join(1, "foo")
# zz = at(join(type(None), operator.getitem), 0)
# reveal_type(zz)
# zz(dict(a="ok"), "yes", 21)
#
# def make_hook(xs: Tuple[Callable[P, R], Callable[P0, R0]]):
#     return hook2(xs[0], xs[1])
#
#
# hook2(to_int, print)(None)
# make_hook((to_int, print))(1)
# make_hook((to_int, print))("ok")
# make_hook((to_int, print))("ok", None, foo=42)
# make_hook((print,))("ok", None, foo=42)
# make_hook(tuple([print, to_int]))("ok", None, foo=42)
#
# def runner(hook: Sequence[Callable[P, R]]) -> Callable[P, R]:
#     def wrapper(*args: P.args, **kws: P.kwargs) -> R:
#         x = None
#         for f in hook:
#             x = f(*args, **kws)
#         return x
#     return wrapper
#
# def run_hook_with_args(hook: List[Callable[P, R]], *args: P.args, **kws: P.kwargs):
#     return runner(hook)(*args, **kws)
#
# # def run_hook_with_args(hook: Callable[P, R], *args: P.args, **kws: P.kwargs) -> R:
# def run_hook_with_args(hook, *args, **kws):
#     for f in hook:
#         # type: (Sequence[Callable[P, R]], P.args, P.kwargs) -> R
#         # return runner(hook)(*args, **kws)
#         return f(*args, **kws)
#
# for f in [print, int]:
#     reveal_type(f)
#     f("ok", 2, 3, file=io.StringIO, omgz=42)
#
# runner([print, int])("ok", 2, 3, file=io.StringIO, omgz=42)
# [f("ok", 2, 3, file=io.StringIO, omgz=42) for f in [py.print, to_int]]
# runner([print])
#
# run_hook_with_args(print, "ok", 2, 3, file=io.StringIO, omgz=42)
# run_hook_with_args([print], file=42)

def dispatch(after=None, around=None):
    def inner(f: Callable[P, R]) -> Callable[P, R]:
        func = functools.singledispatch(f)
        @functools.wraps(func)
        def wrapper(*args, **kws):
            if not null(around):
                result = around(func, *args, **kws)
            else:
                result = func(*args, **kws)
            if after:
                result = after(result, *args, **kws)
            return result
        wrapper.dispatch = func.dispatch
        wrapper.register = func.register
        wrapper.registry = func.registry
        if TYPE_CHECKING:
            wrapper = func
        return wrapper
    return inner

prrepr_ = [lambda x: x if py.isinstance(x, str) and x.isprintable() else repr(x)]

def prrepr(x):
    return prrepr_[0](x)

def inner(s):
    if isinstance(s, std.Sized) and len(s) >= 2:
        return s[1:-1]
    return s

def prcons(self: Cons):
    if consp(cdr(self)):
        if car(self) == "quote":
            return "'" + prrepr(car(cdr(self)))
        if car(self) == "unquote":
            return "," + prrepr(car(cdr(self)))
        if car(self) == "unquote-splicing":
            return ",@" + prrepr(car(cdr(self)))
        if car(self) == "quasiquote":
            return "`" + prrepr(car(cdr(self)))
        if car(self) == "fn" and prrepr(car(cdr(self))) == "(_)" and null(cdr(cdr(cdr(self)))):
            return "[" + inner(prrepr(car(cdr(cdr(self))))) + "]"
    s = []
    for tail in self.tails():
        try:
            s += [prrepr(car(tail))]
        except CircularIteration:
            s += ["circular"]
        # if atom(cdr(tail)):
        it = cdr(tail)
        if not null(it) and not consp(it):
            s += [".", prrepr(cdr(tail))]
    return '(' + ' '.join(s) + ')'

A = TypeVar("A")
D = TypeVar("D")

class Freezable(ABC):
    frozen_: Optional[Literal[False, True, "car", "cdr"]]

    @property
    def frozen(self):
        return self.frozen_

    @frozen.setter
    def frozen(self, value: Optional[Literal[False, True, "car", "cdr"]]):
        self.frozen_ = value

    @property
    def frozen_car(self) -> bool:
        return self.frozen is True or self.frozen == "car"

    @property
    def frozen_cdr(self) -> bool:
        return self.frozen is True or self.frozen == "cdr"

    def __init__(self, *, frozen: Optional[Literal[False, True, "car", "cdr"]] = False):
        self.frozen_ = frozen

class HasCar(Generic[A], Freezable):
    # frozen: Optional[Literal[False, True, "car"]]
    car_: Optional[A]
    # get_car: Optional[Callable[[HasCar[A]], A]]
    # set_car: Optional[Callable[[HasCar[A], A], A]]

    def __init__(self,
                 car: Optional[A] = unset,
                 *,
                 # get_car: Optional[Callable[[HasCar[A]], A]] = None,
                 # set_car: Optional[Callable[[HasCar[A], A], A]] = None,
                 frozen: Optional[Literal[False, True, "car", "cdr"]] = False):
        super().__init__(frozen=frozen)
        self.car_ = init(self.car_default, car)
        # self.set_car = set_car
        # self.get_car = get_car

    def car_default(self) -> A:
        return unset

    @property
    def car(self) -> A:
        # return self.get_car(self) if self.get_car else self.car_
        return self.car_

    @car.setter
    def car(self, value: Optional[A]):
        self.set_car(value)
        # if self.set_car:
        #     self.set_car(self, value)
        # else:
        #     self.car_ = value

    def check_frozen_car(self):
        if self.frozen_car:
            error("Can't set frozen car")

    def set_car(self, value: Optional[A]) -> Optional[A]:
        self.check_frozen_car()
        self.car_ = value
        return value

class HasCdr(Generic[A, D], Freezable):
    # frozen: Optional[Literal[False, True, "cdr"]]
    cdr_: Optional[D]
    # get_cdr: Optional[Callable[[HasCdr[A, D]], D]]
    # set_cdr: Optional[Callable[[HasCdr[A, D], D], D]]

    def __init__(self,
                 cdr: Optional[D] = unset,
                 *,
                 # get_cdr: Optional[Callable[[HasCdr[A, D]], D]] = None,
                 # set_cdr: Optional[Callable[[HasCdr[A, D], A], D]] = None,
                 frozen: Optional[Literal[False, True, "cdr", "cdr"]] = False):
        super().__init__(frozen=frozen)
        self.cdr_ = init(self.cdr_default, cdr)
        # self.set_cdr = set_cdr
        # self.get_cdr = get_cdr

    def cdr_default(self) -> D:
        return unset

    @property
    def cdr(self) -> D:
        # return self.get_cdr(self) if self.get_cdr else self.cdr_
        return self.cdr_

    @cdr.setter
    def cdr(self, value: Optional[D]):
        self.set_cdr(value)
        # if self.set_cdr:
        #     self.set_cdr(self, value)
        # else:
        #     self.cdr_ = value

    def check_frozen_cdr(self):
        if self.frozen_cdr:
            error("Can't set frozen cdr")

    def set_cdr(self, value: Optional[D]) -> Optional[D]:
        self.check_frozen_cdr()
        self.cdr_ = value
        return value

class FrozenCar(HasCar[A]):
    def __init__(self, car: Optional[A] = unset, *, frozen: Optional[Literal[False, True, "cdr", "cdr"]] = "car"):
        assert frozen is True or frozen == "car"
        super().__init__(car, frozen=frozen)

class FrozenCdr(HasCdr[A, D]):
    def __init__(self, cdr: Optional[D] = unset, *, frozen: Optional[Literal[False, True, "cdr", "cdr"]] = "cdr"):
        assert frozen is True or frozen == "cdr"
        super().__init__(cdr, frozen=frozen)

# class HasCdr(Freezable[A, D]):
#     frozen: Optional[Literal[False, True, "cdr"]]
#     cdr_: Optional[D]
#     get_cdr: Optional[Callable[[HasCdr[A, D]], D]]
#     set_cdr: Optional[Callable[[HasCdr[A, D], D], D]]
#
#     @property
#     def cdr(self: HasCdr[A, D]) -> D:
#         return self.get_cdr(self) if self.get_cdr else self.cdr_
#
#     @cdr.setter
#     def cdr(self: HasCdr[A, D], value: D):
#         if self.frozen in [True, "cdr"]:
#             error("Can't set frozen cdr")
#         if self.set_cdr:
#             self.set_cdr(self, value)
#         else:
#             self.cdr_ = value


ConsSelf: TypeAlias = "Cons[A, D]"
ConsNil: TypeAlias = "Cons[A, None]"

ConsSeqSelf = TypeVar("ConsSeqSelf", bound="ConsSeq[A]")

class ConsBase(HasCar[A]):
    @classmethod
    def rest(cls: Type[ConsSeqSelf[A]]) -> Cons[A, ConsSeqSelf[A]]:
        ...

class Cdr(Protocol[D]):
    @property
    @abstractmethod
    def cdr(self) -> D: ...
    @cdr.setter
    def cdr(self, value: D): ...
    @abstractmethod
    def get_cdr(self) -> D: ...
    @abstractmethod
    def set_cdr(self, value: D) -> D: ...

class SeqCdr(Protocol[A]):
    @property
    @abstractmethod
    def cdr(self) -> Optional[Union[List[A], Tuple[A, ...], Cons[A, Optional[SeqCdr[A]]]]]: ...
    @cdr.setter
    def cdr(self, value: Optional[Union[List[A], Tuple[A, ...], Cons[A, Optional[SeqCdr[A]]]]]): ...
    @abstractmethod
    def get_cdr(self) -> Optional[Union[List[A], Tuple[A, ...], Cons[A, Optional[SeqCdr[A]]]]]: ...
    @abstractmethod
    def set_cdr(self, value: Optional[Union[List[A], Tuple[A, ...], Cons[A, Optional[SeqCdr[A]]]]]) -> Optional[Union[List[A], Tuple[A, ...], Cons[A, Optional[SeqCdr[A]]]]]: ...


class ConsSeq(HasCar[A], SeqCdr[A]):
    pass

@functools.total_ordering
# class Cons(HasCar[A], HasCdr[A, D]):
# class Cons(ConsSeq[A], HasCdr[A, D]):
class Cons(HasCar[A], HasCdr[A, D]):
    # if TYPE_CHECKING:
    #     @overload
    #     def __new__(cls: ConsType[A, None], car: Optional[A]) -> Cons[A, Optional[ConsNil]]:
    #         ...
    #     def __new__(cls, car: Optional[A], cdr: Optional[D], frozen: Optional[Literal[False, True, "car", "cdr"]] = False):
    #         ...
    ConsL: TypeAlias = "Cons[A, Optional[ConsL[A]]]"
    @classmethod
    def fromseq(cls: Type[ConsSelf], seq: Iterable[A]) -> ConsSelf[A, Optional[ConsSelf[A]]]:
        raise NotImplementedError

    @classmethod
    def fromlst(cls: Type[ConsSelf], seq: Iterable[A]) -> ConsSelf[A, ConsListBase[A]]:
        raise NotImplementedError

    @classmethod
    def foo(cls: Cons[int, D]):
        cls.make()

    def __init__(self,
                 car: Optional[A] = unset,
                 cdr: Optional[D] = unset,
                 # set_car: Optional[Callable[[Cons[A, D], D], D]] = None,
                 # set_cdr: Optional[Callable[[Cons[A, D], A], A]] = None,
                 # get_car: Optional[Callable[[Cons[A, D]], A]] = None,
                 # get_cdr: Optional[Callable[[Cons[A, D]], D]] = None,
                 frozen: Optional[Literal[False, True, "car", "cdr"]] = False):
        HasCar.__init__(self, car=car, frozen=frozen)
        HasCdr.__init__(self, cdr=cdr, frozen=frozen)
        # super(HasCar, self).__init__(self, car=car, frozen=frozen)
        # self.car_ = car
        # self.cdr_ = cdr
        # self.set_car = set_car
        # self.set_cdr = set_cdr
        # self.get_car = get_car
        # self.get_cdr = get_cdr
        # self.frozen = frozen


    @classmethod
    def new(cls: Type[T], src: Cons[A, D], **kws) -> T:
        # return cls(src.car_, src.cdr_, src.set_car, src.set_cdr, src.get_car, src.get_cdr)
        return cls(src.car_, src.cdr_, frozen=src.frozen_, **kws)

    # @property
    # def car(self) -> A:
    #     return self.get_car() if self.get_car else self.car_
    #
    # @property
    # def cdr(self) -> D:
    #     return self.get_cdr() if self.get_cdr else self.cdr_
    #
    # @car.setter
    # def car(self, value: A):
    #     if self.frozen in [True, "car"]:
    #         error("Can't set frozen car")
    #     if self.set_car:
    #         self.set_car(value)
    #     else:
    #         self.car_ = value
    #
    # @cdr.setter
    # def cdr(self, value: D):
    #     if self.frozen in [True, "cdr"]:
    #         error("Can't set frozen cdr")
    #     if self.set_cdr:
    #         self.set_cdr(value)
    #     else:
    #         self.cdr_ = value

    def for_each_tail_safe(self) -> Generator[Cons[A, D]]:
        tortoise: Optional[Cons[A, D]] = self
        hare: Optional[Cons[A, D]] = self
        while consp(hare):
            yield hare
            hare = cdr(hare)
            if not consp(hare):
                break
            yield hare
            hare = cdr(hare)
            tortoise = cdr(tortoise)
            if eq(hare, tortoise):
                raise CircularIteration()

    def for_each_tail_unsafe(self) -> Generator[Cons[A, D]]:
        hare: Optional[Cons[A, D]] = self
        while consp(hare):
            yield hare
            hare = cdr(hare)

    def tails(self) -> Generator[Cons[A, D]]:
        return self.for_each_tail_safe()
        # return self.for_each_tail_unsafe() # TODO: does this give a speed boost?

    def values(self) -> Generator[A]:
        for tail in self.tails():
            yield car(tail)

    def __iter__(self):
        return iter(self.values())

    def list(self, proper=True):
        if null(self):
            return []
        tail = nil
        return [car(tail := x) for x in self.tails()] + ([] if proper or null(cdr(tail)) or not consp(cdr(tail)) else [".", cdr(tail)])

    @overload
    def at(self, i: Optional[int]) -> Optional[Cons[A, D]]: ...
    @overload
    def at(self, i: slice) -> Optional[List[Cons[A, D]]]: ...
    def at(self, i):
        try:
            out = self.list()[i]
        except IndexError:
            return nil
        if isinstance(i, slice):
            out = XCONS(out)
            if consp(out):
                out = type(self).new(out)
        return out

    def cut(self, i: Optional[int] = None, j: Optional[int] = None):
        return self.at(slice(i, j))

    @reprlib.recursive_repr()
    def __repr__(self):
        # return f'Cons({car(self)!r}, {cdr(self)!r})'
        return prcons(self)

    def __getitem__(self, item):
        return [x for x in self or []][item]
        # for tail in self:
        #     if item <= 0:
        #         break
        #     item -= 1
        # if null(tail):
        #     raise IndexError("cons index out of range")
        # return tail

    def __len__(self):
        n = 0
        if not null(self):
            for _tail in self.tails():
                n += 1
        return n

    def __lt__(self, other):
        if car(self) == car(other):
            return cdr(self) < cdr(other)
        else:
            return car(self) < car(other)

    def __eq__(self, other: Optional[Cons[A, D]]):
        return isinstance(other, Cons) and car(self) == car(other) and cdr(self) == cdr(other)

    def __hash__(self):
        if self.frozen is True:
            return hash((car(self), cdr(self)))
        else:
            return hash(py.id(self))


# if TYPE_CHECKING:
#     class ConsSeq(Generic[D], Cons[A, ConsSeq[A]], ConsSeq[A]):
#         pass
#     # ConsSeq = Cons
#     ConsSeq = TypeVar("ConsSeq", bound=ConsSeq)

ConsList = TypeVar("ConsList", bound="ConsListBase")

class ConsListBase(Generic[A], Cons[A, Optional[D]]):
    @property
    def cdr(self) -> Optional[ConsList[A]]:
        return super().cdr
    @cdr.setter
    def cdr(self, value: ConsList[A]):
        super().cdr = value

    def set_cdr(self, value: Optional[ConsList[A]]) -> Optional[ConsList[A]]:
        return super().set_cdr(value)


if TYPE_CHECKING:
    ConsList = Union[Cons[A, D], Tuple[A, ...], ConsList[A]]

ConsSequence = Optional[Union[Cons[A, Optional[SeqCdr[A]]], ConsSeq[A], Tuple[A, ...], List[A]]]
ConsPair = Optional[Union[Cons[A, D], Tuple[A, D], Mapping[A, D]]]

def _CheckSeq():
    def req(l: ConsSequence[int]):
        return l.cdr
    req(['hi'])
    req([1,2,3])
    def req(l: ConsSequence[ConsPair[str, T]]) -> T:
        for cell in l:
            cell.car
            return cell.cdr

    req([('hi', True)])
    req([dict(hi=True)])
    req([dict([('hi', True)])])
    req([dict([(42, True)])])


def _check():
    zz: ConsList[int]
    zz.car
    zz.cdr
    cdr(zz)

    l: ConsList[int] = Cons(1, Cons(2, "str"))
    l: ConsList[int] = Cons(1, Cons(2, Cons(3, None)))
    l: ConsList[int] = Cons(1, Cons(2, Cons(3, Cons("foo", None))))
    l: ConsList[int] = Cons("str")
    cdr(l)
    car(cdr(l))
    l: ConsList[int] = Cons(1)
    foo: ConsList[int] = Cons("foo",None)
    foo: ConsList[int] = Cons(42,None)
    foo: ConsList[str] = Cons(42)
    foo: ConsList[int] = Cons(42)
    ConsList.new(foo)
    foo.set_cdr(21)
    foo.set_cdr("asdf")
    foo.set_cdr(nil)
    foo.set_car("ok")
    foo.set_car(420)
    foo.set_car(nil)
    foo.set_cdr((1,2,3))
    foo.set_cdr(("a",2,3))
    foo.set_cdr(nil)

    foo1 = foo.cdr
    foo2 = foo1.cdr
    i: int = foo2.car
    baz: ConsList[Cons[str, bool]] = Cons("foo")


class SupportsHash(Protocol):
    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError
    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

class Cell(Cons[SupportsHash, D]):
    # car: str
    # cdr: D
    # car: Tuple[A, Dict[A, D]]
    # cdr: D

    def __init__(self, kvs: Dict[SupportsHash, D], k: SupportsHash, *default: Optional[D],
                 frozen: FrozenSpec = "car",
                 # get_car=None, set_car=None, get_cdr=None, set_cdr=None,
                 ):
        if modulep(kvs):
            kvs = kvs.__dict__
        # if get_cdr is None:
        #     def get_cdr(self: Cons[str, D]) -> D:
        #         if isinstance(kvs, std.Mapping):
        #             return kvs.get(self.car, *default)
        #         else:
        #             assert not consp(kvs)
        #             return getattr(kvs, k, *default)
        # if set_cdr is None:
        #     def set_cdr(v):
        #         if isinstance(kvs, std.Mapping):
        #             if isinstance(kvs, std.MutableMapping):
        #                 kvs[self.car] = v
        #             else:
        #                 raise Error("Can't update non-mutable mapping")
        #         else:
        #             assert not consp(kvs)
        #             setattr(kvs, k, v)
        # super().__init__(car=k, get_car=get_car, set_car=set_car, get_cdr=get_cdr, set_cdr=set_cdr)
        # super().__init__(car=(k, kvs), cdr=default, frozen=frozen)
        super().__init__((k, kvs), default, frozen)

    @property
    def key(self) -> Union[str, SupportsHash]:
        k, kvs = self.car_
        return k

    @property
    def kvs(self) -> Dict[SupportsHash, D]:
        k, kvs = self.car_
        return kvs

    @property
    def car(self) -> SupportsHash:
        return self.key

    @property
    def default(self) -> Tuple[Optional[D], ...]:
        return self.cdr_

    @property
    def cdr(self) -> D:
        k, kvs = self.key, self.kvs
        default = self.default
        if isinstance(kvs, std.Mapping):
            return kvs.get(k, *default)
        else:
            assert isinstance(k, str)
            assert not consp(kvs)
            return getattr(kvs, k, *default)

    @cdr.setter
    def cdr(self, value):
        self.set_cdr(value)

    def set_cdr(self, v: Optional[D]) -> Optional[D]:
        k, kvs = self.key, self.kvs
        if isinstance(kvs, std.Mapping):
            if isinstance(kvs, std.MutableMapping):
                kvs[k] = v
            else:
                raise Error("Can't update non-mutable mapping")
        else:
            assert isinstance(k, str)
            assert not consp(kvs)
            setattr(kvs, k, v)
        return v

    # def cdr_default(self) -> D:
    #     return self.default[0]

# if TYPE_CHECKING:
#     Cell = Optional[Cons[Tuple[SupportsHash, Dict[SupportsHash, D]], D]]

# if TYPE_CHECKING:
#     # class Cell(Cons[str, D], HasCdr[D]):
#     #     ...
#     Cell: Type[Union[Cell[D], Cons[str, D], HasCdr[D]]]

@dispatch()
def XCONS(x: Union[Cons[A, D], Tuple[A, ...], List[A], Dict[A, D]]) -> Cons[A, D]:
    assert isinstance(x, Cons)
    return x

@overload
def XCONS_tuple(x: Tuple[A, D]) -> Cons[A, D]: ...
@overload
def XCONS_tuple(x: Tuple[A, ...]) -> ConsList[A]: ...

@XCONS.register(tuple)
@XCONS.register(list)
def XCONS_tuple(x): #, set_car=None, set_cdr=None):
    if x:
        # return Cons(x[0], XCONS(x[1:]))
        xs = tuple(reversed(x))
        out = nil
        while xs:
            # out = Cons(xs[0], out, set_car, set_cdr)
            out = Cons(xs[0], out)
            xs = xs[1:]
        return out
    else:
        return nil

@XCONS.register(type(None))
def XCONS_nil(x):
    return nil

def _check():
    XCONS_tuple((1,"foo"))
    XCONS_tuple((1,2,3))
# @XCONS.register(dict)
# def XCONS_dict(x):
#     def set_car(v):
#         raise Error("Can't set frozen cell")
#     def make_set_cdr(k):
#         def set_cdr(v):
#             x[k] = v
#             return v
#         return set_cdr
#     # return XCONS_tuple(tuple(Cons(k, v, set_car, make_set_cdr(k)) for k, v in x.items()))
#     return XCONS_tuple(tuple(Cons(k, v, set_car, make_set_cdr(k)) for k, v in x.items()), set_car, set_car)
#
# @XCONS.register(std.Mapping)
# def XCONS_dict(kvs: dict):
#     def set_car(v):
#         raise Error("Can't set frozen car")
#     def set_cdr(v):
#         raise Error("Can't set frozen cdr")
#     return XCONS_tuple(tuple(Cell(kvs, k, frozen=True) for k in kvs.keys()), set_car=set_car, set_cdr=set_cdr)
#
# @XCONS.register(std.MutableMapping)
# def XCONS_dict(kvs: dict):
#     def set_car(v):
#         raise Error("Can't set frozen car")
#     def set_cdr(v):
#         raise Error("Can't set frozen cdr")
#     return XCONS_tuple(tuple(Cell(kvs, k, set_car=set_car) for k in kvs.keys()), set_car=set_car, set_cdr=set_cdr)

@dispatch()
def consp(x) -> bool:
    return isinstance(x, Cons)

@dispatch()
def integerp(x) -> bool:
    return py.type(x) is int or (isinstance(x, int) and not isinstance(x, bool))

@dispatch()
def numberp(x) -> bool:
    if isinstance(x, bool):
        return False
    else:
        return isinstance(x, (int, float))

def string_literal_p(e) -> bool:
    if not isinstance(e, str):
        return False
    if len(e) <= 0:
        return False
    if e.startswith('"') and e.endswith('"'):
        return True
    if e.startswith('-'):
        e = e[1:]
    if len(e) <= 0:
        return False
    return e[0].isdigit()

@dispatch()
def dictp(x) -> bool:
    return isinstance(x, std.Mapping)

@dispatch()
def modulep(x) -> bool:
    return isinstance(x, types.ModuleType)

@dispatch()
def streamp(x) -> bool:
    return isinstance(x, io.IOBase)

def _check():
    class TestCar(HasCar[A]):
        pass
    zz = TestCar(1)
    zz.car
    car(zz)

    foo = Cons(1)
    foo.rest().rest()
    foo.car
    bar: ConsSeq[int] = foo
    bar.car
    bar.cdr.cdr.car
    cdr(bar)
    def requires_cons(c: ConsSeq[Cons[int, bool]]):
        ...
    # requires_cons(bar)
    requires_cons(((1, True),))
    requires_cons(((1, "foo"),))
    requires_cons(Cons(1,2))
    requires_cons(Cons(Cons(1,2), nil))
    zz = Cons(Cons(1, nil),2)
    def foo2(x: ConsSeq[int]):
        x.car
        x.cdr
        x.cdr
        x.car
        ...
    foo2(Cons("hi"))

@overload
def car(x: None): ...
@overload
def car(x: HasCar[A]) -> A: ...
@overload
def car(x: Tuple[A, D]) -> A: ...
@overload
def car(x: Tuple[A, ...]) -> Optional[A]: ...
@overload
def car(x: Cons[A, D]) -> A: ...
# @overload
# def car(x: ConsList[A]) -> A: ...

@dispatch()
# def car(x: Optional[Union[HasCar[A], Cons[A, D], Tuple[A, ...]]]) -> Optional[A]:
def car(x):
    return x if null(x) else x.car

@overload
def cdr(x: None): ...
@overload
def cdr(x: HasCdr[A, D]) -> D: ...
@overload
def cdr(x: Tuple[A, D]) -> D: ...
@overload
def cdr(x: Tuple[A, ...]) -> Optional[Tuple[A, ...]]: ...
@overload
def cdr(x: Cons[A, D]) -> D: ...
# @overload
# def cdr(x: ConsList[A]) -> Optional[ConsList[A]]: ...
# @overload
# def cdr(x: ConsList[A]) -> ConsList[A]: ...
# @overload
# def cdr(x: Cell[D]) -> D: ...
# @overload
# def cdr(x: ConsSeqSelf) -> Optional[ConsSeqSelf]: ...

@overload
def cdr(x: SeqCdr[A]) -> Optional[Union[List[A], Tuple[A, ...], Cons[A, Optional[SeqCdr[A]]]]]: ...

@dispatch()
# def cdr(x: Union[Cons[A, D], Tuple[A, D], HasCdr[D]]) -> D:
# def cdr(x: Optional[Union[HasCdr[A, D], Cons[A, D], Tuple[A, D], Tuple[D, ...]]]) -> Optional[D]:
# def cdr(x: Optional[HasCdr[A, D]]) -> Optional[D]:
def cdr(x):
    return x if null(x) else x.cdr

def _check():
    zz = cdr(Cons(1,2))
    uu = cdr(it := Cons(1,Cons("ok", 99)))
    it.set_cdr(21)
    it.set_cdr(nil)

    a: SeqCdr[int]
    cdr(a)
    def join(l: Cons[Cons[int, bool]]):
        ...
    join(list(1,2))
    join(Cons(Cons(1,True)))

def _check2():
    zz = cdr(Cons(1,2))
    uu = cdr(it := Cons(1,Cons("ok", 99)))
    it.set_cdr(21)
    it.set_cdr(nil)
    # def require(x: Optional[ConsSeq[str]]):
    #     ...
    Str = TypeVar("Str", bound=str)
    def require(x: ConsType[Str]):
        ...
    require(Cons(1,Cons(2, nil)))
    require(Cons('a',Cons('b', nil)))
    require(Cons('a',Cons('b', nil)))

@dispatch()
def eq(x, y) -> bool:
    return x is y

@eq.register(str)
@eq.register(bytes)
@eq.register(numbers.Number)
def eqv(x, y) -> bool:
    return x == y


@consp.register(tuple)
def consp_tuple(x):
    return isinstance(x, tuple)

@null.register(tuple)
def null_tuple(x):
    return len(x) <= 0

@car.register(tuple)
def car_tuple(x):
    return x[0] if x else nil

@cdr.register(tuple)
def cdr_tuple(x):
    return XCONS(x[1:]) if x and x[1:] else nil


lul_init()

def _check():
    zz: Cell[Union[str, SupportsHash], int]
    zz.car
    zz.cdr
    qqc = car(zz)
    qq = cdr(zz)
    def expect(uu: Cell[int]):
        ...
    expect(a1 := Cell(dict(a=("omg", 42)), "a"))
    expect(a2 := Cell(dict(a=99), "a"))
    expect(a3 := Cell(dict(a=99), "a", "bad"))
    expect(a4 := Cell(dict(a=99), "a", 420))
    expect(a5 := Cell(dict(a=99), ["bad"], 420))
    expect(a6 := Cell(dict(a=99), object(), 420))

    Cons.fromseq([1,2,3])
    Cons.fromlst([1,2,3])
    Cons.make()

