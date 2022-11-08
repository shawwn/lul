from __future__ import annotations

import dataclasses
import operator
import sys
import os
import inspect
import numbers
import types
import itertools
from typing_extensions import * # type: ignore
# if not TYPE_CHECKING:
#     from typing_extensions import ParamSpec, ParamSpecArgs, ParamSpecKwargs
from typing import *

# if not TYPE_CHECKING:
#     List = List
#     Tuple = Tuple
# else:
#     List = list
#     Tuple = tuple

import re
import io
import keyword
import functools
import builtins as py
import collections.abc as std
import abc # type: ignore
import collections # type: ignore
import contextvars as CV

def stringp(x):
    return isinstance(x, str)

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

    @property
    def value(self):
        return symbol_value(self)

    @value.setter
    def value(self, value):
        set_symbol_value(self, value)

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
    # V.obarray = make_vector(OBARRAY_SIZE, make_fixnum(0))
    V.obarray = []
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

def symbol_name(sym: Lisp_Object):
    return XSYMBOL(sym).u.s.name

def set_symbol_function(sym: Lisp_Object, function: Lisp_Object):
    XSYMBOL(sym).u.s.function = function

def symbol_function(sym: Lisp_Object):
    return XSYMBOL(sym).u.s.function

def set_symbol_plist(sym: Lisp_Object, plist: Lisp_Object):
    XSYMBOL(sym).u.s.plist = plist

def symbol_plist(sym: Lisp_Object):
    return XSYMBOL(sym).u.s.plist

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

def set_symbol_value(sym: Lisp_Object, v):
    return SET_SYMBOL_VAL(XSYMBOL(sym), v)

def symbol_value(sym: Lisp_Object):
    assert XSYMBOL(sym).u.s.redirect == SYMBOL_PLAINVAL
    assert XSYMBOL(sym).u.s.trapped_write != SYMBOL_NOWRITE, "Symbol is constant"
    return XSYMBOL(sym).u.s.val.value

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




K = TypeVar("K")
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
U = TypeVar("U")

P2 = ParamSpec("P2")
R2 = TypeVar("R2")
T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

def dispatch(arg=0, *, after=None, around=None):
    def inner(f):
        func = functools.singledispatch(f)
        @functools.wraps(func)
        def wrapper(*args, **kws):
            if wrapper.around:
                result = (wrapper.around)(func, *args, **kws)
            else:
                # result = func(*args, **kws)
                if len(args) <= arg:
                    funcname = getattr(func, '__name__', 'singledispatch function')
                    raise TypeError(f'{funcname} requires at least {arg} positional argument(s)')
                result = func.dispatch(args[arg].__class__)(*args, **kws)
            if wrapper.after:
                result = (wrapper.after)(result, *args, **kws)
            return result
        wrapper.dispatch = func.dispatch
        wrapper.register = func.register
        wrapper.registry = func.registry
        wrapper.around = around
        wrapper.after = after
        if TYPE_CHECKING:
            wrapper = func
        return wrapper
    return inner

@dispatch()
def deref(x):
    return x

@deref.register(CV.ContextVar)
def deref_ContextVar(x: CV.ContextVar):
    return x.get()

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

@dispatch()
def functionp(x):
    return inspect.isfunction(x)

@dispatch()
def null(x):
    return x is None

@dispatch()
def truthy(x):
    return bool(x)

@truthy.register(bool)
def truthy_bool(x):
    return x

@truthy.register(numbers.Number)
def truthy_Number(_x):
    return True

def falsep(x):
    return not x
    # if not x:
    #     if x is 0 or x is 0.0:
    #         return False
    #     return True
    # return False
    # return not truthy(x)
    #
    # if x in [True, False, nil]:
    #     return not x
    # return not truthy(x)

def ok(x):
    return not null(x)

def no(x):
    return null(x) or falsep(x)

def yes(x):
    return not no(x)

@dispatch()
def iterate(l):
    return [(k, v) for k, v in py.enumerate(l)]

@iterate.register(type(None))
def iterate_None(l: None):
    return []

@iterate.register(std.Mapping)
def iterate_dict(l: Mapping):
    return [(k, v) for k, v in l.items()]

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

@dispatch()
def prrepr(x):
    return repr(x)

@prrepr.register(types.FunctionType)
@prrepr.register(types.BuiltinFunctionType)
def prrepr_function(x):
    module = getattr(x, "__module__", "") or ""
    name = getattr(x, "__qualname__", getattr(x, "__name__", ""))
    name = name.replace("<lambda>", f"lambda_{py.hex(py.id(x))}")
    name = name.replace(".<locals>", "")
    fqn = f"{module}/{name}" if module else name
    return f"#'{fqn}"

@prrepr.register(type(None))
def prrepr_nil(x):
    return "nil"

@prrepr.register(bool)
def prrepr_bool(x: bool):
    return "t" if x else repr(x)

@prrepr.register(str)
def prrepr_str(x):
    return x

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


@functools.total_ordering
class Cons:
    get_car: Optional[Callable[[], Any]]
    get_cdr: Optional[Callable[[], Any]]
    set_car: Optional[Callable[[T], T]]
    set_cdr: Optional[Callable[[T], T]]
    def __init__(self, car=None, cdr=None, *, set_car=None, set_cdr=None, get_car=None, get_cdr=None):
        self.car_ = car
        self.cdr_ = cdr
        self.set_car = set_car
        self.set_cdr = set_cdr
        self.get_car = get_car
        self.get_cdr = get_cdr

    @property
    def car(self):
        return self.get_car() if callable(self.get_car) else self.car_

    @property
    def cdr(self):
        return self.get_cdr() if callable(self.get_cdr) else self.cdr_

    @car.setter
    def car(self, value):
        if callable(self.set_car):
            self.set_car(value)
        else:
            self.car_ = value

    @cdr.setter
    def cdr(self, value):
        if callable(self.set_cdr):
            self.set_cdr(value)
        else:
            self.cdr_ = value

    # def __iter__(self):
    #     tortoise = self
    #     hare = self
    #     while consp(hare):
    #         yield hare
    #         hare = cdr(hare)
    #         if not consp(hare):
    #             break
    #         yield hare
    #         hare = cdr(hare)
    #         tortoise = cdr(tortoise)
    #         if eq(hare, tortoise):
    #             raise CircularIteration()
    def __iter__(self: Cons):
        return self.step()

    def tails(self, test=None, next=None):
        test = test or consp
        next = next or cdr
        while test(self):
            yield self
            self = next(self)

    def __reversed__(self):
        return tuple(self.tails())[::-1]

    def enumerate(self, start=0, *, values=True, props=True, rest=True) -> Generator[Any, Any]:
        #        (while ,h
        #          (let* ((,k (if (keywordp (car ,h)) (y-%key (car ,h)) (setq ,i (1+ ,i))))
        #                 (,v (if (keywordp (car ,h)) (cadr ,h) (car ,h))))
        #            ,@body)
        #          (setq ,h (if (keywordp (car ,h)) (cddr ,h) (cdr ,h))))
        h: Optional[Union[Cons, Mapping]] = self
        while ok(h):
            if dictp(h):
                h: Mapping
                for k, v in h.items():
                    if values and integerp(k):
                        yield k + start, v
                    elif props:
                        yield k, v
                break
            elif consp(h):
                h: Cons
                while consp(h):
                    v = car(h)
                    h = cdr(h)
                    if not keywordp(v):
                        if values:
                            yield start, v
                        start += 1
                    else:
                        k = keynom(v)
                        if consp(h):
                            v = car(h)
                            h = cdr(h)
                        if props:
                            yield k, v
            else:
                if rest:
                    yield unset, h
                break

    def step(self) -> Generator[Any]:
        for k, v in self.enumerate(values=True, props=False, rest=False):
            yield v

    def props(self) -> Generator[Any, Any]:
        for k, v in self.enumerate(values=False, props=True, rest=False):
            yield k, v

    def rest(self) -> Generator[Any]:
        for k, v in self.enumerate(values=False, props=False, rest=True):
            yield v

    def items(self):
        yield from self.enumerate()

    def keys(self):
        for k, v in self.items():
            yield k

    def values(self):
        for k, v in self.items():
            yield v

    def tail(self):
        if it := py.tuple(v for v in self.rest()):
            if py.len(it) == 1:
                return it[0]
        return it

    def seq(self, props=True, rest=True):
        tail = unset
        for k, v in self.enumerate(props=props, rest=rest):
            if k is unset:
                tail = v
            elif integerp(k):
                yield v
            else:
                yield keysym(k)
                yield v
        if tail is not unset:
            yield "."
            yield tail

    def list(self, props=False, rest=False, cls: Type[List] = py.list) -> List:
        return cls(self.seq(props=props, rest=rest))

    def tuple(self, props=False, rest=False, cls: Type[Tuple] = py.tuple) -> Tuple:
        return cls(self.seq(props=props, rest=rest))

    def dict(self, cls: Type[Dict] = py.dict) -> Dict:
        return cls(self.props())

    def at(self, i: int, *default):
        if i < 0:
            i += py.len(self)
        return self.get(i, *default)

    def get(self, key, *default, test=None):
        test = test or eqv
        for k, v in self.enumerate():
            if test(k, key):
                return v
        if default:
            return default[0]
        raise KeyError(key)

    # @recursive_repr
    def __repr__(self):
        # return f'Cons({car(self)!r}, {cdr(self)!r})'
        return prrepr(self)

    def __getitem__(self, item):
        if isinstance(item, py.slice):
            return self.tuple()[item]
        elif integerp(item):
            return self.at(item)
        return self.get(item)

    def __len__(self):
        n = 0
        for v in self.step():
            n += 1
        return n

    def __bool__(self):
        return True

    def __eq__(self, other):
        if not isinstance(other, Cons):
            return False
        return car(self) == car(other) and cdr(self) == cdr(other)

    def __lt__(self, other):
        if not isinstance(other, Cons):
            return False
        if car(self) == car(other):
            return cdr(self) < cdr(other)
        return car(self) < car(other)

    def __hash__(self):
        return py.hash(py.id(self))

prrepr.register(Cons, prcons)

class Cell(Cons):
    def __init__(self, kvs, k, *default, get_car=None, set_car=None, get_cdr=None, set_cdr=None):
        if get_cdr is None:
            def get_cdr():
                if isinstance(kvs, (std.Mapping, Cons)):
                    return kvs.get(self.car, *default)
                else:
                    assert not consp(kvs)
                    return getattr(kvs, self.car, *default)
        if set_cdr is None:
            def set_cdr(v):
                if isinstance(kvs, std.Mapping):
                    if isinstance(kvs, std.MutableMapping):
                        if unboundp(v):
                            if self.car in kvs:
                                del kvs[self.car]
                        else:
                            kvs[self.car] = v
                    else:
                        raise Error("Can't update non-mutable mapping")
                else:
                    assert not consp(kvs)
                    if unboundp(v):
                        if hasattr(kvs, k):
                            delattr(kvs, k)
                    else:
                        setattr(kvs, k, v)
        super().__init__(car=k, get_car=get_car, set_car=set_car, get_cdr=get_cdr, set_cdr=set_cdr)

@dispatch()
def XCONS(x) -> Cons:
    # assert isinstance(x, Cons)
    return x

@XCONS.register(type(None))
def XCONS_nil(x):
    return x

@XCONS.register(tuple)
@XCONS.register(list)
def XCONS_seq(x, *, set_car=None, set_cdr=None):
    if x:
        # return Cons(x[0], XCONS(x[1:]))
        out = nil
        # if (it := x[-3:]) and (it := x[-2:]) and it[0] == "." and it[1:]:
        #     out = it[1]
        #     x = x[:-2]
        if len(x) == 3 and x[1] == ".":
            return Cons(x[0], x[2], set_car=set_car, set_cdr=set_cdr)
        for v in reversed(x):
            out = Cons(v, out, set_car=set_car, set_cdr=set_cdr)
        return out
    else:
        return nil

@XCONS.register(types.GeneratorType)
def XCONS_GeneratorType(x):
    return XCONS(py.tuple(v for v in x))

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

@XCONS.register(std.Mapping)
def XCONS_Mapping(kvs: Mapping):
    def set_car(v):
        raise Error("Can't set frozen car")
    def set_cdr(v):
        raise Error("Can't set frozen cdr")
    return XCONS_seq(tuple(Cell(kvs, k, set_car=set_car, set_cdr=set_cdr) for k in kvs.keys()), set_car=set_car, set_cdr=set_cdr)

@XCONS.register(std.MutableMapping)
def XCONS_dict(kvs: MutableMapping):
    def set_car(v):
        raise Error("Can't set frozen car")
    def set_cdr(v):
        raise Error("Can't set frozen cdr")
    return XCONS_seq(tuple(Cell(kvs, k, set_car=set_car) for k in kvs.keys()), set_car=set_car, set_cdr=set_cdr)

@dispatch()
def consp(_x):
    return False

@consp.register(Cons)
def consp_Cons(_x):
    return True

@dispatch()
def numberp(_x):
    return False

@numberp.register(bool)
def numberp_bool(_x):
    return False

@numberp.register(numbers.Number)
def numberp_Number(_x):
    return True

assert not numberp(True)
assert numberp(1)

@dispatch()
def integerp(_x):
    return False

@integerp.register(bool)
def integerp_bool(_x):
    return False

@integerp.register(py.int)
def integerp_int(_x):
    return True

@dispatch()
def dictp(x):
    return isinstance(x, std.Mapping)

@dictp.register(Cons)
def dictp_Cons(_x):
    return nil

@dispatch()
def car(x):
    return x if null(x) else x.car
    # try:
    #     return x.car
    # except AttributeError:
    #     return nil
    # return getattr(x, "car", nil)

@dispatch()
def cdr(x):
    return x if null(x) else x.cdr
    # try:
    #     return x.cdr
    # except AttributeError:
    #     return nil
    # return getattr(x, "cdr", nil)

@dispatch()
def eq(x, y):
    return x is y

# built-in types (bool, bytearray, bytes, dict, float, frozenset, int, list, set, str, and tuple)

@eq.register(str)
@eq.register(bytes)
@eq.register(bytearray)
@eq.register(numbers.Number)
def eqv(x, y):
    return x == y


@consp.register(tuple)
def consp_tuple(_x):
    return True

@consp.register(list)
def consp_list(_x):
    return True

@null.register(tuple)
@null.register(list)
def null_seq(x):
    # return not bool(x)
    return nil if x else t

@car.register(tuple)
@car.register(list)
def car_seq(x):
    return x[0] if x else nil

@cdr.register(tuple)
@cdr.register(list)
def cdr_seq(x):
    if x:
        if it := x[1:]:
            if it[0] == ".":
                if rest := it[1:]:
                    if not it[2:]:
                        return rest[0]
            return it
    # if py.len(x) == 3 and x[1] == ".":
    #     return x[2]
    # return nil if (it := x[1:]) else it
    return nil

@dispatch()
def tee(x, n=2):
    return itertools.tee(x, n)

@tee.register(Cons)
def tee_Cons(x, n=2):
    return py.tuple([x] * n)

def memo(f):
    return functools.lru_cache(maxsize=None)(f)

class Delay(Cons):
    def __init__(self, x):
        more = True
        @memo
        def get_car():
            try:
                return car(x)
            except StopIteration:
                nonlocal more
                more = False
                return nil
        @memo
        def get_cdr():
            get_car()
            return Delay(x) if more else nil
        super().__init__(get_car=get_car, get_cdr=get_cdr)

    def __repr__(self):
        return f"(Delay {prrepr(car(self))} ...)"

# @car.register(types.GeneratorType)
# def car_Generator(x: types.GeneratorType):
#     # try:
#     #     return next(x)
#     # except StopIteration:
#     #     return nil
#     return next(x)
#
# @cdr.register(types.GeneratorType)
# def cdr_Generator(x: types.GeneratorType):
#     return x

@dispatch()
def keywordp(_x):
    return False

@keywordp.register(py.str)
def keywordp_str(x):
    return x[0:1] == ":"

@keywordp.register(Cons)
def keywordp_Cons(x: Cons):
    return eqv(car(x), "lit") and eqv(car(cdr(x)), "key")

@dispatch()
def keynom(_x):
    return nil

@keynom.register(py.str)
def keynom_str(x):
    if keywordp_str(x):
        return x[1:]

@keynom.register(Cons)
def keynom_Cons(x):
    if keywordp_Cons(x):
        return car(cdr(cdr(x)))

@dispatch()
def keysym(k):
    return Cons("lit", Cons("key", k))

@keysym.register(py.str)
def keysym_str(x):
    return ":" + x

@keysym.register(type(None))
def keysym_nil(_x):
    return nil

@dispatch()
def each(h: T) -> Generator[Tuple[K, T]]:
    h = XCONS(h)
    i = -1
    while not null(h):
        if ok(k := keynom(car(h))):
            v = car(cdr(h))
            h = cdr(cdr(h))
        else:
            i += 1
            k = i
            v = car(h)
            h = cdr(h)
        yield k, v

@each.register(std.Mapping)
def each_tab(h: Mapping[K, T]) -> Generator[Tuple[K, T]]:
    xs: List[Tuple[int, T]] = []
    for k, v in h.items():
        if integerp(k):
            xs.append((k, v))
        else:
            yield k, v
    # ensure values (integer indices) are yielded in increasing order.
    xs = sorted(xs, key=lambda x: x[0])
    for k, v in xs:
        yield k, v

import argparse

@each.register(types.ModuleType)
@each.register(argparse.Namespace)
def each_ns(h: types.ModuleType) -> Generator[Tuple[K, T]]:
    return each(h.__dict__)

@dispatch()
def step(h):
    for k, v in each(h):
        if integerp(k):
            yield v

@dispatch()
def keys(x):
    for k, v in each(x):
        if not integerp(k):
            yield k, v

def items(x):
    return py.tuple(each(x))

def vals(x):
    return py.tuple(step(x))

def props(x):
    return py.tuple(keys(x))

@iterate.register(Cons)
def iterate_Cons(l: Cons):
    xs = []
    ks = {}
    while consp(l):
        pass

nil = None
t = True

class Unset:
    def __repr__(self):
        return "unset"
unset = Unset()

def unbound():
    return unset

def unboundp(x):
    return x is unset

lul_init()