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




P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
U = TypeVar("U")

P2 = ParamSpec("P2")
R2 = TypeVar("R2")
T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

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
    else:
        return bool(x)

@functools.singledispatch
def falsep(x):
    # return x is False
    return not truthy(x)

def is_p(x):
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
    def inner(f):
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

def prrepr(x):
    return x if py.isinstance(x, str) else repr(x)

def prcons(self):
    if self.car == "quote":
        return "'" + prrepr(car(cdr(self)))
    if self.car == "unquote":
        return "," + prrepr(car(cdr(self)))
    if self.car == "unquote-splicing":
        return ",@" + prrepr(car(cdr(self)))
    if self.car == "quasiquote":
        return "`" + prrepr(car(cdr(self)))
    s = []
    for tail in self:
        try:
            s += [prrepr(car(tail))]
        except CircularIteration:
            s += ["circular"]
        # if atom(cdr(tail)):
        it = cdr(tail)
        if not null(it) and not consp(it):
            s += [".", prrepr(cdr(tail))]
    return '(' + ' '.join(s) + ')'

class Cons:
    def __init__(self, car, cdr, set_car=None, set_cdr=None):
        self.car = car
        self.cdr = cdr
        self.set_car = set_car
        self.set_cdr = set_cdr

    def __iter__(self):
        tortoise = self
        hare = self
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

    def list(self):
        return [car(x) for x in self]

    def at(self, i):
        return self.list()[i]

    # @recursive_repr
    def __repr__(self):
        # return f'Cons({car(self)!r}, {cdr(self)!r})'
        return prcons(self)

    def __getitem__(self, item):
        return [x for x in self][item]
        # for tail in self:
        #     if item <= 0:
        #         break
        #     item -= 1
        # if null(tail):
        #     raise IndexError("cons index out of range")
        # return tail

    def __len__(self):
        n = 0
        for tail in self:
            n += 1
        return n

@dispatch()
def XCONS(x):
    assert isinstance(x, Cons)
    return x

@XCONS.register(tuple)
def XCONS_tuple(x, set_car=None, set_cdr=None):
    if x:
        # return Cons(x[0], XCONS(x[1:]))
        xs = tuple(reversed(x))
        out = nil
        while xs:
            out = Cons(xs[0], out, set_car, set_cdr)
            xs = xs[1:]
        return out
    else:
        return nil

@XCONS.register(dict)
def XCONS_dict(x):
    def set_car(v):
        raise Error("Can't set frozen cell")
    def make_set_cdr(k):
        def set_cdr(v):
            x[k] = v
            return v
        return set_cdr
    # return XCONS_tuple(tuple(Cons(k, v, set_car, make_set_cdr(k)) for k, v in x.items()))
    return XCONS_tuple(tuple(Cons(k, v, set_car, make_set_cdr(k)) for k, v in x.items()), set_car, set_car)

@dispatch()
def consp(x):
    return isinstance(x, Cons)

@dispatch()
def numberp(x):
    if isinstance(x, bool):
        return False
    else:
        return isinstance(x, (int, float))

@dispatch()
def car(x):
    return x if null(x) else x.car

@dispatch()
def cdr(x):
    return x if null(x) else x.cdr

@dispatch()
def eq(x, y):
    return x is y

@eq.register(str)
@eq.register(bytes)
@eq.register(numbers.Number)
def eqv(x, y):
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


nil = None
t = True

lul_init()