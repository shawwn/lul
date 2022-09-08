from __future__ import annotations

from typing import *
import abc
import builtins as py
import operator
from functools import singledispatch

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")
Basic = TypeVar("Number", int, float, bool, type(None))
Symbol = TypeVar("Symbol", bound=str)
Value = TypeVar("Value", Symbol, "Cons")

Car = TypeVar("Car", Symbol, "Cons")
Cdr = TypeVar("Cdr", Symbol, "Cons")

SetCar = Callable[[Car], Car]
SetCdr = Callable[[Cdr], Cdr]
ConsDispatch = Callable[[Car, Cdr, SetCar, SetCdr], Union[Car, Cdr]]
Cons = Callable[[ConsDispatch], Union[Car, Cdr]]
# # Cons = Callable[[Car, Cdr], Cell]
#
# # class Tagged(Generic[T]):
# class Tagged(abc.ABC):
#     # def __init__(self, tag: Type, rep: T):
#     #     #self.rep = rep
#     #     object.__setattr__(self, "tag", tag)
#     #     object.__setattr__(self, "rep", rep)
#     def __new__(cls, tag: Type, rep: T, *, using=object):
#         class Using(using, cls):
#             pass
#         self = using.__new__(Using)
#         object.__setattr__(self, "tag", tag)
#         object.__setattr__(self, "rep", rep)
#         # self.tag = tag
#         # self.rep = rep
#         return self
#     def __getattribute__(self, name):
#         rep = object.__getattribute__(self, "rep")
#         return object.__getattribute__(rep, name)
#     def __setattr__(self, name, rep):
#         rep = object.__getattribute__(self, "rep")
#         return object.__setattr__(rep, name, rep)
#
#
# def cons(a: Car, d: Cdr) -> Cons:
#     def cell(m: ConsDispatch):
#         def set_car(v: Car) -> Car:
#             nonlocal a
#             a = v
#             return a
#         def set_cdr(v: Cdr) -> Cdr:
#             nonlocal d
#             d = v
#             return d
#         return m(a, d, set_car, set_cdr)
#     return cell
#
#
# SetCar = Callable[[T], T]
# SetCdr = Callable[[U], U]
#
# class Dispatch(Generic[T, U]):
#     # def __call__(self, a: T, d: U, set_car: Callable[[T], T], set_cdr: Callable[[U], U]):
#     def __call__(self, a: T, d: U) -> Union[T, U]:
#         ...
#
# class Cons(Generic[T, U]):
#     def __call__(self, m: Dispatch[T, U]) -> Union[T, U]:
#         # return m(self.a, self.d, self.set_car, self.set_cdr)
#         ...
#             # def set_car(v: T) -> T:
#             #     nonlocal a
#             #     a = v
#             #     return a
#             # def set_cdr(v: U) -> U:
#             #     nonlocal d
#             #     d = v
#             #     return d
#             # return m(a, d, set_car, set_cdr)
#     # a: T
#     # d: U
#     # def __init__(self, a: T, d: U):
#     #     self.a = a
#     #     self.d = d
#     # def set_car(self, v: T):
#     #     self.a = v
#     #     return self.a
#     # def set_cdr(self, d: U):
#     #     self.d = d
#     #     return self.d
#
# del Cons
# del ConsCell
#
# # def cons(a: Car, d: Cdr):
# #     global Cons
# #     class Cons(Generic[T, U]):
# #         global ConsCell
# #         class ConsCell:
# #             def __call__(self, a: T, d: U) -> Union[T, U]:
# #                 ...
# #         def __call__(self, m: ConsCell):
# #             return m(a, d)
# #     return Cons()
# del Car
# del Cdr

# Car = TypeVar("Car")
# Cdr = TypeVar("Cdr")
#
#
# Self = TypeVar("Self", bound="Cons")
# CarType = TypeVar("CarType", bound=Type)
# CdrType = TypeVar("CdrType", bound=Type)
#
# class Cons(Generic[Car, Cdr]):
#     SetCar = Callable[[Car], Car]
#     SetCdr = Callable[[Cdr], Cdr]
#     ConsDispatch = Callable[[Car, Cdr, SetCar, SetCdr], Union[Car, Cdr]]
#     Result = Callable[[Self, ConsDispatch], Union[Car, Cdr]]
#     def __new__(cls: Self, a: Car, d: Cdr):
#         def cell(m: Callable[[Car, Cdr, Cons[Car, Cdr].SetCar, Cons[Car, Cdr].SetCdr], Union[Car, Cdr]]):
#             def set_car(v: Car) -> Car:
#                 nonlocal a
#                 a = v
#                 return a
#             def set_cdr(v: Cdr) -> Cdr:
#                 nonlocal d
#                 d = v
#                 return d
#             return m(a, d, set_car, set_cdr)
#         return cell
#     __call__: Result
#
#     # __class_getitem__: Callable[[Type[Cons], Tuple[Type[Car], Type[Cdr]]],
#     # def __class_getitem__(cls, args: Tuple[Type[T], Type[U]]) -> Type[Cons[Type[T], Type[U]]]:
#     def __class_getitem__(cls, args: Tuple[CarType, CdrType]) -> Type[Cons[CarType, CdrType]]:
#         CarType, CdrType = args
#         return Cons[CarType, CdrType]
#
#
# Cons[str, str].__new__(1, 2)
#
# # ConsNum: Type[Cons[int, int]] = Cons
# # ConsNum("foo", 1)
# Cons[int, int]("str", 1)
#
# zz: ConsNum = Cons("asdf", 1)
# zz(lambda a, d: ConsNum("asdf", 1))
# # zz("asdf", "asdf")

# def cons(a: Car, d: Cdr) -> Cons:
#     def cell(m: ConsDispatch):
#         def set_car(v: Car) -> Car:
#             nonlocal a
#             a = v
#             return a
#         def set_cdr(v: Cdr) -> Cdr:
#             nonlocal d
#             d = v
#             return d
#         return m(a, d, set_car, set_cdr)
#     cell.tag = cons
#     return cell

# Self = TypeVar("Self", bound="ConsType")
#
# class ConsType(Generic[T, U]):
#     Car = T
#     Cdr = U
#     SetCar = Callable[[Car], Car]
#     SetCdr = Callable[[Cdr], Cdr]
#     ConsDispatch = Callable[[Car, Cdr, SetCar, SetCdr], Union[Car, Cdr]]
#     Cons = Callable[[ConsDispatch], Union[Car, Cdr]]
#     @classmethod
#     def cons(cls, a: T, d: U) -> Self[T, U].Cons:
#         def cell(m: ConsDispatch):
#             def set_car(v: Car) -> Car:
#                 nonlocal a
#                 a = v
#                 return a
#             def set_cdr(v: Cdr) -> Cdr:
#                 nonlocal d
#                 d = v
#                 return d
#             return m("a", "b", set_car, set_cdr)
#             # return m(a, d, set_car, set_cdr)
#         cell.tag = Cons
#         return cell
#     def __class_getitem__(cls, args: Tuple[T, U]) -> Type[Self[T, U]]:
#         return ConsType[args]
#
# foo: ConsType[int, int].Car
# foo
# ConsType[int, int].Car
#
# class Cons(Generic[Car, Cdr]):
#     # def __class_getitem__(cls, args: Tuple[T, U]) -> Type[Cons[T, U]]:
#     #     return Cons[args]
#
#     # CarType, CdrType = args
#     Cons = Callable[[ConsDispatch], Union[Car, Cdr]]
#     SetCar = Callable[[Car], Car]
#     SetCdr = Callable[[Cdr], Cdr]
#     ConsDispatch = Callable[[Car, Cdr, SetCar, SetCdr], Union[Car, Cdr]]
#     @staticmethod
#     def cons(a: Car, d: Cdr) -> Self[int, int].Cons:
#         def cell(m: Self[int, int].ConsDispatch):
#             def set_car(v: Car) -> Car:
#                 nonlocal a
#                 a = v
#                 return a
#             def set_cdr(v: Cdr) -> Cdr:
#                 nonlocal d
#                 d = v
#                 return d
#             return m("a", "b", set_car, set_cdr)
#             # return m(a, d, set_car, set_cdr)
#         cell.tag = Cons
#         return cell
#     __call__ = Callable[[Self, ConsDispatch], Union[Car, Cdr]]

def tag(type, rep):
    object.__setattr__(rep, '__tag__', type)
    return rep

def kind(rep):
    try:
        return object.__getattribute__(rep, '__tag__')
    except AttributeError:
        return type(rep)

def isa(type):
    # return lambda x: kind(x) == type
    return lambda x: issubclass(kind(x), type)

# def cons(a: Car, d: Cdr) -> Cons:
#     def cell(m: ConsDispatch):
#         def set_car(v: Car) -> Car:
#             nonlocal a
#             a = v
#             return a
#         def set_cdr(v: Cdr) -> Cdr:
#             nonlocal d
#             d = v
#             return d
#         return m(a, d, set_car, set_cdr)
#     # cell.tag = cons
#     return tag(cons, cell)

class cons:
    def __new__(cls, a: Car, d: Cdr) -> Cons:
        def cell(m: ConsDispatch):
            def set_car(v: Car) -> Car:
                nonlocal a
                a = v
                return a
            def set_cdr(v: Cdr) -> Cdr:
                nonlocal d
                d = v
                return d
            return m(a, d, set_car, set_cdr)
        # cell.tag = cons
        return tag(cons, cell)

import functools

class Base:
    pass

def Class(f=None, name=None, bases=(Base,)):
    def Class_(f, name=name, bases=bases):
        name = name or f.__qualname__
        def __new__(cls, *args, **kws):
            dispatch = f(*args, **kws)
            self = object.__new__(cls)
            self.__dispatch__ = dispatch
            return self
        def __call__(self, *args, **kws):
            return self.__dispatch__(*args, **kws)
        # return type(name, bases, dict(__new__=lambda cls, *args, **kws: tag(cls, f(*args, **kws))))
        return type(name, bases, dict(__new__=__new__, __call__=__call__, __name__=name))
    if f is not None:
        return Class_(f, name=name, bases=bases)
    else:
        return Class_


nil = None
t = True

from reprlib import recursive_repr

def null(x):
    return eq(x, nil)


class CircularIteration(Exception):
    pass

import contextvars as CV
import contextlib
import sys
import io

unbound = CV.Token.MISSING

_seen = CV.ContextVar[List]("_seen")
stdout = CV.ContextVar[io.IOBase]("stdout", default=sys.stdout)

def format(string: str, *objects):
    spec = string
    # spec = spec.replace("%s", "{}")
    # spec = spec.replace("%S", "{!r}")
    # spec = spec.replace("%o", "{:o}")
    # spec = spec.replace("%x", "{:x}")
    # spec = spec.replace("%X", "{:x}")
    # spec = spec.replace("%d", "{:d}")
    # spec = spec.replace("%f", "{:f}")
    # spec = spec.replace("%e", "{:e}")
    # spec = spec.replace("%g", "{:g}")
    # spec = spec.replace("%c", "{:c}")
    # spec = spec.replace("%%", "%")
    spec = spec.replace("%S", "%r")
    #return spec.format(*objects)
    return spec % objects

format_message = format

class Error(Exception):
    pass


class SymbolBase:
    @property
    def name(self):
        return symbol_name(self)

    @property
    def value(self):
        return symbol_value(self)

    @property
    def plist(self):
        return symbol_plist(self)

    @plist.setter
    def plist(self, value):
        setplist(self, value)

    def __repr__(self):
        return symbol_repr(self)

SymbolName = str
SymbolValue = Any
SymbolPlist = Cons
SymbolResult = SymbolValue
SymbolSetValue = Callable[[SymbolValue], SymbolValue]
SymbolSetPlist = Callable[[SymbolPlist], SymbolPlist]
SymbolDispatch = Callable[[SymbolName, SymbolValue, SymbolPlist, SymbolSetValue, SymbolSetPlist], SymbolResult]
SymbolDispatcher = Callable[[SymbolDispatch], SymbolResult]
Symbol = Union[SymbolBase, SymbolDispatcher]

@Class(bases=(SymbolBase,))
def sym(name: str, value=unbound, plist=nil) -> Symbol:
    def dispatch(m: SymbolDispatch):
        def set_value(v):
            nonlocal value
            value = v
            return v
        def set_plist(v):
            nonlocal plist
            plist = v
            return v
        return m(name, value, plist, set_value, set_plist)
    return dispatch

Obarray = List[Symbol]
Vobarray: Obarray = []

def intern_soft(name: str, obarray: Obarray = nil):
    obarray = obarray or Vobarray
    for x in obarray:
        if x.name == name:
            return x

def intern(name: str, obarray: Obarray = nil):
    obarray = obarray or Vobarray
    if it := intern_soft(name, obarray):
        return it
    it = sym(name)
    obarray.insert(0, it)
    return it

def symbolp(x):
    return isa(sym)(x)

def compile_id(name: str):
    id = str(name)
    id = ''.join([f'_{ord(c):02d}' if not (c.isalpha() or c == "-") else c for c in id])
    id = id.replace('-', '_')
    if not id.isidentifier():
        id = "_" + id
    if id.startswith("_"):
        id = "ID_" + id
    return id
    # id = name.replace('*', '_STAR_')
    # id = name.replace('?', '_STAR_')
    # if not name.replace('-', '_').isidentifier():
    #     name = '|' + name + '|'

import re

def uncompile_id(id: str):
    def replace(m: re.Match):
        val = m.groups()[0]
        val = int(val)
        return chr(val)
    if id.startswith("ID_"):
        id = id[len("ID_"):]
    id = re.sub("[_]([0-9][0-9])", replace, id)
    id = id.replace("_", "-")
    return id
    # for m in list(re.findall("[_]([0-9][0-9])", id)):
    # assert id.isidentifier()
    # id = id.replace('_STAR_', '*')
    # id = id.replace('_', '-')
    # return id

def symbol_repr(x: Symbol):
    id = symbol_name(x)
    if not id.replace("-", "_").replace("?", "").replace("!", "").replace("*", "").isidentifier():
        id = "|" + id + "|"
    return id


def symbol_name(x: Symbol) -> str:
    return x(lambda name, value, plist, set_value, set_plist: name)

def symbol_value(x: Symbol) -> str:
    return x(lambda name, value, plist, set_value, set_plist: value)

def setq(x: Symbol, v: T) -> T:
    return x(lambda name, value, plist, set_value, set_plist: set_value(v))

def symbol_plist(x: Symbol):
    return x(lambda name, value, plist, set_value, set_plist: plist)

def setplist(x: Symbol, v):
    return x(lambda name, value, plist, set_value, set_plist: set_plist(v))

def error(format_string: str, *args):
    signal("Error", format_message(format_string, *args))

def define_error(name: Symbol, message: str, parent="error"):
    globals()[name] = cons(exn, msg)

def signal(err: Symbol, data=()):
    # if msg is None:
    #     msg = " ".join(["%s" for _ in args])
    # if args:
    #     msg = msg % tuple(repr(x) for x in args)
    # exn: Type[Exception] = get(err)
    # exn(msg, *data)
    # raise exn(msg)
    raise Error(err, *data)

def dispatch(after=None, call=None):
    def inner(f):
        func = singledispatch(f)
        @functools.wraps(func)
        def wrapper(*args, **kws):
            if call:
                result = call(func, *args, **kws)
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


def get__result(result, var, *args):
    if eq(result, unbound):
        error("Unbound variable")
    return result

@dispatch(after=get__result)
def get(var, *args):
    return value(var)

@get.register(CV.ContextVar)
def get__var(var: CV.ContextVar[T]) -> T:
    return var.get()

@get.register(str)
def get__str(var: str):
    return eval(var)

@get.register(sym)
def get__sym(var: Symbol):
    return symbol_value(var)

class QMeta(type):
    def __getattr__(self, id: str) -> Symbol:
        return self[uncompile_id(id)]
    def __getitem__(self, name: str) -> Symbol:
        id = compile_id(name)
        try:
            return object.__getattribute__(self, id)
        except AttributeError:
            x = intern(name)
            setattr(self, id, x)
            return x

class Q(metaclass=QMeta):
    t: Symbol
    nil: Symbol
    unbound: Symbol

# def set__result(_, var, val, env=nil):
#     env = env or globals()

def list(x=nil, y=nil, *args):
    if args:
        return cons(x, list(y, *args))
    else:
        return cons(x, cons(y))

def assign__func(setter, var, val, env=nil):
    done = object()
    prev = value(var, env)
    if not null(env):
        env[var] = val
    else:
        setter(var, val)
    def reset():
        nonlocal prev
        if prev is not done:
            eval(cons("setq", cons(var, cons(val))), env)
            if not null(env):
                if prev is unbound:
                    del env[var]
                else:
                    env[var] = prev
            else:
                setter(var, prev, env)
            prev = done
    return reset

NoneType = type(None)

@dispatch(call=assign__func)
def assign(var, val) -> Callable[[], NoneType]:
    raise NotImplementedError

@assign.register(sym)
def assign__sym(self: Symbol, val: T) -> T:
    return setq(self, val)

@assign.register(CV.ContextVar)
def assign__var(var: CV.ContextVar[T], val: T) -> T:
    var.set(val)
    return val

def set(var, val):
    assign(var, val)

@contextlib.contextmanager
def let(var, val):
    reset = assign(var, val)
    try:
        yield get(var)
    finally:
        reset()

@get.register(str)
def get_sym(var: str):
    return eval(var)

def disp(x):
    print(x, file=get(stdout), end='', flush=True)

@contextlib.contextmanager
def _let(var: CV.ContextVar[T], val: T):
    token = var.set(val)
    try:
        yield var.get()
    finally:
        var.reset(token)


def prcons(self: ConsBase):
    with _let(_seen, _get(_seen) or []) as seen:
        if self in seen:
            raise CircularIteration()
        s = []
        for tail in self:
            try:
                s += [repr(car(tail))]
            except CircularIteration:
                s += ["circular"]
            if atom(cdr(tail)):
                s += [".", repr(cdr(tail))]
        return '(' + ' '.join(s) + ')'


class ConsBase:
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
                raise CircularIteration

    def list(self):
        return [car(x) for x in self]

    def at(self, i):
        return self.list()[i]

    # @recursive_repr
    def __repr__(self):
        #return f'Cons({car(self)!r}, {cdr(self)!r})'
        return prcons(self)

    def __getitem__(self, item):
        for tail in self:
            if item <= 0:
                break
        if null(tail):
            raise IndexError("cons index out of range")
        return tail

    def __len__(self):
        n = 0
        for tail in self:
            n += 1
        return n

@singledispatch
def len(x):
    return py.len(x)

@len.register(type(nil))
def len_none(x):
    return 0


@Class(bases=(ConsBase,))
def cons(a: Car = nil, d: Cdr = nil) -> Cons:
    def dispatch(m: ConsDispatch):
        def set_car(v: Car) -> Car:
            nonlocal a
            a = v
            return a
        def set_cdr(v: Cdr) -> Cdr:
            nonlocal d
            d = v
            return d
        return m(a, d, set_car, set_cdr)
    return dispatch

def car(x: Cons) -> Car:
    return x if null(x) else x(lambda a, d, set_car, set_cdr: a)

def cdr(x: Cons) -> U:
    return x if null(x) else x(lambda a, d, set_car, set_cdr: d)

def replaca(x: Cons, y: Car) -> Car:
    return x(lambda a, d, set_car, set_cdr: set_car(y))

def replacd(x: Cons, y: U) -> U:
    return x(lambda a, d, set_car, set_cdr: set_cdr(y))

def cdar(x): return cdr(car(x))

def caar(x): return car(car(x))

def cadr(x): return car(cdr(x))

def cddr(x): return cdr(cdr(x))

def consp(x):
    return isa(cons)(x)
# consp = isa(cons)

def eq(x, y):
    return x is y

def equal(x, y):
    return x == y

def atom(x):
    return not consp(x)

def numberp(x):
    return isinstance(x, (int, float))

import inspect

def functionp(x):
   return inspect.isfunction(x) or inspect.isclass(x)

def quote(x):
    return x

def literal(x):
    if eq(x, nil):
        return True
    if eq(x, t):
        return True
    if numberp(x):
        return True
    if functionp(x):
        return True


def eval(exp, env=nil, procedures=nil):
    if atom(exp):
        if literal(exp):
            return exp
        else:
            return value(exp, env)
    elif car(exp) == "quote":
        return cadr(exp)
    elif car(exp) == "cond":
        return evcond(cdr(exp), env, procedures)
    else:
        return apply(value(car(exp), procedures),
                     evlis(cdr(exp), env, procedures),
                     procedures)

def apply(fun, args, procedures=nil):
    return fun(*args)
    # raise NotImplementedError

def evcond(clauses, env, procedures=nil):
    raise NotImplementedError

def evlis(arglist=nil, env=nil, procedures=nil):
    if null(arglist):
        return nil
    else:
        return cons(eval(car(arglist), env, procedures),
                    evlis(cdr(arglist), env, procedures))

def value(name, env=nil):
    if null(env):
        env = globals()
    return env.get(name, "&unbound")



if __name__ == '__main__':
    print(cons("foo", cons("bar", cons("baz", "omgz"))))


