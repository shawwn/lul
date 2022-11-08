from __future__ import annotations

from typing import *

from ..common import *

import dataclasses
import functools
import collections.abc as std

class NSMeta(type):
    def __missing__(self, name):
        return name
    def __getattr__(self, id: str):
        return self[uncompile_id(id)]
    def __getitem__(self, name: str):
        id = compile_id(name)
        try:
            return object.__getattribute__(self, id)
        except AttributeError:
            # x = intern(name)
            value = self.__missing__(name)
            setattr(self, id, value)
            return value


def id(x=nil, y=nil):
    if eq(x, y):
        return t
    else:
        return nil

def join(x=nil, y=nil):
    return Cons(x, y)

def type(x):
    if consp(x):
        return quote("pair")
    elif numberp(x):
        return quote("number")
    elif dictp(x):
        return quote("table")
    else:
        return quote("symbol")

@dispatch()
def xar(x, y):
    assert consp(x)
    x.car = y
    return y

@dispatch()
def xdr(x, y):
    assert consp(x)
    x.cdr = y
    return y

@xar.register(py.list)
def xar_list(x: list, y):
    x[0] = y
    return y

@xdr.register(py.list)
def xdr_list(x: list, y):
    x[:] = [car(x), ".", y]
    return y

def sym(x):
    # if x == "t":
    #     return t
    # elif x == "nil":
    #     return nil
    # elif x == "o":
    #     return o
    # elif x == "apply":
    #     return apply
    # elif x == "unset":
    #     return unset
    # else:
    #     return x
    return syms.get(x, x)

def nom(x):
    if x is t:
        return "t"
    elif x is nil:
        return "nil"
    elif x is o:
        return "o"
    elif x is apply:
        return "apply"
    elif x is unset:
        return "unset"
    else:
        return x

def quote(x):
    return sym(x)

def apply(f, *args, **kws):
    xs = args[-1]
    if isinstance(xs, Cons):
        # kvs = xs.dict()
        # kvs.update(kws)
        # kws = kvs
        xs = xs.tuple(props=True)
    elif xs is nil:
        xs  = ()
    args = tuple(args[0:-1]) + tuple(xs)
    return f(*args, **kws)

# globe = globals

o = "o"

syms = dict(t=t, nil=nil, o=o, apply=apply, unset=unset)

def err(x, *args, **kws):
    raise Error(x, *args, *([kws] if kws else []))

def if_f(*clauses: Callable[[], T], test=unset) -> Callable[[], T]:
    if test is unset:
        test = yes
    while len(clauses) >= 2:
        cond, cons, *clauses = clauses
        if test(cond()):
            return cons
    if clauses:
        return clauses[0]
    return lambda: nil
