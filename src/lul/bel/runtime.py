from __future__ import annotations

import types
from typing import *

from ..common import *

import dataclasses
import functools
import collections.abc as std
import json

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

def delay(x):
    return lambda: x

def part(f, *args, **kws):
    return functools.wraps(f)(lambda *args1, **kws1: f(*args, *args1, **{**kws, **kws1}))

def thunk(f, *args, **kws):
    return functools.wraps(f)(lambda *_args, **_kws: f(*args, **kws))

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
        return quote("tab")
    elif modulep(x):
        return quote("mod")
    elif streamp(x):
        return quote("stream")
    elif char(x):
        return quote("char")
    else:
        return quote("symbol")

def xar(x, y):
    assert consp(x)
    x.car = y
    return y

def xdr(x, y):
    assert consp(x)
    x.cdr = y
    return y

def sym(x):
    if x == "t":
        return t
    elif x == "nil":
        return nil
    elif x == "o":
        return o
    # elif x == "apply":
    #     return apply
    else:
        return x

def nom(x):
    if x is t:
        return "t"
    elif x is nil:
        return "nil"
    elif x is o:
        return "o"
    # elif x is apply:
    #     return "apply"
    elif isinstance(x, str):
        return x if x.isprintable() and ' ' not in x else json.dumps(x)
    elif inspect.isfunction(x):
        return f"#'{nameof(x)}"
    else:
        return repr(x)

prrepr_[0] = nom

def quote(x: T) -> T:
    return sym(x)

@dispatch()
def update(out: MutableMapping, *args: Mapping, **kws):
    for x in args:
        if modulep(x):
            x = x.__dict__
        out.update(x)
    out.update(kws)
    return out

@update.register(types.ModuleType)
def update_module(out: types.ModuleType, *args, **kws):
    return update(out.__dict__, *args, **kws)

@update.register(py.type(None))
def update_module(out: None, *args, **kws):
    return update({}, *args, **kws)

def stash(args=nil, kwargs: Dict[str] = nil):
    # kwargs = kwargs or {}
    # args, kws = y_unzip(args)
    kws = kwargs or {}
    if kws:
        return XCONS([quote("lit"), quote("stash"), XCONS(args), kws])
    else:
        return XCONS(args)

def stashp(args):
    if eq(car(args), quote("lit")):
        if eq(car(cdr(args)), quote("stash")):
            return t

def unstash(*args) -> Tuple[Cons, Dict[str]]:
    xs = []
    kws = {}
    for arg in args:
        if stashp(arg):
            it = cdr(cdr(arg))
            xs1, ks1 = car(it), car(cdr(it))
            xs.extend(Cons.list(xs1))
            kws.update(py.dict(ks1))
        else:
            xs.extend(Cons.list(arg))
    # return args, {}
    return XCONS(xs), kws

def cons2vec(xs):
    if isinstance(xs, Cons):
        return xs.list()
    elif null(xs):
        return []
    return py.list(xs)

def applyargs(*args):
    xs = cons2vec(args[-1])
    return tuple(args[0:-1]) + tuple(xs)

def call(f, *args, **kws):
    return f(*args, **kws)

def kwcall(f, *args, **kws):
    args, keys = y_unzip(args)
    return call(f, *args, **kws, **keys)

def apply(f, *args, **kws):
    return call(f, *applyargs(*args), **kws)

def kwapply(f, args=None, kws: Dict[str] = None):
    return call(f, *applyargs(args or []), **(kws or {}))

# unset = join("%unset")
o = "o"

def err(x, *args):
    raise Error(x, *args)

def keyword(x):
    return stringp(x) and len(x) > 1 and x[0] == ":"

def char(x):
    return stringp(x) and len(x) > 1 and x.startswith("\\")

def keynom(x):
    assert keyword(x)
    return x[1:]

def keysym(x):
    if keyword(x):
        return x
    if stringp(x) and not string_literal_p(x):
        return ":" + x
    return x

# (defmacro y-%for (h k v &rest body)
#   (y-let-unique (i)
#     `(let* ((,i -1))
#        (while ,h
#          (let* ((,k (if (keywordp (car ,h)) (y-%key (car ,h)) (setq ,i (1+ ,i))))
#                 (,v (if (keywordp (car ,h)) (cadr ,h) (car ,h))))
#            ,@body)
#          (setq ,h (if (keywordp (car ,h)) (cddr ,h) (cdr ,h))))
#        nil)))
def y_for(h):
    i = -1
    while not null(h):
        if yes(keyword(car(h))):
            k = keynom(car(h))
            v = car(cdr(h))
            h = cdr(cdr(h))
        else:
            i += 1
            k = i
            v = car(h)
            h = cdr(h)
        yield k, v

def y_unzip1(h):
    xs = []
    kvs = {}
    i = -1
    for k, v in y_for(h):
        # if integerp(k) and k >= 0:
        #     if len(xs) <= k:
        #         xs.extend([nil] * ((k + 1) - len(xs)))
        #     xs[k] = v
        if integerp(k):
            i += 1
            assert k == i
            assert len(xs) == i
            xs.append(v)
        else:
            kvs[k] = v
    return xs, kvs

def y_items(h):
    return py.tuple((k, v) for k, v in y_for(h))

def y_keys(h) -> Dict:
    return py.dict((k, v) for k, v in y_for(h) if not integerp(k))

def y_vals(h) -> List:
    return py.list(v for k, v in y_for(h) if integerp(k))

def y_unzip(h) -> Tuple[List, Dict[str]]:
    return y_vals(h), y_keys(h)

def y_zip(args, kws) -> List:
    xs = cons2vec(args)
    for k, v in kws.items():
        xs.extend([keysym(k), v])
    return xs

@nameof.register(Cons)
def nameof_cons(x):
    return repr(x)
