from __future__ import annotations

import types
import weakref
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

def kwcall(f, args=nil, kws=nil):
    xs = tuple(*(args or ()))
    ks = dict(**(kws or {}))
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

KT: TypeAlias = "Union[str, int, SupportsHash]"
K = TypeVar("K", bound=KT)

@dispatch()
def y_each(h: T) -> Generator[Tuple[K, T]]:
    h = XCONS(h)
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

@y_each.register(std.Mapping)
def y_each_tab(h: Mapping[K, T]) -> Generator[Tuple[K, T]]:
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

@y_each.register(types.ModuleType)
@y_each.register(argparse.Namespace)
def y_each_ns(h: types.ModuleType) -> Generator[Tuple[K, T]]:
    return y_each(h.__dict__)

class InconsistentIteration(Error):
    pass

@dispatch()
def y_step(l):
    prev = 0
    for k, v in y_each(l):
        if integerp(k):
            if k < prev:
                return err("backwards-iteration")
            prev = k
            yield v

@dispatch()
def y_length(h, upto: Optional[int] = None):
    if upto is None:
        return 1 + max((k for k, v in y_each(h) if integerp(k)), default=-1)
    else:
        n = 0
        for n, v in enumerate(y_step(h)):
            if n >= upto:
                break
        return n

@y_length.register(std.Mapping)
def y_length_tab(h: Mapping, upto: Optional[int] = None):
    return 1 + max(filter(integerp, h.keys()), default=-1)

def y_none(h): return y_length(h, 0) == 0
def y_some(h): return y_length(h, 0) > 0
def y_many(h): return y_length(h, 1) > 1
def y_one(h): return y_length(h, 1) == 1
def y_two(h): return y_length(h, 2) == 2


import weakref

if 'refs' not in globals():
    refs = weakref.WeakValueDictionary[int, weakref.WeakKeyDictionary["Ref"]]()

LT: TypeAlias = Union[MutableMapping[K, T], MutableSequence[Union[K, T]], MutableSet[Union[K, T]]]
L = TypeVar("L", bound=LT)


@dataclasses.dataclass(eq=False)
class Ref(Generic[K, T], metaclass=ABCMeta):
    key: K
    default: T
    frames: Tuple[Union[MutableMapping[K, T], MutableSequence[Union[K, T]], MutableSet[Union[K, T]]], ...]
    FrameType: TypeAlias = "Tuple[Union[MutableMapping[K, T], MutableSequence[Union[K, T]], MutableSet[Union[K, T]]], ...]"
    def __post_init__(self):
        self.mark()
    @property
    def id(self):
        return py.tuple(py.id(l) for l in self.frames)
    def __hash__(self):
        return hash(self.id)
    @property
    def marks(self) -> Optional[weakref.WeakKeyDictionary[Ref[K, T]]]:
        self.marks_ = refs.setdefault(self.id, weakref.WeakKeyDictionary())
        return self.marks_
    def mark(self):
        self.marks[self] = self.frames
        assert self.marks[self] == self.frames
    def others(self):
        return py.list(self.marks.keys())
    def __repr__(self):
        return repr_call(nameof(py.type(self)), key=self.key, frames=self.frames, id=self.id)


def _check():
    vmark: List[str] = ["%vmark"]
    smark: List[str] = ["%smark"]
    # badmarks: List[Ref[int, str]] = [Ref(i, True, vmark) for i in range(10)]
    vmarks: List[Ref[int, str]] = [Ref(i, "", vmark) for i in range(10)]
    smarks: List[Ref[int, str]] = [Ref(i, "", smark) for i in range(10)]
    uu = vmarks[0].key
    uuv = vmarks[0].frames
    assert len(vmarks[0].others()) == 10
    del vmarks[-1]
    import gc
    gc.collect()
    assert len(vmarks[0].others()) == 9

_check()







# class Slot(Cons[KT, D]):
#     def __init__(self,
#                  key: KT,
#                  cdr: Optional[D] = unset,
#                  frozen: Optional[Literal[False, True, "car", "cdr"]] = False):
#         HasCar.__init__(self, car=car, frozen=frozen)
#         HasCdr.__init__(self, cdr=cdr, frozen=frozen)
#
#     def __init__(self, l, k: KT, ):

@dispatch()
def y_get(l, k: SupportsHash, test: Optional[Callable[KT, KT], Optional[bool]] = None):
    return Cell(l, k)

@y_get.register(Cons)
def y_get(l, k: KT, test: Optional[Callable[[KT, KT], Optional[bool]]] = None):
    if test:
        for k1, v in y_each(l):
            if yes(test(k, k1)):
                return v
    else:
        for k1, v in y_each(l):
            if k == k1:
                return v



def y_items(h: Container[T]) -> List[Tuple[str, T]]:
    return py.list((k, v) for k, v in y_each(h))

def y_keys(h: Container[T]) -> Dict[str, T]:
    return py.dict((k, v) for k, v in y_each(h) if not integerp(k))

def y_vals(h: Container[T]) -> List[T]:
    return py.list(v for k, v in y_each(h) if integerp(k))

def y_unzip(h) -> Tuple[List, Dict[str]]:
    return y_vals(h), y_keys(h)

def y_zip(args, kws) -> List:
    xs = cons2vec(args)
    for k, v in kws.items():
        xs.extend([keysym(k), v])
    return xs

def y_list(*args, **kws):
    return y_zip(args, kws)

def y_seq(l):
    args, kws = y_unzip(l)
    return y_zip(args, kws)

# def y_step(l):
#     for x in y_vals(l):
#         yield x
#
# def y_each(l):
#     for k, v in y_for(l):
#         yield i, x
#     for k, v in y_items(l):
#         if not numberp(k):
#             yield k, v

@nameof.register(Cons)
def nameof_cons(x):
    return repr(x)
