from __future__ import annotations

import contextlib
import inspect
import io
import reprlib
import weakref

from .runtime import *
import json
import sys
from importlib import reload

import __main__ as G
M = sys.modules[__name__]


_R = TypeVar("_R")
_S = TypeVar("_S")
TCons = TypeVar("TCons", bound=Cons, covariant=True)

if TYPE_CHECKING:
    TExpr: TypeAlias = "T"
    TE: TypeAlias = "TExpr"
    # TG = Dict[str, T]
    # TEnvr = Union[Cons[str, T], TG]
    # TS = ConsList[Cons[TExpr,Cons[TEnvr, None]]]
    # TR = ConsList[TExpr]
    # TThread = Cons[TS, Cons[TR, None]]
    # TM = Cons[ConsList[TThread], TG]
    # class ConsList2(Cons[T0, Cons[T1, None]]): ...
    # List1 = Cons[T0, None]
    # List2 = Cons[T0, Cons[T1, None]]
    # List3 = Cons[T0, Cons[T1, Cons[T2, None]]]
    # List4 = Cons[T0, Cons[T1, Cons[T2, Cons[T3, None]]]]
    List1: TypeAlias = "Cons[T0, None]"
    List2: TypeAlias = "Cons[T0, Cons[T1, None]]"
    Rest2: TypeAlias = "Cons[T0, Cons[T1, T2]]"
    List3: TypeAlias = "Cons[T0, Cons[T1, Cons[T2, None]]]"
    List4: TypeAlias = "Cons[T0, Cons[T1, Cons[T2, Cons[T3, None]]]]"

    # class TG(Dict[str, T]): ...
    # class TEnvr(Tuple[Cons[str, T], ...], TG): ...
    # class TS(Tuple[List2[TExpr, TEnvr], ...]): ...
    # class TR(Tuple[TExpr, ...]): ...
    # class TP(List2[TS, TR]): ...
    # TP = List2[TS, TR]
    A = TypeVar("A")
    E = TypeVar("E")
    TCell: TypeAlias = "Union[Cell[str, T], Cons[str, T]]"
    TA: TypeAlias = "Tuple[TCell[T], ...]"
    TE_A: TypeAlias = "List2[E, TA[T]]"
    TS: TypeAlias = "Tuple[TE_A[E, T], ...]"
    TR: TypeAlias = "Tuple[T, ...]"
    TS_R: TypeAlias = "List2[TS[E, T], TR[T]]"
    TP: TypeAlias = "Tuple[TS_R[E, T], ...]"
    TG: TypeAlias = "Dict[str, T]"
    TM: TypeAlias = "List2[TP[E, T], TG[T]]"

@dispatch()
def signature(x):
    try:
        return inspect.signature(x), x
    except (TypeError, ValueError):
        return nil

def afut(x):
    return consp(x) and begins(car(x), list(smark, quote("fut")))

@signature.register(Cons)
def signature_Cons(x):
    if isa(quote("clo"))(x):
        return cadr(cddr(x)), x
    if isa(quote("mac"))(x):
        return signature(caddr(x))
    if afut(x):
        e, a = car(x), cadr(x)
        return signature(caddr(e))
    return nil

# def propget(f, k, default=nil):
#     return getattr(inspect.unwrap(f), k, default)
#
# def propset(f, k, v):
#     return [setattr(inspect.unwrap(f), k, v)] and v
#
# def prop(name, default=nil):
#     def prop_inner(f, *set, init=nil):
#         if set:
#             return propset(f, name, *set)
#         else:
#             return propget(f, name, default=default)
#     return rename(prop_inner, f"{nameof(prop)}!{nom(name)}")

# cache = prop("cache")

# @dispatch()
def memo(x):
    if not callable(x):
        return memo(rename(lambda: x, f"#<memo {nom(x)}>"))
    return functools.wraps(x)(functools.lru_cache(maxsize=None)(x))
    # def memoized(*args, **kws):
    #     get((args, kws), cache(memoized), equal)
    #     put((args, kws), result, getattr(memoized, "cache", nil))

def callback(x):
    return lambda: x

def then(x):
    return lambda: x()

# # @dispatch()
# def delay(x: Callable):
#     # out = memo(x)
#     return rename(memo(lambda: x()), f"#<delay {nom(x)}>")

@memo
def delayed(x):
    return callable(x) and str(signature(x)) == "()"

def force(x):
    if delayed(x):
        return x()
    return x

@car.register(py.list)
def hd(l: List):
    return l[0] if l else nil

@cdr.register(py.list)
def tl(l: List):
    return l[1:] or nil

@overload
def rename(name: str) -> Callable: ...

@overload
def rename(f: T, name: str) -> T: ...

@dispatch()
def rename(f, name):
    f.__qualname__ = f.__name__ = name
    return f

@rename.register(str)
def renaming(name: str):
    # @functools.wraps(rename)
    # def renamer(f):
    #     return rename(f, name)
    return rename(part(rename, name=name), f"[{nameof(rename)} _ '{nom(name)}]")

def compose(*fs):
    return reduce(compose2, fs or (idfn,))

def compose2(f, g):
    @functools.wraps(g)
    def f_then_g(*args, **kws):
        return f(apply(g, args, **kws))
    # f_name = getattr(f, "__qualname__", getattr(f, "__name__", "<unknown>"))
    # g_name = getattr(g, "__qualname__", getattr(g, "__name__", "<unknown>"))
    # f_name = nameof(f)
    # g_name = nameof(g)
    # f_then_g.__qualname__ = f_then_g.__name__ = f"{f_name}:{g_name}"
    return rename(f_then_g, f"{nameof(f)}:{nameof(g)}")


def vec2list(v: Optional[List[T]]) -> Optional[Union[T, ConsList[T]]]:
    if py.isinstance(v, (py.list, py.tuple)):
        if len(v) >= 3 and v[-2] == ".":
            l = vec2list(v[0:-2])
            xdr(lastcdr(l), vec2list(v[-1]))
            return l
        return list(*[vec2list(x) for x in v])
    return quote(v)

def list2vec(l: Cons) -> Optional[List]:
    if consp(l):
        out: List[Any] = [list2vec(x) for x in mkproper(l).list()]
        if no(proper(l)):
            assert len(out) >= 2
            out.insert(-1, ".")
        return out
    if null(l):
        return []
    return l

def cons2vec(xs: Union[T, Cons[A, D]]) -> Union[T, List[A]]:
    if isinstance(xs, Cons):
        return xs.list()
    elif null(xs):
        return []
    return xs

def vec2cons(xs: Iterable[A]) -> Optional[ConsList[A]]:
    if isinstance(xs, Cons):
        return xs
    elif null(xs):
        xs: None
        return xs
    return XCONS(xs)

# def no(x):
#     return id(x, nil) or falsep(x)

def init(x, y):
    return y if x is unset else x

def atom(x):
    return no(id(type(x), quote("pair")))

# def all(f, xs):
#     if no(xs):
#         return t
#     elif no(f(car(xs))):
#         return nil
#     else:
#         return all(f, cdr(xs))

def all(f, xs):
    while True:
        if no(xs):
            return t
        elif no(f(car(xs))):
            return nil
        else:
            xs = cdr(xs)

# def some(f, xs):
#     if no(xs):
#         return nil
#     elif no(f(car(xs))):
#         return some(f, cdr(xs))
#     else:
#         return xs

def some(f, xs):
    while True:
        if no(xs):
            return nil
        elif no(f(car(xs))):
            xs = cdr(xs)
        else:
            return xs

def reduce(f, xs):
    if no(cdr(xs)):
        return car(xs)
    else:
        return f(car(xs), reduce(f, cdr(xs)))

@overload
def cons(a: A, b: D) -> Cons[A, D]: ...
@overload
def cons(a: T0, b: T1, c: T2) -> Cons[T0, Cons[T1, T2]]: ...
@overload
def cons(a: T0, b: T1, c: T2, d: T3) -> Cons[T0, Cons[T1, Cons[T2, T3]]]: ...

def cons(*args):
    return reduce(join, args)

# (def append args
#   (if (no (cdr args)) (car args)
#       (no (car args)) (apply append (cdr args))
#                       (cons (car (car args))
#                             (apply append (cdr (car args))
#                                           (cdr args)))))
def append(*args):
    if no(cdr(args)):
        return car(args)
    elif no(car(args)):
        return apply(append, cdr(args))
    else:
        return cons(car(car(args)),
                    apply(append,
                          cdr(car(args)),
                          cdr(args)))

# (def snoc args
#   (append (car args) (cdr args)))
def snoc(*args):
    return append(car(args), cdr(args))

@overload
def list() -> None: ...
@overload
def list(a: T0) -> Cons[T0, None]: ...
@overload
def list(a: T0, b: T1) -> Cons[T0, Cons[T1, None]]: ...
@overload
def list(a: T0, b: T1, c: T2) -> Cons[T0, Cons[T1, Cons[T2, None]]]: ...
@overload
def list(a: T0, b: T1, c: T2, d: T3) -> Cons[T0, Cons[T1, Cons[T2, Cons[T3, None]]]]: ...
@overload
def list(a: T0, b: T1, c: T2, d: T3, e: T4) -> Cons[T0, Cons[T1, Cons[T2, Cons[T3, Cons[T4, None]]]]]: ...
@overload
def list(a: T0, b: T1, c: T2, d: T3, e: T4, f: T5) -> Cons[T0, Cons[T1, Cons[T2, Cons[T3, Cons[T4, Cons[T5, None]]]]]]: ...
@overload
def list(a: T0, b: T1, c: T2, d: T3, e: T4, f: T5, g: T6) -> Cons[T0, Cons[T1, Cons[T2, Cons[T3, Cons[T4, Cons[T5, Cons[T6, None]]]]]]]: ...
# @overload
# def list(*args: T) -> ConsList[T]: ...

# (def list args
#   (append args nil))
def list(*args, **kws):
    return append(XCONS(y_zip(args, kws)), nil)

assert cons2vec(list(1,2,3)) == [1,2,3]
assert cons2vec(append(list(1,2,3), list(4))) == [1,2,3,4]

# (def map (f . ls)
#   (if (no ls)       nil
#       (some no ls)  nil
#       (no (cdr ls)) (cons (f (car (car ls)))
#                           (map f (cdr (car ls))))
#                     (cons (apply f (map car ls))
#                           (apply map f (map cdr ls)))))
def map(f: Callable[[T], R], l: ConsList[T], *ls: ConsList[T]) -> Cons[R]:
    ls = (l, *ls)
    ls: Tuple[Cons[T], ...]
    if no(ls):
        return nil
    elif yes(some(no, ls)):
        return nil
    elif no(cdr(ls)):
        return cons(f(car(car(ls))),
                    map(f, cdr(car(ls))))
    else:
        ls: Cons[T]
        return cons(apply(f, map(car, ls)),
                    apply(map, f, map(cdr, ls)))

def macro_(f):
    return list(quote("lit"), quote("mac"), compose(vec2list, f))

# (mac fn (parms . body)
#   (if (no (cdr body))
#       `(list 'lit 'clo scope ',parms ',(car body))
#       `(list 'lit 'clo scope ',parms '(do ,@body))))
@macro_
def fn(parms, *body):
    if no(cdr(body)):
        return ["list", ["quote", "lit"], ["quote", "clo"], "scope", ["quote", parms], ["quote", car(body)]]
    else:
        return ["list", ["quote", "lit"], ["quote", "clo"], "scope", ["quote", parms], ["quote", ["do", *body]]]
    # if no(cdr(body)):
    #     return list(quote("list"), quote1("lit"), quote1("clo"), quote("scope"), quote1(parms), quote1(car(body)))
    # else:
    #     return list(quote("list"), quote1("lit"), quote1("clo"), quote("scope"), quote1(parms), list(quote("do"), *body))

# @macro_
# def fn(parms, *body):
#     if no(cdr(body)):
#         return list(quote("list"),
#                     list(quote("quote"), quote("lit")),
#                     list(quote("quote"), quote("clo")),
#                     quote("scope"),
#                     list(quote("quote"), parms),
#                     list(quote("quote"), car(body)))
#     else:
#         return list(quote("list"),
#                     list(quote("quote"), quote("lit")),
#                     list(quote("quote"), quote("clo")),
#                     quote("scope"),
#                     list(quote("quote"), parms),
#                     list(quote("quote"), list(quote("do"), *body)))

# (set vmark (join))
vmark = globals().get("vmark", join("%vmark"))

# uvars = globals().get("uvars", weakref.WeakKeyDictionary())
ucount = globals().get("ucount", [0])

class UVar(Cons):
    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        ucount[0] += 1
        self.n = ucount[0]
    @property
    def name(self):
        return cadr(self) or "uvar"
    def __repr__(self):
        return f"#{self.name}{self.n}"

# (def uvar ()
#   (list vmark))
def uvar(*name):
    # return list(vmark, *name)
    return UVar.new(list(vmark, *name))

# (mac do args
#   (reduce (fn (x y)
#             (list (list 'fn (uvar) y) x))
#           args))
# @macro_
# def do(*args):
#     return reduce(lambda x, y: list(list(quote("fn"), uvar("do"), y), x),
#                   args)

@macro_
def do(*args):
    return ["%do", *args]
    # return cons(quote("%do"), args)

def do_f(x, *args):
    for f in args:
        x = f(x)
    return x

# (mac let (parms val . body)
#   `((fn (,parms) ,@body) ,val))
@macro_
def let(parms, val, *body):
    return ["%let", parms, val, *body]
    # return [["fn", [parms], *body], val]
    # return list(list(quote("fn"), list(parms), *body), val)

# (mac macro args
#   `(list 'lit 'mac (fn ,@args)))
@macro_
def macro(*args):
    return ["list", ["quote", "lit"], ["quote", "mac"], ["fn", *args]]
    # return list(quote("list"),
    #             list(quote("quote"), quote("lit")),
    #             list(quote("quote"), quote("mac")),
    #             list(quote("fn"), *args))


# (mac def (n . rest)
#   `(set ,n (fn ,@rest)))
@macro_
def def_(n, *rest):
    return ["set", n, ["fn", *rest]]
    # return list(quote("set"), n, list(quote("fn"), *rest))
globals()["def"] = def_

# (mac mac (n . rest)
#   `(set ,n (macro ,@rest)))
@macro_
def mac(n, *rest):
    return ["set", n, ["macro", *rest]]
    # return list(quote("set"), n, list(quote("macro"), *rest))

# (mac or args
#   (if (no args)
#       nil
#       (let v (uvar)
#         `(let ,v ,(car args)
#            (if ,v ,v (or ,@(cdr args)))))))
@macro_
def or_(*args):
    if no(args):
        return nil
    v = uvar("or")
    return ["let", v, car(args),
            ["if", v, v, ["or", *(cdr(args) or [])]]]
globals()["or"] = or_

def or_f(*args):
    for f in args:
        x = f()
        if yes(x):
            return x
    return nil

# (mac and args
#   (reduce (fn es (cons 'if es))
#           (or args '(t))))
@macro_
def and_(*args):
    return reduce(lambda *es: ["if", *es],
                  args or [t])
globals()["and"] = and_

def and_f(*args):
    if no(args):
        return t
    else:
        x = nil
        for f in args:
            x = f()
            if no(x):
                return nil
        return x

# (def = args
#   (if (no (cdr args))  t
#       (some atom args) (all [id _ (car args)] (cdr args))
#                        (and (apply = (map car args))
#                             (apply = (map cdr args)))))
def equal(*args):
    if no(cdr(args)):
        return t
    elif no(some(atom, args)):
        return and_f(lambda: apply(equal, map(car, args)),
                     lambda: apply(equal, map(cdr, args)))
    else:
        return all(lambda _: id(_, car(args)), cdr(args))

# (def symbol (x) (= (type x) 'symbol))
def symbol(x):
    return equal(type(x), quote("symbol"))

# (def pair   (x) (= (type x) 'pair))
def pair(x):
    return equal(type(x), quote("pair"))

# (def char   (x) (= (type x) 'char))
def char(x):
    return equal(type(x), quote("char"))

# (def stream (x) (= (type x) 'stream))
def stream(x):
    return equal(type(x), quote("stream"))

def number(x):
    return equal(type(x), quote("number"))

# (def proper (x)
#   (or (no x)
#       (and (pair x) (proper (cdr x)))))
def proper(x):
    return or_f(lambda: no(x),
                lambda: and_f(lambda: pair(x),
                              lambda: proper(cdr(x))))

def mkproper(x) -> Cons:
    return if_f(lambda: null(x),
                lambda: x,
                lambda: pair(x),
                lambda: cons(car(x), mkproper(cdr(x))),
                lambda: list(x))()

# (def string (x)
#   (and (proper x) (all char x)))
def string(x):
    if not proper(x):
        return nil
    else:
        return all(char, x)


# (def mem (x ys (o f =))
#   (some [f _ x] ys))
def mem(x, ys, f=unset):
    if f is unset:
        f = equal
    return some(lambda _: f(_, x), ys)

# (def in (x . ys)
#   (mem x ys))
def in_(x, *ys):
    return mem(x, ys)

@overload
def cadr(x: Tuple[T, ...]) -> T: ...
@overload
def cadr(x: List2[T0, T1]) -> T1: ...
# (def cadr  (x) (car (cdr x)))
# def cadr(x: TCons[T0, TCons[T1, None]]) -> T1:
def cadr(x):
    return car(cdr(x))

@overload
def cddr(x: Tuple[T, ...]) -> T: ...
@overload
def cddr(x: Rest2[T0, T1, T2]) -> T2: ...

# (def cddr  (x) (cdr (cdr x)))
def cddr(x):
    return cdr(cdr(x))

# (def caddr (x) (car (cddr x)))
def caddr(x: TCons[T0, TCons[T1, TCons[T2, T3]]]) -> T2:
    return car(cddr(x))


# (mac case (expr . args)
#   (if (no (cdr args))
#       (car args)
#       (let v (uvar)
#         `(let ,v ,expr
#            (if (= ,v ',(car args))
#                ,(cadr args)
#                (case ,v ,@(cddr args)))))))
@macro_
def case(expr, *args):
    if no(cdr(args)):
        return car(args)
    else:
        v = uvar("case_v")
        return ["let", v, expr,
                ["if", ["=", v, ["quote", car(args)]],
                 cadr(args),
                 ["case", v, *cons2vec(cddr(args))]]]
        # return list(quote("let"), v, expr,
        #             list(quote("if"),
        #                  list(quote("equal"), v, list(quote("quote"), car(args))),
        #                  cadr(args),
        #                  list(quote("case"), v, *args[2:])))


def case_f(expr, *args) -> Callable[[], Any]:
    if no(cdr(args)):
        if no(car(args)):
            return lambda: nil
        else:
            return car(args)
    else:
        if equal(expr, car(args)):
            return cadr(args)
        else:
            return apply(case_f, expr, cddr(args))

# (mac iflet (var . args)
#   (if (no (cdr args))
#       (car args)
#       (let v (uvar)
#         `(let ,v ,(car args)
#            (if ,v
#                (let ,var ,v ,(cadr args))
#                (iflet ,var ,@(cddr args)))))))
@macro_
def iflet(var, *args):
    """
    (mac iflet (var . args)
      (if (no (cdr args))
          (car args)
          (let v (uvar)
            `(let ,v ,(car args)
               (%if ,v
                   (let ,var ,v ,(cadr args))
                   (iflet ,var ,@(cddr args)))))))"""
    if no(cdr(args)):
        return car(args)
    if not consp(var):
        return ["let", var, car(args),
                ["%if", var, cadr(args),
                 ["iflet", var, *(cddr(args) or ())]]]
    else:
        v = uvar("iflet_v")
        return ["let", v, car(args),
                ["%if", v,
                 ["let", var, v, cadr(args)],
                 ["iflet", var, *(cddr(args) or ())]]]

# (mac aif args
#   `(iflet it ,@args))
@macro_
def aif(*args):
    return ["iflet", "it", *args]


# (def find (f xs)
#   (aif (some f xs) (car it)))
def find(f, xs):
    if yes(it := some(f, xs)):
        return car(it)
    else:
        return nil

# (def begins (xs pat (o f =))
#   (if (no pat)               t
#       (atom xs)              nil
#       (f (car xs) (car pat)) (begins (cdr xs) (cdr pat) f)
#                              nil))
def begins(xs, pat, f=unset):
    if f is unset:
        f = equal
    if no(pat):
        return t
    elif atom(xs):
        return nil
    elif yes(f(car(xs), car(pat))):
        return begins(cdr(xs), cdr(pat), f)
    else:
        return nil

# (def caris (x y (o f =))
#   (begins x (list y) f))
def caris(x, y, f=unset):
    if f is unset:
        f = equal
    return begins(x, list(y), f)

# (def hug (xs (o f list))
#   (if (no xs)       nil
#       (no (cdr xs)) (list (f (car xs)))
#                     (cons (f (car xs) (cadr xs))
#                           (hug (cddr xs) f))))
def hug(xs, f=unset):
    if f is unset:
        f = list
    if no(xs):
        return nil
    elif no(cdr(xs)):
        return list(f(car(xs)))
    else:
        return cons(f(car(xs), cadr(xs)),
                    hug(cddr(xs), f))

# (mac with (parms . body)
#   (let ps (hug parms)
#     `((fn ,(map car ps) ,@body)
#       ,@(map cadr ps))))
@macro_
def with_(parms, *body):
    ps = hug(parms)
    return [["fn", map(car, ps), *body],
            *cons2vec(map(cadr, ps))]
globals()["with"] = with_

# (def keep (f xs)
#   (if (no xs)      nil
#       (f (car xs)) (cons (car xs) (keep f (cdr xs)))
#                    (keep f (cdr xs))))
def keep(f, xs):
    if no(xs):
        return nil
    elif yes(f(car(xs))):
        return cons(car(xs), keep(f, cdr(xs)))
    else:
        return keep(f, cdr(xs))

# (def rem (x ys (o f =))
#   (keep [no (f _ x)] ys))
def rem(x, ys, f=unset):
    if f is unset:
        f = equal
    return keep(lambda _: no(f(_, x)), ys)

# (def get (k kvs (o f =))
#   (find [f (car _) k] kvs))
# def get(k, kvs, f=unset):
#     if f is unset:
#         f = equal
#     if null(kvs):
#         return kvs
#     if isinstance(kvs, Cons):
#         # return find(lambda _: f(car(_), k), kvs)
#         for cell in kvs.values():
#             name, value = car(cell), cdr(cell)
#             if value is unset:
#                 continue
#             if f(name, k):
#                 return cell
#         # if dictp(it := cdr(kvs[-1])):
#         #     kvs = it
#         # elif modulep(it):
#         #     kvs = it
#         else:
#             return nil
#     if modulep(kvs):
#         kvs = kvs.__dict__
#     assert f in [equal, id]
#     out = Cell(kvs, k, unset)
#     if cdr(out) is unset:
#         return nil
#     return out

def get(k, kvs, f=unset, new=nil):
    return locate(k, kvs, new, f)

def alistp(kvs):
    return consp(kvs) and consp(car(kvs))

def locatable(kvs):
    return dictp(kvs) or modulep(kvs)

def locate(k, kvs, new=nil, f=unset):
    if f is unset:
        f = equal
    if null(kvs):
        return kvs
    if isinstance(kvs, Cons):
        for cell in kvs.values():
            if locatable(cell):
                if yes(it := locate(k, cell, new, f)):
                    return it
            elif consp(cell):
                name, value = car(cell), cdr(cell)
                if yes(f(k, name)):
                    if value is unset:
                        if yes(new):
                            xdr(cell, nil)
                        else:
                            continue
                    return cell
    else:
        if yes(new) and kvs in [py, M]:
            return nil
        cell = Cell(kvs, k, unset)
        if cdr(cell) is unset:
            if yes(new):
                xdr(cell, nil)
            else:
                return nil
        return cell




# (def put (k v kvs (o f =))
#   (cons (cons k v)
#         (rem k kvs (fn (x y) (f (car x) y)))))
def put(k, v, kvs, f=unset):
    if f is unset:
        f = equal
    return cons(cons(k, v),
                rem(k, kvs, lambda x, y: f(car(x), y)))

# (def rev (xs)
#   (if (no xs)
#       nil
#       (snoc (rev (cdr xs)) (car xs))))
def rev(xs):
    if no(xs):
        return nil
    else:
        return snoc(rev(cdr(xs)), car(xs))

# (def snap (xs ys (o acc))
#   (if (no xs)
#       (list acc ys)
#       (snap (cdr xs) (cdr ys) (snoc acc (car ys)))))
def snap(xs, ys, acc=nil) -> Cons:
    if no(xs):
        return list(acc, ys)
    else:
        return snap(cdr(xs), cdr(ys), snoc(acc, car(ys)))

# (def udrop (xs ys)
#   (cadr (snap xs ys)))
def udrop(xs, ys):
    return cadr(snap(xs, ys))

# (def idfn (x)
#   x)
def idfn(x):
    return x

# (def is (x)
#   [= _ x])
def is_(x: Callable[[], Any]):
    return lambda _: equal(_, x())

# (mac eif (var (o expr) (o fail) (o ok))
#   (with (v (uvar)
#          w (uvar)
#          c (uvar))
#     `(let ,v (join)
#        (let ,w (ccc (fn (,c)
#                       (dyn err [,c (cons ,v _)] ,expr)))
#          (if (caris ,w ,v id)
#              (let ,var (cdr ,w) ,fail)
#              (let ,var ,w ,ok))))))
@macro_
def eif(var, expr=nil, fail=nil, ok=nil):
    v = uvar("eif_v")
    w = uvar("eif_w")
    c = uvar("eif_c")
    return ["let", v, ["join", ["quote", "%eif"]],
            ["let", w, ["ccc", ["fn", [c],
                                ["dyn", "err", ["fn", ["_"], [c, ["cons", v, "_"]]], expr]]],
             ["if", ["caris", w, v, "id"],
              ["let", var, ["cdr", w], fail],
              ["let", var, w, ok]]]]
    # return list(quote("let"), v, list(quote("join")),
    #             list(quote("let"), w, list(quote("ccc"), list(quote("fn"), list(c),
    #                                                           list(quote("dyn"), quote("err"),
    #                                                                list(quote("fn"), list(quote("_")),
    #                                                                     list(c,
    #                                                                          quote("cons"), v, quote("_"))),
    #                                                                expr))),
    #                  list(quote("if"), list(quote("caris"), w, v, quote("id")),
    #                       list(quote("let"), var, list(quote("cdr"), w), fail),
    #                       list(quote("let"), var, w, ok))))



# # (mac fn (parms . body)
# #   (if (no (cdr body))
# #       `(list 'lit 'clo scope ',parms ',(car body))
# #       `(list 'lit 'clo scope ',parms '(do ,@body))))
# @macro_
# def fn(parms, *body):
#     if no(cdr(body)):
#         return list(quote("list"),
#                     list(quote("quote"), quote("lit")),
#                     list(quote("quote"), quote("clo")),
#                     quote("scope"),
#                     list(quote("quote"), parms),
#                     list(quote("quote"), car(body)))
#     else:
#         return list(quote("list"),
#                     list(quote("quote"), quote("lit")),
#                     list(quote("quote"), quote("clo")),
#                     quote("scope"),
#                     list(quote("quote"), parms),
#                     list(quote("quote"), list(quote("do"), *body)))


#
# (mac onerr (e1 e2)
#   (let v (uvar)
#     `(eif ,v ,e2 ,e1 ,v)))
@macro_
def onerr(e1, e2):
    v = uvar("onerr_v")
    return ["eif", v, e2, e1, v]

# (mac safe (expr)
#   `(onerr nil ,expr))
@macro_
def safe(expr):
    return ["onerr", "nil", expr]

# (def literal (e)
#   (or (in e t nil o apply)
#       (in (type e) 'char 'stream)
#       (caris e 'lit)
#       (string e)))
def literal(e):
    # if in_(e, t, nil, o, apply):
    if in_(e, t, nil, o):
        return t
    elif in_(type(e), quote("char"), quote("stream"), quote("tab"), quote("char"), quote("mod")):
        return t
    elif caris(e, quote("lit")):
        return t
    elif number(e):
        return t
    elif consp(e):
        return nil
    elif callable(e):
        return t
    elif string_literal_p(e):
        return t
    elif keyword(e):
        return True
    else:
        return string(e)

namecs = globals().setdefault("namecs", dict(
    bel="\a",
    tab="\t",
    lf="\n",
    cr="\r",
    sp=" "))

def evliteral(e):
    # if string_literal_p(e) and reader.read_from_string(e, more=object())[0] == e:
    if string_literal_p(e):
        return json.loads(e)
    if char(e):
        e: str
        name = e[1:]
        return force(namecs.get(name, name))
    return e

# (def variable (e)
#   (if (atom e)
#       (no (literal e))
#       (id (car e) vmark)))
def variable(e):
    if atom(e):
        return no(literal(e))
    else:
        return id(car(e), vmark)

# (def isa (name)
#   [begins _ `(lit ,name) id])
def isa(name):
    return lambda _: or_f(lambda: begins(_, list(quote("lit"), name), id),
                          lambda: equal(type(_), name))

# (def bel (e (o g globe))
#   (ev (list (list e nil))
#       nil
#       (list nil g)))
def bel(e, g=unset, a=unset, p=unset):
    # breakpoint()
    p_g = jump.p_g if (jump := mev_tail.get()) else nil
    s = jump.s if jump else nil
    r = jump.r if jump else nil
    if a is unset:
        if jump is None:
            # a = nil
            # a = XCONS(G.__dict__)
            # a = G.__dict__
            a = G
        else:
            e_a = car(jump.s)
            a = cadr(e_a)
    if p is unset:
        p = car(p_g)
    if g is unset:
        if jump is None:
            # g = globe()
            # g = nil
            # g = {**py.__dict__, **globals()}
            # g = XCONS(globals())
            # g = append(XCONS(G.__dict__), XCONS(globals()))
            # g = globals()
            g = M
        else:
            g = cadr(p_g)
    return tev(cons(list(e, a), s),
               r,
               list(p, g))

def bel(e, g=unset, a=unset, p=unset):
    s = nil
    r = nil
    if jump := mev_tail.get():
        p_g = jump.p_g
        if g is unset:
            g = cadr(p_g)
        if a is unset:
            a = jump.lexenv
        if p is unset:
            p = car(p_g)
    else:
        if g is unset:
            g = list(G, M, py)
        if a is unset:
            # a = list(G)
            a = nil
        if p is unset:
            p = nil
    return tev(cons(list(e, a), s),
               r,
               list(p, g))

# (def mev (s r (p g))
#   (if (no s)
#       (if p
#           (sched p g)
#           (car r))
#       (sched (if (cdr (binding 'lock s))
#                  (cons (list s r) p)
#                  (snoc p (list s r)))
#              g)))
def mev_(s, r, p_g):
    p, g = car(p_g), cadr(p_g)
    if no(s) and (no(p) or no(cdr(binding(quote("main"), s)))):
        return if_f(lambda: p,
                    lambda: ([ero("discard", r), breakpoint()] and sched(p, g)),
                    lambda: ([ero("leftover", it) if yes(it := cdr(r)) else (ero("return", car(r)) and breakpoint())] and car(r)))()
    else:
        return sched(schedule(s, r, p), g)

def schedule(s, r, p):
    return if_f(lambda: cdr(binding(quote("lock"), s)),
                lambda: cons(list(s, r), p),
                lambda: snoc(p, list(s, r)))()

"""
Each thread is a list 

(s r)

of two stacks: a stack s of expressions to be evaluated, and a stack 
r of return values.

Each element of s is in turn a list

(e a)

where e is an expression to be evaluated, and a is a lexical
environment consisting of a list of (var . val) pairs.

The variable p holds a list of all the threads (usually other than 
the current one).

The other thing we need to pass around in the interpreter is the
global bindings, which is another environment represented as a list 
of (var . val) pairs. I use the variable g for this.

The most common parameter list we'll see is 

(s r m)

where s is the current expression stack, r is the current return 
value stack, and m is a list (p g) of the other threads and the 
global bindings.
"""

# BelThreadCar: TypeAlias = "ConsList[BelExpression]"
TBelThread: TypeAlias = "Union[List2[ConsList[TBelExpression], ConsList[T]], BelThread]"

class BelThread(Cons):
    """Each thread is a list

    (s r)

    of two stacks: a stack s of expressions to be evaluated, and a stack
    r of return values.
    """
    car: ConsList[BelExpression]
    if TYPE_CHECKING:
        @classmethod
        def new(cls, src: Cons[A, D]) -> BelThread: ...
    # def __init__(self):
    #     super().__init__(car=nil, cdr=list(nil))
    @property
    def s(self: TBelThread) -> ConsList[BelExpression]:
        return map(BelExpression.new, car(self))
    @property
    def r(self: TBelThread) -> ConsList[T]:
        return cadr(self)
    @property
    def lexenv(self):
        for expr in cons2vec(self.s): # type: BelExpression
            if ok(it := expr.a):
                return it
    @reprlib.recursive_repr()
    def __repr__(self):
        # return f"BelThread(s={self.s!r}, r={self.r!r})"
        return repr_self(self, ("s", None), ("r", None), ("lexenv", None))

TBelExpression: TypeAlias = "Union[List2[T, TA], BelExpression]"

class BelExpression(Cons):
    @property
    def e(self) -> T:
        return car(self)
    @property
    def a(self) -> TA:
        return cadr(self)
    @reprlib.recursive_repr()
    def __repr__(self):
        # return f"BelExpression(e={self.e!r})"
        # return repr_self(self, ("e", None), ("a", None))
        # return repr_self(self, ("e", None))
        with with_indent(2):
            return "\n" + indentation() + prrepr(self.e)


TBelThreads: TypeAlias = "Union[List2[ConsList[TBelThread], TG], BelThreads]"

class BelThreads(Cons):
    """m is a list (p g) of the other threads and the
    global bindings."""
    @property
    def p(self) -> ConsList[BelThread]:
        return map(BelThread.new, car(self))
    @property
    def g(self) -> TG:
        return cadr(self)


def _check():
    p_g: TBelThreads = nil
    # car(p_g.p)
    # car(p_g.p).s
    p, g = car(p_g), cadr(p_g)
    # s_r: TBelThread = nil
    s_r = car(p)
    s_r.s.car.a
    car(s_r.s)
    s, r = car(s_r), cadr(s_r)
    e_a = car(s)
    e, a = car(e_a), cadr(e_a)


def current():
    return mev_tail.get()

class JumpToMev(Exception):
    def __init__(self, s: TS, r: TR, p_g: TBelThreads, prev: Optional[JumpToMev]):
        self.s = s
        self.r = r
        self.p_g = p_g
        self.prev = prev

    @property
    def thread(self) -> BelThread:
        return BelThread.new(cons(self.s, self.r))

    @property
    def expr(self) -> BelExpression:
        for expr in self.thread.s or []: # type: BelExpression
            return expr

    @property
    def lexenv(self):
        for expr in self.thread.s or []: # type: BelExpression
            if yes(expr.a):
                return expr.a

    @reprlib.recursive_repr()
    def __repr__(self):
        return repr_self(self, ("lexenv", None), ("thread", None), ("prev", None))

if 'mev_tail' not in globals():
    mev_tail = CV.ContextVar[JumpToMev]("mev_tail", default=None)

def frame(e: TE = unset, a: TA = unset, s: TS = unset, r: TR = unset, m: TM = unset, p: TP = unset, g: TG = unset):
    jump = mev_tail.get()
    if e is unset:
        e = jump.expr if jump else nil
    if a is unset:
        a = jump.lexenv if jump else M
    if s is unset:
        s = jump.s if jump else list(list(e, a))
    if r is unset:
        r = jump.r if jump else nil
    # if m is unset:
    #     m = jump.p_g if jump else nil
    # if jump is None:
    #     if s is unset:
    #         s = nil
    #     if r is unset:
    #         r = nil
    #     if p is unset:
    #         assert m is unset, "Can't speecify p and m"
    #     if g is unset:
    #         assert m is unset, "Can't speecify p and m"
    #     if m is unset:
    #         assert p is unset and g is unset
    #         # m = list(nil,

# def let_mev(s, r, p_g):
#     JumpToMev(s, r, p_g)
#     pass

# def tev(s, r, p_g, prev=unset):
#     breakpoint()
#     while True:
#         if prev is unset:
#             prev = nil
#         print("was", mev_tail.get())
#         reset = mev_tail.set(JumpToMev(s, r, p_g, prev))
#         # def unwind():
#         #     nonlocal reset
#         #     if (it := reset) is not None:
#         #         reset = None
#         #         mev_tail.reset(it)
#         try:
#             # while True:
#             #     try:
#             #         return mev_(s, r, p_g)
#             #     except JumpToMev as e:
#             #         mev_tail.set(jump := e)
#             #         s = jump.s
#             #         r = jump.r
#             #         p_g = jump.p_g
#             return mev_(s, r, p_g)
#         except JumpToMev as e:
#             s = e.s
#             r = e.r
#             p_g = e.p_g
#             prev = e.prev
#         finally:
#             mev_tail.reset(reset)

def tev(s, r, p_g, prev=unset):
    if prev is unset:
        prev = mev_tail.get()
    reset = mev_tail.set(JumpToMev(s, r, p_g, prev))
    prev = [prev]
    try:
        while True:
            # print("was", prev)
            try:
                return mev_(s, r, p_g)
            except JumpToMev as e:
                # mev_tail.reset(reset)
                mev_tail.set(jump := e)
                s = jump.s
                r = jump.r
                p_g = jump.p_g
                # prev = jump.prev
                prev.insert(0, jump.prev)
                # print("then", map(car, s), r)
    finally:
        mev_tail.reset(reset)

def mev(s, r, p_g):
    if prev := mev_tail.get():
        raise JumpToMev(s, r, p_g, prev)
        # return tev(s, r, p_g, prev)
    else:
        return tev(s, r, p_g, nil)

# (def sched (((s r) . p) g)
#   (ev s r (list p g)))
def sched(sr_p, g):
    s_r, p = car(sr_p), cdr(sr_p)
    s, r = car(s_r), cadr(s_r)
    return ev(s, r, list(p, g))

def syntaxp(x):
    if isinstance(x, str):
        if not string_literal_p(x):
            if x.startswith("~") and len(x) > 1:
                return t
            # if x.startswith("!") and len(x) > 1:
            #     return t
            if x.startswith(".") and len(x) > 1:
                return t
            for c in [":", "|", "!", "."]:
                if not (x.endswith(c) or x.startswith(c)):
                    if c in x:
                        return t
    return nil

def evsyntax(x, recurse=nil, self=unset, was=unset):
    def go(e, recurse=recurse, self=self, was=x):
        if equal(recurse, quote("once")):
            recurse = nil
        return evsyntax(e, recurse=recurse, self=self, was=was)
    if not syntaxp(x):
        if recurse and consp(x):
            return map(go, x)
        if self is not unset:
            if atom(self):
                if stringp(self):
                    return ":" + self
                return err("nonstring-atom", self, x)
            return list
        return quote(x)
    x: str
    if len(x) > 0 and x[-1] in ":|!.":
        return go(x[:-1]) + x[-1]
    if x.startswith("."):
        rest = go(x[1:])
        if stringp(rest):
            return ":" + rest
        # (.foo.bar x 2) => (:foo (:bar
    if ":" in x and "!" not in x and "|" not in x:
        lh, _, rh = x.partition(":")
        return list(quote("compose"), go(lh), go(rh))
    if x.startswith("!") and len(x) > 1:
        return list(quote("upon"), go(x[1:]))
    if "!" in x and "|" not in x:
        name, _, arg = x.rpartition("!")
        return list(go(name), list(quote("quote"), quote(arg)))
    if "|" in x:
        name, _, test = x.rpartition("|")
        return list(quote("t"), quote(name), go(test))
    if x.startswith("~") and len(x) > 1:
        return list(quote("compose"), quote("no"), go(x[1:]))
    assert False, "Bad syntax"

# (def ev (((e a) . s) r m)
#   (aif (literal e)            (mev s (cons e r) m)
#        (variable e)           (vref e a s r m)
#        (no (proper e))        (sigerr 'malformed s r m)
#        (get (car e) forms id) ((cdr it) (cdr e) a s r m)
#                               (evcall e a s r m)))
def ev(ea_s, r, m):
    e_a, s = car(ea_s), cdr(ea_s)
    e, a = car(e_a), cadr(e_a)
    e = evsyntax(e)
    if yes(literal(e)):
        return mev(s, cons(evliteral(e), r), m)
    elif yes(variable(e)):
        return vref(e, a, s, r, m)
    elif no(proper(e)):
        return sigerr(quote("malformed"), s, r, m)
    elif yes(it := get(car(e), forms, id, car(inwhere(s)))):
        return cdr(it)(cdr(e), a, s, r, m)
    else:
        return evcall(e, a, s, r, m)

# (def vref (v a s r m)
#   (let g (cadr m)
#     (if (inwhere s)
#         (aif (or (lookup v a s g)
#                  (and (car (inwhere s))
#                       (let cell (cons v nil)
#                         (xdr g (cons cell (cdr g)))
#                         cell)))
#              (mev (cdr s) (cons (list it 'd) r) m)
#              (sigerr 'unbound s r m))
#         (aif (lookup v a s g)
#              (mev s (cons (cdr it) r) m)
#              (sigerr (list 'unboundb v) s r m)))))
def vref(k, a, s, r, m):
    p, g = car(m), cadr(m)
    if yes(inwhere(s)):
        if yes(it := or_f(lambda: lookup(k, a, s, r, p, g),
                          lambda: and_f(lambda: car(inwhere(s)),
                                        lambda: xset(k, nil, g)))):
            return mev(cdr(s), cons(list(it, quote("d")), r), m)
        else:
            if yes(it := lookup(k, a, s, r, p, g)):
                return mev(s, cons(cdr(it), r), m)
            else:
                return sigerr(list(quote("unboundb"), k), s, r, m)
    else:
        if yes(it := lookup(k, a, s, r, p, g)):
            return mev(s, cons(cdr(it), r), m)
        else:
            return sigerr(list(quote("unboundb"), k), s, r, m)

def assign(where, v):
    assert not null(where)
    cell, loc = car(where), cadr(where)
    return case_f(loc,
                  quote("a"), lambda: xar(cell, v),
                  quote("d"), lambda: xdr(cell, v),
                  lambda: err(quote("cannot-assign"), loc, cell))()

# def xset(k, v, kvs):
#     assert not null(kvs)
#     if consp(kvs):
#         cell = cons(k, v)
#         xdr(kvs, cons(cell, cdr(kvs)))
#     else:
#         cell = Cell(kvs, k, nil)
#         xdr(cell, v)
#     return cell

def xset(k, v, kvs):
    if yes(cell := locate(k, kvs, quote("new"))):
        return assign(list(cell, quote("d")), v)
    elif consp(kvs):
        cell = cons(k, v)
        xdr(kvs, cons(cell, cdr(kvs)))
    else:
        return err(quote("cannot-set"), k)

# (set smark (join))
smark = globals().get("smark", join("%smark"))

# (def inwhere (s)
#   (let e (car (car s))
#     (and (begins e (list smark 'loc))
#          (cddr e))))
def inwhere(s):
    e = car(car(s))
    return and_f(lambda: begins(e, list(smark, quote("loc"))),
                 lambda: cddr(e))

# (def lookup (e a s g)
#   (or (binding e s)
#       (get e a id)
#       (get e g id)
#       (case e
#         scope (cons e a)
#         globe (cons e g))))
def lookup(e, a, s, r, p, g):
    return or_f(lambda: binding(e, s),
                lambda: get(e, a, id, car(inwhere(s))),
                lambda: get(e, g, id, car(inwhere(s))),
                lambda: case_f(e,
                               quote("scope"), lambda: cons(e, a),
                               quote("stack"), lambda: cons(e, s),
                               quote("outer"), lambda: cons(e, r),
                               quote("other"), lambda: cons(e, p),
                               quote("globe"), lambda: cons(e, g))())

# (def binding (v s)
#   (get v
#        (map caddr (keep [begins _ (list smark 'bind) id]
#                         (map car s)))
#        id))
def binding(k, s):
    return get(k,
               map(caddr, keep(lambda _: begins(_, list(smark, quote("bind")), id),
                               map(car, s))),
               id,
               inwhere(s))

# (def sigerr (msg s r m)
#   (aif (binding 'err s)
#        (applyf (cdr it) (list msg) nil s r m)
#        (err 'no-err)))
def sigerr(msg, s, r, m):
    # print("sigerr", msg)
    if yes(it := binding(quote("err"), s)):
        return applyf(cdr(it), list(msg), nil, s, r, m)
    else:
        if isinstance(msg, Exception):
            raise msg
        else:
            return err(quote("no-err"), msg)


# TS = TypeVar("TS", bound=Cons)
# TR = TypeVar("TR", bound=Cons)
# TG = TypeVar("TG", bound=Dict[str, Any])
# TP = TypeVar("TP", bound=Cons)
# TM = TypeVar("TM", bound=Cons[TG, TP])


def _check():
    # a: TA[T] = nil
    # e_a: TE_A = nil
    p_g: TM = nil
    p, g = car(p_g), cadr(p_g)
    s_r = car(p)
    s, r = car(s_r), cadr(s_r)
    e_a = car(s)
    e, a = car(e_a), cadr(e_a)
    cell = car(a)
    k, v = car(cell), cdr(cell)



    # class TM(Cons[Tuple[TP, ...], Cons[TG, None]]): ...
    # TM = Union[TM, Cons[ConsList[TP], Cons[TG, None]]]
    # TM = List2[Tuple[TP, ...], TG]
    # TM: TypeAlias = "List2[TP, TG]"

    zz: List2[int, str] = list(42, "b")
    car(zz)
    cdr(zz)
    car(cdr(zz))
    cadr(zz)
    cddr(zz)

    # @overload
    # def car(x: TM) -> ConsList[TP]: ...
    # @overload
    # def cdr(x: TM) -> Cons[TG, None]: ...
    # @overload
    # def cadr(x: TM) -> TG: ...

    # m: TM
    # m.car.car.cdr.car
    # car(cdr(car(m)))
    # p, g = car(m), car(cdr(m))
    # # s, r = car(car(p)), cadr(car(p))
    # s_r: TThread = car(p)
    # # s, r = car(s_r), car(cdr(s_r))
    # s, r = car(s_r), cadr(s_r)
    # car(s), cadr(s)
    # cadr(list(1,2))

import ansi_styles
fgcol = ansi_styles.ansiStyles.color.ansi16m
fgclo = ansi_styles.ansiStyles.color.close
bgcol = ansi_styles.ansiStyles.bgColor.ansi16m
bgclo = ansi_styles.ansiStyles.bgColor.close

def color(r: int, g: int, b: int):
    return r, g, b

def gray(v: int):
    return color(v, v, v)

sand = color(246, 246, 239)
site_color = color(180, 180, 180)
border_color = color(180, 180, 180)
textgray = gray(130)
noob_color = color(60, 150, 60)
linkblue = color(0, 0, 190)
orange   = color(255, 102, 0)
darkred  = color(180, 0, 0)
darkblue = color(0, 0, 120)

def prs(*args,
        fg: Optional[Tuple[int, int, int]] = None,
        bg: Optional[Tuple[int, int, int]] = None,
        sep = " ",
        end = "\n",
        flush = False,
        file: io.IOBase = None):
    if ok(fg) and fg is not True: print(end=apply(fgcol, fg), file=file)
    if ok(bg) and bg is not True: print(end=apply(bgcol, bg), file=file)
    if ok(fg): end = fgclo + end
    if ok(bg): end = bgclo + end
    print(*(arg for arg in args if ok(arg)), sep=sep, end=end, flush=flush, file=file)
    return last(args)

prn = rename("prn")(part(prs, sep=""))
pr = rename("pr")(part(prn, end="", flush=True))

ero = rename("ero")(part(print, file=sys.stderr))

def with_tostring(f):
    @functools.wraps(f)
    def captured(*args, **kws):
        return call_with_stdout_to_string(lambda: f(*args, **kws))
    return captured

def highlighter(fg, *, sep=" ", **kws):
    return with_tostring(part(pr, sep=sep, fg=fg, **kws))

hl = rename("hl")(highlighter(orange))
hl2 = rename("hl2")(highlighter(noob_color))


@macro_
def fontcolor(color, *body):
    return ["do",
            ["pr", ["apply", "fgcol", color]],
            ["after", ["do", *body],
             ["pr", "fgclo"]]]

def prcall(f, *args, **kws):
    if ok(f):
        ero(indentation() + repr_call(f, **{str(i): v for i, v in enumerate(args)}, **kws))
    return with_indent(4 if ok(f) else 0)

# def trace(f):
#     cons
#     sig = inspect.signature(f)
#     def inner(*args, **kws):
#         it = sig.bind(*args, **kws)
#         it.args
#
# @contextlib.contextmanager
# def prncall(f, *args, **kws):
#     if ok(f):
#         prcall(f, *args, **kws)
#     with with_indent(2):
#         yield

def almost(l):
    return rev(cdr(rev(l)))




class Fut(Cons):
    def __init__(self, *args, doc=nil, **kws):
        super().__init__(*args, **kws)
        self.doc = doc
    @property
    def f(self):
        return caddr(self)
    @property
    def name(self):
        return nom(self.f)
    @reprlib.recursive_repr()
    def __repr__(self):
        if self.doc:
            return f"#<fut {hl(self.name)} {hl2(prrepr(self.doc()))}>"
        else:
            return f"#<fut {hl(self.name)}>"


"""
(mac fu args
  `(list (list smark 'fut (fn ,@args)) nil))
"""

# def fu(f):
#     return list(list(smark, quote("fut"), f), nil)
def fu(a: Cons, *, doc=nil):
    def fut(f: Callable[[TS, TR, TM], Any]):
        return BelExpression.new(list(Fut.new(list(smark, quote("fut"), f), doc=doc), a))
    return fut

# (def evmark (e a s r m)
#   (case (car e)
#     fut  ((cadr e) s r m)
#     bind (mev s r m)
#     loc  (sigerr 'unfindable s r m)
#     prot (mev (cons (list (cadr e) a)
#                     (fu (s r m) (mev s (cdr r) m))
#                     s)
#               r
#               m)
#          (sigerr 'unknown-mark s r m)))
def evmark(e, a, s, r, m):
    return case_f(
        car(e),
        quote("fut"), (lambda: cadr(e)(s, r, m)),
        quote("bind"), (lambda: mev(s, r, m)),
        quote("loc"), (lambda: sigerr(quote("unfindable"), s, r, m)),
        quote("prot"), (lambda: mev(cons(list(cadr(e), a),
                                         fu(a)(lambda s, r, m: mev(s, cdr(r), m)),
                                         s),
                                    r,
                                    m)),
        lambda: sigerr(quote("unknown-mark"), s, r, m))()

# (set forms (list (cons smark evmark)))
forms = list(cons(smark, evmark))

# (mac form (name parms . body)
#   `(set forms (put ',name ,(formfn parms body) forms)))
#
# (def formfn (parms body)
#   (with (v  (uvar)
#          w  (uvar)
#          ps (parameters (car parms)))
#     `(fn ,v
#        (eif ,w (apply (fn ,(car parms) (list ,@ps))
#                       (car ,v))
#                (apply sigerr 'bad-form (cddr ,v))
#                (let ,ps ,w
#                  (let ,(cdr parms) (cdr ,v) ,@body))))))

def form_(name):
    def formsetter(f):
        global forms
        forms = put(name, f, forms)
        return f
    return formsetter


# (def parameters (p)
#   (if (no p)           nil
#       (variable p)     (list p)
#       (atom p)         (err 'bad-parm)
#       (in (car p) t o) (parameters (cadr p))
#                        (append (parameters (car p))
#                                (parameters (cdr p)))))
#
# (form quote ((e) a s r m)
#   (mev s (cons e r) m))
@form_("quote")
def quote_(es, _a, s, r, m):
    e = car(es)
    return mev(s, cons(e, r), m)

# (form if (es a s r m)
#   (if (no es)
#       (mev s (cons nil r) m)
#       (mev (cons (list (car es) a)
#                  (if (cdr es)
#                      (cons (fu (s r m)
#                              (if2 (cdr es) a s r m))
#                            s)
#                      s))
#            r
#            m)))
@form_("if")
def if_(es, a, s, r, m, test=unset, form=unset):
    if form is unset:
        form = quote("if")
    if no(es):
        return mev(s, cons(nil, r), m)
    else:
        fut = fu(a, doc=lambda: cons(form, es))
        return mev(cons(list(car(es), a),
                        if_f(lambda: cdr(es),
                             lambda: cons(fut((lambda s, r, m:
                                               if2(form, cdr(es), a, s, r, m, test=test))),
                                          s),
                             lambda: s)()),
                   r,
                   m)

@form_("%do")
def do_form(es, a, s, r, m):
    fut = fu(a, doc=lambda: cons("%do", es))
    return mev(append(map(lambda e: list(e, a), es),
                      list(fut(lambda s, r, m: do_form1(es, s, r, m))),
                      s),
               r,
               m)

def do_form1(es, s, r, m):
    v = car(r)
    r2 = cadr(snap(es, r))
    return mev(s, cons(v, r2), m)

@form_("%call")
def call_form(es, a, s, r, m):
    f, args = car(es), cdr(es)
    args, kws = y_unzip(args)
    return mev(append(map(lambda e: list(e, a), es),
                      s),
               r,
               m)

@form_("%if")
def if_form(es, a, s, r, m):
    return if_(es, a, s, r, m, test=ok, form=quote("%if"))

# (def if2 (es a s r m)
#   (mev (cons (list (if (car r)
#                        (car es)
#                        (cons 'if (cdr es)))
#                    a)
#              s)
#        (cdr r)
#        m))
def if2(f, es, a, s, r, m, test=unset):
    return mev(cons(list(if_f(lambda: car(r),
                              lambda: car(es),
                              lambda: cons(f, cdr(es)),
                              test=test)(),
                         a),
                    s),
               cdr(r),
               m)

def if_f(*clauses: Callable[[], Any], test=unset) -> Callable[[], Any]:
    if test is unset:
        test = yes
    while len(clauses) >= 2:
        cond, cons, *clauses = clauses
        if test(cond()):
            return cons
    if clauses:
        return clauses[0]
    return lambda: nil

def aif_f(*clauses: Tuple[Callable[[], _R], Callable[[_R], Any]]):
    for clause in clauses:
        cond, cons = clause
        if yes(it := cond()):
            return cons(it)
    return nil


# (form where ((e (o new)) a s r m)
#   (mev (cons (list e a)
#              (list (list smark 'loc new) nil)
#              s)
#        r
#        m))
@form_("where")
def where(es, a, s, r, m):
    e, new = car(es), cadr(es)
    return mev(cons(list(e, a),
                    list(list(smark, quote("loc"), new),
                         a),
                    s),
               r,
               m)

# (form dyn ((v e1 e2) a s r m)
#   (if (variable v)
#       (mev (cons (list e1 a)
#                  (fu (s r m) (dyn2 v e2 a s r m))
#                  s)
#            r
#            m)
#       (sigerr 'cannot-bind s r m)))
@form_("dyn")
def dyn(es, a, s, r, m):
    fut = fu(a, doc=lambda: cons("dyn", es))
    v, e1, e2 = car(es), cadr(es), caddr(es)
    if yes(variable(v)):
        return mev(cons(list(e1, a),
                        fut(lambda s, r, m: dyn2(v, e2, a, s, r, m)),
                        s),
                   r,
                   m)
    else:
        return sigerr(quote("cannot-bind"), s, r, m)

# (def dyn2 (v e2 a s r m)
#   (mev (cons (list e2 a)
#              (list (list smark 'bind (cons v (car r)))
#                    nil)
#              s)
#        (cdr r)
#        m))
def dyn2(v, e2, a, s, r, m):
    return mev(cons(list(e2, a),
                    list(list(smark, quote("bind"), cons(v, car(r))),
                         a),
                    s),
               snoc(cdr(r), list(smark, quote("bind"), cons(v, car(r)))),
               m)

# (form after ((e1 e2) a s r m)
#   (mev (cons (list e1 a)
#              (list (list smark 'prot e2) a)
#              s)
#        r
#        m))
@form_("after")
def after(es, a, s, r, m):
    e1, e2 = car(es), cadr(es)
    return mev(cons(list(e1, a),
                    list(list(smark, quote("prot"), e2), a),
                    s),
               r,
               m)

# (form ccc ((f) a s r m)
#   (mev (cons (list (list f (list 'lit 'cont s r))
#                    a)
#              s)
#        r
#        m))
@form_("ccc")
def ccc(es, a, s, r, m):
    f = car(es)
    return mev(cons(list(list(f, list(quote("lit"), quote("cont"), s, r)),
                         a),
                    s),
               r,
               m)

# (form thread ((e) a s r (p g))
#   (mev s
#        (cons nil r)
#        (list (cons (list (list (list e a))
#                          nil)
#                    p)
#              g)))
@form_("thread")
def thread(es, a, s, r, m):
    e = cons("do", es) if yes(cdr(es)) else car(es)
    p, g = car(m), cadr(m)
    # breakpoint()
    return mev(s,
               cons(nil, r),
               list(cons(list(list(list(e, a)),
                              nil),
                         p),
                    g))

# @form_("assign")
# def assign(es, a, s, r, m):


# (def evcall (e a s r m)
#   (mev (cons (list (car e) a)
#              (fu (s r m)
#                (evcall2 (cdr e) a s r m))
#              s)
#        r
#        m))
def evcall(e, a, s, r, m):
    return mev(cons(list(car(e), a),
                    fu(a, doc=lambda: e)(lambda s, r, m: evcall2(cdr(e), a, s, r, m)),
                    s),
               r,
               m)

# (def evcall2 (es a s (op . r) m)
#   (if ((isa 'mac) op)
#       (applym op es a s r m)
#       (mev (append (map [list _ a] es)
#                    (cons (fu (s r m)
#                            (let (args r2) (snap es r)
#                              (applyf op (rev args) a s r2 m)))
#                          s))
#            r
#            m)))
def evcall2(es, a, s, op_r, m):
    op, r = car(op_r), cdr(op_r)
    if isa(quote("mac"))(op):
        return applym(op, es, a, s, r, m)
    else:
        # args, kws = belunzip(es)
        # keys = [k for k in kws.keys()]
        # vals = [kws[k] for k in kws.keys()]
        xs, ks, vs = belunzip(es)
        @fu(a, doc=lambda: cons(op, es))
        def f(s, r, m):
            vals, r2 = car(it := snap(vs, r)), cadr(it)
            args, r2 = car(it := snap(xs, r2)), cadr(it)
            args = rev(args)
            vals = rev(vals)
            kws = py.dict(zip(cons2vec(ks), cons2vec(vals)))
            args2 = stash(args, kws) if kws else args
            with prcall("%call" if kws else nil, args=args, keys=ks, vals=vals, kws=kws, argv=args2):
                # args, r2 = car(it), cadr(it)
                return applyf(op, args2, a, s, r2, m)
        return mev(append(map(lambda _: list(_, a), append(xs, vs)),
                          cons(f,
                               s)),
                   r,
                   m)

def belunzip(es) -> Tuple[Cons, ConsList[str], Cons]:
    args, kws = y_unzip(es)
    keys = [k for k in kws.keys()]
    vals = [kws[k] for k in kws.keys()]
    return XCONS(args), XCONS(keys), XCONS(vals)

# (def applym (mac args a s r m)
#   (applyf (caddr mac)
#           args
#           a
#           (cons (fu (s r m)
#                   (mev (cons (list (car r) a) s)
#                        (cdr r)
#                        m))
#                 s)
#           r
#           m))
def applym(mac, args, a, s, r, m):
    return applyf(caddr(mac),
                  args,
                  a,
                  cons(fu(a, doc=lambda: cons(caddr(mac), args))((
                      lambda s, r, m: mev(cons(list(car(r), a), s),
                                          cdr(r),
                                          m))),
                       s),
                  r,
                  m)

# (def applyf (f args a s r m)
#   (if (= f apply)    (applyf (car args) (reduce join (cdr args)) a s r m)
#       (caris f 'lit) (if (proper f)
#                          (applylit f args a s r m)
#                          (sigerr 'bad-lit s r m))
#                      (sigerr 'cannot-apply s r m)))
def applyf(f, args, a, s, r, m):
    # if equal(f, apply):
    #     return applyf(car(args), reduce(join, cdr(args)), a, s, r, m)
    # else:
    return applylit(f, args, a, s, r, m)
    # elif caris(f, quote("lit")):
    #     if proper(f):
    #         return applylit(f, args, a, s, r, m)
    #     else:
    #         return sigerr(list(quote("bad-lit"), f, args), s, r, m)
    # elif callable(f):
    #     return applylit(f, args, a, s, r, m)
    #     # return applyfunc(f, args, a, s, r, m)
    # else:
    #     return sigerr(list(quote("cannot-apply"), f, args), s, r, m)

def applyfunc(f, args, kws, a, s, r, m):
    try:
        v = apply(f, args, **kws)
    except RecursionError:
        raise
    except Exception as v:
        return sigerr(v, s, r, m)
    return mev(s, cons(v, r), m)

# def applyfunc(f, args, a, s, r, m):
#     print("applythen", a, f, args)
#     @fu(a)
#     def f(s, r, m):
#         try:
#             v = apply(f, args)
#         except Exception as v:
#             return sigerr(v, s, r, m)
#         return mev(s, cons(v, r), m)
#         # return mev(cons(list(list(quote("quote"), v), a),
#         #                 s),
#         #            r,
#         #            m)
#     return mev(cons(f, s

# def applyfunc(f, args, a, s, r, m):
#     @fu(a)
#     def then(s, r, m):
#         try:
#             v = apply(f, args)
#         except Exception as v:
#             return sigerr(v, s, r, m)
#         return mev(s, cons(v, r), m)
#     # print("applyfunc", f, args, a)
#     return mev(cons(list(nil, a), then, s), r, m)
#     # map(lambda e_a: list(car(e_a), cons(a, cdr(e_a))), s)
#     # return mev(cons(fu((lambda s, r, m:
#     #                     evcall2(args, a, s, r, m))),
#     #                 s),
#     #            cons(v, r),
#     #            m)


# (def applylit (f args a s r m)
#   (aif (and (inwhere s) (find [(car _) f] locfns))
#        ((cadr it) f args a s r m)
#        (let (tag . rest) (cdr f)
#          (case tag
#            prim (applyprim (car rest) args s r m)
#            clo  (let ((o env) (o parms) (o body) . extra) rest
#                   (if (and (okenv env) (okparms parms))
#                       (applyclo parms args env body s r m)
#                       (sigerr 'bad-clo s r m)))
#            mac  (applym f (map [list 'quote _] args) a s r m)
#            cont (let ((o s2) (o r2) . extra) rest
#                   (if (and (okstack s2) (proper r2))
#                       (applycont s2 r2 args s r m)
#                       (sigerr 'bad-cont s r m)))
#                 (aif (get tag virfns)
#                      (let e ((cdr it) f (map [list 'quote _] args))
#                        (mev (cons (list e a) s) r m))
#                      (sigerr 'unapplyable s r m))))))
def applylit(f, args, a, s, r, m):
    args, kws = unstash(args)
    if yes(it := and_f(lambda: [ero("inwhere", that) if yes(that := inwhere(s)) else nil] and that,
                       lambda: find(lambda _: ero("locfn", _, f, args) or car(_)(f), locfns))):
        return cadr(it)(f, args, a, s, r, m)
    elif callable(f) and not consp(f):
        return applyfunc(f, args, kws, a, s, r, m)
    else:
        if caris(f, quote("lit")):
            tag, rest = cadr(f), cddr(f)
        else:
            tag, rest = type(f), f
        def do_prim():
            return applyprim(car(rest), args, s, r, m)
        def do_clo():
            env, parms, body, extra = car(rest), cadr(rest), caddr(rest), cdr(cddr(rest))
            if yes(okenv(env)) and yes(okparms(parms)):
                return applyclo(parms, args, kws, env, body, s, r, m)
            else:
                return sigerr(list(quote("bad-clo"), parms), s, r, m)
        def do_mac():
            return applym(f, map(lambda _: list(quote("quote"), _), args), a, s, r, m)
        def do_cont():
            s2, r2, extra = car(rest), cadr(rest), cddr(rest)
            if yes(okstack(s2)) and yes(proper(r2)):
                return applycont(s2, r2, args, s, r, m)
            else:
                return sigerr(quote("bad-cont"), s, r, m)
        def do_virfns():
            # if yes(it := get(tag, virfns, unset, car(inwhere(s)))):
            #     e = cdr(it)(f, map(lambda _: list(quote("quote"), _), args))
            #     return mev(cons(list(e, a), s), r, m)
            # else:
            #     return sigerr(quote("unapplyable"), s, r, m)
            return aif_f((lambda: apply(vircall, car(inwhere(s)), tag, f, args),
                          lambda it: mev(cons(list(it, a), s), r, m)),
                         (lambda: t,
                          lambda it: sigerr(list(quote("unapplyable"), tag, f, args), s, r, m)))
        return case_f(
            tag,
            quote("prim"), do_prim,
            quote("clo"), do_clo,
            quote("mac"), do_mac,
            quote("cont"), do_cont,
            do_virfns)()

def vircall(new, tag, f, *args, **kws):
    if yes(it := get(tag, virfns, unset, new)):
        return apply(cdr(it), f, map(lambda _: list(quote("quote"), _), args), **kws)


# (set virfns nil)
virfns = nil

# (mac vir (tag . rest)
#   `(set virfns (put ',tag (fn ,@rest) virfns)))

def vir_(tag):
    def vir_setter(f):
        global virfns
        virfns = put(tag, f, virfns)
        return f
    return vir_setter


# (set locfns nil)
locfns = nil

# (mac loc (test . rest)
#   `(set locfns (cons (list ,test (fn ,@rest)) locfns)))

def loc_(test):
    def loc_setter(f):
        global locfns
        locfns = cons(list(test, f), locfns)
        return f
    return loc_setter

def setter(test):
    def inner(setter):
        @loc_(test)
        def loc_inner(_f, args, _a, s, r, m):
            it = setter(args, car(inwhere(s)))
            # return mev(cdr(s), cons(list(car(args), quote("a")), r), m)
            return mev(cdr(s), cons(it, r), m)
        return loc_inner
    return inner

def whereloc(k, kvs, new=nil):
    if yes(it := locate(k, kvs, new)):
        return list(it, quote("d"))

@setter(is_(lambda: cadr))
def loc_is_cadr(args, new):
    ero(args)
    breakpoint()
    return list(car(args), quote("a"))

@setter(is_(lambda: tabref))
def loc_is_tabref(args, new):
    kvs, k = car(args), cadr(args)
    return whereloc(k, kvs, new)

# (loc (is car) (f args a s r m)
#   (mev (cdr s) (cons (list (car args) 'a) r) m))
@loc_(is_(lambda: car))
def loc_is_car(_f, args, _a, s, r, m):
    return mev(cdr(s), cons(list(car(args), quote("a")), r), m)

# (loc (is cdr) (f args a s r m)
#   (mev (cdr s) (cons (list (car args) 'd) r) m))
@loc_(is_(lambda: cdr))
def loc_is_cdr(_f, args, _a, s, r, m):
    return mev(cdr(s), cons(list(car(args), quote("d")), r), m)

@loc_(print)
def loc_print(_f, args, _a, s, r, m):
    assert False
    # return mev(cdr(s), cons(list(car(args), quote("d")), r), m)

@loc_(is_(lambda: get))
def loc_is_get(_f, args, _a, s, r, m):
    # ero("loc_is_get", _f, args, car(s), car(r))
    k, kvs = car(args), cadr(args)
    cell = get(k, kvs, unset, car(inwhere(s)))
    if null(cell):
        if yes(car(inwhere(s))):
            cell = xset(k, nil, kvs)
    if yes(cell):
        return mev(cdr(s), cons(list(cell, quote("d")), r), m)
    else:
        return sigerr(list(quote("cannot-get"), k, kvs), s, r, m)


# (def okenv (a)
#   (and (proper a) (all pair a)))
def okenv(_a):
    # return yes(proper(a)) and all(pair, a)
    return True

# (def okstack (s)
#   (and (proper s)
#        (all [and (proper _) (cdr _) (okenv (cadr _))]
#             s)))
def okstack(s):
    return and_f(lambda: proper(s),
                 lambda: all(lambda _: and_f(lambda: proper(_),
                                             lambda: cdr(_),
                                             lambda: okenv(cadr(_))),
                             s))

# (def okparms (p)
#   (if (no p)       t
#       (variable p) t
#       (atom p)     nil
#       (caris p t)  (oktoparm p)
#                    (and (if (caris (car p) o)
#                             (oktoparm (car p))
#                             (okparms (car p)))
#                         (okparms (cdr p)))))
def okparms(p):
    if no(p):
        return t
    elif yes(variable(p)):
        return t
    elif yes(keyword(p)):
        return t
    elif yes(atom(p)):
        return nil
    elif yes(caris(p, t)):
        return oktoparm(p)
    else:
        return and_f(lambda: if_f(lambda: caris(car(p), o),
                                  lambda: oktoparm(car(p)),
                                  lambda: okparms(car(p)))(),
                     lambda: okparms(cdr(p)))

# (def oktoparm ((tag (o var) (o e) . extra))
#   (and (okparms var) (or (= tag o) e) (no extra)))
def oktoparm(x):
    tag, var, e, extra = car(x), cadr(x), caddr(x), cdr(cddr(x))
    return and_f(lambda: okparms(var),
                 lambda: or_f(lambda: equal(tag, o),
                              lambda: e),
                 lambda: no(extra))

# (set prims '((id join xar xdr wrb ops)
#              (car cdr type sym nom rdb cls stat sys)
#              (coin)))
prims = list(list(quote("id"), quote("join"), quote("xar"), quote("xdr"), quote("xrb"), quote("ops"), quote("sys")),
             list(quote("car"), quote("cdr"), quote("type"), quote("sym"), quote("nom"), quote("rdb"), quote("cls"), quote("stat"), quote("print")),
             list(quote("coin")))

# (def applyprim (f args s r m)
#   (aif (some [mem f _] prims)
#        (if (udrop (cdr it) args)
#            (sigerr 'overargs s r m)
#            (with (a (car args)
#                   b (cadr args))
#              (eif v (case f
#                       id   (id a b)
#                       join (join a b)
#                       car  (car a)
#                       cdr  (cdr a)
#                       type (type a)
#                       xar  (xar a b)
#                       xdr  (xdr a b)
#                       sym  (sym a)
#                       nom  (nom a)
#                       wrb  (wrb a b)
#                       rdb  (rdb a)
#                       ops  (ops a b)
#                       cls  (cls a)
#                       stat (stat a)
#                       coin (coin)
#                       sys  (sys a))
#                     (sigerr v s r m)
#                     (mev s (cons v r) m))))
#        (sigerr 'unknown-prim s r m)))
def applyprim(f, args, s, r, m):
    if yes(it := some(lambda _: mem(f, _), prims)):
        if yes(udrop(cdr(it), args)):
            return sigerr(quote("overargs"), s, r, m)
        else:
            a = car(args)
            b = cadr(args)
            try:
                v = case_f(f,
                           quote("id"), lambda: id(a, b),
                           quote("join"), lambda: join(a, b),
                           quote("car"), lambda: car(a),
                           quote("cdr"), lambda: cdr(a),
                           quote("type"), lambda: type(a),
                           quote("xar"), lambda: xar(a, b),
                           quote("xdr"), lambda: xdr(a, b),
                           quote("sym"), lambda: sym(a),
                           quote("nom"), lambda: nom(a),
                           quote("wrb"), lambda: wrb(a, b),
                           quote("rdb"), lambda: rdb(a),
                           quote("ops"), lambda: ops(a, b),
                           quote("cls"), lambda: cls(a),
                           quote("stat"), lambda: stat(a),
                           quote("coin"), lambda: coin(),
                           quote("shell"), lambda: shell(a, b),
                           quote("print"), lambda: print(a),
                           lambda: sigerr(quote("bad-prim"), s, r, m))()
            except RecursionError:
                raise
            except Exception as v:
                return sigerr(v, s, r, m)
            return mev(s, cons(v, r), m)
    else:
        return sigerr(quote("unknown-prim"), s, r, m)

# (def applyclo (parms args env body s r m)
#   (mev (cons (fu (s r m)
#                (pass parms args env s r m))
#              (fu (s r m)
#                (mev (cons (list body (car r)) s)
#                     (cdr r)
#                     m))
#              s)
#        r
#        m))
def applyclo(parms, args, kws, env, body, s, r, m):
    def fut(doc):
        return fu(env, doc=lambda: list(f"applyclo{doc}", parms, args, kws, env, body))
    # fut = fu(env, doc=lambda: list(":parms", parms, ":args", args, ":kws", kws, ":env", env, ":body", body))
    return mev(cons(fut("#1")(lambda s, r, m: pass_(parms, args, kws, env, s, r, m)),
                    fut("#2")(lambda s, r, m: mev(cons(list(body, car(r)), s),
                                            cdr(r),
                                            m)),
                    s),
               r,
               m)

@form_("%let")
def let_form(es, a, s, r, m):
    var, val = car(es), cadr(es)
    body = macroexp_do(cddr(es))
    fut = fu(a, doc=lambda: cons("%let", es))
    return mev(cons(list(val, a),
                    fut(lambda s, r, m: applyclo(var, car(r), {}, a, body, s, cdr(r), m)),
                    s),
               r,
               m)

# (def pass (pat arg env s r m)
#   (let ret [mev s (cons _ r) m]
#     (if (no pat)       (if arg
#                            (sigerr 'overargs s r m)
#                            (ret env))
#         (literal pat)  (sigerr 'literal-parm s r m)
#         (variable pat) (ret (cons (cons pat arg) env))
#         (caris pat t)  (typecheck (cdr pat) arg env s r m)
#         (caris pat o)  (pass (cadr pat) arg env s r m)
#                        (destructure pat arg env s r m))))
def pass_(pat, arg, kws, env, s, r, m):
    # prn("pass", pat, arg)
    with prcall("pass_" and nil, pat=pat, arg=arg, kws=kws):
        pat = evsyntax(pat) # for foo|int forms
        def ret(_):
            return mev(s, cons(_, r), m)
        if no(pat):
            if yes(arg):
                return sigerr(quote("overargs"), s, r, m)
            else:
                return ret(env)
        elif yes(literal(pat)):
            return sigerr(quote("literal-parm"), s, r, m)
        elif yes(variable(pat)):
            return ret(cons(cons(pat, arg), env))
        elif yes(caris(pat, t)):
            return typecheck(cdr(pat), arg, kws, env, s, r, m)
        elif yes(caris(pat, o)):
            return pass_(cadr(pat), arg, kws, env, s, r, m)
        elif yes(keyword(car(pat))):
            return pass_(cdr(pat), arg, kws, env, s, r, m)
        else:
            return destructure(pat, arg, kws, env, s, r, m)

# (def typecheck ((var f) arg env s r m)
#   (mev (cons (list (list f (list 'quote arg)) env)
#              (fu (s r m)
#                (if (car r)
#                    (pass var arg env s (cdr r) m)
#                    (sigerr 'mistype s r m)))
#              s)
#        r
#        m))
def typecheck(var_f, arg, kws, env, s, r, m):
    var, f = car(var_f), cadr(var_f)
    return mev(cons(list(list(f, list(quote("quote"), arg)), env),
                    fu(env)((lambda s, r, m:
                             if_f(lambda: car(r),
                                  lambda: pass_(var, arg, kws, env, s, cdr(r), m),
                                  lambda: sigerr(quote("mistype"), s, r, m))())),
                    s),
               r,
               m)

# (def destructure ((p . ps) arg env s r m)
#   (if (no arg)   (if (caris p o)
#                      (mev (cons (list (caddr p) env)
#                                 (fu (s r m)
#                                   (pass (cadr p) (car r) env s (cdr r) m))
#                                 (fu (s r m)
#                                   (pass ps nil (car r) s (cdr r) m))
#                                 s)
#                           r
#                           m)
#                      (sigerr 'underargs s r m))
#       (atom arg) (sigerr 'atom-arg s r m)
#                  (mev (cons (fu (s r m)
#                               (pass p (car arg) env s r m))
#                             (fu (s r m)
#                               (pass ps (cdr arg) (car r) s (cdr r) m))
#                             s)
#                       r
#                       m)))
def destructure(p_ps, arg, kws, env, s, r, m):
    with prcall("destructure" and nil, p_ps=p_ps, arg=arg, kws=kws, env=env):
        # fut = fu(env, doc=lambda: list("destructure", p_ps, arg, kws, ":env", env))
        def fut(doc):
            return fu(env, doc=lambda: list(f"destructure{doc}", p_ps, arg, ":kws", kws))
        # fut = fu(env, doc=lambda: list("destructure", p_ps, arg, kws, env))
        # fut = fu(env, doc=lambda: repr_call("", p_ps=p_ps, arg=arg, kws=kws, env=env))
        p, ps = car(p_ps), cdr(p_ps)
        if no(arg):
            if yes(caris(p, o)):
                return mev(cons(list(caddr(p), env),
                                fut("(o)#1")(lambda s, r, m: pass_(cadr(p), car(r), kws, env, s, cdr(r), m)),
                                fut("(o)#2")(lambda s, r, m: pass_(ps, nil, kws, car(r), s, cdr(r), m)),
                                s),
                           r,
                           m)
            else:
                return sigerr(quote("underargs"), s, r, m)
        elif yes(atom(arg)):
            return sigerr(quote("atom-arg"), s, r, m)
        else:
            return mev(cons(fut("#1")(lambda s, r, m: pass_(p, car(arg), kws, env, s, r, m)),
                            fut("#2")(lambda s, r, m: pass_(ps, cdr(arg), kws, car(r), s, cdr(r), m)),
                            s),
                       r,
                       m)

# (def applycont (s2 r2 args s r m)
#   (if (or (no args) (cdr args))
#       (sigerr 'wrong-no-args s r m)
#       (mev (append (keep [and (protected _) (no (mem _ s2 id))]
#                          s)
#                    s2)
#            (cons (car args) r2)
#            m)))
def applycont(s2, r2, args, s, r, m):
    if no(args) or yes(cdr(args)):
        return sigerr(quote("wrong-no-args"), s, r, m)
    else:
        return mev(append(keep(lambda _: and_f(lambda: protected(_),
                                               lambda: no(mem(_, s2, id))),
                               s),
                          s2),
                   cons(car(args), r2),
                   m)

# (def protected (x)
#   (some [begins (car x) (list smark _) id]
#         '(bind prot)))
def protected(x):
    return some(lambda _: begins(car(x), list(smark, _), id),
                list(quote("bind"), quote("prot")))

# (def function (x)
#   (find [(isa _) x] '(prim clo)))
def function(x):
    return find(lambda _: isa(_)(x), list(quote("prim"), quote("clo")))

# (def con (x)
#   (fn args x))
def con(x):
    return lambda *args, **kws: x

# # (def compose fs
# #   (reduce (fn (f g)
# #             (fn args (f (apply g args))))
# #           (or fs (list idfn))))
# def compose(*fs):
#     return reduce(compose2, fs or (idfn,))
#
# def compose2(f, g):
#     @functools.wraps(g)
#     def f_then_g(*args, **kws):
#         return f(apply(g, args, **kws))
#     # f_name = getattr(f, "__qualname__", getattr(f, "__name__", "<unknown>"))
#     # g_name = getattr(g, "__qualname__", getattr(g, "__name__", "<unknown>"))
#     f_name = nameof(f)
#     g_name = nameof(g)
#     f_then_g.__qualname__ = f_then_g.__name__ = f"{f_name}:{g_name}"
#     return f_then_g

# (def combine (op)
#   (fn fs
#     (reduce (fn (f g)
#               (fn args
#                 (op (apply f args) (apply g args))))
#             (or fs (list (con (op)))))))
def combine_f(op):
    def combiner(*fs):
        def combine2(f, g):
            def combined(*args, **kws):
                return op(lambda: apply(f, args, **kws),
                          lambda: apply(g, args, **kws))
            return combined
        return reduce(combine2, fs or list(con(op())))
    return combiner

# (set cand (combine and)
#      cor  (combine or))
cand = combine_f(and_f)
cor = combine_f(or_f)

# (def foldl (f base . args)
#   (if (or (no args) (some no args))
#       base
#       (apply foldl f
#                    (apply f (snoc (map car args) base))
#                    (map cdr args))))
#
# (def foldr (f base . args)
#   (if (or (no args) (some no args))
#       base
#       (apply f (snoc (map car args)
#                      (apply foldr f base (map cdr args))))))
#
# (def of (f g)
#   (fn args (apply f (map g args))))
def of(f, g):
    def f_of_g(*args, **kws):
        return apply(f, map(g, args), **kws)
    return f_of_g

# (def upon args
#   [apply _ args])
def upon(*args, **kws):
    return lambda _: apply(_, args, **kws)

# (def pairwise (f xs)
#   (or (no (cdr xs))
#       (and (f (car xs) (cadr xs))
#            (pairwise f (cdr xs)))))
def pairwise(f, xs):
    return or_f(lambda: no(cdr(xs)),
                lambda: and_f(lambda: f(car(xs), cadr(xs)),
                              lambda: pairwise(f, cdr(xs))))

# (def fuse (f . args)
#   (apply append (apply map f args)))
def fuse(f, *args, **kws):
    return apply(append, apply(map, f, args, **kws))

# (mac letu (v . body)
#   (if ((cor variable atom) v)
#       `(let ,v (uvar) ,@body)
#       `(with ,(fuse [list _ '(uvar)] v)
#          ,@body)))
@macro_
def letu(v, *body):
    return if_f(lambda: cor(variable, atom)(v),
                lambda: ["let", v, ["uvar", ["quote", v]], *body],
                lambda: ["with", fuse(lambda _: vec2list([_, ["uvar", ["quote", _]]]), v),
                         *body])()
    # if yes(cor(variable, atom)(v)):
    #     return list(quote("let"), v, list(quote("uvar")), *body)
    # else:
    #     return list(quote("with"), fuse(lambda _: list(_, list(quote("uvar"), list(quote("quote"), _))), v),
    #                 *body)

# (mac pcase (expr . args)
#   (if (no (cdr args))
#       (car args)
#       (letu v
#         `(let ,v ,expr
#            (if (,(car args) ,v)
#                ,(cadr args)
#                (pcase ,v ,@(cddr args)))))))


# (def match (x pat)
#   (if (= pat t)                t
#       (function pat)           (pat x)
#       (or (atom x) (atom pat)) (= x pat)
#                                (and (match (car x) (car pat))
#                                     (match (cdr x) (cdr pat)))))
#
# (def split (f xs (o acc))
#   (if ((cor atom f:car) xs)
#       (list acc xs)
#       (split f (cdr xs) (snoc acc (car xs)))))
#
# (mac when (expr . body)
#   `(if ,expr (do ,@body)))
@macro_
def when(expr, *body):
    return ["if", expr, ["do", *body]]
    # return list(quote("if"), expr, list(quote("do"), *body))

# (mac unless (expr . body)
#   `(when (no ,expr) ,@body))
@macro_
def unless(expr, *body):
    return ["when", ["no", expr], *body]
    # return list(quote("when"), list(quote("no"), expr), *body)

# (mac bind (var expr . body)
#   `(dyn ,var ,expr (do ,@body)))
@macro_
def bind(var, expr, *body):
    return ["dyn", var, expr, ["do", *body]]
    # return list(quote("dyn"), var, expr, list(quote("do"), *body))

# (mac atomic body
#   `(bind lock t ,@body))
@macro_
def atomic(*body):
    return ["bind", "lock", "t", *body]
    # return list(quote("bind"), quote("lock"), quote("t"), *body)

# (mac set args
#   (cons 'do
#         (map (fn ((p (o e t)))
#                (letu v
#                  `(atomic (let ,v ,e
#                             (let (cell loc) (where ,p t)
#                               ((case loc a xar d xdr) cell ,v))))))
#              (hug args))))
@macro_
def set(*args):
    def f(x):
        p, e = car(x), (cadr(x) if cdr(x) else t)
        v = uvar("set_v")
        return ["atomic", ["let", v, e,
                           ["let", ["cell", "loc"], ["where", p, "t"],
                            [["case", "loc", "a", "xar", "d", "xdr"], "cell", v]]]]
        # return list(quote("atomic"), list(quote("let"), v, e,
        #                                   list(quote("let"), list(quote("cell"), quote("loc")), list(quote("where"), p, quote("t")),
        #                                        list(list(quote("case"), quote("loc"), quote("a"), quote("xar"), quote("d"), quote("xdr")), quote("cell"), v))))
    return cons(quote("do"), map(f, hug(args)))

def subtract(x, y=unset):
    if y is unset:
        return -x
    return x - y

from math import floor

globals()["print"] = print
globals()["."] = lambda *args: cdr(get(*args))
globals()["+"] = operator.add
globals()["-"] = subtract
globals()["*"] = operator.mul
globals()["/"] = operator.truediv
globals()["//"] = operator.floordiv
globals()["="] = equal
eval = eval
exec = exec
compile = compile

def unbound():
    return unset

import sys
sys.setrecursionlimit(10_000)

# >>> bel(list("join", 1, 2), map(lambda _: cons(_, list(quote("lit"), quote("prim"), _)), apply(append, prims)))
# (1 . 2)
# >>> bel(list("dyn", "err", list("fn", list("x"), list("do", list("print", "x"), list("quote", "hello"))), list("car", list("quote", "b"))))
# 'str' object has no attribute 'car'
# 'hello'

# from lul.common import reader
# belforms = reader.read_all(reader.stream(open("bel.bel").read()))
# [print(repr(x.car)) for x in vec2list(belforms)]

sbits: Dict[int, List[str]] = collections.defaultdict(py.list)

def rd(s: io.IOBase) -> Optional[int]:
    if s.closed:
        err("closed")
    n = 0
    for byte in s.read(1):
        sbits[py.id(s)].extend(bin(byte)[2:].zfill(8)[::-1])
        n += 8
    return nil if n == 0 else n

def rdb(s: io.IOBase):
    if len(bits := sbits[py.id(s)]) > 0:
        return bits.pop(0)

def wrb(s: io.IOBase, c: str):
    assert streamp(s)
    assert c in "10"
    bits = sbits[py.id(s)]
    bits.append(c)
    if len(bits) >= 8 and equal(stat(s), quote("out")):
        byte = int(''.join(bits[0:8][::-1]), 2)
        del bits[0:8]
        s.write(bytes([byte]))
        s.flush()

def ops(x: str, y: Literal["out", "in"]) -> io.IOBase:
    """12. (ops x y)

    Returns a stream that writes to or reads from the place whose name is
    the string x, depending on whether y is out or in respectively.
    Signals an error if it can't, or if y is not out or in."""
    if y not in ["out", "in"]:
        err(quote("y-not-out-or-in"), list(quote("ops"), x, y))
    s = open(x, "rb" if equal(y, "in") else "wb")
    assert streamp(s)
    return s

def cls(x: io.IOBase):
    """13. (cls x)

    Closes the stream x. Signals an error if it can't."""
    x.close()
    try:
        del sbits[py.id(x)]
    except KeyError:
        pass
    return t

def stat(x: io.IOBase):
    """14. (stat x)

    Returns either closed, in, or out depending on whether the stream x
    is closed, or reading from or writing to something respectively.
    Signals an error if it can't."""
    if x.closed:
        return quote("closed")
    mode = getattr(x, "mode", "wb")
    if mode.startswith("r"):
        return quote("in")
    return quote("out")

# https://stackoverflow.com/questions/20936993/how-can-i-create-a-random-number-that-is-cryptographically-secure-in-python
from random import SystemRandom
coingen = SystemRandom()

def coin():
    """15. (coin)

    Returns either t or nil randomly."""
    return nil if coingen.randint(0, 1) == 0 else t

import subprocess
import shlex

def shell(x, args=nil):
    """16. (shell x args)

    Sends x as a command to the operating system."""
    cmd = ' '.join([x] + [shlex.quote(arg) for arg in list2vec(args)])
    return subprocess.run(cmd, shell=True).returncode

def read(x, eof=None):
    if stringp(x):
        form, pos = reader.read_from_string(x, mode="bel")
        return form
    else:
        return reader.read(x, eof=eof)

def readall(string: str) -> List[Union[str, List]]:
    return reader.read_all(reader.stream(string, mode="bel"))

def readallbel(string: str):
    return vec2list(readall(string))

def compilebel(source):
    form = readallbel(source)
    if yes(cdr(form)):
        return cons(quote("do"), form)
    else:
        return car(form)

def evalbel(form, globals=unset, locals=unset):
    return bel(form, globals, locals)

def bell(source, globals=unset, locals=unset):
    return evalbel(compilebel(source), globals=globals, locals=locals)

def escape(x):
    return json.dumps(x)

def quote1(x):
    return list(quote("quote"), x)

def quoted(x):
    # if stringp(x):
    #     return escape(x)
    # elif atom(x):
    #     return x
    # if literal(x):
    #     # if callable(x):
    #     #     return x()
    #     return nom(x)
    # if literal(x):
    #     return nom(x)
    if atom(x):
        return quote1(x)
    elif not proper(x):
        return cons(quote("cons"), map(quoted, mkproper(x)))
    else:
        return cons(quote("list"), map(quoted, x))

def callbel(f, *args, **kws):
    if consp(f):
        if yes(isa(quote("mac"))(f)):
            # return bel(callbel(caddr(f), *args, **kws))
            return bel(callbel(caddr(f), *map(quote1, args), **kws))
        else:
            return bel(cons(f, map(quote1, args)))
    else:
        return f(*args, **kws)

Cons.__call__ = callbel

# bell(source="""
# (def foo (x) (+ x 1))
#
# (def quotingp (depth)
#   (numberp depth))
#
# (def quasiquotingp (depth)
#   (and (quotingp depth) (> depth 0)))
#
# (def can_unquote_p (depth)
#   (and (quoting? depth) (= depth 1)))
#
# (def quasisplicep (x depth)
#   (and (can_unquote_p depth)
#        (not (atom x))
#        (= (car x) 'unquote-splicing)))
#
# """)

def quotingp (depth):
    return numberp(depth)

def quasiquotingp(depth):
    return and_f(lambda: quotingp(depth),
                 lambda: depth > 0)

def can_unquote_p(depth):
    return and_f(lambda: quotingp(depth),
                 lambda: depth == 1)

def quasisplicep(x, depth):
    return and_f(lambda: can_unquote_p(depth),
                 lambda: no(atom(x)),
                 lambda: equal(car(x), quote("unquote-splicing")))

def lastcdr(l: Cons[A, D]) -> Cons[A, D]:
    assert consp(l)
    # return l[-1]
    tail = l
    for tail in l.tails():
        pass
    return tail

def add(l: Cons[A, D], x: A):
    cell = lastcdr(l)
    assert null(cdr(cell))
    xdr(cell, list(x))

def last(l: Sequence[A]) -> A:
    try:
        return l[-1]
    except IndexError:
        pass

# def last(l: Cons[A, D]) -> A:
#     # return car((XCONS(l) or [(nil,)])[-1])
#     return car(lastcdr(l))

def step(form: Iterable[T]) -> Generator[T]:
    if not null(form):
        # for tail in form.tails():
        #     yield car(tail)
        for x in form:
            yield x


# (define quasiquote-list (form depth)
#   (let xs (list '(list))
#     (each (k v) form
#       (unless (number? k)
#         (let v (if (quasisplice? v depth)
#                    ;; don't splice, just expand
#                    (quasiexpand (at v 1))
#                  (quasiexpand v depth))
#           (set (get (last xs) k) v))))
#     ;; collect sibling lists
#     (step x form
#       (if (quasisplice? x depth)
#           (let x (quasiexpand (at x 1))
#             (add xs x)
#             (add xs '(list)))
#         (add (last xs) (quasiexpand x depth))))
#     (let pruned
#         (keep (fn (x)
#                 (or (> (# x) 1)
#                     (not (= (hd x) 'list))
#                     (keys? x)))
#               xs)
#       (if (one? pruned)
#           (hd pruned)
#         `(join ,@pruned)))))
def quasiquote_list(form, depth):
    xs = list(list(quote("cons")))
    # collect sibling lists
    for x in step(mkproper(form)):
        if quasisplicep(x, depth):
            add(last(xs), quote("nil"))
            # add(xs, list(quote("mkproper"), quasiexpand(cadr(x), nil)))
            add(xs, quasiexpand(cadr(x), nil))
            add(xs, list(quote("cons")))
        else:
            add(last(xs), quasiexpand(x, depth))
    if proper(form):
        add(last(xs), quote("nil"))
    xs = rem(list(quote("cons"), quote("nil")), xs)
    xs = map(quasifix, xs)
    return cons(quote("append"), xs) if len(xs) > 1 else car(xs)

def quasifix(form):
    if yes(begins(form, list(quote("cons")))):
        if yes(equal(last(form), quote("nil"))):
            return cons(quote("list"), cdr(almost(form)))
    return form

# (define-global quasiexpand (form depth)
#   (if (quasiquoting? depth)
#       (if (atom? form) (list 'quote form)
#           ;; unquote
#           (and (can-unquote? depth)
#                (= (hd form) 'unquote))
#           (quasiexpand (at form 1))
#           ;; decrease quasiquoting depth
#           (or (= (hd form) 'unquote)
#               (= (hd form) 'unquote-splicing))
#           (quasiquote-list form (- depth 1))
#           ;; increase quasiquoting depth
#           (= (hd form) 'quasiquote)
#           (quasiquote-list form (+ depth 1))
#         (quasiquote-list form depth))
#       (atom? form) form
#       (= (hd form) 'quote) form
#       (= (hd form) 'quasiquote)
#       ;; start quasiquoting
#       (quasiexpand (at form 1) 1)
#     (map (fn (x) (quasiexpand x depth)) form)))

def quasiexpand(form, depth):
    if yes(quasiquotingp(depth)):
        if yes(atom(form)):
            return list(quote("quote"), form)
        # unquote
        if yes(and_f(lambda: can_unquote_p(depth),
                       lambda: equal(car(form), quote("unquote")))):
            return quasiexpand(cadr(form), nil)
        # decrease quasiquoting depth
        if yes(or_f(lambda: equal(car(form), quote("unquote")),
                    lambda: equal(car(form), quote("unquote-splicing")))):
            return quasiquote_list(form, depth - 1)
        # increase quasiquoting depth
        if yes(equal(car(form), quote("quasiquote"))):
            return quasiquote_list(form, depth + 1)
        return quasiquote_list(form, depth)
    if yes(atom(form)):
        return form
    if yes(equal(car(form), quote("quote"))):
        return form
    if yes(equal(car(form), quote("quasiquote"))):
        # start quasiquoting
        return quasiexpand(cadr(form), 1)
    return imap(lambda x: quasiexpand(x, depth), form)

def imap(f, l):
    if proper(l):
        return map(f, l)
    else:
        return apply(cons, map(f, mkproper(l)))

def quasi(form):
    return quasiexpand(form, 1)

@macro_
def quasiquote(form):
    return quasi(form)

# (mac letu (v . body)
#   (if ((cor variable atom) v)
#       `(let ,v (uvar) ,@body)
#       `(with ,(fuse [list _ '(uvar)] v)
#          ,@body)))

# (mac pcase (expr . args)
#   (if (no (cdr args))
#       (car args)
#       (letu v
#         `(let ,v ,expr
#            (if (,(car args) ,v)
#                ,(cadr args)
#                (pcase ,v ,@(cddr args)))))))

# (def bqex (e n)
#   (if (no e)   (list nil nil)
#       (atom e) (list (list 'quote e) nil)
#                (case (car e)
#                  bquote   (bqthru e (list n) 'bquote)
#                  comma    (if (no n)
#                               (list (cadr e) t)
#                               (bqthru e (car n) 'comma))
#                  comma-at (if (no n)
#                               (list (list 'splice (cadr e)) t)
#                               (bqthru e (car n) 'comma-at))
#                           (bqexpair e n))))
#
# (def bqthru (e n op)
#   (let (sub change) (bqex (cadr e) n)
#     (if change
#         (list (if (caris sub 'splice)
#                   `(cons ',op ,(cadr sub))
#                   `(list ',op ,sub))
#               t)
#         (list (list 'quote e) nil))))
#
# (def bqexpair (e n)
#   (with ((a achange) (bqex (car e) n)
#          (d dchange) (bqex (cdr e) n))
#     (if (or achange dchange)
#         (list (if (caris d 'splice)
#                   (if (caris a 'splice)
#                       `(apply append (spa ,(cadr a)) (spd ,(cadr d)))
#                       `(apply cons ,a (spd ,(cadr d))))
#                   (caris a 'splice)
#                   `(append (spa ,(cadr a)) ,d)
#                   `(cons ,a ,d))
#               t)
#         (list (list 'quote e) nil))))
#
# (def spa (x)
#   (if (and x (atom x))
#       (err 'splice-atom)
#       x))
#
# (def spd (x)
#   (pcase x
#     no   (err 'splice-empty-cdr)
#     atom (err 'splice-atom)
#     cdr  (err 'splice-multiple-cdrs)
#          x))


#
# (def nth (n|pint xs|pair)
#   (if (= n 1)
#       (car xs)
#       (nth (- n 1) (cdr xs))))
def nth(n, xs, default=unset):
    if default is unset:
        default = nil
    if consp(xs):
        xs: Cons
        return car(xs[n-1])
    try:
        return xs[n-1]
    except IndexError:
        return default

# (vir num (f args)
# `(nth ,f ,@args))
@vir_(quote("number"))
def vir_number(f, *args):
    return cons(quote("nth"), f, args)

@vir_(quote("symbol"))
def vir_symbol(k, *args):
    if len(args) <= 0:
        return list(quote("fn"), list(quote("_")), list(quoted(k), quote("_")))
    elif len(args) > 1:
        err("todo-multi-args")
    if keyword(k):
        return list(evsyntax("py!getattr"), car(args), quoted(keynom(k)))
    return list(quote("cdr"), cons(quote("get"), quoted(k), args))

# (def table ((o kvs))
#   `(lit tab ,@kvs))
def table(kvs=unset) -> py.dict:
    if kvs is unset:
        kvs = nil
    # return py.dict([(car(x), cdr(x)) for x in (vec2list(kvs).list() if not null(kvs) else [])])
    return py.dict([(car(x), cdr(x)) for x in XCONS(kvs)])

#
# (vir tab (f args)
#   `(tabref ,f ,@args))
@vir_(quote("tab"))
def vir_table(f, *args):
    return cons(quote("tabref"), f, args)

@vir_(quote("mod"))
def vir_module(f, *args):
    return cons(quote("tabref"), list("getattr", f, list(quote("quote"), quote("__dict__"))), args)

# (def tabref (tab key (o default))
#   (aif (get key (cddr tab))
#        (cdr it)
#        default))
def tabref(tab, key, default=unset):
    if default is unset:
        default = nil
    if yes(it := get(key, tab)):
        return cdr(it)
    else:
        return default

# (loc isa!tab (f args a s r m)
#   (let e `(list (tabloc ,f ,@(map [list 'quote _] args)) 'd)
#     (mev (cons (list e a) (cdr s)) r m)))
#
# (def tabloc (tab key)
#   (or (get key (cddr tab))
#       (let kv (cons key nil)
#         (push kv (cddr tab))
#         kv)))
#
# (def tabrem (tab key (o f =))
#   (clean [caris _ key f] (cddr tab)))



# >>> bel( readbel("(join join join)"))
# (<function join at 0x105a02a60> . <function join at 0x105a02a60>)
# >>> bel( readbel("(join join join)"), map(lambda _: cons(_, list(quote("lit"), quote("prim"), _)), apply(append, prims)))
# ((lit prim join) lit prim join)

def repl():
    buf = ""
    def clear():
        nonlocal buf
        buf = ""
        print("> ", end="", flush=True)
    clear()
    more = object()
    while True:
        try:
            line = sys.stdin.readline()
            if line == '': # EOF
                break
            buf += line
            form, pos = reader.read_from_string(buf, more=more)
            if form is more:
                continue
            print(json.dumps(form))
            clear()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            clear()


import code
import codeop

class BelCommandCompiler(codeop.CommandCompiler):
    def __call__(self, source, filename="<input>", symbol="single"):
        form, pos = reader.read_from_string(source, more=(more := object()), mode="bel")
        if form is more:
            return None
        return form

class BelConsole(code.InteractiveConsole):
    def __init__(self, locals=None, filename="<console>"):
        super().__init__(locals=locals, filename=filename)
        self.locals = locals
        self.compile = BelCommandCompiler()
        self.that = nil
        self.thatexpr = nil

    def exec(self, form, locals=None):
        if locals is None:
            locals = self.locals
        # print(json.dumps(form))
        self.thatexpr = vec2list(form)
        self.that = bel(self.thatexpr)
        if self.that is not None:
            print(prrepr(self.that))

    def runcode(self, code):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to
        display a traceback.  All exceptions are caught except
        SystemExit, which is reraised.

        A note about KeyboardInterrupt: this exception may occur
        elsewhere in this code, and may not always be caught.  The
        caller should be prepared to deal with it.

        """
        reload(M)
        try:
            self.exec(code, self.locals)
        except SystemExit:
            raise
        except:
            self.showtraceback()



@contextlib.contextmanager
def letattr(obj, key, val, *default):
    prev = getattr(obj, key, *default)
    setattr(obj, key, val)
    try:
        yield
    finally:
        setattr(obj, key, prev)

def interact(banner=None, readfunc=None, local=None, exitmsg=None):
    """Closely emulate the interactive Python interpreter.

    This is a backwards compatible interface to the InteractiveConsole
    class.  When readfunc is not specified, it attempts to import the
    readline module to enable GNU readline if it is available.

    Arguments (all optional, all default to None):

    banner -- passed to InteractiveConsole.interact()
    readfunc -- if not None, replaces InteractiveConsole.raw_input()
    local -- passed to InteractiveInterpreter.__init__()
    exitmsg -- passed to InteractiveConsole.interact()

    """
    console = BelConsole(local)
    if readfunc is not None:
        console.raw_input = readfunc
    else:
        try:
            import readline
        except ImportError:
            pass
    iterm2_prompt_mark = "\u001b]133;A\u0007\n"
    with letattr(sys, 'ps1', iterm2_prompt_mark + '> ', '>>> '):
        with letattr(sys, 'ps2', '  ', '... '):
            console.interact(banner, exitmsg)



def dbg():
    import pdb
    pdb.pm()


import argparse

if TYPE_CHECKING:
    class BelArguments(Protocol):
        filename: Optional[str]
        verbose: bool
else:
    BelArguments = argparse.Namespace

if 'argv' not in globals():
    argv: BelArguments = BelArguments(filename=None, verbose=False)

def bel_parser():
    parser = argparse.ArgumentParser(
        description="Bel Lisp",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--verbose', '-v',
        action='store_true',
        help='verbose flag')

    parser.add_argument(
        "filename",
        help="Run a bel file as a script",
        nargs="?",
    )
    return parser

def bel_args(args=unset) -> BelArguments:
    if args is unset:
        args = sys.argv[1:]
    return bel_parser().parse_args(args)

def read_string(source: str, filename: Optional[str] = None):
    return readall(source)

def read_file(filename: str):
    with open(filename) as f:
        return f.read()

def macroexp_do(body):
    if null(body):
        return list("do")
    if consp(body):
        if len(body) == 1:
            return car(body)
        return cons("do", body)
    else:
        assert vectorp(body)
        if len(body) == 1:
            return body[0]
        return ["do", *body]

def macroexp_body(form):
    if caris(form, "do"):
        return cdr(form)
    return list(form)

def read_from_file(filename: str) -> List[Union[str, List]]:
    source = read_file(filename)
    body = read_string(source, filename=filename)
    return body

def compile_body(body: List):
    return macroexp_do([compile(form) for form in body])

def compile(form):
    return vec2list(form)

def run_toplevel(form):
    if argv.verbose:
        prn(">", form, fg=orange)
    result = bel(form)
    if argv.verbose:
        prn(result)

def run(*body: Union[str, List], filename=None):
    for form in body:
        expr = compile(form)
        run_toplevel(expr)

def load(filename: str):
    body = read_from_file(filename)
    return run(*body, filename=filename)

def main(args=unset, *, locals=None):
    global argv
    argv = bel_args(args)
    if stringp(argv.filename):
        return load(argv.filename)
    else:
        return interact(local=locals)



if not globals().get("initialized"):
    bell("""
    (mac import (lib (o as)) `(set ,(if as as lib) (__import__ ',lib)))
    (mac w/frame (a . body) (letu ua `(let ,ua (append scope ,a) (dyn scope ,ua ((fn () ,@body))))))
    """)
    initialized = True


# outs = globals().get("outs", sys.stdout)

def call_with_stdout(f, stdout: io.IOBase):
    prev = sys.stdout
    try:
        sys.stdout = stdout
        return f()
    finally:
        sys.stdout = prev

def call_with_stdout_to_string(f):
    out = io.StringIO()
    call_with_stdout(f, out)
    return out.getvalue()

@macro_
def tostring(*body):
    return ["call_with_stdout_to_string", ["fn", [], *body]]





# (safe:read "\"foo") fails to trap the error since compose calls macros directly