from __future__ import annotations

import contextlib

from .runtime import *
import json
import sys

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
def list(*args):
    return append(args, nil)

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
    return list(quote("lit"), quote("mac"), f)

# (mac fn (parms . body)
#   (if (no (cdr body))
#       `(list 'lit 'clo scope ',parms ',(car body))
#       `(list 'lit 'clo scope ',parms '(do ,@body))))
@macro_
def fn(parms, *body):
    if no(cdr(body)):
        return list(quote("list"),
                    list(quote("quote"), quote("lit")),
                    list(quote("quote"), quote("clo")),
                    quote("scope"),
                    list(quote("quote"), parms),
                    list(quote("quote"), car(body)))
    else:
        return list(quote("list"),
                    list(quote("quote"), quote("lit")),
                    list(quote("quote"), quote("clo")),
                    quote("scope"),
                    list(quote("quote"), parms),
                    list(quote("quote"), list(quote("do"), *body)))

# (set vmark (join))
vmark = join("%vmark")

# (def uvar ()
#   (list vmark))
def uvar(*name):
    return list(vmark, *name)

# (mac do args
#   (reduce (fn (x y)
#             (list (list 'fn (uvar) y) x))
#           args))
@macro_
def do(*args):
    return reduce(lambda x, y: list(list(quote("fn"), uvar("do"), y), x),
                  args)

def do_f(x, *args):
    for f in args:
        x = f(x)
    return x

# (mac let (parms val . body)
#   `((fn (,parms) ,@body) ,val))
@macro_
def let(parms, val, *body):
    return list(list(quote("fn"), list(parms), *body), val)

# (mac macro args
#   `(list 'lit 'mac (fn ,@args)))
@macro_
def macro(*args):
    return list(quote("list"),
                list(quote("quote"), quote("lit")),
                list(quote("quote"), quote("mac")),
                list(quote("fn"), *args))


# (mac def (n . rest)
#   `(set ,n (fn ,@rest)))
@macro_
def def_(n, *rest):
    return list(quote("set"), n, list(quote("fn"), *rest))
globals()["def"] = def_

# (mac mac (n . rest)
#   `(set ,n (macro ,@rest)))
@macro_
def mac(n, *rest):
    return list(quote("set"), n, list(quote("macro"), *rest))

# (mac or args
#   (if (no args)
#       nil
#       (let v (uvar)
#         `(let ,v ,(car args)
#            (if ,v ,v (or ,@(cdr args)))))))
def or_(*args):
    # if no(args):
    #     return nil
    # elif yes(v := car(args)):
    #     return v
    # else:
    #     return apply(or_, cdr(args))
    for x in args:
        if yes(x):
            return x
    return nil

def or_f(*args):
    for f in args:
        x = f()
        if yes(x):
            return x
    return nil

# (mac and args
#   (reduce (fn es (cons 'if es))
#           (or args '(t))))
def and_(*args):
    if no(args):
        return t
    else:
        x = nil
        for x in args:
            if no(x):
                return nil
        return x

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
        return and_(apply(equal, map(car, args)),
                    apply(equal, map(cdr, args)))
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
        v = uvar()
        return list(quote("let"), v, expr,
                    list(quote("if"),
                         list(quote("equal"), v, list(quote("quote"), car(args))),
                         cadr(args),
                         list(quote("case"), v, *args[2:])))


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
#
# (mac aif args
#   `(iflet it ,@args))
#
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
#
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
def get(k, kvs, f=unset):
    if f is unset:
        f = equal
    if null(kvs):
        return kvs
    if isinstance(kvs, Cons):
        # return find(lambda _: f(car(_), k), kvs)
        for tail in kvs:
            cell = car(tail)
            name, value = car(cell), cdr(cell)
            if value is unset:
                continue
            if f(name, k):
                return cell
        if dictp(it := cdr(kvs[-1])):
            kvs = it
        elif modulep(it):
            kvs = it
        else:
            return nil
    if modulep(kvs):
        kvs = kvs.__dict__
    assert f in [equal, id]
    out = Cell(kvs, k, unset)
    if cdr(out) is unset:
        return nil
    return out

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
        for tail in kvs:
            cell = car(tail)
            if locatable(cell):
                if yes(it := locate(k, cell, new, f)):
                    return it
            else:
                assert alistp(tail)
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
def snap(xs, ys, acc=nil):
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
def eif(var, expr=unset, fail=unset, ok=unset):
    if expr is unset:
        expr = nil
    if fail is unset:
        fail = nil
    if ok is unset:
        ok = nil
    v = uvar(quote("v"))
    w = uvar(quote("w"))
    c = uvar(quote("c"))
    return list(quote("let"), v, list(quote("join")),
                list(quote("let"), w, list(quote("ccc"), list(quote("fn"), list(c),
                                                              list(quote("dyn"), quote("err"),
                                                                   list(quote("fn"), list(quote("_")),
                                                                        list(c),
                                                                        list(quote("cons"), v, quote("_"))),
                                                                   expr))),
                     list(quote("if"), list(quote("caris"), w, v, quote("id")),
                          list(quote("let"), var, list(quote("cdr"), w), fail),
                          list(quote("let"), var, w, ok))))



# (mac fn (parms . body)
#   (if (no (cdr body))
#       `(list 'lit 'clo scope ',parms ',(car body))
#       `(list 'lit 'clo scope ',parms '(do ,@body))))
@macro_
def fn(parms, *body):
    if no(cdr(body)):
        return list(quote("list"),
                    list(quote("quote"), quote("lit")),
                    list(quote("quote"), quote("clo")),
                    quote("scope"),
                    list(quote("quote"), parms),
                    list(quote("quote"), car(body)))
    else:
        return list(quote("list"),
                    list(quote("quote"), quote("lit")),
                    list(quote("quote"), quote("clo")),
                    quote("scope"),
                    list(quote("quote"), parms),
                    list(quote("quote"), list(quote("do"), *body)))


#
# (mac onerr (e1 e2)
#   (let v (uvar)
#     `(eif ,v ,e2 ,e1 ,v)))
#
# (mac safe (expr)
#   `(onerr nil ,expr))
#
# (def literal (e)
#   (or (in e t nil o apply)
#       (in (type e) 'char 'stream)
#       (caris e 'lit)
#       (string e)))
def literal(e):
    if in_(e, t, nil, o, apply):
        return t
    elif in_(type(e), quote("char"), quote("stream"), quote("tab"), quote("char")):
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

namecs = dict(
    bel="\a",
    tab="\t",
    lf="\n",
    cr="\r",
    sp=" ")

def evliteral(e):
    # if string_literal_p(e) and reader.read_from_string(e, more=object())[0] == e:
    if string_literal_p(e):
        return json.loads(e)
    if char(e):
        e: str
        name = e[1:]
        return namecs.get(name, name)
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
    if no(s):
        return if_f(lambda: p,
                    lambda: ([print("discard", r)] and sched(p, g)),
                    lambda: ([print("leftover", it) if yes(it := cdr(r)) else nil] and car(r)))()
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
    # def __init__(self):
    #     super().__init__(car=nil, cdr=list(nil))
    @property
    def s(self: TBelThread) -> ConsList[BelExpression]:
        return map(BelExpression.new, car(self))
    @property
    def r(self: TBelThread) -> ConsList[T]:
        return cadr(self)
    def __repr__(self):
        # return f"BelThread(s={self.s!r}, r={self.r!r})"
        return repr_self(self, ("s", None), ("r", None))

TBelExpression: TypeAlias = "Union[List2[T, TA], BelExpression]"

class BelExpression(Cons):
    @property
    def e(self) -> T:
        return car(self)
    @property
    def a(self) -> TA:
        return cadr(self)
    def __repr__(self):
        # return f"BelExpression(e={self.e!r})"
        return repr_self(self, ("e", None), ("a", None))

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



# class BelParameters(Cons):
#     @property
#     def e(self):
#         return car(self)
#     @property
#     def a(self):
#         return cadr(self)
#     def __repr__(self):
#         return f"BelExpression(e={self.e!r})"



class JumpToMev(Exception):
    def __init__(self, s: TS, r: TR, p_g: TBelThreads, prev: Optional[JumpToMev]):
        self.s = s
        self.r = r
        self.p_g = p_g
        self.prev = prev

    @property
    def thread(self):
        return BelThread.new(cons(self.s, self.r))

    @property
    def expr(self):
        for expr in self.thread.s:
            if yes(expr.a):
                return expr.e

    @property
    def lexenv(self):
        for expr in self.thread.s:
            if yes(expr.a):
                return expr.a

    def __repr__(self):
        return repr_self(self, ("lexenv", None), ("thread", None), ("prev", None))

    def __str__(self):
        return repr(self)

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
            if x.startswith("!") and len(x) > 1:
                return t
            for c in [":", "|", "!"]:
                if not (x.endswith(c) or x.startswith(c)):
                    if c in x:
                        return t
    return nil

def evsyntax(x):
    if not syntaxp(x):
        return quote(x)
    x: str
    if ":" in x and "!" not in x and "|" not in x:
        lh, _, rh = x.partition(":")
        return list(quote("compose"), evsyntax(lh), evsyntax(rh))
    if x.startswith("!") and len(x) > 1:
        return list(quote("upon"), evsyntax(x[1:]))
    if "!" in x and "|" not in x:
        name, _, arg = x.rpartition("!")
        return list(evsyntax(name), list(quote("quote"), quote(arg)))
    if "|" in x:
        name, _, test = x.rpartition("|")
        return list(quote("t"), quote(name), evsyntax(test))
    if x.startswith("~") and len(x) > 1:
        return list(quote("compose"), quote("no"), evsyntax(x[1:]))
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

def vscope(a, g):
    if not null(a):
        for tail in a:
            if dictp(it := cdr(tail)):
                return it
    return g

def assign(where, v):
    assert not null(where)
    cell, loc = car(where), cdr(where)
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
        return assign(cons(cell, quote("d")), v)
    elif consp(kvs):
        cell = cons(k, v)
        xdr(kvs, cons(cell, cdr(kvs)))
    else:
        return err(quote("cannot-set"), k)

# (set smark (join))
smark = join("%smark")

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




# (mac fu args
#   `(list (list smark 'fut (fn ,@args)) nil))
# def fu(f):
#     return list(list(smark, quote("fut"), f), nil)
def fu(a: Cons):

    def fut(f: Callable[[TS, TR, TM], Any]):
        return list(list(smark, quote("fut"), f), a)
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
def if_(es, a, s, r, m):
    if no(es):
        return mev(s, cons(nil, r), m)
    else:
        return mev(cons(list(car(es), a),
                        if_f(lambda: cdr(es),
                             lambda: cons(fu(a)((lambda s, r, m:
                                                 if2(cdr(es), a, s, r, m))),
                                          s),
                             lambda: s)()),
                   r,
                   m)

# (def if2 (es a s r m)
#   (mev (cons (list (if (car r)
#                        (car es)
#                        (cons 'if (cdr es)))
#                    a)
#              s)
#        (cdr r)
#        m))
def if2(es, a, s, r, m):
    return mev(cons(list(if_f(lambda: car(r),
                              lambda: car(es),
                              lambda: cons(quote("if"), cdr(es)))(),
                         a),
                    s),
               cdr(r),
               m)

def if_f(*clauses: Callable[[], Any]) -> Callable[[], Any]:
    while len(clauses) >= 2:
        cond, cons, *clauses = clauses
        if yes(cond()):
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
    v, e1, e2 = car(es), cadr(es), caddr(es)
    if yes(variable(v)):
        return mev(cons(list(e1, a),
                        fu(a)(lambda s, r, m: dyn2(v, e2, a, s, r, m)),
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
               cdr(r),
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
                    fu(a)(lambda s, r, m: evcall2(cdr(e), a, s, r, m)),
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
        @fu(a)
        def f(s, r, m):
            _ = snap(es, r)
            args, r2 = car(_), cadr(_)
            return applyf(op, rev(args), a, s, r2, m)
        return mev(append(map(lambda _: list(_, a), es),
                          cons(f,
                               s)),
                   r,
                   m)

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
                  cons(fu(a)(lambda s, r, m: mev(cons(list(car(r), a), s),
                                                 cdr(r),
                                                 m)),
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
    if equal(f, apply):
        return applyf(car(args), reduce(join, cdr(args)), a, s, r, m)
    else:
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

def applyfunc(f, args, a, s, r, m):
    try:
        v = apply(f, args)
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
    # if yes(inwhere(s)) and yes(it := find(lambda _: car(_)(f), locfns)):
    if yes(it := and_f(lambda: [print("inwhere", that) if yes(that := inwhere(s)) else nil] and that,
                       lambda: find(lambda _: print("locfn", car(_), f) or car(_)(f), locfns))):
        return cadr(it)(f, args, a, s, r, m)
    elif callable(f):
        return applyfunc(f, args, a, s, r, m)
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
                return applyclo(parms, args, env, body, s, r, m)
            else:
                return sigerr(quote("bad-clo"), s, r, m)
        def do_mac():
            return applym(f, map(lambda _: list(quote("quote"), _), args), a, s, r, m)
        def do_cont():
            s2, r2, extra = car(rest), cadr(rest), cddr(rest)
            if yes(okstack(s2)) and yes(proper(r2)):
                return applycont(s2, r2, args, s, r, m)
            else:
                return sigerr(quote("bad-cont"), s, r, m)
        def do_virfns():
            if yes(it := get(tag, virfns, unset, car(inwhere(s)))):
                e = cdr(it)(f, map(lambda _: list(quote("quote"), _), args))
                return mev(cons(list(e, a), s), r, m)
            else:
                return sigerr(quote("unapplyable"), s, r, m)
        return case_f(
            tag,
            quote("prim"), do_prim,
            quote("clo"), do_clo,
            quote("mac"), do_mac,
            quote("cont"), do_cont,
            do_virfns)()

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
    # print("loc_is_get", _f, args, car(s), car(r))
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
def applyclo(parms, args, env, body, s, r, m):
    return mev(cons(fu(env)(lambda s, r, m: pass_(parms, args, env, s, r, m)),
                    fu(env)(lambda s, r, m: mev(cons(list(body, car(r)), s),
                                                cdr(r),
                                                m)),
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
def pass_(pat, arg, env, s, r, m):
    pat = evsyntax(pat)
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
        return typecheck(cdr(pat), arg, env, s, r, m)
    elif yes(caris(pat, o)):
        return pass_(cadr(pat), arg, env, s, r, m)
    else:
        return destructure(pat, arg, env, s, r, m)

# (def typecheck ((var f) arg env s r m)
#   (mev (cons (list (list f (list 'quote arg)) env)
#              (fu (s r m)
#                (if (car r)
#                    (pass var arg env s (cdr r) m)
#                    (sigerr 'mistype s r m)))
#              s)
#        r
#        m))
def typecheck(var_f, arg, env, s, r, m):
    var, f = car(var_f), cadr(var_f)
    return mev(cons(list(list(f, list(quote("quote"), arg)), env),
                    fu(env)((lambda s, r, m:
                             if_f(lambda: car(r),
                                  lambda: pass_(var, arg, env, s, cdr(r), m),
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
def destructure(p_ps, arg, env, s, r, m):
    p, ps = car(p_ps), cdr(p_ps)
    if no(arg):
        if yes(caris(p, o)):
            return mev(cons(list(caddr(p), env),
                            fu(env)(lambda s, r, m: pass_(cadr(p), car(r), env, s, cdr(r), m)),
                            fu(env)(lambda s, r, m: pass_(ps, nil, car(r), s, cdr(r), m)),
                            s),
                       r,
                       m)
        else:
            return sigerr(quote("underargs"), s, r, m)
    elif yes(atom(arg)):
        return sigerr(quote("atom-arg"), s, r, m)
    else:
        return mev(cons(fu(env)(lambda s, r, m: pass_(p, car(arg), env, s, r, m)),
                        fu(env)(lambda s, r, m: pass_(ps, cdr(arg), car(r), s, cdr(r), m)),
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
#
# (def con (x)
#   (fn args x))
def con(x):
    return lambda *args, **kws: x

# (def compose fs
#   (reduce (fn (f g)
#             (fn args (f (apply g args))))
#           (or fs (list idfn))))
def compose(*fs):
    return reduce(compose2, fs or (idfn,))

def compose2(f, g):
    @functools.wraps(g)
    def f_then_g(*args, **kws):
        return f(apply(g, args, **kws))
    f_name = getattr(f, "__qualname__", getattr(f, "__name__"))
    g_name = getattr(g, "__qualname__", getattr(g, "__name__"))
    f_then_g.__qualname__ = f_then_g.__name__ = f"{f_name}:{g_name}"
    return f_then_g

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
    if yes(cor(variable, atom)(v)):
        return list(quote("let"), v, list(quote("uvar")), *body)
    else:
        return list(quote("with"), fuse(lambda _: list(_, list(quote("quote"), list(quote("uvar")))), v),
                    *body)

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
    return list(quote("if"), expr, list(quote("do"), *body))

# (mac unless (expr . body)
#   `(when (no ,expr) ,@body))
@macro_
def unless(expr, *body):
    return list(quote("when"), list(quote("no"), expr), *body)

# (mac bind (var expr . body)
#   `(dyn ,var ,expr (do ,@body)))
@macro_
def bind(var, expr, *body):
    return list(quote("dyn"), var, expr, list(quote("do"), *body))

# (mac atomic body
#   `(bind lock t ,@body))
@macro_
def atomic(*body):
    return list(quote("bind"), quote("lock"), quote("t"), *body)

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
        return list(quote("atomic"), list(quote("let"), v, e,
                                          list(quote("let"), list(quote("cell"), quote("loc")), list(quote("where"), p, quote("t")),
                                               list(list(quote("case"), quote("loc"), quote("a"), quote("xar"), quote("d"), quote("xdr")), quote("cell"), v))))
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

# import sys
# sys.setrecursionlimit(10_000)

# >>> bel(list("join", 1, 2), map(lambda _: cons(_, list(quote("lit"), quote("prim"), _)), apply(append, prims)))
# (1 . 2)
# >>> bel(list("dyn", "err", list("fn", list("x"), list("do", list("print", "x"), list("quote", "hello"))), list("car", list("quote", "b"))))
# 'str' object has no attribute 'car'
# 'hello'

def vec2list(v: Optional[List[T]]) -> Optional[Union[T, ConsList[T]]]:
    if py.isinstance(v, (py.list, py.tuple)):
        if len(v) >= 3 and v[-2] == ".":
            l = vec2list(v[0:-2])
            xdr(l[-1], vec2list(v[-1]))
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

def readallbel(string):
    return vec2list(reader.read_all(reader.stream(string, mode="bel")))

def readbel(string):
    return vec2list([0])

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

def quoted(x):
    if stringp(x):
        return escape(x)
    elif atom(x):
        return x
    elif not proper(x):
        return cons(quote("cons"), map(quoted, mkproper(x)))
    else:
        return cons(quote("list"), map(quoted, x))

def callbel(f, *args):
    return bel(cons(f, quoted(args)))

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

def lastcdr(l):
    assert consp(l)
    return l[-1]

def add(l, x):
    cell = lastcdr(l)
    assert null(cdr(cell))
    return xdr(cell, list(x))

def last(l):
    assert consp(l)
    return car(l[-1])

def step(form):
    if not null(form):
        tail = nil
        for tail in XCONS(form):
            yield car(tail)
        # if not null(it := cdr(tail)) and atom(it):
        #     yield


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
    xs = list(list("cons"))
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
    return cons(quote("append"), xs) if len(xs) > 1 else car(xs)

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
def vir_number(f, args):
    return cons(quote("nth"), f, args)

# (def table ((o kvs))
#   `(lit tab ,@kvs))
def table(kvs=unset) -> py.dict:
    if kvs is unset:
        kvs = nil
    return py.dict([(car(x), cdr(x)) for x in (vec2list(kvs).list() if not null(kvs) else [])])

#
# (vir tab (f args)
#   `(tabref ,f ,@args))
@vir_(quote("tab"))
def vir_table(f, args):
    return cons(quote("tabref"), f, args)

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





