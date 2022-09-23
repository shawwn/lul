from __future__ import annotations

import contextlib

from .runtime import *
import json

# noinspection PyCompatibility
import __main__ as G


# def no(x):
#     return id(x, nil) or falsep(x)

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

# (def list args
#   (append args nil))
def list(*args):
    # return append(args, nil)
    return XCONS(args)

# (def map (f . ls)
#   (if (no ls)       nil
#       (some no ls)  nil
#       (no (cdr ls)) (cons (f (car (car ls)))
#                           (map f (cdr (car ls))))
#                     (cons (apply f (map car ls))
#                           (apply map f (map cdr ls)))))
def map(f, *ls):
    if no(ls):
        return nil
    elif yes(some(no, ls)):
        return nil
    elif no(cdr(ls)):
        return cons(f(car(car(ls))),
                    map(f, cdr(car(ls))))
    else:
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
    return reduce(lambda x, y: list(list(quote("fn"), uvar(), y), x),
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
    return no(x) or (pair(x) and proper(cdr(x)))

# (def string (x)
#   (and (proper x) (all char x)))
def string(x):
    if not proper(x):
        return nil
    else:
        return all(char, x)


# (def mem (x ys (o f =))
#   (some [f _ x] ys))
@dispatch(1)
def mem(x, ys, f=unset):
    if f is unset:
        f = equal
    return some(lambda _: f(_, x), ys)

@mem.register(std.Mapping)
def mem_Mapping(x, ys: Mapping, f=unset):
    if f in [unset, id]:
        if x in ys:
            return Cell(ys, x)
    for k, v in ys.items():
        if yes(f(it := Cell(ys, k), x)):
            return it
    return nil

# (def in (x . ys)
#   (mem x ys))
def in_(x, *ys):
    return mem(x, ys)

# (def cadr  (x) (car (cdr x)))
def cadr(x):
    return car(cdr(x))

# (def cddr  (x) (cdr (cdr x)))
def cddr(x):
    return cdr(cdr(x))

# (def caddr (x) (car (cddr x)))
def caddr(x):
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


def case_f(expr, *args):
    if no(cdr(args)):
        return car(args)()
    else:
        if equal(expr, car(args)):
            return cadr(args)()
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
@dispatch()
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

@hug.register(std.Mapping)
def hug_Mapping(xs: Mapping, f=unset):
    if f is unset:
        f = list
    return py.tuple(f(k, v) for k, v in xs.items())

# (mac with (parms . body)
#   (let ps (hug parms)
#     `((fn ,(map car ps) ,@body)
#       ,@(map cadr ps))))
#
# (def keep (f xs)
#   (if (no xs)      nil
#       (f (car xs)) (cons (car xs) (keep f (cdr xs)))
#                    (keep f (cdr xs))))
@dispatch(1)
def keep(f, xs):
    if no(xs):
        return nil
    elif yes(f(car(xs))):
        return cons(car(xs), keep(f, cdr(xs)))
    else:
        return keep(f, cdr(xs))

@keep.register(std.Mapping)
def keep_Mapping(f, xs: Mapping):
    return {k: v for k, v in xs.items() if yes(f(Cell(xs, k)))}

# (def rem (x ys (o f =))
#   (keep [no (f _ x)] ys))
def rem(x, ys, f=unset):
    if f is unset:
        f = equal
    return keep(lambda _: no(f(_, x)), ys)

# (def get (k kvs (o f =))
#   (find [f (car _) k] kvs))
@dispatch(1)
def get(k, kvs, f=unset):
    if f is unset:
        f = equal
    if null(kvs):
        return kvs
    if isinstance(kvs, Cons):
        # return find(lambda _: f(car(_), k), kvs)
        for tail in kvs:
            x = car(tail)
            if f(car(x), k):
                return x
        if dictp(it := cdr(kvs[-1])):
            kvs = it
        else:
            return nil
    assert f in [equal, id]
    # if isinstance(kvs, dict):
    #     return join(k, kvs[k]) if k in kvs else nil
    # return join(k, getattr(kvs, k)) if hasattr(kvs, k) else nil
    out = Cell(kvs, k, unset)
    if cdr(out) is unset:
        return nil
    return out

@get.register(std.Mapping)
def get_Mapping(k, kvs: Mapping, f=unset):
    if f in [unset, id]:
        if k in kvs:
            return Cell(kvs, k)
    else:
        return get(k, XCONS(kvs), f)

# (def put (k v kvs (o f =))
#   (cons (cons k v)
#         (rem k kvs (fn (x y) (f (car x) y)))))
@dispatch(2)
def put(k, v, kvs, f=unset):
    if f is unset:
        f = equal
    return cons(cons(k, v),
                rem(k, kvs, lambda x, y: f(car(x), y)))

@put.register(std.Mapping)
def put_Mapping(k, v, kvs: Mapping, f=unset):
    if f is unset:
        return {**kvs, **{k: v}}
    it = put(k, v, XCONS(kvs), f)
    if null(it):
        return {}
    return {car(x): cdr(x) for x in it.list()}

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
def is_(x):
    return lambda _: equal(_, x)

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
    elif in_(type(e), quote("char"), quote("stream")):
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
    else:
        return string(e)

def string_literal_p(e):
    if not isinstance(e, str):
        return False
    if len(e) <= 0:
        return False
    return e[0].isdigit() or (e.startswith('"') and e.endswith('"'))

def evliteral(e):
    if string_literal_p(e) and reader.read_from_string(e, more=object())[0] == e:
        return json.loads(e)
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
    return lambda _: begins(_, list(quote("lit"), name), id)

# (def bel (e (o g globe))
#   (ev (list (list e nil))
#       nil
#       (list nil g)))
def bel(e, g=unset, a=unset):
    if a is unset:
        # a = nil
        # a = XCONS(G.__dict__)
        a = G.__dict__
    if g is unset:
        # g = globe()
        # g = nil
        # g = {**py.__dict__, **globals()}
        # g = XCONS(globals())
        # g = append(XCONS(G.__dict__), XCONS(globals()))
        g = globals()
    return ev(list(list(e, a)),
              nil,
              list(nil, g))

# (def mev (s r (p g))
#   (if (no s)
#       (if p
#           (sched p g)
#           (car r))
#       (sched (if (cdr (binding 'lock s))
#                  (cons (list s r) p)
#                  (snoc p (list s r)))
#              g)))
def mev_(s, r, pg):
    p, g = car(pg), cadr(pg)
    if no(s):
        if yes(p):
            return sched(p, g)
        else:
            return car(r)
    else:
        return sched(cons(list(s, r), p)
                     if yes(cdr(binding(quote("lock"), s))) else
                     snoc(p, list(s, r)),
                     g)

mev_tail = CV.ContextVar[bool]("mev_tail", default=False)

class JumpToMev(Exception):
    def __init__(self, s, r, pg):
        self.s = s
        self.r = r
        self.pg = pg

def mev(s, r, pg):
    if mev_tail.get():
        raise JumpToMev(s, r, pg)
    else:
        reset = mev_tail.set(True)
        try:
            while True:
                try:
                    return mev_(s, r, pg)
                except JumpToMev as e:
                    s = e.s
                    r = e.r
                    pg = e.pg
        finally:
            mev_tail.reset(reset)

# (def sched (((s r) . p) g)
#   (ev s r (list p g)))
def sched(sr_p, g):
    s, r, p = car(car(sr_p)), cadr(car(sr_p)), cdr(sr_p)
    return ev(s, r, list(p, g))

# (def ev (((e a) . s) r m)
#   (aif (literal e)            (mev s (cons e r) m)
#        (variable e)           (vref e a s r m)
#        (no (proper e))        (sigerr 'malformed s r m)
#        (get (car e) forms id) ((cdr it) (cdr e) a s r m)
#                               (evcall e a s r m)))
def ev(ea_s, r, m):
    e, a, s = car(car(ea_s)), cadr(car(ea_s)), cdr(ea_s)
    if yes(literal(e)):
        return mev(s, cons(evliteral(e), r), m)
    elif yes(variable(e)):
        return vref(e, a, s, r, m)
    elif no(proper(e)):
        return sigerr(quote("malformed"), s, r, m)
    elif yes(it := get(car(e), forms, id)):
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
def vref(v, a, s, r, m):
    g = cadr(m)
    if inwhere(s):
        if yes(it := or_f(lambda: lookup(v, a, s, g),
                          lambda: and_f(lambda: car(inwhere(s)),
                                        lambda: gset(v, nil, g)))):
            # lambda: [cell := cons(v, nil),
            #          xdr(g, cons(cell, cdr(g))),
            #          cell][-1]))):
            #     lambda cell: ([xdr(g, cons(cell, cdr(g)))] and cell))
            # # (lambda cell: (lambda a, b: b)(xdr(g, cons(cell, cdr(g))), cell))(
            # #     cons(v, nil))
            # ))):
            return mev(cdr(s), cons(list(it, quote("d")), r), m)
        else:
            if yes(it := lookup(v, a, s, g)):
                return mev(s, cons(cdr(it), r), m)
            else:
                return sigerr(list(quote("unboundb"), v), s, r, m)
    else:
        if yes(it := lookup(v, a, s, g)):
            return mev(s, cons(cdr(it), r), m)
        else:
            return sigerr(list(quote("unboundb"), v), s, r, m)

def gset(k, v, kvs):
    assert not null(kvs)
    if consp(kvs):
        cell = cons(k, v)
        xdr(kvs, cons(cell, cdr(kvs)))
    else:
        cell = Cell(kvs, k, nil)
        xdr(cell, v)
    #return list(cell, quote("d"))
    return cell

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
def lookup(e, a, s, g):
    return or_f(lambda: binding(e, s),
                lambda: get(e, a, id),
                lambda: get(e, g, id),
                lambda: (cons(e, a) if id(e, quote("scope")) else
                         cons(e, g) if id(e, quote("globe")) else
                         nil))

# (def binding (v s)
#   (get v
#        (map caddr (keep [begins _ (list smark 'bind) id]
#                         (map car s)))
#        id))
def binding(v, s):
    return get(v,
               map(caddr, keep(lambda _: begins(_, list(smark, quote("bind")), id),
                               map(car, s))),
               id)

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

# (mac fu args
#   `(list (list smark 'fut (fn ,@args)) nil))
def fu(f):
    return list(list(smark, quote("fut"), f), nil)

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
                                         fu(lambda s, r, m: mev(s, cdr(r), m)),
                                         s),
                                    r,
                                    m)),
        lambda: sigerr(quote("unknown-mark"), s, r, m))

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
                        cons(fu(lambda s, r, m: if2(cdr(es), a, s, r, m)),
                             s) if yes(cdr(es)) else s),
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
    return mev(cons(list(car(es) if yes(car(r)) else cons(quote("if"), cdr(es)),
                         a),
                    s),
               cdr(r),
               m)

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
                    list(list(smark, quote("loc"), new), nil),
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
                        fu(lambda s, r, m: dyn2(v, e2, a, s, r, m)),
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
                         nil),
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
def thread(es, a, s, r, pg):
    e = car(es)
    p, g = car(pg), cadr(pg)
    return mev(s,
               cons(nil, r),
               list(cons(list(list(list(e, a)),
                              nil),
                         p),
                    g))

# (def evcall (e a s r m)
#   (mev (cons (list (car e) a)
#              (fu (s r m)
#                (evcall2 (cdr e) a s r m))
#              s)
#        r
#        m))
def evcall(e, a, s, r, m):
    return mev(cons(list(car(e), a),
                    fu(lambda s, r, m: evcall2(cdr(e), a, s, r, m)),
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
        @fu
        def f(s, r, m):
            args_r2 = snap(es, r)
            args, r2 = car(args_r2), cadr(args_r2)
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
                  cons(fu(lambda s, r, m: mev(cons(list(car(r), a), s),
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
    elif caris(f, quote("lit")):
        if proper(f):
            return applylit(f, args, a, s, r, m)
        else:
            return sigerr(quote("bad-lit"), s, r, m)
    elif callable(f):
        try:
            return mev(s, cons(apply(f, args), r), m)
        except Exception as v:
            return sigerr(v, s, r, m)
    else:
        return sigerr(quote("cannot-apply"), s, r, m)

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
    if yes(inwhere(s)) and yes(it := find(lambda _: car(_)(f), locfns)):
        return cadr(it)(f, args, a, s, r, m)
    else:
        _ = cdr(f)
        tag, rest = car(_), cdr(_)
        def do_clo():
            env, parms, body, extra = car(rest), cadr(rest), caddr(rest), cdr(cddr(rest))
            if yes(okenv(env)) and yes(okparms(parms)):
                return applyclo(parms, args, env, body, s, r, m)
            else:
                return sigerr(quote("bad-clo"), s, r, m)
        def do_cont():
            s2, r2, extra = car(rest), cadr(rest), cddr(rest)
            if yes(okstack(s2)) and yes(proper(r2)):
                return applycont(s2, r2, args, s, r, m)
            else:
                return sigerr(quote("bad-cont"), s, r, m)
        def do_virfns():
            if yes(it := get(tag, virfns)):
                e = cdr(it)(f, map(lambda _: list(quote("quote"), _), args))
                return mev(cons(list(e, a), s), r, m)
            else:
                return sigerr(quote("unapplyable"), s, r, m)
        return case_f(
            tag,
            quote("prim"), (lambda: applyprim(car(rest), args, s, r, m)),
            quote("clo"), do_clo,
            quote("mac"), (lambda: applym(f, map(lambda _: list(quote("quote"), _), args), a, s, r, m)),
            quote("cont"), do_cont,
            do_virfns)

# (set virfns nil)
virfns = nil

# (mac vir (tag . rest)
#   `(set virfns (put ',tag (fn ,@rest) virfns)))

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
@loc_(is_(car))
def loc_is_car(_f, args, _a, s, r, m):
    return mev(cdr(s), cons(list(car(args), quote("a")), r), m)

# (loc (is cdr) (f args a s r m)
#   (mev (cdr s) (cons (list (car args) 'd) r) m))
@loc_(is_(cdr))
def loc_is_cdr(_f, args, _a, s, r, m):
    return mev(cdr(s), cons(list(car(args), quote("d")), r), m)

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
        return and_f((lambda: oktoparm(car(p)) if yes(caris(car(p), o)) else okparms(car(p))),
                     (lambda: okparms(cdr(p))))

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
prims = list(list(quote("id"), quote("join"), quote("xar"), quote("xdr"), quote("xrb"), quote("ops"), quote("print")),
             list(quote("car"), quote("cdr"), quote("type"), quote("sym"), quote("nom"), quote("rdb"), quote("cls"), quote("stat"), quote("sys")),
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
                           # quote("wrb"), lambda: wrb(a, b),
                           # quote("rdb"), lambda: rdb(a),
                           # quote("ops"), lambda: ops(a, b),
                           # quote("cls"), lambda: cls(a),
                           # quote("stat"), lambda: stat(a),
                           # quote("coin"), lambda: coin(),
                           # quote("sys"), lambda: sys(a),
                           quote("print"), lambda: print(a),
                           lambda: sigerr(quote("bad-prim"), s, r, m))
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
    return mev(cons(fu(lambda s, r, m: pass_(parms, args, env, s, r, m)),
                    fu(lambda s, r, m: mev(cons(list(body, car(r)), s),
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
                    fu((lambda s, r, m:
                        pass_(var, arg, env, s, cdr(r), m)
                        if yes(car(r)) else
                        sigerr(quote("mistype"), s, r, m))),
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
                            fu(lambda s, r, m: pass_(cadr(p), car(r), env, s, cdr(r), m)),
                            fu(lambda s, r, m: pass_(ps, nil, car(r), s, cdr(r), m)),
                            s),
                       r,
                       m)
        else:
            return sigerr(quote("underargs"), s, r, m)
    elif yes(atom(arg)):
        return sigerr(quote("atom-arg"), s, r, m)
    else:
        return mev(cons(fu(lambda s, r, m: pass_(p, car(arg), env, s, r, m)),
                        fu(lambda s, r, m: pass_(ps, cdr(arg), car(r), s, cdr(r), m)),
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
#
# (def compose fs
#   (reduce (fn (f g)
#             (fn args (f (apply g args))))
#           (or fs (list idfn))))
#
# (def combine (op)
#   (fn fs
#     (reduce (fn (f g)
#               (fn args
#                 (op (apply f args) (apply g args))))
#             (or fs (list (con (op)))))))
#
# (set cand (combine and)
#      cor  (combine or))
#
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
#
# (def upon args
#   [apply _ args])
#
# (def pairwise (f xs)
#   (or (no (cdr xs))
#       (and (f (car xs) (cadr xs))
#            (pairwise f (cdr xs)))))
#
# (def fuse (f . args)
#   (apply append (apply map f args)))
#
# (mac letu (v . body)
#   (if ((cor variable atom) v)
#       `(let ,v (uvar) ,@body)
#       `(with ,(fuse [list _ '(uvar)] v)
#          ,@body)))
#
# (mac pcase (expr . args)
#   (if (no (cdr args))
#       (car args)
#       (letu v
#         `(let ,v ,expr
#            (if (,(car args) ,v)
#                ,(cadr args)
#                (pcase ,v ,@(cddr args)))))))
#
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
        if len(x) <= 1:
            p, e = car(x), t
        else:
            p, e = car(x), cadr(x)
        v = uvar()
        return list(quote("atomic"), list(quote("let"), v, e,
                                          list(quote("let"), list(quote("cell"), quote("loc")), list(quote("where"), p, quote("t")),
                                               list(list(quote("case"), quote("loc"), quote("a"), quote("xar"), quote("d"), quote("xdr")), quote("cell"), v))))
    return cons(quote("do"), map(f, hug(args)))



globals()["print"] = print
globals()["."] = lambda *args: cdr(get(*args))
globals()["+"] = operator.add
globals()["-"] = operator.sub
globals()["*"] = operator.mul
globals()["/"] = operator.truediv
globals()["//"] = operator.floordiv
eval = eval
exec = exec
compile = compile

# import sys
# sys.setrecursionlimit(10_000)

# >>> bel(list("join", 1, 2), map(lambda _: cons(_, list(quote("lit"), quote("prim"), _)), apply(append, prims)))
# (1 . 2)
# >>> bel(list("dyn", "err", list("fn", list("x"), list("do", list("print", "x"), list("quote", "hello"))), list("car", list("quote", "b"))))
# 'str' object has no attribute 'car'
# 'hello'

def vec2list(v):
    if py.isinstance(v, py.list):
        if len(v) >= 3 and v[-2] == ".":
            l = vec2list(v[0:-2])
            xdr(l, vec2list(v[-1]))
            return l
        return list(*[vec2list(x) for x in v])
    return quote(v)

# from lul.common import reader
# belforms = reader.read_all(reader.stream(open("bel.bel").read()))
# [print(repr(x.car)) for x in vec2list(belforms)]

def readbel(string, more=None):
    return vec2list(reader.read_from_string(string, mode="bel", more=more)[0])

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
        # reload(M)
        # noinspection PyBroadException
        try:
            self.exec(code, self.locals)
        except SystemExit:
            raise
        except:
            self.showtraceback()



@contextlib.contextmanager
def letattr(obj, key, val):
    prev = getattr(obj, key, unset)
    setattr(obj, key, val)
    try:
        yield
    finally:
        if prev is unset:
            delattr(obj, key)
        else:
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
    with letattr(sys, 'ps1', '> '):
        with letattr(sys, 'ps2', '  '):
            console.interact(banner, exitmsg)
