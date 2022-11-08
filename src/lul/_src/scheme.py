from .runtime import *
import inspect
import pydoc
import builtins as py
import re
import collections.abc
import functools

class classproperty(py.object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

def safelen(x, otherwise):
    try:
        return len(x)
    except TypeError:
        return otherwise

class Runtime:
    pass

class Scheme(Runtime):
    nil = None
    true = True
    false = False
    # @classproperty
    # def nil(cls):
    #     return None

    @classmethod
    def null(cls, x):
        return x is cls.nil or (isinstance(x, py.list) and len(x) == 0)

    @classmethod
    def exists(cls, x):
        return not cls.null(x)

    # @classproperty
    # def true(cls):
    #     return True

    @classmethod
    def truep(cls, x):
        return x is cls.true

    # @classproperty
    # def false(cls):
    #     return False

    @classmethod
    def falsep(cls, x):
        return x is cls.false

    @classmethod
    def no(cls, x):
        return cls.null(x) or cls.falsep(x)

    @classmethod
    def yes(cls, x):
        return not cls.no(x)

    @classmethod
    def booleanp(cls, x):
        return cls.truep(x) or cls.falsep(x)

    @classmethod
    def integerp(cls, x):
        return not cls.booleanp(x) and isinstance(x, int)

    @classmethod
    def floatp(cls, x):
        return not cls.booleanp(x) and isinstance(x, float)

    @classmethod
    def numberp(cls, x):
        return not cls.booleanp(x) and isinstance(x, (int, float))

    @classmethod
    def stringp(cls, x):
        return isinstance(x, str)

    @classmethod
    def listp(cls, x):
        return not cls.stringp(x) and isinstance(x, collections.abc.Sequence)

    @classmethod
    def tablep(cls, x):
        return isinstance(x, collections.abc.Mapping)

    @classmethod
    def consp(cls, x):
        return cls.null(x) or cls.listp(x)

    @classmethod
    def atom(cls, exp):
        #return pydoc.isdata(exp)
        if cls.listp(exp):
            return cls.false
        if cls.tablep(exp):
            return cls.false
        return cls.true

    @classmethod
    def either(cls, x, y):
        if cls.null(x):
            return y
        else:
            return x

    @classmethod
    def id(cls, x, y):
        return x is y

    @classmethod
    def equal(cls, x, y):
        if cls.null(x) and cls.null(y):
            return cls.true
        if cls.id(x, y):
            return cls.true
        if cls.booleanp(x) or cls.booleanp(y):
            return cls.false
        return x == y

    @classmethod
    def guard(cls, *errors):
        if not errors:
            errors = (Exception,)
        def wrapper(f):
            @functools.wraps(f)
            def func(*args, **kws):
                try:
                    return f(*args, **kws)
                except errors:
                    return cls.nil
            return func
        return wrapper

    @classmethod
    def errsafe(cls, f, *args, **kws):
        try:
            return f(*args, **kws)
        except Exception:
            return cls.nil

    @classmethod
    def readint(cls, x, base=nil):
        if isinstance(x, int):
            return x
        if cls.stringp(x):
            try:
                return int(x) if cls.null(base) else int(x, base)
            except ValueError:
                pass
        return cls.nil

    @classmethod
    def number(cls, x, base=nil):
        if cls.stringp(x):
            if cls.null(base):
                if m := re.match('^[-+]?0([xbo])', x, re.IGNORECASE):
                    spec = m.group(1)
                    base = dict(x=16, X=16, o=8, O=8, b=2, B=2)[spec]
        if cls.exists(it := cls.readint(x, base)):
            return it
        if cls.null(base):
            if cls.floatp(x):
                return x
            try:
                return float(x)
            except ValueError:
                pass
        return cls.nil

    @classmethod
    def uncompile_id(cls, id: str):
        id = id.replace('_', '-')
        return id

    @classmethod
    def symbolp(cls, x):
        return isinstance(x, str)

    @classmethod
    def symbol_name(cls, x) -> str:
        assert cls.symbolp(x)
        return x

    @classmethod
    def keywordp(cls, x):
        return cls.symbolp(x) and cls.symbol_name(x).startswith(':')

    @classmethod
    def intern(cls, k) -> str:
        assert cls.symbolp(k)
        return k

    @classmethod
    def keyword(cls, k):
        k = str(k)
        if not k.startswith(':'):
            k = ':' + k
        return cls.intern(k)

    @classmethod
    def readatom(cls, s: str):
        if s == 'nil':
            return cls.nil
        if s == 'true' or s == '#t':
            return cls.true
        if s == 'false' or s == '#f':
            return cls.false
        return cls.either(cls.number(s), s)

    @classmethod
    def unkeyword(cls, k):
        if cls.keywordp(k):
            assert k[0] == ':'
            k = k[1:]
        return cls.readatom(k)

    @classmethod
    def id_to_keyword(cls, id: str):
        return cls.keyword(cls.uncompile_id(id))

    @classmethod
    def list(cls, *args, **kwargs):
        l = py.list(args)
        for k, v in kwargs.items():
            key = cls.id_to_keyword(k)
            l.extend([key, v])
        # if len(l) <= 0:
        #     return cls.nil
        return l

    @classmethod
    def quote(cls, x):
        return x

    @classmethod
    def iterate(cls, l):
        if cls.null(l):
            pass
        elif cls.tablep(l):
            l: py.dict
            for k, v in l.items():
                yield k, v
        elif cls.listp(l):
            l: py.list
            unset = cls.cons("unset")
            key = unset
            i = 0
            for v in l:
                if cls.id(key, unset):
                    if cls.keywordp(v):
                        key = v
                        continue
                    else:
                        key = i
                        i += 1
                if key is not None:
                    yield key, v
                    key = unset
        else:
            raise TypeError(f"pairs() expected list or nil, but {type(l).__name__} found")

    @classmethod
    def pairs(cls, l):
        for k, v in cls.iterate(l):
            pass



    @classmethod
    def has(cls, l, idx, test=nil):
        test = cls.either(test, cls.id)
        for k, v in cls.pairs(l):
            if test(k, idx):
                return cls.list(k, v)
        return cls.nil

    @classmethod
    def has63(cls, l, idx):
        return not cls.null(cls.has(l, idx))

    @classmethod
    def get(cls, l, idx, *default):
        if not cls.null(it := cls.has(l, idx)):
            k, v = it
            return v
        if default:
            assert len(default) == 1
            return default[0]
        else:
            raise KeyError(idx)

    @classmethod
    def at(cls, l, i, *default):
        if not default:
            default = (cls.nil,)
        return cls.get(l, i, *default)

    @classmethod
    def cut(cls, l, start=nil, upto=nil):
        start = cls.either(start, 0)
        upto = cls.either(upto, cls.length(l))
        if cls.consp(l):
            ks = {}
            xs = []
            for k, v in cls.iterate(l):
                if cls.numberp(k):
                    if start <= k < upto:
                        print('ok', k, start, upto)
                        xs.append(v)
                    else:
                        print('nope', k, start, upto)
                else:
                    ks[k] = v
            return cls.list(*xs, **ks)
        else:
            return l[start:upto]

    @classmethod
    def length(cls, l, upto=nil):
        try:
            n = -1
            for k, v in cls.pairs(l):
                if cls.numberp(k):
                    if k > n:
                        n = k
                        if upto is not None and n >= upto:
                            break
            return n + 1
        except TypeError:
            return len(l)

    @classmethod
    def unstash(cls, l):
        ks = []
        xs = {}
        lo = cls.nil
        hi = cls.nil
        for k, v in cls.pairs(l):
            if cls.numberp(k):
                lo = cls.either(lo, k)
                hi = cls.either(hi, k)
                if k < lo:
                    lo = k
                if k > hi:
                    hi = k
                xs[k] = v
            else:
                ks.append(k, v)
        return sorted(xs) + ks

    @classmethod
    def iterate(cls, l):
        return self.unstash(l)

    @classmethod
    def ipairs(cls, l):
        i = 0
        for k, v in cls.pairs(l):
            if cls.numberp(k):
                if k != i:
                    return
                yield k, v
                i += 1
        # n = length(l)
        # for i in range(n):
        #     v = at(l, i)
        #     yield i, v

    @classmethod
    def none(cls, l):
        return cls.length(l, 0) == 0

    @classmethod
    def some(cls, l):
        return cls.length(l, 0) > 0

    @classmethod
    def one(cls, l):
        return cls.length(l, 1) == 1

    @classmethod
    def two(cls, l):
        return cls.length(l, 2) == 2

    @classmethod
    def car(cls, x):
        if cls.null(x):
            return x
        return cls.at(x, 0)

    @classmethod
    def cdr(cls, x):
        return cls.cut(x, 1)

    @classmethod
    def join(cls, x=nil, y=nil):
        return cls.list(x, *cls.either(y, ()))

    @classmethod
    def reduce(cls, f, xs):
        if cls.null(cls.cdr(xs)):
            return cls.car(xs)
        else:
            return f(cls.car(xs), cls.reduce(f, cls.cdr(xs)))

    @classmethod
    def cons(cls, *args):
        return cls.reduce(cls.join, args)

    @classmethod
    def lookup(cls, name):
        if isinstance(name, str):
            try:
                return cls.list(name, getattr(cls, name))
            except AttributeError:
                return cls.nil

    @classmethod
    def apply(cls, f, args=nil, kws=nil):
        args = cls.either(args, [])
        kws = cls.either(kws, {})
        if isinstance(f, str):
            f = getattr(cls, f)

        return cls.reduce(cls.join, args)






