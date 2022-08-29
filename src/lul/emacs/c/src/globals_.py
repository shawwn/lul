from ...c import *
if TYPE_CHECKING:
    from .lisp import *

import inspect
import sys as _sys
import re
import keyword
import enum

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


def Q_(*val):
    global Q
    if val:
        Q = val[0]
    return Q

def V_(*val):
    global V
    if val:
        V = val[0]
    return V

def F_(*val):
    global F
    if val:
        F = val[0]
    return F

Q = None
class Q(Singleton):
    __singletons__ = []

V = None
class V(Singleton):
    __singletons__ = []

F = None
class F(Singleton):
    __singletons__ = []

S = None
class S(Singleton):
    __singletons__ = []

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)


# @G_
# class G(G):
@mixin(G)
class G:
    @classproperty
    def lispsym(cls):
        out = []
        # for k, v in Q.__dict__.items():
        for k, v in inspect.getmembers(Q):
            if isinstance(v, cls.Lisp_Symbol):
                out.append(v)
        return out

def init_globals(globals):
    globals['PP'] = PP_()
    globals['G'] = G_()
    globals['F'] = F_()
    globals['V'] = V_()
    globals['Q'] = Q_()
