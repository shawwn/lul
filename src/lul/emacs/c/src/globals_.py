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
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        # next two lines make DynamicClassAttribute act the same as property
        self.__doc__ = doc or fget.__doc__
        self.overwrite_doc = doc is None

    def __get__(self, obj, owner):
        if self.fget is None:
            raise AttributeError("can't get attribute")
        return self.fget(owner)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(instance)

    def getter(self, fget):
        fdoc = fget.__doc__ if self.overwrite_doc else None
        result = type(self)(fget, self.fset, self.fdel, fdoc or self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result

    def setter(self, fset):
        result = type(self)(self.fget, fset, self.fdel, self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result

    def deleter(self, fdel):
        result = type(self)(self.fget, self.fset, fdel, self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result

class ClassPropertyMetaClass(type):
    def __setattr__(self, key, value):
        obj = self.__dict__.get(key, None)
        if obj and type(obj) is ClassPropertyDescriptor:
            return obj.__set__(self, value)
        return super(ClassPropertyMetaClass, self).__setattr__(key, value)

class ClassPropertyDescriptor(metaclass=ClassPropertyMetaClass):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    # def __get__(self, obj, klass=None):
    #     if klass is None:
    #         klass = type(obj)
    #     return self.fget.__get__(obj, klass)()
    def __get__(self, obj, owner):
        return self.fget(owner)

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    # if not isinstance(func, (classmethod, staticmethod)):
    #     func = classmethod(func)

    return ClassPropertyDescriptor(func)



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
