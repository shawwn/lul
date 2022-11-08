from argparse import Namespace
from collections import ChainMap
from collections.abc import MutableMapping
import namez

class Runtime:
    @classmethod
    def is_fqn(cls, name):
        return isinstance(name, str) and '/' in name
    @classmethod
    def fqn(cls, x):
        return x if cls.is_fqn(x) else namez.name(x)
    @classmethod
    def id(cls, x):
        ns, _, name = cls.fqn(x).rpartition('/')
        return ns, name
    @classmethod
    def (cls, x):
        return cls.id(x)[-1]
    @classmethod
    def dirname(cls, x):
        return cls.id(x)[0]

class Frame(Namespace, MutableMapping, Runtime):
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __delitem__(self, key):
        try:
            delattr(self, key)
        except AttributeError:
            raise KeyError(key)
    def __iter__(self):
        return iter(self.__dict__)
    def __len__(self):
        return len(self.__dict__)

    def defconst(self, name=None):
        def func(x):
            if name is None:
                name = self.basename(x)
            self[name] = x
            return x
        return func

    def defvar(self, name=None):
        def func(x):
            if name is None:
                name = self.basename(x)
            if name in self:
                return self[name]
            else:
                self[name] = x
                return x
        return func

class Env(Frame):
    @classmethod
    def inst(cls):
        if not hasattr(cls, '__inst__'):
            cls.__inst__ = Env()
        return cls.__inst__

def env(EnvType=None):
    if EnvType is None:
        EnvType = Env
    return EnvType.inst()

class Symbol:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def get(cls, name):
        return

def listp(x):
    return isinstance(x, list)

def hash_table_p(x):
    return isinstance(x, dict)

def integerp(x):
    return isinstance(x, int)

def gethash(k, h):
    return h[k]

def eq(x, y):
    return x is y

