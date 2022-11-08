from functools import partial

class DEFVAR(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)


class DEFUN(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        print('DEFUN.__get__', self, obj, owner)
        if obj is None:
            return partial(self.f, owner)
        else:
            return partial(self.f, obj)


class Runtime:
    def __init_subclass__(cls, namespace, **kwargs):
        super().__init_subclass__(**kwargs)
        print(f"Called __init_subclass__({cls}, {namespace}, {kwargs!r})")
        print(cls.__dict__)
        cls.namespace = namespace


class Scheme(Runtime, namespace="scheme"):
    @DEFUN
    def print(self, *args, **kws):
        print(*args, **kws)
    @DEFUN
    def baz(self, *args, **kws):
        self.print('baz', self, *args, kws)
        return "Baz"
    @DEFVAR
    def omg(self):
        return 42
    @DEFUN
    def foo(self, *args, **kws):
        return self.baz(self.omg, *args, **kws)


Scheme.foo(1,2,3)

scm = Scheme()
scm.omg = 99
scm.foo(1,2,3)
