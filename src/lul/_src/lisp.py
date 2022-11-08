import builtins as py
import inspect

class Runtime:
    pass

class Lisp(Runtime):
    nil = None
    unset = inspect._empty
    true = True
    false = False

    @classmethod
    def null(cls, x):
        return x is cls.nil or (isinstance(x, py.list) and len(x) == 0)
