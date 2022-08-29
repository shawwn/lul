from typing import *
import ctypes as C
import sys as _sys
import abc
import contextlib
if TYPE_CHECKING:
    import types

false = False
true = True
NULL = None
gid_t = C.c_int

ptrdiff_t = int


def c_limits(c_int_type):
    signed = c_int_type(-1).value < c_int_type(0).value
    bit_size = C.sizeof(c_int_type) * 8
    signed_limit = 2 ** (bit_size - 1)
    return (-signed_limit, signed_limit - 1) if signed else (0, 2 * signed_limit - 1)


INT_MIN, INT_MAX = c_limits(C.c_int)
LONG_MIN, LONG_MAX = c_limits(C.c_long)
LLONG_MIN, LLONG_MAX = c_limits(C.c_longlong)

UINTPTR_MIN, UINTPTR_MAX = c_limits(C.c_size_t)
PTRDIFF_MIN, PTRDIFF_MAX = c_limits(C.c_ssize_t)

SIZE_MIN, SIZE_MAX = UINTPTR_MIN, UINTPTR_MAX

USHRT_MIN, USHRT_MAX = c_limits(C.c_ushort)
USHRT_WIDTH = 8 * C.sizeof(C.c_ushort)

@contextlib.contextmanager
def with_attr(obj, name, value):
    prev = object.__getattribute__(obj, name)
    object.__setattr__(obj, name, value)
    try:
        yield prev
    finally:
        object.__setattr__(obj, name, prev)

class SingletonMeta(type):
    def __getattr__(cls, name):
        if name != '__missing__' and hasattr(cls, '__missing__'):
            return cls.__missing__(cls, name)
        else:
            raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")


class Singleton(metaclass=SingletonMeta):
    __singletons__: ClassVar[List[Type]]
    def __init_subclass__(cls, **kwargs):
        register_singleton(cls.__singletons__, cls)



def register_singleton(singletons: List[type], cls: type):
    # cls.__singletons__ = singletons
    assert cls not in singletons
    singletons.append(cls)
    for current in singletons:
        M: types.ModuleType = _sys.modules[current.__module__]
        assert hasattr(M, current.__name__)
        setattr(M, current.__name__, cls)
    return cls

T = TypeVar("T")
U = TypeVar("U")

def mixin(cls: Type[U]):
    if not isinstance(cls, type):
        cls = type(cls)
    def inner(base: Type[T]) -> Type[U]:
        cls.__bases__ = (base, *cls.__bases__)
        return cls
    return inner
#
# class Mixin:
#     def __new__(cls, *bases, **kws):
#         def inner(cls: type):
#             cls.__bases__ = (*bases, *cls.__bases__)

def G_(*val):
    global G
    if val:
        G = val[0]
    return G

G = None
class G(Singleton):
    __singletons__ = []

def PP_(*val):
    global PP
    if val:
        PP = val[0]
    return PP

PP = None
class PP(Singleton):
    __singletons__ = []

def c_define(name, value=None):
    setattr(PP, name, value)

def c_defined(name):
    return hasattr(PP, name)

def c_undef(name):
    try:
        delattr(PP, name)
    except AttributeError:
        pass

def c_ifdef(name):
    return c_defined(name)

def c_ifndef(name):
    return not c_defined(name)

def assume(cond, globals, locals):
    pass

def strlen(s):
    return len(s)

def calloc(count: int, size: int):
    return bytearray(count * size)

def malloc(size: int):
    v = calloc(1, size)
    v[:] = b'\xFE' * size
    return v

def memset(dst: bytearray, dst_offset: int, val: int, size: int):
    dst[dst_offset:dst_offset+size] = val

def memcpy(dst: bytearray, dst_offset: int, src: Union[bytes, bytearray], src_offset: int, size: int):
    dst[dst_offset:dst_offset+size] = src[src_offset:src_offset+size]