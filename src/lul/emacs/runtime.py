from __future__ import annotations

from backport import typing as t

if t.TYPE_CHECKING:
    from . import buffer

import contextvars as CV
import contextlib
from backport import dataclasses
import abc

NoneType = type(None)
T = t.TypeVar("T")


# @dataclasses.dataclass
# class LispObject(Generic[T]):
#     # _: dataclasses.KW_ONLY
#     # value: Optional[T] = dataclasses.field(default_factory=lambda: Q.nil, kw_only=True)
#     pass

LispObject = object
LispCons = t.Tuple
# if TYPE_CHECKING:
#     LispCons = Tuple

TLisp = t.TypeVar("TLisp", bound=LispObject)

# class LispValue(LispObject[T]):
#     value: Optional[T] = dataclasses.field(default_factory=lambda: Q.nil, kw_only=True)


class lispfwd:
    pass


@dataclasses.dataclass
class LispSymbol(t.Generic[TLisp]):
    # name: str = dataclasses.field(default=dataclasses.MISSING, kw_only=True)
    name: str
    # /* Value of the symbol or Q.unbound if unbound.  Which alternative of the
    # 	 union is used depends on the `redirect' field above.  */
    @dataclasses.dataclass
    class _val(t.Generic[TLisp]):
        value: CV.ContextVar[TLisp] = None
        alias: LispSymbol = None
        blv: Lisp_Buffer_Local_Value[TLisp] = None
        fwd: lispfwd = None
    val: _val[TLisp]
    def __init__(self, name: str, **kws):
        super().__init__()
        self.name = name
        if len(kws) > 1:
            raise ValueError(f"Expected only one keyword, got {kws!r}")
        else:
            kws = dict(value=CV.ContextVar(name, default=Q.nil))
        self.val = self._val(**kws)


# zz: LispSymbol[str] = LispSymbol("foo", value="bar")

# struct Lisp_Buffer_Local_Value
#   {
#     /* True means that merely setting the variable creates a local
#        binding for the current buffer.  */
#     bool_bf local_if_set : 1;
#     /* True means that the binding now loaded was found.
#        Presumably equivalent to (defcell!=valcell).  */
#     bool_bf found : 1;
#     /* If non-NULL, a forwarding to the C var where it should also be set.  */
#     lispfwd fwd;	/* Should never be (Buffer|Kboard)_Objfwd.  */
#     /* The buffer for which the loaded binding was found.  */
#     Lisp_Object where;
#     /* A cons cell that holds the default value.  It has the form
#        (SYMBOL . DEFAULT-VALUE).  */
#     Lisp_Object defcell;
#     /* The cons cell from `where's parameter alist.
#        It always has the form (SYMBOL . VALUE)
#        Note that if `fwd' is non-NULL, VALUE may be out of date.
#        Also if the currently loaded binding is the default binding, then
#        this is `eq'ual to defcell.  */
#     Lisp_Object valcell;
#   };
@dataclasses.dataclass
class Lisp_Buffer_Local_Value(t.Generic[T]):
    #     /* The buffer for which the loaded binding was found.  */
    #     Lisp_Object where;
    where: LispObject
    #     /* A cons cell that holds the default value.  It has the form
    #        (SYMBOL . DEFAULT-VALUE).  */
    #     Lisp_Object defcell;
    defcell: LispCons[LispSymbol, T]
    #     /* The cons cell from `where's parameter alist.
    #        It always has the form (SYMBOL . VALUE)
    #        Note that if `fwd' is non-NULL, VALUE may be out of date.
    #        Also if the currently loaded binding is the default binding, then
    #        this is `eq'ual to defcell.  */
    #     Lisp_Object valcell;
    valcell: LispCons[LispSymbol, T]
    #     /* If non-NULL, a forwarding to the C var where it should also be set.  */
    #     lispfwd fwd;	/* Should never be (Buffer|Kboard)_Objfwd.  */
    fwd: lispfwd = None

# LispSymbol.register(type(None))
# LispSymbol.register(bool)
# LispSymbol.register(int)
# LispSymbol.register(float)
# LispSymbol.register(str)

# if True:
#     blv = Lisp_Buffer_Local_Value()
#     blv.valcell


class QMeta(type):
    def __setattr__(self, key, value):
        frozen = key in ['nil', 't', 'unbound']
        if frozen or not hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError("can't set attribute")


class Q(metaclass=QMeta):
    nil = None
    t = True
    unbound = CV.Token.MISSING
    current_buffer: LispSymbol[buffer.Buffer]
    current_indentation: LispSymbol[str]

global_obarray: t.List[LispSymbol] = []

def intern(name: str, obarray=None):
    if obarray is None:
        obarray = global_obarray
    for sym in obarray:
        if sym.name == name:
            return sym
    sym = LispSymbol(name=name)
    obarray.insert(0, sym)
    return sym

Q.current_buffer = intern("current-buffer")
Q.current_indentation = intern("current-indentation")


class VMeta(type):
    def __setattr__(self, key, value):
        current = getattr(self, key)
        if not isinstance(current, CV.ContextVar):
            raise AttributeError("can't set attribute")
        current.set(value)

class V(metaclass=VMeta):
    current_buffer: CV.ContextVar[buffer.Buffer] = CV.ContextVar(Q.current_buffer.name, default=Q.nil)
    current_indentation: CV.ContextVar[str] = CV.ContextVar(Q.current_indentation.name, default=Q.unbound)

# Qnil = None
# Qt = True
#
# Qerror = "error"
#
# Qunbound = ['%unbound']
# Qcurrent_buffer = "current-buffer"
# Qcurrent_indentation = "current-indentation"

class ElispError(Exception):
    def __init__(self, msg, *args):
        super().__init__(msg, *args)
        self.data = args

def signal(error_symbol, data):
    raise ElispError(error_symbol, data)

def NILP(x):
    return x is None

def EQ(x, y):
    return x is y

def STRINGP(x):
    return isinstance(x, str)

def SDATA(x: T) -> T:
    return x

def name2id(name: str):
    id = name.replace('-', '_')
    if not id.isidentifier():
        raise ValueError(f"Invalid name: {name!r}")
    return id

def lookup(x: t.Union[str, LispSymbol[T]]) -> CV.ContextVar[T]:
    if isinstance(x, LispSymbol):
        x = x.val.value
    if isinstance(x, CV.ContextVar):
        return x
    id = name2id(x)
    return getattr(V, id)

def get(name: t.Union[str, LispSymbol[T]]):
    try:
        val = lookup(name).get()
    except AttributeError:
        val = Q.unbound
    if val is Q.unbound:
        raise AttributeError(f"Symbol’s value as variable is void: {name}")
    return val

def set(name: t.Union[str, LispSymbol[T]], value: t.Optional[T] = None) -> t.Callable[[], None]:
    try:
        var = lookup(name)
    except AttributeError:
        var = CV.ContextVar(name, default=Q.unbound)
        setattr(V, name2id(name), var)
    token = var.set(value)
    def reset():
        nonlocal token
        if token is not None:
            var.reset(token)
            token = None
    return reset

@contextlib.contextmanager
def let(var: t.Union[str, CV.ContextVar[T]], value: t.Optional[T]):
    reset = set(var, value)
    try:
        yield value
    finally:
        reset()

def error(msg, value=None):
    if value is not None:
        msg = msg % value
    return signal(msg, RuntimeError)
