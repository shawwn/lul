from .globals_ import *
from .config import *
from .lisp import *

@mixin(V)
class V:
    quit_flag: bool = Q.nil
    inhibit_quit = Q.nil
    throw_on_input = Q.nil

from .alloc import *
from .lread import *
from .data import *
from .eval import *
from .fns import *
from .keyboard import *
from .atimer import *
from .puresize import *
# from .buffer import *
from .buffer_c import *
from .marker import *
from .print_ import *
from .editfns import *
