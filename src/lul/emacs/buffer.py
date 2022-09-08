from __future__ import annotations

from .runtime import *

@dataclasses.dataclass
class Buffer:
    name: t.Optional[str] = None
    contents: str = ""
    point: int = 1
    all_buffers: t.ClassVar[t.List[Buffer]] = []

    if t.TYPE_CHECKING:
        def __init__(self, name: str):
            ...

    def __repr__(self):
        if self.killed:
            return f'#<killed buffer>'
        else:
            return f'#<buffer {self.name}>'

    @property
    def killed(self):
        return self.name is None

    @killed.setter
    def killed(self, value):
        assert not self.killed
        if value:
            self.name = None

def bufferp(x):
    if isinstance(x, Buffer):
        return True

def XBUFFER(x) -> Buffer:
    assert bufferp(x)
    return x

def nsberror(spec):
    if STRINGP(spec):
        error("No buffer named %s", SDATA(spec))
    error("Invalid buffer argument")

def current_buffer() -> Buffer:
    return get(Q.current_buffer)

def buffer_hidden_p(buffer: Buffer):
    return buffer.name.startswith(' ')

def buffer_live_p(buffer: Buffer):
    return not buffer.killed

def _reap_buffer(buffer: Buffer):
    if buffer_live_p(buffer):
        return buffer
    try:
        Buffer.all_buffers.remove(buffer)
    except ValueError:
        pass

def live_buffers():
    for buffer in list(Buffer.all_buffers):
        if buffer := _reap_buffer(buffer):
            yield buffer

def visible_buffers():
    for buffer in live_buffers():
        if not buffer_hidden_p(buffer):
            yield buffer

def _decode_buffer(buffer: Buffer = None) -> Buffer:
    return current_buffer() if NILP(buffer) else buffer

def buffer_name(buffer: Buffer = None) -> str:
    return _decode_buffer(buffer).name

def get_buffer(buffer_or_name: t.Union[str, Buffer]) -> t.Optional[Buffer]:
    if bufferp(buffer_or_name):
        return buffer_or_name
    for buffer in live_buffers():
        if buffer.name == buffer_or_name:
            return buffer

def set_buffer(buffer_or_name: t.Union[str, Buffer]):
    buffer = get_buffer(buffer_or_name)
    if NILP(buffer):
        nsberror(buffer_or_name)
    if not buffer_live_p(buffer):
        error("Selecting deleted buffer")
    _set_buffer_internal(buffer)
    return buffer

def _set_buffer_internal(b: Buffer):
    if current_buffer() is not b:
        set(Q.current_buffer, b)

def generate_new_buffer_name(starting_name: str, ignore: t.Optional[str] = None):
    name = starting_name
    n = 2
    while get_buffer(name) is not None:
        if name == ignore:
            return name
        name = f'{starting_name}<{n}>'
        n += 1
    return name

def generate_new_buffer(name: str):
    return get_buffer_create(generate_new_buffer_name(name))

def get_buffer_create(buffer_or_name: t.Union[str, Buffer]):
    if it := get_buffer(buffer_or_name):
        return it
    assert isinstance(buffer_or_name, str)
    buffer = Buffer(buffer_or_name)
    Buffer.all_buffers.append(buffer)
    return buffer

def _candidate_buffer(b: Buffer, buffer: Buffer):
    """True if B can be used as 'other-than-BUFFER' buffer."""
    # return (BUFFERP(b) && !EQ (b, buffer)
    #        && BUFFER_LIVE_P(XBUFFER(b))
    #        && !BUFFER_HIDDEN_P(XBUFFER(b)));
    return (bufferp(b) and not EQ(b, buffer)
            and buffer_live_p(b)
            and not buffer_hidden_p(b))

def other_buffer(buffer: Buffer) -> Buffer:
    notsogood = None
    for buf in live_buffers():
        if _candidate_buffer(buf, buffer):
            notsogood = buf
    if not NILP(notsogood):
        return notsogood
    scratch = "*scratch*"
    if NILP(buf := get_buffer(scratch)):
        buf = get_buffer_create(scratch)
        # Fset_buffer_major_mode(buf);
    return buf

def kill_buffer(buffer_or_name: t.Union[str, Buffer] = None):
    if NILP(buffer_or_name):
        buffer = current_buffer()
    else:
        buffer = get_buffer(buffer_or_name)

    if NILP(buffer):
        nsberror(buffer_or_name)

    b = XBUFFER(buffer)

    # Avoid trouble for buffer already dead.
    if not buffer_live_p(b):
        return Q.nil

    # Make this buffer not be current.  Exit if it is the sole visible
    # buffer.
    if b is current_buffer():
        tem = other_buffer(buffer)
        set_buffer(tem)
        if b is current_buffer():
            return Q.nil

    b.killed = True
    return Q.t

@contextlib.contextmanager
def with_current_buffer(buffer_or_name: t.Union[str, Buffer]):
    prev = current_buffer()
    try:
        set_buffer(buffer_or_name)
        yield
    finally:
        set_buffer(prev)

def save_current_buffer():
    return with_current_buffer(current_buffer())

@contextlib.contextmanager
def with_temp_buffer():
    temp_buffer = generate_new_buffer(" *temp*")
    with with_current_buffer(temp_buffer):
        try:
            yield
        finally:
            if buffer_name(temp_buffer):
                kill_buffer(temp_buffer)

set_buffer(get_buffer_create("*scratch*"))