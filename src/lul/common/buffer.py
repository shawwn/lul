from __future__ import annotations

from .runtime import *

from dataclasses import dataclass, field
import contextvars as CV

@dataclass
class BufferText:
    beg: str = ""

@dataclass
class Buffer:
    name: str

    #   /* This structure holds the coordinates of the buffer contents
    #      in ordinary buffers.  In indirect buffers, this is not used.  */
    own_text: BufferText = field(default_factory=BufferText)

    #   /* This points to the `struct buffer_text' that used for this buffer.
    #      In an ordinary buffer, this is the own_text field above.
    #      In an indirect buffer, this is the own_text field of another buffer.  */
    text: BufferText = None

    # /* Char position of point in buffer.  */
    pt: int = 1

    # /* Char position of beginning of accessible range.  */
    begv: int = 1

    # /* Char position of end of accessible range.  */
    zv: int = 1

    #   /* In an indirect buffer, this points to the base buffer.
    #      In an ordinary buffer, it is 0.  */
    base_buffer: Buffer = None

    def __post_init__(self):
        self.text = self.own_text

    all_buffers: ClassVar[List[Buffer]] = []

def bufferp(x):
    return isinstance(x, Buffer)

def nsberror(spec):
    if stringp(spec):
        error("No buffer named %s", SDATA(spec))
    error("Invalid buffer argument")

def buffer_hidden_p(buffer: Buffer):
    return buffer.name.startswith(' ')

def buffer_live_p(buffer: Buffer):
    return buffer.name is not None

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

def current_buffer() -> Buffer:
    return get(Q.current_buffer)



