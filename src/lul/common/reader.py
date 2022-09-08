from .runtime import *

from dataclasses import dataclass
import contextvars as CV
import sys

# @dataclass
# class Stream:
#     string: str
#     pos: int = 0
#     end: int = 0

class EndOfFile(Exception):
    pass

def delimiter_p(c):
    return c in "();\r\n"

def whitespace_p(c):
    return c in " \t\r\n"

def looking_at(s, predicate):
    return predicate(peek_char(s))

def stream(string: str, start=0, end=None, more=None):
    end = len(string) if end is None else end
    return [string, start, end, more]

def stream_item(s, idx, *val):
    if val:
        s[idx] = val[0]
    return s[idx]

def stream_string(s, *val) -> str:
    return stream_item(s, 0, *val)

def stream_pos(s, *val) -> int:
    return stream_item(s, 1, *val)

def stream_end(s, *val) -> int:
    return stream_item(s, 2, *val)

def forward_char(s, count=1):
    stream_pos(s, stream_pos(s) + count)

def forward_char(s, count=1):
    stream_pos(s, stream_pos(s) + count)

def peek_char(s):
    if (pos := stream_pos(s)) < stream_end(s):
        return stream_string(s)[pos]

def read_char(s):
    if (c := peek_char(s)) is not None:
        stream_pos(s, stream_pos(s) + 1)
        return c
    else:
        raise EndOfFile()

def read_line(s):
    s = []
    while (c := read_char(s)) != "\n":
        s.append(c)
    return ''.join(s) + "\n"

def skip_non_code(s):
    while c := peek_char(s):
        if whitespace_p(c):
            read_char(s)
        elif c == ";":
            read_line(s)
        else:
            break

def read_from_string(string, start=0, end=0):
    return read(stream(string, start=start, end=end))

def read(s):
    skip_non_code(s)
    c = peek_char(s)
    if c == "(":
        return read_list(s, "(", ")")
    else:
        return read_atom(s)

def read_list(s, open: str, close: str):
    assert read_char(s) == open
    out = []
    while peek_char(s) != close:
        out.append(read(s))
        skip_non_code(s)
    return out

def read_atom(s):
    skip_non_code(s)
    start = stream_pos(s)
    while looking_at(s, whitespace_p) or looking_at(s, delimiter_p):
        read_char(s)
    while (c := peek_char(s)) and not whitespace_p(c) and not delimiter_p(c):
        read_char(s)




