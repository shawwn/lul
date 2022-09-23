from .runtime import *

# import dataclasses
# from dataclasses import dataclass
# import contextvars as CV
# import sys

# @dataclass
# class Stream:
#     string: str
#     pos: int = 0
#     end: int = 0
#     _: dataclasses.KW_ONLY = None
#     more: Any = None

class ReaderError(Exception):
    pass

class EndOfFile(ReaderError):
    pass

def delimiter_p(c):
    return c and c in "\"()[]{};\r\n"

def elisp_delimiter_p(c):
    return c and c in "\"()[];\r\n"

def delimiter_fn(s):
    if stream_mode(s) == 'elisp':
        return elisp_delimiter_p
    return delimiter_p

def closing_p(c):
    return c and c in ")]}"

def elisp_closing_p(c):
    return c and c in ")]"

def closing_fn(s):
    if stream_mode(s) == 'elisp':
        return elisp_closing_p
    return closing_p

def whitespace_p(c):
    return c and c in " \t\r\n"

def end_of_input_p(c):
    return c is None

def looking_at(s, predicate):
    c = peek_char(s)
    return predicate(c) if callable(predicate) else c == predicate

def stream(string: str, start=0, end=None, more=None, mode=None):
    end = len(string) if end is None else end
    return [string, start, end, more, mode or "lisp"]

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

def stream_more(s, *val) -> Any:
    return stream_item(s, 3, *val)

def stream_mode(s, *val) -> Any:
    return stream_item(s, 4, *val)

def forward_char(s, count=1):
    stream_pos(s, stream_pos(s) + count)

def peek_char(s):
    if (pos := stream_pos(s)) < stream_end(s):
        return stream_string(s)[pos]

def read_char(s):
    if (c := peek_char(s)) is not None:
        stream_pos(s, stream_pos(s) + 1)
        return c

def read_line(s):
    r = []
    while (c := read_char(s)) and c != "\n":
        r.append(c)
    return ''.join(r) + "\n"

def skip_non_code(s):
    while c := peek_char(s):
        if whitespace_p(c):
            read_char(s)
        elif c == ";":
            read_line(s)
        else:
            break

def read_from_string(string, start=0, end=None, more=None, mode=None):
    s = stream(string, start=start, end=end, more=more, mode=mode)
    return read(s), stream_pos(s)

def read(s, eof=None, start=None):
    skip_non_code(s)
    if start is None:
        start = stream_pos(s)
    c = peek_char(s)
    if c is None:
        return eof
    elif c == "(":
        return read_list(s, "(", ")", start=start)
    elif c == "[":
        form = read_list(s, "[", "]", start=start)
        if stream_mode(s) in ["bel", "arc"]:
            return ["fn", ["_"], form]
        return ["lit", "brackets", form]
    elif c == "{" and stream_mode(s) != "elisp":
        return ["lit", "braces", read_list(s, "{", "}", start=start)]
    elif c == "\"":
        return read_string(s, "\"", "\"", True)
    # elif c == "|":
    #     return read_string(s, "|", "|", False)
    elif c == "'":
        read_char(s)
        return wrap(s, "quote", start=start)
    elif c == "`":
        read_char(s)
        return wrap(s, "quasiquote", start=start)
    elif c == ("~" if stream_mode(s) == "clojure" else ","):
        read_char(s)
        if peek_char(s) == "@":
            read_char(s)
            return wrap(s, "unquote-splicing", start=start)
        return wrap(s, "unquote", start=start)
    elif closing_fn(s)(c):
        raise SyntaxError(f"Unexpected {peek_char(s)!r} at {format_line_info(s, stream_pos(s))} from {format_line_info(s, start)}")
    else:
        return read_atom(s)

def read_all(s, *, verbose=False):
    out = []
    eof = object()
    if verbose:
        prev = stream_pos(s)
        import tqdm
        with tqdm.tqdm(total=stream_end(s), position=stream_pos(s)) as pbar:
            while (x := read(s, eof)) is not eof:
                out.append(x)
                pbar.update(stream_pos(s) - prev)
                prev = stream_pos(s)
    else:
        while (x := read(s, eof)) is not eof:
            out.append(x)
    return out

def line_info(s, pos: int):
    # s1 = stream(stream_string(s), end=stream_end(s))
    # line = 1
    # col = 1
    # while stream_pos(s1) < pos and (c := read_char(s1)):
    #     if c == "\n":
    #         col = 1
    #         line += 1
    #     else:
    #         col += 1
    lines = stream_string(s)[0:pos+1].splitlines(keepends=True)
    line = len(lines)
    col = len(lines[-1]) + 1
    return line, col

def format_line_info(s, pos: int):
    line, col = line_info(s, pos)
    return f"{pos} {line}:{col}"

def expected(s, c: str, start: int):
    if (more := stream_more(s)) is not None:
        return more
    raise EndOfFile(f"Expected {c!r} at {format_line_info(s, stream_pos(s))} from {format_line_info(s, start)}")

def read_list(s, open: str, close: str, start=None):
    start = stream_pos(s)
    assert read_char(s) == open
    out = []
    skip_non_code(s)
    while (c := peek_char(s)) and c != close:
        out.append(read(s, start=start))
        skip_non_code(s)
    if c != close:
        return expected(s, close, start)
    assert read_char(s) == close
    return out

def read_atom(s):
    skip_non_code(s)
    while looking_at(s, whitespace_p) or looking_at(s, delimiter_fn(s)):
        read_char(s)
    out = []
    while True:
        c = peek_char(s)
        if c == '\\' or (c == '?' and len(out) == 0):
            out.append(read_char(s))
            out.append(c1 := read_char(s))
            if c == '?' and c1 == '\\':
                out.append(read_char(s))
            continue
        if not c or whitespace_p(c) or delimiter_fn(s)(c):
            break
        out.append(read_char(s))
    return "".join(out)

def read_string(s, open: str, close: str, backquote: Optional[bool] = None):
    start = stream_pos(s)
    assert read_char(s) == open
    out = []
    while (c := peek_char(s)) and c != close:
        if backquote is not None and c == "\\":
            if backquote:
                out.append(read_char(s))
            else:
                read_char(s)
        out.append(read_char(s))
    if c != close:
        return expected(s, close, start)
    assert read_char(s) == close
    return open + "".join(out) + close


def wrap(s, x, start=None):
    if (y := read(s, start=start)) == stream_more(s):
        return y
    else:
        return [x, y]



# s = reader.stream(open(os.path.expanduser("~/ml/bel/bel.bel")).read(), mode='bel'); bel = reader.read_all(s, verbose=True)
# s = reader.stream(open(os.path.expanduser("~/all-emacs.el")).read(), mode='elisp'); emacs = reader.read_all(s, verbose=True)
