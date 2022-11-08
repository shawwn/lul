from .runtime import *

import sys

def call_with_file(f=None, path=None, mode=None):
  with open(path, mode) as h:
    return f(h)

def read_file(path=None, mode="r"):
  def __f(f=None):
    return f.read()
  return call_with_file(__f, path, mode)

def write_file(path=None, data=None, mode="w"):
  def __f1(f=None):
    return f.write(data)
  return call_with_file(__f1, path, mode)

