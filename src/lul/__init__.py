# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

from lul._src.runtime import *
from lul._src import system
from lul._src import reader