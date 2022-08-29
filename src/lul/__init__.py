# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

from . import emacs

emacs.c.main(0, [])
