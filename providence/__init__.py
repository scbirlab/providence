from importlib.metadata import version

app_name = "providence"
__author__ = "Eachan Johnson"
__version__ = version(app_name)

from .data import DataResolver
