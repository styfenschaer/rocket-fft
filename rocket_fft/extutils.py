from functools import cached_property
from pathlib import Path
from sysconfig import get_config_var

from llvmlite.binding import load_library_permanently


class ExtensionLibrary:
    def __init__(self, library_name):
        self.library_name = library_name

    @cached_property
    def path(self):
        search_path = Path(__file__).parent.parent
        ext_suffix = get_config_var("EXT_SUFFIX")
        ext_path = f"**/{self.library_name}{ext_suffix}"
        matches = search_path.glob(ext_path)
        return str(next(matches))

    def load_permanently(self):
        load_library_permanently(self.path)
