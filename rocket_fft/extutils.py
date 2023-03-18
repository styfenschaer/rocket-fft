from distutils.sysconfig import get_config_var
from pathlib import Path

from llvmlite.binding import load_library_permanently


def get_extension_path(lib_name):
    search_path = Path(__file__).parent.parent
    ext_suffix = get_config_var("EXT_SUFFIX")
    ext_path = f"**/{lib_name}{ext_suffix}"
    matches = search_path.glob(ext_path)
    lib_path = str(next(matches))
    return lib_path


def load_extension_library_permanently(lib_name):
    lib_path = get_extension_path(lib_name)
    load_library_permanently(lib_path)