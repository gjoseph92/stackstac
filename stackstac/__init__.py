from .rio_env import LayeredEnv
from .rio_reader import DEFAULT_GDAL_ENV, MULTITHREADED_DRIVER_ALLOWLIST
from .stack import stack

# try:
from . import show as _show
from .show import show, add_to_map
from .ops import mosaic

# except ImportError:
#     pass

__all__ = [
    "LayeredEnv",
    "DEFAULT_GDAL_ENV",
    "MULTITHREADED_DRIVER_ALLOWLIST",
    "stack",
    "show",
    "add_to_map",
    "mosaic",
]
