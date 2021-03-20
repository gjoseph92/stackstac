from .rio_env import LayeredEnv
from .rio_reader import DEFAULT_GDAL_ENV, MULTITHREADED_DRIVER_ALLOWLIST
from .stack import stack

# try:
from .show import show

# except ImportError:
#     pass

__all__ = [
    "LayeredEnv",
    "DEFAULT_GDAL_ENV",
    "MULTITHREADED_DRIVER_ALLOWLIST",
    "stack",
    "show",
]
