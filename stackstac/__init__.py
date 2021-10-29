from .rio_env import LayeredEnv
from .rio_reader import DEFAULT_GDAL_ENV, MULTITHREADED_DRIVER_ALLOWLIST
from .stack import stack
from .ops import mosaic
from .geom_utils import reproject_array, array_bounds, array_epsg, xyztile_of_array

try:
    from . import show as _show  # helpful for debugging
    from .show import show, add_to_map, server_stats
except ImportError:
    import traceback as _traceback

    msg = _traceback.format_exc()

    def _missing_imports(*args, **kwargs):

        raise ImportError(
            "Optional dependencies for map visualization are missing.\n"
            "Please re-install stackstac with the `viz` extra:\n"
            "$ pip install --upgrade 'stackstac[viz]'\n\n"
            f"The original error was:\n{msg}"
        )

    show = add_to_map = server_stats = _missing_imports

# Single-source version from pyproject.toml: https://github.com/python-poetry/poetry/issues/273#issuecomment-769269759
# Note that this will be incorrect for local installs
import importlib.metadata

__version__ = importlib.metadata.version("stackstac")
del importlib


__all__ = [
    "LayeredEnv",
    "DEFAULT_GDAL_ENV",
    "MULTITHREADED_DRIVER_ALLOWLIST",
    "stack",
    "show",
    "_show",
    "add_to_map",
    "mosaic",
    "reproject_array",
    "array_bounds",
    "array_epsg",
    "xyztile_of_array",
    "__version__",
]
