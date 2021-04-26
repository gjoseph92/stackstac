from .rio_env import LayeredEnv
from .rio_reader import DEFAULT_GDAL_ENV, MULTITHREADED_DRIVER_ALLOWLIST
from .stack import stack
from .ops import mosaic
from .geom_utils import reproject_array, array_bounds, array_epsg

try:
    from .show import show, add_to_map
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

    show = add_to_map = _missing_imports


__all__ = [
    "LayeredEnv",
    "DEFAULT_GDAL_ENV",
    "MULTITHREADED_DRIVER_ALLOWLIST",
    "stack",
    "show",
    "add_to_map",
    "mosaic",
    "reproject_array",
    "array_bounds",
    "array_epsg",
]
