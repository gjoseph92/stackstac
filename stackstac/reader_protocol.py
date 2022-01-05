from __future__ import annotations
from typing import Optional, Protocol, Tuple, Type, TYPE_CHECKING, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from .raster_spec import RasterSpec
    from .rio_env import LayeredEnv
    from rasterio.windows import Window
    from rasterio.enums import Resampling


PickleState = TypeVar("PickleState")


class Pickleable(Protocol[PickleState]):
    def __getstate__(self) -> PickleState:
        ...

    def __setstate__(self, state: PickleState) -> None:
        ...


class Reader(Pickleable, Protocol):
    """
    Protocol for a thread-safe, lazily-loaded object for reading data from a single-band STAC asset.
    """

    def __init__(
        self,
        *,
        url: str,
        spec: RasterSpec,
        resampling: Resampling,
        dtype: np.dtype,
        fill_value: Union[int, float],
        rescale: bool,
        gdal_env: Optional[LayeredEnv],
        errors_as_nodata: Tuple[Exception, ...] = (),
    ) -> None:
        """
        Construct the Dataset *without* fetching any data.

        Parameters
        ----------
        url:
            Fetch data from the asset at this URL.
        spec:
            Reproject data to match this georeferencing information.
        resampling:
            When reprojecting, resample in this way.
        dtype:
            Return arrays in this dtype. If ``fill_value`` is not None,
            must be the same type as ``fill_value``.
        fill_value:
            Fill nodata pixels in the output array with this value.
            If None, whatever nodata value is set in the asset will be used.
        rescale:
            Rescale the output array according to any scales and offsets set in the asset.
        gdal_env:
            A `~.LayeredEnv` of GDAL configuration options to use while opening
            and reading datasets. If None (default), `~.DEFAULT_GDAL_ENV` is used.
        errors_as_nodata:
            Exception patterns to ignore when opening datasets or reading data.
            Exceptions matching the pattern will be logged as warnings, and just
            produce nodata (``fill_value``).

            The exception patterns should be instances of an Exception type to catch,
            where ``str(exception_pattern)`` is a regex pattern to match against
            ``str(raised_exception)``.
        """
        # TODO colormaps?

    def read(self, window: Window) -> np.ndarray:
        """
        Read a portion of data from the Dataset as a NumPy array.

        If the Dataset has not been opened yet, `read` will open it
        (likely incurring much higher latency than subsequent reads).

        This method must be thread-safe, and will wait to acquire any locks
        as necessary.

        Parameters
        ----------
        window: The window to read, relative to the bounds and CRS of ``self.spec``

        Returns
        -------
        array: The window of data read
        """
        ...

    def close(self) -> None:
        """
        Close all resources for the Dataset.

        This method must be thread-safe, and will wait to acquire any locks
        as necessary.

        After `close` has returned, the behavior of subsequent calls to `read`
        is undefined. They may just behave as though the Dataset was never
        opened, or they may fail.
        """
        ...


class FakeReader:
    """
    Fake Reader that just returns random numbers.

    Meant for performance debugging, to isolate whether performance issues are due to rasterio,
    or inherent to the dask graph.
    """

    def __init__(self, *, dtype: np.dtype, **kwargs) -> None:
        self.dtype = dtype

    def read(self, window: Window, **kwargs) -> np.ndarray:
        return np.random.random((window.height, window.width)).astype(self.dtype)

    def close(self) -> None:
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass


# Type assertion
_: Type[Reader] = FakeReader
