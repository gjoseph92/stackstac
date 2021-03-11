from __future__ import annotations
from typing import Optional, Protocol, Type, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from .raster_spec import RasterSpec
    from rasterio.windows import Window
    from rasterio.enums import Resampling


class Reader(Protocol):
    """
    Protocol for a thread-safe, lazily-loaded object for reading data from a single-band STAC asset.
    """

    def __init__(
        self,
        url: str,
        spec: RasterSpec,
        resampling: Resampling,
        dtype: np.dtype,
        fill_value: Optional[Union[int, float]] = np.nan,
        rescale: bool = True,
        **kwargs,
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
        """
        if fill_value is not None and not np.can_cast(fill_value, dtype):
            raise ValueError(
                f"The fill_value {fill_value} is incompatible with the output dtype {dtype}. "
                f"Try using `dtype={np.array(fill_value).dtype.name!r}`."
            )
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

    def __init__(self, url: str, spec: RasterSpec, *args, **kwargs) -> None:
        pass
        # self.url = url
        # self.spec = spec

    def read(self, window: Window, **kwargs) -> np.ndarray:
        return np.random.random((window.height, window.width))

    def close(self) -> None:
        pass


# Type assertion
_: Type[Reader] = FakeReader
