from __future__ import annotations

import numpy as np
from rasterio import windows
from dask.array.utils import assert_eq

from stackstac.raster_spec import RasterSpec
from stackstac.prepare import ASSET_TABLE_DT
from stackstac.to_dask import items_to_dask


def test_items_to_dask_basic():
    asset_table = np.array(
        [
            # Encode the (i, j) index in the table in the URL
            [("fake://0/0", [0, 0, 2, 2]), ("fake://0/1", [0, 0, 2, 2])],
            [("fake://1/0", [0, 3, 2, 5]), ("fake://1/1", [10, 13, 12, 15])],
            [("fake://2/0", [1, 3, 2, 6]), ("fake://2/1", [1, 3, 2, 7])],
            [(None, None), (None, None)],
        ],
        dtype=ASSET_TABLE_DT,
    )
    spec_ = RasterSpec(4326, (0, 0, 7, 8), (0.2, 0.2))
    chunksize = 2
    dtype_ = np.dtype("int32")
    fill_value_ = -1

    # Build expected array of the final stack.
    # Start with all nodata, then write data in for each asset.
    # The `TestReader` will then just read from this final array, sliced to the appropriate window.
    # (This is much easier than calculating where to put nodata values ourselves.)

    asset_windows: dict[str, windows.Window] = {}
    results = np.full(asset_table.shape + spec_.shape, fill_value_, dtype_)
    for i, item in enumerate(asset_table):
        for j, asset in enumerate(item):
            url = asset["url"]
            if url is None:
                continue
            assert url == f"fake://{i}/{j}"

            window = windows.from_bounds(
                *asset["bounds"],
                transform=spec_.transform,
                precision=0.0
                # ^ `precision=0.0`: https://github.com/rasterio/rasterio/issues/2374
            )
            # convert to int so `toslices` works for indexing
            window_int = window.round_lengths().round_offsets()
            # sanity check; rounding should not have changed anything
            assert window_int == window
            asset_windows[url] = window_int

            chunk = results[(i, j) + window_int.toslices()]
            if chunk.size:
                # Asset falls within final bounds
                chunk[:] = np.random.default_rng().integers(
                    0, 10000, (int(window_int.height), int(window_int.width)), dtype_
                )

    class TestReader:
        def __init__(
            self,
            *,
            url: str,
            spec: RasterSpec,
            dtype: np.dtype,
            fill_value: int | float,
            **kwargs,
        ) -> None:
            i, j = map(int, url[7:].split("/"))
            self.full_data = results[i, j]
            self.window = asset_windows[url]

            assert spec == spec_
            assert dtype == dtype_
            assert fill_value == fill_value_
            # NOTE: needed for `Reader` interface:
            self.dtype = dtype
            self.fill_value = fill_value

        def read(self, window: windows.Window) -> np.ndarray:
            assert (window.height, window.width) == (chunksize, chunksize)
            # Read should be bypassed entirely if windows don't intersect
            assert windows.intersect(window, self.window)
            return self.full_data[window.toslices()]

        def close(self) -> None:
            pass

        def __getstate__(self) -> dict:
            return self.__dict__

        def __setstate__(self, state):
            self.__init__(**state)

    arr = items_to_dask(
        asset_table,
        spec_,
        chunksize,
        dtype=dtype_,
        fill_value=fill_value_,
        reader=TestReader,
    )
    assert arr.chunksize == (1, 1, chunksize, chunksize)
    assert arr.dtype == dtype_

    assert_eq(arr, results)
