from __future__ import annotations
from threading import Lock
from typing import ClassVar

from hypothesis import given, strategies as st
import hypothesis.extra.numpy as st_np
import numpy as np
from rasterio import windows
from dask.array.utils import assert_eq

from stackstac.raster_spec import RasterSpec
from stackstac.prepare import ASSET_TABLE_DT
from stackstac.to_dask import items_to_dask
from stackstac.testing import strategies as st_stc


@given(st.data(), st_stc.raster_dtypes)
def test_items_to_dask_basic(data: st.DataObject, dtype_: np.dtype):
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
    spec_ = RasterSpec(4326, (0, 0, 7, 8), (0.5, 0.5))
    chunksize = 2
    fill_value_ = data.draw(st_np.from_dtype(dtype_), label="fill_value")

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
                # ^ https://github.com/rasterio/rasterio/issues/2374
            )
            asset_windows[url] = window

            chunk = results[(i, j) + windows.window_index(window)]
            if chunk.size:
                # Asset falls within final bounds
                chunk[:] = np.random.default_rng().uniform(0, 128, chunk.shape)

    class TestReader:
        opened: ClassVar[set[str]] = set()
        lock: ClassVar[Lock] = Lock()

        def __init__(
            self,
            *,
            url: str,
            spec: RasterSpec,
            dtype: np.dtype,
            fill_value: int | float,
            **kwargs,
        ) -> None:
            with self.lock:
                # Each URL should only be opened once.
                # The `dask.annotate` on the `asset_table_to_reader_and_window` step is necessary for this,
                # otherwise blockwise fusion would merge the Reader creation into every `fetch_raster_window`!
                assert url not in self.opened
                self.opened.add(url)
            i, j = map(int, url[7:].split("/"))
            self.full_data = results[i, j]
            self.window = asset_windows[url]

            assert spec == spec_
            assert dtype == dtype_
            np.testing.assert_equal(fill_value, fill_value_)

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

    assert_eq(arr, results, equal_nan=True)
