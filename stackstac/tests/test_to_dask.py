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
            [("fake://1/0", [0, 3, 2, 5]), ("fake://1/1", [0, 3, 2, 5])],
            [("fake://2/0", [1, 3, 2, 6]), ("fake://2/1", [1, 3, 2, 7])],
            [(None, None), (None, None)],
        ],
        dtype=ASSET_TABLE_DT,
    )

    def parse_url(url: str) -> tuple[int, int]:
        assert url.startswith("fake://")
        i, j = url[7:].split("/")
        return int(i), int(j)

    def value(i: int, j: int) -> int:
        return i << 1 & j

    spec_ = RasterSpec(4326, (0, 0, 7, 8), (0.2, 0.2))
    chunksize = 2
    dtype_ = np.dtype("int32")
    fill_value_ = -1

    class TestReader:
        def __init__(
            self,
            *,
            url: str,
            spec: RasterSpec,
            dtype: np.dtype,
            fill_value: int | float,
            **kwargs
        ) -> None:
            i, j = parse_url(url)
            entry = asset_table[i, j]
            self._window = windows.from_bounds(
                *entry["bounds"], transform=spec.transform
            )
            self._value = value(i, j)

            self.url = url
            assert spec == spec_
            self.spec = spec
            assert dtype == dtype_
            self.dtype = dtype
            assert fill_value == fill_value_
            self.fill_value = fill_value

        def read(self, window: windows.Window) -> np.ndarray:
            assert (window.height, window.width) == (chunksize, chunksize)
            assert windows.intersect(window, self._window)
            return np.full((window.height, window.width), self._value, self.dtype)

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
    assert arr.shape == asset_table.shape + spec_.shape
    assert arr.chunksize == (1, 1, chunksize, chunksize)
    assert arr.dtype == dtype_

    expected = np.full(arr.shape, fill_value_, dtype_)

    for i in range(asset_table.shape[0]):
        for b in range(asset_table.shape[1]):
            entry = asset_table[i, b]
            v = value(*parse_url(entry["url"]))
            window = windows.from_bounds(*entry["bounds"], transform=spec_.transform)
            for y in range(spec_.shape[0]):
                for x in range(spec_.shape[1]):
                    expected[
                        i,
                        b,
                        y * chunksize : y * chunksize + chunksize,
                        x * chunksize : x * chunksize + chunksize,
                    ] = v

    # expected = [
    #     [
    #         np.full(
    #             (chunksize, chunksize),
    #             value(*parse_url(asset["url"])) if asset["url"] else fill_value_,
    #             dtype_,
    #         )
    #         for asset in row
    #     ]
    #     for row in asset_table
    ]
    assert_eq(arr, expected)
