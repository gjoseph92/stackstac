from __future__ import annotations
from threading import Lock
from typing import ClassVar

from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as st_np
import numpy as np
from rasterio import windows
import dask.core
import dask.threaded
from dask.array.utils import assert_eq

from stackstac.raster_spec import Bbox, RasterSpec
from stackstac.prepare import ASSET_TABLE_DT
from stackstac.to_dask import (
    ChunksParam,
    items_to_dask,
    normalize_chunks,
)
from stackstac.testing import strategies as st_stc


@st.composite
def asset_tables(
    draw: st.DrawFn,
    max_side: int | None = None,
) -> np.ndarray:
    """
    Generate asset tables where entries have random bounds, and are randomly missing.
    Each URL is of the form ``"fake://{i}/{j}"``, so you can parse it within a Reader
    to know the (time, band) coordinates of that asset. Bounds may be zero-size (the min
    and max are equal).

    An example of an asset table:

        np.array(
            [
                # Encode the (i, j) index in the table in the URL
                [("fake://0/0", [0, 0, 2, 2]), ("fake://0/1", [0, 0, 2, 2])],
                [("fake://1/0", [0, 3, 2, 5]), ("fake://1/1", [10, 13, 12, 15])],
                [("fake://2/0", [1, 3, 2, 6]), ("fake://2/1", [1, 3, 2, 7])],
                [(None, None), (None, None)],
            ],
            dtype=ASSET_TABLE_DT,
        )
    """
    shape = draw(
        st_np.array_shapes(min_dims=2, max_dims=2, max_side=max_side), label="shape"
    )
    bounds_arr = draw(
        st_np.arrays(
            object,
            shape,
            elements=st_stc.simple_bboxes(),
            fill=st.none(),
        ),
        label="bounds_arr",
    )

    asset_table = np.empty_like(bounds_arr, ASSET_TABLE_DT)
    for (i, j), bounds in np.ndenumerate(bounds_arr):
        if bounds:
            # Encode the (i, j) index in the table in the URL
            asset_table[i, j] = (f"fake://{i}/{j}", bounds)

    return asset_table


@given(
    st.data(),
    asset_tables(max_side=5),
    st_stc.simple_bboxes(-4, -4, 4, 4, zero_size=False),
    st_stc.raster_dtypes,
    st_stc.chunksizes(
        4,
        max_side=10,
        auto=False,
        bytes=False,
        none=False,
        minus_one=False,
        dicts=False,
        singleton=False,
    ),
)
@settings(max_examples=500, print_blob=True)
def test_items_to_dask(
    data: st.DataObject,
    asset_table: np.ndarray,
    bounds: Bbox,
    dtype_: np.dtype,
    chunksize: tuple[int, int, int, int],
):
    spec_ = RasterSpec(4326, bounds, (0.5, 0.5))
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

            asset_bounds: Bbox = asset["bounds"]
            window = windows.from_bounds(*asset_bounds, spec_.transform)
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
            assert 0 < window.height <= chunksize[2]
            assert 0 < window.width <= chunksize[3]
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
    assert arr.chunksize == tuple(
        min(x, y) for x, y in zip(asset_table.shape + spec_.shape, chunksize)
    )
    assert arr.dtype == dtype_

    assert_eq(arr, results, equal_nan=True)

    # Check that entirely-empty chunks are broadcast-tricked into being tiny.
    # NOTE: unfortunately, this computes the array again, which slows down tests.
    # But passing a computed array into `assert_eq` would skip handy checks for chunks, meta, etc.
    TestReader.opened.clear()
    chunks = dask.threaded.get(arr.dask, list(dask.core.flatten(arr.__dask_keys__())))
    for chunk in chunks:
        if (
            np.isnan(chunk) if np.isnan(fill_value_) else np.equal(chunk, fill_value_)
        ).all():
            assert chunk.strides == (0, 0, 0, 0)


@given(
    st_stc.chunksizes(4, max_side=1000),
    st_np.array_shapes(min_dims=4, max_dims=4),
    st_stc.raster_dtypes,
)
def test_normalize_chunks(
    chunksize: ChunksParam, shape: tuple[int, int, int, int], dtype: np.dtype
):
    chunks = normalize_chunks(chunksize, shape, dtype)
    numblocks = tuple(map(len, chunks))
    assert len(numblocks) == 4
    assert all(x >= 1 for t in chunks for x in t)
    if isinstance(chunksize, int) or isinstance(chunks, tuple) and len(chunks) == 2:
        assert numblocks[:2] == shape[:2]
