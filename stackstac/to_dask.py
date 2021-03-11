from __future__ import annotations

import itertools
from typing import Optional, Tuple, Type, TypeVar, Union
import warnings

import dask
import dask.array as da
import numpy as np
from rasterio import windows
from rasterio.enums import Resampling

from .raster_spec import Bbox, RasterSpec
from .rio_reader import AutoParallelRioReader, LayeredEnv
from .reader_protocol import Reader

# from xarray.backends import CachingFileManager
# from xarray.backends.locks import DummyLock


def items_to_dask(
    asset_table: np.ndarray,
    spec: RasterSpec,
    chunksize: int,
    resampling: Resampling = Resampling.nearest,
    dtype: np.dtype = np.dtype("float64"),
    fill_value: Optional[Union[int, float]] = np.nan,
    rescale: bool = True,
    reader: Type[Reader] = AutoParallelRioReader,
    gdal_env: Optional[LayeredEnv] = None,
) -> da.Array:
    # The overall strategy in this function is to materialize the outer two dimensions (items, assets)
    # as one dask array, then the chunks of the inner two dimensions (y, x) as another dask array, then use
    # Blockwise to represent the cartesian product between them, to avoid materializing that entire graph.
    # Materializing the (items, assets) dimensions is unavoidable: every asset has a distinct URL, so that information
    # has to be included somehow.

    # make URLs into dask array with 1-element chunks (one chunk per asset)
    asset_table_dask = da.from_array(
        asset_table,
        chunks=1,
        inline_array=True,
        name="asset-table-" + dask.base.tokenize(asset_table),
    )

    # then map a function over each chunk that opens that URL as a rasterio dataset
    # HACK: `url_to_ds` doesn't even return a NumPy array, so `datasets.compute()` would fail
    # (because the chunks can't be `np.concatenate`d together).
    # but we're just using this for the graph.
    # So now we have an array of shape (items, assets), chunksize 1---the outer two dimensions of our final array.
    datasets = asset_table_dask.map_blocks(
        asset_entry_to_reader_and_window,
        spec,
        resampling,
        dtype,
        fill_value,
        rescale,
        gdal_env,
        reader,
        meta=asset_table_dask._meta,
    )

    # MEGAHACK: generate a fake array for our spatial dimensions following `shape` and `chunksize`,
    # but where each chunk is not actually NumPy a array, but just a 2-tuple of the (y-slice, x-slice)
    # to `read` from the rasterio dataset to fetch that window of data.
    shape = spec.shape
    name = "slices-" + dask.base.tokenize(chunksize, shape)
    chunks = da.core.normalize_chunks(chunksize, shape)
    keys = itertools.product([name], *(range(len(bds)) for bds in chunks))
    slices = da.core.slices_from_chunks(chunks)
    # HACK: `slices_fake_arr` is in no way a real dask array: the chunks aren't ndarrays; they're tuples of slices!
    # We just stick it in an Array container to make dask's blockwise logic handle broadcasting between the graphs of
    # `datasets` and `slices_fake_arr`.
    slices_fake_arr = da.Array(
        dict(zip(keys, slices)), name, chunks, meta=datasets._meta
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=da.core.PerformanceWarning)

        rasters = da.blockwise(
            fetch_raster_window,
            "tbyx",
            datasets,
            "tb",
            slices_fake_arr,
            "yx",
            meta=np.ndarray((), dtype=dtype),  # TODO dtype
        )
    return rasters


ReaderT = TypeVar("ReaderT", bound=Reader)


def asset_entry_to_reader_and_window(
    asset_entry: np.ndarray,
    spec: RasterSpec,
    resampling: Resampling,
    dtype: np.dtype,
    fill_value: Optional[Union[int, float]],
    rescale: bool,
    gdal_env: Optional[LayeredEnv],
    reader: Type[ReaderT],
) -> Optional[Tuple[ReaderT, windows.Window]]:
    asset_entry = asset_entry[0, 0]
    # ^ because dask adds extra outer dims in `from_array`
    url = asset_entry["url"]
    if url is None:
        return None

    asset_bounds: Bbox = asset_entry["bounds"]
    asset_window = windows.from_bounds(*asset_bounds, transform=spec.transform)

    return (
        # CachingFileManager(
        #     AutoParallelRioBackend,  # TODO other backends
        #     url,
        #     spec,
        #     resampling,
        #     gdal_env,
        #     lock=DummyLock(),
        #     # ^ NOTE: this lock only protects the file cache, not the file itself.
        #     # xarray's file cache is already thread-safe, so using a lock is pointless.
        # ),
        # NOTE: skip the `CachingFileManager` for now to be sure datasets aren't leaked
        reader(
            url,
            spec,
            resampling,
            dtype,
            fill_value,
            rescale,
            gdal_env=gdal_env,
        ),
        asset_window,
    )


def fetch_raster_window(
    asset_entry: Optional[Tuple[Reader, windows.Window]],
    slices: Tuple[slice, ...],
) -> np.ndarray:
    current_window = windows.Window.from_slices(*slices)
    if asset_entry is not None:
        reader, asset_window = asset_entry

        # check that the window we're fetching overlaps with the asset
        if windows.intersect(current_window, asset_window):
            # backend: Backend = manager.acquire(needs_lock=False)
            data = reader.read(current_window)

            return data[None, None]

    # no dataset, or we didn't overlap it: return empty data.
    # use the broadcast trick for even fewer memz
    return np.broadcast_to(np.nan, (1, 1) + windows.shape(current_window))
