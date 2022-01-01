from __future__ import annotations

import itertools
from typing import ClassVar, Optional, Tuple, Type, TypeVar, Union
import warnings

import dask
import dask.array as da
from dask.blockwise import BlockwiseDep, blockwise
from dask.highlevelgraph import HighLevelGraph
import numpy as np
from rasterio import windows
from rasterio.enums import Resampling

from .raster_spec import Bbox, RasterSpec
from .rio_reader import AutoParallelRioReader, LayeredEnv
from .reader_protocol import Reader


def items_to_dask(
    asset_table: np.ndarray,
    spec: RasterSpec,
    chunksize: int,
    resampling: Resampling = Resampling.nearest,
    dtype: np.dtype = np.dtype("float64"),
    fill_value: Union[int, float] = np.nan,
    rescale: bool = True,
    reader: Type[Reader] = AutoParallelRioReader,
    gdal_env: Optional[LayeredEnv] = None,
    errors_as_nodata: Tuple[Exception, ...] = (),
) -> da.Array:
    errors_as_nodata = errors_as_nodata or ()  # be sure it's not None

    if not np.can_cast(fill_value, dtype):
        raise ValueError(
            f"The fill_value {fill_value} is incompatible with the output dtype {dtype}. "
            f"Either use `dtype={np.array(fill_value).dtype.name!r}`, or pick a different `fill_value`."
        )

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
        errors_as_nodata,
        reader,
        meta=asset_table_dask._meta,
    )

    shape_yx = spec.shape
    chunks_yx = da.core.normalize_chunks(chunksize, shape_yx)
    chunks = datasets.chunks + chunks_yx

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=da.core.PerformanceWarning)

        name = f"fetch_raster_window-{dask.base.tokenize(datasets, chunks)}"
        # TODO use `da.blockwise` once it supports `BlockwiseDep`s as arguments
        lyr = blockwise(
            fetch_raster_window,
            name,
            "tbyx",
            datasets.name,
            "tb",
            Slices(chunks_yx),
            "yx",
            numblocks={datasets.name: datasets.numblocks},  # ugh
        )
        dsk = HighLevelGraph.from_collections(name, lyr, [datasets])
        rasters = da.Array(dsk, name, chunks, meta=np.ndarray((), dtype=dtype))

    return rasters


ReaderT = TypeVar("ReaderT", bound=Reader)


def asset_entry_to_reader_and_window(
    asset_entry: np.ndarray,
    spec: RasterSpec,
    resampling: Resampling,
    dtype: np.dtype,
    fill_value: Union[int, float],
    rescale: bool,
    gdal_env: Optional[LayeredEnv],
    errors_as_nodata: Tuple[Exception, ...],
    reader: Type[ReaderT],
) -> Tuple[ReaderT, windows.Window] | np.ndarray:
    asset_entry = asset_entry[0, 0]
    # ^ because dask adds extra outer dims in `from_array`
    url = asset_entry["url"]
    if url is None:
        # Signifies empty value
        return np.array(fill_value, dtype)

    asset_bounds: Bbox = asset_entry["bounds"]
    asset_window = windows.from_bounds(
        *asset_bounds,
        transform=spec.transform,
        precision=0.0
        # ^ `precision=0.0`: https://github.com/rasterio/rasterio/issues/2374
    )

    return (
        reader(
            url=url,
            spec=spec,
            resampling=resampling,
            dtype=dtype,
            fill_value=fill_value,
            rescale=rescale,
            gdal_env=gdal_env,
            errors_as_nodata=errors_as_nodata,
        ),
        asset_window,
    )


def fetch_raster_window(
    asset_entry: Tuple[ReaderT, windows.Window] | np.ndarray,
    slices: Tuple[slice, slice],
) -> np.ndarray:
    assert len(slices) == 2, slices
    current_window = windows.Window.from_slices(*slices)
    if isinstance(asset_entry, tuple):
        reader, asset_window = asset_entry

        # check that the window we're fetching overlaps with the asset
        if windows.intersect(current_window, asset_window):
            data = reader.read(current_window)

            return data[None, None]
        fill_arr = np.array(reader.fill_value, reader.dtype)
    else:
        fill_arr: np.ndarray = asset_entry

    # no dataset, or we didn't overlap it: return empty data.
    # use the broadcast trick for even fewer memz
    return np.broadcast_to(fill_arr, (1, 1) + windows.shape(current_window))


class Slices(BlockwiseDep):
    starts: list[tuple[int, ...]]
    produces_tasks: ClassVar[bool] = False

    def __init__(self, chunks: Tuple[Tuple[int, ...], ...]):
        self.starts = [tuple(itertools.accumulate(c, initial=0)) for c in chunks]

    def __getitem__(self, idx: Tuple[int, ...]) -> Tuple[slice, ...]:
        return tuple(
            slice(start[i], start[i + 1]) for i, start in zip(idx, self.starts)
        )

    @property
    def numblocks(self) -> list[int]:
        return [len(s) - 1 for s in self.starts]

    def __dask_distributed_pack__(
        self, required_indices: Optional[list[Tuple[int, ...]]] = None
    ) -> list[Tuple[int, ...]]:
        return self.starts

    @classmethod
    def __dask_distributed_unpack__(cls, state: list[Tuple[int, ...]]) -> Slices:
        self = cls.__new__(cls)
        self.starts = state
        return self
