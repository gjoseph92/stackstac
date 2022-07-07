from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Type, Union
import warnings

from affine import Affine
import dask
import dask.array as da
from dask.blockwise import blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArraySliceDep
import numpy as np
from rasterio import windows
from rasterio.enums import Resampling

from .raster_spec import Bbox, RasterSpec
from .rio_reader import AutoParallelRioReader, LayeredEnv
from .reader_protocol import Reader

ChunkVal = Union[int, Literal["auto"], str, None]
ChunksParam = Union[ChunkVal, Tuple[ChunkVal, ...], Dict[int, ChunkVal]]


def items_to_dask(
    asset_table: np.ndarray,
    spec: RasterSpec,
    chunksize: ChunksParam,
    resampling: Resampling = Resampling.nearest,
    dtype: np.dtype = np.dtype("float64"),
    fill_value: Union[int, float] = np.nan,
    rescale: bool = True,
    reader: Type[Reader] = AutoParallelRioReader,
    gdal_env: Optional[LayeredEnv] = None,
    errors_as_nodata: Tuple[Exception, ...] = (),
) -> da.Array:
    "Create a dask Array from an asset table"
    errors_as_nodata = errors_as_nodata or ()  # be sure it's not None

    if not np.can_cast(fill_value, dtype):
        raise ValueError(
            f"The fill_value {fill_value} is incompatible with the output dtype {dtype}. "
            f"Either use `dtype={np.array(fill_value).dtype.name!r}`, or pick a different `fill_value`."
        )

    chunks = normalize_chunks(chunksize, asset_table.shape + spec.shape, dtype)
    chunks_tb, chunks_yx = chunks[:2], chunks[2:]

    # The overall strategy in this function is to materialize the outer two dimensions (items, assets)
    # as one dask array (the "asset table"), then map a function over it which opens each URL as a `Reader`
    # instance (the "reader table").
    # Then, we use the `ArraySliceDep` `BlockwiseDep` to represent the inner inner two dimensions (y, x), and
    # `Blockwise` to create the cartesian product between them, avoiding materializing that entire graph.
    # Materializing the (items, assets) dimensions is unavoidable: every asset has a distinct URL, so that information
    # has to be included somehow.

    # make URLs into dask array, chunked as requested for the time,band dimensions
    asset_table_dask = da.from_array(
        asset_table,
        chunks=chunks_tb,
        inline_array=True,
        name="asset-table-" + dask.base.tokenize(asset_table),
    )

    # map a function over each chunk that opens that URL as a rasterio dataset
    with dask.annotate(fuse=False):
        # ^ HACK: prevent this layer from fusing to the next `fetch_raster_window` one.
        # This uses the fact that blockwise fusion doesn't happen when the layers' annotations
        # don't match, which may not be behavior we can rely on.
        # (The actual content of the annotation is irrelevant here, just that there is one.)
        reader_table = asset_table_dask.map_blocks(
            asset_table_to_reader_and_window,
            spec,
            resampling,
            dtype,
            fill_value,
            rescale,
            gdal_env,
            errors_as_nodata,
            reader,
            dtype=object,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=da.core.PerformanceWarning)

        name = f"fetch_raster_window-{dask.base.tokenize(reader_table, chunks, dtype, fill_value)}"
        # TODO use `da.blockwise` once it supports `BlockwiseDep`s as arguments
        lyr = blockwise(
            fetch_raster_window,
            name,
            "tbyx",
            reader_table.name,
            "tb",
            ArraySliceDep(chunks_yx),
            "yx",
            dtype,
            None,
            fill_value,
            None,
            numblocks={reader_table.name: reader_table.numblocks},  # ugh
        )
        dsk = HighLevelGraph.from_collections(name, lyr, [reader_table])
        rasters = da.Array(dsk, name, chunks, meta=np.ndarray((), dtype=dtype))

    return rasters


ReaderTableEntry = Optional[Tuple[Reader, windows.Window]]


def asset_table_to_reader_and_window(
    asset_table: np.ndarray,
    spec: RasterSpec,
    resampling: Resampling,
    dtype: np.dtype,
    fill_value: Union[int, float],
    rescale: bool,
    gdal_env: Optional[LayeredEnv],
    errors_as_nodata: Tuple[Exception, ...],
    reader: Type[Reader],
) -> np.ndarray:
    """
    "Open" an asset table by creating a `Reader` for each asset.

    This function converts the asset table (or chunks thereof) into an object array,
    where each element contains a tuple of the `Reader` and `Window` for that asset,
    or None if the element has no URL.
    """
    reader_table = np.empty_like(asset_table, dtype=object)
    for index, asset_entry in np.ndenumerate(asset_table):
        url: str | None = asset_entry["url"]
        if url:
            asset_bounds: Bbox = asset_entry["bounds"]
            asset_window = windows.from_bounds(*asset_bounds, spec.transform)

            entry: ReaderTableEntry = (
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
            reader_table[index] = entry
    return reader_table


def fetch_raster_window(
    reader_table: np.ndarray,
    slices: Tuple[slice, slice],
    dtype: np.dtype,
    fill_value: Union[int, float],
) -> np.ndarray:
    "Do a spatially-windowed read of raster data from all the Readers in the table."
    assert len(slices) == 2, slices
    current_window = windows.Window.from_slices(*slices)

    assert reader_table.size, f"Empty reader_table: {reader_table.shape=}"
    # Start with an empty output array, using the broadcast trick for even fewer memz.
    # If none of the assets end up actually existing, or overlapping the current window,
    # or containing data, we'll just return this 1-element array that's been broadcast
    # to look like a full-size array.
    output = np.broadcast_to(
        np.array(fill_value, dtype),
        reader_table.shape + (current_window.height, current_window.width),
    )

    all_empty: bool = True
    entry: ReaderTableEntry
    for index, entry in np.ndenumerate(reader_table):
        if entry:
            reader, asset_window = entry
            # Only read if the window we're fetching actually overlaps with the asset
            if windows.intersect(current_window, asset_window):
                # NOTE: when there are multiple assets, we _could_ parallelize these reads with our own threadpool.
                # However, that would probably increase memory usage, since the internal, thread-local GDAL datasets
                # would end up copied to even more threads.

                # TODO when the Reader won't be rescaling, support passing `output` to avoid the copy?
                data = reader.read(current_window)

                if all_empty:
                    # Turn `output` from a broadcast-trick array to a real array, so it's writeable
                    if (
                        np.isnan(data)
                        if np.isnan(fill_value)
                        else np.equal(data, fill_value)
                    ).all():
                        # Unless the data we just read is all empty anyway
                        continue
                    output = np.array(output)
                    all_empty = False

                output[index] = data

    return output


def normalize_chunks(
    chunks: ChunksParam, shape: Tuple[int, int, int, int], dtype: np.dtype
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Normalize chunks to tuple of tuples, assuming 1D and 2D chunks only apply to spatial coordinates

    If only 1 or 2 chunks are given, assume they're for the ``y, x`` coordinates,
    and that the ``time, band`` coordinates should be chunksize 1.
    """
    # TODO implement our own auto-chunking that makes the time,band coordinates
    # >1 if the spatial chunking would create too many tasks?
    if isinstance(chunks, int):
        chunks = (1, 1, chunks, chunks)
    elif isinstance(chunks, tuple) and len(chunks) == 2:
        chunks = (1, 1) + chunks

    return da.core.normalize_chunks(
        chunks,
        shape,
        dtype=dtype,
        previous_chunks=((1,) * shape[0], (1,) * shape[1], (shape[2],), (shape[3],)),
        # ^ Give dask some hint of the physical layout of the data, so it prefers widening
        # the spatial chunks over bundling together items/assets. This isn't totally accurate.
    )
