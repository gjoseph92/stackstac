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
ChunksParam = Union[
    ChunkVal, Tuple[Union[ChunkVal, Tuple[ChunkVal, ...]], ...], Dict[int, ChunkVal]
]
TBYXChunks = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


def items_to_dask(
    asset_table: np.ndarray,
    spec: RasterSpec,
    chunksize: ChunksParam,
    nbands_per_asset: tuple[int, ...],
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

    chunks, asset_table_band_chunks = normalize_chunks(
        chunksize, asset_table.shape + spec.shape, nbands_per_asset, dtype
    )
    chunks_tb = chunks[:1] + asset_table_band_chunks
    chunks_yx = chunks[2:]

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
        chunks=chunks_tb,  # type: ignore
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
            nbands_per_asset,
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
            asset_window = window_from_bounds(asset_bounds, spec.transform)

            entry: ReaderTableEntry = (
                reader(
                    url=url,
                    bands=asset_entry["bands"],
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
    nbands_per_asset: tuple[int, ...],
) -> np.ndarray:
    "Do a spatially-windowed read of raster data from all the Readers in the table."
    assert len(slices) == 2, slices
    current_window = windows.Window.from_slices(*slices)

    assert reader_table.size, f"Empty reader_table: {reader_table.shape=}"
    assert (
        len(nbands_per_asset) == reader_table.shape[1]
    ), f"{nbands_per_asset=}, {reader_table.shape[1]=}"
    # Start with an empty output array, using the broadcast trick for even fewer memz.
    # If none of the assets end up actually existing, or overlapping the current window,
    # or containing data, we'll just return this 1-element array that's been broadcast
    # to look like a full-size array.
    output = np.broadcast_to(
        np.array(fill_value, dtype),
        (
            reader_table.shape[0],
            sum(nbands_per_asset),
            current_window.height,
            current_window.width,
        ),
    )

    asset_i_to_band = np.cumsum(nbands_per_asset)
    all_empty: bool = True
    entry: ReaderTableEntry
    for (time_i, asset_i), entry in np.ndenumerate(reader_table):
        if entry:
            reader, asset_window = entry
            # Only read if the window we're fetching actually overlaps with the asset
            if windows.intersect(current_window, asset_window):
                # NOTE: when there are multiple assets, we _could_ parallelize these reads with our own threadpool.
                # However, that would probably increase memory usage, since the internal, thread-local GDAL datasets
                # would end up copied to even more threads.

                # TODO when the Reader won't be rescaling, support passing `output` to avoid the copy?
                data = reader.read(current_window)
                assert (
                    data.shape[0] == nbands_per_asset[asset_i]
                ), f"Band count mismatch: {nbands_per_asset[asset_i]=}, {data.shape[0]=}"

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

                band_i = asset_i_to_band[asset_i]
                output[time_i, band_i : band_i + data.shape[0]] = data

    return output


def normalize_chunks(
    chunks: ChunksParam,
    shape: Tuple[int, int, int, int],
    nbands_per_asset: tuple[int, ...],
    dtype: np.dtype,
) -> tuple[TBYXChunks, tuple[int, ...]]:
    """
    Normalize chunks to tuple of tuples, assuming 1D and 2D chunks only apply to spatial coordinates

    If only 1 or 2 chunks are given, assume they're for the ``y, x`` coordinates,
    and that the ``time, band`` coordinates should be chunksize 1.

    If "auto" is given for bands, uses ``nbands_per_asset``.

    Returns
    -------
    chunks:
        Normalized chunks
    asset_table_band_chunks:
        Band chunks to apply to the asset table (see `process_multiband_chunks`)
    """
    # TODO implement our own auto-chunking that makes the time,band coordinates
    # >1 if the spatial chunking would create too many tasks?
    if isinstance(chunks, int):
        chunks = (1, nbands_per_asset, chunks, chunks)
    elif isinstance(chunks, tuple) and len(chunks) == 2:
        chunks = (1, nbands_per_asset) + chunks
    elif isinstance(chunks, tuple) and len(chunks) == 4 and chunks[1] == "auto":
        chunks = (chunks[0], nbands_per_asset, chunks[2], chunks[3])

    norm: TBYXChunks = da.core.normalize_chunks(
        chunks,
        shape,
        dtype=dtype,
        previous_chunks=((1,) * shape[0], nbands_per_asset, (shape[2],), (shape[3],)),
        # ^ Give dask some hint of the physical layout of the data, so it prefers widening
        # the spatial chunks over bundling together items/assets. This isn't totally accurate.
    )

    # Ensure we aren't trying to split apart multi-band assets. This would require rewriting
    # the asset table (adding duplicate columns) and is generally not what you want, assuming
    # that in multi-band assets, the bands are stored interleaved, so reading one requires reading
    # them all anyway.
    asset_table_band_chunks = process_multiband_chunks(norm[1], nbands_per_asset)
    return norm, asset_table_band_chunks


def process_multiband_chunks(
    chunks: tuple[int, ...], nbands_per_asset: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Validate that the bands chunks don't try to split apart any multi-band assets.

    Returns
    -------
    asset_table_band_chunks:
        Band chunks to apply to the asset table (so that assets are combined into single chunks as necessary).
        ``len(asset_table_band_chunks) == len(chunks)``. In other words, along the bands, we'll have the same
        ``numblocks`` in the asset table as ``numblocks`` in the final array. But each block in the final array
        may be longer (have more bands) than the number of assets (when they're multi-band assets).
    """
    n_chunks = len(chunks)
    n_assets = len(nbands_per_asset)

    final_msg = (
        f"Requested bands chunks: {chunks}\n"
        f"Physical bands chunks: {nbands_per_asset}\n"
        "This would entail splitting apart multi-band assets. This typically (but not always) has "
        "much worse performance, since GeoTIFF bands are generally interleaved (so reading one "
        "band from a file requires reading them all).\n"
        "If you have a use-case for this, please discuss on https://github.com/gjoseph92/stackstac/issues."
    )

    if n_chunks > n_assets:
        raise NotImplementedError(
            f"Refusing to make {n_chunks} chunk(s) for the bands when there are only {n_assets} bands asset(s).\n"
            + final_msg
        )
    elif n_chunks == n_assets:
        if chunks != nbands_per_asset:
            raise NotImplementedError(final_msg)
        return chunks
    else:
        # Trying to combine multiple assets into one chunk; must be whole multiples.
        # n_chunks < n_assets
        asset_table_band_chunks: list[int] = []
        i = nbands_so_far = nbands_requested = n_assets_so_far = 0
        for nb in nbands_per_asset:
            if nbands_requested == 0:
                if i == n_chunks:
                    raise ValueError(
                        f"Invalid chunks for {sum(nbands_per_asset)} band(s): only {sum(chunks)} band(s) used.\n"
                        f"Requested bands chunks: {chunks}\n"
                        f"Physical bands chunks: {nbands_per_asset}\n"
                    )
                nbands_requested = chunks[i]

            nbands_so_far += nb
            n_assets_so_far += 1
            if nbands_so_far < nbands_requested:
                continue
            elif nbands_so_far == nbands_requested:
                # nailed it
                i += 1
                nbands_so_far = 0
                nbands_requested = 0
                asset_table_band_chunks.append(n_assets_so_far)
                n_assets_so_far = 0
            else:
                # `nbands_so_far > nbands_requested`
                raise NotImplementedError(
                    f"Specified chunks do not evenly combine multi-band assets: chunk {i} would split one apart.\n"
                    + final_msg
                )
        return tuple(asset_table_band_chunks)


# FIXME remove this once rasterio bugs are fixed
def window_from_bounds(bounds: Bbox, transform: Affine) -> windows.Window:
    "Get the window corresponding to the bounding coordinates (correcting for rasterio bugs)"
    window = windows.from_bounds(
        *bounds,
        transform=transform,
        precision=0.0
        # ^ https://github.com/rasterio/rasterio/issues/2374
    )

    # Trim negative `row_off`/`col_off` to work around https://github.com/rasterio/rasterio/issues/2378
    # Note this does actually alter the window: it clips off anything that was out-of-bounds to the
    # west/north of the `transform`'s origin. So the size and origin of the window is no longer accurate.
    # This is okay for our purposes, since we only use these windows for intersection-testing to see if
    # an asset intersects our current chunk. We don't care about the parts of the asset that fall
    # outside our AOI.
    window = windows.Window(
        max(window.col_off, 0),  # type: ignore "Expected 0 positional arguments"
        max(window.row_off, 0),
        (max(window.col_off + window.width, 0) if window.col_off < 0 else window.width),
        (
            max(window.row_off + window.height, 0)
            if window.row_off < 0
            else window.height
        ),
    )
    return window
