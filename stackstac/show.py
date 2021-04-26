from __future__ import annotations
import functools

from typing import Awaitable, Dict, Literal, NamedTuple, Optional, Tuple, Union, cast
import math
import io
import asyncio
import logging
import warnings

from aiohttp import web
import mercantile
import dask
import distributed
import numpy as np
import xarray as xr
from PIL import Image
import ipyleaflet
import dask.array as da
import matplotlib.cm
import matplotlib.colors

from . import geom_utils
from .raster_spec import RasterSpec

Range = Tuple[float, float]

PORT = 8000
routes = web.RouteTableDef()

# TODO figure out weakref situation.
# We don't want to be the only ones holding onto a persisted Dask collection;
# this would make it impossible for users to release a persisted collection from distributed memory.
# However, just holding onto weakrefs can cause other issues: it's common to display an ephemeral
# result like `.show(x + 1)`; if the value is dropped from our lookup dict immediately,
# then the HTTP requests to render tiles will 404.
# For now, we're calling future-leaking a future problem, and storing everything in a normal dict
# until what we want from the visualization API is more fleshed out.

# arrs: weakref.WeakValueDictionary[str, xr.DataArray] = weakref.WeakValueDictionary()


class Displayable(NamedTuple):
    arr: xr.DataArray
    range: Range
    colormap: Optional[matplotlib.colors.Colormap]
    checkerboard: bool
    tilesize: int
    interpolation: Literal["linear", "nearest"]


TOKEN_TO_ARRAY: Dict[str, Displayable] = {}


def show(
    arr: xr.DataArray,
    center=None,
    zoom=None,
    range: Optional[Range] = None,
    colormap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    checkerboard: bool = True,
    interpolation: Literal["linear", "nearest"] = "linear",
    **map_kwargs,
) -> ipyleaflet.Map:
    """
    Quickly create an ipyleaflet :doc:`api_reference/map` displaying a `~xarray.DataArray`.

    As you pan around the map, the part of the array that's in view is computed on the fly by dask.
    This requires using a :doc:`dask distributed <distributed:index>` cluster.

    Parameters
    ----------
    arr:
        `~xarray.DataArray` to visualize. Must have ``x`` and ``y``, and optionally ``band`` dims,
        and the ``epsg`` coordinate set.

        ``arr`` must have 1-3 bands. Single-band data can be colormapped; multi-band data will be
        displayed as RGB. For 2-band arrays, the first band will be duplicated into the third band's spot,
        then shown as RGB.
    center:
        Centerpoint for the map. If None (default), the map will automatically be centered on the array.
    zoom:
        Initial zoom level for the map. If None (default), the map will automatically be zoomed to fit the entire array.
    range:
        Min and max values in ``arr`` which will become black (0) and white (255) in the visualization.

        If None (default), it will automatically use the 2nd/98th percentile values of the *entire array*
        (unless it's a boolean array; then we just use 0-1).
        For large arrays, this can be very slow and expensive, and slow down tile rendering a lot, so
        passing an explicit range is usually a good idea.
    colormap:
        Colormap to use for single-band data. Can be a
        :doc:`matplotlib colormap name <gallery/color/colormap_reference>` as a string,
        or a `~matplotlib.colors.Colormap` object for custom colormapping.

        If None (default), the default matplotlib colormap (usually ``viridis``) will automatically
        be used for 1-band data. Setting a colormap for multi-band data is an error.
    checkerboard:
        Whether to show a checkerboard pattern for missing data (default), or leave it fully transparent.

        Note that only NaN is considered a missing value; any custom fill value should be converted
        to NaN before visualizing.
    interpolation:
        Interpolation method to use while reprojecting: ``"linear"`` or ``"nearest"`` (default ``"linear"``).
        Use ``"linear"`` for continuous data, such as imagery, SAR, DEMs, weather data, etc. Use ``"nearest"``
        for discrete/categorical data, such as classification maps.

    Note
    ----
    Why do the tiles seem to show up in batches, and why does it take along time for all the batches to load?

    Unfortunately, this is a web browser limitation, and there's not much stackstac can do about it.
    Each 256x256 tile visible on the map consumes one connection, and Web browsers only allow
    a fixed number of connections to be open at once (for example, 6 per host in Chrome).
    It's the browser requesting a tile that causes dask to start computing it, so in Chrome's case,
    only 6 tiles can ever be computing at once. This limits dask's parallelism in computing tiles,
    and causes the "bunching" effect.

    Returns
    -------
    ipyleaflet.Map:
        The new map showing this array.
    """
    map_ = ipyleaflet.Map(**map_kwargs)
    if center is None:
        west, south, east, north = geom_utils.array_bounds(arr, to_epsg=4326)
        map_.fit_bounds([[south, east], [north, west]])
        # TODO ipyleaflet `fit_bounds` doesn't do a very good job
    else:
        map_.center = center

    if zoom is not None:
        map_.zoom = zoom

    add_to_map(
        arr,
        map_,
        range=range,
        colormap=colormap,
        checkerboard=checkerboard,
        interpolation=interpolation,
    )
    return map_


def add_to_map(
    arr: xr.DataArray,
    map: ipyleaflet.Map,
    name: Optional[str] = None,
    range: Optional[Range] = None,
    colormap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    checkerboard: bool = True,
    interpolation: Literal["linear", "nearest"] = "linear",
) -> ipyleaflet.Layer:
    """
    Add the `~xarray.DataArray` to an ipyleaflet :doc:`api_reference/map` as a new layer or replacing an existing one.

    By giving a name, you can change and re-run notebook cells without piling up extraneous layers on
    your map.

    As you pan around the map, the part of the array that's in view is computed on the fly by dask.
    This requires using a :doc:`dask distributed <distributed:index>` cluster.

    Parameters
    ----------
    arr:
        `~xarray.DataArray` to visualize. Must have ``x`` and ``y``, and optionally ``band`` dims,
        and the ``epsg`` coordinate set.

        ``arr`` must have 1-3 bands. Single-band data can be colormapped; multi-band data will be
        displayed as RGB. For 2-band arrays, the first band will be duplicated into the third band's spot,
        then shown as RGB.
    map:
        ipyleaflet :doc:`api_reference/map` to show the array on.
    name: str
        Name of the layer. If there's already a layer with this name on the map, its URL will be updated.
        Otherwise, a new layer is added.

        If None (default), a new layer is always added, which will have the name ``arr.name``.
    range:
        Min and max values in ``arr`` which will become black (0) and white (255) in the visualization.

        If None (default), it will automatically use the 2nd/98th percentile values of the *entire array*
        (unless it's a boolean array; then we just use 0-1).
        For large arrays, this can be very slow and expensive, and slow down tile rendering a lot, so
        passing an explicit range is usually a good idea.
    colormap:
        Colormap to use for single-band data. Can be a
        :doc:`matplotlib colormap name <gallery/color/colormap_reference>` as a string,
        or a `~matplotlib.colors.Colormap` object for custom colormapping.

        If None (default), the default matplotlib colormap (usually ``viridis``) will automatically
        be used for 1-band data. Setting a colormap for multi-band data is an error.
    checkerboard:
        Whether to show a checkerboard pattern for missing data (default), or leave it fully transparent.

        Note that only NaN is considered a missing value; any custom fill value should be converted
        to NaN before visualizing.
    interpolation:
        Interpolation method to use while reprojecting: ``"linear"`` or ``"nearest"`` (default ``"linear"``).
        Use ``"linear"`` for continuous data, such as imagery, SAR, DEMs, weather data, etc. Use ``"nearest"``
        for discrete/categorical data, such as classification maps.


    Note
    ----
    Why do the tiles seem to show up in batches, and why does it take along time for all the batches to load?

    Unfortunately, this is a web browser limitation, and there's not much stackstac can do about it.
    Each 256x256 tile visible on the map consumes one connection, and Web browsers only allow
    a fixed number of connections to be open at once (for example, 6 per host in Chrome).
    It's the browser requesting a tile that causes dask to start computing it, so in Chrome's case,
    only 6 tiles can ever be computing at once. This limits dask's parallelism in computing tiles,
    and causes the "bunching" effect.

    Returns
    -------
    ipyleaflet.Layer:
        The new or existing layer for visualizing this array.
    """
    url = register(
        arr,
        range=range,
        colormap=colormap,
        checkerboard=checkerboard,
        interpolation=interpolation,
    )
    if name is not None:
        for lyr in map.layers:
            if lyr.name == name:
                lyr.url = url
                break
        else:
            lyr = ipyleaflet.TileLayer(name=name, url=url)
            map.add_layer(lyr)
    else:
        lyr = ipyleaflet.TileLayer(name=arr.name, url=url)
        map.add_layer(lyr)
    return lyr


def register(
    arr: xr.DataArray,
    range: Optional[Range] = None,
    colormap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    checkerboard: bool = True,
    tilesize: int = 256,
    interpolation: Literal["linear", "nearest"] = "linear",
) -> str:
    """
    Low-level method to register a `DataArray` for display on a web map, and spin up the HTTP server if necessary.

    Registration is necessary so that the local web server can look up the right
    `~xarray.DataArray` object to render based on the URL requested.

    A `distributed.Client` must already be created (and set as default) before calling this.

    Once registered, an array cannot currently be un-registered. Beware of this when visualizing
    things you've called `~distributed.Client.persist` on: even if you try to release a persisted
    object from your own code, if you've ever registered it for visualization, it won't be freed
    from distributed memory.
    """
    ensure_server()

    geom_utils.array_epsg(arr)  # just for the error
    if arr.ndim == 2:
        assert set(arr.dims) == {"x", "y"}
        arr = arr.expand_dims("band")
    elif arr.ndim == 3:
        assert set(arr.dims) == {"band", "x", "y"}
        nbands = arr.sizes["band"]
        assert 1 <= nbands <= 3, f"Array must have 1-3 bands, not {nbands}."
    else:
        raise ValueError(
            f"Array must have the dimensions 'x', 'y', and optionally 'band', not {arr.dims!r}"
        )

    arr = arr.transpose("band", "y", "x")

    if arr.shape[0] == 1:
        if colormap is None:
            # use the default colormap for 1-band data (usually viridis)
            colormap = matplotlib.cm.get_cmap()
    elif colormap is not None:
        raise ValueError(
            f"Colormaps are only possible on single-band data; this array has {arr.shape[0]} bands: "
            f"{arr.bands.data.tolist()}"
        )

    if isinstance(colormap, str):
        colormap = matplotlib.cm.get_cmap(colormap)

    if range is None:
        if arr.dtype.kind == "b":
            range = (0, 1)
        else:
            warnings.warn(
                "Calculating 2nd and 98th percentile of the entire array, since no range was given. "
                "This could be expensive!"
            )

            flat = arr.data.flatten()
            mins, maxes = (
                # TODO auto-use tdigest if available
                # NOTE: we persist the percentiles to be sure they aren't recomputed when you pan/zoom
                da.percentile(flat, 2).persist(),
                da.percentile(flat, 98).persist(),
            )
            arr = (arr - mins) / (maxes - mins)
            range = (0, 1)
    else:
        vmin, vmax = range
        if vmin > vmax:
            raise ValueError(f"Invalid range: min value {vmin} > max value {vmax}")

    assert tilesize > 1, f"Tilesize must be greater than zero, not {tilesize}"

    disp = Displayable(arr, range, colormap, checkerboard, tilesize, interpolation)
    token = dask.base.tokenize(disp)
    TOKEN_TO_ARRAY[token] = disp

    # TODO proxy through jupyter so another port doesn't have to be open to the internet
    return f"http://localhost:{PORT}/{token}/{{z}}/{{y}}/{{x}}.png"
    # TODO some way to unregister (this is hard because we may have modified `arr` in `register`)


def ensure_server():
    if not ensure_server._started:
        _launch_server()
        ensure_server._started = True


ensure_server._started = False


def _launch_server():
    client = distributed.get_client()  # if there isn't one yet, this will error

    app = web.Application()
    app.add_routes(routes)

    async def run():
        runner = web.AppRunner(app, logger=logging.getLogger("root"))
        # ^ NOTE: logs only seem to show up in Jupyter if we use the root logger, AND
        # set the log level in the JupyterLab UI to `info` or `debug`. This makes no sense.
        try:
            await runner.setup()
            site = web.TCPSite(runner, "localhost", PORT)
            await site.start()

            while True:
                await asyncio.sleep(3600)  # sleep forever
        finally:
            await runner.cleanup()

    # Ensure our server runs in the same event loop as the distributed client,
    # so the server can properly `await` results
    client.loop.spawn_callback(run)
    # ^ NOTE: tornado equivalent of `asyncio.create_task(run())`
    # (note that it takes the function itself, not the coroutine object)


@routes.get("/{hash}/{z}/{y}/{x}.png")
async def handler(request: web.Request) -> web.Response:
    "Handle a HTTP GET request for an XYZ tile"
    hash = request.match_info["hash"]
    try:
        disp = TOKEN_TO_ARRAY[hash]
    except KeyError:
        raise web.HTTPNotFound(
            reason=f"{hash} not found. Have {list(TOKEN_TO_ARRAY)}."
        ) from None

    try:
        z = int(request.match_info["z"])
        y = int(request.match_info["y"])
        x = int(request.match_info["x"])
    except TypeError:
        raise web.HTTPBadRequest(reason="z, y, and z parameters must be ints")

    # TODO request cancellation
    png = await compute_tile(disp, z, y, x)
    return web.Response(
        body=png,
        status=200,
        content_type="image/png",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def compute_tile(disp: Displayable, z: int, y: int, x: int) -> bytes:
    "Send an XYZ tile to be computed by the distributed client, and wait for it."
    client = distributed.get_client()
    # TODO assert the client's loop is the same as our current event loop.
    # If not... tell the server to shut down and restart on the new event loop?
    # (could also do this within a watch loop in `_launch_server`.)
    bounds = mercantile.xy_bounds(mercantile.Tile(x, y, z))

    if not geom_utils.bounds_overlap(
        bounds, geom_utils.array_bounds(disp.arr, to_epsg=3857)
    ):
        return empty_tile(disp.tilesize, disp.checkerboard)

    minx, miny, maxx, maxy = bounds
    # FIXME: `reproject_array` is really, really slow for large arrays
    # because of all the dask-graph-munging. Having a blocking, GIL-bound
    # function within an async handler like this also means we're basically
    # sending requests out serially per tile
    tile = geom_utils.reproject_array(
        disp.arr,
        RasterSpec(
            epsg=3857,
            bounds=bounds,
            resolutions_xy=(
                (maxx - minx) / disp.tilesize,
                (maxy - miny) / disp.tilesize,
            ),
        ),
        interpolation=disp.interpolation,
    )
    assert tile.shape[1:] == (
        disp.tilesize,
        disp.tilesize,
    ), f"Wrong shape after interpolation: {tile.shape}"

    delayed_png = delayed_arr_to_png(
        tile.data,
        range=disp.range,
        colormap=disp.colormap,
        checkerboard=disp.checkerboard,
    )
    future = cast(distributed.Future, client.compute(delayed_png, sync=False))

    awaitable = future if client.asynchronous else future._result()
    # ^ sneak into the async api if the client isn't set up to be async.
    # this _should_ be okay, since we're running within the client's own event loop.
    return await cast(Awaitable[bytes], awaitable)


def arr_to_png(
    arr: np.ndarray,
    range: Range,
    colormap: Optional[matplotlib.colors.Colormap] = None,
    checkerboard: bool = True,
) -> bytes:
    "Convert an ndarray into a PNG"
    # TODO multi-band scales?
    # TODO non-nan fill values
    assert len(arr) <= 3, f"Array must have at most 3 bands. Array shape: {arr.shape}"

    working_dtype = arr.dtype
    # https://github.com/matplotlib/matplotlib/blob/8b02ed1f0af7956c7f42b76bf40a86f048a0454e/lib/matplotlib/colors.py#L1169-L1171
    if np.issubdtype(working_dtype, np.integer) or working_dtype.type is np.bool_:
        # bool_/int8/int16 -> float32; int32/int64 -> float64
        working_dtype = np.promote_types(working_dtype, np.float32)

    vmin, vmax = range
    norm_arr = arr.astype(working_dtype, copy=True)
    if vmin == vmax:
        norm_arr.fill(0)
    else:
        norm_arr -= vmin
        norm_arr /= vmax - vmin

    if colormap is not None:
        # NOTE: `Colormap` automatically uses `np.isnan(x)` as the mask
        cmapped = colormap(norm_arr, bytes=True)
        cmapped = np.moveaxis(np.squeeze(cmapped), -1, 0)
        u8_arr, alpha = cmapped[:-1], cmapped[-1]
    else:
        u8_arr = np.clip(np.nan_to_num(norm_arr * 255), 0, 255).astype("uint8")
        mask = np.isnan(arr).any(axis=0)
        alpha = (~mask).astype("uint8", copy=False)
        alpha *= 255

    if checkerboard:
        checkers = make_checkerboard(max(arr.shape[1:]), 8)
        checkers = checkers[: arr.shape[1], : arr.shape[2]]
        alpha[(alpha == 0) & checkers] = 30

    img_arr = np.concatenate(
        [u8_arr, alpha[None]]
        if len(u8_arr) != 2
        else [u8_arr, u8_arr[[0]], alpha[None]],
        axis=0,
    )
    img_arr = np.moveaxis(img_arr, 0, -1)

    image = Image.fromarray(img_arr)
    file = io.BytesIO()
    image.save(file, format="png")
    return file.getvalue()


delayed_arr_to_png = dask.delayed(arr_to_png, pure=True)


def make_checkerboard(arr_size: int, checker_size: int):
    n_cells = arr_size / checker_size
    n_half_cells = math.ceil(n_cells / 2)
    base = [[True, False] * n_half_cells, [False, True] * n_half_cells] * n_half_cells
    board = np.kron(base, np.ones((checker_size, checker_size), dtype=bool))
    return board


@functools.lru_cache(maxsize=64)
def empty_tile(tilesize: int, checkerboard: bool) -> bytes:
    empty = np.full((1, tilesize, tilesize), np.nan)
    return arr_to_png(empty, range=(0, 1), checkerboard=checkerboard)
