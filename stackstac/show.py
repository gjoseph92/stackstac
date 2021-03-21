from __future__ import annotations
from stackstac.raster_spec import RasterSpec

from typing import Awaitable, Dict, NamedTuple, Optional, Tuple, cast
import io
import asyncio
import logging
import warnings

# import urllib.parse

from aiohttp import web
import mercantile
import dask
import distributed
import numpy as np
import xarray as xr
from PIL import Image
import ipyleaflet
import dask.array as da

from . import geom_utils

Range = Tuple[float, float]

TILESIZE = 256
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
    checkerboard: bool


TOKEN_TO_ARRAY: Dict[str, Displayable] = {}


def show(
    arr: xr.DataArray,
    center=None,
    zoom=None,
    range: Optional[Range] = None,
    checkerboard: bool = True,
    **map_kwargs,
) -> ipyleaflet.Map:
    """
    Quickly create a Map displaying a DataArray
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

    add_to_map(arr, map_, range=range, checkerboard=checkerboard)
    return map_


def add_to_map(
    arr: xr.DataArray,
    map: ipyleaflet.Map,
    name: Optional[str] = None,
    range: Optional[Range] = None,
    checkerboard: bool = True,
) -> ipyleaflet.Layer:
    """
    Add the DataArray to a Map, as a new layer or replacing an existing one with the same name.

    By giving a name, you can change and re-run notebook cells without piling up extraneous layers on
    your map.
    """
    url = register(arr, range=range, checkerboard=checkerboard)
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
    arr: xr.DataArray, range: Optional[Range] = None, checkerboard: bool = True
) -> str:
    ensure_server()

    if arr.dtype.kind == "b":
        raise NotImplementedError(
            "Boolean arrays aren't supported yet. Please convert to a numeric type."
        )

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

    if range is None:
        warnings.warn(
            "Calculating 2nd and 98th percentile of the entire array, since no range was given. "
            "This could be expensive!"
        )

        def bandwise_percentile(arr: xr.DataArray, p):
            # TODO auto-use tdigest if available
            percentiles = da.concatenate(
                [da.percentile(band.flatten(), p) for band in arr.data]
            )
            return xr.DataArray(percentiles, dims=["band"], coords=dict(band=arr.band))

        flat = arr.data.flatten()
        mins, maxes = (
            da.percentile(flat, 2).persist(),
            da.percentile(flat, 98).persist()
            # bandwise_percentile(arr, 2).persist(),
            # bandwise_percentile(arr, 98).persist(),
        )
        arr = (arr - mins) / (maxes - mins)
        range = (0, 1)

    disp = Displayable(arr, range, checkerboard)
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
    client = distributed.get_client()
    bounds = mercantile.xy_bounds(mercantile.Tile(x, y, z))

    if not geom_utils.bounds_overlap(
        bounds, geom_utils.array_bounds(disp.arr, to_epsg=3857)
    ):
        return EMPTY_TILE_CHECKERBOARD if disp.checkerboard else EMPTY_TILE

    minx, miny, maxx, maxy = bounds
    tile = geom_utils.reproject_array(
        disp.arr,
        RasterSpec(
            epsg=3857,
            bounds=bounds,
            resolutions_xy=((maxx - minx) / TILESIZE, (maxy - miny) / TILESIZE),
        ),
    )
    assert tile.shape[1:] == (
        TILESIZE,
        TILESIZE,
    ), f"Wrong shape after interpolation: {tile.shape}"

    ordered = tile.transpose("band", "y", "x")
    delayed_png = delayed_arr_to_png(
        ordered.data, range=disp.range, checkerboard=disp.checkerboard
    )
    future = cast(distributed.Future, client.compute(delayed_png, sync=False))

    awaitable = future if client.asynchronous else future._result()
    # ^ sneak into the async api if the client isn't set up to be async.
    # this _should_ be okay, since we're running within the client's own event loop.
    return await cast(Awaitable[bytes], awaitable)


def arr_to_png(arr: np.ndarray, range: Range, checkerboard: bool) -> bytes:
    assert len(arr) <= 3, f"Array must have at most 3 bands. Array shape: {arr.shape}"

    alpha_mask = ~np.isnan(arr).all(axis=0)  # TODO non-nan fill value
    alpha = alpha_mask.astype("uint8", copy=False) * 255

    # TODO boolean arrays
    # TODO multi-band scales?
    # TODO colormap within here??
    min_, max_ = range
    if min_ != max_:
        arr -= min_
        arr /= max_ - min_
        arr *= 255
    else:
        arr[:] = 0

    np.nan_to_num(arr, copy=False)
    np.clip(arr, 0, 255, out=arr)
    arr = arr.astype("uint8", copy=False)

    if checkerboard:
        checkers = make_checkerboard(max(arr.shape[1:]), 8)
        checkers = checkers[: arr.shape[1], : arr.shape[2]]
        alpha[~alpha_mask & checkers] = 30

    arr = np.concatenate(
        [arr, alpha[None]] if len(arr) != 2 else [arr, arr[[0]], alpha[None]],
        axis=0,
    )
    arr = np.moveaxis(arr, 0, -1).astype("uint8", copy=False)

    image = Image.fromarray(arr)
    file = io.BytesIO()
    image.save(file, format="png")
    return file.getvalue()


delayed_arr_to_png = dask.delayed(arr_to_png, pure=True)


def make_checkerboard(arr_size: int, checker_size: int):
    n_cells = arr_size // checker_size
    n_half_cells = n_cells // 2
    base = [[True, False] * n_half_cells, [False, True] * n_half_cells] * n_half_cells
    board = np.kron(base, np.ones((checker_size, checker_size), dtype=bool))
    return board


EMPTY_TILE_CHECKERBOARD = arr_to_png(
    np.full((1, TILESIZE, TILESIZE), np.nan), range=(0, 1), checkerboard=True
)
EMPTY_TILE = arr_to_png(
    np.full((1, TILESIZE, TILESIZE), np.nan), range=(0, 1), checkerboard=False
)
