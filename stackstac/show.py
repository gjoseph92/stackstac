from __future__ import annotations

from typing import Awaitable, Optional, Tuple, cast
import sys
import functools
import traceback
import weakref
import io
import asyncio
import logging

# import urllib.parse

from aiohttp import web
import mercantile
import dask
import distributed
import numpy as np
import xarray as xr
from PIL import Image
import ipyleaflet

from .prepare import bounds_overlap

Scales = Tuple[float, float]

TILESIZE = 256
PORT = 8000

arrs: weakref.WeakValueDictionary[str, xr.DataArray] = weakref.WeakValueDictionary()
routes = web.RouteTableDef()

logger = logging.getLogger(__file__)


def show(arr: xr.DataArray, map_: ipyleaflet.Map) -> ipyleaflet.Layer:
    ensure_server()

    assert getattr(arr, "crs", None) in (
        "epsg:3857",
        None,
    ), "Only Web Mercator (EPSG:3857) arrays are supported currently"
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

    token = dask.base.tokenize(arr)
    arrs[token] = arr

    # TODO proxy through jupyter so another port doesn't have to be open to the internet

    # browser_url = urllib.parse.urlparse(map_.window_url)
    # base_url = f"{browser_url.scheme}://{browser_url.netloc}"
    # url = urllib.parse.urljoin(
    #     base_url, f"proxy/127.0.0.1/{PORT}/{token}/{{z}}/{{y}}/{{x}}.png"
    # )
    url = f"http://localhost:{PORT}/{token}/{{z}}/{{y}}/{{x}}.png"

    # TODO everything around named layers. Maybe will switch keys from tokens to names too??
    lyr = ipyleaflet.TileLayer(name=arr.name, url=url)
    map_.add_layer(lyr)
    return lyr


# _async_client: Optional[distributed.Client] = None
_started: bool = False


# def get_async_client():
#     global _async_client
#     current_cluster = distributed.get_client().cluster
#     if _async_client is None or _async_client.cluster is not current_cluster:
#         _async_client = distributed.Client(
#             current_cluster, asynchronous=True, set_as_default=False
#         )
#     return _async_client


def ensure_server():
    global _started
    if not _started:
        _launch_server()
        _started = True


def _launch_server():
    distributed.get_client()  # just for the error; make sure there is one

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

    asyncio.create_task(run())


@routes.get("/{hash}/{z}/{y}/{x}.png")
async def handler(request: web.Request) -> web.Response:
    hash = request.match_info["hash"]
    try:
        arr = arrs[hash]
    except KeyError:
        raise web.HTTPNotFound(reason=f"{hash} not found. Have {list(arrs)}.") from None

    try:
        z = int(request.match_info["z"])
        y = int(request.match_info["y"])
        x = int(request.match_info["x"])
    except TypeError:
        raise web.HTTPBadRequest(reason="z, y, and z parameters must be ints")

    # TODO request cancellation
    png = await compute_tile(arr, z, y, x)
    return web.Response(
        body=png,
        status=200,
        content_type="image/png",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def compute_tile(arr: xr.DataArray, z: int, y: int, x: int) -> bytes:
    client = distributed.get_client()
    bounds = mercantile.xy_bounds(mercantile.Tile(x, y, z))

    # TODO reprojection! (try just using xarray interp for warping for now, might actually work)
    # for now, assume array is 3857.
    arr_bounds = (
        arr.x.min().item(),
        arr.y.min().item(),
        arr.x.max().item(),
        arr.y.max().item(),
    )
    if not bounds_overlap(bounds, arr_bounds):
        return EMPTY_TILE

    # TODO this seems to be off-by-one-ish. Also pixel centers vs corners?
    xs = np.linspace(bounds.left, bounds.right, TILESIZE)
    ys = np.linspace(bounds.top, bounds.bottom, TILESIZE)
    tile = arr.interp(x=xs, y=ys, method="linear", kwargs=dict(fill_value=None))
    assert tile.shape[1:] == (
        TILESIZE,
        TILESIZE,
    ), f"Wrong shape after interpolation: {tile.shape}"

    ordered = tile.transpose("band", "y", "x")
    delayed_png = delayed_arr_to_png(ordered.data, scales=(0, 4000))
    future = client.compute(delayed_png, sync=False)

    awaitable = future if client.asynchronous else future._result()
    return await awaitable  # sneak into async api

    # return await cast(Awaitable[bytes], client.compute(png))


def arr_to_png(arr: np.ndarray, scales: Scales, checkerboard: bool = True) -> bytes:
    assert len(arr) <= 3, f"Array must have at most 3 bands. Array shape: {arr.shape}"

    alpha_mask = ~np.isnan(arr).all(axis=0)  # TODO non-nan fill value
    alpha = alpha_mask.astype("uint8", copy=False) * 255

    # TODO colormap within here??
    # TODO boolean arrays?
    min_, max_ = scales
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
        checkers = _checkerboard(max(arr.shape[1:]), 8)
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


def _checkerboard(size, cell_size):
    n_cells = size // cell_size
    n_half_cells = n_cells // 2
    base = [[True, False] * n_half_cells, [False, True] * n_half_cells] * n_half_cells
    board = np.kron(base, np.ones((cell_size, cell_size), dtype=bool))
    return board


EMPTY_TILE = arr_to_png(np.full((1, TILESIZE, TILESIZE), np.nan), scales=(0, 1))
