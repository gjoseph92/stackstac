from __future__ import annotations
from asyncio.tasks import Task
from dataclasses import dataclass
import re

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
import math
import io
import asyncio
import logging
import warnings
import functools

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
import ipywidgets
from traitlets import traitlets  # pylance prefers this

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


class ServerStats(ipywidgets.VBox):
    """
    ipywidget for monitoring the local webserver's progress rendering map tiles with Dask

    Do not instantiate directly; use `stackstac.server_stats` instead.
    """

    computing = traitlets.Int(default_value=0)
    requested = traitlets.Int(default_value=0)
    completed = traitlets.Int(default_value=0)
    cancelled = traitlets.Int(default_value=0)
    errored = traitlets.Int(default_value=0)

    def __init__(self, name: str = "", **kwargs) -> None:
        self.output = ipywidgets.Output()
        self._computing_progress = ipywidgets.IntProgress(
            value=0, min=0, max=6, description="0 computing"
        )
        self._requested_progress = ipywidgets.IntProgress(
            value=0, min=0, max=6, description="0 requested"
        )
        completed = ipywidgets.Label(value="0 completed")
        traitlets.dlink(
            (self, "completed"), (completed, "value"), lambda c: f"{c} completed"
        )
        cancelled = ipywidgets.Label(value="0 cancelled")
        traitlets.dlink(
            (self, "cancelled"), (cancelled, "value"), lambda c: f"{c} cancelled"
        )
        errored = ipywidgets.Label(value="0 errored")
        traitlets.dlink((self, "errored"), (errored, "value"), lambda c: f"{c} errored")
        super().__init__(
            children=[
                ipywidgets.HTML(value=f"<b>{name}</b>"),
                self._requested_progress,
                self._computing_progress,
                completed,
                cancelled,
                errored,
                self.output,
            ],
            **kwargs,
        )

    @traitlets.observe("computing", "requested")
    def _progbar_change(self, event):
        attr = event["name"]
        progbar = getattr(self, f"_{attr}_progress")
        value = event["new"]
        with progbar.hold_trait_notifications():
            progbar.value = value
            progbar.description = f"{value} {progbar.description.split(' ')[1]}"
            if value > progbar.max:
                progbar.max = value


server_stats = ipywidgets.VBox(children=[])


class TileManager:
    @dataclass
    class TileRef:
        task: Task[bytes]
        done_callback: Callable[[], Any]
        refcount: int = 0
        released: bool = False

        def incref(self):
            assert (
                not self.released
            ), f"Incref on a released task. {self.task.cancelled()=} {self.refcount=}"
            assert (
                not self.task.cancelled()
            ), f"Incref on a cancelled task, {self.refcount=}"
            self.refcount += 1

        def decref(self):
            if self.refcount > 0:
                self.refcount -= 1
                if self.refcount == 0:
                    self.task.cancel()
                    self.released = True
                    self.done_callback()

        def cancel(self):
            if self.refcount or not self.released:
                self.refcount = 1
                self.decref()

        def __enter__(self) -> None:
            self.incref()

        def __exit__(self, *args) -> None:
            self.decref()

        def __del__(self):
            if not self.released:
                self.task.cancel()
                self.done_callback()
            assert (
                self.refcount == 0
            ), f"__del__ on TileRef with nonzero refcount: {self.refcount=} {self.released=}"

    def __init__(
        self, disp: Displayable, token: str, name: str, loop: asyncio.AbstractEventLoop
    ):
        self.disp = disp
        self.token = token
        self.loop = loop
        self.tiles: Dict[Tuple[int, int, int], TileManager.TileRef] = {}
        self.stats = ServerStats(name=name)

    def url(self, base_url: str) -> str:
        """
        URL template for tiles for this layer (routed through jupyter-server-proxy)

        Base URL should be the notebook base URL as described at
        https://jupyter-server-proxy.readthedocs.io/en/latest/arbitrary-ports-hosts.html
        """
        return f"{base_url}/proxy/{PORT}/{self.token}/{{z}}/{{y}}/{{x}}.png"

    def update_viewport(self, tiles: Set[Tuple[int, int, int]]) -> None:
        """
        Update the set of currently-visible tiles.

        Tiles that are not already computing will be started.
        Tiles that were computing and had no other requests waiting for them will be cancelled.

        Safe to call from other threads.
        """
        viewport = set(tiles)
        current = self.tiles.keys()
        for release_xyz in current - viewport:
            release_ref = self.tiles[release_xyz]
            self.loop.call_soon_threadsafe(release_ref.decref)

        for new_xyz in viewport - current:
            self.loop.call_soon_threadsafe(self.fire_and_forget, new_xyz)

    async def fetch(self, x: int, y: int, z: int) -> bytes:
        """
        Wait for a tile to compute.

        If a computation is already running for this tile, it will be reused.
        Otherwise, a new computation is started.
        """
        xyz = (x, y, z)
        tile_ref = self.submit(xyz)

        with tile_ref:
            self.stats.requested += 1
            try:
                return await tile_ref.task
            finally:
                self.stats.requested -= 1

    def stop(self) -> None:
        "Stop all active computations, even if others hold references to them"
        for ref in self.tiles.values():
            self.loop.call_soon_threadsafe(ref.cancel)
        # NOTE: no need for `self.tiles.clear()`; the `done_callback` on `ref` should do that thread-safely

    def submit(self, xyz: Tuple[int, int, int]) -> TileManager.TileRef:
        "Get the TileRef for a tile, starting a new computation if necessary"
        tile_ref = self.tiles.get(xyz, None)
        if tile_ref is not None:
            return tile_ref

        task = asyncio.create_task(self._compute_tile(*xyz))
        task.add_done_callback(self._finalize)

        tile_ref = self.tiles[xyz] = self.TileRef(
            task, lambda: self.tiles.pop(xyz, None)
        )
        self.stats.computing += 1
        return tile_ref

    def fire_and_forget(self, xyz: Tuple[int, int, int]) -> None:
        "Submit, then immediately `incref()`"
        self.submit(xyz).incref()

    def _finalize(self, future: asyncio.Future) -> None:
        "Internal: update stats when a computation task finishes"
        self.stats.computing -= 1

        try:
            exc = future.exception()
        except asyncio.CancelledError as e:
            exc = e

        if exc is None:
            self.stats.completed += 1
        elif isinstance(exc, asyncio.CancelledError):
            self.stats.cancelled += 1
        else:
            self.stats.errored += 1

    async def _compute_tile(self, x: int, y: int, z: int) -> bytes:
        "Send an XYZ tile to be computed by the distributed client, and wait for it."
        disp = self.disp
        client = distributed.get_client()
        # TODO assert the client's loop is the same as our current event loop.
        # If not... tell the server to shut down and restart on the new event loop?
        # (could also do this within a watch loop in `_launch_server`.)

        tile = xyztile_of_array(
            disp.arr, disp.tilesize, x, y, z, interpolation=disp.interpolation
        )
        if tile is None:
            return empty_tile(disp.tilesize, disp.checkerboard)

        delayed_png = delayed_arr_to_png(
            tile.data,
            range=disp.range,
            colormap=disp.colormap,
            checkerboard=disp.checkerboard,
        )

        future = client.compute(delayed_png, sync=False)
        future = cast(distributed.Future, future)

        awaitable = future if client.asynchronous else future._result()
        # ^ sneak into the async api if the client isn't set up to be async.
        # this _should_ be okay, since we're running within the client's own event loop.
        awaitable = cast(Awaitable[bytes], awaitable)
        try:
            return await awaitable
        except Exception:
            # Typically an `asyncio.CancelledError` from the request being cancelled,
            # but no matter what it is, we want to ensure we drop the distributed Future.

            # There may be some state issues in `distributed.Client` around handling `CancelledError`s;
            # occasionally there are still references to them held within frames/tracebacks (this may also
            # have to do with aiohttp's error handing, our ours, or ipython).
            # So we very aggressively try to cancel and release references to the future.
            try:
                await future.cancel(asynchronous=True)
            except asyncio.CancelledError:
                # Unlikely, but anytime we `await`, we could get cancelled.
                # We're already cleaning up, so ignore cancellation here.
                pass

            future.release()
            # ^ We can still hold a reference to a Future after it's cancelled?
            # And occasionally data will stay in distributed memory in that case?

            raise

    def __del__(self):
        self.stop()


TOKEN_TO_TILE_MANAGER: Dict[str, TileManager] = {}


def register(
    arr: xr.DataArray,
    map: ipyleaflet.Map,
    layer: ipyleaflet.TileLayer,
    range: Optional[Range] = None,
    colormap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    checkerboard: bool = True,
    tilesize: int = 256,
    interpolation: Literal["linear", "nearest"] = "linear",
):
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
    loop = ensure_server()

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
            f"Array must only have the dimensions 'x', 'y', and optionally 'band', not {arr.dims!r}"
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
    manager = TOKEN_TO_TILE_MANAGER.get(token, None)
    if manager is None:
        manager = TOKEN_TO_TILE_MANAGER[token] = TileManager(
            disp, token, layer.name, loop
        )

    MapObserver.set_up_for_map(map, layer, manager)
    server_stats.children = [
        manager.stats for manager in TOKEN_TO_TILE_MANAGER.values()
    ]

    # TODO some way to unregister (this is hard because we may have modified `arr` in `register`)


class MapObserver:
    @classmethod
    def set_up_for_map(
        cls, map: ipyleaflet.Map, layer: ipyleaflet.TileLayer, manager: TileManager
    ) -> None:
        "Create a new or set up an existing `MapObserver` for an `ipyleaflet.Map`"
        if base_url := cls.base_url_from_window_location(map.window_url):
            new_url = manager.url(base_url)
            if new_url != layer.url:
                # TODO cancel the manager for the old URL
                layer.url = new_url
                layer.redraw()

        try:
            notifiers = map._trait_notifiers["window_url"]["change"]
        except KeyError:
            pass
        else:
            for map_observer in notifiers:
                if isinstance(map_observer, cls):
                    if manager not in map_observer.managers:
                        map_observer.managers.append(manager)
                        map_observer.layers.append(layer)
                        map_observer.bounds_changed(dict(map=map, bounds=map.bounds))
                    return

        map_observer = cls()
        map_observer.managers.append(manager)
        map_observer.layers.append(layer)

        map.observe(map_observer, names="window_url")
        map.observe(map_observer, names="bounds")
        map.observe(map_observer, names="layers")

        # Trigger initial viewport to load
        map_observer.bounds_changed(dict(map=map, bounds=map.bounds))

    def __init__(self) -> None:
        self.managers: List[TileManager] = []
        self.layers: List[ipyleaflet.TileLayer] = []

    def __call__(self, change: dict):
        try:
            name = change["name"]
        except KeyError:
            return
        if name == "window_url":
            self.window_url_changed(change)
        elif name == "bounds":
            self.bounds_changed(change)
        elif name == "layers":
            self.layers_changed(change)

    def window_url_changed(self, change: dict):
        try:
            url = change["new"]
        except KeyError:
            return

        base_url = self.base_url_from_window_location(url)
        if base_url:
            for lyr, manager in zip(self.layers, self.managers):
                lyr.url = manager.url(base_url)
                lyr.redraw()

    def bounds_changed(self, change: dict):
        "When the map moves, update all our managers' viewports"
        try:
            map: ipyleaflet.Map = change["owner"]
            bounds: Tuple[Tuple[float, float], Tuple[float, float]] = change["new"]
        except KeyError:
            return

        (south, west), (north, east) = bounds
        if south == west == north == east == 0:
            return
        tiles = set(
            (t.x, t.y, t.z)  # unnecessary copy, makes typechecker happy
            for t in mercantile.tiles(west, south, east, north, int(map.zoom))
        )

        for manager in self.managers:
            manager.update_viewport(tiles)

    def layers_changed(self, change: dict):
        "If one of our layers is removed from the map, forget about it, and stop its manager"
        # NOTE: this assumes a layer is only on at most one map at a time
        try:
            new = change["new"]
        except KeyError:
            return

        drop_is = []
        for i, (lyr, manager) in enumerate(zip(self.layers, self.managers)):
            if lyr not in new:
                manager.stop()
                drop_is.append(i)
                TOKEN_TO_TILE_MANAGER.pop(manager.token, None)

        for i in drop_is:
            del self.layers[i]
            del self.managers[i]

        if drop_is:
            server_stats.children = [
                manager.stats for manager in TOKEN_TO_TILE_MANAGER.values()
            ]

    @staticmethod
    def base_url_from_window_location(url: str) -> Optional[str]:
        if not url:
            return None
        if path_re := re.match(r"(^.+)\/(?:lab|notebook|voila)", url):
            return path_re.group(1)
        return None


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

        If None (default), ``arr.name`` is used as the name. If a layer with this name already exists,
        it will be replaced.
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
    if name is None:
        name = str(arr.name)
    for lyr in map.layers:
        if lyr.name == name:
            break
    else:
        lyr = ipyleaflet.TileLayer(name=name, url="")
        map.add_layer(lyr)

    register(
        arr,
        map=map,
        layer=lyr,
        range=range,
        colormap=colormap,
        checkerboard=checkerboard,
        interpolation=interpolation,
    )
    return lyr


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


def ensure_server() -> asyncio.AbstractEventLoop:
    "Ensure the webserver is running, and return the event loop for it"
    if ensure_server._loop is None:
        ensure_server._loop = _launch_server()
    return ensure_server._loop


ensure_server._loop = None


def _launch_server() -> asyncio.AbstractEventLoop:
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

    return client.loop.asyncio_loop  # type: ignore
    # Cannot access member "asyncio_loop" for type "IOLoop"
    # Member "asyncio_loop" is unknownPylancereportGeneralTypeIssues
    # This is entirely correct---we'll fail if the Tornado loop isn't aio-backed


@routes.get("/{hash}/{z}/{y}/{x}.png")
async def handler(request: web.Request) -> web.Response:
    "Handle a HTTP GET request for an XYZ tile"
    hash = request.match_info["hash"]
    try:
        manager = TOKEN_TO_TILE_MANAGER[hash]
    except KeyError:
        raise web.HTTPNotFound(
            reason=f"{hash} not found. Have {list(TOKEN_TO_TILE_MANAGER)}."
        ) from None

    try:
        z = int(request.match_info["z"])
        y = int(request.match_info["y"])
        x = int(request.match_info["x"])
    except ValueError:
        raise web.HTTPBadRequest(reason="z, y, and z parameters must be ints")

    png = await manager.fetch(x, y, z)
    return web.Response(
        body=png,
        status=200,
        content_type="image/png",
        headers={"Access-Control-Allow-Origin": "*"},
    )


def xyztile_of_array(
    arr: xr.DataArray,
    tilesize: int,
    x: int,
    y: int,
    z: int,
    interpolation: Literal["linear", "nearest"],
) -> Optional[xr.DataArray]:
    "Slice an XYZ tile out of a DataArray. Returns None if the tile does not overlap."
    bounds = mercantile.xy_bounds(mercantile.Tile(x, y, z))

    if not geom_utils.bounds_overlap(
        bounds, geom_utils.array_bounds(arr, to_epsg=3857)
    ):
        return None

    minx, miny, maxx, maxy = bounds
    # FIXME: `reproject_array` is really, really slow for large arrays
    # because of all the dask-graph-munging. Having a blocking, GIL-bound
    # function within an async handler like this also means we're basically
    # sending requests out serially per tile
    tile = geom_utils.reproject_array(
        arr,
        RasterSpec(
            epsg=3857,
            bounds=bounds,
            resolutions_xy=(
                (maxx - minx) / tilesize,
                (maxy - miny) / tilesize,
            ),
        ),
        interpolation=interpolation,
    )
    assert tile.shape[1:] == (
        tilesize,
        tilesize,
    ), f"Wrong shape after interpolation: {tile.shape}"
    return tile


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
