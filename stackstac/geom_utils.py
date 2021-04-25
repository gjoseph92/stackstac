import functools
import math
from typing import Literal, Optional, Union
import pandas as pd

import pyproj
import affine
import xarray as xr
import numpy as np

from .raster_spec import Bbox, RasterSpec, Resolutions

NO_DEFAULT = "__no_default__"
NO_DEFAULT_LITERAL = Literal["__no_default__"]
cached_transformer = functools.lru_cache(pyproj.Transformer.from_crs)


def bounds_from_affine(
    af: affine.Affine, ysize: int, xsize: int, from_epsg: int, to_epsg: int
) -> Bbox:
    ul_x, ul_y = af * (0, 0)
    ll_x, ll_y = af * (0, ysize)
    lr_x, lr_y = af * (xsize, ysize)
    ur_x, ur_y = af * (0, xsize)

    xs = [ul_x, ll_x, lr_x, ur_x]
    ys = [ul_y, ll_y, lr_y, ur_y]

    if from_epsg != to_epsg:
        transformer = cached_transformer(
            from_epsg, to_epsg, skip_equivalent=True, always_xy=True
        )
        # TODO handle error
        xs_proj, ys_proj = transformer.transform(xs, ys, errcheck=True)
    else:
        xs_proj = xs
        ys_proj = ys

    return min(xs_proj), min(ys_proj), max(xs_proj), max(ys_proj)


# TODO: use `rio.warp.transform_bounds` instead, for densification? or add our own densification?
# ours is likely faster due to transformer caching.
def reproject_bounds(bounds: Bbox, from_epsg: int, to_epsg: int) -> Bbox:
    if from_epsg == to_epsg:
        return bounds

    minx, miny, maxx, maxy = bounds
    # generate the four corners (in CCW order, starting at upper-left)
    # read this in pairs, downward by column
    xs = [minx, minx, maxx, maxx]
    ys = [maxy, miny, miny, maxy]
    transformer = cached_transformer(
        from_epsg, to_epsg, skip_equivalent=True, always_xy=True
    )
    xs_proj, ys_proj = transformer.transform(xs, ys, errcheck=True)  # TODO handle error
    return min(xs_proj), min(ys_proj), max(xs_proj), max(ys_proj)


def union_bounds(*bounds: Bbox) -> Bbox:
    pairs = zip(*bounds)
    return (
        min(next(pairs)),
        min(next(pairs)),
        max(next(pairs)),
        max(next(pairs)),
    )


def bounds_overlap(*bounds: Bbox) -> bool:
    min_xs, min_ys, max_xs, max_ys = zip(*bounds)
    return max(min_xs) < min(max_xs) and max(min_ys) < min(max_ys)


def snapped_bounds(bounds: Bbox, resolutions_xy: Resolutions) -> Bbox:
    minx, miny, maxx, maxy = bounds
    xres, yres = resolutions_xy

    minx = math.floor(minx / xres) * xres
    maxx = math.ceil(maxx / xres) * xres
    miny = math.floor(miny / yres) * yres
    maxy = math.ceil(maxy / yres) * yres

    return (minx, miny, maxx, maxy)


def array_epsg(
    arr: xr.DataArray, default: Union[int, NO_DEFAULT_LITERAL] = NO_DEFAULT
) -> int:
    # TODO look at `crs` in attrs; more compatibility with rioxarray data model
    try:
        epsg = arr.epsg
    except AttributeError:
        if default != NO_DEFAULT:
            return default
    else:
        return epsg.item()

    # NOTE: raise out here for shorter traceback
    raise ValueError(
        f"DataArray {arr.name!r} does not have the `epsg` coordinate set, "
        "so we don't know what coordinate reference system it's in.\n"
        "Please set it using `arr.assign_coords(epsg=<EPSG code as int>)` "
        "(note that this returns a new object)."
    )


def array_bounds(arr: xr.DataArray, to_epsg: Optional[int] = None) -> Bbox:
    try:
        bounds = arr.spec.bounds
    except AttributeError:
        try:
            x: pd.Index = arr.indexes["x"]
            y: pd.Index = arr.indexes["y"]
        except KeyError:
            raise ValueError(
                "Cannot determine bounds of the array, since it has no `x` and `y` coordinates, nor a `spec` attribute."
            ) from None

        # Check for monotonicity. There should be no way to produce non-monotonic indexes from stackstac,
        # but it's still good to check our assumptions, just in case you've messed with the indices in a strange way.
        # Note that `is_monotonic_*` is cached on `pd.Index`, so this check should be cheap.
        for index in (x, y):
            if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
                raise ValueError(
                    f"Cannot determine bounds of the DataArray, since the {index.name} coordinate is non-monotonic. "
                    f"Consider `arr.sortby({index.name!r})`, or add a `stackstac.RasterSpec` as the `spec` attribute."
                )

        # Bounds go from the top _left_ pixel corner to the top _right_ pixel corner, so we need to account
        # for that one extra pixel's worth of width/height.
        # Assume the interval between xs/ys is constant to calculate this.
        xstep = x[1] - x[0]
        ystep = y[1] - y[0]
        xs = (x[0], x[-1] + xstep)
        ys = (y[0], y[-1] + ystep)
        # Don't assume coordinates are north-up, east-right.
        bounds = (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    if to_epsg is None:
        return bounds

    return reproject_bounds(bounds, array_epsg(arr), to_epsg)


def reproject_array(
    arr: xr.DataArray,
    spec: RasterSpec,
    method: Union[Literal["linear"], Literal["nearest"]] = "linear",
) -> xr.DataArray:
    # TODO this scipy/`interp`-based approach still isn't block-parallel
    # (seems like xarray just rechunks to fuse all the spatial chunks first),
    # so this both won't scale, and can be crazy slow in dask graph construction
    # (and the rechunk probably eliminates any hope of sending an HLG to the scheduler).

    from_epsg = array_epsg(arr)
    if (
        from_epsg == spec.epsg
        and array_bounds(arr) == spec.bounds
        and arr.shape[:-2] == spec.shape
    ):
        return arr

    as_bool = False
    if arr.dtype.kind == "b":
        # `interp` can't handle boolean arrays
        arr = arr.astype("uint8")
        as_bool = True

    # TODO fastpath when there's no overlap? (graph shouldn't have any IO in it.)
    # Or does that already happen?

    # TODO pixel centers vs topleft? `spec` assumes topleft;
    # if the x/y coords on the array are center, this will be a half
    # pixel off.
    minx, miny, maxx, maxy = spec.bounds
    height, width = spec.shape

    x = np.linspace(minx, maxx, width, endpoint=False)
    y = np.linspace(maxy, miny, height, endpoint=False)

    if from_epsg == spec.epsg:
        # Simpler case: just interpolate within the same CRS
        result = arr.interp(x=x, y=y, method=method)
        return result.astype(bool) if as_bool else result

    # Different CRSs: need to do a 2D interpolation.
    # We do this by, for each point in the output grid, generating
    # the coordinates in the _input_ CRS that correspond to that point.

    reverse_transformer = cached_transformer(
        spec.epsg, from_epsg, skip_equivalent=True, always_xy=True
    )

    xs, ys = np.meshgrid(x, y, copy=False)
    src_xs, src_ys = reverse_transformer.transform(xs, ys, errcheck=True)

    xs_indexer = xr.DataArray(src_xs, dims=["y", "x"], coords=dict(y=y, x=x))
    ys_indexer = xr.DataArray(src_ys, dims=["y", "x"], coords=dict(y=y, x=x))

    # TODO maybe just drop old dims instead?
    old_xdim = f"x_{from_epsg}"
    old_ydim = f"y_{from_epsg}"

    result = arr.rename(x=old_xdim, y=old_ydim).interp(
        {old_xdim: xs_indexer, old_ydim: ys_indexer}, method=method
    )
    return result.astype(bool) if as_bool else result
