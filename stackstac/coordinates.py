from __future__ import annotations

from typing import Any, Collection, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from stackstac.coordinates_utils import (
    Coordinates,
    items_to_coords,
    unnested_items,
    unpacked_per_band_asset_fields,
)

from .raster_spec import RasterSpec
from .stac_types import ItemSequence

ASSET_TABLE_DT = np.dtype(
    [("url", object), ("bounds", "float64", 4), ("scale_offset", "float64", 2)]
)

# Asset fields which are a list with one item per band in the asset.
# For one-band assets, they should be a list of length 1.
# We'll unpack those 1-length lists, so the subfields can be flattened into
# top-level coordinates.
# This is how we get `eo:bands_common_name` or `raster:bands_scale` coordinates.
PER_BAND_ASSET_FIELDS = {
    "eo:bands",
    "raster:bands",
}


def to_coords(
    items: ItemSequence,
    asset_ids: List[str],
    spec: RasterSpec,
    xy_coords: Literal["center", "topleft", False] = "topleft",
    properties: bool = True,
    band_coords: bool = True,
) -> Tuple[Coordinates, List[str]]:
    times = pd.to_datetime(
        [item["properties"]["datetime"] for item in items],
        infer_datetime_format=True,
        errors="coerce",
    )
    if times.tz is not None:
        # xarray can't handle tz-aware DatetimeIndexes, so we convert to UTC and drop the timezone
        # https://github.com/pydata/xarray/issues/3291.
        # The `tz is None` case is typically a manifestation of https://github.com/pandas-dev/pandas/issues/41047.
        # Since all STAC timestamps should be UTC (https://github.com/radiantearth/stac-spec/issues/1095),
        # we feel safe assuming that any tz-naive datetimes are already in UTC.
        times = times.tz_convert(None)

    dims = ["time", "band", "y", "x"]
    coords = {
        "time": times,
        "id": xr.Variable("time", [item["id"] for item in items]),
        "band": asset_ids,
    }

    if xy_coords is not False:
        if xy_coords == "center":
            pixel_center = True
        elif xy_coords == "topleft":
            pixel_center = False
        else:
            raise ValueError(
                f"xy_coords must be 'center', 'topleft', or False, not {xy_coords!r}"
            )

        transform = spec.transform
        # We generate the transform ourselves in `RasterSpec`, and it's always constructed to be rectilinear.
        # Someday, this should not always be the case, in order to support non-rectilinear data without warping.
        assert (
            transform.is_rectilinear
        ), f"Non-rectilinear transform generated: {transform}"
        minx, miny, maxx, maxy = spec.bounds
        xres, yres = spec.resolutions_xy

        if pixel_center:
            half_xpixel, half_ypixel = xres / 2, yres / 2
            minx, miny, maxx, maxy = (
                minx + half_xpixel,
                miny - half_ypixel,
                maxx + half_xpixel,
                maxy - half_ypixel,
            )

        height, width = spec.shape
        # Wish pandas had an RangeIndex that supported floats...
        # https://github.com/pandas-dev/pandas/issues/46484
        xs = pd.Index(np.linspace(minx, maxx, width, endpoint=False), dtype="float64")
        ys = pd.Index(np.linspace(maxy, miny, height, endpoint=False), dtype="float64")

        coords["x"] = xs
        coords["y"] = ys

    if properties:
        assert properties is True, (
            "Passing specific properties is no longer supported. "
            "The `properties` argument must only be True or False. "
            "If you have a use case for this, please open an issue."
        )
        coords.update(items_to_property_coords(items))

    if band_coords:
        coords.update(items_to_band_coords(items, asset_ids))

    # Add `epsg` last in case it's also a field in properties; our data model assumes it's a coordinate
    coords["epsg"] = spec.epsg

    return coords, dims


def items_to_property_coords(
    items: ItemSequence,
    skip_fields: Collection[str] = frozenset(["datetime", "id"]),
) -> Coordinates:
    return items_to_coords(
        (
            ((i,), k, v)
            for i, item in enumerate(items)
            # TODO: should we unnest properties?
            for k, v in item["properties"].items()
            if k not in skip_fields
        ),
        shape=(len(items),),
        dims=("time",),
    )


def items_to_band_coords(
    items: ItemSequence,
    asset_ids: List[str],
    skip_fields: Collection[str] = frozenset(["href", "type"]),
) -> Coordinates:
    def fields_values_generator():
        for ii, item in enumerate(items):
            for ai, id in enumerate(asset_ids):
                try:
                    asset = item["assets"][id]
                except KeyError:
                    continue

                for field, value in unnested_items(
                    unpacked_per_band_asset_fields(asset.items(), PER_BAND_ASSET_FIELDS)
                ):
                    field = rename_some_band_fields(field)
                    if field not in skip_fields:
                        yield (ii, ai), field, value

    return items_to_coords(
        fields_values_generator(),
        shape=(len(items), len(asset_ids)),
        dims=("time", "band"),
    )


def rename_some_band_fields(field: str) -> str:
    """
    Apply renamings to band fields for "convenience".

    This is just for backwards compatibility.
    These renamings should probably be removed for simplicity and consistency.
    """
    if field == "sar:polarizations":
        return "polarization"
    return field.removeprefix("eo:bands_")


def spec_to_attrs(spec: RasterSpec) -> Dict[str, Any]:
    attrs = {"spec": spec, "crs": f"epsg:{spec.epsg}", "transform": spec.transform}

    resolutions = spec.resolutions_xy
    if resolutions[0] == resolutions[1]:
        attrs["resolution"] = resolutions[0]
    else:
        attrs["resolution_xy"] = resolutions
    return attrs
