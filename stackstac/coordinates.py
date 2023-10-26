from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from stackstac.coordinates_utils import (
    deduplicate_axes,
    descalar_obj_array,
    scalar_sequence,
    unnest_dicts,
    unnested_items,
    unpack_per_band_asset_fields,
    unpacked_per_band_asset_fields,
)

from . import accumulate_metadata
from .raster_spec import RasterSpec
from .stac_types import ItemSequence

ASSET_TABLE_DT = np.dtype(
    [("url", object), ("bounds", "float64", 4), ("scale_offset", "float64", 2)]
)
Coordinates = Mapping[str, Union[pd.Index, np.ndarray, xr.Variable, list]]

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
    properties: Union[bool, str, Sequence[str]] = True,
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
        coords.update(items_to_property_coords(items, properties))

    if band_coords:
        coords.update(items_to_band_coords2(items, asset_ids))

    # Add `epsg` last in case it's also a field in properties; our data model assumes it's a coordinate
    coords["epsg"] = spec.epsg

    return coords, dims


def items_to_property_coords(
    items: ItemSequence, properties: Union[str, Sequence[str], Literal[True]]
) -> Coordinates:
    return accumulate_metadata.metadata_to_coords(
        (item["properties"] for item in items),
        "time",
        fields=properties,
        skip_fields={"datetime"},
        # skip_fields={"datetime", "providers"},
    )


def items_to_band_coords(
    items: ItemSequence,
    asset_ids: List[str],
) -> Coordinates:
    flattened_metadata_by_asset = [
        accumulate_metadata.accumulate_metadata(
            (item["assets"].get(asset_id, {}) for item in items),
            # skip_fields={"href", "type", "roles"},
        )
        for asset_id in asset_ids
    ]

    eo_by_asset = []
    for meta in flattened_metadata_by_asset:
        # NOTE: we look for `eo:bands` in each Asset's metadata, not as an Item-level list.
        # This only became available in STAC 1.0.0-beta.1, so we'll fail on older collections.
        # See https://github.com/radiantearth/stac-spec/tree/master/extensions/eo#item-fields
        eo = meta.pop("eo:bands", {})
        if isinstance(eo, list):
            eo = eo[0] if len(eo) == 1 else {}
            # ^ `eo:bands` should be a list when present, but >1 item means it's probably a multi-band asset,
            # which we can't currently handle, so we ignore it. we don't error here, because
            # as long as you don't actually _use_ that asset, everything will be fine. we could
            # warn, but that would probably just get annoying.
        eo_by_asset.append(eo)
        try:
            meta["polarization"] = meta.pop("sar:polarizations")
        except KeyError:
            pass

    coords = accumulate_metadata.metadata_to_coords(
        flattened_metadata_by_asset,
        "band",
        # skip_fields={"href"},
        # skip_fields={"href", "title", "description", "type", "roles"},
    )
    if any(eo_by_asset):
        coords.update(
            accumulate_metadata.metadata_to_coords(
                eo_by_asset,
                "band",
                fields=["common_name", "center_wavelength", "full_width_half_max"],
            )
        )
    return coords


# TODO
def items_to_property_coords2(
    items: ItemSequence, properties: Union[str, Sequence[str], Literal[True]]
) -> Coordinates:
    # How to factor out into shared code?
    # TODO use `properties` arg

    unnested_props = [unnest_dicts(item["properties"]) for item in items]
    all_fields = set().union(*(p.keys() for p in unnested_props))

    coords_arrs = {
        field: descalar_obj_array(
            np.array([scalar_sequence(prop.get(field)) for prop in unnested_props])
        )
        for field in all_fields
    }

    deduped = {field: deduplicate_axes(arr) for field, arr in coords_arrs.items()}

    return {
        field: xr.Variable(["time"], arr).squeeze() for field, arr in deduped.items()
    }


def items_to_band_coords2(
    items: ItemSequence,
    asset_ids: List[str],
) -> Coordinates:

    unnested_assets = [
        {
            k: unnest_dicts(unpack_per_band_asset_fields(v, PER_BAND_ASSET_FIELDS))
            for k, v in item["assets"].items()
            if k in asset_ids
        }
        for item in items
    ]
    all_fields = sorted(
        set().union(*(asset.keys() for ia in unnested_assets for asset in ia.values()))
    )

    # Building up arrays like:
    # {
    #     "field 0": np.array([
    #         [  # item 0
    #             value_for_asset_0, value_for_asset_1, ...
    #         ],
    #         [  # item 1
    #             value_for_asset_0, value_for_asset_1, ...
    #         ],
    #         ...
    #     ])
    #     ...
    # }
    coords_arrs = {
        field: descalar_obj_array(
            np.array(
                [
                    [
                        scalar_sequence(assets.get(id, {}).get(field))
                        for id in asset_ids
                        # desequence because if the field contains a list, we want to
                        # treat that as though it's a scalar value.
                    ]
                    for assets in unnested_assets
                ]
            )
        )
        for field in all_fields
    }

    # # Maybe a way to improve locality and not iterate over all items many times.
    # # TODO: benchmark
    # coords_lists = collections.defaultdict(list)
    # for assets in unnested_assets:
    #     values = collections.defaultdict(list)
    #     for id in asset_ids:
    #         asset = assets.get(id, {})
    #         for field in all_fields:
    #             values[field].append(desequence(asset.get(field)))
    #     for k, v in values.items():
    #         coords_lists[k].append(v)

    deduped = {field: deduplicate_axes(arr) for field, arr in coords_arrs.items()}

    return {
        field: xr.Variable(["time", "band"], arr).squeeze()
        for field, arr in deduped.items()
    }


def items_to_band_coords_simple(
    items: ItemSequence,
    asset_ids: List[str],
) -> Coordinates:
    # Interestingly this is slightly faster in benchmarking than `items_to_band_coords2`
    unnested_assets = [
        {
            k: unnest_dicts(unpack_per_band_asset_fields(v, PER_BAND_ASSET_FIELDS))
            for k, v in item["assets"].items()
            if k in asset_ids
        }
        for item in items
    ]
    all_fields = sorted(
        set().union(*(asset.keys() for ia in unnested_assets for asset in ia.values()))
    )

    asset_arr = [
        [[ia.get(aid, {}).get(f) for f in all_fields] for aid in asset_ids]
        for ia in unnested_assets
    ]
    da = xr.DataArray(
        asset_arr,
        coords={
            "band": asset_ids,
            "field": all_fields,
        },
        dims=["time", "band", "field"],
    )
    ds = da.to_dataset("field")
    ds = ds.map(
        # TODO better way?
        lambda da: xr.DataArray(deduplicate_axes(da.data), dims=da.dims).squeeze()
    )
    return ds.variables


def items_to_band_coords_locality(
    items: ItemSequence,
    asset_ids: List[str],
) -> Coordinates:
    # {field:
    #   [
    #       [v_asset_0, v_asset_1, ...],  # item 0
    #       [v_asset_0, v_asset_1, ...],  # item 1
    #   ]
    # }
    fields = {}
    for ii, item in enumerate(items):
        for ai, id in enumerate(asset_ids):
            try:
                asset = item["assets"][id]
            except KeyError:
                continue

            for field, value in unnested_items(
                unpacked_per_band_asset_fields(asset.items(), PER_BAND_ASSET_FIELDS)
            ):
                try:
                    values = fields[field]
                except KeyError:
                    values = fields[field] = np.empty(
                        (len(items), len(asset_ids)), dtype=object
                    )

                values[ii, ai] = value

    # TODO un-object-ify each field
    fields = {field: np.array(arr.tolist()) for field, arr in fields.items()}
    deduped = {field: deduplicate_axes(arr) for field, arr in fields.items()}

    return {
        field: xr.Variable(["time", "band"], arr).squeeze()
        for field, arr in deduped.items()
    }


def spec_to_attrs(spec: RasterSpec) -> Dict[str, Any]:
    attrs = {"spec": spec, "crs": f"epsg:{spec.epsg}", "transform": spec.transform}

    resolutions = spec.resolutions_xy
    if resolutions[0] == resolutions[1]:
        attrs["resolution"] = resolutions[0]
    else:
        attrs["resolution_xy"] = resolutions
    return attrs
