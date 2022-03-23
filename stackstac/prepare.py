from __future__ import annotations

import collections
from typing import (
    AbstractSet,
    Literal,
    NamedTuple,
    Sequence,
    Optional,
    Set,
    Union,
    Tuple,
    List,
    Dict,
    Any,
    cast,
)
import warnings


import affine
import numpy as np
import pandas as pd
import xarray as xr

from .raster_spec import IntFloat, Bbox, Resolutions, RasterSpec
from .stac_types import ItemSequence
from . import accumulate_metadata, geom_utils

ASSET_TABLE_DT = np.dtype([("url", object), ("bounds", "float64", 4)])


class Mimetype(NamedTuple):
    type: str
    subtype: Optional[str]
    parameters: Set[str]

    @classmethod
    def from_str(cls, mimetype: str) -> Mimetype:
        parts = [p.strip() for p in mimetype.split(";")]
        type, *subtype = parts[0].split("/")
        if len(subtype) == 0:
            subtype = None
        else:
            assert len(subtype) == 1
            subtype = subtype[0]
        return cls(type, subtype, set(parts[1:]))

    def is_valid_for(self, target: Mimetype) -> bool:
        return (
            self.type == target.type
            and (self.subtype == target.subtype if target.subtype else True)
            and target.parameters.issubset(self.parameters)
        )


def prepare_items(
    items: ItemSequence,
    assets: Optional[Union[List[str], AbstractSet[str]]] = frozenset(
        ["image/tiff", "image/x.geotiff", "image/vnd.stac.geotiff", "image/jp2"]
    ),
    epsg: Optional[int] = None,
    resolution: Optional[Union[IntFloat, Resolutions]] = None,
    bounds: Optional[Bbox] = None,
    bounds_latlon: Optional[Bbox] = None,
    snap_bounds: bool = True,
) -> Tuple[np.ndarray, RasterSpec, List[str], ItemSequence]:

    if bounds is not None and bounds_latlon is not None:
        raise ValueError(
            f"Cannot give both `bounds` {bounds} and `bounds_latlon` {bounds_latlon}."
        )

    out_epsg = epsg
    out_bounds = bounds
    if resolution is not None and not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    out_resolutions_xy = resolution

    if assets is None:
        # use the keys from the item with the most assets (do this rather than a `set()` to preserve order)
        asset_ids = list(max((item["assets"] for item in items), key=len))
    elif isinstance(assets, AbstractSet):
        allowed_mimetypes = [Mimetype.from_str(t) for t in assets]
        type_strs_by_id: dict[str, set[Optional[str]]] = collections.defaultdict(set)
        for item in items:
            for asset_id, asset in item["assets"].items():
                type_strs_by_id[asset_id].add(asset.get("type"))

        mimetypes_by_id = {
            id: [Mimetype.from_str(t) for t in types if t is not None]
            for id, types in type_strs_by_id.items()
        }

        ids_without_type = [
            id for id, mimetypes in mimetypes_by_id.items() if not mimetypes
        ]
        if ids_without_type:
            warnings.warn(
                f"You're filtering for assets that match the mimetype(s) {assets}, but since {len(ids_without_type)} "
                f"(out of {len(type_strs_by_id)}) asset(s) have no `type` specified on any item, those will be "
                "dropped. Consider passing a list of asset IDs instead to the `assets=` parameter.\n"
                f"Assets with no type: {ids_without_type}",
            )

        asset_ids = [
            asset_id
            for asset_id, asset_mimetypes in mimetypes_by_id.items()
            if asset_mimetypes
            and all(
                any(
                    asset_mt.is_valid_for(allowed_mt)
                    for allowed_mt in allowed_mimetypes
                )
                for asset_mt in asset_mimetypes
            )
        ]
    else:
        asset_ids = assets

    asset_table = np.full((len(items), len(asset_ids)), None, dtype=ASSET_TABLE_DT)

    # TODO support item-assets https://github.com/radiantearth/stac-spec/tree/master/extensions/item-assets

    if len(items) == 0:
        raise ValueError("No items")
    if len(asset_ids) == 0:
        raise ValueError("Zero asset IDs requested")

    for item_i, item in enumerate(items):
        item_epsg = item["properties"].get("proj:epsg")
        item_bbox = item["properties"].get("proj:bbox")
        item_shape = item["properties"].get("proj:shape")
        item_transform = item["properties"].get("proj:transform")

        item_bbox_proj = None
        for asset_i, id in enumerate(asset_ids):
            try:
                asset = item["assets"][id]
            except KeyError:
                continue

            asset_epsg = asset.get("proj:epsg", item_epsg)
            asset_bbox = asset.get("proj:bbox", item_bbox)
            asset_shape = asset.get("proj:shape", item_shape)
            asset_transform = asset.get("proj:transform", item_transform)
            asset_affine = None

            # Auto-compute CRS
            if epsg is None:
                if asset_epsg is None:
                    raise ValueError(
                        f"Cannot pick a common CRS, since asset {id!r} of item "
                        f"{item_i} {item['id']!r} does not have one.\n\n"
                        "Please specify a CRS with the `epsg=` argument."
                    )
                if out_epsg is None:
                    out_epsg = asset_epsg
                elif out_epsg != asset_epsg:
                    raise ValueError(
                        f"Cannot pick a common CRS, since assets have multiple CRSs: asset {id!r} of item "
                        f"{item_i} {item['id']!r} is in EPSG:{asset_epsg}, "
                        f"but assets before it were in EPSG:{out_epsg}.\n\n"
                        "Please specify a CRS with the `epsg=` argument."
                    )

            out_epsg = cast(int, out_epsg)
            # ^ because if it was None initially, and we didn't error out in the above check, it's now always set

            if bounds_latlon is not None and out_bounds is None:
                out_bounds = bounds = geom_utils.reproject_bounds(
                    bounds_latlon, 4326, out_epsg
                )
                # NOTE: we wait to reproject until now, so we can use the inferred CRS

            # Compute the asset's bbox in the output CRS.
            # Use `proj:bbox` if there is one, otherwise compute it from `proj:shape` and `proj:transform`.
            # And if none of that info exists, then estimate it from the item's lat-lon geometry/bbox.

            # If there's a `proj:bbox` (and it's actually for the asset, not the item overall),
            # just reproject that and use it
            if (
                asset_bbox is not None
                and asset_epsg is not None
                and asset_transform == item_transform
                and asset_shape == item_shape
                # TODO this still misses the case where the asset overrides bbox, but not transform/shape.
                # At that point would need to significantly restructure the code
                # to express this complex of prioritization:
                # bbox from asset, then transform from asset, then bbox from item,
                # then transform from item, then latlon bbox from item
            ):
                asset_bbox_proj = geom_utils.reproject_bounds(
                    asset_bbox, asset_epsg, out_epsg
                )

            # If there's no bbox (or asset-level metadata is more accurate), compute one from the shape and geotrans
            else:
                if (
                    asset_transform is not None
                    and asset_shape is not None
                    and asset_epsg is not None
                ):
                    asset_affine = affine.Affine(*asset_transform[:6])
                    asset_bbox_proj = geom_utils.bounds_from_affine(
                        asset_affine,
                        asset_shape[0],
                        asset_shape[1],
                        asset_epsg,
                        out_epsg,
                    )

                # There's no bbox, nor shape and transform. The only info we have is `item.bbox` in lat-lon.
                else:
                    if item_bbox_proj is None:
                        try:
                            bbox_lonlat = item["bbox"]
                        except KeyError:
                            asset_bbox_proj = None
                        else:
                            # TODO handle error
                            asset_bbox_proj = geom_utils.reproject_bounds(
                                bbox_lonlat, 4326, out_epsg
                            )
                            item_bbox_proj = asset_bbox_proj
                            # ^ so we can reuse for other assets
                    else:
                        asset_bbox_proj = item_bbox_proj

            # Auto-compute bounds
            if bounds is None:
                if asset_bbox_proj is None:
                    raise ValueError(
                        f"Cannot automatically compute the bounds, "
                        f"since asset {id!r} on item {item_i} {item['id']!r} "
                        f"doesn't provide enough metadata to determine its spatial extent.\n"
                        f"We'd need at least one of (in order of preference):\n"
                        f"- The `proj:bbox` field set on the asset, or on the item\n"
                        f"- The `proj:shape` and `proj:transform` fields set on the asset, or on the item\n"
                        f"- A `bbox` set on the item {item['id']!r}\n\n"
                        "Please specify the `bounds=` or `bounds_latlon=` argument to set the output bounds manually."
                    )
                out_bounds = (
                    asset_bbox_proj
                    if out_bounds is None
                    else geom_utils.union_bounds(asset_bbox_proj, out_bounds)
                )
            else:
                # Drop asset if it doesn't overlap with the output bounds at all
                if asset_bbox_proj is not None and not geom_utils.bounds_overlap(
                    asset_bbox_proj, bounds
                ):
                    # I've got a blank space in my ndarray, baby / And I'll write your name
                    continue

            # Auto-compute resolutions
            if resolution is None:
                # Prefer computing resolutions from a geotrans, if it exists
                if asset_transform is not None and asset_epsg is not None:
                    asset_affine = asset_affine or affine.Affine(*asset_transform[:6])
                    if asset_epsg == out_epsg:
                        # Fastpath-ish when asset is already in the output CRS:
                        # pull directly from geotrans coefficients
                        if not asset_affine.is_rectilinear:
                            raise NotImplementedError(
                                f"Cannot automatically compute the resolution, "
                                f"since asset {id!r} on item {item_i} {item['id']!r} "
                                "has a non-rectilinear geotrans "
                                f'(its data is is not axis-aligned, or "north-up"): {asset_transform}. '
                                "We should be able to handle this but just don't want to deal with it right now.\n\n"
                                "Please specify the `resolution=` argument."
                            )
                        res_x, res_y = abs(asset_affine.a), abs(asset_affine.e)
                    else:
                        # Asset is in a different CRS; create a 1-pixel box and reproject
                        # to figure out its width/height in the output CRS
                        px_corner_xs, px_corner_ys = (
                            asset_affine * np.array([(0, 0), (0, 1), (1, 1), (1, 0)]).T
                        )

                        transformer = geom_utils.cached_transformer(
                            asset_epsg, out_epsg, skip_equivalent=True, always_xy=True
                        )
                        out_px_corner_xs, out_px_corner_ys = transformer.transform(
                            px_corner_xs, px_corner_ys, errcheck=True
                        )

                        res_y = max(out_px_corner_ys) - min(out_px_corner_ys)
                        res_x = max(out_px_corner_xs) - min(out_px_corner_xs)

                # If there's no geotrans, compute resolutions from `proj:shape`
                else:
                    if asset_bbox_proj is None or asset_shape is None:
                        raise ValueError(
                            f"Cannot automatically compute the resolution, "
                            f"since asset {id!r} on item {item_i} {item['id']!r} "
                            f"doesn't provide enough metadata to determine its native resolution.\n"
                            f"We'd need at least one of (in order of preference):\n"
                            f"- The `proj:transform` and `proj:epsg` fields set on the asset, or on the item\n"
                            f"- The `proj:shape` and one of `proj:bbox` or `bbox` fields set on the asset, "
                            "or on the item\n\n"
                            "Please specify the `resolution=` argument to set the output resolution manually. "
                            f"(Remember that resolution must be in the units of your CRS (http://epsg.io/{out_epsg})"
                            "---not necessarily meters."
                        )

                    # NOTE: this would be inaccurate if `proj:bbox` was provided,
                    # but the geotrans was non-rectilinear
                    # TODO check for that if there's a geotrans??
                    res_y = (asset_bbox_proj[3] - asset_bbox_proj[1]) / asset_shape[0]
                    res_x = (asset_bbox_proj[2] - asset_bbox_proj[0]) / asset_shape[1]

                if out_resolutions_xy is None:
                    out_resolutions_xy = (res_x, res_y)
                else:
                    out_resolutions_xy = (
                        # TODO do you always want the smallest resolution?
                        # Maybe support setting for controlling this (min, max, mode, etc)?
                        min(res_x, out_resolutions_xy[0]),
                        min(res_y, out_resolutions_xy[1]),
                    )

            # Phew, we figured out all the spatial stuff! Now actually store the information we care about.
            asset_table[item_i, asset_i] = (asset["href"], asset_bbox_proj)
            # ^ NOTE: If `asset_bbox_proj` is None, NumPy automatically converts it to NaNs

    # At this point, everything has been set (or there was as error)
    out_bounds = cast(Bbox, out_bounds)
    out_resolutions_xy = cast(Resolutions, out_resolutions_xy)
    out_epsg = cast(int, out_epsg)

    if snap_bounds:
        out_bounds = geom_utils.snapped_bounds(out_bounds, out_resolutions_xy)
    spec = RasterSpec(
        epsg=out_epsg,
        bounds=out_bounds,
        resolutions_xy=out_resolutions_xy,
    )

    # Drop items / asset IDs that are all to-be-skipped (either the asset didn't exist, or it was fully out-of-bounds)
    isnan_table = np.isnan(asset_table["bounds"]).all(axis=-1)
    # ^ use `"bounds"` only because np.isnan doesn't work on object dtype
    item_isnan = isnan_table.all(axis=1)
    asset_id_isnan = isnan_table.all(axis=0)

    if item_isnan.any() or asset_id_isnan.any():
        asset_table = asset_table[np.ix_(~item_isnan, ~asset_id_isnan)]
        asset_ids = [id for id, isnan in zip(asset_ids, asset_id_isnan) if not isnan]
        items = [item for item, isnan in zip(items, item_isnan) if not isnan]

    return asset_table, spec, asset_ids, items


def to_coords(
    items: ItemSequence,
    asset_ids: List[str],
    spec: RasterSpec,
    xy_coords: Literal["center", "topleft", False] = "topleft",
    properties: Union[bool, str, Sequence[str]] = True,
    band_coords: bool = True,
) -> Tuple[Dict[str, Union[pd.Index, np.ndarray, list]], List[str]]:

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
        coords.update(
            accumulate_metadata.metadata_to_coords(
                (item["properties"] for item in items),
                "time",
                fields=properties,
                skip_fields={"datetime"},
                # skip_fields={"datetime", "providers"},
            )
        )

        # Property-merging code using awkward array. Slightly shorter, not sure if it's faster,
        # probably not worth the dependency

        #     import awkward as ak

        #     awk_props = ak.Array([item._data for item in items]).properties
        #     for field, props in zip(ak.fields(awk_props), ak.unzip(awk_props)):
        #         if field == "datetime":
        #             continue

        #         # if all values are the same, collapse to a 0D coordinate
        #         try:
        #             if len(ak.run_lengths(props)) == 1:
        #                 props = ak.to_list(props[0])
        #                 # ^ NOTE: `to_list` because `ak.to_numpy` on string scalars (`ak.CharBehavior`)
        #                 # turns them into an int array of the characters!
        #         except NotImplementedError:
        #             # generally because it's an OptionArray (so there's >1 value anyway)
        #             pass

        #         try:
        #             props = np.squeeze(ak.to_numpy(props))
        #         except ValueError:
        #             continue

        #         coords[field] = xr.Variable(
        #             (("time",) + tuple(f"dim_{i}" for i in range(1, props.ndim)))
        #             if np.ndim(props) > 0
        #             else (),
        #             props,
        #         )
        # else:
        #     # For now don't use awkward when the field names are already known,
        #     # mostly so users don't have to have it installed.
        #     if isinstance(properties, str):
        #         properties = (properties,)
        #     for prop in properties:  # type: ignore (`properties` cannot be True at this point)
        #         coords[prop] = xr.Variable(
        #             "time", [item["properties"].get(prop) for item in items]
        #         )

    if band_coords:
        flattened_metadata_by_asset = [
            accumulate_metadata.accumulate_metadata_only_allsame(
                (item["assets"].get(asset_id, {}) for item in items),
                skip_fields={"href", "type", "roles"},
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

        coords.update(
            accumulate_metadata.metadata_to_coords(
                flattened_metadata_by_asset,
                "band",
                skip_fields={"href"},
                # skip_fields={"href", "title", "description", "type", "roles"},
            )
        )
        if any(eo_by_asset):
            coords.update(
                accumulate_metadata.metadata_to_coords(
                    eo_by_asset,
                    "band",
                    fields=["common_name", "center_wavelength", "full_width_half_max"],
                )
            )

    # Add `epsg` last in case it's also a field in properties; our data model assumes it's a coordinate
    coords["epsg"] = spec.epsg

    return coords, dims


def to_attrs(spec: RasterSpec) -> Dict[str, Any]:
    attrs = {"spec": spec, "crs": f"epsg:{spec.epsg}", "transform": spec.transform}

    resolutions = spec.resolutions_xy
    if resolutions[0] == resolutions[1]:
        attrs["resolution"] = resolutions[0]
    else:
        attrs["resolution_xy"] = resolutions
    return attrs
