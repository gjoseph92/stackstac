from __future__ import annotations

import collections
from typing import (
    AbstractSet,
    NamedTuple,
    Optional,
    Set,
    Union,
    Tuple,
    List,
)
import warnings


import affine
import numpy as np

from .raster_spec import IntFloat, Bbox, Resolutions, RasterSpec

from .stac_types import ItemSequence
from . import geom_utils

ASSET_TABLE_DT = np.dtype(
    [("url", object), ("bounds", "float64", 4), ("scale_offset", "float64", 2)]
)


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
    rescale: bool = True,
    dtype: np.dtype = np.dtype("float64"),
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
            raster_bands = asset.get("raster:bands")

            if raster_bands is not None:
                if len(raster_bands) != 1:
                    raise ValueError(
                        f"raster:bands has {len(raster_bands)} elements for asset {id!r}. "
                        "Multi-band rasters are not currently supported.\n"
                        "If you don't care about this asset, you can skip it by giving a list "
                        "of asset IDs you *do* want in `assets=`, and leaving this one out."
                        "For example:\n"
                        f"`assets={[x for x in asset_ids if x != id]!r}`"
                    )
                asset_scale = raster_bands[0].get("scale", 1)
                asset_offset = raster_bands[0].get("offset", 0)
            else:
                asset_scale = 1
                asset_offset = 0

            if rescale:
                if not np.can_cast(asset_scale, dtype):
                    raise ValueError(
                        f"`rescale=True`, but safe casting cannot be completed between "
                        f"asset scale value {asset_scale} and output dtype {dtype}.\n"
                        f"To continue using `{dtype=}`, pass `rescale=False` to retrieve "
                        "data in its raw, unscaled values. Or, if you want rescaled "
                        "values, pass a different `dtype=` (typically `float`)."
                    )

                if not np.can_cast(asset_offset, dtype):
                    raise ValueError(
                        f"`rescale=True`, but safe casting cannot be completed between "
                        f"asset offset value {asset_offset} and output dtype {dtype}.\n"
                        f"To continue using `{dtype=}`, pass `rescale=False` to retrieve "
                        "data in its raw, unscaled values. Or, if you want rescaled "
                        "values, pass a different `dtype=` (typically `float`)."
                    )

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

            assert isinstance(out_epsg, int), f"`out_epsg` not found. {out_epsg=}"
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
                            asset_epsg, out_epsg, always_xy=True
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

            # Auto-compute bounds
            # We do this last, so that if we have to skip all items (due to non-overlap),
            # we still get the spatial information needed to construct an array of NaNs.
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

            # Phew, we figured out all the spatial stuff! Now actually store the information we care about.
            asset_table[item_i, asset_i] = (
                asset["href"],
                asset_bbox_proj,
                (asset_scale, asset_offset),
            )
            # ^ NOTE: If `asset_bbox_proj` is None, NumPy automatically converts it to NaNs

    # At this point, everything has been set (or there was as error)
    assert out_bounds, f"{out_bounds=}"
    assert out_resolutions_xy is not None, f"{out_resolutions_xy=}"
    assert out_epsg is not None, f"{out_epsg=}"

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
