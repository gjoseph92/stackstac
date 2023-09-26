import numpy as np
import pandas as pd
import xarray as xr
from itertools import chain


def is_constant(arr):
    """
    Return True if arr is constant along the first dimension

    Parameters
    ----------
    arr:
        list or array
    """
    is_constant = True
    first_value = arr[0]
    for i in arr:
        if i != first_value:
            is_constant = False
            break
    return is_constant


def flatten(arr):
    """
    Flatten a list

    Parameters
    ----------
    arr:
        list or array
    """
    return [item for sublist in arr for item in sublist]


def accumulate_assets_coords(items, asset_ids, coords):
    """
    Accumulate and deduplicate coordinnates (as per xarray nomenclature) from the items 'assets'

    Parameters
    ----------
    items:
        list of stac items (in dictionnary format)
    asset_ids:
        list asset keys to be considered
    coords:
        dictionnary of existing coords to avoid overwriting them
    """

    # Selecting assets from 'asset_ids' and ordering them
    assets_items = [
        [item["assets"][asset_id] for asset_id in asset_ids] for item in items
    ]

    # List of all assets keys (deduplicated)
    asset_keys = [
        key
        for key in set(chain.from_iterable(flatten(assets_items)))
        if key not in coords.keys()
    ]

    # Flatening cases with 'eo:bands'
    if "eo:bands" in asset_keys:
        out = []
        for assets_item in assets_items:
            for asset in assets_item:
                eo = asset.pop("eo:bands", [{}])
                asset.update(eo[0] if isinstance(eo, list) else {})
            out.append(assets_item)
        assets_items = out
        asset_keys = set(chain.from_iterable(flatten(assets_items)))

    # Making sure all assets have all asset_keys and filling the others with np.nan
    assets_items = [
        [[asset.get(key, np.nan) for key in asset_keys] for asset in assets_item]
        for assets_item in assets_items
    ]

    # Facilitating transposing via numpy for duplicates finding
    assets_items = np.array(assets_items, dtype=object)

    # Dropping all nulls
    nan_mask = pd.notnull(assets_items).any(axis=2).any(axis=0)
    assets_items = assets_items[:, nan_mask, :]

    # Looping through assets_keys and determining what is unique and what is not
    assets_coords = {}
    for asset_key, band_oriented_coords, time_oriented_coords in zip(
        asset_keys, assets_items.transpose(2, 1, 0), assets_items.transpose(2, 0, 1)
    ):
        is_band_dependant = is_constant(band_oriented_coords.tolist())
        is_time_dependant = is_constant(time_oriented_coords.tolist())

        if is_time_dependant and is_band_dependant:
            assets_coords[asset_key] = time_oriented_coords[
                0, 0
            ]  # same than band_oriented_coords[0, 0]
        elif is_time_dependant:
            assets_coords[asset_key] = xr.Variable(["band"], time_oriented_coords[0])
        elif is_band_dependant:
            assets_coords[asset_key] = xr.Variable(["time"], band_oriented_coords[0])
        else:
            # rioxarray convention is ordered: time, band, y, x
            assets_coords[asset_key] = xr.Variable(
                ["time", "band"], time_oriented_coords
            )

    return assets_coords


def accumulate_properties_coords(items, coords):
    """
    Accumulate and deduplicate coordinnates (as per xarray nomenclature) from the items 'properties'

    Parameters
    ----------
    items:
        list of stac items (in dictionnary format)
    coords:
        dictionnary of existing coords to avoid overwriting them
    """
    # Selecting properties only
    properties_items = [item["properties"] for item in items]

    # List of all assets keys (deduplicated)
    properties_keys = [
        key
        for key in set(chain.from_iterable(properties_items))
        if key not in coords.keys()
    ]

    # Making sure all properties have all properties_keys and ordering them
    properties_items = [
        [properties_item.get(key, np.nan) for key in properties_keys]
        for properties_item in properties_items
    ]

    # Facilitating transposing via numpy for duplicates finding
    properties_items = np.array(properties_items, dtype=object)

    # Dropping all nulls
    nan_mask = pd.notnull(properties_items).any(axis=0)
    properties_items = properties_items[:, nan_mask]

    # Looping through properties_keys and determining what is unique and what is not
    properties_coords = {}
    for properties_key, time_oriented_coords in zip(
        properties_keys, properties_items.T
    ):
        is_time_dependant = not is_constant(time_oriented_coords.tolist())

        if is_time_dependant:
            properties_coords[properties_key] = xr.Variable(
                ["time"], time_oriented_coords
            )
        else:
            # rioxarray convention is ordered: time, band, y, x
            properties_coords[properties_key] = time_oriented_coords[0]

    return properties_coords
