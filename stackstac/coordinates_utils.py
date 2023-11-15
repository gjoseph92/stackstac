from __future__ import annotations

from typing import Any, Container, Iterable, Iterator, Mapping, TypeVar, Union

import numpy as np
import pandas as pd
import xarray as xr

Coordinates = Mapping[str, Union[pd.Index, np.ndarray, xr.Variable, list]]


def items_to_coords(
    items: Iterable[tuple[tuple[int, ...], str, object]],
    *,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
) -> dict[str, xr.Variable]:
    """
    Create coordinates from ``(index, field, value)`` tuples.

    Say you want to make coordinates for a 2x2 DataArray with dimensions ``time, band``.
    Your metadata might look like::

        ((0, 0), "color", "red"),
        ((0, 0), "day", "mon"),
        ((0, 0), "href", "0/red"),

        ((0, 1), "color", "green"),
        ((0, 1), "day", "mon"),
        ((0, 1), "href", "0/green"),

        ((1, 0), "color", "red"),
        ((1, 0), "day", "tues"),
        ((1, 0), "href", "1/red"),

        ((1, 1), "color", "green"),
        ((1, 1), "day", "tues"),
        # ((1, 1), "href", "1/green") skip this to see how missing values are handled

    This would produce a dict of coordinates like::

        {
            "color": xr.Variable(
                dims=["band"], data=["red", "green"],
            ),
            "day": xr.Variable(
                dims=["time"], data=["mon", "tues"],
            ),
            "href": xr.Variable(
                dims=["time", "band"],
                data=[
                    ["0/red", "0/green"],
                    ["1/red", None],
                ],
            ),
        }

    Each ``item`` in the iterator gives the value for a particular field at a particular
    coordinate in the array. Per field, those are combined and de-duplicated: you can
    see how "color" only labels the "band" dimension, because it's the same across all
    times; "day" only labels the time dimension, because it's the same across all bands,
    and "href" labels both, because it varies across both dimensions.

    Parameters
    ----------
    items:
        Iterable of ``(index, field, value)`` tuples. ``index`` is a tuple of
        integers for the position in the array that this item labels.
        ``field`` is the name of a coordinate, and ``value`` is its value.
    shape:
        The shape of the data being labeled. If a coordinate covered all of the
        dimensions, this is the shape it would have. Each ``index`` in ``items``
        must be the same length (same number of dimensions) as ``shape``.
    dims:
        Dimension names corresponding to ``shape``. Must be the same length
        as ``shape``.

    Returns
    -------
    coords: dict of xarray Variables, one per unique ``field`` in the items.
    """
    assert len(shape) == len(
        dims
    ), f"{shape=} has {len(shape)} dimensions; {dims=} has {len(dims)}"

    fields = {}
    # {field:
    #   np.array([
    #       [v_asset_0, v_asset_1, ...],  # item 0
    #       [v_asset_0, v_asset_1, ...],  # item 1
    #       ...,
    #   ])
    # }
    for idx, field, value in items:
        assert len(idx) == len(
            shape
        ), f"Expected {len(shape)}-dimensional index, got {idx}"
        try:
            values = fields[field]
        except KeyError:
            # Haven't seen this field before, so create the array of its values.
            # We guess whether the dtype will be numeric, or str/object, based
            # on this first value.
            if isinstance(value, (int, float)):
                dtype = float
                fill = np.nan
                # NOTE: we don't use int64, even for ints, because there'd be no
                # way to represent missing values. Using pandas nullable arrays
                # could be interesting at some point.
            else:
                dtype = object
                fill = None

            values = fields[field] = np.full(shape, fill, dtype=dtype)

        try:
            values[idx] = value
        except (TypeError, ValueError):
            # If our dtype guess was wrong, or a field has values of multiple types,
            # promote the whole array to a more generic dtype.
            # A `ValueError` might be "could not convert string to float".
            # (so if there did happen to be string values that could be parsed as numbers,
            # we'd do that, which is probably ok?)
            try:
                new_dtype = np.result_type(value, values)
            except TypeError:
                # Thrown when "no common DType exists for the given inputs.
                # For example they cannot be stored in a single array unless the
                # dtype is `object`"
                new_dtype = object

            values = fields[field] = values.astype(new_dtype)
            values[idx] = value

    deduped = {field: deduplicate_axes(arr) for field, arr in fields.items()}

    return {field: xr.Variable(dims, arr).squeeze() for field, arr in deduped.items()}


def deduplicate_axes(arr: np.ndarray) -> np.ndarray:
    "Flatten dimensions to length 1 where all values are duplicated"
    if arr.dtype.kind in ("f", "c"):
        arr_nan = np.isnan(arr)
        return _nandeduplicate_axes(arr, arr_nan, np.where(arr_nan, True, arr))
    return _deduplicate_axes(arr)


def _deduplicate_axes(arr: np.ndarray) -> np.ndarray:
    if arr.size <= 1:
        return arr

    for axis in range(arr.ndim):
        if arr.shape[axis] <= 1:
            continue
        first = arr.take([0], axis=axis)
        # ^ note `[0]` instead of `0`: that keeps the dimension
        # as length 1 instead of dropping it
        allsame = (arr == first).all(axis=axis)
        if allsame.all():
            return deduplicate_axes(first)
    return arr


def _nandeduplicate_axes(
    arr: np.ndarray, arr_nan: np.ndarray, arr_filled: np.ndarray
) -> np.ndarray:
    if arr.size <= 1:
        return arr

    for axis in range(arr.ndim):
        if arr.shape[axis] <= 1:
            continue
        first_nan = arr_nan.take([0], axis=axis)
        first_filled = arr_filled.take([0], axis=axis)
        # ^ note `[0]` instead of `0`: that keeps the dimension
        # as length 1 instead of dropping it
        allsame = (arr_filled == first_filled).all(axis=axis) & (
            arr_nan == first_nan
        ).all(axis=axis)
        if allsame.all():
            return _nandeduplicate_axes(
                arr.take([0], axis=axis), first_nan, first_filled
            )
    return arr


VT = TypeVar("VT")


def unnested_items(
    items: Iterable[tuple[str, VT]], prefix: tuple[str, ...] = (), sep: str = "_"
) -> Iterator[tuple[str, VT]]:
    """
    Iterate over flattened dicts, prefixing sub-keys with the name of their parent key.

    Example
    -------
    >>> list(unnested_dict_items({
    ...     "foo": 1,
    ...     "bar": {
    ...         "a": 2,
    ...         "foo": 3,
    ...     },
    ... }.items()))
    [
        ("foo", 1),
        ("bar_a", 2),
        ("bar_foo", 3),
    ]
    """
    for k, v in items:
        if isinstance(v, dict) and v:
            yield from unnested_items(v.items(), prefix=prefix + (k,), sep=sep)
        else:
            yield sep.join(prefix + (k,)) if prefix else k, v

    # Note that we don't descend into lists/tuples. For the purposes of STAC metadata,
    # there'd be no reason to do this: we're not going to make an xarray coordinate like
    # `classification:bitfields_0_name`, `classification:bitfields_0_fill`, ...,
    # `classification:bitfields_8_name`, `classification:bitfields_8_fill`
    # and unpack a separate coordinate for every field in a sequence. Rather, we
    # `preserve anything that's a sequence as a single element in an object array.


def scalar_sequence(x):
    """
    Convert sequence inputs into NumPy scalars.

    Use this to wrap inputs to `np.array` that you don't want to be treated as
    additional axes in the array.

    Example
    -------
    >>> s = scalar_sequence([1, 2])
    >>> s
    array(list([1, 2]), dtype=object)
    >>> s.shape
    ()
    >>> s.item()
    [1, 2]
    >>> arr = np.array([s])
    >>> arr
    >>> arr.shape
    (1,)
    >>> # for comparision, if we hadn't wrapped it:
    >>> np.array([[1, 2]]).shape
    >>> (1, 2)
    """
    if not isinstance(x, (list, tuple)):
        return x

    scalar = np.empty((), dtype=object)  # basically a pointer
    scalar[()] = x
    return scalar


def unpacked_per_band_asset_fields(
    asset: Iterable[tuple[str, Any]], fields: Container
) -> Iterator[tuple[str, Any]]:
    """
    Unpack 1-length list/tuple values for the given ``fields``.

    For keys of ``asset`` in ``fields``, if the value is a 1-length
    list or tuple, use its single value. Otherwise, use an empty dict.
    """
    # NOTE: this will have to change a lot when we support multi-band assets;
    # this is predicated on each asset having exactly 1 band.
    for k, v in asset:
        if k in fields:
            if isinstance(v, (list, tuple)):
                if len(v) == 1:
                    v = v[0]
                else:
                    # For >1 band, drop metadata entirely (you can't use the data anyway).
                    # Otherwise, coordinates would be a mess: both unpacked `eo:bands`
                    # fields like `eo:bands_common_name`, and plain `eo:bands` which would
                    # be None for all 1-band assets, and contain the dicts for multi-band
                    # assets.
                    continue
            elif v is None:
                continue

        yield k, v
