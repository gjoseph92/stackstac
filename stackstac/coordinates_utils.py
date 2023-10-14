from __future__ import annotations

import numpy as np


def deduplicate_axes(arr: np.ndarray) -> np.ndarray:
    "Flatten dimensions to length 1 where all values are duplicated"
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


def unnest_dicts(item, prefix=(), sep="_"):
    """
    Flatten nested dicts, prefixing sub-keys with the name of their parent key.

    Example
    -------
    >>> unnest_dicts({
    ...     "foo": 1,
    ...     "bar": {
    ...         "a": 2,
    ...         "foo": 3,
    ...     },
    ... })
    {
        "foo": 1,
        "bar_a": 2,
        "bar_foo": 3,
    }
    """
    if isinstance(item, dict):
        result = {}
        for k, v in item.items():
            sub_prefix = prefix + (k,)
            unnested = unnest_dicts(v, prefix=sub_prefix, sep=sep)
            if isinstance(unnested, dict):
                result.update(unnested)
            else:
                result[sep.join(sub_prefix)] = unnested
        return result

    if isinstance(item, (list, tuple)) and len(item) == 1:
        return unnest_dicts(item[0], prefix=prefix, sep=sep)

    # Note that we don't descend into lists/tuples. For the purposes of STAC metadata,
    # there'd be no reason to do this: we're not going to make an xarray coordinate like
    # `classification:bitfields_0_name`, `classification:bitfields_0_fill`, ...,
    # `classification:bitfields_8_name`, `classification:bitfields_0_fill`
    # and unpack a separate coordinate for every field in a sequence. Rather, we rely on
    # `scalar_sequence` to preserve anything that's a sequence into a single coordinate.

    return item


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


def descalar_obj_array(arr: np.ndarray) -> np.ndarray:
    """
    In an object array containing NumPy object scalars, unpack the scalars.

    Note that this may mutate the array.

    Example
    -------
    >>> s = scalar_sequence([1, 2])
    >>> arr = np.array([s])
    >>> arr[0]
    array(list([1, 2]), dtype=object)
    >>> # remove the indirection of the NumPy scalar
    >>> unpacked = descalar_obj_array(arr)
    >>> unpacked[0]
    >>> [1, 2]

    """
    if arr.dtype.kind != "O":
        return arr

    for idx in np.ndindex(arr.shape):
        x = arr[idx]
        if isinstance(x, np.ndarray) and x.shape == ():
            arr[idx] = x.item()
    return arr
