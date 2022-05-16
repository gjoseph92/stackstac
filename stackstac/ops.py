from __future__ import annotations

from functools import partial
from typing import Hashable, Tuple, List, Union, Optional

import numpy as np
import dask.array as da
import xarray as xr


def _mosaic_base(
    arr: np.ndarray,
    axis: int,
    *,
    reverse: bool,
    nodata: Union[int, float],
    keepdims: bool = False,
    initial: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Internal function to mosaic a NumPy array along an axis. Does some extra funky stuff to be useful with dask.

    Parameters
    ----------
    arr: array to mosaic, must be at least 1D
    axis: axis to reduce
    reverse: False = last on top, True = first on top
    nodata: the value to treat as invalid
    keepdims: whether to preserve the reduced axis as 1-length
    initial:
        Array to use as the initial layer, instead of the first index into ``arr``.
        Used when chaning mosaics together.

    Returns
    -------
    result, done
        The mosaicked array, and a bool of whether mosaicing is complete (every pixel in ``result`` is valid)
    """
    if isinstance(axis, tuple):
        # `da.reduction` likes to turn int axes into tuples
        assert len(axis) == 1, f"Multiple axes to mosaic not supported: {axis=}"
        axis = axis[0]

    ax_length = arr.shape[axis] if np.ndim(arr) else 1

    # "normal" means last -> first, "reversed" means first -> last,
    # so we apply `reversed` in the opposite case you'd expect
    indices = iter(range(ax_length) if reverse else reversed(range(ax_length)))

    if initial is None:
        i = next(indices)
        out = np.take(arr, [i] if keepdims else i, axis=axis)
    else:
        out = initial

    # TODO implement this with Numba, would be much faster
    done = False
    for i in indices:
        where = np.isnan(out) if np.isnan(nodata) else out == nodata
        if not where.any():
            # Short-circuit when mosaic is complete (no invalid pixels left)
            done = True
            break
        layer = np.take(arr, [i] if keepdims else i, axis=axis)
        out = np.where(where, layer, out)

    # Final `where` may have completed the mosaic, but we wouldn't have checked
    if not done:
        where = np.isnan(out) if np.isnan(nodata) else out == nodata
        done = not where.any()

    return out, done


def _mosaic_np(
    arr: np.ndarray,
    axis: int,
    *,
    reverse: bool,
    nodata: Union[int, float],
    keepdims: bool = False,
) -> np.ndarray:
    "Mosaic a NumPy array, without returning ``done``"
    return _mosaic_base(arr, axis, reverse=reverse, nodata=nodata, keepdims=keepdims)[0]


def _mosaic_dask_aggregate(
    arrs: Union[List[np.ndarray], np.ndarray],
    axis: Union[int, Tuple[int]],
    keepdims: bool,
    *,
    reverse: bool,
    nodata: Union[int, float],
) -> np.ndarray | np.ndarray:
    """
    Mosaic a list of NumPy arrays (or a single one) without concatenating them all first.

    Avoiding concatenation lets us avoid unnecessary copies and memory spikes.
    """
    if not isinstance(arrs, list):
        arrs = [arrs]

    if isinstance(axis, tuple):
        # `da.reduction` likes to turn int axes into tuples
        assert len(axis) == 1, f"Multiple axes to mosaic not supported: {axis=}"
        axis = axis[0]

    out: Optional[np.ndarray] = None
    for arr in arrs if reverse else arrs[::-1]:
        # ^ Remember, `reverse` is backwards of what you'd think
        if out is None and (arr.ndim == 0 or arr.shape[axis] <= 1):
            # There's nothing to mosaic in this chunk (it's effectively 1-length along `axis` already).
            # Skip mosaicing, just drop the axis if we're supposed to.
            out = np.take(arr, 0, axis=axis) if not keepdims else arr
        else:
            # Multiple entries along the axis: mosaic them, using any
            # mosaic we've already built up as a starting point.
            out, done = _mosaic_base(
                arr,
                axis,
                reverse=reverse,
                nodata=nodata,
                keepdims=keepdims,
                initial=out,
            )
            if done:
                break

    assert out is not None, "Cannot mosaic zero arrays"
    return out


def _mosaic_dask(
    arr: da.Array,
    axis: int,
    *,
    reverse: bool,
    nodata: Union[int, float],
    split_every: Union[None, int],
) -> da.Array:
    "Tree-reduction-based mosaic for Dask arrays."
    return da.reduction(
        arr,
        partial(_mosaic_np, reverse=reverse, nodata=nodata),
        partial(_mosaic_dask_aggregate, reverse=reverse, nodata=nodata),
        axis=axis,
        keepdims=False,
        dtype=arr.dtype,
        split_every=split_every,
        name="mosaic",
        meta=da.utils.meta_from_array(arr, ndim=arr.ndim - 1),
        concatenate=False,
    )


def mosaic(
    arr: xr.DataArray,
    dim: Union[None, Hashable] = None,
    axis: Union[None, int] = 0,
    *,
    reverse: bool = False,
    nodata: Union[int, float] = np.nan,
    split_every: Union[None, int] = None,
):
    """
    Flatten a dimension of a `~xarray.DataArray` by picking the first valid pixel.

    The order of mosaicing is from last to first, meaning the last item is on top.

    Parameters
    ----------
    arr:
        The `DataArray` to mosaic.
    dim:
        The dimension name to mosaic. Default: None.
    axis:
        The axis number to mosaic. Default: 0. If ``dim`` is given, ``axis``
        is ignored.
    reverse:
        If False (default), the last item along the dimension is on top.
        If True, the first item in the dimension is on top.
    nodata:
        The value to treat as invalid. Default: NaN.

        To catch common mis-use, raises a ``ValueError`` if ``nodata=nan``
        is used when the array has an integer or boolean dtype. Since NaN
        cannot exist in those arrays, this indicates a different ``nodata``
        value needs to be used.
    split_every:
        For Dask arrays: how many *chunks* to mosaic together at once.

        The Dask default is 4. A higher number will mean a smaller graph,
        but higher peak memory usage.

        Ignored for NumPy arrays.

    Returns
    -------
    xarray.DataArray:
        The mosaicked `~xarray.DataArray`.
    """
    if np.isnan(nodata) and arr.dtype.kind in "biu":
        # Try to catch usage errors forgetting to set `nodata=`
        raise ValueError(
            "You've probably forgotten to pass a custom `nodata=` argument. "
            f"Cannot use {nodata=} (the default) when mosaicing a {arr.dtype} array, "
            f"since {nodata} cannot exist in that dtype. "
        )

    axis = None if dim is not None else axis

    func = (
        partial(_mosaic_dask, split_every=split_every)
        if isinstance(arr.data, da.Array)
        else _mosaic_np
    )
    return arr.reduce(
        func,
        dim=dim,
        axis=axis,
        keep_attrs=True,
        reverse=reverse,
        nodata=nodata,
    )
