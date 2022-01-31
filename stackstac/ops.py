from typing import Hashable, Sequence, Union
import numpy as np
import xarray as xr


def _mosaic(arr, axis, reverse: bool = False, nodata: Union[int, float] = np.nan):
    ax_length = arr.shape[axis]

    # "normal" means last -> first, "reversed" means first -> last,
    # so we apply `reversed` in the opposite case you'd expect
    indices = iter(range(ax_length) if reverse else reversed(range(ax_length)))
    out = np.take(arr, next(indices), axis=axis)

    for i in indices:
        layer = np.take(arr, i, axis=axis)
        where = np.isnan(out) if np.isnan(nodata) else out == nodata
        out = np.where(where, layer, out)
    return out


def mosaic(
    arr: xr.DataArray,
    dim: Union[None, Hashable, Sequence[Hashable]] = None,
    axis: Union[None, int, Sequence[int]] = 0,
    reverse: bool = False,
    nodata: Union[int, float] = np.nan,
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
        The axis number to mosaic. Default: 0. Only one of
        ``dim`` and ``axis`` can be given.
    reverse:
        If False (default), the last item along the dimension is on top.
        If True, the first item in the dimension is on top.
    nodata:
        The value to treat as invalid. Default: NaN.

        To catch common mis-use, raises a ``ValueError`` if ``nodata=nan``
        is used when the array has an integer or boolean dtype. Since NaN
        cannot exist in those arrays, this indicates a different ``nodata``
        value needs to be used.

    Returns
    -------
    xarray.DataArray:
        The mosaicked `~xarray.DataArray`.
    """
    if np.isnan(nodata) and arr.dtype.kind in "biu":
        # Try to catch usage errors forgetting to set `nodata=`
        raise ValueError(
            f"Cannot use {nodata=} when mosaicing a {arr.dtype} array, since {nodata} cannot exist in the array."
        )
    return arr.reduce(
        _mosaic,
        dim=dim,
        axis=axis,
        keep_attrs=True,
        reverse=reverse,
        nodata=nodata,
    )
