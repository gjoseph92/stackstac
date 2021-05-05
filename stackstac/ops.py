from typing import Hashable, Sequence, Union
import numpy as np
import xarray as xr


# TODO `fill_value`s besides NaN
def _mosaic(arr, axis, reverse: bool = False):
    ax_length = arr.shape[axis]

    # "normal" means last -> first, "reversed" means first -> last,
    # so we apply `reversed` in the opposite case you'd expect
    indices = iter(range(ax_length) if reverse else reversed(range(ax_length)))
    out = np.take(arr, next(indices), axis=axis)

    for i in indices:
        layer = np.take(arr, i, axis=axis)
        out = np.where(np.isnan(out), layer, out)
    return out


def mosaic(
    arr: xr.DataArray,
    dim: Union[None, Hashable, Sequence[Hashable]] = None,
    axis: Union[None, int, Sequence[int]] = 0,
    reverse: bool = False,
):
    """
    Flatten a dimension of a `~xarray.DataArray` by picking the first non-NaN pixel.

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

    Returns
    -------
    xarray.DataArray:
        The mosaicked `~xarray.DataArray`.
    """
    return arr.reduce(_mosaic, dim=dim, axis=axis, keep_attrs=True, reverse=reverse)
