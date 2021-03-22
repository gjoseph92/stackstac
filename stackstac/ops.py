import numpy as np
import xarray as xr


# TODO `fill_value`s besides NaN
def _mosaic(chunk, axis):
    ax_length = chunk.shape[axis]
    if ax_length <= 1:
        return chunk
    out = np.take(chunk, 0, axis=axis)
    for i in range(1, ax_length):
        layer = np.take(chunk, i, axis=axis)
        out = np.where(np.isnan(out), layer, out)
    return out


def mosaic(arr: xr.DataArray, axis: int = 0):
    return arr.reduce(_mosaic, axis=axis)
