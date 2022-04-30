from typing import Tuple
from hypothesis import given, assume, strategies as st
from hypothesis.extra import numpy as st_np
import pytest

import xarray as xr
import dask.array as da
from dask.array.utils import assert_eq
import numpy as np

from stackstac.ops import mosaic
from stackstac.testing import strategies as st_stc


@pytest.mark.parametrize("dask", [False, True])
def test_mosaic_basic(dask: bool):
    arr = np.array(
        [
            [np.nan, 1, 2, np.nan],
            [np.nan, 10, 20, 30],
            [np.nan, 100, 200, np.nan],
        ]
    )
    if dask:
        arr = da.from_array(arr, chunks=1)

    xarr = xr.DataArray(arr)

    fwd = mosaic(xarr, axis=0)
    # Remember, forward means "last item on top" (because last item == last date == newest)
    assert_eq(fwd.data, np.array([np.nan, 100, 200, 30]), equal_nan=True)

    rev = mosaic(xarr, axis=0, reverse=True)
    assert_eq(rev.data, np.array([np.nan, 1, 2, 30]), equal_nan=True)


@pytest.mark.parametrize("dtype", [np.dtype("bool"), np.dtype("int"), np.dtype("uint")])
def test_mosaic_dtype_error(dtype: np.dtype):
    arr = xr.DataArray(np.arange(3).astype(dtype))
    with pytest.raises(ValueError, match="Cannot use"):
        mosaic(arr)


@given(
    st.data(),
    st_stc.raster_dtypes,
    st_np.array_shapes(max_dims=4, max_side=5),
    st.booleans(),
    st.booleans(),
)
def test_fuzz_mosaic(
    data: st.DataObject, dtype: np.dtype, shape: Tuple[int, ...], reverse: bool, use_dim: bool,
):
    """
    See if we can break mosaic.

    Not testing correctness much here, since that's hard to do without rewriting a mosaic implementation.
    """
    # `np.where` doesn't seem to preserve endianness.
    # Even with our best efforts to add an `astype` at the end, this will fail (on little-endian systems at least?)
    # with big-endian dtypes.
    assume(dtype.byteorder != ">")

    fill_value = data.draw(st_np.from_dtype(dtype), label="fill_value")
    arr = data.draw(st_np.arrays(dtype, shape, fill=st.just(fill_value)), label="arr")
    chunkshape = data.draw(
        st_np.array_shapes(
            min_dims=arr.ndim, max_dims=arr.ndim, max_side=max(arr.shape)
        ), label="chunkshape"
    )
    axis = data.draw(st.integers(-arr.ndim + 1, arr.ndim - 1), label="axis")

    darr = da.from_array(arr, chunks=chunkshape)
    split_every = data.draw(st.integers(1, darr.numblocks[axis]), label="split_every")
    xarr = xr.DataArray(darr)

    if use_dim:
        kwargs = dict(dim=xarr.dims[axis])
    else:
        kwargs = dict(axis=axis)

    result = mosaic(
        xarr, reverse=reverse, nodata=fill_value, split_every=split_every, **kwargs
    )
    assert result.dtype == arr.dtype
    result_np = mosaic(xr.DataArray(arr), axis=axis, reverse=reverse, nodata=fill_value)
    assert_eq(result.data, result_np.data, equal_nan=True)
