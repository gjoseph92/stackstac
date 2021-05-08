from typing import Optional

import xarray as xr
import dask.array as da
import numpy as np


def _unpack_block(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = np.moveaxis(arr, axis, -1)
    # ^ When we do `.view("uint8")`, if the array is >8-bit, NumPy will {double, quadruple, etc.}
    # the length of the last dimension to account for the extra elements in the 8-bit view.
    # We want the expanded dimension to be our new `bits` dimension, not some other one.
    u8_lil = (
        arr.astype(arr.dtype.newbyteorder("L"), copy=False)
        # ^ ensure it's little-endian
        .view("uint8")
    )
    bits = np.unpackbits(u8_lil, axis=-1, bitorder="little")
    # ^ `bitorder="little"` so element 0 -> bit 0
    return np.moveaxis(bits, -1, axis)


def expand_bits(
    arr: xr.DataArray, astype: Optional[np.dtype] = None, keep_mask=True, bit_axis=0
) -> xr.DataArray:
    """
    Expand a DataArray of bitpacked values, giving each bit its own entry in a new ``"bits"`` dimension.

    For ``dtype="uint16"``, the result would have a new dimension ``"bit"`` with 16 elements.
    The Nth item in ``"bit"`` would correspond to the Nth bit in the original values
    (after they'd been converted to uint16), starting from the least-significant bit.

    This ordering is consistent with most data sheets from imagery providers;
    if it says "Bit 3" on a data sheet, that's probaby ``unpack_bits(arr, ...).sel(bit=3)``.

    Parameters
    ---------
    arr:
        Array whose values are all bitpacked
    astype:
        dtype of the original data source (typically ``uint8`` or ``uint18``).
        ``arr`` is first converted to this type. For example, if ``arr`` is a float64
        array loaded from Landsat-8 data where the original GeoTIFFs are in uint16,
        you'd pass ``"uint16"`` here.
    keep_mask:
        If True (default), the ``arr.isnull()`` mask is re-applied to the result.
        Note that this will result in a ``float64`` array, where the possible values are 0, 1, or NaN.

        If False, any NaN values will be filled with 0s before unpacking, and the result will be a ``uint8`` array
        of values 0 or 1.

        If ``arr`` has an integer dtype, ``keep_mask`` is ignored and always considered False
        (since ``arr`` cannot contain NaNs).
    bit_axis:
        Insert the new ``"bit"`` dimension at this axis (default 0).

    Returns
    -------
    xarray.DataArray:
        DataArray with a ``"bit"`` dimension added, which is equal in length to the number of bytes in ``astype``.

    Example
    -------
    >>> import xarray as xr
    >>> import stackstac
    >>> arr = xr.DataArray([0b10000011_11111110, 0b11100011_00000001], dims="x")
    >>> arr
    <xarray.DataArray (x: 2)>
    array([33790, 58113])
    Dimensions without coordinates: x

    >>> expand_bits(arr, astype="uint16", bit_axis=-1)
    <xarray.DataArray (x: 2, bit: 16)>
    array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]], dtype=uint8)
    Dimensions without coordinates: x, bit

    Notice how the bits are printed in opposite order of how we wrote them above.
    That's because binary literals in Python (``0b10000011_11111110``) are right-to-left,
    with the least-significant bit---bit 0---furthest right.
    `expand_bits` puts bit 0 as item 0 in the ``bit`` dimension, which is then printed in
    left-to-right order.
    """
    if astype is not None:
        astype = np.dtype(astype)
    if arr.dtype.kind in ("i", "u"):
        keep_mask = False
        prep = arr
        if astype is None:
            astype = arr.dtype
    else:
        if astype is None:
            raise ValueError(
                f"Refusing to interpret {arr.dtype} data as bitpacked "
                "(bitpacked data almost always should have an integer dtype). "
                "Pass `astype=` to specify the dtype to convert this data to first "
                "(typically 'uint8' or 'uint16')."
            )
        prep = arr.fillna(0)

    prep = prep.astype(astype, copy=False).expand_dims("bit", bit_axis)

    # TODO use xr.apply_ufunc? Or just add `unpackbits` to dask
    if isinstance(prep.data, da.Array):
        bits_arr = prep.data.map_blocks(
            _unpack_block,
            chunks=prep.chunks[:bit_axis]
            + (astype.itemsize * 8,)
            + prep.chunks[bit_axis + 1 :],
            dtype="uint8",
            axis=bit_axis,
        )
    else:
        bits_arr = _unpack_block(prep.data, axis=bit_axis)

    bits_arr_xr = xr.DataArray(bits_arr, coords=prep.coords)
    if keep_mask:
        bits_arr_xr = bits_arr_xr.where(~arr.isnull())
    return bits_arr_xr
