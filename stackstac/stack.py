from __future__ import annotations

from typing import AbstractSet, List, Literal, Optional, Sequence, Type, Union

import numpy as np
import xarray as xr
import dask
from rasterio.enums import Resampling

from .prepare import prepare_items, to_attrs, to_coords
from .raster_spec import Bbox, IntFloat, Resolutions
from .reader_protocol import Reader
from .rio_env import LayeredEnv
from .rio_reader import AutoParallelRioReader
from .stac_types import ItemCollectionIsh, ItemIsh, items_to_plain
from .to_dask import items_to_dask


def stack(
    items: Union[ItemCollectionIsh, ItemIsh],
    assets: Optional[Union[List[str], AbstractSet[str]]] = frozenset(
        ["image/tiff", "image/x.geotiff", "image/vnd.stac.geotiff", "image/jp2"]
    ),
    epsg: Optional[int] = None,
    resolution: Optional[Union[IntFloat, Resolutions]] = None,
    bounds: Optional[Bbox] = None,
    bounds_latlon: Optional[Bbox] = None,
    snap_bounds: bool = True,
    resampling: Resampling = Resampling.nearest,
    chunksize: int = 1024,
    dtype: np.dtype = np.dtype("float64"),
    fill_value: Optional[Union[int, float]] = np.nan,
    rescale: bool = True,
    sortby_date: Literal["asc", "desc", False] = "asc",
    xy_coords: Literal["center", "topleft", False] = "center",
    properties: Union[bool, str, Sequence[str]] = True,
    band_coords: bool = True,
    gdal_env: Optional[LayeredEnv] = None,
    reader: Type[Reader] = AutoParallelRioReader,
) -> xr.DataArray:
    """
    Create an `xarray.DataArray` of all the STAC items, reprojected to the same grid and stacked by time.

    The DataArray's dimensions will be ``("time", "band", "y", "x")``. It's backed by
    a lazy `Dask array <dask.array.Array>`, so you can manipulate it without touching any data.

    We'll try to choose the output coordinate reference system, resolution, and bounds
    based on the metadata in the STAC items. However, if not all items have the necessary
    metadata, or aren't in the same coordinate reference system, you'll have specify these
    yourself---``epsg`` and ``resolution`` are the two parameters you'll set most often.

    Examples
    --------
    >>> import stackstac
    >>> import satsearch
    >>> items = satsearch.Search(...).items()

    >>> # Use default CRS, resolution, bounding box, etc.
    >>> xr_stack = stackstac.stack(items)
    >>>
    >>> # Reproject to 100-meter resolution in web mercator
    >>> xr_stack = stackstac.stack(items, epsg=3857, resolution=100)
    >>>
    >>> # Only use specific asset IDs
    >>> xr_stack = stackstac.stack(items, assets=["B01", "B03", "B02"])
    >>>
    >>> # Clip to a custom bounding box
    >>> xr_stack = stackstac.stack(items, bounds_latlon=[-106.2, 35.6, -105.6, 36])
    >>>
    >>> # Turn off all metadata if you don't need it
    >>> xr_stack = stackstac.stack(
    ...     items, properties=False, bands_coords=False, xy_coords=False, sortby_date=False
    ... )
    >>>
    >>> # Custom dtype and fill_value
    >>> xr_stack = stackstac.stack(items, rescale=False, fill_value=0, dtype="uint16")

    Note
    ----
    Don't be scared of all the parameters!

    Though there are lots of options, you can leave nearly all of them as their defaults.

    Parameters
    ----------
    items:
        The STAC items to stack. Can be a plain Python list of dicts
        following the STAC JSON specification, or objects from
        the `satstac <https://github.com/sat-utils/sat-stac>`_ (preferred) or
        `pystac <https://github.com/stac-utils/pystac>`_ libraries.
    assets:
        Which asset IDs to use. Any Items missing a particular Asset will return an array
        of ``fill_value`` for that Asset. By default, returns all assets with a GeoTIFF
        or JPEG2000 ``type``.

        If None, all assets are used.

        If a list of strings, those asset IDs are used.

        If a set, only assets compatible with those mimetypes are used (according to the
        ``type`` field on each asset). Note that if you give ``assets={"image/tiff"}``,
        and the asset ``B1`` has ``type="image/tiff"`` on some items but ``type="image/png"``
        on others, then ``B1`` will not be included. Mimetypes structure is respected, so
        ``image/tiff`` will also match ``image/tiff; application=geotiff``; ``image`` will match
        ``image/tiff`` and ``image/jp2``, etc. See the `STAC common media types <MT>`_ for ideas.

        Note: each asset's data must contain exactly one band. Multi-band assets (like an RGB GeoTIFF)
        are not yet supported.

        .. _MT: https://github.com/radiantearth/stac-spec/blob/master/best-practices.md#common-media-types-in-stac
    epsg:
        Reproject into this coordinate reference system, as given by an `EPSG code <http://epsg.io>`_.
        If None (default), uses whatever CRS is set on all the items. In this case, all Items/Assets
        must have the ``proj:epsg`` field, and it must be the same value for all of them.
    resolution:
        Output resolution. Careful: this must be given in the output CRS's units!
        For example, with ``epsg=4326`` (meaning lat-lon), the units are degrees of
        latitude/longitude, not meters. Giving ``resolution=20`` in that case would mean
        each pixel is 20ºx20º (probably not what you wanted). You can also give pair of
        ``(x_resolution, y_resolution)``.

        If None (default), we try to calculate each Asset's resolution based on whatever metadata is available,
        and pick the minimum of all the resolutions---meaning by default, all data will be upscaled to
        match the "finest" or "highest-resolution" Asset.

        To estimate resolution, these combinations of fields must be set on each Asset or Item
        (in order of preference):

        * The ``proj:transform`` and ``proj:epsg`` fields
        * The ``proj:shape`` and one of ``proj:bbox`` or ``bbox`` fields

    bounds:
        Output spatial bounding-box, as a tuple of ``(min_x, min_y, max_x, max_y)``.
        This defines the ``(west, south, east, north)`` rectangle the output array will cover.
        Values must be in the same coordinate reference system as ``epsg``.

        If None (default), the bounding box of all the input items is automatically used.
        (This only requires the ``bbox`` field to be set on each Item, which is a required
        field in the STAC specification, so only in rare cases will auto-calculating the bounds
        fail.) So in most cases, you can leave ``bounds`` as None. You'd only need to set it
        when you want to use a custom bounding box.

        When ``bounds`` is given, any assets that don't overlap those bounds are dropped.
    bounds_latlon:
        Same as ``bounds``, but given in degrees (latitude and longitude) for convenience.
        Only one of ``bounds`` and ``bounds_latlon`` can be used.
    snap_bounds:
        Whether to snap the bounds to whole-number intervals of ``resolution`` to prevent
        fraction-of-a-pixel offsets. Default: True.

        This is equivalent to the ``-tap`` or
        `target-align pixels <https://gis.stackexchange.com/questions/165402/how-to-use-tap-in-gdal-rasterize>`_
        argument in GDAL.
    resampling:
        The rasterio resampling method to use when reprojecting or rescaling data to a different CRS or resolution.
        Default: ``rasterio.enums.Resampling.nearest``.
    chunksize:
        The chunksize to use for the spatial component of the Dask array, in pixels.
        Default: 1024. Can be given in any format Dask understands for a ``chunks=`` argument,
        such as ``1024``, ``(1024, 2048)``, ``15 MB``, etc.

        This is basically the "tile size" all your operations will be parallelized over.
        Generally, you should use the internal tile size/block size of whatever data
        you're accessing (or a multiple of that). For example, if all the assets are in
        Cloud-Optimized GeoTIFF files with an internal tilesize of 512, pass ``chunksize=512``.

        You want the chunks of the Dask array to align with the internal tiles of the data.
        Otherwise, if those grids don't line up, processing one chunk of your Dask array could
        require reading many tiles from the data, parts of which will then be thrown out.
        Additionally, those same data tiles might need to be re-read (and re-thrown-out)
        for a neighboring Dask chunk, which is just as inefficient as it sounds (though setting
        a high ``GDAL_CACHEMAX`` via ``gdal_env`` will help keep more data tiles cached,
        at the expense of using more memory).

        Of course, when reprojecting data to a new grid, the internal tilings of each input
        almost certainly won't line up anyway, so some misalignment is inevitable, and not
        that big of a deal.

        **This is the one parameter we can't pick for you automatically**, because the STAC
        specification offers no metadata about the internal tiling of the assets. We'd
        have to open the data files to find out, which is very slow. But to make an educated
        guess, you should look at ``rasterio.open(url).block_shapes`` for a few sample assets.

        The most important thing to avoid is making your ``chunksize`` here *smaller* than
        the internal tilesize of the data. If you want small Dask chunks for other reasons,
        don't set it here---instead, call ``.chunk`` on the DataArray to re-chunk it.
    dtype:
        The NumPy data type of the output array. Default: ``float64``. Must be a data type
        that's compatible with ``fill_value``. Note that if ``fill_value`` is None, whatever nodata
        value is set in each asset's file will be used, so that value needs to be compatible
        with ``dtype`` as well.
    fill_value:
        Value to fill nodata/masked pixels with. Default: ``np.nan``.

        Using NaN is generally the best approach, since many functions already know how to
        handle/propagate NaNs, or have NaN-equivalents (``dask.array.nanmean`` vs ``dask.array.mean``,
        for example). However, NaN requires a floating-point ``dtype``. If you know the data can
        be represented in a smaller data type (like ``uint16``), using a different ``fill_value``
        (like 0) and managing it yourself could save a lot of memory.
    rescale:
        Whether to rescale pixel values by the scale and offset set on the dataset.
        Default: True. Note that this could produce floating-point data when the
        original values are ints, so set ``dtype`` accordingly. You will NOT be warned
        if the cast to ``dtype`` is losing information!
    sortby_date:
        Whether to pre-sort the items by date (from the ``properties["datetime"]`` field).
        One of ``"asc"``, ``"desc"``, or False to disable sorting. Default: ``"asc"``.
        Note that if you set ``sortby_date=False``, selecting date ranges from the DataArray
        (like ``da.loc["2020-01":"2020-04"]``) may behave strangely, because xarray/pandas
        needs indexes to be sorted.
    xy_coords:
        Whether to add geospatial coordinate labels for the ``x`` and ``y`` dimensions of the DataArray,
        allowing for spatial indexing. The coordinates will be in the coordinate reference system given
        by ``epsg``

        If ``"center"`` (default), the coordinates are for each pixel's centroid, following xarray conventions.

        If ``"topleft"``, the coordinates are for each pixel's upper-left corner.

        If False, ``x`` and ``y`` will just be indexed by row/column numbers, saving a small amount of time
        and local memory.
    properties:
        Which fields from each STAC Item's ``properties`` to add as coordinates to the DataArray, indexing the "time"
        dimension.

        If None (default), all properties will be used. If a string or sequence of strings, only those fields
        will be used. For each Item missing a particular field, its value for that Item will be None.

        If False, no properties will be added.
    band_coords:
        Whether to include Asset-level metadata as coordinates for the ``bands`` dimension.

        If True (default), for each asset ID, the field(s) that have the same value across all Items
        will be added as coordinates.

        The ``eo:bands`` field is also unpacked if present, and ``sar:polarizations`` is renamed to
        ``polarization`` for convenience.
    gdal_env:
        Advanced use: a `~.LayeredEnv` of GDAL configuration options to use while opening
        and reading datasets. If None (default), `~.DEFAULT_GDAL_ENV` is used.
        See ``rio_reader.py`` for notes on why these default options were chosen.
    reader:
        Advanced use: the `~.Reader` type to use. Currently there is only one real reader type:
        `~.AutoParallelRioReader`. However, there's also `~.FakeReader` (which doesn't read data at all,
        just returns random numbers), which can be helpful for isolating whether performace issues are
        due to IO and GDAL, or inherent to dask.

    Returns
    -------
    xarray.DataArray:
        xarray DataArray, backed by a Dask array. No IO will happen until calling ``.compute()``,
        or accessing ``.values``. The dimensions will be ``("time", "band", "y", "x")``.

        ``time`` will be equal in length to the number of items you pass in, and indexed by date. Note that this means
        multiple entries could have the same index.

        ``band`` will be equal in length to the number of asset IDs used (see the ``assets`` parameter for more).

        The size of ``y`` and ``x`` will be determined by ``resolution`` and ``bounds``, which in many cases are
        automatically computed from the items you pass in.
    """
    plain_items = items_to_plain(items)

    if sortby_date is not False:
        plain_items = sorted(
            plain_items,
            key=lambda item: item["properties"].get("datetime", ""),
            reverse=sortby_date == "desc",
        )  # type: ignore

    asset_table, spec, asset_ids, plain_items = prepare_items(
        plain_items,
        assets=assets,
        epsg=epsg,
        resolution=resolution,
        bounds=bounds,
        bounds_latlon=bounds_latlon,
        snap_bounds=snap_bounds,
    )
    arr = items_to_dask(
        asset_table,
        spec,
        chunksize=chunksize,
        dtype=dtype,
        resampling=resampling,
        fill_value=fill_value,
        rescale=rescale,
        reader=reader,
        gdal_env=gdal_env,
    )

    return xr.DataArray(
        arr,
        *to_coords(
            plain_items,
            asset_ids,
            spec,
            xy_coords=xy_coords,
            properties=properties,
            band_coords=band_coords,
        ),
        attrs=to_attrs(spec),
        name="stackstac-" + dask.base.tokenize(arr)
    )
