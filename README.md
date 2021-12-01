# StackSTAC

[![Documentation Status](https://readthedocs.org/projects/stackstac/badge/?version=latest)](https://stackstac.readthedocs.io/en/latest/?badge=latest) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gjoseph92/stackstac/main?labpath=%2Fdocs%2Fbasic.ipynb%3Ffile-browser-path%3D%2Fexamples)

Turn a list of [STAC](http://stacspec.org) items into a 4D [xarray](http://xarray.pydata.org/en/stable/) DataArray (dims: `time, band, y, x`), including reprojection to a common grid. The array is a lazy [Dask array](https://docs.dask.org/en/latest/array.html), so loading and processing the data in parallel—locally or [on a cluster](https://coiled.io/)—is just a `compute()` call away.

For more information and examples, please [see the documentation](https://stackstac.readthedocs.io).

```python
import stackstac
import pystac_client

URL = "https://earth-search.aws.element84.com/v0"
catalog = pystac_client.Client.open(URL)

stac_items = catalog.search(
    intersects=dict(type="Point", coordinates=[-105.78, 35.79]),
    collections=["sentinel-s2-l2a-cogs"],
    datetime="2020-04-01/2020-05-01"
).get_all_items()

stack = stackstac.stack(stac_items)
print(stack)
```
```
<xarray.DataArray 'stackstac-fefccf3d6b2f9922dc658c114e79865b' (time: 13, band: 17, y: 10980, x: 10980)>
dask.array<fetch_raster_window, shape=(13, 17, 10980, 10980), dtype=float64, chunksize=(1, 1, 1024, 1024), chunktype=numpy.ndarray>
Coordinates: (12/24)
  * time                        (time) datetime64[ns] 2020-04-01T18:04:04 ......
    id                          (time) <U24 'S2B_13SDV_20200401_0_L2A' ... 'S...
  * band                        (band) <U8 'overview' 'visual' ... 'WVP' 'SCL'
  * x                           (x) float64 4e+05 4e+05 ... 5.097e+05 5.098e+05
  * y                           (y) float64 4e+06 4e+06 ... 3.89e+06 3.89e+06
    view:off_nadir              int64 0
    ...                          ...
    instruments                 <U3 'msi'
    created                     (time) <U24 '2020-09-05T06:23:47.836Z' ... '2...
    sentinel:sequence           <U1 '0'
    sentinel:grid_square        <U2 'DV'
    title                       (band) object None ... 'Scene Classification ...
    epsg                        int64 32613
Attributes:
    spec:        RasterSpec(epsg=32613, bounds=(399960.0, 3890220.0, 509760.0...
    crs:         epsg:32613
    transform:   | 10.00, 0.00, 399960.00|\n| 0.00,-10.00, 4000020.00|\n| 0.0...
    resolution:  10.0
```

Once in xarray form, many operations become easy. For example, we can compute a low-cloud weekly mean-NDVI timeseries:

```python
lowcloud = stack[stack["eo:cloud_cover"] < 40]
nir, red = lowcloud.sel(band="B08"), lowcloud.sel(band="B04")
ndvi = (nir - red) / (nir + red)
weekly_ndvi = ndvi.resample(time="1w").mean(dim=("time", "x", "y")).rename("NDVI")
# Call `weekly_ndvi.compute()` to process ~25GiB of raster data in parallel. Might want a dask cluster for that!
```

## Installation

```
pip install stackstac
```

Windows notes:

It's a good idea to use `conda` to handle installing rasterio on Windows. There's considerably more pain involved with GDAL-type installations using pip. Then `pip install stackstac`.

## Things `stackstac` does for you:

* Figure out the geospatial parameters from the STAC metadata (if possible): a coordinate reference system, resolution, and bounding box.
* Transfer the STAC metadata into [xarray coordinates](http://xarray.pydata.org/en/stable/data-structures.html#coordinates) for easy indexing, filtering, and provenance of metadata.
* Efficiently generate a Dask graph for loading the data in parallel.
* Mediate between Dask's parallelism and GDAL's aversion to it, allowing for fast, multi-threaded reads when possible, and at least preventing segfaults when not.
* Mask nodata and rescale by dataset-level scales/offsets.
* Display data in interactive maps in a notebook, computed on the fly by Dask.

## Limitations:

* **Raster data only!** We are currently ignoring other types of assets you might find in a STAC (XML/JSON metadata, point clouds, video, etc.).
* **Single-band raster data only!** Each band has to be a separate STAC asset—a separate `red`, `green`, and `blue` asset on each Item is great, but a single `RGB` asset containing a 3-band GeoTIFF is not supported yet.
* [COG](https://www.cogeo.org)s work best. "Normal" GeoTIFFs that aren't internally tiled, or don't have overviews, will see much worse performance. Sidecar files (like `.msk` files) are ignored for performance. JPEG2000 will probably work, but probably be slow unless you buy kakadu. [Formats make a big difference](https://medium.com/@_VincentS_/do-you-really-want-people-using-your-data-ec94cd94dc3f).
* BYOBlocksize. STAC doesn't offer any metadata about the internal tiling scheme of the data. Knowing it can make IO more efficient, but actually reading the data to figure it out is slow. So it's on you to set this parameter. (But if you don't, things should be fine for any reasonable COG.)
* Doesn't make geospatial data any easier to work with in xarray. Common operations (picking bands, clipping to bounds, etc.) are tedious to type out. Real geospatial operations (shapestats on a GeoDataFrame, reprojection, etc.) aren't supported at all. [rioxarray](https://corteva.github.io/rioxarray/stable/readme.html) might help with some of these, but it has limited support for Dask, so be careful you don't kick off a huge computation accidentally.
* I haven't even written tests yet! Don't use this in production. Or do, I guess. Up to you.

## Roadmap:

Short-term:

- Write tests and add CI (including typechecking)
- Support multi-band assets
- Easier access to `s3://`-style URIs (right now, you'll need to pass in `gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(always=dict(session=rio.session.AWSSession(...)))`)
- Utility to guess blocksize (open a few assets)
- Support [item assets](https://github.com/radiantearth/stac-spec/tree/master/extensions/item-assets) to provide more useful metadata with collections that use it (like S2 on AWS)
- Rewrite dask graph generation once the [Blockwise IO API](https://github.com/dask/dask/pull/7281) settles

Long term (if anyone uses this thing):
- Support other readers ([aiocogeo](https://github.com/geospatial-jeff/aiocogeo)?) that may perform better than GDAL for specific formats
- Interactive mapping with [xarray_leaflet](https://github.com/davidbrochart/xarray_leaflet), made performant with some Dask graph-rewriting tricks to do the initial IO at coarser resolution for lower zoom levels (otherwize zooming out could process terabytes of data)
- Improve ergonomics of xarray for raster data (in collaboration with [rioxarray](https://corteva.github.io/rioxarray/stable/readme.html))
- Implement core geospatial routines (warp, vectorize, vector stats, [GeoPandas](https://geopandas.org)/[spatialpandas](https://github.com/holoviz/spatialpandas) interop) in Dask
