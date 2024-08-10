# Changelog

## 0.5.1 (2024-08-09)
- Support NumPy 2.0 [@floriandeboissieu](https://github.com/floriandeboissieu) [@gjoseph92](https://github.com/gjoseph92)
- Require pandas 2.0, fix `'infer_datetime_format' is deprecated` warning, fix NaT values in time dimension [@gjoseph92](https://github.com/gjoseph92) [bastien-k](https://github.com/bastien-k) [clausmichele](https://github.com/clausmichele) [J6767](https://github.com/J6767)
- Fix `Unexpected shape` error with `errors_as_nodata` [@gjoseph92](https://github.com/gjoseph92) [fkroeber](https://github.com/fkroeber)
- Fix error when generating error message for multi-band rasters [yellowcap](https://github.com/yellowcap)
- Remove annoying `Skipping VRT` log statement [@gjoseph92](https://github.com/gjoseph92)

## 0.5.0 (2023-09-19)
- **Breaking change in specific scenarios:** `rescale=True` now uses STAC metadata from `raster:bands`, instead of metadata set per GeoTIFF [@RSchueder](https://github.com/RSchueder)
- Fix black stripes when mosaicing (introduced in 0.4.4) [@gjoseph92](https://github.com/gjoseph92) [@lhcorralo](https://github.com/lhcorralo)
- Update examples to point to newer Sentinel-2 endpoint on AWS [@ljstrnadiii](https://github.com/ljstrnadiii)
- Ensure `jupyter-server-proxy` is installed with `pip install 'stackstac[viz]'` [@gjoseph92](https://github.com/gjoseph92)

## 0.4.4 (2023-06-21)
- Resolve compatibility with NumPy >= 1.24.0 [@gjoseph92](https://github.com/gjoseph92)
- Fix TypeError when no items overlap with the given `bounds` [@gjoseph92](https://github.com/gjoseph92)
- Fix scale and offset application when only one is available [@sharkinspatial](https://github.com/sharkinsspatial)
- Change timer logging to DEBUG level [@jorge-cervest](https://github.com/jorge-cervest)

## 0.4.3 (2022-09-14)
- Support sequences of `Item`s [@gjoseph92](https://github.com/gjoseph92)
- Fix compatibility with `pyproj>=3.4.0` [@gjoseph92](https://github.com/gjoseph92)
- stackstac has switched from Poetry to PDM as its package manager. This does not affect users, only developers.

## 0.4.2 (2022-07-06)
- Support (and require) rasterio 1.3.0 [@carderne](https://github.com/carderne) [@gjoseph92](https://github.com/gjoseph92)
- Fix `ValueError` when passing `dim=` to `mosaic` [@aazuspan](https://github.com/aazuspan)

## 0.4.1 (2022-04-15)
- Use `pd.Index` instead of deprecated `pd.Float64Index` [@TomAugspurger](https://github.com/TomAugspurger)
- Better error when forgetting a custom nodata value to `mosaic` [@gjoseph92](https://github.com/gjoseph92)

## 0.4.0 (2022-03-16)
- Support specifying a chunk pattern for the `time`/`band` dimensions, allowing you to load multiple items in one spatial chunk (like `stackstac.stack(..., chunksize=(-1, 1, 512, 512))`). This can moderately to significantly decrease memory usage and improve performance when creating composites (like `stack(..., chunksize=(-1, 1, "128MiB", "128MiB")).median("time")`). See [#116 (comment)](https://github.com/gjoseph92/stackstac/pull/116#issuecomment-1027606996) for details.
- `stackstac.mosaic` generates a more efficient dask graph, with hopefully lower memory usage [@gjoseph92](https://github.com/gjoseph92)
- Clearer errors when versions of `pystac` or `satstac` are incompatible [@gjoseph92](https://github.com/gjoseph92)
- Support newer versions of `xarray` using [CalVer](https://github.com/pydata/xarray/issues/6176) [@scottyhq](https://github.com/scottyhq)

## 0.3.1 (2022-01-20)
- Support `nodata=` argument to `stackstac.mosaic` [@TomAugspurger](https://github.com/TomAugspurger) [@gjoseph92](https://github.com/gjoseph92)

## 0.3.0 (2022-01-20)
- **Breaking change:** `fill_value=None` is no longer supported. You must always specify a `fill_value` (default is still NaN); it can no longer be inferred from the GeoTIFF files. [@gjoseph92](https://github.com/gjoseph92)
- Respect `fill_value` for array chunks that don't overlap with any Asset, instead of always using NaN [@gjoseph92](https://github.com/gjoseph92) [@TomAugspurger](https://github.com/TomAugspurger)
- Fix bugs with `stackstac.show` when the path to your notebook file had the word `notebook`, `lab`, or `voila` in it [@robintw](https://github.com/robintw)
- Support 2022 version of Dask [@gjoseph92](https://github.com/gjoseph92)
- Relax NumPy requirement, supporting any NumPy version supported by Dask [@gjoseph92](https://github.com/gjoseph92) [@scottyhq](https://github.com/scottyhq)
- Require minimum Pillow version of 9.0, due to vulnerabilities reported in older versions (in features unused by stackstac)

## 0.2.2 (2021-12-03)
- Support [pystac](https://github.com/stac-utils/pystac) ItemCollections [@TomAugspurger](https://github.com/TomAugspurger), [@scottyhq](https://github.com/scottyhq)
- Fix bug where repeated metadata values would be None [@gjoseph92](https://github.com/gjoseph92) [@scottyhq](https://github.com/scottyhq)
- Fix one-pixel shift when using `xy_coords="center"` [@gjoseph92](https://github.com/gjoseph92) [@Kirill888](https://github.com/Kirill888) [@maawoo](https://github.com/maawoo)
- Fix upper-right-hand corner calculation in `bounds_from_affine` [@g2giovanni](https://github.com/g2giovanni)
- Fix error with valid STAC items that don't have a `type` field [@scottyhq](https://github.com/scottyhq)
- Allow all file extensions (not just `.tif`) [@JamesOConnor](https://github.com/JamesOConnor) [@gjoseph92](https://github.com/gjoseph92)
- Fix `Error when attempting to bind on address ::1` with `stackstac.show` [@gjoseph92](https://github.com/gjoseph92)
- Fix empty map from `stackstac.show` due to `jupyter-server-proxy` not being installed [@robintw](https://github.com/robintw) [@gjoseph92](https://github.com/gjoseph92)
- Fix occasional `RuntimeError: Set changed size during iteration` errors while computing [@gjoseph92](https://github.com/gjoseph92)
- Use BinderHub instead of Coiled notebook for runnable examples in documentation [@gjoseph92](https://github.com/gjoseph92)

## 0.2.1 (2021-05-07)
Support [xarray 0.18](http://xarray.pydata.org/en/stable/whats-new.html#v0-18-0-6-may-2021) and beyond, as well as looser version requirements for some other dependencies.

## 0.2.0 (2021-05-05)
Call [`stackstac.show`](https://stackstac.readthedocs.io/en/latest/api/main/stackstac.show.html) to render DataArrays in interactive ipyleaflet maps in your notebook! See [this example](https://stackstac.readthedocs.io/en/latest/examples/show.html) for more.

- [`mosaic`](https://stackstac.readthedocs.io/en/latest/api/main/stackstac.mosaic.html) function
- Exposed some assorted [handy geospatial ops](https://stackstac.readthedocs.io/en/latest/api/main.html#operations)
- More robustly get band metadata (if one Item is missing some asset metadata, just ignore it instead of dropping the whole thing)
- Fixed occasional `ValueError: conflicting sizes for dimension`
- Resolved issue where timestamps became integers
- Support [pystac-client](https://github.com/stac-utils/pystac-client) ItemCollections
- Minimum Python version is now (accurately) 3.8. Stackstac would have failed upon import on 3.7 before.

## 0.1.1 (2021-04-16)
- Passing a `gdal_env` now works when using a distributed cluster (before, you got a pickle error when calling `.compute()`)
- Many typos fixed, thanks [@kylebarron](https://github.com/kylebarron) and [@RichardScottOZ](https://github.com/RichardScottOZ)!

## 0.1.0 (2021-03-10)
Initial release 🎉
