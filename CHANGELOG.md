# Changelog

## 0.2.2 (...)
- Support [pystac](https://github.com/stac-utils/pystac) ItemCollections
- Fix bug where repeated metadata values would be None
- Fix one-pixel shift when using `xy_coords="center"` [@gjoseph92](https://github.com/gjoseph92) [@Kirill888](https://github.com/Kirill888) [@maawoo](https://github.com/maawoo)

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
