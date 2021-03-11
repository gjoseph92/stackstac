``stackstac``: easier coud-native geoprocessing
================================================

`stackstac.stack` turns a `STAC <http://stacspec.org/>`_ collection into a lazy `xarray.DataArray`, backed by :doc:`dask <dask:array>`.

This lets you easily compute composites, mosaics, and any other sorts of fun aggregations on raster data. Run it in parallel on one machine, or distribute it across many in the cloud, using `Coiled <https://coiled.io/>`_, `Pangeo <https://pangeo.io/>`_, or your own `distributed <https://distributed.dask.org>`_ deployment.

.. Huh? What does that mean?

.. STAC—or `Spatio-Temporal Asset Catalog <http://stacspec.org/>`_—is a standard for providing metadata about geospatial data: where and when the data is from (on the Earth), what kind of data it is, and where the data files can actually be found. But once you know where the files are, how do you use them? Particularly when there are *lots* (terabytes) of data you need to process?

Things ``stackstac`` does for you:
----------------------------------

* Figure out the **geospatial parameters** from the STAC metadata (if possible): a coordinate reference system, resolution, and bounding box.
* Transfer the STAC metadata into :ref:`xarray coordinates <xarray:coordinates>` for **easy indexing**, filtering, and provenance of metadata.
* Efficiently generate a Dask graph for **loading the data in parallel**.
* Mediate between Dask's parallelism and GDAL's aversion to it, allowing for **fast, multi-threaded reads** when possible, and at least preventing segfaults when not.

Installation
------------

``stackstac`` is available on pip::

   pip install stackstac

Its main dependencies are rasterio, pyproj, dask, and xarray, all of which should pose no problems to install—no need to build GDAL from source here.

Contents
--------

.. toctree::
   :maxdepth: 1

   basic
   cluster
   api/main
   api/internal


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
