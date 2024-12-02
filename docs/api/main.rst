API reference
-------------

.. currentmodule:: stackstac

Core
~~~~

This is what it's all about:

.. autosummary::
    :toctree: main
    :nosignatures:

    stack

Visualization
~~~~~~~~~~~~~

Easily render your data on an :doc:`ipyleaflet <ipyleaflet:index>` map. Requires installing ``'stackstac[viz]'``, and running with a :doc:`dask distributed <distributed:index>` cluster.

As you pan around the map, the part of the array that's in view is computed on the fly by dask.

**Warning**: zooming in and out will *not* load underlying data at different resolutions. Assets will be loaded at whatever resolution you initially got from `~.stack`. If your array is large and high-resolution, zooming out could trigger very slow (and expensive) computations.


.. autosummary::
    :toctree: main
    :nosignatures:

    show
    add_to_map

Operations
~~~~~~~~~~

Utilities to work with geospatial DataArrays produced by stackstac.

**Warning**: these functions may be reorganized in the near future.

.. autosummary::
    :toctree: main
    :nosignatures:

    mosaic
    reproject_array
    xyztile_of_array
    array_bounds
    array_crs
