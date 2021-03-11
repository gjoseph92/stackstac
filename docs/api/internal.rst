Internal API reference
----------------------

You shouldn't need to look here unless you're curious, but while this library is still so not-production-ready, you might need to understand some of these internal types for now.

.. currentmodule:: stackstac

.. autosummary::
    :toctree: internal

    rio_env.LayeredEnv
    raster_spec.RasterSpec
    reader_protocol.Reader
    rio_reader.AutoParallelRioReader
    rio_reader.ThreadLocalRioDataset
    rio_reader.SingleThreadedRioDataset
    reader_protocol.FakeReader
