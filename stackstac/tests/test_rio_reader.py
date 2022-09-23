import tempfile

import numpy as np
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import Window
from stackstac.raster_spec import RasterSpec
from stackstac.rio_reader import AutoParallelRioReader


def test_dataset_read_with_gcps():
    """
    Ensure that GeoTIFFs with ground control points (gcps) can be read using
    AutoParallelRioReader.

    Regression test for https://github.com/gjoseph92/stackstac/issues/181.
    """
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmpfile:
        src_gcps = [
            GroundControlPoint(row=0, col=0, x=156113, y=2818720, z=0),
            GroundControlPoint(row=0, col=800, x=338353, y=2785790, z=0),
            GroundControlPoint(row=800, col=800, x=297939, y=2618518, z=0),
            GroundControlPoint(row=800, col=0, x=115698, y=2651448, z=0),
        ]
        crs = CRS.from_epsg(32618)
        with rasterio.open(
            tmpfile.name,
            mode="w",
            height=800,
            width=800,
            count=1,
            dtype=np.uint8,
            driver="GTiff",
        ) as source:
            source.gcps = (src_gcps, crs)

        reader = AutoParallelRioReader(
            url=tmpfile.name,
            spec=RasterSpec(
                epsg=4326, bounds=(90, -10, 110, 10), resolutions_xy=(10, 10)
            ),
            resampling=rasterio.enums.Resampling.bilinear,
            dtype=np.float32,
            fill_value=np.nan,
            rescale=True,
        )
        array = reader.read(window=Window(col_off=0, row_off=0, width=10, height=10))

    np.testing.assert_allclose(
        actual=array, desired=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    )
