from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, Union

import affine

IntFloat = Union[int, float]
Bbox = Tuple[IntFloat, IntFloat, IntFloat, IntFloat]
Resolutions = Tuple[IntFloat, IntFloat]


@dataclass
class RasterSpec:
    """
    Spatial parameters defining the grid for a raster.
    """

    epsg: int
    bounds: Bbox
    resolutions_xy: Resolutions

    def __post_init__(self):
        xres, yres = self.resolutions_xy
        assert xres > 0, f"X resolution {xres} must be > 0"
        assert yres > 0, f"Y resolution {yres} must be > 0"

        minx, miny, maxx, maxy = self.bounds
        assert minx < maxx, f"Invalid bounds: {minx=} >= {maxx=}"
        assert miny < maxy, f"Invalid bounds: {miny=} >= {maxy=}"

    @cached_property
    def transform(self) -> affine.Affine:
        return affine.Affine(
            self.resolutions_xy[0],  # xscale
            0.0,
            self.bounds[0],  # xoff
            0.0,
            -self.resolutions_xy[1],  # yscale
            self.bounds[3],  # yoff
        )

    @cached_property
    def shape(self) -> Tuple[int, int]:
        minx, miny, maxx, maxy = self.bounds
        xres, yres = self.resolutions_xy

        # This is how GDAL rounds/snaps the calculation, so we do it too
        # https://github.com/OSGeo/gdal/blob/00615775bff0681a7fbce17eb187dcfc0e000c15/gdal/apps/gdalwarp_lib.cpp#L3394-L3399
        # (it's not quite the same as `round`)
        width = int((maxx - minx + (xres / 2)) / xres)
        height = int((maxy - miny + (yres / 2)) / yres)

        return (height, width)

    @cached_property
    def vrt_params(self) -> dict:
        height, width = self.shape
        return {
            "crs": self.epsg,
            "transform": self.transform,
            "height": height,
            "width": width,
        }
