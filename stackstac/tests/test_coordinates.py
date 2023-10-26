import json

import pytest
import xarray as xr
import numpy as np

from stackstac.coordinates import (
    items_to_band_coords,
    items_to_band_coords2,
    items_to_band_coords_simple,
    items_to_band_coords_locality,
)
from stackstac.coordinates_utils import scalar_sequence


@pytest.fixture
def landsat_c2_l2_json():
    with open("stackstac/tests/items-landsat-c2-l2.json") as f:
        return json.load(f)


def test_band_coords(landsat_c2_l2_json):
    ids = ["red", "green", "qa_pixel", "qa_radsat"]
    # coords = items_to_band_coords(landsat_c2_l2_json, ids)
    coords = items_to_band_coords_locality(landsat_c2_l2_json, ids)
    # print(coords)

    # 0D coordinate
    type = coords["type"]
    assert isinstance(type, xr.Variable)
    assert type.shape == ()
    assert type.item() == "image/tiff; application=geotiff; profile=cloud-optimized"
    assert np.issubdtype(type.dtype, str)  # shouldn't be `O`, should be `U56`

    # 1D coordinate along bands
    title = coords["title"]
    assert isinstance(title, xr.Variable)
    assert np.issubdtype(title.dtype, str)  # also shouldn't be `O`
    assert (
        title
        == [
            "Red Band",
            "Green Band",
            "Pixel Quality Assessment Band",
            "Radiometric Saturation and Terrain Occlusion Quality Assessment Band",
        ]
    ).all()

    # 1D coordinate along bands, where each element is a variable-length list
    roles = coords["roles"]
    assert isinstance(roles, xr.Variable)
    assert roles.shape == (len(ids),)
    # `roles` is an array of lists:
    # <xarray.Variable (band: 4)>
    # array([list(['data', 'reflectance']), list(['data', 'reflectance']),
    #        list(['cloud', 'cloud-shadow', 'snow-ice', 'water-mask']),
    #        list(['saturation'])], dtype=object)
    #
    # Actually working with this in xarray is awkward. I'm not sure how users
    # would ever usefully interact with this besides printing it, because
    # just an equality check requires wrapping your list in `scalar_sequence`,
    # and if you wanted to do some sort of set operation (only 'data' roles, say),
    # I don't think it's even possible in xarray.
    assert roles[0] == scalar_sequence(["data", "reflectance"])
    assert roles[2] == scalar_sequence(
        ["cloud", "cloud-shadow", "snow-ice", "water-mask"]
    )
    assert roles[3] == scalar_sequence(["saturation"])

    # Here's a 2D coordinate along both time and band
    href = coords["href"]
    assert isinstance(href, xr.Variable)
    assert href.dims == ("time", "band")
    assert href.shape == (len(landsat_c2_l2_json), len(ids))

    # `eo:bands` should be unpacked
    # TODO: for backwards compatibility, we should de-prefix `eo:bands` and `sar:`
    assert "eo:bands" not in coords
    assert "eo:bands_description" in coords
    common_name = coords["eo:bands_common_name"]
    assert isinstance(common_name, xr.Variable)
    assert common_name.dims == ("band",)
    assert (common_name == ['red', 'green', None, None]).all()

    # `raster:bands` is also unpacked
    assert "raster:bands" not in coords
    data_type = coords["raster:bands_data_type"]
    assert isinstance(data_type, xr.Variable)
    assert data_type.shape == ()
    assert data_type == "uint16"

    # missing values in `raster:bands_unit` are imputed with None
    unit = coords["raster:bands_unit"]
    assert isinstance(unit, xr.Variable)
    assert (unit == [None, None, "bit index", "bit index"]).all()

    # `classification:bitfields` contains scalar sequences of dicts.
    # Again, quite awkward to work with in xarray, but at least it's properly there?
    bitfields = coords["classification:bitfields"]
    assert isinstance(bitfields, xr.Variable)
    assert bitfields.dims == ("band",)
    assert bitfields[0] == None
    c2 = bitfields[2].item()
    assert isinstance(c2, list)
    assert len(c2) == 12
    assert all([isinstance(c, dict) for c in c2])


@pytest.mark.parametrize(
    "func",
    [
        items_to_band_coords,
        items_to_band_coords2,
        items_to_band_coords_simple,
        items_to_band_coords_locality
    ],
)
def test_benchmark_band_coords(func, landsat_c2_l2_json, benchmark):
    ids = [
        "ang",
        "red",
        "blue",
        "green",
        "nir08",
        "swir16",
        "swir22",
        "coastal",
        "mtl.txt",
        "mtl.xml",
        "mtl.json",
        "qa_pixel",
        "qa_radsat",
        "qa_aerosol",
        "tilejson",
        "rendered_preview",
    ]
    bigger = landsat_c2_l2_json * 10  # 200 items
    benchmark(func, bigger, ids)
    # print(benchmark.stats)
