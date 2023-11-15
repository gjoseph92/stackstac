import json

import pytest
import xarray as xr
import numpy as np

from stackstac.coordinates import (
    items_to_band_coords,
    items_to_band_coords,
    rename_some_band_fields,
)
from stackstac.coordinates_utils import scalar_sequence


@pytest.fixture
def landsat_c2_l2_json():
    with open("stackstac/tests/items-landsat-c2-l2.json") as f:
        return json.load(f)


def test_band_coords(landsat_c2_l2_json):
    ids = ["red", "green", "qa_pixel", "qa_radsat"]
    # coords = items_to_band_coords(landsat_c2_l2_json, ids)
    coords = items_to_band_coords(landsat_c2_l2_json, ids, skip_fields=set())

    # Note that we intentionally keep some coordinates that would normally be dropped,
    # since they're handy for testing

    # 0D coordinate
    type = coords["type"]
    assert isinstance(type, xr.Variable)
    assert type.shape == ()
    assert type.item() == "image/tiff; application=geotiff; profile=cloud-optimized"

    # 1D coordinate along bands
    title = coords["title"]
    assert isinstance(title, xr.Variable)
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
    # NOTE: for backwards compatibility, we de-prefix `eo:bands`.
    # I'd like to remove this; `eo:bands` doesn't deserve more special
    # treatment than `raster:bands`, for instance.
    assert "eo:bands" not in coords
    assert "description" in coords
    common_name = coords["common_name"]
    assert isinstance(common_name, xr.Variable)
    assert common_name.dims == ("band",)
    assert (common_name == ["red", "green", None, None]).all()

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


def test_band_coords_type_promotion():
    stac = [
        {
            "assets": {
                "foo": {
                    "complex_field": 0,
                    "numeric": 0,
                    "mixed_numeric_object": 0,
                    "mixed_numeric_str": 0,
                }
            }
        },
        {
            "assets": {
                "foo": {
                    "complex_field": 2 - 4.0j,
                    "numeric": None,
                    "mixed_numeric_object": [],
                    "mixed_numeric_str": "baz",
                    "partially_missing_str": "woof",
                    "partially_missing_numeric": -1,
                    "partially_missing_object": {},
                }
            }
        },
    ]
    coords = items_to_band_coords(stac, ["foo"])

    complex = coords["complex_field"]
    assert isinstance(complex, xr.Variable)
    assert complex.dtype.kind == "c"
    assert (complex == [0, 2 - 4.0j]).all()

    numeric = coords["numeric"]
    assert isinstance(numeric, xr.Variable)
    assert numeric.dtype == np.dtype(float)
    assert numeric[0] == 0
    assert np.isnan(numeric[1])

    mixed_numeric_object = coords["mixed_numeric_object"]
    assert isinstance(mixed_numeric_object, xr.Variable)
    assert mixed_numeric_object.dtype == np.dtype(object)
    assert (mixed_numeric_object == [0, []]).all()

    mixed_numeric_str = coords["mixed_numeric_str"]
    assert isinstance(mixed_numeric_str, xr.Variable)
    assert mixed_numeric_str.dtype == np.dtype(object)
    assert (mixed_numeric_str == np.array([0.0, "baz"], dtype=object)).all()
    # ^ without explicitly specifying `dtype=object`, NumPy would weirdly have
    # turned our list into a string array (`<U32`), which would fail equality.

    partially_missing_str = coords["partially_missing_str"]
    assert isinstance(partially_missing_str, xr.Variable)
    assert partially_missing_str.dtype == np.dtype(object)
    assert (partially_missing_str == [None, "woof"]).all()

    partially_missing_numeric = coords["partially_missing_numeric"]
    assert isinstance(partially_missing_numeric, xr.Variable)
    assert partially_missing_numeric.dtype == np.dtype(float)
    assert np.isnan(partially_missing_numeric[0])
    assert partially_missing_numeric[1] == -1

    partially_missing_object = coords["partially_missing_object"]
    assert isinstance(partially_missing_object, xr.Variable)
    assert partially_missing_object.dtype == np.dtype(object)
    assert (partially_missing_object == [None, {}]).all()


@pytest.mark.parametrize(
    "input, expected",
    [
        ("sar:polarizations", "polarization"),
        ("eo:bands_common_name", "common_name"),
        ("eo:bands_foobar", "foobar"),
        ("sar:frequency_band", "sar:frequency_band"),
        ("eo:cloud_cover", "eo:cloud_cover"),
        (
            "somethingsomething_sar:polarizations",
            "somethingsomething_sar:polarizations",
        ),
        ("somethingsomething_eo:bands_name", "somethingsomething_eo:bands_name"),
    ],
)
def test_rename_band_fields(input: str, expected: str):
    assert rename_some_band_fields(input) == expected
