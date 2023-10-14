import json

import pytest

from stackstac.coordinates import (
    items_to_band_coords,
    items_to_band_coords2,
)


@pytest.fixture
def stac_json():
    with open("stackstac/tests/items.json") as f:
        return json.load(f)


def test_band_coords(stac_json):
    ids = ["red", "green", "qa_pixel", "qa_radsat"]
    coords = items_to_band_coords(stac_json, ids)
    c2 = items_to_band_coords2(stac_json, ids)
    print(coords)
    print(c2)
