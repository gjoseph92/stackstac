import json
from stackstac import accumulate_metadata


with open("stackstac/tests/test_stac_items.json") as json_data:
    stac_items = json.load(json_data)

properties_coords = accumulate_metadata.accumulate_properties_coords(
    stac_items, coords={}
)
assert (
    properties_coords["view:off_nadir"] == 0
), "the coord 'view:off_nadir' is not constant"
assert properties_coords["datetime"].dims == (
    "time",
), "the coord 'time' should vary along the time axis only"

assets_coords = accumulate_metadata.accumulate_assets_coords(
    stac_items, asset_ids=["B2", "B3", "B4"], coords={}
)
assert (
    assets_coords["type"] == "image/vnd.stac.geotiff; cloud-optimized=true"
), "the coord 'type' is not constant"
assert assets_coords["id2"].dims == (
    "time",
), "the coord 'id2' should vary along the time axis only"
assert assets_coords["title"].dims == (
    "band",
), "the coord 'title' should vary along the band axis only"
assert assets_coords["href"].dims == (
    "time",
    "band",
), "the property 'href' should vary along both band and time axis"
