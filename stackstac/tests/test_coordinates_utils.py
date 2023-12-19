from datetime import datetime
import numpy as np
import pytest

from stackstac.coordinates_utils import (
    DtypeUpdatingArray,
    deduplicate_axes,
    items_to_coords,
    unnested_items,
    unpacked_per_band_asset_fields,
)


def test_deduplicate_axes():
    a1 = np.arange(5)
    d = deduplicate_axes(a1)
    np.testing.assert_equal(d, a1)

    a1_d0 = np.repeat(1, 5)
    d = deduplicate_axes(a1_d0)
    assert d.shape == (1,)
    np.testing.assert_equal(d, [1])

    a2 = np.arange(3 * 4).reshape(3, 4)
    d = deduplicate_axes(a2)
    np.testing.assert_equal(d, a2)

    a2_d0 = np.stack([np.arange(4)] * 3)
    d = deduplicate_axes(a2_d0)
    assert d.shape == (1, a2_d0.shape[1])
    np.testing.assert_equal(d, a2_d0[[0]])

    a2_d1 = a2_d0.T
    d = deduplicate_axes(a2_d1)
    assert d.shape == (a2_d1.shape[0], 1)
    np.testing.assert_equal(d, a2_d1[:, [0]])

    a2_d01 = np.broadcast_to(1, (3, 4))
    d = deduplicate_axes(a2_d01)
    assert d.shape == (1, 1)
    np.testing.assert_equal(d, np.broadcast_to(1, (1, 1)))


def test_deduplicate_axes_nan():
    a2_d0 = np.stack([np.array([1, np.nan, 2])] * 3)
    d = deduplicate_axes(a2_d0)
    assert d.shape == (1, a2_d0.shape[1])
    np.testing.assert_equal(d, a2_d0[[0]])

    a2_d1 = a2_d0.T
    d = deduplicate_axes(a2_d1)
    assert d.shape == (a2_d1.shape[0], 1)
    np.testing.assert_equal(d, a2_d1[:, [0]])


@pytest.mark.parametrize(
    "input, expected",
    [
        # Unchanged, no nesting
        ({"a": 1, "b": "foo"}, {"a": 1, "b": "foo"}),
        # Single level nesting
        ({"a": 1, "b": {"a": "foo"}}, {"a": 1, "b_a": "foo"}),
        # Single level nesting, multiple subkeys
        ({"a": 1, "b": {"a": "foo", "b": "bar"}}, {"a": 1, "b_a": "foo", "b_b": "bar"}),
        (
            # Double level nesting
            {"a": 1, "b": {"a": "foo", "b": {"x": 0}}},
            {"a": 1, "b_a": "foo", "b_b_x": 0},
        ),
        # Empty dicts are preserved
        ({"a": {}, "b": {"c": {}}}, {"a": {}, "b_c": {}}),
    ],
)
def test_unnested_items(input, expected):
    assert list(unnested_items(input.items())) == list(expected.items())


@pytest.mark.parametrize(
    "input, fields, expected",
    [
        # No fields
        ({"a": None}, [], {"a": None}),
        # `None` value is dropped
        ({"a": None}, ["a"], {}),
        # Not a sequence
        ({"a": 1}, ["a"], {"a": 1}),
        # Dropped: not 1-length
        ({"a": []}, ["a"], {}),
        # Unpacked: 1-length
        ({"a": [1]}, ["a"], {"a": 1}),
        # Dropped: not 1-length
        ({"a": [1, 2]}, ["a"], {}),
        # No fields match
        ({"a": None}, ["b"], {"a": None}),
        # No fields match
        ({"a": [1]}, ["b"], {"a": [1]}),
        # Unpacked: 1-length, with extraneous fields
        ({"a": [1]}, ["a", "b"], {"a": 1}),
        # Dropped: not 1-length, with extraneous fields
        ({"a": [1, 2]}, ["a", "b"], {}),
        # Multiple fields: unpacked, not sequence
        ({"a": [1], "b": 2}, ["a", "b"], {"a": 1, "b": 2}),
        # Multiple fields: unpacked, dropped
        ({"a": [1], "b": ()}, ["a", "b"], {"a": 1}),
        # Multiple fields: unpacked, not matched
        ({"a": [1], "c": ()}, ["a", "b"], {"a": 1, "c": ()}),
        # Multiple fields: unpacked, unpacked
        ({"a": [1], "b": (2,)}, ["a", "b"], {"a": 1, "b": 2}),
    ],
)
def test_unpacked_per_band_asset_fields(input, fields, expected):
    result = list(unpacked_per_band_asset_fields(input.items(), fields))
    assert result == list(expected.items())


def test_items_to_coords_3d_same_bands():
    # 3D coordinates don't actually happen in stackstac, but this is a nice stress test that the logic is correct.
    data = [
        {
            "smallsat": {
                "red": {
                    "type": "geotiff",
                    "cloudy": True,
                },
                "nir": {
                    "type": "geotiff",
                    "cloudy": True,
                },
            },
            "bigsat": {
                "red": {
                    "type": "geotiff",
                    "cloudy": False,
                },
                "nir": {
                    "type": "geotiff",
                    "cloudy": False,
                },
            },
        },
        {
            "smallsat": {
                "red": {
                    "type": "geotiff",
                    "cloudy": True,
                },
                "nir": {
                    "type": "geotiff",
                    "cloudy": True,
                },
            },
            "bigsat": {
                "red": {
                    "type": "geotiff",
                    "cloudy": True,
                },
                "nir": {
                    "type": "geotiff",
                    "cloudy": True,
                },
            },
        },
        {
            "smallsat": {
                "red": {
                    "type": "geotiff",
                    "cloudy": False,
                },
                "nir": {
                    "type": "geotiff",
                    "cloudy": False,
                },
            },
            "bigsat": {
                "red": {
                    "type": "geotiff",
                    # "cloudy": False,
                },
                "nir": {
                    "type": "geotiff",
                    # "cloudy": False,
                },
            },
        },
    ]

    coords = items_to_coords(
        (
            ((i, j, k), field, value)
            for i, item in enumerate(data)
            for j, (sat_key, assets) in enumerate(item.items())
            for k, (asset_id, asset) in enumerate(assets.items())
            for field, value in asset.items()
        ),
        shape=(len(data), 2, 2),
        dims=("time", "platform", "band"),
    )

    cloudy = coords["cloudy"]
    assert cloudy.dims == ("time", "platform")
    np.testing.assert_equal(
        cloudy.values,
        [
            [True, False],
            [True, True],
            [False, None],
        ],
    )

    # TODO because there's no cloud mask band for smallsat,
    # it's None, which I suppose is correct/fair, but slightly annoying.
    type_ = coords["type"]
    assert type_.dims == ()
    assert type_ == "geotiff"


def test_items_to_coords_3d_different_bands():
    # similar to above, but `bigsat` has a band that `smallsat` doesn't.
    data = [
        {
            "smallsat": {
                "red": {
                    "desc": "red-ish",
                    "resolution": 15,
                    "id": "smallsat-red-00",
                },
                "nir": {
                    "desc": "near infrared",
                    "resolution": 30,
                    "id": "smallsat-nir-00",
                },
            },
            "bigsat": {
                "red": {
                    "desc": "red-ish",
                    "resolution": 5,
                    "id": "bigsat-red-00",
                },
                "nir": {
                    "desc": "near infrared",
                    "resolution": 10,
                    "id": "bigsat-nir-00",
                },
                "cloud": {
                    "desc": "cloud mask",
                    "id": "bigsat-cloud-00",
                },
            },
        },
        {
            "smallsat": {
                "red": {
                    "desc": "red-ish",
                    "resolution": 15,
                    "id": "smallsat-red-01",
                },
                "nir": {
                    "desc": "near infrared",
                    "resolution": 30,
                    "id": "smallsat-nir-01",
                },
            },
            "bigsat": {
                "red": {
                    "desc": "red-ish",
                    "resolution": 5,
                    "id": "bigsat-red-01",
                },
                "nir": {
                    "desc": "near infrared",
                    "resolution": 10,
                    "id": "bigsat-nir-01",
                },
                "cloud": {
                    "desc": "cloud mask",
                    "id": "bigsat-cloud-01",
                },
            },
        },
        {
            "smallsat": {
                "red": {
                    "desc": "red-ish",
                    "resolution": 15,
                    "id": "smallsat-red-02",
                },
                "nir": {
                    "desc": "near infrared",
                    "resolution": 30,
                    "id": "smallsat-nir-02",
                },
            },
            "bigsat": {
                "red": {
                    "desc": "red-ish",
                    "resolution": 5,
                    "id": "bigsat-red-02",
                },
                "nir": {
                    "desc": "near infrared",
                    "resolution": 10,
                    "id": "bigsat-nir-02",
                },
                "cloud": {
                    "desc": "cloud mask",
                    "id": "bigsat-cloud-02",
                },
            },
        },
    ]

    coords = items_to_coords(
        (
            ((i, j, k), field, value)
            for i, item in enumerate(data)
            for j, (sat_key, assets) in enumerate(item.items())
            for k, (asset_id, asset) in enumerate(assets.items())
            for field, value in asset.items()
        ),
        shape=(len(data), 2, 3),
        dims=("time", "platform", "band"),
    )

    assert coords.keys() == {"desc", "resolution", "id"}

    ids = coords["id"]
    assert ids.dims == ("time", "platform", "band")
    assert ids.shape == (len(data), 2, 3)
    assert ids[0, 0, 0] == "smallsat-red-00"
    assert ids[-1, -1, -1] == "bigsat-cloud-02"

    desc = coords["desc"]
    assert desc.dims == ("platform", "band")
    np.testing.assert_equal(
        desc.values,
        [
            ["red-ish", "near infrared", None],
            ["red-ish", "near infrared", "cloud mask"],
        ],
    )

    resolution = coords["resolution"]
    assert resolution.dims == ("platform", "band")
    np.testing.assert_equal(
        resolution.values,
        [
            [15, 30, np.nan],
            [5, 10, np.nan],
        ],
    )


@pytest.mark.parametrize(
    "values, expected, dtype",
    [
        # NOTE: the array will always be initialized to size 4
        ([3, 4, 5], [3.0, 4.0, 5.0, np.nan], float),
        ([3, 4, 5, 6], [3.0, 4.0, 5.0, 6.0], float),
        ([3, 4.1, 5.2], [3.0, 4.1, 5.2, np.nan], float),
        ([3, "foo"], [3, "foo", None, None], object),
        (["foo", 3], ["foo", 3, None, None], object),
        ([3, "4", "5"], [3, "4", "5", None], object),
        ([2.2, {"x": 1}, 3.3, 4], [2.2, {"x": 1}, 3.3, 4], object),
        ([True, True, False], [True, True, False, None], object),
        ([1, True, False], [1, True, False, None], object),
        (
            [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
            [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1), np.nan],
            np.datetime64,
        ),
    ],
)
def test_dtype_updating_array(values, expected, dtype):
    a = DtypeUpdatingArray((4,))
    for i, x in enumerate(values):
        a[(i,)] = x

    arr = a.arr
    np.testing.assert_array_equal(arr, expected)
    assert arr.dtype == np.dtype(dtype)
