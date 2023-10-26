from string import printable

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from stackstac.coordinates_utils import (
    deduplicate_axes,
    descalar_obj_array,
    scalar_sequence,
    unnest_dicts,
    unnested_items,
    unpack_per_band_asset_fields,
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
        (
            # Sequences are _not_ traversed
            [{"a": {"b": "c"}}, {"a2": {"b2": "c2"}}],
            [{"a": {"b": "c"}}, {"a2": {"b2": "c2"}}],
        ),
        # Basics are unchanged
        ("abc", "abc"),
        (1, 1),
        (None, None),
        ([1, 2, "foo", True], [1, 2, "foo", True]),
        ({"a": 1, "b": "foo"}, {"a": 1, "b": "foo"}),
    ],
)
def test_unnest_dicts(input, expected):
    assert unnest_dicts(input) == expected


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
        # (
        #     # Sequences are _not_ traversed
        #     [{"a": {"b": "c"}}, {"a2": {"b2": "c2"}}],
        #     [{"a": {"b": "c"}}, {"a2": {"b2": "c2"}}],
        # ),
        # # Basics are unchanged
        # ("abc", "abc"),
        # (1, 1),
        # (None, None),
        # ([1, 2, "foo", True], [1, 2, "foo", True]),
        # ({"a": 1, "b": "foo"}, {"a": 1, "b": "foo"}),
    ],
)
def test_unnested_items(input, expected):
    assert list(unnested_items(input)) == list(expected.items())


jsons = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats()
    | st.datetimes()
    | st.text(printable),
    lambda children: st.lists(children) | st.dictionaries(st.text(printable), children),
)
# ^ modified from https://hypothesis.readthedocs.io/en/latest/data.html#recursive-data


@given(jsons)
def test_scalar_sequence_roundtrip(x):
    wrapped = scalar_sequence(x)
    arr = np.array([wrapped])
    assert arr.shape == (1,)
    descalared = descalar_obj_array(arr)
    assert descalared[0] == x


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
def test_unpack_per_band_asset_fields(input, fields, expected):
    result = unpack_per_band_asset_fields(input, fields)
    assert result is not input
    assert result == expected


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
