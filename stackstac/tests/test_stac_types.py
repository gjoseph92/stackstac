import importlib
from datetime import datetime
import sys
from types import ModuleType

import pytest
import satstac
import pystac

from stackstac import stac_types

satstac_itemcollection = satstac.itemcollection.ItemCollection(
    [
        satstac.item.Item(
            {"id": "foo"},
        ),
        satstac.item.Item(
            {"id": "bar"},
        ),
    ]
)


pystac_catalog = pystac.Catalog("foo", "bar")
pystac_catalog.add_items(
    [
        pystac.Item("foo", None, None, datetime(2000, 1, 1), {}),
        pystac.Item("bar", None, None, datetime(2001, 1, 1), {}),
    ]
)
pystac_foo_dict = {
    k: v
    for k, v in pystac_catalog.get_item("foo").to_dict().items()
    if k
    in (
        "type",
        "stac_version",
        "id",
        "properties",
        "geometry",
        "href",
        "assets",
        "stac_extensions",
    )
}
pystac_bar_dict = {
    k: v
    for k, v in pystac_catalog.get_item("bar").to_dict().items()
    if k
    in (
        "type",
        "stac_version",
        "id",
        "properties",
        "geometry",
        "assets",
        "stac_extensions",
    )
}


@pytest.mark.parametrize(
    "input, expected",
    [
        ({"id": "foo"}, [{"id": "foo"}]),
        ([{"id": "foo"}, {"id": "bar"}], [{"id": "foo"}, {"id": "bar"}]),
        # satstac,
        (satstac_itemcollection[0], [{"id": "foo"}]),
        (satstac_itemcollection, [{"id": "foo"}, {"id": "bar"}]),
        (satstac_itemcollection[:], [{"id": "foo"}, {"id": "bar"}]),
        # pystac,
        (pystac_catalog.get_item("foo"), [pystac_foo_dict]),
        (
            pystac.ItemCollection(pystac_catalog.get_all_items()),
            [pystac_foo_dict, pystac_bar_dict],
        ),
        (
            pystac_catalog,
            [pystac_foo_dict, pystac_bar_dict],
        ),
        (list(pystac_catalog.get_all_items()), [pystac_foo_dict, pystac_bar_dict]),
    ],
)
def test_basic(input, expected):
    results = stac_types.items_to_plain(input)
    assert isinstance(results, list)
    assert len(results) == len(expected)
    for result, exp in zip(results, expected):
        # Only check fields stackstac actually cares about (we don't use the `link` field, for example)
        subset = {k: v for k, v in result.items() if k in exp}
        assert subset == exp


def test_normal_case():
    assert stac_types.SatstacItem is satstac.item.Item
    assert stac_types.SatstacItemCollection is satstac.itemcollection.ItemCollection

    assert stac_types.PystacItem is pystac.Item
    assert stac_types.PystacItemCollection is pystac.ItemCollection
    assert stac_types.PystacCatalog is pystac.Catalog


def test_missing_satstac(monkeypatch: pytest.MonkeyPatch):
    "Test importing works without satstac"
    monkeypatch.setitem(sys.modules, "satstac", None)  # type: ignore
    monkeypatch.setitem(sys.modules, "satstac.item", None)  # type: ignore
    monkeypatch.setitem(sys.modules, "satstac.itemcollection", None)  # type: ignore
    # Type "ModuleType | None" cannot be assigned to type "ModuleType"

    reloaded_stac_types = importlib.reload(stac_types)
    assert "stackstac" in reloaded_stac_types.SatstacItem.__module__
    assert "stackstac" in reloaded_stac_types.SatstacItemCollection.__module__

    assert not reloaded_stac_types.possible_problems
    # clean things up for other tests
    monkeypatch.undo()
    importlib.reload(stac_types)


def test_missing_pystac(monkeypatch: pytest.MonkeyPatch):
    "Test importing works without pystac"
    monkeypatch.setitem(sys.modules, "pystac", None)  # type: ignore
    # Type "ModuleType | None" cannot be assigned to type "ModuleType"

    reloaded_stac_types = importlib.reload(stac_types)
    assert "stackstac" in reloaded_stac_types.PystacItem.__module__
    assert "stackstac" in reloaded_stac_types.PystacCatalog.__module__
    assert "stackstac" in reloaded_stac_types.PystacItemCollection.__module__

    assert not reloaded_stac_types.possible_problems
    # clean things up for other tests
    monkeypatch.undo()
    importlib.reload(stac_types)


@pytest.mark.parametrize(
    "module, path, inst",
    [
        (
            satstac,
            "item.Item",
            satstac_itemcollection[0],
        ),
        (
            satstac,
            "itemcollection.ItemCollection",
            satstac_itemcollection,
        ),
        (
            satstac,
            "item.Item",
            list(satstac_itemcollection),
        ),
        (pystac, "Item", pystac_catalog.get_item("foo")),
        (pystac, "Catalog", pystac_catalog),
        (
            pystac,
            "ItemCollection",
            pystac.ItemCollection([]),
        ),
        (pystac, "Item", list(pystac_catalog.get_all_items())),
    ],
)
def test_unimportable_path(
    module: ModuleType, path: str, inst: object, monkeypatch: pytest.MonkeyPatch
):
    """
    Test that importing still works when a type isn't importable from pystac/satstac,
    but the overall module is importable. (Simulating a breaking change/incompatible version.)

    Verify that a useful error is shown when `items_to_plain` fails.
    """
    parts = path.split(".")
    modname = module.__name__

    # Delete the `path` from `module`. We do this instead of just putting None in `sys.modules`,
    # so that we get a proper `ImportError` instead of `ModuleNotFoundError`.
    for p in parts:
        prev = module
        module = getattr(module, p)
        monkeypatch.delattr(prev, p)

    reloaded_stac_types = importlib.reload(stac_types)

    with pytest.raises(TypeError, match=f"Your version of `{modname}` is too old"):
        reloaded_stac_types.items_to_plain(inst)

    # clean things up for other tests
    monkeypatch.undo()
    importlib.reload(stac_types)
