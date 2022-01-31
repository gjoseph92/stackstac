import importlib
from datetime import datetime
import sys
from types import ModuleType

import pytest
import satstac
import pystac

from stackstac import stac_types


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


def test_missing_pystac(monkeypatch: pytest.MonkeyPatch):
    "Test importing works without pystac"
    monkeypatch.setitem(sys.modules, "pystac", None)  # type: ignore
    # Type "ModuleType | None" cannot be assigned to type "ModuleType"

    reloaded_stac_types = importlib.reload(stac_types)
    assert "stackstac" in reloaded_stac_types.PystacItem.__module__
    assert "stackstac" in reloaded_stac_types.PystacCatalog.__module__
    assert "stackstac" in reloaded_stac_types.PystacItemCollection.__module__

    assert not reloaded_stac_types.possible_problems


@pytest.mark.parametrize(
    "module, path, inst",
    [
        (
            satstac,
            "item.Item",
            satstac.item.Item(
                {"id": "foo"},
            ),
        ),
        (
            satstac,
            "itemcollection.ItemCollection",
            satstac.itemcollection.ItemCollection(
                [],
            ),
        ),
        (pystac, "Item", pystac.Item("foo", None, None, datetime(2000, 1, 1), {})),
        (pystac, "Catalog", pystac.Catalog("foo", "bar")),
        (
            pystac,
            "ItemCollection",
            pystac.ItemCollection(
                [],
            ),
        ),
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
