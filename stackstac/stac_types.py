from __future__ import annotations

"""
Compatibility methods for all the different ways of representing STAC catalogs in Python.

Because dicts and lists just are never enough to represent JSON data.
"""

from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)

possible_problems: list[str] = []

try:
    from satstac.item import Item as SatstacItem
    from satstac.itemcollection import ItemCollection as SatstacItemCollection
except ImportError as e:

    class SatstacItem:
        _data: ItemDict

    class SatstacItemCollection:
        def __iter__(self) -> Iterator[SatstacItem]:
            ...

    if not isinstance(e, ModuleNotFoundError):
        possible_problems.append(
            "Your version of `satstac` is too old (or new) for stackstac. "
            "`satstac.Item` and `satstac.ItemCollection` aren't supported "
            f"because they could not be imported: {e!r}"
        )
    del e


try:
    from pystac import Catalog as PystacCatalog
    from pystac import Item as PystacItem
    from pystac import ItemCollection as PystacItemCollection
except ImportError as e:

    class PystacItem:
        def to_dict(self) -> ItemDict:
            ...

    class PystacCatalog:
        def get_all_items(self) -> Iterator[PystacItem]:
            ...

    class PystacItemCollection:
        features: List[PystacItem]

        def __iter__(self) -> Iterator[PystacItem]:
            ...

    if not isinstance(e, ModuleNotFoundError):
        possible_problems.append(
            "Your version of `pystac` is too old (or new) for stackstac. "
            "`pystac.Item`, `pystac.ItemCollection`, and `pystac.Catalog` aren't supported "
            f"because they could not be imported: {e!r}"
        )
    del e


class EOBand(TypedDict, total=False):
    name: str
    common_name: str
    description: str
    center_wavelength: float
    full_width_half_max: float


AssetDict = TypedDict(
    "AssetDict",
    {
        "href": str,
        "title": str,
        "description": str,
        "type": str,
        "roles": List[str],
        "proj:shape": Tuple[int, int],
        "proj:transform": Union[
            Tuple[int, int, int, int, int, int],
            Tuple[int, int, int, int, int, int, int, int, int],
        ],
        "eo:bands": EOBand,
        "sar:polarizations": List[str],
    },
    total=False,
)

PropertiesDict = TypedDict(
    "PropertiesDict",
    {
        "datetime": Optional[str],
        "proj:epsg": Optional[int],
        "proj:bbox": Tuple[float, float, float, float],
        "proj:shape": Tuple[int, int],
        "proj:transform": Union[
            Tuple[int, int, int, int, int, int],
            Tuple[int, int, int, int, int, int, int, int, int],
        ],
    },
    total=False,
)


class ItemDict(TypedDict):
    stac_version: str
    id: str
    type: Literal["Feature"]
    geometry: Optional[dict]
    bbox: Tuple[float, float, float, float]
    properties: PropertiesDict
    assets: Dict[str, AssetDict]
    stac_extensions: List[str]
    collection: str


ItemSequence = Sequence[ItemDict]

ItemIsh = Union[SatstacItem, PystacItem, ItemDict]
ItemCollectionIsh = Union[
    SatstacItemCollection, PystacCatalog, PystacItemCollection, ItemSequence
]


def items_to_plain(items: Union[ItemCollectionIsh, ItemIsh]) -> ItemSequence:
    """
    Convert something like a collection/Catalog of STAC items into a list of plain dicts

    Currently works on ``satstac.ItemCollection``, ``pystac.Catalog`` (inefficiently),
    ``pystac.ItemCollection`` (inefficiently), and plain Python lists-of-dicts.
    """

    if isinstance(items, dict):
        # singleton item
        return [items]

    if isinstance(items, Sequence):
        # slicing a satstac `ItemCollection` produces a list, not another `ItemCollection`,
        # so having a `List[SatstacItem]` is quite possible
        try:
            return [item._data for item in cast(SatstacItemCollection, items)]
        except AttributeError:
            return items

    if isinstance(items, SatstacItem):
        return [items._data]

    if isinstance(items, PystacItem):
        # TODO this is wasteful.
        return [items.to_dict()]

    if isinstance(items, SatstacItemCollection):
        return [item._data for item in items]

    # TODO all our `to_dict()` with pystac is wasteful. Instead of this `items_to_plain` function,
    # switch to `get_items`, `get_properties`, `get_assets`, etc. style functions
    # which can handle each object type, preventing the need for this sort of copying.
    if isinstance(items, PystacCatalog):
        return [item.to_dict() for item in items.get_all_items()]

    if isinstance(items, PystacItemCollection):
        return [item.to_dict() for item in items]

    raise TypeError(
        f"Unrecognized STAC collection type {type(items)}: {items!r}"
        + (
            "\n".join(["\nPossible problems:"] + possible_problems)
            if possible_problems
            else ""
        )
    )
