from typing import (
    Container,
    Dict,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    Union,
    TypeVar,
)

import numpy as np
import xarray as xr


# properties can contain lists; need some way to tell them a singleton list
# apart from the list of properties we're collecting
class _ourlist(list):
    pass


def metadata_to_coords(
    items: Iterable[Mapping[str, object]],
    dim_name: str,
    fields: Union[str, Sequence[str], Literal[True]] = True,
    skip_fields: Container[str] = (),
) -> Dict[str, xr.Variable]:
    return dict_to_coords(
        accumulate_metadata(
            items,
            fields=[fields] if isinstance(fields, str) else fields,
            skip_fields=skip_fields,
        ),
        dim_name,
    )


T = TypeVar("T", bound=Hashable)


def accumulate_metadata(
    items: Iterable[Mapping[T, object]],
    fields: Union[Sequence[T], Literal[True]] = True,
    skip_fields: Container[T] = (),
) -> Dict[T, object]:
    """
    Accumulate a sequence of multiple similar dicts into a single dict of lists.

    Each field will contain a list of all the values for that field (equal length to ``items``).
    For items where the field didn't exist, None is used.

    Fields with only one unique value are flattened down to just that single value.

    Parameters
    ----------
    items:
        Iterable of dicts to accumulate
    fields:
        Only use these fields. If True, use all fields.
    skip_fields:
        Skip these fields.
    """
    all_fields: Dict[T, object] = {}
    for i, item in enumerate(items):
        # Inductive case: update existing fields
        for existing_field, existing_value in all_fields.items():
            new_value = item.get(existing_field, None)
            if new_value == existing_value:
                # leave fields that are the same for every item as singletons
                continue
            if isinstance(existing_value, _ourlist):
                # we already have a list going; add to it
                existing_value.append(new_value)
            else:
                # all prior values were the same; this is the first different one
                all_fields[existing_field] = _ourlist(
                    [existing_value] * i + [new_value]
                )

        # Base case 1: add any never-before-seen fields, when inferring field names
        if fields is True:
            for new_field in item.keys() - all_fields.keys():
                if new_field in skip_fields:
                    continue
                value = item[new_field]
                all_fields[new_field] = (
                    value if i == 0 else _ourlist([None] * i + [value])
                )
        # Base case 2: initialize with predefined fields
        elif i == 0:
            all_fields.update(
                (field, item.get(field, None))
                for field in fields
                if field not in skip_fields
            )

    return all_fields


def accumulate_metadata_only_allsame(
    items: Iterable[Mapping[T, object]],
    skip_fields: Container[T] = (),
) -> Dict[T, object]:
    """
    Accumulate multiple similar dicts into a single flattened dict of only consistent values.

    If the value of a field differs between items, the field is dropped.
    If the value of a field is the same for all items that contain that field, the field is kept.

    Note this means that missing fields are ignored, not treated as different.

    Parameters
    ----------
    items:
        Iterable of dicts to accumulate
    skip_fields:
        Skip these fields when ``fields`` is True.
    """
    all_fields: Dict[T, object] = {}
    for item in items:
        for field, value in item.items():
            if field in skip_fields:
                continue
            if field not in all_fields:
                all_fields[field] = value
            else:
                if value != all_fields[field]:
                    all_fields[field] = None

    return {field: value for field, value in all_fields.items() if value is not None}


def dict_to_coords(
    metadata: Dict[str, object], dim_name: str
) -> Dict[str, xr.Variable]:
    """
    Convert the output of `accumulate_metadata` into a dict of xarray Variables.

    1-length lists and scalar values become 0D variables.

    Instances of ``_ourlist`` become 1D variables for ``dim_name``.

    Any other things with >= 1 dimension are dropped, because they probably don't line up
    with the other dimensions of the final array.
    """
    coords = {}
    for field, props in metadata.items():
        while isinstance(props, list) and not isinstance(props, _ourlist):
            # a list scalar (like `instruments = ['OLI', 'TIRS']`).

            # first, unpack (arbitrarily-nested) 1-element lists.
            # keep re-checking if it's still a list
            if len(props) == 1:
                props = props[0]
                continue

            # for now, treat multi-item lists as a set so xarray can interpret them as 0D variables.
            # (numpy very much does not like to create object arrays containing python lists;
            # `set` is basically a hack to make a 0D ndarray containing a Python object with multiple items.)
            try:
                props = set(props)
            except TypeError:
                # if it's not set-able, just give up
                break

        props_arr = np.squeeze(
            np.array(
                props,
                # Avoid DeprecationWarning creating ragged arrays when elements are lists/tuples of different lengths
                dtype="object"
                if (
                    isinstance(props, _ourlist)
                    and len(set(len(x) for x in props if isinstance(x, (list, tuple))))
                    > 1
                )
                else None,
            )
        )

        if (
            props_arr.ndim > 1
            or props_arr.ndim == 1
            and not isinstance(props, _ourlist)
        ):
            # probably a list-of-lists situation. the other dims likely don't correspond to
            # our "bands", "y", and "x" dimensions, and xarray won't let us use unrelated
            # dimensions. so just skip it for now.
            continue

        coords[field] = xr.Variable(
            (dim_name,) if props_arr.ndim == 1 else (),
            props_arr,
        )

    return coords
