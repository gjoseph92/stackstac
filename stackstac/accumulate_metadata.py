from typing import (
    Any,
    Container,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    Union,
    cast,
)

import numpy as np
import xarray as xr


# properties can contain lists; need some way to tell them a singleton list
# apart from the list of properties we're collecting
class _ourlist(list):
    pass


def metadata_to_coords(
    items: Iterable[Mapping[str, Any]],
    dim_name: str,
    fields: Union[str, Sequence[str], Literal[True]] = True,
    skip_fields: Container[str] = (),
    only_allsame: bool = False,
):
    return dict_to_coords(
        accumulate_metadata(
            items, fields=fields, skip_fields=skip_fields, only_allsame=only_allsame
        ),
        dim_name,
    )


def accumulate_metadata(
    items: Iterable[Mapping[str, Any]],
    fields: Union[str, Sequence[str], Literal[True]] = True,
    skip_fields: Container[str] = (),
    only_allsame: bool = False,
) -> Dict[str, Any]:
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
        Skip these fields when ``fields`` is True.
    """
    if isinstance(fields, str):
        fields = (fields,)

    all_fields: Dict[str, Any] = {}
    i = 0
    for i, item in enumerate(items):
        for existing_field in all_fields.keys():
            value = item.get(existing_field, None)
            existing_value = all_fields[existing_field]
            if existing_value == value:
                # leave fields that are the same for every item as singletons
                continue

            if isinstance(existing_value, _ourlist):
                # we already have a list going; add do it
                existing_value.append(value)
            else:
                if only_allsame:
                    all_fields[existing_field] = None
                else:
                    # all prior values for this field were the same (or missing).
                    # start a new list collecting them, including Nones at the front
                    # for however many items were missing the field.
                    all_fields[existing_field] = _ourlist(
                        [None] * (i - 1) + [existing_value, value]
                    )

        if fields is True:
            # want all properties - add in any ones we haven't processed already
            for new_field in item.keys() - all_fields.keys():
                if new_field in skip_fields:
                    continue
                all_fields[new_field] = item[new_field]
        else:
            # just want some properties
            for field in cast(Iterable[str], fields):
                # ^ cast: pyright isn't smart enough to know the `else` branch means `properties` isn't True
                # https://github.com/microsoft/pyright/issues/1573
                if field not in all_fields.keys():
                    try:
                        all_fields[field] = item[field]
                    except KeyError:
                        pass

    if only_allsame:
        return {
            field: value for field, value in all_fields.items() if value is not None
        }

    return all_fields


def dict_to_coords(metadata: Dict[str, Any], dim_name: str) -> Dict[str, xr.Variable]:
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

        props_arr = np.squeeze(np.array(props))
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
