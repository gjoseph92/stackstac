def select_unique_vars(ds, dim=["band", "time"]):
    """
    Return the deduplicated data variables from a xarray.Dataset
    that are constant along specified dimensions.

    Parameters
    ----------
    ds:
        xarray.dataset
    dim:
        Name of the dimension(s) to drop duplicates along
    """
    dim = [dim] if type(dim) is str else dim
    sel_dim = {i: 0 for i in dim}
    constant_vars = ds.apply(lambda x: (x == x.isel(sel_dim)).all()).to_array()
    constant_vars = constant_vars[constant_vars]["variable"].data
    if len(constant_vars) > 0:
        constant_ds = ds[constant_vars]
        constant_ds = constant_ds.apply(lambda x: x.isel(sel_dim))  # Keeping the first value only
    else:
        constant_ds = {}

    return constant_ds


def drop_allnull_vars(ds):
    """
    Dropping data variable in a xarray.Dataset that are always nulls

    Parameters
    ----------
    ds:
        xarray.dataset
    """
    nulls = ds.isnull().all().to_array()
    no_nulls = nulls[~nulls]["variable"]
    return ds[no_nulls.data]
