from __future__ import annotations

from hypothesis import strategies as st
import hypothesis.extra.numpy as st_np

from stackstac.to_dask import ChunksParam


@st.composite
def simple_bboxes(
    draw: st.DrawFn,
    minx: int = -100,
    miny: int = -100,
    maxx: int = 100,
    maxy: int = 100,
    *,
    zero_size: bool = True,
) -> tuple[int, int, int, int]:
    west = draw(st.integers(minx, maxx - 1))
    south = draw(st.integers(miny, maxy - 1))
    east = draw(st.integers(west if zero_size else west + 1, maxy))
    north = draw(st.integers(south if zero_size else south + 1, maxy))
    return (west, south, east, north)


raster_dtypes = (
    st_np.unsigned_integer_dtypes()
    | st_np.integer_dtypes()
    | st_np.floating_dtypes()
    | st_np.complex_number_dtypes()
)


def chunksizes(
    ndim: int,
    *,
    max_side: int | None = None,
    ints: bool = True,
    auto: bool = True,
    bytes: bool = True,
    none: bool = True,
    minus_one: bool = True,
    tuples: bool = True,
    dicts: bool = True,
    singleton: bool = True,
) -> st.SearchStrategy[ChunksParam]:
    "Generates arguments for ``chunks=`` for Dask."

    sizes = st.shared(
        st.sampled_from(["8B", f"{max_side*8}B" if max_side else "100KiB"]), key="size"
    )
    chunk_val_strategies = []
    if ints:
        chunk_val_strategies.append(st.integers(1, max_side))
    if auto:
        chunk_val_strategies.append(st.just("auto"))
    if bytes:
        chunk_val_strategies.append(sizes)

    toplevel_chunk_vals = st.one_of(chunk_val_strategies)

    if none:
        chunk_val_strategies.append(st.none())
    if minus_one:
        chunk_val_strategies.append(st.just(-1))

    chunk_vals = st.one_of(chunk_val_strategies)

    final = []
    if singleton:
        final.append(toplevel_chunk_vals)
    if tuples:
        final.append(st.tuples(*(chunk_vals,) * ndim))
    if dicts:
        final.append(
            st.dictionaries(st.integers(1, ndim), chunk_vals, min_size=1, max_size=ndim)
        )

    return st.one_of(final)
