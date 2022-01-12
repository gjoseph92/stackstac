from __future__ import annotations

from hypothesis import strategies as st
import hypothesis.extra.numpy as st_np


@st.composite
def simple_bboxes(
    draw: st.DrawFn,
    minx: int = -100,
    miny: int = -100,
    maxx: int = 100,
    maxy: int = 100,
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
