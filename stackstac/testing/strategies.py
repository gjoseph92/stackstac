from __future__ import annotations

# from math import isfinite
# import sys

from hypothesis import assume, reject, strategies as st
import hypothesis.extra.numpy as st_np

# import numpy as np

from stackstac.geom_utils import Bbox

# from stackstac.prepare import ASSET_TABLE_DT
# from stackstac.raster_spec import RasterSpec


# @st.composite
# def bboxes(
#     draw: st.DrawFn,
#     minx: int | float | None = None,
#     miny: int | float | None = None,
#     maxx: int | float | None = None,
#     maxy: int | float | None = None,
#     min_width: int | float = sys.float_info.epsilon,
#     max_width: int | float | None = None,
#     min_height: int | float = sys.float_info.epsilon,
#     max_height: int | float | None = None,
# ) -> Bbox:
#     if minx is not None and maxx is not None:
#         assert maxx > minx
#     if miny is not None and maxy is not None:
#         assert maxy > miny

#     assert min_width > 0
#     assert min_height > 0
#     if max_width is not None:
#         assert max_width > min_width
#     if max_height is not None:
#         assert max_height > min_height

#     maxx = sys.float_info.max if maxx is None else maxx
#     maxy = sys.float_info.max if maxy is None else maxy
#     # ^ https://github.com/HypothesisWorks/hypothesis/issues/3208
#     west = draw(
#         st.floats(
#             min_value=minx,
#             max_value=maxx - min_width,
#             allow_nan=False,
#             allow_infinity=False,
#             allow_subnormal=False,
#             exclude_max=True,
#         ),
#         label="west",
#     )
#     assume(isfinite(west + min_width))
#     if max_width:
#         assume(west + max_width > west)
#     south = draw(
#         st.floats(
#             min_value=miny,
#             max_value=maxy - min_height,
#             allow_nan=False,
#             allow_infinity=False,
#             allow_subnormal=False,
#             exclude_max=True,
#         ),
#         label="south",
#     )
#     assume(isfinite(south + min_height))
#     if max_height:
#         assume(south + max_height > south)
#     east = draw(
#         st.floats(
#             min_value=west + min_width,
#             max_value=min(maxx, west + max_width) if max_width else maxx,
#             allow_nan=False,
#             allow_infinity=False,
#             allow_subnormal=False,
#             exclude_min=True,
#         ),
#         label="east",
#     )
#     assume(east - west >= min_width)
#     if max_width:
#         assume(east - west <= max_width)
#     north = draw(
#         st.floats(
#             min_value=south + min_height,
#             max_value=min(maxy, south + max_height) if max_height else maxy,
#             allow_nan=False,
#             allow_infinity=False,
#             allow_subnormal=False,
#             exclude_min=True,
#         ),
#         label="north",
#     )
#     assume(north - south >= min_height)
#     if max_height:
#         assume(north - south <= max_height)
#     return (west, south, east, north)


@st.composite
def simple_bboxes(
    draw: st.DrawFn,
    minx: int = -100,
    miny: int = -100,
    maxx: int = 100,
    maxy: int = 100,
    zero_size: bool = True,
) -> Bbox:
    west = draw(st.integers(minx, maxx - 1))
    south = draw(st.integers(miny, maxy - 1))
    east = draw(st.integers(west if zero_size else west + 1, maxy))
    north = draw(st.integers(south if zero_size else south + 1, maxy))
    return (west, south, east, north)


# def resolutions(
#     min_x: int | float | None = sys.float_info.epsilon,
#     max_x: int | float | None = sys.float_info.max / 2,
#     min_y: int | float | None = sys.float_info.epsilon,
#     max_y: int | float | None = sys.float_info.max / 2,
#     equal: bool = False,
# ) -> st.SearchStrategy[tuple[float, float]]:
#     """
#     Strategy that generates tuples of x, y resolutions

#     With ``equal=True``, equal x and y resolutions will be picked
#     that satisfy all the min/max constraints. Otherwise, the resolutions
#     along x vs y may be different.

#     By default, this strategy does not use the full range of floating-point values,
#     because very large or small numbers generally cause numerical errors downstream.
#     """
#     # Notes on default values:
#     # Floats small enough that `x * -x == 0` will generally cause numerical errors downstream.
#     # The machine epsilon is not the smallest _possible_ value that will work here, but since
#     # we don't particularly care about these extreme ranges for the sorts of tests we're doing,
#     # epsilon will suffice as a "smallest reasonable resolution".
#     # Similarly, very large resolutions may overflow the `.shape` calculation on `RasterSpec`,
#     # since it uses non-standard GDAL-style rounding.
#     if equal:
#         min_ = max(filter(None, [min_x, min_y]), default=None)
#         max_ = min(filter(None, [max_x, max_y]), default=None)
#         if min_ is not None and max_ is not None:
#             assert min_ < max_, (
#                 "Cannot fit an equal resolution within these constraints: "
#                 f"{min_x=} {max_x=} {min_y=} {max_y=}"
#             )
#         return _equal_resolutions(min_, max_)
#     return _resolutions_xy(min_x, max_x, min_y, max_y)


# def _resolutions_xy(
#     min_x: int | float | None = None,
#     max_x: int | float | None = None,
#     min_y: int | float | None = None,
#     max_y: int | float | None = None,
# ) -> st.SearchStrategy[tuple[float, float]]:
#     min_x = min_x or 0
#     min_y = min_y or 0
#     assert min_x >= 0, min_x
#     assert min_y >= 0, min_y
#     return st.tuples(
#         st.floats(
#             min_x,
#             max_x,
#             allow_nan=False,
#             allow_infinity=False,
#             allow_subnormal=False,  # tiny resolutions are too small to work with
#             exclude_min=min_x == 0,
#         ),
#         st.floats(
#             min_y,
#             max_y,
#             allow_nan=False,
#             allow_infinity=False,
#             allow_subnormal=False,  # tiny resolutions are too small to work with
#             exclude_min=min_y == 0,
#         ),
#     ).filter(lambda res_xy: res_xy[0] * -res_xy[1] != 0)


# def _equal_resolutions(
#     min: int | float | None = None,
#     max: int | float | None = None,
# ) -> st.SearchStrategy[tuple[float, float]]:
#     min = min or sys.float_info.epsilon
#     assert min >= 0, min
#     return (
#         st.floats(
#             min,
#             max,
#             allow_nan=False,
#             allow_infinity=False,
#             allow_subnormal=False,
#             exclude_min=min == 0,
#         )
#         .filter(lambda x: x * -x != 0)
#         .map(lambda r: (r, r))
#     )


# @st.composite
# def raster_specs(
#     draw: st.DrawFn,
#     min_side: int | None = 1,
#     max_side: int | None = 10,
#     equal_resolution: bool = False,
# ) -> RasterSpec:
#     if min_side is not None:
#         assert min_side > 0
#     if max_side is not None:
#         assert max_side > 0
#     if min_side is not None and max_side is not None:
#         assert max_side >= min_side

#     epsg = draw(st.sampled_from([4326, 3857, 32612]), label="epsg")

#     bounds = draw(simple_bboxes(zero_size=False))
#     width, height = bounds_width_height(bounds)
#     resolutions_xy = draw(
#         resolutions(
#             width / max_side if max_side else None,
#             width / min_side if min_side else None,
#             height / max_side if max_side else None,
#             height / min_side if min_side else None,
#             equal_resolution,
#         ),
#         label="resolutions_xy",
#     )

#     # res_x, res_y = draw(
#     #     resolutions(
#     #         max_x=sys.float_info.max / 2 / max_side if max_side else None,
#     #         max_y=sys.float_info.max / 2 / max_side if max_side else None,
#     #         equal=equal_resolution,
#     #     ),
#     #     label="resolutions_xy",
#     # )
#     # bounds = draw(
#     #     bboxes(
#     #         min_width=res_x * min_side if min_side else None,
#     #         max_width=res_x * max_side if max_side else None,
#     #         min_height=res_y * min_side if min_side else None,
#     #         max_height=res_y * max_side if max_side else None,
#     #     ),
#     #     label="bounds",
#     # )

#     # # FIXME something more reasonable than this. We just tend to produce
#     # # specs where calculating the shape overflows, because the width/height
#     # # or resolution are too close to inf.
#     # # reasonableness_bound = sys.float_info.max / 4
#     # bounds = draw(
#     #     bboxes(
#     #         # -reasonableness_bound,
#     #         # -reasonableness_bound,
#     #         # reasonableness_bound,
#     #         # reasonableness_bound,
#     #     ),
#     #     label="bounds",
#     # )

#     # width, height = bounds_width_height(bounds)
#     # resolutions_xy = draw(
#     #     resolutions(
#     #         width / max_side if max_side else None,
#     #         width / min_side if min_side else None,
#     #         height / max_side if max_side else None,
#     #         height / min_side if min_side else None,
#     #         equal_resolution,
#     #     ),
#     #     label="resolutions_xy",
#     # )

#     spec = RasterSpec(epsg, bounds, resolutions_xy)
#     try:
#         shape = spec.shape
#     except AssertionError:
#         # Shape is inf/-inf/nan
#         reject()

#     if min_side:
#         assert min(shape) >= min_side, f"{shape=}, {min_side=}"
#     if max_side:
#         assert max(shape) <= max_side, f"{shape=}, {max_side=}"

#     return spec


raster_dtypes = (
    st_np.unsigned_integer_dtypes()
    | st_np.integer_dtypes()
    | st_np.floating_dtypes()
    | st_np.complex_number_dtypes()
)
