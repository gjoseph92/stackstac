from typing import Optional, Tuple, Type, Union, cast
import re

import numpy as np
from rasterio.windows import Window

from .reader_protocol import Reader

State = Tuple[np.dtype, Optional[Union[int, float]]]


class NodataReader:
    "Reader that returns a constant (nodata) value for all reads"
    scale_offset = (1.0, 0.0)

    def __init__(
        self,
        *,
        dtype: np.dtype,
        fill_value: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> None:
        self.dtype = dtype
        self.fill_value = fill_value

    def read(self, window: Window, **kwargs) -> np.ndarray:
        return nodata_for_window(window, self.fill_value, self.dtype)

    def close(self) -> None:
        pass

    def __getstate__(self) -> State:
        return (self.dtype, self.fill_value)

    def __setstate__(self, state: State) -> None:
        self.dtype, self.fill_value = state


def nodata_for_window(
    window: Window, fill_value: Optional[Union[int, float]], dtype: np.dtype
):
    assert (
        fill_value is not None
    ), "Trying to convert an exception to nodata, but `fill_value` is None"

    height = cast(int, window.height)
    width = cast(int, window.width)
    # Argument of type "tuple[_T@attrib, _T@attrib]" cannot be assigned to parameter "shape" of type "_ShapeLike"
    # in function "full"
    return np.full((height, width), fill_value, dtype)


def exception_matches(e: Exception, patterns: Tuple[Exception, ...]) -> bool:
    """
    Whether an exception matches one of the pattern exceptions

    Parameters
    ----------
    e:
        The exception to check
    patterns:
        Instances of an Exception type to catch, where ``str(exception_pattern)``
        is a regex pattern to match against ``str(e)``.
    """
    e_type = type(e)
    e_msg = str(e)
    for pattern in patterns:
        if issubclass(e_type, type(pattern)):
            if re.match(str(pattern), e_msg):
                return True
    return False


# Type assertion
_: Type[Reader] = NodataReader
