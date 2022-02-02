from typing import Tuple, Type, Union
import re

import numpy as np
from rasterio.windows import Window

from .reader_protocol import Reader

State = Tuple[np.dtype, Union[int, float]]


class NodataReader:
    "Reader that returns a constant (nodata) value for all reads"
    scale_offset = (1.0, 0.0)

    def __init__(
        self,
        *,
        dtype: np.dtype,
        fill_value: Union[int, float],
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


def nodata_for_window(window: Window, fill_value: Union[int, float], dtype: np.dtype):
    return np.full((window.height, window.width), fill_value, dtype)


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
