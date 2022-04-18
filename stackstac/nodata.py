from typing import Tuple, Union
import re

import numpy as np
from rasterio.windows import Window

State = Tuple[np.dtype, Union[int, float]]


def nodata_for_window(
    ndim: int, window: Window, fill_value: Union[int, float], dtype: np.dtype
):
    return np.full((ndim, window.height, window.width), fill_value, dtype)


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
