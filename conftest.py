import os
from hypothesis import settings
import pytest

pytest.register_assert_rewrite(
    "dask.array.utils", "dask.dataframe.utils", "dask.bag.utils"
)

settings.register_profile("default", settings.default, print_blob=True)
settings.register_profile("ci", max_examples=1000, derandomize=True)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
