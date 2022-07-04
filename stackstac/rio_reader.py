from __future__ import annotations

import logging
import threading
import warnings
from typing import TYPE_CHECKING, Optional, Protocol, Tuple, Type, TypedDict, Union

import numpy as np
import rasterio as rio
from rasterio.vrt import WarpedVRT

from .rio_env import LayeredEnv
from .timer import time
from .reader_protocol import Reader
from .raster_spec import RasterSpec
from .nodata_reader import NodataReader, exception_matches, nodata_for_window

if TYPE_CHECKING:
    from rasterio.enums import Resampling
    from rasterio.windows import Window


logger = logging.getLogger(__name__)


# TODO remove logging code?


def _curthread():
    return threading.current_thread().name


# /TODO


# Default GDAL configuration options
DEFAULT_GDAL_ENV = LayeredEnv(
    always=dict(
        GDAL_HTTP_MULTIRANGE="YES",  # unclear if this actually works
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        # ^ unclear if this works either. won't do much when our dask chunks are aligned to the dataset's chunks.
    ),
    open=dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        # ^ stop GDAL from requesting `.aux` and `.msk` files from the bucket (speeds up `open` time a lot)
        VSI_CACHE=True
        # ^ cache HTTP requests for opening datasets. This is critical for `ThreadLocalRioDataset`,
        # which re-opens the same URL many times---having the request cached makes subsequent `open`s
        # in different threads snappy.
    ),
    read=dict(
        VSI_CACHE=False
        # ^ *don't* cache HTTP requests for actual data. We don't expect to re-request data,
        # so this would just blow out the HTTP cache that we rely on to make repeated `open`s fast
        # (see above)
    ),
)

# These GDAL _drivers_ are safe to run in multiple threads. Note that GDAL _datasets_ are never
# safe to access across different threads. But if we create a copy of the dataset for each thread,
# and each copy uses its own file descriptor (`sharing=False`), then each thread can safely access
# its own dataset in parallel. Compare this to the hdf5 driver for example, which assumes only one
# thread is accessing the entire library at a time.
# See `ThreadLocalRioDataset` for more.
# https://github.com/pangeo-data/pangeo-example-notebooks/issues/21#issuecomment-432457955
# https://gdal.org/drivers/raster/vrt.html#multi-threading-issues
MULTITHREADED_DRIVER_ALLOWLIST = {"GTiff"}


class ThreadsafeRioDataset(Protocol):
    scale_offset: Tuple[float, float]

    def read(self, window: Window, **kwargs) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class SingleThreadedRioDataset:
    """
    Interface for a rasterio dataset whose driver is inherently single-threaded (like hdf5).

    Concurrent reads are protected by a lock.
    """

    def __init__(
        self,
        env: LayeredEnv,
        ds: rio.DatasetReader,
        vrt: Optional[WarpedVRT] = None,
    ) -> None:
        self.env = env
        self.ds = ds
        self.vrt = vrt

        # Cache this for non-locking access
        self.scale_offset = (ds.scales[0], ds.offsets[0])

        self._lock = threading.Lock()

    def read(self, window: Window, **kwargs) -> np.ndarray:
        "Acquire the lock, then read from the dataset"
        reader = self.vrt or self.ds
        with self._lock, self.env.read:
            return reader.read(1, window=window, **kwargs)

    def close(self) -> None:
        "Acquire the lock, then close the dataset"
        with self._lock:
            if self.vrt:
                self.vrt.close()
            self.ds.close()

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> SingleThreadedRioDataset:
        self._lock.acquire()
        return self

    def __exit__(self, *args) -> None:
        self._lock.release()


class ThreadLocalRioDataset:
    """
    Creates a copy of the dataset and VRT for every thread that reads from it.

    In GDAL, nothing allows you to read from the same dataset from multiple threads.
    The best concurrency support available is that you can use the same *driver*, on
    separate dataset objects, from different threads (so long as those datasets don't share
    a file descriptor). Also, the thread that reads from a dataset must be the one that creates it.

    This wrapper transparently re-opens the dataset (with ``sharing=False``, to use a separate file
    descriptor) for each new thread that accesses it. Subsequent reads by that thread will reuse that
    dataset.

    Note
    ----
    When using a large number of threads, this could potentially use a lot of memory!
    GDAL datasets are not lightweight objects.
    """

    def __init__(
        self,
        env: LayeredEnv,
        ds: rio.DatasetReader,
        vrt: Optional[WarpedVRT] = None,
    ) -> None:
        self._env = env
        self._url = ds.name
        self._driver = ds.driver
        self._open_options = ds.options

        # Cache this for non-locking access
        self.scale_offset = (ds.scales[0], ds.offsets[0])

        if vrt is not None:
            self._vrt_params = dict(
                # src_crs=vrt.src_crs.to_string(),
                # ^ we won't use this, and loading proj4 CRSs is slow
                crs=vrt.crs.to_string(),
                # ^ we _do_ ser-de the CRS to re-create it per thread,
                # because pyproj.CRS objects probably aren't thread-safe?
                resampling=vrt.resampling,
                tolerance=vrt.tolerance,
                src_nodata=vrt.src_nodata,
                nodata=vrt.nodata,
                width=vrt.width,
                height=vrt.height,
                src_transform=vrt.src_transform,
                transform=vrt.transform,
                dtype=vrt.working_dtype,
                warp_extras=vrt.warp_extras,
            )
            # ^ copied from rioxarray
            # https://github.com/corteva/rioxarray/blob/0804791a44f65ac4f303dd286e94b3eaee81f72b/rioxarray/_io.py#L720-L734
        else:
            self._vrt_params = None

        self._threadlocal = threading.local()
        self._threadlocal.ds = ds
        self._threadlocal.vrt = vrt
        # ^ NOTE: we fill these in *only for this thread*; in other threads, the attributes won't be set.
        # Instead, `_open` will lazily fill them in.

        self._lock = threading.Lock()
        # ^ NOTE this lock protects any mutation of `self`---namely, changing `self._threadlocal`.
        # The `threading.local` object is itself thread-safe (the `.x` part of `self._threadlocal.x` is protected),
        # but because `close` closes datasets across all threads by simply deleting the current threadlocal
        # and replacing it with an empty one, we have to synchronize all access to `self._threadlocal`.

    def _open(self) -> Union[SelfCleaningDatasetReader, WarpedVRT]:
        with self._env.open:
            with time(f"Reopen {self._url!r} in {_curthread()}: {{t}}"):
                result = ds = SelfCleaningDatasetReader(
                    self._url,
                    sharing=False,
                    driver=self._driver,
                    **self._open_options,
                )
            if self._vrt_params:
                with self._env.open_vrt:
                    result = vrt = WarpedVRT(ds, sharing=False, **self._vrt_params)
            else:
                vrt = None

        with self._lock:
            self._threadlocal.ds = ds
            self._threadlocal.vrt = vrt

        return result

    @property
    def dataset(self) -> Union[SelfCleaningDatasetReader, WarpedVRT]:
        try:
            with self._lock:
                return self._threadlocal.vrt or self._threadlocal.ds
        except AttributeError:
            return self._open()

    def read(self, window: Window, **kwargs) -> np.ndarray:
        "Read from the current thread's dataset, opening a new copy of the dataset on first access from each thread."
        with time(f"Read {self._url!r} in {_curthread()}: {{t}}"):
            with self._env.read:
                return self.dataset.read(1, window=window, **kwargs)

    def close(self) -> None:
        """
        Release every thread's reference to its dataset, allowing them to close.

        This method is thread-safe. After `close` returns, any `read` calls will
        open new datasets for their threads. However, for best performance, be
        sure that no thread will need to access the dataset again before
        calling `close`.

        If `close` is called while a thread-local copy of a dataset is opening,
        that thread will still receive the newly-opened dataset. The next read
        from that thread may or may not open the dataset yet again.

        Note that the underlying rasterio dataset/VRT may not be immediately closed
        upon calling this method; it will take until the next garbage-collection cycle.
        Indeed, *if any other code holds a reference to one of the rasterio datasets,
        it will not be closed at all*. This method just releases our references and relies
        on garbage collection to do the rest.
        """
        # We can't just call `close` on `self._threadlocal.ds`, because we want to close _all_
        # the datasets held by all threads.
        # It is (reasonably) very hard to access a different thread's storage on a `threading.local`
        # object, so we can't just iterate through them all and call `close`.
        # Instead, we simply replace the thread-local with a new empty one. Dropping our reference
        # to the old thread-local will cause it to delete its internal dict, thereby dropping references
        # to all the rasterio datasets contained therein.
        # Then, the `__del__` method on `WarpedVRT` and `SelfCleaningDatasetReader` will close those
        # datasets.
        # NOTE: we're assuming here that closing a GDAL dataset from a thread other than the one that created
        # it is safe to do, which, knowing GDAL, is quite possibly untrue.
        with self._lock:
            self._threadlocal = threading.local()

    def __getstate__(self):
        raise RuntimeError("Don't pickle me bro!")

    def __setstate__(self, state):
        raise RuntimeError("Don't un-pickle me bro!")


class SelfCleaningDatasetReader(rio.DatasetReader):
    # Unclear if this is even necessary, since `DatasetBase` implements `__dealloc__`,
    # but better to be safe?
    # https://github.com/mapbox/rasterio/blob/0a52d52b0c19094cd906c25fe3c23ddb48ee1f48/rasterio/_base.pyx#L445-L447
    def __del__(self):
        self.close()


class PickleState(TypedDict):
    url: str
    spec: RasterSpec
    resampling: Resampling
    dtype: np.dtype
    fill_value: Union[int, float]
    rescale: bool
    gdal_env: Optional[LayeredEnv]
    errors_as_nodata: Tuple[Exception, ...]


class AutoParallelRioReader:
    """
    rasterio-based Reader that picks the appropriate concurrency mechanism after opening the file.

    After opening the ``url`` and seeing which GDAL driver it uses, it'll use
    `ThreadLocalRioDataset` (full concurrency, but higher memory usage) if the
    driver is in `MULTITHREADED_DRIVER_ALLOWLIST`, otherwise `SingleThreadedRioDataset`
    for non-thread-safe drivers.
    """

    def __init__(
        self,
        *,
        url: str,
        spec: RasterSpec,
        resampling: Resampling,
        dtype: np.dtype,
        fill_value: Union[int, float],
        rescale: bool,
        gdal_env: Optional[LayeredEnv] = None,
        errors_as_nodata: Tuple[Exception, ...] = (),
    ) -> None:
        self.url = url
        self.spec = spec
        self.resampling = resampling
        self.dtype = dtype
        self.rescale = rescale
        self.fill_value = fill_value
        self.gdal_env = gdal_env or DEFAULT_GDAL_ENV
        self.errors_as_nodata = errors_as_nodata

        self._dataset: Optional[ThreadsafeRioDataset] = None
        self._dataset_lock = threading.Lock()

    def _open(self) -> ThreadsafeRioDataset:
        with self.gdal_env.open:
            with time(f"Initial read for {self.url!r} on {_curthread()}: {{t}}"):
                try:
                    ds = SelfCleaningDatasetReader(
                        self.url, sharing=False
                    )
                except Exception as e:
                    msg = f"Error opening {self.url!r}: {e!r}"
                    if exception_matches(e, self.errors_as_nodata):
                        warnings.warn(msg)
                        return NodataReader(
                            dtype=self.dtype, fill_value=self.fill_value
                        )

                    raise RuntimeError(msg) from e
            if ds.count != 1:
                ds.close()
                raise RuntimeError(
                    f"Assets must have exactly 1 band, but file {self.url!r} has {ds.count}. "
                    "We can't currently handle multi-band rasters (each band has to be "
                    "a separate STAC asset), so you'll need to exclude this asset from your analysis."
                )

            # Only make a VRT if the dataset doesn't match the spatial spec we want
            if self.spec.vrt_params != {
                "crs": ds.crs.to_epsg(),
                "transform": ds.transform,
                "height": ds.height,
                "width": ds.width,
            }:
                with self.gdal_env.open_vrt:
                    vrt = WarpedVRT(
                        ds,
                        sharing=False,
                        resampling=self.resampling,
                        **self.spec.vrt_params,
                    )
            else:
                logger.info(f"Skipping VRT for {self.url!r}")
                vrt = None

        if ds.driver in MULTITHREADED_DRIVER_ALLOWLIST:
            return ThreadLocalRioDataset(self.gdal_env, ds, vrt=vrt)
            # ^ NOTE: this forces all threads to wait for the `open()` we just did before they can open their
            # thread-local datasets. In principle, this would double the wall-clock open time, but if the above `open()`
            # is cached, it can actually be faster than all threads duplicating the same request in parallel.
            # This is worth profiling eventually for cases when STAC tells us the media type is a GeoTIFF.
        else:
            # logger.warning(
            #     f"Falling back on single-threaded reader for {self.url!r} (driver: {ds.driver!r}). "
            #     "This will be slow!"
            # )
            return SingleThreadedRioDataset(self.gdal_env, ds, vrt=vrt)

    @property
    def dataset(self):
        with self._dataset_lock:
            if self._dataset is None:
                self._dataset = self._open()
            return self._dataset

    def read(self, window: Window, **kwargs) -> np.ndarray:
        reader = self.dataset
        try:
            result = reader.read(
                window=window,
                masked=True,
                # ^ NOTE: we always do a masked array, so we can safely apply scales and offsets
                # without potentially altering pixels that should have been the ``fill_value``
                **kwargs,
            )
        except Exception as e:
            msg = f"Error reading {window} from {self.url!r}: {e!r}"
            if exception_matches(e, self.errors_as_nodata):
                warnings.warn(msg)
                return nodata_for_window(window, self.fill_value, self.dtype)

            raise RuntimeError(msg) from e

        if self.rescale:
            scale, offset = reader.scale_offset
            if scale != 1 and offset != 0:
                result *= scale
                result += offset

        result = result.astype(self.dtype, copy=False)
        result = np.ma.filled(result, fill_value=self.fill_value)
        return result

    def close(self) -> None:
        with self._dataset_lock:
            if self._dataset is None:
                return
            self._dataset.close()
            self._dataset = None

    def __del__(self) -> None:
        try:
            self.close()
        except AttributeError:
            # AttributeError: 'AutoParallelRioReader' object has no attribute '_dataset_lock'
            # can happen when running multithreaded. I think this somehow occurs when `__del__`
            # happens before `__init__` has even run? Is that possible?
            pass

    def __getstate__(
        self,
    ) -> PickleState:
        return {
            "url": self.url,
            "spec": self.spec,
            "resampling": self.resampling,
            "dtype": self.dtype,
            "fill_value": self.fill_value,
            "rescale": self.rescale,
            "gdal_env": self.gdal_env,
            "errors_as_nodata": self.errors_as_nodata,
        }

    def __setstate__(
        self,
        state: PickleState,
    ):
        self.__init__(**state)
        # NOTE: typechecking may not catch errors here https://github.com/microsoft/pylance-release/issues/374


# Type assertion
_: Type[Reader] = AutoParallelRioReader
