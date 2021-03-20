from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple, Union

import rasterio as rio

EnvDict = Dict[str, Union[str, bool, int, float]]


class LayeredEnv:
    """
    Manage GDAL configuration options for different situations (always, opening, reading).

    Access the `rasterio.Env` for each situation with ``layered_env.always``,
    ``layered_env.read``, etc. To ensure thread-safety, each thread accessing
    these properties gets its own thread-local copy of a `rasterio.Env`.

    Options are layered as follows:

    * ``always``: the base set of options
    * ``open``: ``always`` + ``open``
    * ``open_vrt``: ``open`` + ``open_vrt``
    * ``read``: ``always`` + ``read``
    """

    def __init__(
        self,
        always: EnvDict,
        open: Optional[EnvDict] = None,
        open_vrt: Optional[EnvDict] = None,
        read: Optional[EnvDict] = None,
    ) -> None:
        self._always = always
        self._open = open or {}
        self._open_vrt = open_vrt or {}
        self._read = read or {}

        self._threadlocal = threading.local()

    @property
    def always(self) -> rio.Env:
        "Base `rasterio.Env` object"
        try:
            return self._threadlocal.always
        except AttributeError:
            env = self._threadlocal.always = rio.Env(**self._always)
            return env

    def _get_layered_option(self, name: str, if_empty: str = "always") -> rio.Env:
        try:
            return getattr(self._threadlocal, name)
        except AttributeError:
            opts = getattr(self, "_" + name)
            env = (
                rio.Env(**dict(self._always, **opts))
                if opts
                else getattr(self, if_empty)
            )
            setattr(self._threadlocal, name, env)
            return env

    @property
    def open(self) -> rio.Env:
        "`rasterio.Env` object to use while opening datasets"
        return self._get_layered_option("open")

    @property
    def open_vrt(self) -> rio.Env:
        "`rasterio.Env` object to use while opening VRTs"
        return self._get_layered_option("open_vrt", if_empty="open")

    @property
    def read(self) -> rio.Env:
        "`rasterio.Env` object to use while reading from datasets"
        return self._get_layered_option("read")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            f"    always={self._always},\n"
            f"    open={self._open},\n"
            f"    open_vrt={self._open_vrt},\n"
            f"    read={self._read},\n"
            ")"
        )

    def updated(
        self,
        always: Optional[EnvDict] = None,
        open: Optional[EnvDict] = None,
        open_vrt: Optional[EnvDict] = None,
        read: Optional[EnvDict] = None,
    ) -> LayeredEnv:
        """
        Duplicate this LayeredEnv, adding additional options for each situation.
        """
        _always = dict(self._always, **always) if always else self._always
        _open = dict(self._open, **open) if open else self._open
        _open_vrt = dict(self._open_vrt, **open_vrt) if open_vrt else self._open_vrt
        _read = dict(self._read, **read) if read else self._read

        return type(self)(always=_always, open=_open, open_vrt=_open_vrt, read=_read)

    def __getstate__(
        self,
    ) -> Tuple[EnvDict, Optional[EnvDict], Optional[EnvDict], Optional[EnvDict]]:
        return (self._always, self._open, self._open_vrt, self._read)

    def __setstate__(
        self,
        state: Tuple[EnvDict, Optional[EnvDict], Optional[EnvDict], Optional[EnvDict]],
    ):
        self.__init__(*state)
