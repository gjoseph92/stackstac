from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Literal, Optional, TYPE_CHECKING, Tuple, Union, cast
import io

import dask
from dask.delayed import Delayed
import dask.array as da
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm
import matplotlib.colors
import xarray as xr
from xarray.plot.utils import _rescale_imshow_rgb

if TYPE_CHECKING:
    import IPython.display


def _validate_arr_for_gif(
    arr: xr.DataArray,
    cmap: Optional[Union[str, matplotlib.colors.Colormap]],
    date_format: Optional[str],
    date_position: Literal["ul", "ur", "ll", "lr"],
) -> Tuple[xr.DataArray, Optional[matplotlib.colors.Colormap]]:
    if arr.ndim not in (3, 4):
        raise ValueError(
            f"Array must only have the dimensions 'time', 'y', 'x', and optionally 'band', not {arr.dims!r}"
        )
    if arr.ndim == 3:
        arr = arr.expand_dims("band", axis=1)

    if arr.shape[1] not in (1, 3):
        raise ValueError(f"Array must have 1 or 3 bands, not {arr.shape[1]}")

    if arr.shape[1] == 1:
        cmap = (
            # this will use the default colormap (usually viridis) if it's None
            matplotlib.cm.get_cmap(cmap)
            if not isinstance(cmap, matplotlib.colors.Colormap)
            else cmap
        )
    elif cmap is not None:
        raise ValueError(
            f"Colormaps are only possible on single-band data; this array has {arr.shape[1]} bands: "
            f"{arr[arr.dims[1]].data.tolist()}"
        )

    if date_format:
        time_coord = arr[arr.dims[0]]
        try:
            time_coord.dt
        except TypeError:
            raise ValueError(
                f"Coordinates for the {time_coord.name} dimension are not datetimes or timedeltas. "
                "Set `date_format=False`"
            )
        assert date_position in (
            "ul",
            "ur",
            "ll",
            "lr",
        ), f"date_position must be one of ('ul', 'ur', 'll', 'lr'), not {date_position}."

    return (arr, cmap)


def gif(
    arr: xr.DataArray,
    *,
    to: Optional[Union[str, Path, BinaryIO]] = None,
    fps: int = 16,
    robust: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    date_format: Optional[str] = "%Y-%m-%d",
    date_position: Literal["ul", "ur", "ll", "lr"] = "ul",
    date_color: Tuple[int, int, int] = (255, 255, 255),
    date_bg: Optional[Tuple[int, int, int]] = (0, 0, 0),
) -> Union[IPython.display.Image, None]:
    """
    Render a `~xarray.DataArray` timestack (``time``, ``band``, ``y``, ``x``) into a GIF.

    If the `~xarray.DataArray` contains a `dask.Array`, use `dgif` (delayed-GIF) instead.

    The `~xarray.DataArray` must have 1 or 3 bands.

    Unless ``date_format=None``, a small timestamp will be printed onto each frame of the animation.
    You can control the position and styling of this with the ``date_position``, ``date_color``, and
    ``date_bg`` arguments.

    Parameters
    ----------
    arr:
        Time-stacked array to animate. Must have 3 or 4 dimensions, which are assumed to be
        in the order ``time``, [optional ``band``], ``y``, ``x``.
    to:
        Where to write the GIF. If None (default), an `IPython.display.Image` is returned,
        which will display the GIF in your Jupyter notebook.
    fps:
        Frames per second
    robust:
        Calculate ``vmin`` and ``vmax`` from the 2nd and 98th percentiles of the data
        (default True)
    vmin:
        Value in the data to map to 0 (black). If None (default), it's calculated
        from the minimum value of the data or the 2nd percentile, depending on ``robust``.
    vmax:
        Value in the data to map to 255 (white). If None (default), it's calculated
        from the maximum value of the data or the 98nd percentile, depending on ``robust``.
    cmap:
        Colormap to use for single-band data. Can be a
        :doc:`matplotlib colormap name <gallery/color/colormap_reference>` as a string,
        or a `~matplotlib.colors.Colormap` object for custom colormapping.

        If None (default), the default matplotlib colormap (usually ``viridis``) will automatically
        be used for 1-band data. Setting a colormap for multi-band data is an error.
    date_format:
        Date format string (like ``"%Y-%m-%d"``, the default) used to format the timestamps
        written onto each frame of the animation. If the coordinates for axis 0 of the
        `~xarray.DataArray` are not timestamps or timedeltas, you must explicitly pass
        ``date_format=None``.

        See the `Python string format doc
        <https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>`__
        for details.
    date_position:
        Where to print the timestamp on each frame.
        One of ``"ul"`` (upper-left), ``"ur"`` (upper-right), ``"ll"`` (lower-left),
        ``"lr"`` (lower-right), default ``"ul"``.
    date_color:
        Color for the timestamp font, as an RGB 3-tuple. Default: ``(255, 255, 255)``
        (white).
    date_bg:
        Fill color to draw behind the timestamp (for legibility), as an RGB 3-tuple.
        Default: ``(0, 0, 0)`` (black). Set to None to disable.

    Returns
    -------
    IPython.display.Image or None
        If ``to`` is None, returns an `IPython.display.Image`, which will display the
        GIF in a Jupyter Notebook. (You can also get the GIF data as bytes from the Image's
        ``.data`` attribute.)

        Otherwise, returns None, and the GIF data is written to ``to``.

    Examples
    --------
    Generate a GIF and show it in your notebook:
    >>> stackstac.gif(arr, date_format="Year: %Y")

    Write the GIF to a file, with no timestamp printed:
    >>> stackstac.gif(arr, to="animation.gif", fps=24, date_format=None)

    Show a colormapped GIF of single-band data in your notebook,
    with the timestamp font in black and no background behind it:
    >>> stackstac.gif(arr.sel(band="ndvi"), cmap="YlGn", date_color=(0, 0, 0), date_bg=None)
    """
    if isinstance(arr.data, da.Array):
        raise TypeError("DataArray contains delayed data; use `dgif` instead.")

    arr, cmap = _validate_arr_for_gif(arr, cmap, date_format, date_position)

    # Rescale
    if not robust and vmin is None and vmax is None:
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
    rescaled: xr.DataArray = _rescale_imshow_rgb(arr, vmin, vmax, robust)
    data: np.ndarray = rescaled.data

    # Colormap
    if arr.shape[1] == 1:
        assert isinstance(cmap, matplotlib.colors.Colormap)
        data = data[:, 0]
        data = cmap(data)
        data = np.moveaxis(data, -1, -3)  # colormap puts RGB last

    # Convert to uint8
    u8 = (data * 255).astype("uint8")
    u8 = np.clip(u8, 0, 255, out=u8)
    u8 = np.moveaxis(u8, -3, -1)

    # Add alpha mask
    if data.shape[1] == 4:
        # colormap has already added the alpha band
        frames = u8
    else:
        mask: np.ndarray = arr.isnull().data.any(axis=-3)
        alpha = (~mask).astype("uint8", copy=False) * 255
        frames = np.concatenate([u8, alpha[..., None]], axis=-1)

    imgs = [Image.fromarray(frame) for frame in frames]

    # Write timestamps onto each frame
    if date_format:
        time_coord = arr[arr.dims[0]]
        labels = time_coord.dt.strftime(date_format).data

        fnt = ImageFont.load_default()
        for label, img in zip(labels, imgs):
            # get a drawing context
            d = ImageDraw.Draw(img)
            d = cast(ImageDraw.ImageDraw, d)

            width, height = img.size
            t_width, t_height = fnt.getsize(label)

            offset = 15
            if date_position[0] == "u":
                y = offset
            else:
                y = height - t_height - offset

            if date_position[1] == "l":
                x = offset
            else:
                x = width - t_width - offset

            if date_bg:
                d.rectangle((x, y, x + t_width, y + t_height), fill=date_bg)
            # draw text
            d.multiline_text((x, y), label, font=fnt, fill=date_color)

    out = to if to is not None else io.BytesIO()
    imgs[0].save(
        out,
        format="gif",
        save_all=True,
        append_images=imgs[1:],
        duration=1 / fps * 1000,  # ms
        loop=False,
    )
    if to is None and isinstance(out, io.BytesIO):
        # second `isinstace` is just for the typechecker
        try:
            import IPython.display
        except ImportError:
            raise ImportError(
                "Cannot return an Image to display in a notebook, since IPython is not installed. "
                "Pass a path or file to save the GIF to as the `to=` argument. "
                "To get the GIF data as bytes, pass an instance of `io.BytesIO()`.\n"
                "If this error is coming from your distributed cluster and you called `dgif`, "
                "then IPython is not installed on your dask workers. Either install it, or "
                "pass `dgif(arr, bytes=True)` to return the GIF as bytes. "
                "Then use `IPython.display.Image(data=computed_bytes)` to show the image."
            )
        else:
            return IPython.display.Image(data=out.getvalue())


def _gif(arr: xr.DataArray, bytes=False, **kwargs):
    to = io.BytesIO() if bytes else None
    out = gif(arr, to=to, **kwargs)
    return to.getvalue() if bytes else out


_dgif = dask.delayed(_gif, pure=True)


def dgif(
    arr: xr.DataArray,
    *,
    bytes=False,
    fps: int = 10,
    robust: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    date_format: Optional[str] = "%Y-%m-%d",
    date_position: Literal["ul", "ur", "ll", "lr"] = "ul",
    date_color: Tuple[int, int, int] = (255, 255, 255),
    date_bg: Optional[Tuple[int, int, int]] = (0, 0, 0),
) -> Delayed:
    """
    Turn a dask-backed `~xarray.DataArray` timestack into a GIF, as a `dask.Delayed` object.

    The `~xarray.DataArray` must have 1 or 3 bands, and dimensions in
    (``time``, [optional ``band``], ``y``, ``x``) order.

    If all you want is a GIF, `dgif` can be faster than calling ``.compute()`` and then `gif`:
    since GIFs are smaller and reduced in quality from NumPy arrays, there's less data to transfer
    from the cluster back to your computer. Or, if you want to generate lots of GIFs and store them
    in a bucket, `dgif` makes it easier to parallelize that with Dask.

    If the `~xarray.DataArray` is not delayed, and just contains a NumPy array, use `gif` instead.

    Unless ``date_format=None``, a small timestamp will be printed onto each frame of the animation.
    You can control the position and styling of this with the ``date_position``, ``date_color``, and
    ``date_bg`` arguments.

    Parameters
    ----------
    arr:
        Time-stacked array to animate. Must have 3 or 4 dimensions, which are assumed to be
        in the order ``time``, [optional ``band``], ``y``, ``x``.
    bytes:
        Whether to return the GIF as bytes, or as an `IPython.display.Image`.
        If ``bytes=False`` (default), then ``dgif(...).compute()`` will (eventually)
        display the image in your Jupyter notebook without any extra steps.

        Note that you can also access the raw bytes from an `IPython.display.Image`
        with the ``.data`` attribute.
    fps:
        Frames per second
    robust:
        Calculate ``vmin`` and ``vmax`` from the 2nd and 98th percentiles of the data
        (default True)
    vmin:
        Value in the data to map to 0 (black). If None (default), it's calculated
        from the minimum value of the data or the 2nd percentile, depending on ``robust``.
    vmax:
        Value in the data to map to 255 (white). If None (default), it's calculated
        from the maximum value of the data or the 98nd percentile, depending on ``robust``.
    cmap:
        Colormap to use for single-band data. Can be a
        :doc:`matplotlib colormap name <gallery/color/colormap_reference>` as a string,
        or a `~matplotlib.colors.Colormap` object for custom colormapping.

        If None (default), the default matplotlib colormap (usually ``viridis``) will automatically
        be used for 1-band data. Setting a colormap for multi-band data is an error.
    date_format:
        Date format string (like ``"%Y-%m-%d"``, the default) used to format the timestamps
        written onto each frame of the animation. If the coordinates for axis 0 of the
        `~xarray.DataArray` are not timestamps or timedeltas, you must explicitly pass
        ``date_format=None``.

        See the `Python string format doc
        <https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>`__
        for details.
    date_position:
        Where to print the timestamp on each frame.
        One of ``"ul"`` (upper-left), ``"ur"`` (upper-right), ``"ll"`` (lower-left),
        ``"lr"`` (lower-right), default ``"ul"``.
    date_color:
        Color for the timestamp font, as an RGB 3-tuple. Default: ``(255, 255, 255)``
        (white).
    date_bg:
        Fill color to draw behind the timestamp (for legibility), as an RGB 3-tuple.
        Default: ``(0, 0, 0)`` (black). Set to None to disable.

    Returns
    -------
    dask.Delayed
        Delayed object which, when computed, resolves to either an `IPython.display.Image`,
        or `bytes`.

    Examples
    --------
    Compute a GIF on the cluster and show it in your notebook:
    >>> stackstac.dgif(arr, date_format="Year: %Y").compute()

    Compute a GIF on the cluster, get back the bytes, and write them to a file:
    >>> gif_data = stackstac.dgif(arr, bytes=True).compute()
    >>> with open("animation.gif", "wb") as f:
    ...     f.write(gif_data)

    Compute a GIF on the cluster, and write it to an S3 bucket:
    >>> import fsspec
    >>> import dask.delayed
    >>> bucket = dask.delayed(fsspec.get_mapper('s3://my-sweet-gifs/latest'))
    >>> gif = stackstac.dgif(arr, bytes=True)
    >>> bucket.setitems({"neat.gif": gif}).compute()
    """

    if not isinstance(arr.data, da.Array):
        raise TypeError(
            "DataArray does not contain delayed (Dask) data; use `gif` instead to render a GIF locally."
        )

    # Do some quick sanity checks to save you a lot of compute
    _validate_arr_for_gif(arr, cmap, date_format, date_position)

    if not bytes:
        try:
            import IPython.display  # noqa: F401
        except ImportError:
            raise ImportError(
                "Cannot return an Image to display in a notebook, since IPython is not installed "
                "(so you also must not be running in a notebook).\n"
                "Pass `bytes=True` to get back the GIF as plain bytes instead. "
                "You could then save the bytes to a file, and open that with an image-viewing program:\n\n"
                "gif_bytes = dgif(arr, bytes=True, ...).compute()\n"
                "with open('animation.gif', 'wb') as f:\n"
                "    f.write(gif_bytes)"
            )

    # Array optimizations won't be applied once we convert to delayed, so do it now.
    # https://github.com/dask/dask/issues/7587
    # TODO condition this on a LooseVersion check for this once #7587 is closed
    (arr,) = dask.optimize(arr)

    return _dgif(
        arr,
        bytes=bytes,
        fps=fps,
        robust=robust,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        date_format=date_format,
        date_position=date_position,
        date_color=date_color,
        date_bg=date_bg,
    )
