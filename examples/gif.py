import dask
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import IPython.display as dsp


def gif(arr: xr.DataArray, fps=5, dpi=72, filename=None):
    if filename is None:
        filename = f"animation-{dask.base.tokenize(arr, fps, dpi)}.gif"

    # http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    rescaled = xr.plot.utils._rescale_imshow_rgb(arr.ffill("time")[1:], None, None, True)

    snapshots = np.moveaxis(rescaled.data, -3, -1)

    # First set up the figure, the axis, and the plot element we want to animate
    height, width = arr.shape[-2] / dpi, arr.shape[-1] / dpi
    fig = plt.figure(figsize=(width, height), frameon=False, tight_layout=True)

    a = snapshots[0]
    empty = a * 0
    im = plt.imshow(a, interpolation="none", aspect="equal", vmin=0, vmax=1)

    # initialization function: plot the background of each frame
    # FWIW this doesn't seem to do anything
    def init():
        im.set_data(empty)
        plt.title("")
        return [im]

    def animate_func(frame, *fargs):
        if frame % fps == 0:
            print(".", end="")

        im.set_data(snapshots[frame])
        plt.title(rescaled.time[frame].data)
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        init_func=init,
        frames=len(snapshots),
        blit=True,
        interval=1000 / fps,  # in ms
    )

    anim.save(filename, fps=fps, writer="Pillow")
    plt.close()
    return dsp.Image(filename)


@dask.delayed(pure=True)
def delayed_gif(arr: xr.DataArray, fps=5, dpi=72, filename=None):
    if filename is None:
        filename = f"animation-{dask.base.tokenize(arr, fps, dpi)}.gif"

    gif(arr, fps=fps, dpi=dpi, filename=filename)

    with open(filename, "rb") as f:
        return f.read()
