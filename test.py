import numpy as np
import matplotlib.cm

import stackstac

test = np.arange(10000).reshape(100, 100)[None].astype(float)
test[:, :40, :40] = np.nan

pb = stackstac._show.arr_to_png(test, (0, 10000), matplotlib.cm.get_cmap("magma"), checkerboard=True)

with open("test.png", "wb") as f:
    f.write(pb)
