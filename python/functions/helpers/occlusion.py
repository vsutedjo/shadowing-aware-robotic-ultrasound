import numpy as np
from numpy import shape

from functions.helpers.plot_functions import setup_plot


def occlusion_fun(x, y):
    # The occluding surface function can be altered however we want it.
    # For now, we keep it as a square on level 2.
    return np.ones(shape(x)) + 1


def occlusion():
    # The occluding surface's width/length
    from_x = 0.5
    to_x = 2.5
    from_y = 0.7
    to_y = 1.9
    x = np.arange(from_x, to_x + 0.0001, 0.5)
    y = np.arange(from_y, to_y + 0.0001, 0.5)
    X, Y = np.meshgrid(x, y)
    zs = np.array(occlusion_fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    return X, Y, Z


def visualize_occlusion(nr_rows, nr_cols, nr_planes):
    setup_plot(nr_cols, np.ones((nr_rows, nr_cols, nr_planes)), occlusion_volume=occlusion(), plot_multiple=False,
               freeze=True)
