import numpy as np

from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.stats import multivariate_normal


def bivariate_gauss(nr_rows: int, nr_cols: int, centers: [(float, float, float)]) -> ndarray:
    """Calculates a gaussian 2d grid with peaks at the given centers"""
    x = np.linspace(0, nr_cols, nr_cols + 1)
    y = np.linspace(0, nr_rows, nr_rows + 1)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = np.zeros((nr_rows + 1, nr_cols + 1))

    # Create grid and multivariate normal
    for center in centers:
        _, mu_x, mu_y = center
        Z *= multivariate_normal([mu_x, mu_y], [[0.5, 0], [0, 0.5]]).pdf(pos)

    # Make a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.show(block=True)
    return Z