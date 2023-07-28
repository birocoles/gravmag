import numpy as np
from . import check


def regular_grid_xy(area, shape, ordering="xy", check_input=True):
    """
    Define a horizontal grid of points x and y. This function uses numpy.broadcast_to to avoid copies
    of repeating coordinates (see https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html).

    parameters
    ----------
    area : list
        List of min x, max x, min y and max y.
    shape : tuple
        Tuple defining the total number of points along x and y directions, respectively.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y). 
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
        Default is 'xy'.

    returns
    -------
    x, y : numpy arrays 1d
        Numpy arrays 1d containing the coordinates x and y of the grid points.
        These vectors are 
    """