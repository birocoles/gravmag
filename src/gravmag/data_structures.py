import numpy as np
from scipy.fft import fftfreq, fftshift
from . import check, utils


def grid_xy(area, shape, z0, check_input=True):
    """
    Define the data structure for a horizontal grid of points x and y.

    parameters
    ----------
    area : list
        List of min x, max x, min y and max y.
    shape : tuple
        Tuple defining the total number of points along x and y directions, respectively.
    z0 : scalar
        Constant vertical coordinate of the grid.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    grid : dictionary containing the following keys
        'x' : numpy array 1d with shape = (Nx, ), where Nx is the number of data along x-axis.
        'y' : numpy array 1d with shape = (Ny, ), where Ny is the number of data along y-axis.
        'z' : scalar (float or int) defining the constant vertical coordinate of the grid.
        'area' : list
            List of min x, max x, min y and max y (the same as input)
        'shape' : tuple
            Tuple defining the total number of points along x and y directions,
            respectively (the same as input).
    """
    if check_input == True:
        check.is_area(area=area)
        check.is_shape(shape=shape)
        check.is_scalar(x=z0, positive=False)

    grid = {
        "x": np.linspace(area[0], area[1], shape[0]),
        "y": np.linspace(area[2], area[3], shape[1]),
        "z": z0,
        "area": area,
        "shape": shape,
    }

    return grid


def grid_xy_to_full_flatten(grid, grid_orientation, check_input=True):
    """
    Compute the full grid from the metadata obtained from 'data_structures.grid_xy'.
    The coordinates are collapsed into one dimension, according to the given 'grid_orientation'.

    parameters
    ----------
    grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'grid_orientation'. Output of the function 'data_structures.grid_xy'.
    grid_orientation : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    full_grid : dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z', respectively.
        All keys are numpy arrays 1d having the same number of elements.
    """
    if check_input == True:
        check.is_grid_xy(grid)
        check.is_grid_orientation(grid_orientation)

    if grid_orientation == "xy":
        x, y = np.meshgrid(grid["x"], grid["y"], indexing="xy")
    else:  # grid_orientation == 'yx'
        x, y = np.meshgrid(grid["x"], grid["y"], indexing="ij")

    N = grid["shape"][0] * grid["shape"][1]
    full_grid = {
        "x": np.ravel(x),
        "y": np.ravel(y),
        "z": np.zeros(N, dtype=float) + grid["z"],
    }

    return full_grid


def grid_xy_full_flatten_to_matrix(
    data, grid_orientation, shape, check_input=True
):
    """
    Let a 'data' vector be computed at a grid of points with a given 'grid_orientation' and 'shape'.
    The present function reshape the 'data' into a matrix having the given 'shape', according to the
    'grid_orientation' of its corresponding grid of points.

    parameters
    ----------
    data : numpy array 1d
        Data vector.
    grid_orientation : string
        Defines how the points are ordered after the first point (min x, min y) in the
        corresponding grid of points.
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    shape : tuple
        Tuple defining the total number of points along x and y directions, respectively.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    data_matrix : numpy array 2d
        Data vector rearranged into a matrix according to the 'grid_orientation' of its
        corresponding grid of points.
    """
    if check_input == True:
        check.is_array(x=data, ndim=1)
        check.is_grid_orientation(grid_orientation)
        check.is_shape(shape)
        if shape[0] * shape[1] != data.size:
            raise Valuerror("shape mismatch data")

    if grid_orientation == "xy":
        return np.reshape(data, shape[::-1]).T
    else:  # grid_orientation == 'yx'
        return np.reshape(data, shape)


def grid_xy_full_matrix_to_flatten(grid, grid_orientation, check_input=True):
    """
    Rearrange a full grid matrix into a flattened array 1d according to the given 'grid_orientation'.

    parameters
    ----------
    grid : numpy array 2d
        Full grid of points.
    grid_orientation : string
        Defines how the points are ordered after the first point (min x, min y) in the
        corresponding grid of points.
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    data : numpy array 1d
        Full grid rearranged into a 1d array according to the given 'grid_orientation'.
    """
    if check_input == True:
        check.is_array(x=grid, ndim=2)
        check.is_grid_orientation(grid_orientation)

    if grid_orientation == "xy":
        return grid.T.ravel()
    else:  # grid_orientation == 'yx'
        return grid.ravel()


def grid_xy_to_full_matrices_view(x, y, shape, check_input=True):
    """
    Broadcast to matrices the coordinates 'x' and 'y' of a 'grid' with given 'shape',
    according to the given 'grid_orientation'.

    parameters
    ----------
    'x' : numpy array 1d
        Array with shape = (Nx, ), where Nx is the number of data along x-axis.
    'y' : numpy array 1d
        Array with shape = (Ny, ), where Ny is the number of data along y-axis.
    shape : tuple
        Tuple defining the total number of points along x and y directions, respectively.
    grid_orientation : string
        Defines how the points are ordered after the first point (min x, min y) in the
        corresponding grid of points.
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    X, Y : numpy arrays 2d
        Views of the grid coordinates.
    """
    if check_input == True:
        check.is_shape(shape=shape)
        check.is_array(x=x, ndim=1, shape=(shape[0],))
        check.is_array(x=y, ndim=1, shape=(shape[1],))

    X = np.broadcast_to(x, shape[::-1]).T
    Y = np.broadcast_to(y, shape)

    return X, Y


def grid_xy_spacing(area, shape, check_input=True):
    """
    Compute the grid spacing along the x and y directions.
    The grid spacing between N data points is defined by the ratio of
    'total extension' and 'N - 1', where 'N - 1' is the number of
    intervals between the N data points.

    parameters
    ----------
    area : list
        List of min x, max x, min y and max y.
    shape : tuple
        Tuple defining the total number of points along x and y directions, respectively.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    spacing : tuple
        Tuple containing the grid spacing along the x and y directions, respectively.
    """
    if check_input == True:
        check.is_area(area=area)
        check.is_shape(shape=shape)

    spacing = (
        (area[1] - area[0]) / (shape[0] - 1),
        (area[3] - area[2]) / (shape[1] - 1),
    )

    return spacing


def grid_wavenumbers(grid, pad_size=None, check_input=True):
    """
    Compute the wavenumbers associated with a regular grid of data.

    parameters
    ----------
    grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'grid_orientation'. Output of the function 'data_structures.grid_xy'.
    pad_size : integer or None
        If not None, it defines the size of padding along axes
        0 (rows) and 1 (columns). The shape of the padded data is
        (data.shape[0] * (2 * pad_size + 1), data.shape[1] * (2 * pad_size + 1)).
        Default is 1.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    wavenumbers: dictionary containing the following keys
        'x' : numpy array 1d
            Vector with shape = (Nx, ), where Nx is the number of
            data long x-axis, if 'pad' is False, or the number of padded data, if 'pad' is True.
            This numpy array contains the discrete wavenumbers along the x-axis.
        'y' : numpy array 1d
            Vector with shape = (Ny, ), where Ny is the number of
            data long y-axis, if 'pad' is False, or the number of padded data, if 'pad' is True.
            This numpy array contains the discrete wavenumbers along the y-axis.
        'z' : numpy array 2d
            Matrix with shape (kx.size, ky.size) containing the wavenumbers along
            the z-axis by considering that the generating data grid in space domain
            contains potential-field data on a horizontal plane.
        'shape' : tuple
            If 'pad' is False, it returns the parameter 'shape' of the given 'grid'.
            Otherwise, it returns the 'shape' of the padded 'grid'.
        'spacing' : tuple
            The input parameter 'spacing'
    """

    if check_input is True:
        check.is_grid_xy(grid=grid)
        if pad_size is not None:
            check.is_integer(x=pad_size, positive=True)

    # get the original shape and area
    shape = grid["shape"]
    area = grid["area"]
    # compute the grid spacing
    spacing = grid_xy_spacing(area=area, shape=shape, check_input=False)
    # redefine 'shape' according to 'pad'
    if pad_size is not None:
        shape = ((2 * pad_size + 1) * shape[0], (2 * pad_size + 1) * shape[1])

    # wavenumbers kx = 2pi fx and ky = 2pi fy
    kx = 2 * np.pi * fftfreq(n=shape[0], d=spacing[0])
    ky = 2 * np.pi * fftfreq(n=shape[1], d=spacing[1])

    # this is valid for potential fields on a plane
    # the line below generates a numpy array 2d with shape (kx.size, ky.size)
    KX, KY = grid_xy_to_full_matrices_view(
        x=kx, y=ky, shape=shape, check_input=False
    )
    kz = np.sqrt(KX**2 + KY**2)

    # shift the wavenumbers so that their values goes from negative to positive values
    # kx = fftshift(kx)
    # ky = fftshift(ky)
    # kz = fftshift(kz)

    wavenumbers = {
        "x": kx,
        "y": ky,
        "z": kz,
        "shape": shape,
        "spacing": spacing,
    }

    return wavenumbers
