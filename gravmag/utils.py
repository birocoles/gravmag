import numpy as np
from numba import njit
from . import check


@njit
def safe_atan2_entrywise(y, x):
    """
    Principal value of the arctangent expressed as a two variable function

    This modification has to be made to the arctangent function so the
    gravitational field of the prism satisfies the Poisson's equation.
    Therefore, it guarantees that the fields satisfies the symmetry properties
    of the prism. This modified function has been defined according to
    Fukushima (2020, eq. 72).
    """
    if x != 0.0:
        result = np.arctan(y / x)
    else:
        if y > 0.0:
            result = np.pi / 2
        elif y < 0.0:
            result = -np.pi / 2
        else:
            result = 0.0
    return result


@njit
def safe_atan2(y, x):
    """
    Principal value of the arctangent expressed as a two variable function

    This modification has to be made to the arctangent function so the
    gravitational field of the prism satisfies the Poisson's equation.
    Therefore, it guarantees that the fields satisfies the symmetry properties
    of the prism. This modified function has been defined according to
    Fukushima (2020, eq. 72).
    """
    result = np.empty_like(x)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if x[i, j] != 0.0:
                result[i, j] = np.arctan(y[i, j] / x[i, j])
            else:
                if y[i, j] > 0.0:
                    result[i, j] = np.pi / 2
                elif y[i, j] < 0.0:
                    result[i, j] = -np.pi / 2
                else:
                    result[i, j] = 0.0
    return result


def safe_atan2_np(y, x):
    """
    Principal value of the arctangent expressed as a two variable function

    This modification has to be made to the arctangent function so the
    gravitational field of the prism satisfies the Poisson's equation.
    Therefore, it guarantees that the fields satisfies the symmetry properties
    of the prism. This modified function has been defined according to
    Fukushima (2020, eq. 72).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = np.zeros_like(y)

    # x != 0
    nonzero_x = x != 0
    result[nonzero_x] = np.arctan(y[nonzero_x] / x[nonzero_x])

    # x == 0
    zero_x = x == 0
    result[zero_x] = np.sign(y[zero_x]) * np.pi / 2
    return result


@njit
def safe_log_entrywise(x):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0.
    """
    if np.abs(x) < 1e-10:
        result = 0.0
    else:
        result = np.log(x)
    return result


@njit
def safe_log(x):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0.
    """
    result = np.empty_like(x)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if np.abs(x[i, j]) < 1e-10:
                result[i, j] = 0.0
            else:
                result[i, j] = np.log(x[i, j])
    return result


def safe_log_np(x):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0.
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    # abs(x) >= 1e-10
    indices_x = np.abs(x) >= 1e-10
    result[indices_x] = np.log(x[indices_x])
    return result


def magnetization_components(magnetization):
    """
    Given the total-magnetization (moment) intensity, inclination and declination,
    compute the Cartesian components mx, my and mz.
    Run ``check.rectangular_prisms_magnetization`` before.
    """
    # transform inclination and declination from degrees to radians
    inc = np.deg2rad(magnetization[:, 1])
    dec = np.deg2rad(magnetization[:, 2])
    # compute the sines and cosines
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)
    cos_dec = np.cos(dec)
    sin_dec = np.sin(dec)
    # compute the Cartesian components
    mx = magnetization[:, 0] * cos_inc * cos_dec
    my = magnetization[:, 0] * cos_inc * sin_dec
    mz = magnetization[:, 0] * sin_inc
    return mx, my, mz


def unit_vector(inc, dec, check_input=True):
    """
    Compute the Cartesian components of a unit vector
    as a function of its inclination inc and declination dec

    parameters
    ----------
    inc, dec: scalars
        Inclination and declination of the unit vector (in degrees)
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    vector: numpy array 1D
        Unit vector with inclination inc and declination dec
    """
    if check_input is True:
        assert isinstance(inc, (float, int)), "inc must be a scalar"
        assert isinstance(dec, (float, int)), "dec must be a scalar"

    # convert inclination and declination to radians
    I_rad = np.deg2rad(inc)
    D_rad = np.deg2rad(dec)

    # compute cosine and sine
    cosI = np.cos(I_rad)
    sinI = np.sin(I_rad)
    cosD = np.cos(D_rad)
    sinD = np.sin(D_rad)

    # compute vector components
    vector = np.array([cosI * cosD, cosI * sinD, sinI])

    return vector


def direction(vector, check_input=True):
    """
    Convert a 3-component vector to intensity, inclination and
    declination.

    parameters
    ----------
    vector : numpy array 1d
        Real vector with 3 elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    intensity, inclination, declination: floats - intensity,
        inclination and declination (in degrees).

    """
    vector = np.asarray(vector)
    if check_input is True:
        assert vector.ndim == 1, "vector must be a vector"
        assert vector.size == 3, "vector must have 3 elements"
    intensity = np.linalg.norm(vector)
    x, y, z = vector
    declination = np.rad2deg(np.arctan2(y, x))
    inclination = np.rad2deg(np.arcsin(z / intensity))
    return intensity, inclination, declination


def rotation_matrix(I, D, dI, dD):
    """
    Compute the rotation matrix transforming the unit vector
    with inclination I and declination D into the unit vector
    with inclination I + dI and declination D + dD.

    parameters
    ----------
    I, D: floats - inclination and declination (in degrees) of the
        unit vector to be rotated.
    dI, dD: floats - differences (in degrees) between the
        inclination and declination of the rotated and original unit
        vectors.

    returns
    -------
    R: numpy array 2d - rotation matrix.
    """
    I_rad = np.deg2rad(I)
    D_rad = np.deg2rad(D)

    cosI = np.cos(I_rad)
    sinI = np.sin(I_rad)
    cosD = np.cos(D_rad)
    sinD = np.sin(D_rad)

    dI_rad = np.deg2rad(dI)
    dD_rad = np.deg2rad(dD)

    cosdI = np.cos(dI_rad)
    sindI = np.sin(dI_rad)
    cosdD = np.cos(dD_rad)
    sindD = np.sin(dD_rad)

    I_dI_rad = np.deg2rad(I + dI)
    D_dD_rad = np.deg2rad(D + dD)

    cosI_dI = np.cos(I_dI_rad)
    sinI_dI = np.sin(I_dI_rad)
    cosD_dD = np.cos(D_dD_rad)
    sinD_dD = np.sin(D_dD_rad)

    r00 = sinD_dD * sinD + cosD_dD * cosdI * cosD
    r10 = -cosD_dD * sinD + sinD_dD * cosdI * cosD
    r20 = sindI * cosD
    r01 = -sinD_dD * cosD + cosD_dD * cosdI * sinD
    r11 = cosD_dD * cosD + sinD_dD * cosdI * sinD
    r21 = sindI * sinD
    r02 = -cosD_dD * sindI
    r12 = -sinD_dD * sindI
    r22 = cosdI

    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return R


def coordinate_transform(x, y, theta):
    """
    Compute the Cartesian coordinates (x',y') obtained by
    rotating the coordinates (x,y) with "theta" degrees.

    parameters
    ----------
    x, y: numpy array 2D - Cartesian coordinates to be rotated.
    theta: float - Rotation angle in degrees (positive clockwise).

    returns
    -------
    x_prime, y_prime: numpy arrays 2D - Rotated coordinates.
    u_prime, v_prime: floats - Horizontal componentes of the unit
        vector with direction defined by "theta".
    """
    assert x.shape == y.shape, "x and y must have the same shape"
    assert isinstance(theta, (float, int)), "theta must be a scalar"
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x0 = 0.5 * (np.max(x) + np.min(x))
    y0 = 0.5 * (np.max(y) + np.min(y))
    Dx = x - x0
    Dy = y - y0
    x_prime = cos_theta * Dx + sin_theta * Dy + x0
    y_prime = -sin_theta * Dx + cos_theta * Dy + y0
    u_prime = cos_theta * cos_theta + sin_theta * sin_theta
    v_prime = -sin_theta * cos_theta + cos_theta * sin_theta
    return x_prime, y_prime, u_prime, v_prime


def prisms_volume(prisms):
    """
    Compute the volume of each prism forming the model.

    parameters
    ----------
    prisms : 2d-array
        2d-array containing the coordinates of the prisms. Each line must contain
        the coordinates of a single prism in the following order:
        south (x1), north (x2), west (y1), east (y2), top (z1) and bottom (z2).
        All coordinates should be in meters.

    returns
    -------
    volume : 1d-array
        1d-array containing the volume of each prism in prisms.
    """

    # Verify the input parameters
    check.are_rectangular_prisms(prisms)

    volume = (
        (prisms["x2"] - prisms["x1"])
        * (prisms["y2"] - prisms["y1"])
        * (prisms["z2"] - prisms["z1"])
    )

    return volume


def block_data(x, y, area, shape):
    """
    Split a dataset into a grid of Nx X Ny blocks within a predefined area.

    parameters
    ----------
    x, y : numpy arrays 1d
        Vectors containing the x and y coordinates of a scattered set of points.
    area : list
        List formed by [x1, x2, y1, y2], where x2 > x1 and y2 > y1 define the
        boundaries along the x and y directions, respectively.
    shape : tuple of ints
        Positive integers defining the number of blocks along the x and y directions, respectively.

    returns
    -------
    blocks_indices : list of lists
        Lists containing the indices of the data at each block.
    """
    if (type(x) != np.ndarray) or (type(y) != np.ndarray):
        raise ValueError("x and y must be numpy arrays")
    if (x.ndim != 1) or (y.ndim != 1):
        raise ValueError("x and y must have ndim = 1")
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    if type(area) != list:
        raise ValueError("area must be a list")
    if len(area) != 4:
        raise ValueError("area must have four elements")
    if area[1] <= area[0]:
        raise ValueError("area[1] must be greater than area[0]")
    if area[3] <= area[2]:
        raise ValueError("area[3] must be greater than area[2]")
    if isinstance(shape, tuple) == False:
        raise ValueError("shape must be a tuple")
    if len(shape) != 2:
        raise ValueError("shape must have 2 elements")
    if (isinstance(shape[0], int) == False) or (shape[0] <= 0):
        raise ValueError("shape[0] must be a positive integer")
    if (isinstance(shape[1], int) == False) or (shape[1] <= 0):
        raise ValueError("shape[1] must be a positive integer")

    # compute spacing along x and y
    dx = (area[1] - area[0]) / shape[0]
    dy = (area[3] - area[2]) / shape[1]

    # reduced_data = np.empty(shape=(Nx,Ny), dtype=float)
    x_indices = np.array((x - area[0]) / dx, dtype=int)
    y_indices = np.array((y - area[2]) / dy, dtype=int)

    # blocks = shape[0]*[shape[1]*[[]]]
    blocks_indices = []
    for i in range(shape[0]):
        blocks_indices.append([])
    for block_row in blocks_indices:
        for j in range(shape[1]):
            block_row.append([])
    for index, (i, j) in enumerate(zip(x_indices, y_indices)):
        if (i < shape[0]) and (i >= 0) and (j < shape[1]) and (j >= 0):
            blocks_indices[i][j].append(index)

    return blocks_indices


def reduce_data(data, blocks_indices, function="mean", remove_nan=False):
    """
    Apply func to the values at each element in blocks.

    parameters
    ----------
    data : numpy array 1d
        Vector containing the data at the scattered set of points.
    blocks_indices : list of lists
        Lists containing the indices of the data at each block.
    function : string
        Function to be applied to compute the reduced data at the
        center of each block. The possibilities are "mean" or "median".
    remove_nan : boolean
        If True, keep the elements containing nan values and return a 2d array of reduced data.
        Otherwise, remove elements containing nan values and return an 1d array of reduced data.
        Default is False.

    returns
    -------
    reduced_data : numpy array 1d or 2d
        Matrix containing the reduced data at each block.
    """
    if type(data) != np.ndarray:
        raise ValueError("data must be numpy arrays")
    if data.ndim != 1:
        raise ValueError("data must have ndim = 1")
    if type(blocks_indices) != list:
        raise ValueError("blocks_indices must be a list")
    if type(function) != str:
        raise ValueError("function must be string")
    if function not in ["mean", "median"]:
        raise ValueError("invalid function {}".format(function))
    if remove_nan not in [True, False]:
        raise ValueError("{} does not have a valid format".format(remove_nan))

    if function == "mean":
        func = np.mean
    else:  # function == "median"
        func = np.median

    Nx = len(blocks_indices)
    Ny = len(blocks_indices[0])

    reduced_data = np.empty(shape=(Nx, Ny), dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            if len(blocks_indices[i][j]) == 0:
                reduced_data[i, j] = np.nan
            else:
                reduced_data[i, j] = func(data[blocks_indices[i][j]])

    if remove_nan == True:
        reduced_data = reduced_data.ravel()
        nan_elements = np.isnan(reduced_data)
        reduced_data = np.delete(reduced_data, nan_elements)

    return reduced_data
