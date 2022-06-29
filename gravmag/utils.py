import numpy as np
from numba import njit


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
    if x != 0:
        result = np.arctan(y / x)
    else:
        if y > 0:
            result = np.pi / 2
        elif y < 0:
            result = -np.pi / 2
        else:
            result = 0
    return result


@njit
def safe_log(x):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0.
    """
    if np.abs(x) < 1e-10:
        result = 0
    else:
        result = np.log(x)
    return result


def magnetization_components(magnetization):
    """
    Given the total-magnetization intensity, inclination and declination,
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
