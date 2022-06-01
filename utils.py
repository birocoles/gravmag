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
    inc = np.deg2rad(magnetization[:,1])
    dec = np.deg2rad(magnetization[:,2])
    # compute the sines and cosines
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)
    cos_dec = np.cos(dec)
    sin_dec = np.sin(dec)
    # compute the Cartesian components
    mx = magnetization[:,0]*cos_inc*cos_dec
    my = magnetization[:,0]*cos_inc*sin_dec
    mz = magnetization[:,0]*sin_inc
    return mx, my, mz
