import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from . import utils
from . import check


def wavenumbers(shape, dx, dy, check_input=True):
    """
    Compute the wavenumbers kx, ky and kz associated with a regular
    grid of data.

    parameters
    ----------
    dx, dy : floats
        Grid spacing along x and y directions.
    shape : tuple or list
        Sequence containing the number of points of data grid
        along x and y directions.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions.
    """

    if check_input is True:
        assert np.isscalar(dx) and (dx > 0), "dx must be a positive scalar"
        assert np.isscalar(dy) and (dy > 0), "dy must be a positive scalar"
        assert len(shape) == 2, "shape must have 2 elements"
        assert isinstance(shape[0], int) and (
            shape[0] > 0
        ), "shape[0] must \
        be a positive integer"
        assert isinstance(shape[1], int) and (
            shape[1] > 0
        ), "shape[1] must \
        be a positive integer"

    # wavenumbers kx = 2pi fx and ky = 2pi fy
    kx = 2 * np.pi * fftfreq(n=shape[0], d=dx)
    ky = 2 * np.pi * fftfreq(n=shape[1], d=dy)
    ky, kx = np.meshgrid(ky, kx)

    # this is valid for potential fields on a plane
    kz = np.sqrt(kx ** 2 + ky ** 2)

    return kx, ky, kz


def direction(kx, ky, kz, inc, dec, check_input=True):
    """
    Compute the 2D direction filter associated with the real
    unit vector u.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions computed according to
        function 'wavenumbers'.
    inc, dec: scalars
        Inclination and declination of the unit vector (in degrees)
        defining the direction filter.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    theta: numpy array 2D
        Direction filter evaluated at the wavenumbers kx, ky and kz.
    """

    if check_input is True:
        check.wavenumbers(kx, ky, kz)
        assert np.isscalar(inc), "inc must be a scalar"
        assert np.isscalar(dec), "dec must be a scalar"

    # define unit vector
    u = utils.unit_vector(inc, dec, check_input=False)

    # kz[0,0] is zero and raises a division-by-zero problem
    # because of that, we set kz[0,0] == 1 to compute the filter
    kz[0, 0] = 1

    # compute the filter
    theta = u[2] + 1j * (kx * u[0] + ky * u[1]) / kz

    # return kz[0,0] to zero
    kz[0, 0] = 0.0

    return theta


def rtp(kx, ky, kz, inc0, dec0, inc, dec, check_input=True):
    """
    Compute the reduction to the pole filter.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions computed according to
        function 'wavenumbers'.
    inc0, dec0: scalars
        Constant inclination and declination (in degrees) of the main
        geomagnetic field.
    inc, dec: scalars
        Constant inclination and declination (in degrees) of the sources
        total-magnetization.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    filter : numpy array 2D
        RTP filter evaluated at the wavenumbers kx, ky and kz.
    """

    if check_input is True:
        check.wavenumbers(kx, ky, kz)
        assert np.isscalar(inc0), "inc0 must be a scalar"
        assert np.isscalar(dec0), "dec0 must be a scalar"
        assert np.isscalar(inc), "inc must be a scalar"
        assert np.isscalar(dec), "dec must be a scalar"

    # compute the direction filter for the main field
    theta_main_field = direction(kx, ky, kz, inc0, dec0, check_input=False)

    # compute the direction filter for the total magnetization of the sources
    theta_magnetization = direction(kx, ky, kz, inc, dec, check_input=False)

    # theta_main_field[0,0] and theta_magnetization[0,0] are zero and
    # it causes a division-by-zero problem. Because of that, we
    # set theta_main_field[0,0] and theta_magnetization[0,0] equal to 1
    theta_main_field[0, 0] = 1.0
    theta_magnetization[0, 0] = 1.0
    filter = 1 / (theta_main_field * theta_magnetization)

    return filter


def derivative(kx, ky, kz, axes, check_input=True):
    """
    Compute the derivative filter.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions computed according to
        function 'wavenumbers'.
    axes : list or tuple of strings
        Sequence of strings defining the axes along which the partial derivative
        will be computed. Possible values are 'x', 'y' or 'z'.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    filter : numpy array 2D
        Derivative filter evaluated at the wavenumbers kx, ky and kz.
    """
    if check_input is True:
        check.wavenumbers(kx, ky, kz)
        assert len(axes) > 0, "axes must have at least one element"
        for axis in axes:
            assert axis in ["x", "y", "z"], "invalid axis {}".format(axis)

    # define only the derivatives along the axes contained in 'axes'
    exponents = [axes.count("x"), axes.count("y"), axes.count("z")]
    filter = []
    if exponents[0] > 0:
        filter.append((1j * kx) ** exponents[0])
    if exponents[1] > 0:
        filter.append((1j * ky) ** exponents[1])
    if exponents[2] > 0:
        filter.append(kz ** exponents[2])
    filter = np.prod(filter, axis=0)

    return filter
