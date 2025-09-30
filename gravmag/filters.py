import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from . import utils, check, data_structures


def direction(wavenumbers, inc, dec, check_input=True):
    """
    Compute the 2D directional derivative filter associated with the real
    unit vector u (Gunn, 1975; Blakely, 1996, p. 278).

    parameters
    ----------
    wavenumbers: dictionary
        Dictionary containing the metadata of the wavenumber grid
        (See description at function 'data_structures.regular_grid_wavenumbers').
    inc, dec: scalars
        Inclination and declination of the unit vector (in degrees)
        defining the direction filter.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    theta: numpy array 2D
        Directional derivative filter evaluated at the
        wavenumbers kx, ky and kz.
    """

    if check_input is True:
        check.is_grid_wavenumbers(wavenumbers)
        check.is_scalar(x=inc, positive=False)
        check.is_scalar(x=dec, positive=False)

    # define unit vector
    u = utils.unit_vector(inc, dec, check_input=False)

    # compute the filter
    theta = (wavenumbers["z"] * u[2]) + 1j * (
        wavenumbers["x"][:, np.newaxis] * u[0] + wavenumbers["y"] * u[1]
    )

    return theta


def rtp(wavenumbers, inc0, dec0, inc, dec, check_input=True):
    """
    Compute the reduction to the pole filter (Gunn, 1975; Blakely, 1996, p. 330-331).

    parameters
    ----------
    wavenumbers: dictionary
        Dictionary containing the metadata of the wavenumber grid
        (See description at function 'data_structures.regular_grid_wavenumbers').
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
        check.is_grid_wavenumbers(wavenumbers)
        check.is_scalar(x=inc0, positive=False)
        check.is_scalar(x=dec0, positive=False)
        check.is_scalar(x=inc, positive=False)
        check.is_scalar(x=dec, positive=False)

    # compute the direction filter for the main field
    theta_main_field = direction(wavenumbers, inc0, dec0, check_input=False)

    # compute the direction filter for the total magnetization of the sources
    theta_magnetization = direction(wavenumbers, inc, dec, check_input=False)

    # theta_main_field[0,0] and theta_magnetization[0,0] are zero and
    # it causes a division-by-zero problem. Because of that, we
    # set theta_main_field[0,0] and theta_magnetization[0,0] equal to 1
    theta_main_field[0, 0] = 1.0
    theta_magnetization[0, 0] = 1.0
    rtp_filter = (wavenumbers["z"] * wavenumbers["z"]) / (
        theta_main_field * theta_magnetization
    )

    return rtp_filter


def derivative(wavenumbers, axes, check_input=True):
    """
    Compute the derivative filter (Gunn, 1975; Blakely, 1996, p. 323-324).

    parameters
    ----------
    wavenumbers: dictionary
        Dictionary containing the metadata of the wavenumber grid
        (See description at function 'data_structures.regular_grid_wavenumbers').
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
        check.is_grid_wavenumbers(wavenumbers)
        if len(axes) <= 0:
            raise ValueError("axes must have at least one element")
        for axis in axes:
            if axis not in ["x", "y", "z"]:
                raise ValueError("invalid axis {}".format(axis))

    # matrix "views' of the original wavenumbers
    KX, KY = data_structures.grid_xy_to_full_matrices_view(
        x=wavenumbers["x"], y=wavenumbers["y"], shape=wavenumbers["shape"]
    )

    # define only the derivatives along the axes contained in 'axes'
    exponents = [axes.count("x"), axes.count("y"), axes.count("z")]
    deriv_filter = []
    if exponents[0] > 0:
        deriv_filter.append((1j * KX) ** exponents[0])
    if exponents[1] > 0:
        deriv_filter.append((1j * KY) ** exponents[1])
    if exponents[2] > 0:
        deriv_filter.append(wavenumbers["z"] ** exponents[2])
    deriv_filter = np.prod(deriv_filter, axis=0)

    return deriv_filter


def continuation(wavenumbers, dz, regularization=0., check_input=True):
    """
    Compute the level-to-level upward/downward continuation filter
    (Gunn, 1975; Blakely, 1996, p. 317) with regularization defined
    by Pašteka et al. (2012).

    parameters
    ----------
    wavenumbers: dictionary
        Dictionary containing the metadata of the wavenumber grid
        (See description at function 'data_structures.grid_wavenumbers').
    dz : int or float
        Scalar defining the difference between the constant vertical coordinate
        of the continuation plane and the constant vertical coordinate of the
        original data. Negative and positive values define upward and downward
        continuations, respectively.
    regularization : int or float
        Positive scalar defining the regularization parameter for continuation
        operator (Pašteka et al, 2012).
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    result : numpy array 2D
        Resultant filter evaluated at the wavenumber kz.
    """

    if check_input is True:
        check.is_grid_wavenumbers(wavenumbers)
        check.is_scalar(x=dz, positive=True)
        check.is_scalar(x=regularization)
        if regularization < 0.:
            raise ValueError('regularization must be zero or positive')

    # compute the continuation filter
    cont_filter = np.exp(dz * wavenumbers["z"])

    if regularization == 0:
        result = cont_filter
    else:
        aux = regularization * (wavenumbers["z"]**2) * cont_filter
        result = cont_filter / (1 + aux)

    return result

7
def cuttof_frequency(wavenumbers, max_freq, check_input=True):
    """
    Compute a simple low-pass filter.

    parameters
    ----------
    wavenumbers: dictionary
        Dictionary containing the metadata of the wavenumber grid
        (See description at function 'data_structures.regular_grid_wavenumbers').
    max_freq : int or float
        Scalar defining the maximum frequency of the filtered data..
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    filter : numpy array 2D
        Low-pass filter evaluated at the wavenumber kz.
    """

    if check_input is True:
        check.is_grid_wavenumbers(wavenumbers)
        check.is_scalar(x=max_freq, positive=True)

    dead_zone = wavenumbers["z"] >= max_freq
    cutoff_filter = np.ones_like(wavenumbers["z"])
    cutoff_filter[dead_zone] = 0.0

    return cutoff_filter
