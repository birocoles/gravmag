import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from . import utils
from . import check


def wavenumbers(shape, dx, dy, check_input=True):
    """
    Compute the wavenumbers associated with a regular grid of data.

    parameters
    ----------
    dx, dy : floats
        Grid spacing along x and y directions.
    shape : tuple of ints
        Tuple containing the number of points of data grid
        along x and y directions.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    wnum: dictionary formed by numpy arrays 2D
        Dictionary having 3 keys, 'x', 'y' and 'z', representing the
        wavenumbers along 'x', 'y' and 'z' directions, respectively.
    """

    if check_input is True:
        if type(shape) != tuple:
            raise ValueError("shape must be a tuple")
        if len(shape) != 2:
            raise ValueError("shape must have 2 elements")
        check.is_integer(x=shape[0], positive=True)
        check.is_integer(x=shape[1], positive=True)
        check.is_scalar(x=dx, positive=True)
        check.is_scalar(x=dy, positive=True)

    # wavenumbers kx = 2pi fx and ky = 2pi fy
    kx = 2 * np.pi * fftfreq(n=shape[0], d=dx)
    ky = 2 * np.pi * fftfreq(n=shape[1], d=dy)
    ky, kx = np.meshgrid(ky, kx)

    # this is valid for potential fields on a plane
    kz = np.sqrt(kx**2 + ky**2)

    # shift the wavenumbers according to the FFT routines 
    kx = fftshift(kx)
    ky = fftshift(ky)
    kz = fftshift(kz)

    wavenumbers_dict = {
        'x': kx,
        'y': ky,
        'z': kz
    }

    return wavenumbers_dict


def direction(kx, ky, kz, inc, dec, check_input=True):
    """
    Compute the 2D directional derivative filter associated with the real
    unit vector u.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions computed according to
        function 'gravmag.filters.wavenumbers'.
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
        check.wavenumbers(kx, ky, kz)
        assert isinstance(inc, (float, int)), "inc must be a scalar"
        assert isinstance(dec, (float, int)), "dec must be a scalar"

    # define unit vector
    u = utils.unit_vector(inc, dec, check_input=False)

    # compute the filter
    theta = (kz * u[2]) + 1j * (kx * u[0] + ky * u[1])

    return theta


def rtp(kx, ky, kz, inc0, dec0, inc, dec, check_input=True):
    """
    Compute the reduction to the pole filter.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions computed according to
        function 'gravmag.filters.wavenumbers'.
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
        assert isinstance(inc0, (float, int)), "inc0 must be a scalar"
        assert isinstance(dec0, (float, int)), "dec0 must be a scalar"
        assert isinstance(inc, (float, int)), "inc must be a scalar"
        assert isinstance(dec, (float, int)), "dec must be a scalar"

    # compute the direction filter for the main field
    theta_main_field = direction(kx, ky, kz, inc0, dec0, check_input=False)

    # compute the direction filter for the total magnetization of the sources
    theta_magnetization = direction(kx, ky, kz, inc, dec, check_input=False)

    # theta_main_field[0,0] and theta_magnetization[0,0] are zero and
    # it causes a division-by-zero problem. Because of that, we
    # set theta_main_field[0,0] and theta_magnetization[0,0] equal to 1
    theta_main_field[0, 0] = 1.0
    theta_magnetization[0, 0] = 1.0
    filter = (kz * kz) / (theta_main_field * theta_magnetization)

    return filter


def derivative(kx, ky, kz, axes, check_input=True):
    """
    Compute the derivative filter.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions computed according to
        function 'gravmag.filters.wavenumbers'.
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


def continuation(kz, dz, check_input=True):
    """
    Compute the level-to-level upward/downward continuation filter.

    parameters
    ----------
    kz: numpy array 2D
        Wavenumber in z direction computed according to function
        'gravmag.filters.wavenumbers'.
    dz : int or float
        Scalar defining the difference between the constant vertical coordinate
        of the continuation plane and the constant vertical coordinate of the
        original data. Negative and positive values define upward and downward
        continuations, respectively.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    filter : numpy array 2D
        Continuation filter evaluated at the wavenumber kz.
    """

    if check_input is True:
        kz = np.asarray(kz)
        assert kz.ndim == 2, "kz must be a matrix"
        assert np.all(kz >= 0), "elements of kz must be >= 0"
        assert isinstance(dz, (int, float)), "dz must be int or float"

    filter = np.exp(dz * kz)

    return filter


def cuttof_frequency(kz, max_freq, check_input=True):
    """
    Compute a simple low-pass filter.

    parameters
    ----------
    kz: numpy array 2D
        Wavenumber in z direction computed according to function
        'gravmag.filters.wavenumbers'.
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
        kz = np.asarray(kz)
        assert kz.ndim == 2, "kz must be a matrix"
        assert np.all(kz >= 0), "elements of kz must be >= 0"
        assert isinstance(max_freq, (int, float)) and (
            max_freq > 0
        ), "max_freq must be a positive scalar"

    dead_zone = kz >= max_freq
    filter = np.ones_like(kz)
    filter[dead_zone] = 0.0

    return filter
