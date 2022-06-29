import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift


def wavenumbers(shape, dx, dy, check_input=True):
    """
    Compute the wavenumbers kx, ky and kz associated with a regular
    grid of data.

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
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions.
    """

    if check_input is True:
        assert isinstance(shape, tuple), "shape must be a tuple"
        assert len(shape) == 2, "shape must have 2 elements"
        assert isinstance(shape[0], int) and (
            shape[0] > 0
        ), "shape[0] must be a positive integer"
        assert isinstance(shape[1], int) and (
            shape[1] > 0
        ), "shape[1] must be a positive integer"
        assert isinstance(dx, (float, int)) and (
            dx > 0
        ), "dx must be a positive scalar"
        assert isinstance(dy, (float, int)) and (
            dy > 0
        ), "dy must be a positive scalar"

    # wavenumbers kx = 2pi fx and ky = 2pi fy
    kx = 2 * np.pi * fftfreq(n=shape[0], d=dx)
    ky = 2 * np.pi * fftfreq(n=shape[1], d=dy)
    ky, kx = np.meshgrid(ky, kx)

    # this is valid for potential fields on a plane
    kz = np.sqrt(kx ** 2 + ky ** 2)

    return kx, ky, kz


def DFT(data, pad_mode=None, check_input=True):
    """
    Compute the Discrete Fourier Transform (DFT) of a potential-field data set
    arranged as regular grid on a horizontal surface.

    parameters
    ----------
    data : numpy array 2D
        Matrix containing the regular grid of potential-field data.
    pad_mode : None or string
        If not None, it defines the method available in the routine 'numpy.pad'
        to apply padding. Default is None.
    check_input : boolean
        If True, it verifies if the input is valid. Default is True.

    returns
    -------
    FT_data : numpy array 2D
        DFT of the potential-field data grid. If "pad_width" is not None, then
        the shape of FT_data will be greater than that of the input 'data'.
    """

    # convert data to numpy array
    data = np.asarray(data)

    if check_input is True:
        assert data.ndim == 2, "data must be a matrix"
        assert isinstance(
            pad_mode, (type(None), str)
        ), "pad_mode must be None or a string (see the routine numpy.pad)"

    if pad_mode is not None:
        # define the padded data
        data_padded = np.pad(
            data, pad_width=((data.shape[0],), (data.shape[1],)), mode=pad_mode
        )
        # compute the 2D DFT of the padded data using the Fast Fourier
        # Transform algorithm implemented at scipy.fft.fft2
        FT_data = fft2(data_padded)
    else:
        # compute the 2D DFT of the original data using the Fast Fourier
        # Transform algorithm implemented at scipy.fft.fft2
        FT_data = fft2(data)

    return FT_data


def IDFT(FT_data, unpad=False, grid=True, check_input=True):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of a potential-field
    data set arranged as regular grid on a horizontal surface.

    parameters
    ----------
    FT_data : numpy array 2D
        Matrix containing the DFT of a potential-field data set arranged as
        regular grid on a horizontal surface.
    unpad : boolean
        If True, remove the padding applied according to the function 'DFT'.
    grid : boolean
        Defines the output shape. If True, the output has the same shape as the
        input. If False, then the output is a flattened 1D array.
    check_input : boolean
        If True, it verifies if the input is valid. Default is True.

    returns
    -------
    IFT_data : numpy array 2D
        Matrix containing the regular grid of potential-field data obtained
        from IDFT.
    """

    # convert data to numpy array
    FT_data = np.asarray(FT_data)

    if check_input is True:
        assert np.iscomplexobj(FT_data), "FT_data must be a complex array"
        assert FT_data.ndim == 2, "FT_data must be a matrix"
        assert isinstance(unpad, bool), "pad_width must be True or False"
        assert isinstance(grid, bool), "grid must be True or False"

    # compute the 2D IDFT of FT_data using the Fast Fourier
    # Transform algorithm implemented at scipy.fft.ifft2
    IFT_data = ifft2(FT_data).real

    if unpad is True:
        # define number of values padded to the edges of original data
        pad_width = (FT_data.shape[0] // 3, FT_data.shape[1] // 3)
        # remove the padding
        IFT_data = IFT_data[
            pad_width[0] : 2 * pad_width[0], pad_width[1] : 2 * pad_width[1]
        ]

    if grid is False:
        IFT_data = IFT_data.ravel()

    return IFT_data
