import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from . import utils
from . import constants as cts


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
        assert isinstance(unpad, bool), "unpad must be True or False"
        assert isinstance(grid, bool), "grid must be True or False"

    # compute the 2D IDFT of FT_data using the Fast Fourier
    # Transform algorithm implemented at scipy.fft.ifft2
    IFT_data = ifft2(FT_data).real

    if unpad is True:
        IFT_data = _unpad(IFT_data)

    if grid is False:
        IFT_data = IFT_data.ravel()

    return IFT_data


def spectra(
    FT_data,
    shift=True,
    types=["amplitude", "phase", "power"],
    check_input=True,
):
    """
    Compute the amplitude, phase and/or power spectra of a potential-field data
    set arranged as regular grid on a horizontal surface.

    parameters
    ----------
    FT_data : numpy array 2D
        Matrix containing the DFT of a potential-field data set arranged as
        regular grid on a horizontal surface.
    shift : boolean
        If True, swaps half-spaces for x and y directions so that the
        zero-frequency component is placed to the center of the spectrum.
    types : list of strings
        List of strings defining the spectra to be computed.
        Default is ['amplitude', 'phase', 'power'], which contains all available
        spectra. Repeated types are ignored.
    check_input : boolean
        If True, it verifies if the input is valid. Default is True.

    returns
    -------
    spectra_list : list of numpy arrays 2D
        Matrices containing the computed spectra.
    """

    # convert data to numpy array
    FT_data = np.asarray(FT_data)

    if check_input is True:
        assert np.iscomplexobj(FT_data), "FT_data must be a complex array"
        assert FT_data.ndim == 2, "FT_data must be a matrix"
        assert isinstance(shift, bool), "shift must be True or False"
        # check number of elements in types
        if len(types) > 3:
            raise ValueError("types must have at most 3 elements")
        # convert types to array of strings
        # repeated elements are ignored
        # the code below removes possibly duplicated elements in types
        _, _indices = np.unique(np.asarray(types, dtype=str), return_index=True)
        _types = np.array(types)[np.sort(_indices)]
        # check if types are valid
        for t in _types:
            if t not in ["amplitude", "phase", "power"]:
                raise ValueError("invalid type {}".format(t))
    else:
        _types = types

    spectra_list = []

    for t in _types:
        if t == "amplitude":
            spectra_list.append(np.abs(FT_data))
        elif t == "phase":
            spectra_list.append(np.angle(FT_data, deg=True))
        else:  # type is "power"
            spectra_list.append(np.abs(FT_data) ** 2)

    if shift is True:
        # swaps half-spaces for x and y directions so that the zero-frequency
        # component is placed to the center of the spectrum.
        for i in range(len(spectra_list)):
            spectra_list[i] = fftshift(spectra_list[i])

    return spectra_list


def _unpad(data):
    """
    Remove padded values at the edges of data.
    """
    # define number of values padded to the edges of original data
    pad_width = (data.shape[0] // 3, data.shape[1] // 3)
    # remove padded values ate the edges of data
    data = data[
        pad_width[0] : 2 * pad_width[0], pad_width[1] : 2 * pad_width[1]
    ]

    return data


# def FT_magnetized_rectangular_prism(
#     kx, ky, kz, prisms, magnetization, inct, dect
# ):
#     """
#     Prototype of closed-form spectrum associated with the approximated
#     total-field anomaly produced by a rectangular prism (Bhattacharyya, 1966)

#     Bhattacharyya, B. K., 1966. Continuous spectrum of the
#     total-magnetic-field anomaly due to a rectangular prismatic body,
#     Geophysics, 31, 97-121. https://doi.org/10.1190/1.1439767
#     """

#     result = np.zeros(kx.shape, dtype="complex")

#     # unit vector associated with constant main field
#     l, m, n = utils.unit_vector(inct, dect)
#     # unit vectors associated with the constant total magnetization
#     # of each prism
#     mx, my, mz = utils.magnetization_components(magnetization)

#     # iterate over prisms
#     for prism, L, M, N, Ip in zip(prisms, mx, my, mz, magnetization[:, 0]):
#         # horizontal dimension x
#         b0 = prism[1] - prism[0]
#         # horizontal dimension y
#         c0 = prism[3] - prism[2]
#         # geometrical factor depending on the horizontal dimensions
#         B = np.zeros_like(kx)
#         B[1:, 1:] = (
#             4 * np.sin(kx[1:, 1:] * b0 / 2) * np.sin(ky[1:, 1:] * c0 / 2)
#         ) / (kx[1:, 1:] * ky[1:, 1:])
#         # auxiliary constants
#         a12 = L * m + M * l
#         a13 = L * n + N * l
#         a23 = M * n + N * m
#         # dimensionless factor dependent on the main
#         # field and total-magnetization directions
#         kz[0, 0] = 1.0
#         D_real = (
#             -l * L * kx ** 2 - m * M * ky ** 2 + n * N * kz ** 2 - a12 * kx * ky
#         ) / kz ** 2
#         D_imag = (a13 * kx + a23 * ky) / kz
#         D = D_real + 1j * D_imag
#         kz[0, 0] = 0.0
#         # geometrical factor dependent on the top and bottom of the prism
#         H = np.exp(-kz * prism[4]) - np.exp(-kz * prism[5])
#         # Foutier transform of the prism
#         result += 2 * np.pi * B * D * H * Ip

#     return result


# def FT_grav_potential_rectangular_prism(
#     shape, dx, dy, prisms, density, z0, scale=True
# ):
#     """
#     Prototype of closed-form spectrum associated with the gravitational
#     potential produced by a rectangular prism (Bhattacharyya, 1966)

#     Bhattacharyya, B. K., 1966. Continuous spectrum of the
#     total-magnetic-field anomaly due to a rectangular prismatic body,
#     Geophysics, 31, 97-121. https://doi.org/10.1190/1.1439767
#     """

#     # compute the wavenumbers
#     kx, ky, kz = wavenumbers(shape, dx, dy)

#     result = np.zeros(shape, dtype="complex")

#     # iterate over prisms
#     for prism, rho in zip(prisms, density):
#         # horizontal dimension x
#         b0 = prism[1] - prism[0]
#         # horizontal dimension y
#         c0 = prism[3] - prism[2]
#         # top of the prism
#         ht = prism[4] - z0
#         # bottom of the prism
#         hb = prism[5] - z0
#         # horizontal coordinates of the center
#         xc = 0.5 * (prism[1] + prism[0])
#         yc = 0.5 * (prism[3] + prism[2])

#         # geometrical factor depending on the horizontal dimensions
#         B = np.zeros_like(kx)
#         B[1:, 1:] = (
#             4 * np.sin(kx[1:, 1:] * b0 / 2) * np.sin(ky[1:, 1:] * c0 / 2)
#         ) / (kx[1:, 1:] * ky[1:, 1:])
#         # B = (
#         #     4*np.sin(kx*b0/2)*np.sin(ky*c0/2)
#         # )/(kx*ky)
#         kz[0, 0] = 1.0
#         B /= kz ** 2
#         kz[0, 0] = 0.0
#         # geometrical factor dependent on the top and bottom of the prism
#         H = np.exp(-kz * ht) - np.exp(-kz * hb)
#         # shifting factor
#         # S = np.exp(-1j*(kx*xc + ky*yc))
#         # S = np.exp(1j*(kx*xc + ky*yc))
#         # Fourier transform of the prism
#         # result += 2*np.pi*B*H*rho
#         result += B * H * rho / (2 * np.pi)
#         # result += 2*np.pi*B*H*S*rho

#     # reorganize wavenumbers according to fftshift
#     # result = fftshift(result)

#     # remove FFT normalization and restore amplitude
#     # result *= np.sqrt(shape[0]*shape[1])/(dx*dy)

#     # multiply the computed field by the corresponding scale factors
#     if scale is True:
#         result *= cts.GRAVITATIONAL_CONST

#     return result
