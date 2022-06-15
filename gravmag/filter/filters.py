import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq


def direction(kx, ky, kz, u):
    '''
    Compute the 2D direction filter associated with the
    unit vector u.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions.
    u: numpy array 1D
            Unit vector.

    returns
    -------
    theta: numpy array 2D
        Direction filter evaluated at the wavenumbers kx, ky and kz.
    '''
    assert kx.ndim == ky.ndim == kz.ndim == 2, 'kx, ky and kz must be matrices'
    assert np.all(kz >= 0), 'kz elements must be >= 0'
    assert (u.ndim == 1) and (u.size == 3), 'u must be a 3-component vector'
    assert np.allclose(np.sum(u*u), 1), 'u must be a unit vector'

    with np.errstate(divide='ignore', invalid='ignore'):
        theta = u[2] + 1j*((kx*u[0] + ky*u[1])/kz)

    return theta
