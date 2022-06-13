'''
Algoritmhs for computing the gravitational field of point masses and the
magnetic induction field of dipoles.
'''

import numpy as np
import scipy.spatial as spt
from scipy.fft import fft2, ifft2
from .. import constants as cts
from .. import check


def sedm(P, S):
    '''
    Compute Squared Euclidean Distance Matrix (SEDM) between the observation
    points and the sources.

    parameters
    ----------
    P: numpy array 2d
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N observation points. The ith column contains the
        coordinates of the ith observation point.
    S: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.


    returns
    -------
    R2: numpy array 2d
        N x M SEDM between P and S.
    '''
    check.coordinates(P)
    check.coordinates(S)

    R2 = spt.distance.cdist(P.T, S.T, 'sqeuclidean')

    return R2


def first_derivatives(P, S, R2):
    '''
    Compute the first derivatives of the inverse distance function between the
    observation points and the sources.

    parameters
    ----------
    P: numpy array 2d
        3 x N matrix containing the coordinates
        x (1rt row), y (2nd row), z (3rd row) of N observation points.
        The ith column contains the coordinates of the ith observation point.
    S: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.
    R2: numpy array 2d
        Squared Euclidean distance matrix between the N observation points P
        and the M equivalent sources.


    returns
    -------
    Kx, Ky, Kz: numpy arrays 2d
        N x M matrices containing the first derivatives x, y, z.
    '''

    check.coordinates(P)
    check.coordinates(S)
    assert R2.shape == (P.shape[1], S.shape[0]), 'R2 does not match P and S'

    R3 = R2*np.sqrt(R2)

    X = P[0][:, np.newaxis] - S[0]
    Y = P[1][:, np.newaxis] - S[1]
    Z = P[2][:, np.newaxis] - S[2]

    Kx = -X/R3
    Ky = -Y/R3
    Kz = -Z/R3

    return Kx, Ky, Kz


def second_derivatives(P, S, R2):
    '''
    Compute the second derivatives of the inverse
    distance function between the observation points
    and the sources.

    parameters
    ----------
    P: numpy array 2d
        3 x N matrix containing the coordinates
        x (1rt row), y (2nd row), z (3rd row) of N observation points.
        The ith column contains the coordinates of the ith observation point.
    S: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.
    R2: numpy array 2d
        Squared Euclidean distance matrix between the N observation points P
        and the M equivalent sources.


    returns
    -------
    Kxx, Kxy, Kxz, Kyy, Kyz: numpy arrays 2d
        N x M matrices containing the second derivatives xx, xy, xz, yy and yz
        (zz is not computed because it can be obtained from xx and yy).
    '''

    check.coordinates(P)
    check.coordinates(S)
    assert R2.shape == (P.shape[1], S.shape[0]), 'R2 does not match P and S'

    R3 = R2*np.sqrt(R2)
    R5 = R3*R2

    X = P[0][:, np.newaxis] - S[0]
    Y = P[1][:, np.newaxis] - S[1]
    Z = P[2][:, np.newaxis] - S[2]

    Kxx = (3*X*X)/R5 - 1/R3
    Kxy = (3*X*Y)/R5
    Kxz = (3*X*Z)/R5
    Kyy = (3*Y*Y)/R5 - 1/R3
    Kyz = (3*Y*Z)/R5

    return Kxx, Kxy, Kxz, Kyy, Kyz


def matrices_A(F, Kxx, Kxy, Kxz, Kyy, Kyz):
    '''
    Compute the matrices Ax, Ay and Az containing, respectively, the x, y and z
    components of the vector defined by the product of the main-field unit
    vector and the gradient tensor of the inverse distance function.

    parameters
    ----------
    F: numpy array 1d
        Unit vector defining the main-field direction.
    Kxx, Kxy, Kxz, Kyy, Kyz: numpyy arrays 2d
        N x M matrices containing the second derivatives xx, xy, xz, yy and yz
        (zz is not computed because it can be obtained from xx and yy).

    returns
    -------
    Ax, Ay, Az: numpy array 2d
        Matrices containing the x, y and z components of the unit vector F and
        2nd derivatives matrix of the inverse distance function.
    '''

    assert (F.ndim == 1) and (F.size == 3), 'F must be a vector with 3 elements'
    assert Kxx.ndim == 2, 'Kxx must be a matrix'
    assert Kxy.ndim == 2, 'Kxy must be a matrix'
    assert Kxz.ndim == 2, 'Kxz must be a matrix'
    assert Kyy.ndim == 2, 'Kyy must be a matrix'
    assert Kzz.ndim == 2, 'Kzz must be a matrix'
    shape_xx = Kxx.shape
    assert Kxy.shape == shape_xx, 'Kxx and Kxy must have the same shape'
    assert Kxz.shape == shape_xx, 'Kxx and Kxz must have the same shape'
    assert Kyy.shape == shape_xx, 'Kxx and Kyy must have the same shape'
    assert Kyz.shape == shape_xx, 'Kxx and Kyz must have the same shape'

    Ax = F[0]*Kxx + F[1]*Kxy + F[2]*Kxz
    Ay = F[0]*Kxy + F[1]*Kyy + F[2]*Kyz
    Az = F[0]*Kxz + F[1]*Kyz - F[2]*(Kxx + Kyy)

    return Ax, Ay, Az


def dipole_matrix(h, Ax, Ay, Az):
    '''
    Compute the matrix whose element ij is the approximated
    total-field anomaly produced by a dipole with total
    magnetization vector h.

    parameters
    ----------
    h: numpy array 1d
        Unit vector defining the total magnetization of the dipoles.

    Ax, Ay, Az: numpy arrays 2d
        Matrices depending on the direction of the main field and the second
        derivatives of the inverse distance function.

    returns
    -------
    G: numpy array 2d
        Dipole matrix.
    '''

    assert (h.ndim == 1) and (h.size == 3), 'h must be a vector with 3 elements'
    assert Ax.ndim == 2, 'Ax must be a matrix'
    assert Ay.ndim == 2, 'Ay must be a matrix'
    assert Az.ndim == 2, 'Az must be a matrix'
    shape_x = Ax.shape
    assert Ay.shape == shape_x, 'Ay and Ax must have the same shape'
    assert Az.shape == shape_x, 'Az and Ax must have the same shape'

    G = cts.CMT2NT*(h[0]*Ax + h[1]*Ay + h[2]*Az)

    return G
