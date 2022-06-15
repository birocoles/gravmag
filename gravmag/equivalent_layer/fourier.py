'''
This file contains Python codes for dealing with 2D discrete convolutions.
'''

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import constants as cts


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


def embedding_BCCB_first_column(b0, Q, P, symmetry):
    '''
    Compute the first column "c0" of the embedding BCCB matrix
    from the first column "b0" of the BTTB matrix.

    parameters
    ----------
    b0: numpy array 1d
        First column of the BTTB matrix.
    Q: int
        Number of blocks along a column/row of the BTTB.
    P: int
        Order of each block forming the BTTB.
    symmetry: string
        Define the symmetry type. There are four types: "skew-skew",
        "skew-symm", "symm-skew" and "symm-symm". The first term defines the
        symmetry of the blocks set. The second term defines the symmetry in
        each block. Thse symmetries are defined by "skew" (skew-symmetric) and
        "symm" (symmetric).

    returns
    -------
    c0: numpy array 1d
        First column of the embedding BCCB matrix.
    '''
    assert b0.ndim == 1, 'b0 must be an array 1d'
    assert (isinstance(Q, int)) and (Q > 0), 'Q must be a positive integer'
    assert (isinstance(P, int)) and (P > 0), 'P must be a positive integer'
    assert b0.size == Q*P, 'b0 must have Q*P elements'
    assert symmetry in ['skew-skew', 'skew-symm', 'symm-skew', 'symm-symm'], 'invalid symmetry'

    # split b into Q parts
    b_parts = np.split(b0, Q)

    c0 = []

    # define the vector c0 for symmetry 'skew-skew'
    if symmetry == 'skew-skew':
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, -bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2*P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([-bi, 0, bi[:0:-1]]))

    # define the vector c for symmetry 'skew-symm'
    if symmetry == 'skew-symm':
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2*P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([-bi, 0, -bi[:0:-1]]))

    # define the vector c for symmetry 'symm-skew'
    if symmetry == 'symm-skew':
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, -bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2*P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([bi, 0, -bi[:0:-1]]))

    # define the vector c for symmetry 'symm-symm'
    if symmetry == 'symm-symm':
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2*P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([bi, 0, bi[:0:-1]]))

    # concatenate c0 in a single vector
    c0 = np.concatenate(c0)

    return c0


def eigenvalues_BCCB(c0, Q, P, ordering='row'):
    '''
    Compute the eigenvalues of a Block Circulant formed
    by Circulant Blocks (BCCB) matrix C. The eigenvalues
    are rearranged along the rows or columns of a matrix L.

    parameters
    ----------
    c0: numpy array 1D
        First column of C.
    Q: int
        Number of blocks along a column/row of the BTTB.
    P: int
        Order of each block forming the BTTB.
    ordering: string
        If "row", the eigenvalues are arranged along the rows of a matrix L;
        if "column", they are arranged along the columns of a matrix L.

    returns
    -------
    L: numpy array 2D
        Matrix formed by the eigenvalues of the BCCB.
    '''
    assert (isinstance(Q, int)) and (Q > 0), 'Q must be a positive integer'
    assert (isinstance(P, int)) and (P > 0), 'P must be a positive integer'
    assert c0.ndim == 1, 'c0 must be an array 1d'
    assert c0.size == 4*Q*P, 'c0 must have 4*Q*P elements'
    assert ordering in ['row', 'column'], 'invalid ordering'

    if ordering == 'row':
        # matrix containing the elements of c0 arranged along its rows
        G = np.reshape(c0, (2*Q, 2*P))
    else: # if ordering == 'column':
        # matrix containing the elements of vector a arranged along its columns
        G = np.reshape(c0, (2*Q, 2*P)).T

    L = np.sqrt(4*Q*P)*fft2(x=G, norm='ortho')

    return L


def product_BCCB_vector(L, Q, P, v, ordering='row'):
    '''
    Compute the product of a BCCB matrix and a vector
    v by using the eigenvalues of the BCCB. This BCCB embeds
    a BTTB matrix formed by Q x Q blocks, each one with
    P x P elements.

    parameters
    ----------
    L: numpy array 2D
        Matrix formed by the eigenvalues of the embedding BCCB. This matrix
        must have the ordering defined by the parameter 'ordering' (see its
        description below).
    Q: int
        Number of blocks along a column/row of the BTTB.
    P: int
        Order of each block forming the BTTB.
    v: numpy array 1d
        Vector to be multiplied by the BCCB matrix.
    ordering: string
        If "row", the eigenvalues are arranged along the rows of a matrix L;
        if "column", they are arranged along the columns of a matrix L.

    returns
    -------
    W: numpy array 2d
        Matrix containing the non-null elements of the product of the BCCB
        matrix and vector v. Its elements are arranged according to the
        ordering defined by the parameter 'ordering' (described above).
    '''
    assert (isinstance(Q, int)) and (Q > 0), 'Q must be a positive integer'
    assert (isinstance(P, int)) and (P > 0), 'P must be a positive integer'
    assert v.ndim == 1, 'v must be an array 1d'
    assert v.size == Q*P, 'v must have Q*P elements'
    assert L.ndim == 2, 'L must be an array 2d'
    assert L.size == 4*Q*P, 'c0 must have 4*Q*P elements'
    assert ordering in ['row', 'column'], 'invalid ordering'

    if ordering == 'row':
        assert L.shape == (2*Q, 2*P), 'L must have shape (2*Q, 2*P)'
        # matrix containing the elements of vector a arranged along its rows
        V = np.reshape(v, (Q, P))
        V = np.hstack([V, np.zeros((Q,P))])
        V = np.vstack([V, np.zeros((Q,2*P))])
    else: # if ordering == 'column':
        assert L.shape == (2*P, 2*Q), 'L must have shape (2*P, 2*Q)'
        # matrix containing the elements of vector a arranged along its columns
        V = np.reshape(v, (Q, P)).T
        V = np.hstack([V, np.zeros((P,Q))])
        V = np.vstack([V, np.zeros((P,2*Q))])

    # matrix obtained by computing the Hadamard product
    H = L*fft2(x=V, norm='ortho')

    # matrix containing the non-null elements of the product BCCB v
    # arranged according to the parameter 'ordering'
    # the non-null elements are located in the first quadrant.
    if ordering == 'row':
        W = ifft2(x=H, norm='ortho')[:Q,:P].real
    else: # if ordering == 'column':
        W = ifft2(x=H, norm='ortho')[:P,:Q].real

    return W


def eigenvalues_matrix(h_hat, u_hat, eigenvalues_K,
                       N_blocks, N_points_per_block):
    '''
    Compute the eigenvalues matrix L of the "u_hat" magnetic field component
    produced by a dipole layer with magnetization direction "h_hat".
    '''
    f0 = h_hat[0]*u_hat[0] - h_hat[2]*u_hat[2]
    f1 = h_hat[0]*u_hat[1] + h_hat[1]*u_hat[0]
    f2 = h_hat[0]*u_hat[2] + h_hat[2]*u_hat[0]
    f3 = h_hat[1]*u_hat[1] - h_hat[2]*u_hat[2]
    f4 = h_hat[1]*u_hat[2] + h_hat[2]*u_hat[1]
    factors = [f0, f1, f2, f3, f4]

    L = np.zeros((2*N_blocks, 2*N_points_per_block), dtype='complex')

    for factor, eigenvalues_Ki in zip(factors, eigenvalues_K):

        # compute the matrix of eigenvalues of the embedding BCCB
        L += factor*eigenvalues_Ki

    L *= cts.CMT2NT

    return L


def H_matrix(y, n):
    '''
    Matrix of the Fourier series model for producing
    the annihilator model.

    parameters
    ----------
    y: numpy array 2D
        Rotated coordinate y computed with function
        "utils.coordinate_transform".
    n: int
        Positive integer defining the maximum degree of the Fourier series
        model.

    returns
    -------
    H: numpy array 2D
        Matrix of the Fourier series model.
    '''
    assert (isinstance(n, int)) and (n > 0), 'n must be a positive integer'
    shapey = y.shape
    L = np.max(y) - np.min((y))
    arg = 2*np.pi*np.outer(y.ravel(), np.arange(n+1))/L
    H = np.hstack([np.cos(arg), np.sin(arg)])
    return H


# def cgnr_method(A, dobs, p0, tol):
#     '''
#     Solve a linear system by using the conjugate gradient
#     normal equation residual method (Golub and Van Loan, 2013,
#     modified Algorithm 11.3.3 according to Figure 11.3.1 ,
#     p. 637).

#     Parameters:
#     -----------
#     A : array 2D
#         Rectangular N x M matrix.
#     dobs : array 1D
#         Observed data vector with N elements.
#     p0 : array 1D
#         Initial approximation of the M x 1 solution p.
#     tol : float
#         Positive scalar controlling the termination criterion.

#     Returns:
#     --------
#     p : array 1D
#         Solution of the linear system.
#     dpred : array 1D
#         Predicted data vector produced by p.
#     residuals_L2_norm_values : list
#         L2 norm of the residuals along the iterations.
#     '''

#     A = np.asarray(A)
#     dobs = np.asarray(dobs)
#     p0 = np.asarray(p0)
#     assert dobs.size == A.shape[0], 'A order and dobs size must be the same'
#     assert p0.size == A.shape[1], 'A order and p0 size must be the same'
#     assert np.isscalar(tol) & (tol > 0.), 'tol must be a positive scalar'

#     N = dobs.size
#     p = p0.copy()
#     # residuals vector
#     res = dobs - np.dot(A, p)

#     # auxiliary variable
#     z = np.dot(A.T, res)

#     # L2 norm of z
#     z_L2 = np.dot(z,z)
#     # Euclidean norm of the residuals
#     res_norm = np.linalg.norm(res)
#     # List of Euclidean norm of the residuals
#     residuals_norm_values = [res_norm]
#     # positive scalar controlling convergence
#     delta = tol*np.linalg.norm(dobs)

#     # iteration 1
#     if res_norm > delta:
#         q = z
#         w = np.dot(A, q)
#         mu = z_L2/np.dot(w,w)
#         p += mu*q
#         res -= mu*w
#         z = np.dot(A.T, res)
#         z_L2_ = z_L2
#         z_L2 = np.dot(z,z)
#         res_norm = np.linalg.norm(res)

#     residuals_norm_values.append(res_norm)

#     # remaining iterations
#     while res_norm > delta:
#         tau = z_L2/z_L2_
#         q = z + tau*q
#         w = np.dot(A, q)
#         mu = z_L2/np.dot(w,w)
#         p += mu*q
#         res -= mu*w
#         z = np.dot(A.T, res)
#         z_L2_ = z_L2
#         z_L2 = np.dot(z,z)
#         res_norm = np.linalg.norm(res)
#         residuals_norm_values.append(res_norm)

#     dpred = np.dot(A,p)

#     return p, dpred, residuals_norm_values
