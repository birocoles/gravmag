"""
This file contains Python codes for dealing with 2D discrete convolutions.
"""

import numpy as np
from scipy.linalg import toeplitz, circulant
from scipy.fft import fft2, ifft2


def compute(FT_data, filters, check_input=True):
    """
    Compute the convolution in Fourier domain as the Hadamard (or element-wise)
    product of the Fourier-Transformed data and a sequence of filters.

    parameters
    ----------
    FT_data : numpy array 2D
        Matrix obtained by computing the 2D Discrete Fourier Transform of a
        regular grid of potential-field data located on a horizontal plane.
    filter : list of numpy arrays 2D
        List of matrices having the same shape of FT_data. These matrices
        represent the sequence of filters to be applied in Fourier domain.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    convolved_data : numpy array 2D or flattened array 1D
        Convolved data as a grid or a flattened array 1D, depending on
        the parameter "grid".
    """

    # convert FT_data to numpy array
    FT_data = np.asarray(FT_data)

    assert isinstance(filters, list), "filters must be a list"

    if check_input is True:
        assert np.iscomplexobj(FT_data), "FT_data must be a complex array"
        assert FT_data.ndim == 2, "FT_data must be a matrix"
        shape_data = FT_data.shape
        assert len(filters) > 0, "filters must have at least one element"
        for filter in filters:
            # convert filters elements into numpy arrays
            filter = np.asarray(filter)
            # verify filter ndim
            assert filter.ndim == 2, "filter must be a matrix"
            # verify filter shape
            assert (
                filter.shape == shape_data
            ), "filter must have the same shape as data"

    # create a single filter by multiplying all those
    # defined in filters
    resultant_filter = np.prod(filters, axis=0)

    # compute the convolved data in Fourier domain
    convolved_data = FT_data * resultant_filter

    return convolved_data


def general_BTTB(num_blocks, columns_blocks, rows_blocks=None):
    """
    Generate a Block Toeplitz formed by Toeplitz Blocks (BTTB)
    matrix T from the first columns and first rows of its non-repeating
    blocks.

    The matrix T has Q x Q blocks, each one with P x P elements.

    The first column and row of blocks forming the BTTB matrix T are
    represented as follows:

        |T11 T12 ... T1Q|
        |T21            |
    T = |.              | .
        |:              |
        |TQ1            |


    parameters
    ----------
    num_blocks: int
        Number of blocks (Q) of T along column and row.
    columns_blocks: numpy array 2D
        Matrix whose rows are the first columns cij of the non-repeating blocks
        Tij of T. They must be ordered as follows: c11, c12, ..., c1Q,
        c21, ..., cQ1.
    rows_blocks: None or numpy array 2D
        If not None, it is a matrix whose rows are the first rows rij of the
        non-repeating blocks Tij of T, without the diagonal term. They must
        be ordered as follows: r11, r12, ..., r1M, r21, ..., rQ1.

    If "rows_blocks" is None, there are two possible situations:

    1) the number of rows forming "columns_blocks" is equal to
        "number of blocks". In this case, the T matrix is a Symmetric Block
        Toeplitz formed by Symmetric Toeplitz Blocks (SBTSTB).
    2) the number of rows forming "columns_blocks" is equal to
        (2 * "number of blocks") - 1. In this case, the T matrix is a Block
        Toeplitz formed by Symmetric Toeplitz Blocks (BTSTB).

    If "rows_blocks" is not None, its number of rows must be equal to that
    of "columns_blocks". In this case, there are two possible situations:

    1) the number of rows forming "rows_blocks"/"columns_blocks" is equal
        to "number of blocks". In this case, the T matrix is a Symmetric Block
        Toeplitz formed by Toeplitz Blocks (SBTTB).
    2) the number of rows forming "rows_blocks"/"columns_blocks" is equal
        to (2 * "number of blocks") - 1. In this case, the T matrix is a Block
        Toeplitz formed by Toeplitz Blocks (BTTB).

    returns
    -------
    T: numpy array 2D
        BTTB matrix.
    """

    assert (isinstance(num_blocks, int)) and (
        num_blocks > 1
    ), "num_blocks must be a positive integer greater than 1"

    columns_blocks = np.asarray(columns_blocks)
    assert columns_blocks.ndim == 2, "columns_blocks must be a matrix"
    assert (columns_blocks.shape[0] == num_blocks) or (
        columns_blocks.shape[0] == (2 * num_blocks - 1)
    ), 'the number of rows in "columns_blocks" must be equal to "num_blocks" or equal to (2 * "num_blocks") - 1'

    block_size = columns_blocks.shape[1]
    ind_col_blocks, ind_row_blocks = np.ogrid[
        0:block_size, block_size - 1 : -1 : -1
    ]
    ind_blocks = ind_col_blocks + ind_row_blocks

    if rows_blocks is None:
        # Case SBTSTB
        if columns_blocks.shape[0] == num_blocks:
            blocks = []
            for column in columns_blocks:
                blocks.append(
                    np.concatenate((column[-1:0:-1], column))[ind_blocks]
                )
            concatenated_blocks = np.concatenate(
                (np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )

        # Case BTSTB generalized
        if columns_blocks.shape[0] == (2 * num_blocks - 1):
            blocks_1j = []
            for column in columns_blocks[:num_blocks]:
                blocks_1j.append(
                    np.concatenate((column[-1:0:-1], column))[ind_blocks]
                )
            blocks_i1 = []
            for column in columns_blocks[num_blocks:]:
                blocks_i1.append(
                    np.concatenate((column[-1:0:-1], column))[ind_blocks]
                )
            concatenated_blocks = np.concatenate(
                (np.stack(blocks_i1)[::-1], np.stack(blocks_1j))
            )

    else:
        rows_blocks = np.asarray(rows_blocks)
        assert (
            columns_blocks.shape[0] == rows_blocks.shape[0]
        ), 'the number of rows in "rows_and" and columns_blocks" must be equal to each other'
        assert columns_blocks.shape[1] == (
            rows_blocks.shape[1] + 1
        ), 'the number of column in "columns_blocks must be equal that in "rows_blocks" + 1'

        # Case SBTTB
        if columns_blocks.shape[0] == num_blocks:
            blocks = []
            for (column, row) in zip(columns_blocks, rows_blocks):
                blocks.append(np.concatenate((row[::-1], column))[ind_blocks])
            concatenated_blocks = np.concatenate(
                (np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )

        # Case BTTB generalized
        if columns_blocks.shape[0] == (2 * num_blocks - 1):
            blocks_1j = []
            for (column, row) in zip(
                columns_blocks[:num_blocks], rows_blocks[:num_blocks]
            ):
                blocks_1j.append(
                    np.concatenate((row[::-1], column))[ind_blocks]
                )
            blocks_i1 = []
            for (column, row) in zip(
                columns_blocks[num_blocks:], rows_blocks[num_blocks:]
            ):
                blocks_i1.append(
                    np.concatenate((row[::-1], column))[ind_blocks]
                )
            concatenated_blocks = np.concatenate(
                (np.stack(blocks_i1)[::-1], np.stack(blocks_1j))
            )

    ind_col, ind_row = np.ogrid[0:num_blocks, num_blocks - 1 : -1 : -1]
    indices = ind_col + ind_row

    T = np.hstack(np.hstack(concatenated_blocks[indices]))

    return T


def C_from_T(T_column, T_row=None):
    """
    Generate the full circulant Circulant matrix C which embbeds a
    Toeplitz matrix T.

    The Toeplitz matrix T has P x P elements. The embedding circulant matrix C
    has 2P x 2P elements.

    parameters
    ----------
    T_column: numpy array 1D
        First column of T.
    T_row: numpy array 1D or None
        If not None, it is the first row of T, without the diagonal element. In
        this case, T is non-symmetric and C is formed by using the T_column and
        T_row. If None, matrix T is symmetric and matrix C is generated by
        using only T_column. Default is None.

    returns
    -------
    C: numpy array 2D
        Full 2P x 2P circulant matrix.
    """

    T_column = np.asarray(T_column)
    assert T_column.ndim == 1, "T_column must be an array 1D"

    if T_row is None:
        # The first column of C (C_column) is formed by stacking the first
        # column of T (T_column), a zero and the vector "T_column[-1:0:-1]",
        # which is the reversed T_column without its first element. The first
        # element of T_column is the first element of the main diagonal of
        # matrix T.
        C_column = np.hstack([T_column, 0, T_column[-1:0:-1]])
        ind_col, ind_row = np.ogrid[0 : len(C_column), 0 : -len(C_column) : -1]
        indices = ind_col + ind_row
        C = C_column[indices]
    else:
        # The first column of C (C_column) is formed by stacking the first
        # column of T (T_column), a zero and the vector "T_row[::-1]",
        # which is the reversed first row of matrix T without the first element
        # that lies on the main diagonal of matrix T.
        T_row = np.asarray(T_row)
        assert T_row.ndim == 1, "T_row must be an array 1D"
        assert T_column.size == (
            T_row.size + 1
        ), "T_column size must be equal to T_row size + 1"
        C_column = np.hstack([T_column, 0, T_row[::-1]])
        ind_col, ind_row = np.ogrid[0 : len(C_column), 0 : -len(C_column) : -1]
        indices = ind_col + ind_row
        C = C_column[indices]

    return C


def BCCB_from_BTTB(num_blocks, columns_blocks, rows_blocks=None):
    """
    Generate a circulant Block Circulant formed by Circulant Matrices (BCCB)
    which embeds a Block Toeplitz formed by Toeplitz Blocks (BTTB) matrix T.

    The matrix BTTB has Q x Q blocks, each one with P x P elements. The
    embedding circulant matrix BCCB has 2Q x 2Q blocks, each one with 2P x 2P
    elements.

    The BCCB inherits the symmetry of BTTB matrix. It means that:

    1) An arbitrary BTTB matrix produces an arbitrary BCCB matrix;
    2) A Symmetric Block Toeplitz formed by Symmetric Toeplitz Blocks (SBTSTB)
    produces a Symmetric Block Circulant formed by Symmetric Circulant Blocks
    (SBCSCB);
    3) A Block Toeplitz formed by Symmetric Toeplitz Blocks (BTSTB) produces a
    Block Circulant formed by Symmetric Circulant Blocks (BCSCB);
    4) A Symmetric Block Toeplitz formed by Toeplitz Blocks (SBTTB) produces a
    Symmetric Block Circulant formed by Circulant Blocks (SBCCB).
    For details, see function 'general_BTTB'.

    parameters
    ----------
    See function 'general_BTTB' for a description of the input parameters.

    returns
    -------
    C: numpy array 2D
        BCCB matrix.
    """

    assert (isinstance(num_blocks, int)) and (
        num_blocks > 1
    ), "num_blocks must be a positive integer greater than 1"

    columns_blocks = np.asarray(columns_blocks)
    assert columns_blocks.ndim == 2, "columns_blocks must be a matrix"
    assert (columns_blocks.shape[0] == num_blocks) or (
        columns_blocks.shape[0] == (2 * num_blocks - 1)
    ), 'the number of rows in "columns_blocks" must be equal to "num_blocks" or equal to (2 * "num_blocks") - 1'

    if rows_blocks is None:
        T_block_size = columns_blocks.shape[1]
        nonull_blocks = []
        for column in columns_blocks:
            nonull_blocks.append(C_from_T(column))
        C_block_size = 2 * T_block_size

        # Case SBTSTB
        # The first block column of the BCCB matrix C is formed by stacking the
        # first block column of the SBTSTB matrix T (nonull_blocks), a block
        # with null elements and the reversed block column "nonull_blocks"
        # without its first block. The first block of "nonull_blocks" lies on
        # the main block diagonal of matrix T.
        if columns_blocks.shape[0] == num_blocks:
            C_blocks = np.concatenate(
                (
                    np.stack(nonull_blocks),
                    np.zeros((1, C_block_size, C_block_size)),
                    np.stack(nonull_blocks[-1:0:-1]),
                )
            )

        # Case BTSTB
        # The first block column of the BCCB matrix C is formed by stacking the
        # first block column of the BTSTB matrix T (nonull_blocks[:num_blocks]),
        # a block with null elements and the reversed block row
        # "nonull_blocks[-1:num_blocks-1:-1]" without the block that lies on
        # main block diagonal of matrix T.
        if columns_blocks.shape[0] == (2 * num_blocks - 1):
            C_blocks = np.concatenate(
                (
                    np.stack(nonull_blocks[:num_blocks]),
                    np.zeros((1, C_block_size, C_block_size)),
                    np.stack(nonull_blocks[-1 : num_blocks - 1 : -1]),
                )
            )

    else:
        rows_blocks = np.asarray(rows_blocks)
        assert (
            columns_blocks.shape[0] == rows_blocks.shape[0]
        ), 'the number of rows in "rows_and" and columns_blocks" must be equal to each other'
        assert columns_blocks.shape[1] == (
            rows_blocks.shape[1] + 1
        ), 'the number of column in "columns_blocks must be equal that in "rows_blocks" + 1'
        T_block_size = columns_blocks.shape[1]
        nonull_blocks = []
        for (column, rows) in zip(columns_blocks, rows_blocks):
            nonull_blocks.append(C_from_T(column, rows))
        C_block_size = 2 * T_block_size

        # Case SBTTB
        # The first block column of the BCCB matrix C is formed by stacking the
        # first block column of the SBTTB matrix T (nonull_blocks), a block
        # with null elements and the reversed block column "nonull_blocks"
        # without its first block. The first block of "nonull_blocks" lies on
        # the main block diagonal of matrix T.
        if columns_blocks.shape[0] == num_blocks:
            C_blocks = np.concatenate(
                (
                    np.stack(nonull_blocks),
                    np.zeros((1, C_block_size, C_block_size)),
                    np.stack(nonull_blocks[-1:0:-1]),
                )
            )

        # Case BTTB generalized
        # The first block column of the BCCB matrix C is formed by stacking the
        # first block column of the BTTB matrix T (nonull_blocks[:num_blocks]),
        # a block with null elements and the reversed block row
        # "nonull_blocks[-1:num_blocks-1:-1]" without the block that lies on
        # main block diagonal of matrix T.
        if columns_blocks.shape[0] == (2 * num_blocks - 1):
            C_blocks = np.concatenate(
                (
                    np.stack(nonull_blocks[:num_blocks]),
                    np.zeros((1, C_block_size, C_block_size)),
                    np.stack(nonull_blocks[-1 : num_blocks - 1 : -1]),
                )
            )

    num_blocks_C = 2 * num_blocks

    ind_col, ind_row = np.ogrid[0:num_blocks_C, 0:-num_blocks_C:-1]
    indices = ind_col + ind_row
    C = np.hstack(np.hstack(C_blocks[indices]))

    return C


def embedding_BCCB_first_column(b0, Q, P, symmetry):
    """
    Compute the first column "c0" of the embedding BCCB matrix
    from the first column "b0" of a BTTB matrix.

    parameters
    ----------
    b0: numpy array 1d
        First column of the BTTB matrix.
    Q: int
        Number of blocks along a column/row of the BTTB.
    P: int
        Number of rows/columns of each block forming the BTTB.
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
    """

    b0 = np.asarray(b0)
    # check b0 is a vector
    if b0.ndim != 1:
        raise ValueError("b0 must be an array 1d")
    # check if Q and P are positive integers
    assert (isinstance(Q, int)) and (Q > 0), "Q must be a positive integer"
    assert (isinstance(P, int)) and (P > 0), "P must be a positive integer"
    # check if b0 match Q and P
    if b0.size != Q * P:
        raise ValueError("b0 must have Q*P elements")
    # check if symmetry is valid
    if symmetry not in ["skew-skew", "skew-symm", "symm-skew", "symm-symm"]:
        raise ValueError("invalid {} symmetry".format(symmetry))

    # split b into Q parts
    b_parts = np.split(b0, Q)

    # define a list to store the pieces
    # of c0 that will be computed below
    c0 = []

    # define the elements of c0 for symmetry 'skew-skew'
    if symmetry == "skew-skew":
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, -bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2 * P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([-bi, 0, bi[:0:-1]]))

    # define the elements of c0 for symmetry 'skew-symm'
    if symmetry == "skew-symm":
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2 * P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([-bi, 0, -bi[:0:-1]]))

    # define the elements of c0 for symmetry 'symm-skew'
    if symmetry == "symm-skew":
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, -bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2 * P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([bi, 0, -bi[:0:-1]]))

    # define the elements of c0 for symmetry 'symm-symm'
    if symmetry == "symm-symm":
        # run the first block column of the BTTB
        for bi in b_parts:
            c0.append(np.hstack([bi, 0, bi[:0:-1]]))
        # include zeros
        c0.append(np.zeros(2 * P))
        # run the first block row of the BTTB
        for bi in b_parts[:0:-1]:
            c0.append(np.hstack([bi, 0, bi[:0:-1]]))

    # concatenate c0 in a single vector
    c0 = np.concatenate(c0)

    return c0


def eigenvalues_BCCB(c0, Q, P, ordering="row"):
    """
    Compute the eigenvalues of a Block Circulant formed
    by Circulant Blocks (BCCB) matrix C. The eigenvalues
    are rearranged along the rows or columns of a matrix L.

    parameters
    ----------
    c0: numpy array 1D
        First column of C.
    Q: int
        Number of blocks along a column/row of the corresponding BTTB matrix.
        See funtion 'embedding_BCCB_first_column'.
    P: int
        Number of rows/columns of each block forming the corresponding BTTB
        matrix. See funtion 'embedding_BCCB_first_column'.
    ordering: string
        If "row", the eigenvalues are arranged along the rows of a matrix L;
        if "column", they are arranged along the columns of a matrix L.

    returns
    -------
    L: numpy array 2D
        Matrix formed by the eigenvalues of the BCCB.
    """

    c0 = np.asarray(c0)
    # verify if c0 is a vector
    if c0.ndim != 1:
        raise ValueError("c0 must be an array 1d")
    # check if Q and P are positive integers
    assert (isinstance(Q, int)) and (Q > 0), "Q must be a positive integer"
    assert (isinstance(P, int)) and (P > 0), "P must be a positive integer"
    # check size of c0
    if c0.size != 4 * Q * P:
        raise ValueError("c0 must have 4*Q*P elements")
    # check if ordering is valid
    if ordering not in ["row", "column"]:
        raise ValueError("invalid {} ordering".format(ordering))

    # reshape c0 according to ordering
    if ordering == "row":
        # matrix containing the elements of c0 arranged along its rows
        G = np.reshape(c0, (2 * Q, 2 * P))
    else:  # if ordering == 'column':
        # matrix containing the elements of vector a arranged along its columns
        G = np.reshape(c0, (2 * Q, 2 * P)).T

    # compute the matrix L containing the eigenvalues
    L = np.sqrt(4 * Q * P) * fft2(x=G, norm="ortho")

    return L


def product_BCCB_vector(L, Q, P, v, ordering="row"):
    """
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
    w: numpy array 1d
        Vector containing the non-null elements of the product of the BCCB
        matrix and vector v.
    """

    L = np.asarray(L)
    assert L.ndim == 2, "L must be an array 2d"
    assert (isinstance(Q, int)) and (Q > 0), "Q must be a positive integer"
    assert (isinstance(P, int)) and (P > 0), "P must be a positive integer"
    v = np.asarray(v)
    assert v.ndim == 1, "v must be an array 1d"
    assert v.size == Q * P, "v must have Q*P elements"
    assert ordering in ["row", "column"], "invalid ordering"

    if ordering == "row":
        assert L.shape == (2 * Q, 2 * P), "L must have shape (2*Q, 2*P)"
        # matrix containing the elements of vector a arranged along its rows
        V = np.reshape(v, (Q, P))
        V = np.hstack([V, np.zeros((Q, P))])
        V = np.vstack([V, np.zeros((Q, 2 * P))])
    else:  # if ordering == 'column':
        assert L.shape == (2 * P, 2 * Q), "L must have shape (2*P, 2*Q)"
        # matrix containing the elements of vector a arranged along its columns
        V = np.reshape(v, (Q, P)).T
        V = np.hstack([V, np.zeros((P, Q))])
        V = np.vstack([V, np.zeros((P, 2 * Q))])

    # matrix obtained by computing the Hadamard product
    H = L * fft2(x=V, norm="ortho")

    # matrix containing the non-null elements of the product BCCB v
    # arranged according to the parameter 'ordering'
    # the non-null elements are located in the first quadrant.
    if ordering == "row":
        w = ifft2(x=H, norm="ortho")[:Q, :P].real
        w = w.ravel()
    else:  # if ordering == 'column':
        w = ifft2(x=H, norm="ortho")[:P, :Q].real
        w = w.T.ravel()

    return w


# def eigenvalues_matrix(h_hat, u_hat, eigenvalues_K,
#                        N_blocks, N_points_per_block):
#     '''
#     Compute the eigenvalues matrix L of the "u_hat" magnetic field component
#     produced by a dipole layer with magnetization direction "h_hat".
#     '''
#     f0 = h_hat[0]*u_hat[0] - h_hat[2]*u_hat[2]
#     f1 = h_hat[0]*u_hat[1] + h_hat[1]*u_hat[0]
#     f2 = h_hat[0]*u_hat[2] + h_hat[2]*u_hat[0]
#     f3 = h_hat[1]*u_hat[1] - h_hat[2]*u_hat[2]
#     f4 = h_hat[1]*u_hat[2] + h_hat[2]*u_hat[1]
#     factors = [f0, f1, f2, f3, f4]
#
#     L = np.zeros((2*N_blocks, 2*N_points_per_block), dtype='complex')
#
#     for factor, eigenvalues_Ki in zip(factors, eigenvalues_K):
#
#         # compute the matrix of eigenvalues of the embedding BCCB
#         L += factor*eigenvalues_Ki
#
#     L *= cts.CMT2NT
#
#     return L


# def H_matrix(y, n):
#     '''
#     Matrix of the Fourier series model for producing
#     the annihilator model.
#
#     parameters
#     ----------
#     y: numpy array 2D
#         Rotated coordinate y computed with function
#         "utils.coordinate_transform".
#     n: int
#         Positive integer defining the maximum degree of the Fourier series
#         model.
#
#     returns
#     -------
#     H: numpy array 2D
#         Matrix of the Fourier series model.
#     '''
#     assert (isinstance(n, int)) and (n > 0), 'n must be a positive integer'
#     shapey = y.shape
#     L = np.max(y) - np.min((y))
#     arg = 2*np.pi*np.outer(y.ravel(), np.arange(n+1))/L
#     H = np.hstack([np.cos(arg), np.sin(arg)])
#     return H


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
# assert (
#     np.isistance(tol, (float, int)) & (tol > 0.)
# ), 'tol must be a positive scalar'

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
