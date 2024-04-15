"""
This file contains Python codes for dealing with 2D discrete convolutions.
"""

import numpy as np
from scipy.linalg import toeplitz, circulant
from scipy.fft import fft2, ifft2
from . import check, data_structures


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

    if check_input is True:
        check.is_array(x=FT_data, ndim=2)
        shape_data = FT_data.shape
        if np.iscomplexobj(FT_data) == False:
            raise ValueError("FT_data must be a complex array")
        if type(filters) != list:
            raise ValueError("filters must be a list")
        if len(filters) == 0:
            raise ValueError("filters must have at least one element")
        for filter in filters:
            check.is_array(x=filter, ndim=2)
            shape_filter = filter.shape
            if shape_filter != shape_data:
                raise ValueError("filter must have the same shape as data")

    # create a single filter by multiplying all those
    # defined in filters
    resultant_filter = np.prod(filters, axis=0)

    # compute the convolved data in Fourier domain
    convolved_data = FT_data * resultant_filter

    return convolved_data


def Circulant_from_Toeplitz(Toeplitz, full=False, check_input=True):
    """
    Generate the Circulant matrix C which embbeds a Toeplitz matrix T.

    The Toeplitz matrix T has P x P elements. The embedding circulant matrix C has 2P x 2P elements.

    Matrix T is represented as follows:

        |t11 t12 ... t1P|
        |t21            |
    T = |.              | .
        |:              |
        |tP1            |


    We consider that matrix T may have three symmetry types:
    * gene - it denotes 'generic' and it means that there is no symmetry.
    * symm - it denotes 'symmetric' and it means that there is a perfect symmetry.
    * skew - it denotes 'skew-symmetric' and it means that the elements above the main diagonal
        have opposite signal with respect to those below the main diagonal.

    parameters
    ----------
    Toeplitz : dictionary containing the following keys
        symmetry : string
            Defines the type of symmetry between elements above and below the main diagonal.
            It can be 'gene', 'symm' or 'skew' (see the explanation above).
        column : numpy array 1D
            First column of T.
        row : None or numpy array 1D
            If not None, it is the first row of T, without the diagonal element. In
            this case, T does not have the assumed symmetries (see the text above).
            If None, matrix T is symmetric or skew-symmetric. Default is None.
    full : boolean
        If True, returns the full Circulant matrix C. Otherwise, returns only its first column. Default is False.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    C: numpy array 1D or 2D
        The full 2P x 2P circulant matrix or only its first column (see parameter 'full').
    """

    if check_input == True:
        check.Toeplitz_metadata(Toeplitz)
        if full not in [True, False]:
            raise ValueError("invalid parameter full ({})".format(full))

    # get the parameters defining the Toeplitz matrix
    symmetry = Toeplitz["symmetry"]
    column = Toeplitz["column"]
    row = Toeplitz["row"]

    # order of the Toeplitz matrix T
    P = column.size

    # define the first column of the BCCB matrix C by
    # concatenating (i) column in the correct order, (ii) a zero and (iii) row in the reversed order
    # 'row' is defined in terms of 'column' if symmetry is 'symm' or 'skew'
    if symmetry == "symm":
        C = np.hstack([column, 0, column[-1:0:-1]])
    elif symmetry == "skew":
        C = np.hstack([column, 0, -column[-1:0:-1]])
    else:  # symmetry == "gene"
        C = np.hstack([column, 0, row[::-1]])

    if full == True:
        # auxiliary variable used to create full Circulant matrices
        # P = 4
        # ind_col =
        # [[0]
        #  [1]
        #  [2]
        #  [3]
        #  [4]
        #  [5]
        #  [6]
        #  [7]]
        # ind_row =
        # [[ 0 -1 -2 -3 -4 -5 -6 -7]]
        # indices =
        # [[ 0 -1 -2 -3 -4 -5 -6 -7]
        #  [ 1  0 -1 -2 -3 -4 -5 -6]
        #  [ 2  1  0 -1 -2 -3 -4 -5]
        #  [ 3  2  1  0 -1 -2 -3 -4]
        #  [ 4  3  2  1  0 -1 -2 -3]
        #  [ 5  4  3  2  1  0 -1 -2]
        #  [ 6  5  4  3  2  1  0 -1]
        #  [ 7  6  5  4  3  2  1  0]]
        ind_col, ind_row = np.ogrid[0 : 2 * P, 0 : -2 * P : -1]
        indices = ind_col + ind_row
        return C[indices]
    else:  # full is False
        return C


def BTTB_from_metadata(BTTB_metadata, check_input=True):
    """
    Generate the data structure for a full Block Toeplitz formed by Toeplitz Blocks (BTTB)
    matrix T from the first columns and first rows of its non-repeating blocks.

    The matrix T has nblocks x nblocks blocks, each one with npoints_per_block x npoints_per_block elements.

    The first column and row of blocks forming the BTTB matrix T are represented as follows:

        |T11 T12 ... T1Q|
        |T21            |
    T = |.              | .
        |:              |
        |TQ1            |


    There are two symmetries:
    * symmetry_structure - between all blocks above and below the main block diagonal.
    * symmetry_blocks    - between all elements above and below the main diagonal within each block.
    Each symmetry pattern has three possible types:
    * gene - it denotes 'generic' and it means that there is no symmetry.
    * symm - it denotes 'symmetric' and it means that there is a perfect symmetry.
    * skew - it denotes 'skew-symmetric' and it means that the elements above the main diagonal
        have opposite signal with respect to those below the main diagonal.
    Hence, we consider that the BTTB matrix T has nine possible symmetry patterns:
    * 'symm-symm' - Symmetric Block Toeplitz formed by Symmetric Toeplitz Blocks
    * 'symm-skew' - Symmetric Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
    * 'symm-gene' - Symmetric Block Toeplitz formed by Generic Toeplitz Blocks
    * 'skew-symm' - Skew-Symmetric Block Toeplitz formed by Symmetric Toeplitz Blocks
    * 'skew-skew' - Skew-Symmetric Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
    * 'skew-gene' - Skew-Symmetric Block Toeplitz formed by Generic Toeplitz Blocks
    * 'gene-symm' - Generic Block Toeplitz formed by Symmetric Toeplitz Blocks
    * 'gene-skew' - Generic Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
    * 'gene-gene' - Generic Block Toeplitz formed by Generic Toeplitz Blocks

    parameters
    ----------
    BTTB_metadata : dictionary containing the following keys:
        symmetry_structure : string
            Defines the type of symmetry between all blocks above and below the main block diagonal.
            It can be 'gene', 'symm' or 'skew' (see the explanation above).
        symmetry_blocks : string
            Defines the type of symmetry between elements above and below the main diagonal within all blocks.
            It can be 'gene', 'symm' or 'skew' (see the explanation above).
        nblocks : int
            Number of blocks (nblocks) of T along column and row.
        columns : numpy array
            Matrix whose rows are the first columns cij of the non-repeating blocks
            Tij of T. They must be ordered as follows: c11, c21, ..., cQ1,
            c12, ..., c1Q.
        rows : None or numpy array 2D
            If not None, it is a matrix whose rows are the first rows rij of the
            non-repeating blocks Tij of T, without the diagonal term. They must
            be ordered as follows: r11, r21, ..., rQ1, r12, ..., r1Q.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    T: numpy array 2D
        The full BTTB matrix.
    """

    if check_input == True:
        check.BTTB_metadata(BTTB=BTTB_metadata)

    # get the parameters defining the BTTB matrix
    symmetry_structure = BTTB_metadata["symmetry_structure"]
    symmetry_blocks = BTTB_metadata["symmetry_blocks"]
    nblocks = BTTB_metadata["nblocks"]
    columns = BTTB_metadata["columns"]
    rows = BTTB_metadata["rows"]

    # number of points per block row/column
    npoints_per_block = columns.shape[1]
    # auxiliary variable used to create Toeplitz matrices
    # npoints_per_block = 4
    # ind_col_blocks =
    # [[0]
    #  [1]
    #  [2]
    #  [3]]
    # ind_row_blocks =
    # [[3 2 1 0]]
    # ind_blocks =
    # [[3 2 1 0]
    #  [4 3 2 1]
    #  [5 4 3 2]
    #  [6 5 4 3]]
    ind_col_blocks, ind_row_blocks = np.ogrid[
        0:npoints_per_block, npoints_per_block - 1 : -1 : -1
    ]
    ind_blocks = ind_col_blocks + ind_row_blocks

    if symmetry_structure == "symm":
        if symmetry_blocks == "symm":
            # 'symm-symm' - Symmetric Block Toeplitz formed by Symmetric Toeplitz Blocks
            blocks = []
            for column in columns:
                # create the block column in the correct order
                blocks.append(
                    # concatenate (i) row in reversed order and (ii) column in the correct order
                    # create a matrix by indexing the concatenated vector with the auxiliary variable ind_blocks
                    np.concatenate((column[-1:0:-1], column))[ind_blocks]
                )
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )
        elif symmetry_blocks == "skew":
            # 'symm-skew' - Symmetric Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
            blocks = []
            for column in columns:
                # create the block column in the correct order
                blocks.append(
                    # concatenate (i) row in reversed order and (ii) column in the correct order
                    # create a matrix by indexing the concatenated vector with the auxiliary variable ind_blocks
                    np.concatenate((-column[-1:0:-1], column))[ind_blocks]
                )
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )
        else:  # symmetry_blocks == 'gene'
            # 'symm-gene' - Symmetric Block Toeplitz formed by Generic Toeplitz Blocks
            blocks = []
            # create the block column in the correct order
            for column, row in zip(columns, rows):
                # concatenate (i) row in reversed order and (ii) column in the correct order
                # create a matrix by indexing the concatenated vector with the auxiliary variable ind_blocks
                blocks.append(np.concatenate((row[::-1], column))[ind_blocks])
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )
    elif symmetry_structure == "skew":
        if symmetry_blocks == "symm":
            # 'skew-symm' - Skew-Symmetric Block Toeplitz formed by Symmetric Toeplitz Blocks
            blocks = []
            for column in columns:
                # create the block column in the correct order
                blocks.append(
                    # concatenate (i) row in reversed order and (ii) column in the correct order
                    # create a matrix by indexing the concatenated vector with the auxiliary variable ind_blocks
                    np.concatenate((column[-1:0:-1], column))[ind_blocks]
                )
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (-np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )
        elif symmetry_blocks == "skew":
            # 'skew-skew' - Skew-Symmetric Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
            blocks = []
            for column in columns:
                # create the block column in the correct order
                blocks.append(
                    # concatenate (i) row in reversed order and (ii) column in the correct order
                    # create a matrix by indexing the concatenated vector with the auxiliary variable ind_blocks
                    np.concatenate((-column[-1:0:-1], column))[ind_blocks]
                )
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (-np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )
        else:  # symmetry_blocks == 'gene'
            # 'skew-gene' - Skew-Symmetric Block Toeplitz formed by Generic Toeplitz Blocks
            blocks = []
            # create the block column in the correct order
            for column, row in zip(columns, rows):
                # concatenate (i) row in reversed order and (ii) column in the correct order
                # create a matrix by indexing the concatenated vector with the auxiliary variable ind_blocks
                blocks.append(np.concatenate((row[::-1], column))[ind_blocks])
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (-np.stack(blocks)[-1:0:-1], np.stack(blocks))
            )
    else:  # symmetry_structure == 'gene'
        if symmetry_blocks == "symm":
            # 'gene-symm' - Generic Block Toeplitz formed by Symmetric Toeplitz Blocks
            blocks_1j = []
            # create the block column in the correct order
            for column in columns[:nblocks]:
                blocks_1j.append(
                    np.concatenate((column[-1:0:-1], column))[ind_blocks]
                )
            blocks_i1 = []
            # create the block row in the correct order
            for column in columns[nblocks:]:
                blocks_i1.append(
                    np.concatenate((column[-1:0:-1], column))[ind_blocks]
                )
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (np.stack(blocks_i1)[::-1], np.stack(blocks_1j))
            )
        elif symmetry_blocks == "skew":
            # 'gene-skew' - Generic Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
            blocks_1j = []
            # create the block column in the correct order
            for column in columns[:nblocks]:
                blocks_1j.append(
                    np.concatenate((-column[-1:0:-1], column))[ind_blocks]
                )
            blocks_i1 = []
            # create the block row in the correct order
            for column in columns[nblocks:]:
                blocks_i1.append(
                    np.concatenate((-column[-1:0:-1], column))[ind_blocks]
                )
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (np.stack(blocks_i1)[::-1], np.stack(blocks_1j))
            )
        else:  # symmetry_blocks == 'gene'
            # 'gene-gene' - Generic Block Toeplitz formed by Generic Toeplitz Blocks
            blocks_1j = []
            # create the block column in the correct order
            for column, row in zip(columns[:nblocks], rows[:nblocks]):
                blocks_1j.append(
                    np.concatenate((row[::-1], column))[ind_blocks]
                )
            blocks_i1 = []
            # create the block row in the correct order
            for column, row in zip(columns[nblocks:], rows[nblocks:]):
                blocks_i1.append(
                    np.concatenate((row[::-1], column))[ind_blocks]
                )
            # concatenate (i) block row in reversed order and (ii) block column in the correct order
            concatenated_blocks = np.concatenate(
                (np.stack(blocks_i1)[::-1], np.stack(blocks_1j))
            )

    # auxiliary variable similar to ind_blocks
    ind_col, ind_row = np.ogrid[0:nblocks, nblocks - 1 : -1 : -1]
    indices = ind_col + ind_row

    # create the full BTTB matrix T by indexing the concatenated blocks
    # with the auxiliary variable indices
    T = np.hstack(np.hstack(concatenated_blocks[indices]))

    return T



def embedding_BCCB(BTTB_metadata, full=False, check_input=True):
    """
    Generate the first column or the full Block Circulant formed by Circulant Blocks (BCCB)
    matrix that embeds a given Block Toeplitz formed by Toeplitz Blocks (BTTB).

    See details in the function 'data_structures.BTTB_metadata'.

    parameters
    ----------
    BTTB_metadata : dictionary
        See function 'check.BTTB_metadata' for a description of the input parameters.
    full : boolean
        If True, returns the full BCCB matrix C. Otherwise, returns only its first column. Default is False.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    C: numpy array 2D or 1D
        The full embedding BCCB matrix or only its first column (see parameter 'full').
    """

    if check_input == True:
        check.BTTB_metadata(BTTB_metadata)
        if full not in [True, False]:
            raise ValueError("invalid parameter full ({})".format(full))

    # get the parameters defining the BTTB matrix
    symmetry_structure = BTTB_metadata["symmetry_structure"]
    symmetry_blocks = BTTB_metadata["symmetry_blocks"]
    nblocks = BTTB_metadata["nblocks"]
    columns = BTTB_metadata["columns"]
    rows = BTTB_metadata["rows"]

    nblocks_C = 2 * nblocks
    npoints_per_block_C = 2 * columns.shape[1]

    # list to store the first (block) column
    c0 = []

    if symmetry_blocks in ["symm", "skew"]:
        # iterate over columns
        for t0 in columns:
            Toeplitz = {
                "symmetry": symmetry_blocks,
                "column": t0,
                "row": None,
            }
            c0.append(
                Circulant_from_Toeplitz(Toeplitz, full=False, check_input=False)
            )
    else:  # symmetry_blocks == "gene"
        # iterate over columns and rows
        for t0, r0 in zip(columns, rows):
            Toeplitz = {
                "symmetry": symmetry_blocks,
                "column": t0,
                "row": r0,
            }
            c0.append(
                Circulant_from_Toeplitz(Toeplitz, full=False, check_input=False)
            )

    if symmetry_structure == "symm":
        c0 = np.hstack(
            (
                np.hstack(c0),
                np.zeros_like(c0[0]),
                np.hstack(c0[-1:0:-1]),
            )
        )
    elif symmetry_structure == "skew":
        c0 = np.hstack(
            (
                np.hstack(c0),
                np.zeros_like(c0[0]),
                -np.hstack(c0[-1:0:-1]),
            )
        )
    else:  # symmetry_structure == "gene"
        c0 = np.hstack(
            (
                np.hstack(c0[:nblocks]),
                np.zeros_like(c0[0]),
                np.hstack(c0[-1 : nblocks - 1 : -1]),
            )
        )

    if full == True:
        C = []
        for c0i in np.split(c0, nblocks_C):
            C.append(circulant(c0i))
        C = np.stack(C)
        ind_col, ind_row = np.ogrid[0:nblocks_C, 0:-nblocks_C:-1]
        indices = ind_col + ind_row
        C = np.hstack(np.hstack(C[indices]))

    else:  # full == False
        # concatenate all pieces to form the first column of the BCCB matrix
        C = np.hstack(c0)

    return C


def eigenvalues_BCCB(BTTB_metadata, ordering="row", check_input=True):
    """
    Compute the eigenvalues of a Block Circulant formed by Circulant Blocks (BCCB) matrix C
    that embeds a given Block Toeplitz formed by Toeplitz Blocks (BTTB) matrix. The eigenvalues
    are rearranged along the rows or columns of a matrix L.

    parameters
    ----------
    BTTB_metadata : dictionary
        See function 'check.BTTB_metadata' for a description of the input parameters.
    ordering: string
        If "row", the eigenvalues will be arranged along the rows of a matrix L;
        if "column", they will be arranged along the columns of a matrix L.
        Default is 'row'.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    L : numpy array 2D
        Matrix formed by the eigenvalues of the BCCB.
    """

    if check_input == True:
        # check if the associated BTTB is valid
        check.BTTB_metadata(BTTB_metadata)
        # check if ordering is valid
        if ordering not in ["row", "column"]:
            raise ValueError("invalid {} ordering".format(ordering))

    # get the parameters defining the associated BTTB matrix
    symmetry_structure = BTTB_metadata["symmetry_structure"]
    symmetry_blocks = BTTB_metadata["symmetry_blocks"]
    nblocks_BTTB = BTTB_metadata["nblocks"]
    columns_BTTB = BTTB_metadata["columns"]
    rows_BTTB = BTTB_metadata["rows"]

    npoints_per_block_BTTB = columns_BTTB.shape[1]

    # compute the first column of the BCCB matrix
    c0 = embedding_BCCB(BTTB_metadata, full=False, check_input=False)

    # reshape c0 according to ordering
    if ordering == "row":
        # matrix containing the elements of c0 arranged along its rows
        G = np.reshape(c0, (2 * nblocks_BTTB, 2 * npoints_per_block_BTTB))
    else:  # if ordering == 'column':
        # matrix containing the elements of vector a arranged along its columns
        G = np.reshape(c0, (2 * nblocks_BTTB, 2 * npoints_per_block_BTTB)).T

    # compute the matrix L containing the eigenvalues
    L = np.sqrt(4 * nblocks_BTTB * npoints_per_block_BTTB) * fft2(
        x=G, norm="ortho"
    )

    return L


def product_BCCB_vector(eigenvalues, ordering, v, check_input=True):
    """
    Compute the product of a BCCB matrix and a vector v by using the eigenvalues of the BCCB.

    parameters
    ----------
    L : numpy array 2D
        Matrix formed by the eigenvalues of the BCCB.
    ordering: string
        If "row", the eigenvalues are arranged along the rows of matrix L;
        if "column", they are arranged along the columns of L.
    v: numpy array 1d
        Vector to be multiplied by the BCCB matrix.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    w: numpy array 1d
        Vector containing the non-null elements of the product of the BCCB
        matrix and vector v.
    """

    if check_input == True:
        check.is_array(x=eigenvalues, ndim=2)
        if ordering not in ["row", "column"]:
            raise ValueError("invalid ordering {}".format(ordering))
        check.is_array(x=v, ndim=1)
        if eigenvalues.size != 4 * v.size:
            raise ValueError(
                "'eigenvalues' size ({}) must be equal to 4 times v size ({})".format(
                    eigenvalues.size, 4 * v.size
                )
            )

    # rearrange vector v into a matrix and pad with zeros
    if ordering == "row":
        # define the number of blocks and points per block of the
        # BTTB matrix associated with the BCCB matrix
        nblocks_BTTB = eigenvalues.shape[0] // 2
        npoints_per_block_BTTB = eigenvalues.shape[1] // 2
        # matrix containing the elements of vector a arranged along its rows
        V = np.reshape(v, (nblocks_BTTB, npoints_per_block_BTTB))
        V = np.hstack([V, np.zeros((nblocks_BTTB, npoints_per_block_BTTB))])
        V = np.vstack([V, np.zeros((nblocks_BTTB, 2 * npoints_per_block_BTTB))])
    else:  # if ordering == 'column':
        # define the number of blocks and points per block of the
        # BTTB matrix associated with the BCCB matrix
        nblocks_BTTB = eigenvalues.shape[1] // 2
        npoints_per_block_BTTB = eigenvalues.shape[0] // 2
        # matrix containing the elements of vector a arranged along its columns
        V = np.reshape(v, (nblocks_BTTB, npoints_per_block_BTTB)).T
        V = np.hstack([V, np.zeros((npoints_per_block_BTTB, nblocks_BTTB))])
        V = np.vstack([V, np.zeros((npoints_per_block_BTTB, 2 * nblocks_BTTB))])

    # matrix obtained by computing the Hadamard product
    H = eigenvalues * fft2(x=V, norm="ortho")

    # matrix containing the non-null elements of the product BCCB v
    # arranged according to the parameter 'ordering'
    # the non-null elements are located in the first quadrant.
    if ordering == "row":
        w = ifft2(x=H, norm="ortho")[
            :nblocks_BTTB, :npoints_per_block_BTTB
        ].real
        w = w.ravel()
    else:  # if ordering == 'column':
        w = ifft2(x=H, norm="ortho")[
            :npoints_per_block_BTTB, :nblocks_BTTB
        ].real
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
