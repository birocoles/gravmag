import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from numpy.linalg import multi_dot
from scipy.linalg import toeplitz, circulant, dft
from pytest import raises
from .. import convolve as cv


def test_compute_FT_data_not_complex_matrix():
    "must raise AssertionError if FT_data not a complex matrix"
    filters = [np.ones((5, 5))]
    # FT_data as a complex vector
    FT_data = np.ones(5) - 1j * np.ones(5)
    with raises(AssertionError):
        cv.compute(FT_data, filters)
    # FT_data as a real matrix
    FT_data = np.ones((5, 5))
    with raises(AssertionError):
        cv.compute(FT_data, filters)


def test_compute_filters_not_complex_matrices():
    "must raise AssertionError if filters does not contain complex matrices"
    FT_data = np.ones((5, 5)) - 1j * np.ones((5, 5))
    # filters without any element
    filters = []
    with raises(AssertionError):
        cv.compute(FT_data, filters)
    # filters as a scalar
    filters = 3
    with raises(AssertionError):
        cv.compute(FT_data, filters)
    # filters with vectors
    filters = [np.ones(3)]
    with raises(AssertionError):
        cv.compute(FT_data, filters)
    # filters with matrices having a sahpe different from FT_data
    filters = [np.ones((3, 3))]
    with raises(AssertionError):
        cv.compute(FT_data, filters)


def test_general_BTTB_bad_num_blocks():
    "must raise AssertionError for bad num_blocks"
    columns_blocks = np.ones((3, 2))
    # num_blocks negative integer
    num_blocks = -5
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)
    # num_blocks float
    num_blocks = 4.2
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)
    # num_blocks equal to 1
    num_blocks = 1
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)


def test_general_BTTB_columns_blocks_not_matrix():
    "must raise AssertionError for columns_blocks not matrix"
    num_blocks = 5
    # columns_blocks a float
    columns_blocks = 2.9
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)
    # columns_blocks a vector
    columns_blocks = np.ones(5)
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)


def test_general_BTTB_bad_columns_blocks_without_rows_blocks():
    "must raise AssertionError for bad columns_blocks shape"
    num_blocks = 5
    # columns_blocks number of rows different from
    # (num_blocks) and (2*num_blocks - 1)
    columns_blocks = np.ones((7, 3))
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)


def test_general_BTTB_arbitrary():
    "verify if returned matrix is BTTB"
    Q = 4  # number of blocks along rows/columns
    P = 3  # number of rows/columns in each block
    # matrix containing the columns of each block
    columns = np.arange((2 * Q - 1) * P).reshape((2 * Q - 1, P))
    rows = np.arange((2 * Q - 1) * (P - 1)).reshape((2 * Q - 1, P - 1)) + 100
    BTTB = cv.general_BTTB(Q, columns, rows)
    # split matrix in blocks
    blocks = []
    for i in range(0, Q * P, P):
        blocks_i = []
        for j in range(0, Q * P, P):
            blocks_i.append(BTTB[i : i + P, j : j + P])
        blocks.append(blocks_i)
    # verify if each block is a toeplitz matrix
    for i in range(Q):
        for j in range(Q):
            block = blocks[i][j]
            block_ref = toeplitz(block[:, 0], block[0, :])
            ae(block, block_ref)
    # verify if each block is not symmetric
    for i in range(Q):
        for j in range(Q):
            ar(AssertionError, ae, blocks[i][j], blocks[i][j].T)
    # verify if block_ij is different from block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ar(AssertionError, ae, blocks[i][j], blocks[j][i])


def test_general_BTTB_SBTSTB():
    "verify if returned matrix is SBTSTB"
    Q = 4  # number of blocks along rows/columns
    P = 3  # number of rows/columns in each block
    # matrix containing the columns of each block
    columns = np.arange(Q * P).reshape((Q, P))
    BTTB = cv.general_BTTB(Q, columns)
    # split matrix in blocks
    blocks = []
    for i in range(0, Q * P, P):
        blocks_i = []
        for j in range(0, Q * P, P):
            blocks_i.append(BTTB[i : i + P, j : j + P])
        blocks.append(blocks_i)
    # verify if each block is symmetric
    for i in range(Q):
        for j in range(Q):
            ae(blocks[i][j], blocks[i][j].T)
    # verify if block_ij is equal to block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ae(blocks[i][j], blocks[j][i])


def test_general_BTTB_BTSTB():
    "verify if returned matrix is BTSTB"
    Q = 4  # number of blocks along rows/columns
    P = 3  # number of rows/columns in each block
    # matrix containing the columns of each block
    columns = np.arange((2 * Q - 1) * P).reshape((2 * Q - 1, P))
    BTTB = cv.general_BTTB(Q, columns)
    # split matrix in blocks
    blocks = []
    for i in range(0, Q * P, P):
        blocks_i = []
        for j in range(0, Q * P, P):
            blocks_i.append(BTTB[i : i + P, j : j + P])
        blocks.append(blocks_i)
    # verify if each block is symmetric
    for i in range(Q):
        for j in range(Q):
            ae(blocks[i][j], blocks[i][j].T)
    # verify if block_ij is different from block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ar(AssertionError, ae, blocks[i][j], blocks[j][i])


def test_general_BTTB_SBTTB():
    "verify if returned matrix is SBTTB"
    Q = 4  # number of blocks along rows/columns
    P = 3  # number of rows/columns in each block
    # matrix containing the columns of each block
    columns = np.arange(Q * P).reshape((Q, P))
    rows = np.arange(Q * (P - 1)).reshape((Q, P - 1)) + 100
    BTTB = cv.general_BTTB(Q, columns, rows)
    # split matrix in blocks
    blocks = []
    for i in range(0, Q * P, P):
        blocks_i = []
        for j in range(0, Q * P, P):
            blocks_i.append(BTTB[i : i + P, j : j + P])
        blocks.append(blocks_i)
    # verify if each block is a toeplitz matrix
    for i in range(Q):
        for j in range(Q):
            block = blocks[i][j]
            block_ref = toeplitz(block[:, 0], block[0, :])
            ae(block, block_ref)
    # verify if each block is not symmetric
    for i in range(Q):
        for j in range(Q):
            ar(AssertionError, ae, blocks[i][j], blocks[i][j].T)
    # verify if block_ij is equal to block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ae(blocks[i][j], blocks[j][i])


def test_general_BTTB_bad_columns_blocks_with_rows_blocks():
    "must raise AssertionError for inconsistent columns_blocks and row_blocks"
    num_blocks = 5
    # columns_blocks.shape[0] different from rows_blocks.shape[0]
    columns_blocks = np.ones((7, 3))
    row_blocks = np.ones((6, 3))
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)
    # columns_blocks.shape[1] different from (rows_blocks.shape[1]+1
    columns_blocks = np.ones((7, 3))
    row_blocks = np.ones((7, 3))
    with raises(AssertionError):
        cv.general_BTTB(num_blocks, columns_blocks)


def test_C_from_T_T_column_not_vector():
    "must raise AssertionError if T_column not a vector"
    # T_column a float
    T_column = 2.9
    with raises(AssertionError):
        cv.C_from_T(T_column)
    # T_column a matrix
    T_column = np.ones((5, 4))
    with raises(AssertionError):
        cv.C_from_T(T_column)


def test_C_from_T_row_not_vector():
    "must raise AssertionError if T_row not a vector"
    # T_row a float
    T_column = np.ones(3)
    T_row = 3.2
    with raises(AssertionError):
        cv.C_from_T(T_column, T_row)
    # T_row a matrix
    T_column = np.ones(3)
    T_row = np.ones((3, 2))
    with raises(AssertionError):
        cv.C_from_T(T_column, T_row)


def test_C_from_bad_T_column_T_row_sizes():
    "must raise AssertionError for inconsistent T_column and T_row sizes"
    # T_row size smaller than T_column.size - 1
    T_column = np.ones(5)
    T_row = np.ones(3)
    with raises(AssertionError):
        cv.C_from_T(T_column, T_row)
    # T_row size greater than T_column.size - 1
    T_column = np.ones(5)
    T_row = np.ones(7)
    with raises(AssertionError):
        cv.C_from_T(T_column, T_row)


def test_C_from_known_values():
    "compare result with a reference"
    # with T_row None
    T_column = np.arange(1, 4)
    C_ref = circulant(np.array([1, 2, 3, 0, 3, 2]))
    C = cv.C_from_T(T_column)
    ae(C, C_ref)
    # with T_row not None
    T_column = np.arange(1, 4)
    T_row = np.array([-0.8, 30])
    C_ref = circulant(np.array([1, 2, 3, 0, 30, -0.8]))
    C = cv.C_from_T(T_column, T_row)
    ae(C, C_ref)


def test_BCCB_from_BTTB_bad_num_blocks():
    "must raise AssertionError for bad num_blocks"
    columns_blocks = np.ones((3, 2))
    # num_blocks negative integer
    num_blocks = -5
    with raises(AssertionError):
        cv.BCCB_from_BTTB(num_blocks, columns_blocks)
    # num_blocks float
    num_blocks = 4.2
    with raises(AssertionError):
        cv.BCCB_from_BTTB(num_blocks, columns_blocks)
    # num_blocks equal to 1
    num_blocks = 1
    with raises(AssertionError):
        cv.BCCB_from_BTTB(num_blocks, columns_blocks)


def test_BCCB_from_BTTB_columns_blocks_not_matrix():
    "must raise AssertionError for columns_blocks not matrix"
    num_blocks = 5
    # columns_blocks a float
    columns_blocks = 2.9
    with raises(AssertionError):
        cv.BCCB_from_BTTB(num_blocks, columns_blocks)
    # columns_blocks a vector
    columns_blocks = np.ones(5)
    with raises(AssertionError):
        cv.BCCB_from_BTTB(num_blocks, columns_blocks)


def test_BCCB_from_BTTB_bad_columns_blocks_without_rows_blocks():
    "must raise AssertionError for bad columns_blocks shape"
    num_blocks = 5
    # columns_blocks number of rows different from
    # (num_blocks) and (2*num_blocks - 1)
    columns_blocks = np.ones((7, 3))
    with raises(AssertionError):
        cv.BCCB_from_BTTB(num_blocks, columns_blocks)


def test_BCCB_from_BTTB_arbitrary():
    "verify if returned matrix is BCCB"
    Q = 4  # number of blocks along rows/columns of BTTB matrix
    P = 3  # number of rows/columns in each block of BTTB matrix
    # matrix containing the columns of each block  of BTTB matrix
    columns = np.arange((2 * Q - 1) * P).reshape((2 * Q - 1, P))
    rows = np.arange((2 * Q - 1) * (P - 1)).reshape((2 * Q - 1, P - 1)) + 100
    BCCB = cv.BCCB_from_BTTB(Q, columns, rows)
    # split matrix in blocks
    blocks = []
    for i in range(0, 4 * Q * P, 2 * P):
        blocks_i = []
        for j in range(0, 4 * Q * P, 2 * P):
            blocks_i.append(BCCB[i : i + (2 * P), j : j + (2 * P)])
        blocks.append(blocks_i)
    # verify if each block is a circulant matrix
    for i in range(Q):
        for j in range(Q):
            block_ref = circulant(blocks[i][j][:, 0])
            ae(blocks[i][j], block_ref)
    # verify if each block is not symmetric
    for i in range(Q):
        for j in range(Q):
            ar(AssertionError, ae, blocks[i][j], blocks[i][j].T)
    # verify if block_ij is different from block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ar(AssertionError, ae, blocks[i][j], blocks[j][i])


def test_BCCB_from_BTTB_SBCSCB():
    "verify if returned matrix is SBCSCB"
    Q = 4  # number of blocks along rows/columns
    P = 3  # number of rows/columns in each block
    # matrix containing the columns of each block
    columns = np.arange(Q * P).reshape((Q, P))
    BCCB = cv.BCCB_from_BTTB(Q, columns)
    # split matrix in blocks
    blocks = []
    for i in range(0, 4 * Q * P, 2 * P):
        blocks_i = []
        for j in range(0, 4 * Q * P, 2 * P):
            blocks_i.append(BCCB[i : i + (2 * P), j : j + (2 * P)])
        blocks.append(blocks_i)
    # verify if each block is a circulant matrix
    for i in range(Q):
        for j in range(Q):
            block_ref = circulant(blocks[i][j][:, 0])
            ae(blocks[i][j], block_ref)
    # verify if each block is symmetric
    for i in range(Q):
        for j in range(Q):
            ae(blocks[i][j], blocks[i][j].T)
    # verify if block_ij is equal to block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ae(blocks[i][j], blocks[j][i])


def test_BCCB_from_BTTB_BCSCB():
    "verify if returned matrix is BCSCB"
    Q = 4  # number of blocks along rows/columns
    P = 3  # number of rows/columns in each block
    # matrix containing the columns of each block
    columns = np.arange((2 * Q - 1) * P).reshape((2 * Q - 1, P))
    BCCB = cv.BCCB_from_BTTB(Q, columns)
    # split matrix in blocks
    blocks = []
    for i in range(0, 4 * Q * P, 2 * P):
        blocks_i = []
        for j in range(0, 4 * Q * P, 2 * P):
            blocks_i.append(BCCB[i : i + (2 * P), j : j + (2 * P)])
        blocks.append(blocks_i)
    # verify if each block is a circulant matrix
    for i in range(Q):
        for j in range(Q):
            block_ref = circulant(blocks[i][j][:, 0])
            ae(blocks[i][j], block_ref)
    # verify if each block is symmetric
    for i in range(Q):
        for j in range(Q):
            ae(blocks[i][j], blocks[i][j].T)
    # verify if block_ij is different from block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ar(AssertionError, ae, blocks[i][j], blocks[j][i])


def test_BCCB_from_BTTB_SBCCB():
    "verify if returned matrix is SBCCB"
    Q = 4  # number of blocks along rows/columns
    P = 3  # number of rows/columns in each block
    # matrix containing the columns of each block
    columns = np.arange(Q * P).reshape((Q, P))
    rows = np.arange(Q * (P - 1)).reshape((Q, P - 1)) + 100
    BCCB = cv.BCCB_from_BTTB(Q, columns, rows)
    # split matrix in blocks
    blocks = []
    for i in range(0, 4 * Q * P, 2 * P):
        blocks_i = []
        for j in range(0, 4 * Q * P, 2 * P):
            blocks_i.append(BCCB[i : i + (2 * P), j : j + (2 * P)])
        blocks.append(blocks_i)
    # verify if each block is a circulant matrix
    for i in range(Q):
        for j in range(Q):
            block_ref = circulant(blocks[i][j][:, 0])
            ae(blocks[i][j], block_ref)
    # verify if each block is not symmetric
    for i in range(Q):
        for j in range(Q):
            ar(AssertionError, ae, blocks[i][j], blocks[i][j].T)
    # verify if block_ij is different from block_ji
    for i in range(Q):
        for j in range(i + 1, Q):
            ae(blocks[i][j], blocks[j][i])


def test_embedding_BCCB_first_column_bad_b0():
    "must raise ValueError for invalid b0"
    Q = 4
    P = 3
    symmetry = "symm-symm"
    # set b0 as a float
    b0 = 3.5
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    # set b0 as a matrix
    b0 = np.ones((3, 3))
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)


def test_embedding_BCCB_first_column_bad_QP():
    "must raise AssertionError for invalids Q and P"
    b0 = np.zeros(4)
    symmetry = "symm-symm"
    # set Q negative
    Q = -4
    P = 3
    with raises(AssertionError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    # set P negative
    Q = 4
    P = -3
    with raises(AssertionError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    # set Q as a float
    Q = 4.1
    P = 3
    with raises(AssertionError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    # set P as a float
    Q = 4
    P = 3.2
    with raises(AssertionError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)


def test_embedding_BCCB_first_column_bad_b0_QP():
    "must raise ValueError if b0.size is not Q*P"
    Q = 4
    P = 3
    symmetry = "symm-symm"
    # set b0.size greater than Q*P
    b0 = np.zeros(Q * P + 2)
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    # set b0.size smaller than Q*P
    b0 = np.zeros(Q * P - 1)
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)


def test_embedding_BCCB_first_column_bad_symmetry():
    "must raise ValueError for invalid symmetry"
    Q = 4
    P = 3
    b0 = np.zeros(Q * P)
    symmetry = "invalid-symmetry"
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)


def test_embedding_BCCB_first_column_known_values():
    "verify result obtained with specific input"
    Q = 2
    P = 3
    # verify result with symmetry 'symm-symm'
    symmetry = "skew-skew"
    b0 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    c0_true = np.array(
        [
            1.0,
            1.0,
            1.0,
            0.0,
            -1.0,
            -1.0,
            2.0,
            2.0,
            2.0,
            0.0,
            -2.0,
            -2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.0,
            -2.0,
            -2.0,
            0.0,
            2.0,
            2.0,
        ]
    )
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)
    # verify result with symmetry 'symm-symm'
    symmetry = "skew-symm"
    b0 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    c0_true = np.array(
        [
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            0.0,
            2.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.0,
            -2.0,
            -2.0,
            0.0,
            -2.0,
            -2.0,
        ]
    )
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)
    # verify result with symmetry 'symm-symm'
    symmetry = "symm-skew"
    b0 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    c0_true = np.array(
        [
            1.0,
            1.0,
            1.0,
            0.0,
            -1.0,
            -1.0,
            2.0,
            2.0,
            2.0,
            0.0,
            -2.0,
            -2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            2.0,
            0.0,
            -2.0,
            -2.0,
        ]
    )
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)
    # verify result with symmetry 'symm-symm'
    symmetry = "symm-symm"
    b0 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    c0_true = np.array(
        [
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            0.0,
            2.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            2.0,
            0.0,
            2.0,
            2.0,
        ]
    )
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)


def test_eigenvalues_BCCB_bad_c0():
    "must raise ValueError for invalid c0"
    Q = 4
    P = 3
    ordering = "row"
    # set c0 as a float
    c0 = 3.5
    with raises(ValueError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)
    # set c0 as a matrix
    c0 = np.ones((3, 3))
    with raises(ValueError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)


def test_eigenvalues_BCCB_bad_QP():
    "must raise AssertionError for invalids Q and P"
    c0 = np.zeros(4)
    ordering = "row"
    # set Q negative
    Q = -4
    P = 3
    with raises(AssertionError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)
    # set P negative
    Q = 4
    P = -3
    with raises(AssertionError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)
    # set Q as a float
    Q = 4.1
    P = 3
    with raises(AssertionError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)
    # set P as a float
    Q = 4
    P = 3.2
    with raises(AssertionError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)


def test_eigenvalues_BCCB_bad_c0_QP():
    "must raise ValueError if c0.size is not 4*Q*P"
    Q = 4
    P = 3
    ordering = "row"
    # set c0.size greater than 4*Q*P
    c0 = np.zeros(4 * Q * P + 2)
    with raises(ValueError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)
    # set c0.size smaller than 4*Q*P
    c0 = np.zeros(4 * Q * P - 1)
    with raises(ValueError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)


def test_eigenvalues_BCCB_bad_ordering():
    "must raise ValueError for invalid symmetry"
    Q = 4
    P = 3
    c0 = np.zeros(4 * Q * P)
    ordering = "invalid-ordering"
    with raises(ValueError):
        cv.eigenvalues_BCCB(c0, Q, P, ordering)


def test_eigenvalues_BCCB_known_values():
    "compare result with reference"
    Q = 4  # number of blocks along rows/columns of BTTB matrix
    P = 3  # number of rows/columns in each block of BTTB matrix
    # matrix containing the columns of each block  of BTTB matrix
    columns = np.arange((2 * Q - 1) * P).reshape((2 * Q - 1, P))
    rows = np.arange((2 * Q - 1) * (P - 1)).reshape((2 * Q - 1, P - 1)) + 100
    BCCB = cv.BCCB_from_BTTB(Q, columns, rows)
    # define unitaty DFT matrices
    F2Q = dft(n=2 * Q, scale="sqrtn")
    F2P = dft(n=2 * P, scale="sqrtn")
    # compute the Kronecker product between them
    F2Q_kron_F2P = np.kron(F2Q, F2P)
    # compute the reference eigenvalues of BCCB from its first column
    lambda_ref = np.sqrt(4 * Q * P) * np.dot(F2Q_kron_F2P, BCCB[:, 0])
    # compute eigenvalues with ordering='row'
    L_row = cv.eigenvalues_BCCB(BCCB[:, 0], Q, P, ordering="row")
    lambda_row = L_row.ravel()
    aae(lambda_row, lambda_ref, decimal=10)
    # compute eigenvalues with ordering='column'
    L_col = cv.eigenvalues_BCCB(BCCB[:, 0], Q, P, ordering="column")
    lambda_col = L_col.T.ravel()
    aae(lambda_col, lambda_ref, decimal=10)


def test_product_BCCB_vector_bad_L():
    "must raise AssertionError for bad L"
    Q = 4
    P = 3
    v = np.zeros(Q * P)
    # L float
    L = 1.3
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v)
    # L vector
    L = np.ones(3)
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v)
    # L matrix with shape different from (2*Q, 2*P) and ordering='row'
    L = np.zeros((2 * Q + 1, 2 * P - 1))
    ordering = "row"
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)
    # L matrix with shape different from (2*P, 2*Q) and ordering='column'
    L = np.zeros((2 * P + 1, 2 * Q - 1))
    ordering = "column"
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)


def test_product_BCCB_vector_bad_QP():
    "must raise AssertionError for bad Q and P"
    # define L and v considering Q = 4 and P = 3
    L = np.zeros((8, 6))  # shape (2*Q, 2*P)
    v = np.zeros(12)  # size Q*P
    ordering = "row"
    # Q negative
    Q = -4
    P = 3
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)
    # P negative
    Q = 4
    P = -3
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)
    # Q float
    Q = 4.1
    P = 3
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)
    # P float
    Q = 4
    P = 3.0
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)


def test_product_BCCB_vector_bad_v():
    "must raise AssertionError for bad v"
    Q = 4
    P = 3
    L = np.zeros((2 * Q, 2 * P))
    ordering = "row"
    # v float
    v = 1.3
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)
    # v matrix
    v = np.ones((3, 3))
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)
    # v vector with size different from Q*P
    v = np.ones(Q * P + 5)
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)


def test_product_BCCB_vector_bad_ordering():
    "must raise AssertionError for invalid ordering"
    Q = 4
    P = 3
    L = np.zeros((2 * Q, 2 * P))
    v = np.ones(Q * P)
    ordering = "invalid-ordering"
    with raises(AssertionError):
        cv.product_BCCB_vector(L, Q, P, v, ordering)


def test_product_BCCB_vector_compare_matrix_vector():
    "compare values with that obtained via matrix-vector product"
    Q = 4  # number of blocks along rows/columns of BTTB matrix
    P = 3  # number of rows/columns in each block of BTTB matrix
    # matrix containing the columns of each block  of BTTB matrix
    columns = np.arange((2 * Q - 1) * P).reshape((2 * Q - 1, P))
    rows = np.arange((2 * Q - 1) * (P - 1)).reshape((2 * Q - 1, P - 1)) + 100
    BTTB = cv.general_BTTB(Q, columns, rows)
    BCCB = cv.BCCB_from_BTTB(Q, columns, rows)
    # define a vector v
    np.random.seed(5)
    v = np.random.rand(Q * P)
    # define reference
    w_matvec = BTTB @ v
    # compute the product with function convolve.product_BCCB_vector
    # ordering='row'
    L = cv.eigenvalues_BCCB(BCCB[:, 0], Q, P, ordering="row")
    w_conv_row = cv.product_BCCB_vector(L, Q, P, v, ordering="row")
    aae(w_conv_row, w_matvec, decimal=12)
    # ordering='column'
    L = cv.eigenvalues_BCCB(BCCB[:, 0], Q, P, ordering="column")
    w_conv_col = cv.product_BCCB_vector(L, Q, P, v, ordering="column")
    aae(w_conv_col, w_matvec, decimal=12)
