import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import raises
from .. import convolve as cv


def test_embedding_BCCB_first_column_bad_b0():
    'must raise ValueError for invalid b0'
    Q = 4
    P = 3
    symmetry = 'symm-symm'
    # set b0 as a float
    b0 = 3.5
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    # set b0 as a matrix
    b0 = np.ones((3,3))
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)


def test_embedding_BCCB_first_column_bad_QP():
    'must raise AssertionError for invalids Q and P'
    b0 = np.zeros(4)
    symmetry = 'symm-symm'
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
    'must raise ValueError if b0.size is not Q*P'
    Q = 4
    P = 3
    symmetry = 'symm-symm'
    # set b0.size greater than Q*P
    b0 = np.zeros(Q*P + 2)
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    # set b0.size smaller than Q*P
    b0 = np.zeros(Q*P - 1)
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)


def test_embedding_BCCB_first_column_bad_symmetry():
    'must raise ValueError for invalid symmetry'
    Q = 4
    P = 3
    b0 = np.zeros(Q*P)
    symmetry = 'invalid-symmetry'
    with raises(ValueError):
        cv.embedding_BCCB_first_column(b0, Q, P, symmetry)


def test_embedding_BCCB_first_column_known_values():
    'verify result obtained with specific input'
    Q = 2
    P = 3
    # verify result with symmetry 'symm-symm'
    symmetry = 'skew-skew'
    b0 = np.array([1., 1., 1., 2., 2., 2.])
    c0_true = np.array([ 1.,  1.,  1.,  0., -1., -1.,
                         2.,  2.,  2.,  0., -2., -2.,
                         0.,  0.,  0.,  0.,  0.,  0.,
                        -2., -2., -2.,  0.,  2.,  2.])
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)
    # verify result with symmetry 'symm-symm'
    symmetry = 'skew-symm'
    b0 = np.array([1., 1., 1., 2., 2., 2.])
    c0_true = np.array([ 1.,  1.,  1.,  0.,  1.,  1.,
                         2.,  2.,  2.,  0.,  2.,  2.,
                         0.,  0.,  0.,  0.,  0.,  0.,
                        -2., -2., -2.,  0., -2., -2.])
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)
    # verify result with symmetry 'symm-symm'
    symmetry = 'symm-skew'
    b0 = np.array([1., 1., 1., 2., 2., 2.])
    c0_true = np.array([ 1.,  1.,  1.,  0., -1., -1.,
                         2.,  2.,  2.,  0., -2., -2.,
                         0.,  0.,  0.,  0.,  0.,  0.,
                         2.,  2.,  2.,  0., -2., -2.])
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)
    # verify result with symmetry 'symm-symm'
    symmetry = 'symm-symm'
    b0 = np.array([1., 1., 1., 2., 2., 2.])
    c0_true = np.array([ 1.,  1.,  1.,  0.,  1.,  1.,
                         2.,  2.,  2.,  0.,  2.,  2.,
                         0.,  0.,  0.,  0.,  0.,  0.,
                         2.,  2.,  2.,  0.,  2.,  2.])
    c0 = cv.embedding_BCCB_first_column(b0, Q, P, symmetry)
    aae(c0, c0_true, decimal=15)
