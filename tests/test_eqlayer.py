import numpy as np
from scipy.linalg import toeplitz, circulant, hankel
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from pytest import raises
from gravmag import eqlayer

##### method_CGLS


def test_method_CGLS_invalid_sensitivity_matrices():
    "Check if passing an invalid list of sensibility matrices raises an error"
    data = [np.zeros(4), np.empty(6), np.ones(3)]
    eps = 1e-3
    ITMAX = 10
    # sensibility not array 2d
    G = [np.zeros((4, 5)), np.empty((6, 5)), ["invalid-matrix"]]
    with raises(ValueError):
        eqlayer.method_CGLS(
            sensitivity_matrices=G,
            data_vectors=data,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
    # less matrices than datasets
    G = [np.zeros((4, 5)), np.empty((6, 5))]
    with raises(ValueError):
        eqlayer.method_CGLS(
            sensitivity_matrices=G,
            data_vectors=data,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
    # one matrix with wrong number of columns
    G = [np.zeros((4, 6)), np.empty((6, 5)), np.ones((3, 5))]
    with raises(ValueError):
        eqlayer.method_CGLS(
            sensitivity_matrices=G,
            data_vectors=data,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )


def test_method_CGLS_invalid_data_vectors():
    "Check if passing an invalid list of data vectors raises an error"
    G = [np.zeros((4, 5)), np.empty((6, 5)), np.ones((3, 5))]
    eps = 1e-3
    ITMAX = 10
    # not array 1d
    data = [np.zeros(4), ["invalid-data-vector"], np.ones(3)]
    with raises(ValueError):
        eqlayer.method_CGLS(
            sensitivity_matrices=G,
            data_vectors=data,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
    # less datasets than matrices
    data = [np.empty(6), np.ones(3)]
    with raises(ValueError):
        eqlayer.method_CGLS(
            sensitivity_matrices=G,
            data_vectors=data,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
    # one data vector with wrong number of elements
    data = [np.zeros(4), np.empty(7), np.ones(3)]
    with raises(ValueError):
        eqlayer.method_CGLS(
            sensitivity_matrices=G,
            data_vectors=data,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )


def test_method_CGLS_true_parameter_vector():
    "Check if passing the method retrieves the true parameter vector"
    eps = 1e-3
    ITMAX = 10
    # define square matrices with order 5
    matrices = [
        toeplitz(np.arange(1, 6)),
        circulant(np.linspace(3.1, 11.0, 5)),
        hankel([2, 3.5, 7.0, 1, 9.3]),
    ]
    # compute data vectors with a non-null parameter vector
    data = []
    parameters_true = np.array([2.0, 3.1, 7.0, 1.0, 4.5])
    for G in matrices:
        data.append(G @ parameters_true)
    # run the method CGLS
    delta_list, parameters = eqlayer.method_CGLS(
        sensitivity_matrices=matrices,
        data_vectors=data,
        epsilon=eps,
        ITMAX=ITMAX,
        check_input=True,
    )
    aae(parameters, parameters_true, decimal=10)


#### method_column_action_C92


def test_method_column_action_C92_invalid_sensibility_matrices():
    "Check if passing a invalid sensibility matrix raises an error"
    data = np.empty(6)
    coords = {
        "x": np.zeros(6),
        "y": np.zeros(6),
        "z": np.ones(6),
    }
    eps = 1e-3
    ITMAX = 10
    z_layer = 300.0
    # sensibility not array 2d
    G = ["invalid-matrix"]
    with raises(ValueError):
        eqlayer.method_column_action_C92(
            sensitivity_matrix=G,
            data=data,
            data_points=coords,
            zlayer=z_layer,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
    # matrix with number of columns smaller than number of data
    G = np.empty((7, 5))
    with raises(ValueError):
        eqlayer.method_column_action_C92(
            sensitivity_matrix=G,
            data=data,
            data_points=coords,
            zlayer=z_layer,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )


def test_method_column_action_C92_invalid_data_vectors():
    "Check if passing an invalid data vector raises an error"
    G = np.empty((7, 5))
    coords = {
        "x": np.zeros(6),
        "y": np.zeros(6),
        "z": np.ones(6),
    }
    eps = 1e-3
    ITMAX = 10
    z_layer = 300.0
    # data not array 1d
    data = ["invalid-vector"]
    with raises(ValueError):
        eqlayer.method_column_action_C92(
            sensitivity_matrix=G,
            data=data,
            data_points=coords,
            zlayer=z_layer,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
    # vector with more elements than columns of sensibilit matrix
    data = np.empty(6)
    with raises(ValueError):
        eqlayer.method_column_action_C92(
            sensitivity_matrix=G,
            data=data,
            data_points=coords,
            zlayer=z_layer,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )


def test_method_column_action_C92_invalid_layer_depth():
    "Check if passing an invalid layer depth raises an error"
    G = np.empty((7, 6))
    data = np.empty(6)
    coords = {
        "x": np.zeros(6),
        "y": np.zeros(6),
        "z": np.ones(6),
    }
    eps = 1e-3
    ITMAX = 10
    # layer depth not scalar
    z_layer = "invalid-layer-depth"
    with raises(ValueError):
        eqlayer.method_column_action_C92(
            sensitivity_matrix=G,
            data=data,
            data_points=coords,
            zlayer=z_layer,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
    # layer above data
    z_layer = 0.5
    with raises(ValueError):
        eqlayer.method_column_action_C92(
            sensitivity_matrix=G,
            data=data,
            data_points=coords,
            zlayer=z_layer,
            epsilon=eps,
            ITMAX=ITMAX,
            check_input=True,
        )
