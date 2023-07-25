import numpy as np
import numpy.testing as npt
import pytest
from .. import check

##### rectangular prism

def test_rectangular_prism_n_prisms():
    "Check if return the correct number of prisms"
    model = {
        'x1' : np.array([-100, -100, -100]),
        'x2' : np.array([ 100,  100,  100]),
        'y1' : np.array([-100, -100, -100]),
        'y2' : np.array([ 100,  100,  100]),
        'z1' : np.array([ 100,  100,  100]),
        'z2' : np.array([ 200,  200,  200])
        }
    P = check.rectangular_prisms(model)
    assert P == 3


def test_invalid_rectangular_prism_boundaries():
    "Check if passing an invalid prism boundaries raises an error"
    # wrong x boundaries
    model = {
        'x1' : np.array([ 100, -100, -100]),
        'x2' : np.array([-100,  100,  100]),
        'y1' : np.array([-100, -100, -100]),
        'y2' : np.array([ 100,  100,  100]),
        'z1' : np.array([ 100,  100,  100]),
        'z2' : np.array([ 200,  200,  200])
        }
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # wrong y boundaries
    model = {
        'x1' : np.array([-100, -100, -100]),
        'x2' : np.array([ 100,  100,  100]),
        'y1' : np.array([-100,  100, -100]),
        'y2' : np.array([ 100, -100,  100]),
        'z1' : np.array([ 100,  100,  100]),
        'z2' : np.array([ 200,  200,  200])
        }
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # wrong z boundaries
    model = {
        'x1' : np.array([-100, -100, -100]),
        'x2' : np.array([ 100,  100,  100]),
        'y1' : np.array([-100, -100, -100]),
        'y2' : np.array([ 100,  100,  100]),
        'z1' : np.array([ 100,  100,  300]),
        'z2' : np.array([ 200,  200,  200])
        }
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)


def test_invalid_rectangular_prism():
    "Check if passing an invalid prism raises an error"
    # array
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # array with shape (2,4)
    model = np.empty((2, 4))
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # array with shape (1,4)
    model = np.empty((1, 5))
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # tuple
    model = (1, 5)
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # list
    model = [1,2,3.]
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # float
    model = 10.2
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # dict with correct keys in wrong order
    model = {
        'x1' : np.array([-100, -100, -100]),
        'x2' : np.array([ 100,  100,  100]),
        'y2' : np.array([ 100,  100,  100]),
        'y1' : np.array([-100, -100, -100]),
        'z1' : np.array([ 100,  100,  100]),
        'z2' : np.array([ 200,  200,  200])
        }
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # dict with one extra key
    model = {
        'x1' : np.array([-100, -100, -100]),
        'x2' : np.array([ 100,  100,  100]),
        'y1' : np.array([-100, -100, -100]),
        'y2' : np.array([ 100,  100,  100]),
        'z1' : np.array([ 100,  100,  100]),
        'z2' : np.array([ 200,  200,  200]),
        'qw' : np.array([ 200,  200,  200])
        }
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # dict with one wrong key
    model = {
        'x1' : np.array([-100, -100, -100]),
        'X2' : np.array([ 100,  100,  100]),
        'y1' : np.array([-100, -100, -100]),
        'y2' : np.array([ 100,  100,  100]),
        'z1' : np.array([ 100,  100,  100]),
        'z2' : np.array([ 200,  200,  200])
        }
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)

##### coordinates

def test_coordinates_n_points():
    "Check if return the correct number of points"
    coordinates = {
        'x' : np.array([-100, 100, -100, 100, 100, 200]),
        'y' : np.array([-100, 100, -100, 100, 100, 200]),
        'z' : np.array([-100, 100, -100, 100, 100, 200])
        }
    D = check.coordinates(coordinates)
    assert D == 6


def test_invalid_coordinates():
    "Check if passing an invalid coordinates raises an error"
    # array
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        check.coordinates(coordinates)
    # array
    coordinates = np.zeros((4, 3))
    with pytest.raises(ValueError):
        check.coordinates(coordinates)
    # tuple
    coordinates = (1, 5)
    with pytest.raises(ValueError):
        check.coordinates(coordinates)
    # list
    coordinates = [1,2,3.]
    with pytest.raises(ValueError):
        check.coordinates(coordinates)
    # float
    coordinates = 10.2
    with pytest.raises(ValueError):
        check.coordinates(coordinates)
    # dictionary with one extra key
    coordinates = {
        'x' : np.array([-100, 100, -100, 100, 100, 200]),
        'y' : np.array([-100, 100, -100, 100, 100, 200]),
        'z' : np.array([-100, 100, -100, 100, 100, 200]),
        'k' : np.array([-100, 100, -100, 100, 100, 200])
        }
    with pytest.raises(ValueError):
        check.coordinates(coordinates)
    # dictionary with one wrong key
    coordinates = {
        'x' : np.array([-100, 100, -100, 100, 100, 200]),
        'y' : np.array([-100, 100, -100, 100, 100, 200]),
        'Z' : np.array([-100, 100, -100, 100, 100, 200])
        }
    with pytest.raises(ValueError):
        check.coordinates(coordinates)

##### is_scalar

def test_invalid_scalar():
    "Check if passing a non-scalar raises an error"
    # tuple
    y = (5, 7, 8.7)
    with pytest.raises(ValueError):
        check.is_scalar(x=y, positive=False)
    # list
    y = [5, 7, 8.7]
    with pytest.raises(ValueError):
        check.is_scalar(x=y, positive=False)
    # array
    y = np.array([5, 7, 8.7])
    with pytest.raises(ValueError):
        check.is_scalar(x=y, positive=False)
    # complex
    y = 34. + 5j
    with pytest.raises(ValueError):
        check.is_scalar(x=y, positive=False)


def test_scalar_non_positive():
    "Check if passing a non-positive raises an error"
    # zero
    y = 0.
    with pytest.raises(ValueError):
        check.is_scalar(x=y, positive=True)
    # negative
    y = -5.7
    with pytest.raises(ValueError):
        check.is_scalar(x=y, positive=True)


##### is_integer

def test_invalid_integer():
    "Check if passing a non-integer raises an error"
    # tuple
    y = (5, 7, 8.7)
    with pytest.raises(ValueError):
        check.is_integer(x=y, positive=False)
    # list
    y = [5, 7, 8.7]
    with pytest.raises(ValueError):
        check.is_integer(x=y, positive=False)
    # array
    y = np.array([5, 7, 8.7])
    with pytest.raises(ValueError):
        check.is_integer(x=y, positive=False)
    # complex
    y = 34. + 5j
    with pytest.raises(ValueError):
        check.is_integer(x=y, positive=False)
    # float
    y = 34.1
    with pytest.raises(ValueError):
        check.is_integer(x=y, positive=False)


def test_integer_non_positive():
    "Check if passing a non-integer raises an error"
    # zero
    y = 0
    with pytest.raises(ValueError):
        check.is_integer(x=y, positive=True)
    # negative
    y = -5
    with pytest.raises(ValueError):
        check.is_integer(x=y, positive=True)


##### is_array

def test_invalid_array():
    "Check if passing a non-array raises an error"
    # tuple
    y = (5, 7, 8.7)
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=1, shape=(4,))
    # list
    y = [5, 7, 8.7]
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=1, shape=(4,))
    # complex
    y = 34. + 5j
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=1, shape=(2,))


def test_array_wrong_ndim_shape():
    "Check if passing an array with wrong ndim and/or shape raises an error"
    # wrong ndim
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=2, shape=(3,1))
    # wrong ndim
    y = np.array([[1], [2], [3]])
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=1, shape=(3,1))
    # wrong shape
    y = np.array(
        [[1, 2, 3],
         [0, 0, 0]])
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=2, shape=(2,2))


##### sensibility matrix and data vector

def test_sensibility_matrix_and_data_non_arrays():
    "Check is passing non-arrays raises an error"
    # G array, d tuple
    G = np.empty((4,3))
    data = (9, 7.1)
    with pytest.raises(ValueError):
        check.sensibility_matrix_and_data(G=G, data=data)
    # G float, d array
    G = 7.5
    data = np.empty(3)
    with pytest.raises(ValueError):
        check.sensibility_matrix_and_data(G=G, data=data)
    # G 1d array, d 1d array
    G = np.empty(5)
    data = np.empty(5)
    with pytest.raises(ValueError):
        check.sensibility_matrix_and_data(G=G, data=data)
    # G 2d array, d 2d array
    G = np.empty((4,3))
    data = np.empty((3,5))
    with pytest.raises(ValueError):
        check.sensibility_matrix_and_data(G=G, data=data)
    # G complex, d list
    G = 7.2+9.1j
    data = [3, 5, 6.6]
    with pytest.raises(ValueError):
        check.sensibility_matrix_and_data(G=G, data=data)