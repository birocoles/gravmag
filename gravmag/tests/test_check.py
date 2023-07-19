import numpy as np
import numpy.testing as npt
import pytest
from .. import check

##### rectangular prism

def test_rectangular_prism_n_prisms():
    "Check if return the correct number of prisms"
    model = np.array([
        [-100, 100, -100, 100, 100, 200],
        [-100, 100, -100, 100, 100, 200],
        [-100, 100, -100, 100, 100, 200]
        ])
    P = check.rectangular_prisms(model)
    assert P == 3


def test_invalid_rectangular_prism_boundaries():
    "Check if passing an invalid prism boundaries raises an error"
    # wrong x boundaries
    model = np.array([[100, -100, -100, 100, 100, 200]])
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # wrong y boundaries
    model = np.array([[-100, 100, 100, -100, 100, 200]])
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # wrong z boundaries
    model = np.array([[-100, 100, -100, 100, 200, 100]])
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)


def test_invalid_rectangular_prism():
    "Check if passing an invalid prism raises an error"
    # array with shape (1,)
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

##### coordinates

def test_coordinates_n_points():
    "Check if return the correct number of points"
    coordinates = np.array([
        [-100, 100, -100, 100, 100, 200],
        [-100, 100, -100, 100, 100, 200],
        [-100, 100, -100, 100, 100, 200]
        ])
    D = check.coordinates(coordinates)
    assert D == 6


def test_invalid_coordinates():
    "Check if passing an invalid coordinates raises an error"
    # shape (1,)
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        check.coordinates(coordinates)
    # shape (4,3)
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


##### scalar properties


def test_invalid_rectangular_prism_densities():
    "Check if passing invalid densities raises an error"
    # number of prisms
    P = 2
    # array with ndim != 1
    density = np.array([[1000], [2000]])
    with pytest.raises(ValueError):
        check.scalar_prop(density, P)
    # tuple
    density = (1, 5)
    with pytest.raises(ValueError):
        check.scalar_prop(density, P)
    # list
    density = [1,2,3.]
    with pytest.raises(ValueError):
        check.scalar_prop(density, P)
    # float
    density = 10.2
    with pytest.raises(ValueError):
        check.scalar_prop(density, P)
    # density.size != P
    density = np.array([1000, 2000, 66])
    with pytest.raises(ValueError):
        check.scalar_prop(density, P)

##### vector properties


def test_invalid_rectangular_prism_magnetizations():
    "Check if passing invalid magnetizations raises an error"
    # number of prisms
    P = 3
    # array with ndim != 2
    magnetization = np.array([2.1, -13, 22])
    with pytest.raises(ValueError):
        check.vector_prop(magnetization, P)
    # array with shape[1] != 3
    magnetization = np.array([[2.1, -13], [1.1, 45], [0.5, 10]])
    with pytest.raises(ValueError):
        check.vector_prop(magnetization, P)
    # tuple
    magnetization = (1, 5)
    with pytest.raises(ValueError):
        check.vector_prop(magnetization, P)
    # list
    magnetization = [1,2,3.]
    with pytest.raises(ValueError):
        check.vector_prop(magnetization, P)
    # float
    magnetization = 10.2
    with pytest.raises(ValueError):
        check.vector_prop(magnetization, P)
    # magnetization.shape[0] != P
    magnetization = np.zeros((2,6))
    with pytest.raises(ValueError):
        check.vector_prop(magnetization, P)

##### scalar

def test_invalid_scalar():
    "Check if passing a non-scalar raises an error"
    # tuple
    y = (5, 7, 8.7)
    with pytest.raises(ValueError):
        check.scalar(y, positive=False)
    # list
    y = [5, 7, 8.7]
    with pytest.raises(ValueError):
        check.scalar(y, positive=False)
    # array
    y = np.array([5, 7, 8.7])
    with pytest.raises(ValueError):
        check.scalar(y, positive=False)
    # complex
    y = 34. + 5j
    with pytest.raises(ValueError):
        check.scalar(y, positive=False)


def test_scalar_non_positive():
    "Check if passing a non-positive raises an error"
    # zero
    y = 0.
    with pytest.raises(ValueError):
        check.scalar(y, positive=True)
    # negative
    y = -5.7
    with pytest.raises(ValueError):
        check.scalar(y, positive=True)