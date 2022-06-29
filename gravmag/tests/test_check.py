import numpy as np
import numpy.testing as npt
import pytest
from .. import check


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
    # shape (1,)
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # shape (2,4)
    model = np.empty((2, 4))
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)
    # shape (1,4)
    model = np.empty((1, 5))
    with pytest.raises(ValueError):
        check.rectangular_prisms(model)


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


def test_invalid_rectangular_prism_densities():
    "Check if passing invalid densities raises an error"
    model = np.array(
        [[-100, 100, -100, 100, 100, 200], [-300, 100, -84, 100, 400, 500]]
    )
    # density.ndim != 1
    density = np.array([[1000], [2000]])
    with pytest.raises(ValueError):
        check.density(density, model)
    # density.size != prisms.shape[0]
    density = np.array([1000, 2000, 66])
    with pytest.raises(ValueError):
        check.density(density, model)


def test_invalid_rectangular_prism_magnetizations():
    "Check if passing invalid magnetizations raises an error"
    model = np.array(
        [[-100, 100, -100, 100, 100, 200], [-300, 100, -84, 100, 400, 500]]
    )
    # magnetization.ndim != 2
    magnetization = np.array([2.1, -13, 22])
    with pytest.raises(ValueError):
        check.magnetization(magnetization, model)
    # magnetization.shape[1] != 3
    magnetization = np.array([[2.1, -13], [1.1, 45], [0.5, 10]])
    with pytest.raises(ValueError):
        check.magnetization(magnetization, model)
    # magnetization.shape[0] != prisms.shape[0]
    magnetization = np.array([[2.1, -13, 10], [1.1, 45, -18], [0.5, -31, 70]])
    with pytest.raises(ValueError):
        check.magnetization(magnetization, model)
