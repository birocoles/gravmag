import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from pytest import raises
from .. import data_structures as ds

# grid_xy


def test_grid_xy_bad_input():
    "must raise error if receive bad input"
    # good parameters
    area = [1, 2.2, -3.1, 4]
    shape = (12, 11)
    z0 = 12.3

    # area not list
    with raises(ValueError):
        ds.grid_xy(area=(1, 2.2, -3.1, 4), shape=shape, z0=z0)
    # len area different from 4
    with raises(ValueError):
        ds.grid_xy(area=[1, 2.2, -3.1], shape=shape, z0=z0)
    # shape not tuple
    with raises(ValueError):
        ds.grid_xy(area=area, shape=[12, 11], z0=z0)
    # len shape different from 2
    with raises(ValueError):
        ds.grid_xy(area=area, shape=(12, 11, 10), z0=z0)
    # z0 complex
    with raises(ValueError):
        ds.grid_xy(area=area, shape=shape, z0=12.3 + 5j)
    # z0 list
    with raises(ValueError):
        ds.grid_xy(area=area, shape=shape, z0=[12.3])


def test_grid_xy_output():
    "compare output with reference"
    area = [1, 5, 14.5, 17.5]
    shape = (5, 4)
    z0 = 10
    reference = {
        "x": np.array([1, 2, 3, 4, 5]),
        "y": np.array([14.5, 15.5, 16.5, 17.5]),
        "z": 10,
        "area": area,
        "shape": shape,
    }
    computed = ds.grid_xy(area=area, shape=shape, z0=z0)
    ae(reference, computed)


# grid_xy_to_full_flatten


def test_grid_xy_to_full_flatten_known_values():
    "compare results with reference values obtained for specific input"
    # reference grid
    grid = {
        "x": np.arange(3),
        "y": np.array([10.3, 12.4]),
        "z": 10.0,
        "area": [0, 3, 10.3, 12.4],
        "shape": (3, 2),
    }
    reference_xy = {
        "x": np.array([0, 1, 2, 0, 1, 2]),
        "y": np.array([10.3, 10.3, 10.3, 12.4, 12.4, 12.4]),
        "z": np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
    }
    reference_yx = {
        "x": np.array([0, 0, 1, 1, 2, 2]),
        "y": np.array([10.3, 12.4, 10.3, 12.4, 10.3, 12.4]),
        "z": np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
    }
    computed_xy = ds.grid_xy_to_full_flatten(grid=grid, grid_orientation="xy")
    computed_yx = ds.grid_xy_to_full_flatten(grid=grid, grid_orientation="yx")
    ae(computed_xy, reference_xy)
    ae(computed_yx, reference_yx)


# grid_xy_full_flatten_to_matrix


def test_grid_xy_full_flatten_to_matrix():
    "compare result with reference values obtained for specific input"
    data = np.array([10, 20, 30, 40, 50, 60])
    shape = (3, 2)
    reference_xy = np.array([[10, 40], [20, 50], [30, 60]])
    reference_yx = np.array([[10, 20], [30, 40], [50, 60]])
    computed_xy = ds.grid_xy_full_flatten_to_matrix(
        data=data, grid_orientation="xy", shape=shape
    )
    computed_yx = ds.grid_xy_full_flatten_to_matrix(
        data=data, grid_orientation="yx", shape=shape
    )
    ae(computed_xy, reference_xy)
    ae(computed_yx, reference_yx)


# grid_xy_full_matrix_to_flatten


def test_grid_xy_full_matrix_to_flatten():
    "compare result with reference values obtained for specific input"
    grid_xy = np.array([[10, 40], [20, 50], [30, 60]])
    grid_yx = np.array([[10, 20], [30, 40], [50, 60]])
    reference = np.array([10, 20, 30, 40, 50, 60])
    computed_xy = ds.grid_xy_full_matrix_to_flatten(grid=grid_xy, grid_orientation="xy")
    computed_yx = ds.grid_xy_full_matrix_to_flatten(grid=grid_yx, grid_orientation="yx")
    ae(computed_xy, reference)
    ae(computed_yx, reference)


# grid_xy_to_full_matrices_view


def test_grid_xy_to_full_matrices_view():
    "compare result with reference values obtained for specific input"
    x = np.array([0, 1, 2])
    y = np.array([10.3, 12.4])
    shape = (3, 2)
    X = np.array([[0, 0], [1, 1], [2, 2]])
    Y = np.array([[10.3, 12.4], [10.3, 12.4], [10.3, 12.4]])
    comp_X, comp_Y = ds.grid_xy_to_full_matrices_view(x=x, y=y, shape=shape)
    # compare with reference values
    ae(comp_X, X)
    ae(comp_Y, Y)
    # verify if computed are actually views from original data
    with raises(ValueError):
        comp_X[0, 0] = -1.2
    with raises(ValueError):
        comp_Y[0, 0] = -1.2


# grid_xy_spacing


def test_grid_xy_spacing():
    "compare result with reference values obtained for specific input"
    area = [0, 10, 5.5, 7.5]
    shape = (5, 3)
    reference = (2.5, 1.0)
    computed = ds.grid_xy_spacing(area=area, shape=shape)
    ae(computed, reference)


# grid_wavenumbers


def test_grid_wavenumbers_output_without_pad():
    "compare output with reference"
    grid = {
        "x": np.arange(5) * 1.3 + 10.0,
        "y": np.arange(4) * 1.1 - 3.2,
        "z": 10.0,
        "area": [10, 10 + 4 * 1.3, -3.2, -3.2 + 3 * 1.1],
        "shape": (5, 4),
    }
    x_ref = 2 * np.pi * np.array([0, 1, 2, -2, -1]) / (5 * 1.3)
    y_ref = 2 * np.pi * np.array([0, 1, -2, -1]) / (4 * 1.1)
    reference = {
        "x": x_ref,
        "y": y_ref,
        "z": np.sqrt(x_ref[:, np.newaxis] ** 2 + y_ref**2),
        "shape": (5, 4),
        "spacing": (1.3, 1.1),
    }
    computed = ds.grid_wavenumbers(grid=grid)
    aae(reference["x"], computed["x"], decimal=15)
    aae(reference["y"], computed["y"], decimal=15)
    aae(reference["z"], computed["z"], decimal=15)
    ae(reference["shape"], computed["shape"])
    aae(reference["spacing"], computed["spacing"], decimal=15)


def test_grid_wavenumbers_output_with_pad():
    "compare output with reference"
    grid = {
        "x": np.arange(5) * 1.3 + 10.0,
        "y": np.arange(4) * 1.1 - 3.2,
        "z": 10.0,
        "area": [10, 10 + 4 * 1.3, -3.2, -3.2 + 3 * 1.1],
        "shape": (5, 4),
    }
    x_ref = (
        2
        * np.pi
        * np.array([0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1])
        / (15 * 1.3)
    )
    y_ref = (
        2
        * np.pi
        * np.array([0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1])
        / (12 * 1.1)
    )
    reference = {
        "x": x_ref,
        "y": y_ref,
        "z": np.sqrt(x_ref[:, np.newaxis] ** 2 + y_ref**2),
        "shape": (15, 12),
        "spacing": (1.3, 1.1),
    }
    computed = ds.grid_wavenumbers(grid=grid, pad=True)
    aae(reference["x"], computed["x"], decimal=15)
    aae(reference["y"], computed["y"], decimal=15)
    aae(reference["z"], computed["z"], decimal=15)
    ae(reference["shape"], computed["shape"])
    aae(reference["spacing"], computed["spacing"], decimal=15)
