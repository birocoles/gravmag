import numpy as np
import numpy.testing as npt
import pytest
from .. import check

##### rectangular prism


def test_rectangular_prism_n_prisms():
    "Check if return the correct number of prisms"
    model = {
        "x1": np.array([-100, -100, -100]),
        "x2": np.array([100, 100, 100]),
        "y1": np.array([-100, -100, -100]),
        "y2": np.array([100, 100, 100]),
        "z1": np.array([100, 100, 100]),
        "z2": np.array([200, 200, 200]),
    }
    P = check.are_rectangular_prisms(model)
    assert P == 3


def test_invalid_rectangular_prism_boundaries():
    "Check if passing an invalid prism boundaries raises an error"
    # wrong x boundaries
    model = {
        "x1": np.array([100, -100, -100]),
        "x2": np.array([-100, 100, 100]),
        "y1": np.array([-100, -100, -100]),
        "y2": np.array([100, 100, 100]),
        "z1": np.array([100, 100, 100]),
        "z2": np.array([200, 200, 200]),
    }
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # wrong y boundaries
    model = {
        "x1": np.array([-100, -100, -100]),
        "x2": np.array([100, 100, 100]),
        "y1": np.array([-100, 100, -100]),
        "y2": np.array([100, -100, 100]),
        "z1": np.array([100, 100, 100]),
        "z2": np.array([200, 200, 200]),
    }
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # wrong z boundaries
    model = {
        "x1": np.array([-100, -100, -100]),
        "x2": np.array([100, 100, 100]),
        "y1": np.array([-100, -100, -100]),
        "y2": np.array([100, 100, 100]),
        "z1": np.array([100, 100, 300]),
        "z2": np.array([200, 200, 200]),
    }
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)


def test_invalid_rectangular_prism():
    "Check if passing an invalid prism raises an error"
    # array
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # array with shape (2,4)
    model = np.empty((2, 4))
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # array with shape (1,4)
    model = np.empty((1, 5))
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # tuple
    model = (1, 5)
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # list
    model = [1, 2, 3.0]
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # float
    model = 10.2
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # dict with correct keys in wrong order
    model = {
        "x1": np.array([-100, -100, -100]),
        "x2": np.array([100, 100, 100]),
        "y2": np.array([100, 100, 100]),
        "y1": np.array([-100, -100, -100]),
        "z1": np.array([100, 100, 100]),
        "z2": np.array([200, 200, 200]),
    }
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # dict with one extra key
    model = {
        "x1": np.array([-100, -100, -100]),
        "x2": np.array([100, 100, 100]),
        "y1": np.array([-100, -100, -100]),
        "y2": np.array([100, 100, 100]),
        "z1": np.array([100, 100, 100]),
        "z2": np.array([200, 200, 200]),
        "qw": np.array([200, 200, 200]),
    }
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)
    # dict with one wrong key
    model = {
        "x1": np.array([-100, -100, -100]),
        "X2": np.array([100, 100, 100]),
        "y1": np.array([-100, -100, -100]),
        "y2": np.array([100, 100, 100]),
        "z1": np.array([100, 100, 100]),
        "z2": np.array([200, 200, 200]),
    }
    with pytest.raises(ValueError):
        check.are_rectangular_prisms(model)


##### coordinates


def test_coordinates_n_points():
    "Check if return the correct number of points"
    coordinates = {
        "x": np.array([-100, 100, -100, 100, 100, 200]),
        "y": np.array([-100, 100, -100, 100, 100, 200]),
        "z": np.array([-100, 100, -100, 100, 100, 200]),
    }
    D = check.are_coordinates(coordinates)
    assert D == 6


def test_invalid_coordinates():
    "Check if passing an invalid coordinates raises an error"
    # array
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        check.are_coordinates(coordinates)
    # array
    coordinates = np.zeros((4, 3))
    with pytest.raises(ValueError):
        check.are_coordinates(coordinates)
    # tuple
    coordinates = (1, 5)
    with pytest.raises(ValueError):
        check.are_coordinates(coordinates)
    # list
    coordinates = [1, 2, 3.0]
    with pytest.raises(ValueError):
        check.are_coordinates(coordinates)
    # float
    coordinates = 10.2
    with pytest.raises(ValueError):
        check.are_coordinates(coordinates)
    # dictionary with one extra key
    coordinates = {
        "x": np.array([-100, 100, -100, 100, 100, 200]),
        "y": np.array([-100, 100, -100, 100, 100, 200]),
        "z": np.array([-100, 100, -100, 100, 100, 200]),
        "k": np.array([-100, 100, -100, 100, 100, 200]),
    }
    with pytest.raises(ValueError):
        check.are_coordinates(coordinates)
    # dictionary with one wrong key
    coordinates = {
        "x": np.array([-100, 100, -100, 100, 100, 200]),
        "y": np.array([-100, 100, -100, 100, 100, 200]),
        "Z": np.array([-100, 100, -100, 100, 100, 200]),
    }
    with pytest.raises(ValueError):
        check.are_coordinates(coordinates)


##### grid


def test_is_planar_grid_n_points():
    "Check if return the correct number of points"
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "xy",
    }
    D = check.is_planar_grid(coordinates)
    assert D == 12


def test_is_planar_grid_non_dict_input():
    "Check if passing a non-dictionary grid raises an error"
    # array
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # array
    coordinates = np.zeros((4, 3))
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # tuple
    coordinates = (1, 5)
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # list
    coordinates = [1, 2, 3.0]
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # float
    coordinates = 10.2
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)


def test_is_planar_grid_invalid_keys():
    "Check if passing a dictionary with invalid keys raises an error"
    # dictionary with one extra key
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "xy",
        "k": np.array([-100, 100, -100, 100, 100, 200]),
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # dictionary with one wrong key (y in uppercase)
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "Y": np.ones(3),
        "z": 18.2,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # dictionary with 'z' key not float or int
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2 + 3j,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": np.array(18.2),
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # dictionary with 'ordering' key neither 'xy' nor 'yx'
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "y",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "x",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "Xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": ["xy"],
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": ("yx",),
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)


def test_is_planar_grid_invalid_x_key():
    "Check if passing an invalid x key raises an error"
    # array 1d
    coordinates = {
        "x": np.arange(4),
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # list
    coordinates = {
        "x": [0, 1, 2, 3, 4],
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # tuple
    coordinates = {
        "x": (0, 1, 2, 3, 4),
        "y": np.ones(3),
        "z": 18.2,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)


def test_is_planar_grid_invalid_y_key():
    "Check if passing an invalid y key raises an error"
    # array 2d
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3)[np.newaxis, :],
        "z": 18.2,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # list
    coordinates = {
        "x": np.ones(3),
        "y": [0, 1, 2, 3, 4],
        "z": 18.2,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # tuple
    coordinates = {
        "x": np.ones(4),
        "y": (0, 1, 2, 3),
        "z": 18.2,
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)


def test_is_planar_grid_invalid_z_key():
    "Check if passing an invalid z key raises an error"
    # array 2d
    coordinates = {
        "x": np.arange(4)[:, np.newaxis],
        "y": np.ones(3),
        "z": np.array(18.2),
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # list
    coordinates = {
        "x": np.ones(4),
        "y": np.ones(3),
        "z": [18.2],
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)
    # tuple
    coordinates = {
        "x": np.ones(4),
        "y": np.ones(3),
        "z": (18.2,),
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_planar_grid(coordinates)


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
    y = 34.0 + 5j
    with pytest.raises(ValueError):
        check.is_scalar(x=y, positive=False)


def test_scalar_non_positive():
    "Check if passing a non-positive raises an error"
    # zero
    y = 0.0
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
    y = 34.0 + 5j
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
    y = 34.0 + 5j
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=1, shape=(2,))


def test_array_wrong_ndim_shape():
    "Check if passing an array with wrong ndim and/or shape raises an error"
    # wrong ndim
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=2, shape=(3, 1))
    # wrong ndim
    y = np.array([[1], [2], [3]])
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=1, shape=(3, 1))
    # wrong shape
    y = np.array([[1, 2, 3], [0, 0, 0]])
    with pytest.raises(ValueError):
        check.is_array(x=y, ndim=2, shape=(2, 2))


##### wavenumbers


def test_invalid_wavenumbers():
    "Check if passing non-dictionary wavenumbers raises an error"
    # tuple
    w = (5, 7, 8.7)
    with pytest.raises(ValueError):
        check.are_wavenumbers(wavenumbers=w)
    # list
    w = [5, 7, 8.7]
    with pytest.raises(ValueError):
        check.are_wavenumbers(wavenumbers=w)
    # complex
    w = 34.0 + 5j
    with pytest.raises(ValueError):
        check.are_wavenumbers(wavenumbers=w)


def test_invalid_wavenumbers_xyz():
    "Check if passing dictionary of invalid wavenumbers x, y and z raises an error"
    # wrong 'x'
    kx = np.ones((3, 4))
    ky = np.ones((3, 4))
    ky[:, 0] = 0.0
    kz = np.ones((3, 4))
    w = {"x": kx, "y": ky, "z": kz}
    with pytest.raises(ValueError):
        check.are_wavenumbers(wavenumbers=w)
    # wrong 'y'
    kx = np.ones((3, 4))
    kx[0, :] = 0.0
    ky = np.ones((3, 4))
    kz = np.ones((3, 4))
    w = {"x": kx, "y": ky, "z": kz}
    with pytest.raises(ValueError):
        check.are_wavenumbers(wavenumbers=w)
    # wrong 'z'
    kx = np.ones((3, 4))
    kx[0, :] = 0.0
    ky = np.ones((3, 4))
    ky[:, 0] = 0.0
    kz = np.ones((3, 4))
    kz[1, 1] = -2.0
    w = {"x": kx, "y": ky, "z": kz}
    with pytest.raises(ValueError):
        check.are_wavenumbers(wavenumbers=w)


##### sensitivity matrix and data vector


def test_sensitivity_matrix_and_data_non_arrays():
    "Check if passing non-arrays raises an error"
    # G array, d tuple
    G = np.empty((4, 3))
    data = (9, 7.1)
    with pytest.raises(ValueError):
        check.sensitivity_matrix_and_data(matrix=G, data=data)
    # G float, d array
    G = 7.5
    data = np.empty(3)
    with pytest.raises(ValueError):
        check.sensitivity_matrix_and_data(matrix=G, data=data)
    # G 1d array, d 1d array
    G = np.empty(5)
    data = np.empty(5)
    with pytest.raises(ValueError):
        check.sensitivity_matrix_and_data(matrix=G, data=data)
    # G 2d array, d 2d array
    G = np.empty((4, 3))
    data = np.empty((3, 5))
    with pytest.raises(ValueError):
        check.sensitivity_matrix_and_data(matrix=G, data=data)
    # G complex, d list
    G = 7.2 + 9.1j
    data = [3, 5, 6.6]
    with pytest.raises(ValueError):
        check.sensitivity_matrix_and_data(matrix=G, data=data)


def test_sensitivity_matrix_and_data_mismatch():
    "Check if passing incompatible matrix and data raises an error"
    G = np.empty((5, 3))
    data = np.empty(4)
    with pytest.raises(ValueError):
        check.sensitivity_matrix_and_data(matrix=G, data=data)


###### Toeplitz_metadata


def test_Toeplitz_metadata_bad_symmetry():
    "must raise an error for bad symmetry"
    # wrong upercase
    Toeplitz = {
        "symmetry": "Symm",
        "column": np.ones(5),
        "row": None,
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)
    # invalid string
    Toeplitz = {
        "symmetry": "invalid-symmetry",
        "column": np.ones(5),
        "row": None,
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)
    # not string
    Toeplitz = {
        "symmetry": 5.1,
        "column": np.ones(5),
        "row": None,
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)


def test_Toeplitz_metadata_bad_column():
    "must raise an error if column is not a numpy array 1D"
    # array 2D
    Toeplitz = {
        "symmetry": "symm",
        "column": np.ones((5, 4)),
        "row": None,
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)
    # list
    Toeplitz = {
        "symmetry": "symm",
        "column": [1, 2, 3, 4],
        "row": None,
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)
    # float
    Toeplitz = {
        "symmetry": "symm",
        "column": 5.4,
        "row": None,
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)


def test_Toeplitz_metadata_consistency_symmetry_column_row():
    "must raise an error for inconsistent symmetry, column and row"
    # symmetry symm or skew and row not None
    Toeplitz = {
        "symmetry": "symm",
        "column": np.ones(5),
        "row": np.zeros(4),
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)
    Toeplitz = {
        "symmetry": "skew",
        "column": np.ones(5),
        "row": np.zeros(4),
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)
    # symmetry gene and row None
    Toeplitz = {
        "symmetry": "gene",
        "column": np.ones(5),
        "row": None,
    }
    with pytest.raises(ValueError):
        check.Toeplitz_metadata(Toeplitz)


##### BTTB_metadata


def test_BTTB_metadata_bad_symmetries():
    "must raise an error for bad symmetries"
    # wrong upercase
    BTTB = {
        "symmetry_structure": "Symm",
        "symmetry_blocks": "symm",
        "nblocks": 5,
        "columns": np.ones((5, 4)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # invalid string
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "invalid-symmetry",
        "nblocks": 5,
        "columns": np.ones((5, 4)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # not string
    BTTB = {
        "symmetry_structure": 4.5,
        "symmetry_blocks": "symm",
        "nblocks": 5,
        "columns": np.ones((5, 4)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)


def test_BTTB_metadata_bad_nblocks():
    "must raise an error for bad number of blocks"
    # negative integer
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": -5,
        "columns": np.ones((5, 4)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # not integer
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 5.0,
        "columns": np.ones((5, 4)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)


def test_BTTB_metadata_columns_not_array_2d():
    "must raise an error if columns is not a numpy array 2D"
    # array 1d
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 5,
        "columns": np.ones(5),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # list of lists
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": -5,
        "columns": [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
        ],
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)


def test_BTTB_metadata_consistency_symmetry_structure_symm():
    "must raise an error for inconsistent symmetries, nblocks and columns"
    # columns.shape[0] not equal to nblocks
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 5,
        "columns": np.ones((4, 5)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "skew",
        "nblocks": 5,
        "columns": np.ones((6, 5)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # symmetry_blocks is skew, rows not None
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "skew",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": np.zeros((5, 4)),
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # symmetry_blocks is skew, num rows in rows is not equal to nblocks
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "skew",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": np.zeros((9, 4)),
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # symmetry_structure is symm, symmetry_blocks is gene, num columns in rows is not equal to that in columns - 1
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "gene",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": np.zeros((5, 5)),
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)


def test_BTTB_metadata_consistency_symmetry_structure_skew():
    "must raise an error for inconsistent symmetries, nblocks and columns"
    # columns.shape[0] not equal to nblocks
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "symm",
        "nblocks": 5,
        "columns": np.ones((4, 5)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "symm",
        "nblocks": 5,
        "columns": np.ones((6, 5)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # symmetry_blocks is skew, rows not None
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "skew",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": np.ones((5, 4)),
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # symmetry_blocks is skew, num rows in rows is not equal to nblocks
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "skew",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": np.ones((9, 4)),
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # symmetry_blocks is gene, num columns in rows is not equal to that in columns - 1
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "gene",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": np.ones((5, 5)),
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)


def test_BTTB_metadata_consistency_symmetry_structure_gene():
    "must raise an error for inconsistent symmetries, nblocks and columns"
    # columns.shape[0] not equal to 2*nblocks - 1
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "symm",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "skew",
        "nblocks": 5,
        "columns": np.ones((10, 5)),
        "rows": None,
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
    # symmetry_structure is gene, symmetry_blocks is gene, num columns in rows is not equal to that in columns - 1
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 5,
        "columns": np.ones((5, 5)),
        "rows": np.zeros((9, 5)),
    }
    with pytest.raises(ValueError):
        check.BTTB_metadata(BTTB)
