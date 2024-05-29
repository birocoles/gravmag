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


##### grid_xy


def test_is_grid_xy_n_points():
    "Check if return the correct number of points"
    grid = {
        "x": np.arange(4),
        "y": np.ones(3),
        "z": 18.2,
        "area": [0, 1, 2, 3],
        "shape": (4, 3)
    }
    D = check.is_grid_xy(grid)
    assert D == 12


def test_is_grid_xy_non_dict_input():
    "Check if passing a non-dictionary grid raises an error"
    # array
    grid = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # array
    grid = np.zeros((4, 3))
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # tuple
    grid = (1, 5)
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # list
    grid = [1, 2, 3.0]
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # float
    grid = 10.2
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)


def test_is_grid_xy_invalid_keys():
    "Check if passing a dictionary with invalid keys raises an error"
    # correct keys
    x = np.arange(4)
    y = np.ones(3)
    z = 18.2
    area = [0, 1, 2, 3]
    shape = (4, 3)
    # dictionary with one extra key
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": area,
        "shape": shape,
        "k": np.array([-100, 100, -100, 100, 100, 200]),
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # dictionary with one wrong key (y in uppercase)
    grid = {
        "x": x,
        "Y": y,
        "z": z,
        "area": area,
        "shape": shape,
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # dictionary with 'z' key not float or int
    grid = {
        "x": x,
        "y": y,
        "z": 18.2 + 3j,
        "area": area,
        "shape": shape,
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    grid = {
        "x": x,
        "y": y,
        "z": np.array([18.2]),
        "area": area,
        "shape": shape,
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # dictionary with 'area' key not list
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": np.array(area),
        "shape": shape,
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": (0, 2, 4, 5),
        "shape": shape,
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # dictionary with inconsistent 'area' key
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": [0, -4, 3, 6],
        "shape": shape,
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": [1, 4, 7, 5],
        "shape": shape,
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # dictionary with 'shape' key not tuple
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": area,
        "shape": [x.size, y.size],
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": area,
        "shape": np.array(shape),
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # dictionary with inconsistent 'shape' key
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": area,
        "shape": (-1, 4),
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "area": area,
        "shape": (3, 3),
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)    


def test_is_grid_xy_invalid_x_key():
    "Check if passing an invalid x key raises an error"
    # correct keys
    y = np.ones(3)
    z = 18.2
    area = [0, 1, 2, 3]
    shape = (4, 3)
    # array 2d
    grid = {
        "x": np.arange(4)[:, np.newaxis],
        "y": y,
        "z": z,
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # list
    grid = {
        "x": [0, 1, 2, 3, 4],
        "y": y,
        "z": z,
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # tuple
    grid = {
        "x": (0, 1, 2, 3, 4),
        "y": y,
        "z": z,
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)


def test_is_grid_xy_invalid_y_key():
    "Check if passing an invalid y key raises an error"
    # correct keys
    x = np.arange(4)
    z = 18.2
    area = [0, 1, 2, 3]
    shape = (4, 3)
    # array 2d
    grid = {
        "x": x,
        "y": np.ones(3)[np.newaxis, :],
        "z": z,
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # list
    grid = {
        "x": x,
        "y": [0, 1, 2, 3, 4],
        "z": z,
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # tuple
    grid = {
        "x": x,
        "y": (0, 1, 2, 3),
        "z": z,
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)


def test_is_grid_xy_invalid_z_key():
    "Check if passing an invalid z key raises an error"
    # correct keys
    x = np.arange(4)
    y = np.ones(3)
    area = [0, 1, 2, 3]
    shape = (4, 3)
    # array 2d
    coordinates = {
        "x": np.arange(4),
        "y": np.ones(3),
        "z": np.array(18.2),
        "ordering": "xy",
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(coordinates)
    # list
    grid = {
        "x": x,
        "y": y,
        "z": [18.2],
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)
    # tuple
    grid = {
        "x": x,
        "y": y,
        "z": (18.2,),
        "area": area,
        "shape": shape
    }
    with pytest.raises(ValueError):
        check.is_grid_xy(grid)


##### grid_wavenumbers


def test_is_grid_wavenumbers_non_dict_input():
    "Check if passing a non-dictionary grid raises an error"
    # array
    wavenumbers = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # array
    wavenumbers = np.zeros((4, 3))
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # tuple
    wavenumbers = (1, 5)
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # list
    wavenumbers = [1, 2, 3.0]
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # float
    wavenumbers = 10.2
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)


def test_is_grid_wavenumbers_invalid_keys():
    "Check if passing a dictionary with invalid keys raises an error"
    # correct keys
    x = np.arange(4)
    y = np.ones(3)
    z = np.zeros((4,3)) + 1.2
    shape = (4, 3)
    spacing = (1.1, 1.3)
    # set dict with invalid x key
    wavenumbers = {
        "x": np.arange(4)[:,np.newaxis],
        "y": y,
        "z": z,
        "shape": shape,
        "spacing": spacing
    }
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # set dict with invalid y key
    wavenumbers = {
        "x": x,
        "y": [0, 1, 2],
        "z": z,
        "shape": shape,
        "spacing": spacing
    }
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # set dict with invalid z key (scalar instead numpy array 2d)
    wavenumbers = {
        "x": x,
        "y": y,
        "z": 3.,
        "shape": shape,
        "spacing": spacing
    }
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # set dict with invalid z key with a negative element
    z_with_negative = np.copy(z)
    z_with_negative[1,1] *= -1 
    wavenumbers = {
        "x": x,
        "y": y,
        "z": z_with_negative,
        "shape": shape,
        "spacing": spacing
    }
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # set dict with invalid shape key
    wavenumbers = {
        "x": x,
        "y": y,
        "z": z,
        "shape": [4,3],
        "spacing": spacing
    }
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
    # set dict with invalid spacing key
    wavenumbers = {
        "x": x,
        "y": y,
        "z": z,
        "shape": shape,
        "spacing": np.array([1.1, 1.3])
    }
    with pytest.raises(ValueError):
        check.is_grid_wavenumbers(wavenumbers)
   


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


##### is_area

def test_area_not_list():
    "Check if passing a non-list raises an error"
    # scalar
    area = 0.
    with pytest.raises(ValueError):
        check.is_area(area=area)
    # tuple
    area = (1, 2, 3, 4)
    with pytest.raises(ValueError):
        check.is_area(area=area)
    # array
    area = np.arange(4)
    with pytest.raises(ValueError):
        check.is_area(area=area)


def test_area_not_4_elements():
    "Check if passing list with number of elements different from 4"
    # 3 elements
    area = [0, 1, 2]
    with pytest.raises(ValueError):
        check.is_area(area=area)
    # 5 elements
    area = [0, 1, 2, 4, 5]
    with pytest.raises(ValueError):
        check.is_area(area=area)


def test_area_inconsistent_elements():
    "Check if passing an inconsistent list raises an error"
    # first element greater than second
    area = [1, 0, 2, 3]
    with pytest.raises(ValueError):
        check.is_area(area=area)
    # first element equal to second
    area = [0, 0., 3, 4]
    with pytest.raises(ValueError):
        check.is_area(area=area)
    # third element greater than fouth
    area = [0, 1, 4, 3]
    with pytest.raises(ValueError):
        check.is_area(area=area)
    # third element equal to fouth
    area = [0, 1, 3, 3]
    with pytest.raises(ValueError):
        check.is_area(area=area)


##### is_shape

def test_shape_not_tuple():
    "Check if passing a non-tuple raises an error"
    # scalar
    shape = 0.
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # list
    shape = [1, 2]
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # array
    shape = np.arange(2)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)


def test_shape_not_2_elements():
    "Check if passing list with number of elements different from 4"
    # 1 element
    shape = (0)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # 5 elements
    shape = (0, 1, 2, 4, 5)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)


def test_shape_inconsistent_elements():
    "Check if passing an inconsistent tuple raises an error"
    # first element negative
    shape = (-4, 5)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # second element negative
    shape = (4, -5)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # first element zero
    shape = (0, 5)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # second element zero
    shape = (4, 0)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # first element positive float
    shape = (4.1, 5)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)
    # second element positive float
    shape = (4, 5.2)
    with pytest.raises(ValueError):
        check.is_shape(shape=shape)


##### is_spacing

def test_spacing_not_tuple():
    "Check if passing a non-tuple raises an error"
    # scalar
    spacing = 0.
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)
    # list
    spacing = [1, 2]
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)
    # array
    spacing = np.arange(2)
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)


def test_spacing_not_2_elements():
    "Check if passing list with number of elements different from 4"
    # 1 element
    spacing = (0)
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)
    # 5 elements
    spacing = (0, 1, 2, 4, 5)
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)


def test_spacing_inconsistent_elements():
    "Check if passing an inconsistent tuple raises an error"
    # first element negative
    spacing = (-4.1, 5)
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)
    # second element negative
    spacing = (4, -5)
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)
    # first element zero
    spacing = (0, 5.)
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)
    # second element zero
    spacing = (4, 0)
    with pytest.raises(ValueError):
        check.is_spacing(spacing=spacing)


###### is ordering
def test_ordering_inconsistent_elements():
    "Check if passing an inconsistent ordering raises an error"
    ordering = "Xy"
    with pytest.raises(ValueError):
        check.is_ordering(ordering=ordering)
    ordering = "xY"
    with pytest.raises(ValueError):
        check.is_ordering(ordering=ordering)
    ordering = "x"
    with pytest.raises(ValueError):
        check.is_ordering(ordering=ordering)
    ordering = "y"
    with pytest.raises(ValueError):
        check.is_ordering(ordering=ordering)
    ordering = "xyz"
    with pytest.raises(ValueError):
        check.is_ordering(ordering=ordering)


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