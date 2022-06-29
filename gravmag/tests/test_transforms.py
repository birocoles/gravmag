import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from pytest import raises
from .. import transforms as tr


def test_wavenumbers_bad_shape():
    "must raise AssertionError if shape not a tuple of positive integers"
    dx = 1.0
    dy = 1.0
    # shape not tuple
    with raises(AssertionError):
        tr.wavenumbers(np.ones(4), dx, dy)
    # shape tuple with more than 2 elements
    with raises(AssertionError):
        tr.wavenumbers((3, 4, 5), dx, dy)
    # shape with a float
    with raises(AssertionError):
        tr.wavenumbers((3.0, 4), dx, dy)
    # shape with negative integer
    with raises(AssertionError):
        tr.wavenumbers((-3, 4), dx, dy)


def test_wavenumbers_bad_dxdy():
    "must raise AssertionError if dx/dy are not positive scalars"
    shape = (3, 4)
    dx = 1.0
    dy = 1.0
    # dx not scalar
    with raises(AssertionError):
        tr.wavenumbers(shape, np.ones(3), dy)
    # dy not scalar
    with raises(AssertionError):
        tr.wavenumbers(shape, dx, "not-scalar")
    # dx negative
    with raises(AssertionError):
        tr.wavenumbers(shape, -1.0, dy)
    # dy negative
    with raises(AssertionError):
        tr.wavenumbers(shape, dx, -1.0)


def test_DFT_data_not_matrix():
    "must raise AssertionError if data not a matrix"
    # data as a float
    with raises(AssertionError):
        tr.DFT(3.0)
    # data as vector
    with raises(AssertionError):
        tr.DFT(np.ones(3))
    # data as a string
    with raises(AssertionError):
        tr.DFT("not-a-matrix")


def test_DFT_pad_mode_not_None_or_string():
    "must raise AssertionError if pad_mode is not None or a string"
    data = np.ones((5, 5))
    # pad_mode as a float
    with raises(AssertionError):
        tr.DFT(data, pad_mode=3.0)
    # pad_mode as a vector
    with raises(AssertionError):
        tr.DFT(data, pad_mode=np.ones(3))


def test_DFT_return_shape():
    "verify is the shape of returned array is correct"
    data = np.ones((5, 5))
    # pad_mode is None
    correct_shape = data.shape
    FT_data = tr.DFT(data, pad_mode=None)
    ae(FT_data.shape, correct_shape)
    # pad_mode is not None
    correct_shape = (15, 15)
    FT_data = tr.DFT(data, pad_mode="constant")
    ae(FT_data.shape, correct_shape)


def test_IDFT_FT_data_not_complex_matrix():
    "must raise AssertionError if data not a matrix"
    # Ft_data as a float
    with raises(AssertionError):
        tr.IDFT(3.0)
    # FT_data as vector
    with raises(AssertionError):
        tr.IDFT(np.ones(3))
    # FT_data as a string
    with raises(AssertionError):
        tr.IDFT("not-a-matrix")
    # FT_data as a real matrix
    with raises(AssertionError):
        tr.IDFT(np.ones((3, 3)))


def test_IDFT_unpad_not_boolean():
    "must raise AssertionError if unpad is not boolean"
    with raises(AssertionError):
        FT_data = np.ones((3, 3)) + 1j * np.ones((3, 3))
        tr.IDFT(FT_data=FT_data, unpad="invalid-unpad")


def test_IDFT_grid_not_boolean():
    "must raise AssertionError if grid is not boolean"
    with raises(AssertionError):
        FT_data = np.ones((3, 3)) + 1j * np.ones((3, 3))
        unpad = True
        tr.IDFT(FT_data=FT_data, unpad=unpad, grid="invalid-grid")


def test_IDFT_return_shape():
    "verify is the shape of returned array is correct"
    FT_data = np.ones((15, 15)) - 1j * np.ones((15, 15))
    # unpad=False and grid=True
    correct_shape = (15, 15)
    data = tr.IDFT(FT_data, unpad=False, grid=True)
    ae(data.shape, correct_shape)
    # unpad=True and grid=True
    correct_shape = (5, 5)
    data = tr.IDFT(FT_data, unpad=True, grid=True)
    ae(data.shape, correct_shape)
    # unpad=False and grid=False
    correct_shape = (15 * 15,)
    data = tr.IDFT(FT_data, unpad=False, grid=False)
    ae(data.shape, correct_shape)
    # unpad=True and grid=False
    correct_shape = (5 * 5,)
    data = tr.IDFT(FT_data, unpad=True, grid=False)
    ae(data.shape, correct_shape)
