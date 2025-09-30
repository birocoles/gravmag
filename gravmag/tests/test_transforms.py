import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from pytest import raises
from .. import transforms as tr


def test_DFT_data_not_matrix():
    "must raise ValueError if data not a matrix"
    # data as a float
    with raises(ValueError):
        tr.DFT(3.0)
    # data as vector
    with raises(ValueError):
        tr.DFT(np.ones(3))
    # data as a string
    with raises(ValueError):
        tr.DFT("not-a-matrix")


def test_DFT_pad_mode_not_None_or_string():
    "must raise ValueError if pad_mode is not None or a string"
    data = np.ones((5, 5))
    # pad_mode as a float
    with raises(ValueError):
        tr.DFT(data, pad_mode=3.0)
    # pad_mode as a vector
    with raises(ValueError):
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
    "must raise ValueError if data not a matrix"
    # Ft_data as a float
    with raises(ValueError):
        tr.IDFT(3.0)
    # FT_data as vector
    with raises(ValueError):
        tr.IDFT(np.ones(3))
    # FT_data as a string
    with raises(ValueError):
        tr.IDFT("not-a-matrix")
    # FT_data as a real matrix
    with raises(ValueError):
        tr.IDFT(np.ones((3, 3)))


def test_IDFT_unpad_not_boolean():
    "must raise ValueError if unpad is not boolean"
    with raises(ValueError):
        FT_data = np.ones((3, 3)) + 1j * np.ones((3, 3))
        tr.IDFT(FT_data=FT_data, unpad="invalid-unpad")


def test_IDFT_return_shape():
    "verify is the shape of returned array is correct"
    FT_data = np.ones((15, 15)) - 1j * np.ones((15, 15))
    # unpad=False
    correct_shape = (15, 15)
    data = tr.IDFT(FT_data, unpad=False)
    ae(data.shape, correct_shape)
    # unpad=True
    correct_shape = (5, 5)
    data = tr.IDFT(FT_data, unpad=True)
    ae(data.shape, correct_shape)
