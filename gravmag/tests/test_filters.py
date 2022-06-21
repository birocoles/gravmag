import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from pytest import raises
from .. import filters as ft


def test_wavenumbers_bad_dx_dy():
    'must raise AssertionError if dx or dy are not positive scalars'
    shape = (3,5)
    # dx not scalar
    dx = np.ones(3)
    dy = 1.
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)
    # dy not scalar
    dy = np.ones(3)
    dx = 1.
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)
    # dx negative scalar
    dx = -2.
    dy = 1.
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)
    # dy negative scalar
    dy = -2.
    dx = 1.
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)


def test_wavenumbers_shape_without_2_elements():
    'must raise AssertionError if shape does not have 2 elements'
    dx = 1.
    dy = 1.
    # shape with one element
    shape = (4,)
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)
    # shape with 3 elements
    shape = (4, 5, 2)
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)


def test_wavenumbers_shape_not_positive_integers():
    'must raise AssertionError if shape does not contain positive integers'
    dx = 1.
    dy = 1.
    # shape with floats
    shape = (4.1, 5.)
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)
    # shape with negative integers
    shape = (4, -5)
    with raises(AssertionError):
        ft.wavenumbers(shape, dx, dy)


def test_wavenumbers_known_values():
    'compare result with reference values'
    shape = (8,7)
    dx = 1.
    dy = 1.
    kx_ref = np.resize(
                        np.array([0., 1., 2., 3., -4., -3., -2., -1.])/8,
                        (7,8)
                      ).T
    ky_ref = np.resize(
                        np.array([0., 1., 2., 3., -3., -2., -1.])/7,
                        (8,7)
                       )
    kz_ref = np.sqrt(kx_ref**2 + ky_ref**2)
    kx, ky, kz = ft.wavenumbers(shape, dx, dy)
    ae(kx, kx_ref)
    ae(ky, ky_ref)
    ae(kz, kz_ref)
