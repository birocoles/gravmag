import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from pytest import raises
from .. import filters as ft


def test_direction_kxkykz_not_matrices():
    "must raise AssertionError if wavenumbers are not matrices"
    kx = np.ones((4, 4))
    kx[0, :] = 0.0
    ky = np.ones((4, 4))
    ky[:, 0] = 0.0
    kz = np.ones((4, 4))
    inc = 10.0
    dec = -3.0
    # kx not a matrix
    with raises(AssertionError):
        ft.direction(np.ones(4), ky, kz, inc, dec, check_input=True)
    # ky not a matrix
    with raises(AssertionError):
        ft.direction(kx, (3, 4.0), kz, inc, dec, check_input=True)
    # kz not a matrix
    with raises(AssertionError):
        ft.direction(kx, ky, 4, inc, dec, check_input=True)


def test_direction_kxkykz_different_shapes():
    "must raise AssertionError if wavenumbers have different shapes"
    kx = np.ones((4, 4))
    kx[0, :] = 0.0
    ky = np.ones((4, 4))
    ky[:, 0] = 0.0
    kz = np.ones((4, 4))
    inc = 10.0
    dec = -3.0
    # kx with wrong shape
    with raises(AssertionError):
        ft.direction(np.ones((4, 5)), ky, kz, inc, dec, check_input=True)
    # ky with wrong shape
    with raises(AssertionError):
        ft.direction(kx, np.ones((3, 4)), kz, inc, dec, check_input=True)
    # kz with wrong shape
    with raises(AssertionError):
        ft.direction(kx, ky, np.ones((1, 1)), inc, dec, check_input=True)


def test_direction_kx_with_nonnull_first_row():
    "must raise AssertionError if kx has nonnull values in first line"
    kx = np.ones((4, 4))
    ky = np.ones((4, 4))
    ky[:, 0] = 0.0
    kz = np.ones((4, 4))
    inc = 10.0
    dec = -3.0
    with raises(AssertionError):
        ft.direction(kx, ky, kz, inc, dec, check_input=True)


def test_direction_ky_with_nonnull_first_column():
    "must raise AssertionError if ky has nonnull values in first column"
    kx = np.ones((4, 4))
    kx[0, :] = 0.0
    ky = np.ones((4, 4))
    kz = np.ones((4, 4))
    inc = 10.0
    dec = -3.0
    with raises(AssertionError):
        ft.direction(kx, ky, kz, inc, dec, check_input=True)


def test_direction_kz_with_negative_values():
    "must raise AssertionError if kz has negative values"
    kx = np.ones((4, 4))
    kx[0, :] = 0.0
    ky = np.ones((4, 4))
    ky[:, 0] = 0.0
    kz = -np.ones((4, 4))
    inc = 10.0
    dec = -3.0
    with raises(AssertionError):
        ft.direction(kx, ky, kz, inc, dec, check_input=True)


def test_direction_inc_dec_not_scalars():
    "must raise AssertionError if inc/dec are not scalars"
    kx = np.ones((4, 4))
    kx[0, :] = 0.0
    ky = np.ones((4, 4))
    ky[:, 0] = 0.0
    kz = np.ones((4, 4))
    inc = 10.0
    dec = -3.0
    # inc not a scalar
    with raises(AssertionError):
        ft.direction(kx, ky, kz, np.ones(4), dec)
    # dec not a scalar
    with raises(AssertionError):
        ft.direction(kx, ky, kz, inc, (3, 4.0))


def test_rtp_inc0_dec0_inc_dec_not_scalars():
    "must raise AssertionError if inc0/dec0/inc/dec are not scalars"
    kx = np.ones((4, 4))
    kx[0, :] = 0.0
    ky = np.ones((4, 4))
    ky[:, 0] = 0.0
    kz = np.ones((4, 4))
    inc0 = 10
    dec0 = 9
    inc = 5
    dec = 0
    # inc0 not scalar
    with raises(AssertionError):
        ft.rtp(kx, ky, kz, (2, 1), dec0, inc, dec, check_input=True)
    # dec0 not scalar
    with raises(AssertionError):
        ft.rtp(kx, ky, kz, inc0, "not-scalar", inc, dec, check_input=True)
    # inc not scalar
    with raises(AssertionError):
        ft.rtp(kx, ky, kz, inc0, dec0, np.ones(3), dec, check_input=True)
    # dec not scalar
    with raises(AssertionError):
        ft.rtp(kx, ky, kz, inc0, dec0, inc, np.zeros((4, 3)), check_input=True)


def test_derivative_invalid_axes():
    "must raise AssertionError if axes is empty or has invalid elements"
    kx = np.ones((4, 4))
    kx[0, :] = 0.0
    ky = np.ones((4, 4))
    ky[:, 0] = 0.0
    kz = np.ones((4, 4))
    # axes without elements
    with raises(AssertionError):
        ft.derivative(kx, ky, kz, [], check_input=True)
    # axes with elements different from 'x', 'y' or 'z'
    with raises(AssertionError):
        ft.derivative(kx, ky, kz, ["xx", "y", "z"], check_input=True)
    with raises(AssertionError):
        ft.derivative(kx, ky, kz, ["x", "Y", "z"], check_input=True)
    with raises(AssertionError):
        ft.derivative(kx, ky, kz, ["x", "y", "xz"], check_input=True)
