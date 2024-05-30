import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from pytest import raises
from .. import filters as ft


def test_direction_inc_dec_not_scalars():
    "must raise ValueError if inc/dec are not scalars"
    # correct input
    x = np.arange(4)
    y = np.ones(3)
    z = np.zeros((4, 3)) + 1.2
    ordering = "xy"
    shape = (4, 3)
    spacing = (1.1, 1.3)
    wavenumbers = {
        "x": x,
        "y": y,
        "z": z,
        "ordering": ordering,
        "shape": shape,
        "spacing": spacing,
    }
    inc = 10.0
    dec = -3.0
    # inc not a scalar
    with raises(ValueError):
        ft.direction(wavenumbers, np.ones(4), dec)
    # dec not a scalar
    with raises(ValueError):
        ft.direction(wavenumbers, inc, (3, 4.0))


def test_rtp_inc0_dec0_inc_dec_not_scalars():
    "must raise ValueError if inc0/dec0/inc/dec are not scalars"
    # correct input
    x = np.arange(4)
    y = np.ones(3)
    z = np.zeros((4, 3)) + 1.2
    ordering = "xy"
    shape = (4, 3)
    spacing = (1.1, 1.3)
    wavenumbers = {
        "x": x,
        "y": y,
        "z": z,
        "ordering": ordering,
        "shape": shape,
        "spacing": spacing,
    }
    inc0 = 10
    dec0 = 9
    inc = 5
    dec = 0
    # inc0 not scalar
    with raises(ValueError):
        ft.rtp(wavenumbers, (2, 1), dec0, inc, dec, check_input=True)
    # dec0 not scalar
    with raises(ValueError):
        ft.rtp(wavenumbers, inc0, "not-scalar", inc, dec, check_input=True)
    # inc not scalar
    with raises(ValueError):
        ft.rtp(wavenumbers, inc0, dec0, np.ones(3), dec, check_input=True)
    # dec not scalar
    with raises(ValueError):
        ft.rtp(wavenumbers, inc0, dec0, inc, np.zeros((4, 3)), check_input=True)


def test_derivative_invalid_axes():
    "must raise ValueError if axes is empty or has invalid elements"
    x = np.arange(4)
    y = np.ones(3)
    z = np.zeros((4, 3)) + 1.2
    ordering = "xy"
    shape = (4, 3)
    spacing = (1.1, 1.3)
    wavenumbers = {
        "x": x,
        "y": y,
        "z": z,
        "ordering": ordering,
        "shape": shape,
        "spacing": spacing,
    }
    # axes without elements
    with raises(ValueError):
        ft.derivative(wavenumbers, [], check_input=True)
    # axes with elements different from 'x', 'y' or 'z'
    with raises(ValueError):
        ft.derivative(wavenumbers, ["xx", "y", "z"], check_input=True)
    with raises(ValueError):
        ft.derivative(wavenumbers, ["x", "Y", "z"], check_input=True)
    with raises(ValueError):
        ft.derivative(wavenumbers, ["x", "y", "xz"], check_input=True)
