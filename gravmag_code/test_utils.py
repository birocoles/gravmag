import numpy as np
import numpy.testing as npt
import pytest
import utils


def test_safe_atan2():
    "Test the safe_atan2 function"
    # Test safe_atan2 for one point per quadrant
    # First quadrant
    x, y = 1, 1
    npt.assert_allclose(utils.safe_atan2(y, x), np.pi / 4)
    # Second quadrant
    x, y = -1, 1
    npt.assert_allclose(utils.safe_atan2(y, x), -np.pi / 4)
    # Third quadrant
    x, y = -1, -1
    npt.assert_allclose(utils.safe_atan2(y, x), np.pi / 4)
    # Forth quadrant
    x, y = 1, -1
    npt.assert_allclose(utils.safe_atan2(y, x), -np.pi / 4)
    # Test safe_atan2 if the denominator is equal to zero
    npt.assert_allclose(utils.safe_atan2(1, 0), np.pi / 2)
    npt.assert_allclose(utils.safe_atan2(-1, 0), -np.pi / 2)
    # Test safe_atan2 if both numerator and denominator are equal to zero
    npt.assert_allclose(utils.safe_atan2(0, 0), 0)


def test_safe_log():
    "Test the safe_log function"
    # Check if safe_log function satisfies safe_log(0) == 0
    npt.assert_allclose(utils.safe_log(0), 0)
    # Check if safe_log behaves like the natural logarithm in case that x != 0
    x = np.linspace(1, 100, 101)
    for x_i in x:
        npt.assert_allclose(utils.safe_log(x_i), np.log(x_i))


def test_magnetization_components():
    "Compare reference values with those obtained from magnetization_components"
    magnetization = np.array([[ 1., -30,  45],
                              [10.,  60, -30]])

    mx_ref = 1.*np.array([np.sqrt(6)/4, 10*np.sqrt(3)/4])
    my_ref = np.array([np.sqrt(6)/4, -10/4])
    mz_ref = np.array([-1/2, 10*np.sqrt(3)/2])
    mx, my, mz = utils.magnetization_components(magnetization)
    # mx, my, mz with ndim != 1
    npt.assert_allclose(mx.ndim, 1)
    npt.assert_allclose(my.ndim, 1)
    npt.assert_allclose(mz.ndim, 1)
    # mx, my, mz are close to reference values
    npt.assert_allclose(mx, mx_ref)
    npt.assert_allclose(my, my_ref)
    npt.assert_allclose(mz, mz_ref)
