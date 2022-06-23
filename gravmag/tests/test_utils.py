import numpy as np
import numpy.testing as npt
import pytest
from .. import utils


def test_unit_vector_magnitude():
    "check if the unit vector has magnitude 1"
    I = [10, -30, 0, 90, -90, 180, 45, 73, -3]
    D = [-28, 47, 5, 18, 0, 90, -90, 7, 89]
    for inc, dec in zip(I, D):
        u = utils.unit_vector(inc, dec)
        npt.assert_allclose(np.sum(u * u), 1, atol=1e-15)


def test_unit_vector_known_values():
    "compare computed unit vector with reference values"
    I = [0, 0, 90, -90, 0]
    D = [0, 90, 0, 0, 45]
    reference_outputs = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
        np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0]),
    ]
    for inc, dec, ref in zip(I, D, reference_outputs):
        u = utils.unit_vector(inc, dec)
        npt.assert_allclose(u, ref, atol=1e-15)


def test_direction_known_values():
    "compare computed direction with reference values"
    reference_I = [0, 0, 90, -90, 0]
    reference_D = [0, 90, 0, 0, 45]
    reference_inputs = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
        np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0]),
    ]
    for ref_inc, ref_dec, ref_input in zip(
        reference_I, reference_D, reference_inputs
    ):
        intens, inc, dec = utils.direction(ref_input)
        npt.assert_allclose(intens, 1)
        npt.assert_allclose(inc, ref_inc)
        npt.assert_allclose(dec, ref_dec)


def test_rotation_matrix_orthonormal():
    "check if the rotation matrix is orthonormal"
    I = [10, -30, 0, 90, -90, 180, 45, 73, -3]
    D = [-28, 47, 5, 18, 0, 90, -90, 7, 89]
    dI = [1, 18, 24, 13, 0, 40, 5, -3, -3]
    dD = [8, 7, -51, 108, 19.4, 0, 6, -7, 389]
    for inc, dec, dinc, ddec in zip(I, D, dI, dD):
        R = utils.rotation_matrix(inc, dec, dinc, ddec)
        npt.assert_allclose(np.dot(R.T, R), np.identity(3), atol=1e-15)
        npt.assert_allclose(np.dot(R, R.T), np.identity(3), atol=1e-15)


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
    magnetization = np.array([[1.0, -30, 45], [10.0, 60, -30]])

    mx_ref = 1.0 * np.array([np.sqrt(6) / 4, 10 * np.sqrt(3) / 4])
    my_ref = np.array([np.sqrt(6) / 4, -10 / 4])
    mz_ref = np.array([-1 / 2, 10 * np.sqrt(3) / 2])
    mx, my, mz = utils.magnetization_components(magnetization)
    # mx, my, mz with ndim != 1
    npt.assert_allclose(mx.ndim, 1)
    npt.assert_allclose(my.ndim, 1)
    npt.assert_allclose(mz.ndim, 1)
    # mx, my, mz are close to reference values
    npt.assert_allclose(mx, mx_ref)
    npt.assert_allclose(my, my_ref)
    npt.assert_allclose(mz, mz_ref)
