import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
import pytest
from numba import njit
from .. import utils


def test_unit_vector_magnitude():
    "check if the unit vector has magnitude 1"
    I = [10, -30, 0, 90, -90, 180, 45, 73, -3]
    D = [-28, 47, 5, 18, 0, 90, -90, 7, 89]
    for inc, dec in zip(I, D):
        u = utils.unit_vector(inc, dec)
        aae(np.sum(u * u), 1, decimal=15)


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
        aae(u, ref, decimal=15)


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
        aae(intens, 1, decimal=15)
        aae(inc, ref_inc, decimal=15)
        aae(dec, ref_dec, decimal=15)


def test_rotation_matrix_orthonormal():
    "check if the rotation matrix is orthonormal"
    I = [10, -30, 0, 90, -90, 180, 45, 73, -3]
    D = [-28, 47, 5, 18, 0, 90, -90, 7, 89]
    dI = [1, 18, 24, 13, 0, 40, 5, -3, -3]
    dD = [8, 7, -51, 108, 19.4, 0, 6, -7, 389]
    for inc, dec, dinc, ddec in zip(I, D, dI, dD):
        R = utils.rotation_matrix(inc, dec, dinc, ddec)
        aae(np.dot(R.T, R), np.identity(3), decimal=15)
        aae(np.dot(R, R.T), np.identity(3), decimal=15)


def test_safe_atan2():
    "Test the safe_atan2 function"
    # Test safe_atan2 for one point per quadrant
    x = np.array([[1.,-1.],[-1.,1.]])
    y = np.array([[1.,1.],[-1.,-1.]])
    reference = np.array([[np.pi / 4, -np.pi / 4],[np.pi / 4, -np.pi / 4]])
    aae(utils.safe_atan2(y, x), reference, decimal=15)
    # Test safe_atan2 if the denominator is equal to zero
    x = np.array([[0.,0.]])
    y = np.array([[1.,-1.]])
    reference = np.array([[np.pi / 2, -np.pi / 2]])
    aae(utils.safe_atan2(y, x), reference, decimal=15)
    # Test safe_atan2 if both numerator and denominator are equal to zero
    x = np.array([[0.,0.]])
    y = np.array([[0.,0.]])
    reference = np.array([[0, 0]])
    aae(utils.safe_atan2(y, x), reference, decimal=15)


def test_safe_log():
    "Test the safe_log function"
    # Check if safe_log function satisfies safe_log(0) == 0
    x = np.array([[0., 0.]])
    reference = np.zeros((1,2))
    aae(utils.safe_log(x), reference, decimal=15)
    # Check if safe_log behaves like the natural logarithm in case that x != 0
    x = np.linspace(1, 100, 100).reshape((4,25))
    aae(utils.safe_log(x), np.log(x), decimal=15)


def test_magnetization_components():
    "Compare reference values with those obtained from magnetization_components"
    magnetization = np.array([[1.0, -30, 45], [10.0, 60, -30]])

    mx_ref = 1.0 * np.array([np.sqrt(6) / 4, 10 * np.sqrt(3) / 4])
    my_ref = np.array([np.sqrt(6) / 4, -10 / 4])
    mz_ref = np.array([-1 / 2, 10 * np.sqrt(3) / 2])
    mx, my, mz = utils.magnetization_components(magnetization)
    # mx, my, mz are close to reference values
    aae(mx, mx_ref, decimal=13)
    aae(my, my_ref, decimal=13)
    aae(mz, mz_ref, decimal=13)
