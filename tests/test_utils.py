import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
import pytest
from numba import njit
from gravmag import utils


# safe_atan2


def test_safe_atan2_compare_functions():
    "verify consistency between all safe_atan2 functions"
    y = np.array([[0.0, 30.0], [-45.0, 60.0], [-90.0, 180.0]])
    x = np.array([[10.0, 12.0], [-73.0, 3.0], [18.0, 0.0]])
    result_numba = utils.safe_atan2(y=y, x=x)
    result_numpy = utils.safe_atan2_np(y=y, x=x)
    aae(result_numba, result_numpy, decimal=15)
    for yi, xi, result_i in zip(y.ravel(), x.ravel(), result_numpy.ravel()):
        result_numba_entrywise = utils.safe_atan2_entrywise(y=yi, x=xi)
        aae(result_numba_entrywise, result_i, decimal=15)


def test_safe_atan2_entrywise():
    "Test the safe_atan2 function"
    # Test safe_atan2 for one point per quadrant
    x = np.array([1.0, -1.0, -1.0, 1.0])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    reference = np.array([np.pi / 4, -np.pi / 4, np.pi / 4, -np.pi / 4])
    for xi, yi, ri in zip(x, y, reference):
        aae(utils.safe_atan2_entrywise(yi, xi), ri, decimal=15)
    # Test safe_atan2 if the denominator is equal to zero
    x = np.array([0.0, 0.0])
    y = np.array([1.0, -1.0])
    reference = np.array([np.pi / 2, -np.pi / 2])
    for xi, yi, ri in zip(x, y, reference):
        aae(utils.safe_atan2_entrywise(yi, xi), ri, decimal=15)
    # Test safe_atan2 if both numerator and denominator are equal to zero
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])
    reference = np.array([0, 0])
    for xi, yi, ri in zip(x, y, reference):
        aae(utils.safe_atan2_entrywise(yi, xi), ri, decimal=15)


def test_safe_atan2():
    "Test the safe_atan2 function"
    # Test safe_atan2 for one point per quadrant
    x = np.array([[1.0, -1.0], [-1.0, 1.0]])
    y = np.array([[1.0, 1.0], [-1.0, -1.0]])
    reference = np.array([[np.pi / 4, -np.pi / 4], [np.pi / 4, -np.pi / 4]])
    aae(utils.safe_atan2(y, x), reference, decimal=15)
    # Test safe_atan2 if the denominator is equal to zero
    x = np.array([[0.0, 0.0]])
    y = np.array([[1.0, -1.0]])
    reference = np.array([[np.pi / 2, -np.pi / 2]])
    aae(utils.safe_atan2(y, x), reference, decimal=15)
    # Test safe_atan2 if both numerator and denominator are equal to zero
    x = np.array([[0.0, 0.0]])
    y = np.array([[0.0, 0.0]])
    reference = np.array([[0, 0]])
    aae(utils.safe_atan2(y, x), reference, decimal=15)


# safe_log


def test_safe_log_compare_functions():
    "verify consistency between all safe_log functions"
    x = np.array([[0.0, 12.0], [-3.0, 7.3], [1018.0, -1018.0]])
    result_numba = utils.safe_log(x=x)
    result_numpy = utils.safe_log_np(x=x)
    aae(result_numba, result_numpy, decimal=15)
    for xi, result_i in zip(x.ravel(), result_numpy.ravel()):
        result_numba_entrywise = utils.safe_log_entrywise(x=xi)
        aae(result_numba_entrywise, result_i, decimal=15)


def test_safe_log_entrywise():
    "Test the safe_log function"
    # Check if safe_log function satisfies safe_log(0) == 0
    x = np.array([0.0, 0.0])
    reference = np.zeros(2)
    for xi, ri in zip(x, reference):
        aae(utils.safe_log_entrywise(xi), ri, decimal=15)
    # Check if safe_log behaves like the natural logarithm in case that x != 0
    x = np.linspace(1, 100, 100)
    reference = np.log(x)
    for xi, ri in zip(x, reference):
        aae(utils.safe_log_entrywise(xi), ri, decimal=15)


def test_safe_log():
    "Test the safe_log function"
    # Check if safe_log function satisfies safe_log(0) == 0
    x = np.array([[0.0, 0.0]])
    reference = np.zeros((1, 2))
    aae(utils.safe_log(x), reference, decimal=15)
    # Check if safe_log behaves like the natural logarithm in case that x != 0
    x = np.linspace(1, 100, 100).reshape((4, 25))
    aae(utils.safe_log(x), np.log(x), decimal=15)


# magnetization_components


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


# unit_vector


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


# direction


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


# rotation_matrix


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


# coordinate_transform


# prisms_volume


def test_prisms_volume_compare_known_values():
    "verify if computed volumes are equal to reference values"
    model = {
        "x1": np.array([-100, 2000, -34]),
        "x2": np.array([100, 2500, 66]),
        "y1": np.array([230, -1400, 350.0]),
        "y2": np.array([430, 0, 700.0]),
        "z1": np.array([10, -100, 0]),
        "z2": np.array([90, 600.5, 348.0]),
    }
    reference = np.array([200 * 200 * 80, 500 * 1400 * 700.5, 100 * 350 * 348])
    computed = utils.prisms_volume(prisms=model)
    aae(computed, reference, decimal=15)


# block_data


def test_block_data():
    "compare computed blocks with reference values"
    x = np.array(
        [
            310.0,
            290.0,
            403.0,
            500.0,
            -107.5,
            18.9,
            200.0,
            -12.3,
            -99.7,
            598.0,
            -100.0,
            0.5,
            150.0,
            110.0,
            290.0,
            -275.3,
            590.0,
        ]
    )
    y = np.array(
        [
            134.0,
            201.0,
            370.0,
            260.0,
            199.3,
            101.1,
            340.0,
            207.0,
            318.1,
            150.0,
            130.0,
            300.1,
            150.0,
            170.0,
            240.0,
            310.8,
            311.0,
        ]
    )
    reference = [
        [[4, 10], [7], [8, 15]],
        [[5, 12, 13], [1, 14], [6, 11]],
        [[0, 9], [3], [2, 16]],
    ]
    computed = utils.block_data(
        x=x, y=y, area=[-300.0, 600.0, 100.0, 400.0], shape=(3, 3)
    )
    ae(computed, reference)


# reduce_data


def test_reduce_data():
    "compare computed blocks with reference values"
    x = np.array(
        [
            310.0,
            290.0,
            403.0,
            500.0,
            -107.5,
            18.9,
            200.0,
            -12.3,
            -99.7,
            598.0,
            -100.0,
            0.5,
            150.0,
            110.0,
            290.0,
            -275.3,
            590.0,
        ]
    )
    y = np.array(
        [
            134.0,
            201.0,
            370.0,
            260.0,
            199.3,
            101.1,
            340.0,
            207.0,
            318.1,
            150.0,
            130.0,
            300.1,
            150.0,
            170.0,
            240.0,
            310.8,
            311.0,
        ]
    )
    data = np.arange(17) * 10 + 10.0
    blocks = utils.block_data(
        x=x, y=y, area=[-300.0, 600.0, 100.0, 400.0], shape=(3, 3)
    )
    reference = np.array(
        [
            [(50 + 110) / 2, (60 + 130 + 140) / 3, (10 + 100) / 2],
            [80, (20 + 150) / 2, 40],
            [(90 + 160) / 2, (70 + 120) / 2, (30 + 170) / 2],
        ]
    ).T
    computed = utils.reduce_data(
        data=data, blocks_indices=blocks, function="mean", remove_nan=False
    )
    aae(computed, reference, decimal=15)
