import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import raises
from .. import inverse_distance as id


def test_sedm_symmetric_points():
    "verify results obtained for symmetrically positioned sources"

    # single source
    S = S = np.array([0, 0, 0]).reshape((3, 1))

    # computation points
    P = np.vstack(
        [
            [-100, 0, 0, 100, 0, 0, 100 * np.sqrt(2) * 0.5],
            [0, -100, 0, 0, 100, 0, 100 * np.sqrt(2) * 0.5],
            [0, 0, -100, 0, 0, 100, 0],
        ]
    )

    # multiple sources
    np.random.seed(10)
    np.random.rand(34)
    S = np.vstack(
        [
            -50 + 100 * np.random.rand(123),
            -50 + 100 * np.random.rand(123),
            np.zeros(123),
        ]
    )

    # computation points
    P_up = np.copy(S)
    P_up[2] -= 64
    P_down = np.copy(S)
    P_down[2] += 64

    V_up = 1.0 / np.sqrt(id.sedm(P_up, S))
    V_down = 1.0 / np.sqrt(id.sedm(P_down, S))
    aae(V_up, V_down, decimal=15)


def test_sedm_known_points():
    "verify results obtained for specific points"

    # single source
    S = np.array([0, 0, 10]).reshape((3, 1))

    # computation points
    P = np.vstack(
        [[-10, -10, 0, 10, 0, 0], [0, -10, 0, 0, 10, 0], [0, 0, -10, 0, 0, 0]]
    )

    V_ref = np.array(
        [
            [1 / np.sqrt(200)],
            [1 / np.sqrt(300)],
            [1 / np.sqrt(400)],
            [1 / np.sqrt(200)],
            [1 / np.sqrt(200)],
            [0.1],
        ]
    )

    V = 1.0 / np.sqrt(id.sedm(P, S))
    aae(V, V_ref, decimal=15)


def test_grad_invalid_component():
    "must raise ValueError for invalid components"
    # single source
    S = np.array([0, 0, 0]).reshape((3, 1))
    # singe data point
    P = np.array([0, 0, -10]).reshape((3, 1))
    R2 = id.sedm(P, S)
    # components with more than 3 elements
    components = ["x", "y", "z", "z"]
    with raises(ValueError):
        id.grad(P, S, R2, components)
    # invalid component
    components = ["x", "h"]
    with raises(ValueError):
        id.grad(P, S, R2, components)


def test_grad_invalid_SEDM():
    "must raise ValueError for invalid SEDM"
    # single source
    S = np.array([0, 0, 0]).reshape((3, 1))
    # singe data point
    P = np.array([0, 0, -10]).reshape((3, 1))
    components = ["x", "y", "z"]
    # SEDM with shape different from (1,1)
    with raises(ValueError):
        id.grad(P, S, np.ones((2, 2)), components)


def test_grad_known_points():
    "verify results obtained for specific points"

    # single source
    S = np.array([0, 0, 10]).reshape((3, 1))

    # computation points
    P = np.vstack(
        [
            [-10, -10, 10, 10, 0, 0],
            [0, -10, 0, -10, 10, 0],
            [0, 0, -10, 0, 0, 0],
        ]
    )

    Vx_ref = np.array(
        [
            [-(-10) / (np.sqrt(200) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(10) / (np.sqrt(500) ** 3)],
            [-(10) / (np.sqrt(300) ** 3)],
            [-(0) / (np.sqrt(200) ** 3)],
            [-(0) / (10 ** 3)],
        ]
    )

    Vy_ref = np.array(
        [
            [-(0) / (np.sqrt(200) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(0) / (np.sqrt(500) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(10) / (np.sqrt(200) ** 3)],
            [-(0) / (10 ** 3)],
        ]
    )

    Vz_ref = np.array(
        [
            [-(-10) / (np.sqrt(200) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(-20) / (np.sqrt(500) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(-10) / (np.sqrt(200) ** 3)],
            [-(-10) / (10 ** 3)],
        ]
    )

    R2 = id.sedm(P, S)

    # all components
    Vx, Vy, Vz = id.grad(P, S, R2)
    aae(Vx, Vx_ref, decimal=15)
    aae(Vy, Vy_ref, decimal=15)
    aae(Vz, Vz_ref, decimal=15)

    # x and y components
    Vx, Vy = id.grad(P, S, R2, ["x", "y"])
    aae(Vx, Vx_ref, decimal=15)
    aae(Vy, Vy_ref, decimal=15)

    # x and z components
    Vx, Vz = id.grad(P, S, R2, ["x", "z"])
    aae(Vx, Vx_ref, decimal=15)
    aae(Vz, Vz_ref, decimal=15)

    # z and y components
    Vz, Vy = id.grad(P, S, R2, ["z", "y"])
    aae(Vz, Vz_ref, decimal=15)
    aae(Vy, Vy_ref, decimal=15)


def test_grad_tensor_invalid_component():
    "must raise ValueError for invalid components"
    # single source
    S = np.array([0, 0, 0]).reshape((3, 1))
    # singe data point
    P = np.array([0, 0, -10]).reshape((3, 1))
    R2 = id.sedm(P, S)
    # components with more than 6 elements
    components = ["xx", "xy", "xz", "yy", "yz", "zz", "zz"]
    with raises(ValueError):
        id.grad_tensor(P, S, R2, components)
    # invalid component
    components = ["xx", "xh"]
    with raises(ValueError):
        id.grad_tensor(P, S, R2, components)


def test_grad_tensor_invalid_SEDM():
    "must raise ValueError for invalid SEDM"
    # single source
    S = np.array([0, 0, 0]).reshape((3, 1))
    # singe data point
    P = np.array([0, 0, -10]).reshape((3, 1))
    # SEDM with shape different from (1,1)
    with raises(ValueError):
        id.grad_tensor(P, S, np.ones((2, 2)))


def test_grad_tensor_xx_symmetric_points():
    "verify results obtained for symmetrically positioned sources"

    # sources
    S = np.array([[0, 0], [-100, 100], [0, 0]])

    # computation points
    P = np.array([[-140, 140], [0, 0], [0, 0]])

    R2 = id.sedm(P, S)
    Vxx = id.grad_tensor(P, S, R2, ["xx"])

    aae(Vxx[0][0, :], Vxx[0][1, :], decimal=15)


def test_grad_tensor_yy_symmetric_points():
    "verify results obtained for symmetrically positioned sources"

    # sources
    S = np.array([[-100, 100], [0, 0], [0, 0]])

    # computation points
    P = np.array([[0, 0], [-140, 140], [0, 0]])

    R2 = id.sedm(P, S)
    Vyy = id.grad_tensor(P, S, R2, ["yy"])

    aae(Vyy[0][0, :], Vyy[0][1, :], decimal=15)


def test_grad_tensor_zz_symmetric_points():
    "verify results obtained for symmetrically positioned sources"

    # sources
    S = np.array([[0, 0], [0, 0], [100, 200]])

    # computation points
    P = np.array([[0, 140], [-140, 0], [0, 0]])

    R2 = id.sedm(P, S)
    Vzz = id.grad_tensor(P, S, R2, ["zz"])

    aae(Vzz[0][0, :], Vzz[0][1, :], decimal=15)


def test_grad_tensor_Laplace():
    "abs values of second derivatives must decay with distance"

    # single source
    S = np.array([0, 0, 10]).reshape((3, 1))

    # computation points
    P = np.vstack(
        [[-10, -10, 0, 10, 0, 0], [0, -10, 0, 0, 10, 0], [0, 0, -10, 0, 0, 0]]
    )

    # second derivatives produced by shallow sources
    R2 = id.sedm(P, S)
    Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = id.grad_tensor(P, S, R2)

    aae(Vzz, -Vxx - Vyy, decimal=15)


def test_grad_tensor_known_points():
    "verify results obtained for specific points"

    # single source
    S = np.array([[0], [0], [10]])

    # computation points
    P = np.vstack(
        [
            [-10, -10, 10, 10, 0, 0],
            [0, -10, 0, -10, 10, 0],
            [0, 0, -10, 0, 0, 0],
        ]
    )

    Vxx_ref = np.array(
        [
            [3 * (-10) * (-10) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (10) * (10) / (np.sqrt(500) ** 5) - 1 / (np.sqrt(500) ** 3)],
            [3 * (10) * (10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (0) * (0) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (0) * (0) / (10 ** 5) - 1 / (10 ** 3)],
        ]
    )

    Vxy_ref = np.array(
        [
            [3 * (-10) * (0) / (np.sqrt(200) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (10) * (0) / (np.sqrt(500) ** 5)],
            [3 * (10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (0) * (10) / (np.sqrt(200) ** 5)],
            [3 * (0) * (0) / (10 ** 5)],
        ]
    )

    Vxz_ref = np.array(
        [
            [3 * (-10) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (10) * (-20) / (np.sqrt(500) ** 5)],
            [3 * (10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (0) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (0) * (-10) / (10 ** 5)],
        ]
    )

    Vyy_ref = np.array(
        [
            [3 * (0) * (0) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (0) * (0) / (np.sqrt(500) ** 5) - 1 / (np.sqrt(500) ** 3)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (10) * (10) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (0) * (0) / (10 ** 5) - 1 / (10 ** 3)],
        ]
    )

    Vyz_ref = np.array(
        [
            [3 * (0) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (0) * (-20) / (np.sqrt(500) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (10) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (0) * (-10) / (10 ** 5)],
        ]
    )

    R2 = id.sedm(P, S)
    Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = id.grad_tensor(P, S, R2)
    aae(Vxx, Vxx_ref, decimal=15)
    aae(Vxy, Vxy_ref, decimal=15)
    aae(Vxz, Vxz_ref, decimal=15)
    aae(Vyy, Vyy_ref, decimal=15)
    aae(Vyz, Vyz_ref, decimal=15)
