import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_array_less as aal
from pytest import raises
from ..equivalent_layer import space as eqls


def test_inverse_distance_symmetric_points():
    'verify results obtained for symmetrically positioned sources'

    # single source
    S = np.array([[0],
                  [0],
                  [0]])

    # computation points
    P = np.vstack([[-100, 0, 0, 100, 0, 0, 100*np.sqrt(2)*0.5],
                   [0, -100, 0, 0, 100, 0, 100*np.sqrt(2)*0.5],
                   [0, 0, -100, 0, 0, 100, 0]])

    # multiple sources
    np.random.seed(10)
    np.random.rand(34)
    S = np.vstack([-50 + 100*np.random.rand(123),
                   -50 + 100*np.random.rand(123),
                   np.zeros(123)])

    # computation points
    P_up = np.copy(S)
    P_up[2] -= 64
    P_down = np.copy(S)
    P_down[2] += 64

    V_up = 1./np.sqrt(eqls.sedm(P_up, S))
    V_down = 1./np.sqrt(eqls.sedm(P_down, S))
    aae(V_up, V_down, decimal=15)


def test_kernel_xx_symmetric_points():
    'verify results obtained for symmetrically positioned sources'

    # sources
    S = np.array([[0, 0],
                  [-100, 100],
                  [0, 0]])

    # computation points
    P = np.array([[-140, 140],
                  [0, 0],
                  [0, 0]])

    Vxx, _, _, _, _ = eqls.second_derivatives(P, S)

    aae(Vxx[0,:], Vxx[1,:], decimal=15)



def test_kernel_yy_symmetric_points():
    'verify results obtained for symmetrically positioned sources'

    # sources
    S = np.array([[-100, 100],
                  [0, 0],
                  [0, 0]])

    # computation points
    P = np.array([[0, 0],
                  [-140, 140],
                  [0, 0]])

    _, _, _, Vyy, _ = eqls.second_derivatives(P, S)

    aae(Vyy[0,:], Vyy[1,:], decimal=15)


def test_kernel_zz_symmetric_points():
    'verify results obtained for symmetrically positioned sources'

    # sources
    S = np.array([[0, 0],
                  [0, 0],
                  [100, 200]])

    # computation points
    P = np.array([[0, 140],
                  [-140, 0],
                  [0, 0]])

    Vxx, _, _, Vyy, _ = eqls.second_derivatives(P, S)
    Vzz = -(Vxx+Vyy)

    aae(Vzz[0,:], Vzz[1,:], decimal=15)


def test_second_derivatives_decay():
    'abs values of second derivatives must decay with distance'

    # computation points
    P = np.array([[0, 10,  0,  0, -10],
                  [0,  0, 10,  0, -10],
                  [0,  0,  0,  0,   0]])

    # shallow sources
    S = np.array([[ 0],
                  [ 0],
                  [20]])

    # second derivatives produced by shallow sources
    Vxx1, Vxy1, Vxz1, Vyy1, Vyz1 = eqls.second_derivatives(P, S)

    Vxx1 = np.abs(np.sum(Vxx1, axis=1))
    Vxy1 = np.abs(np.sum(Vxy1, axis=1))
    Vxz1 = np.abs(np.sum(Vxz1, axis=1))
    Vyy1 = np.abs(np.sum(Vyy1, axis=1))
    Vyz1 = np.abs(np.sum(Vyz1, axis=1))

    # deeper sources
    S[2] = 40

    # second derivatives produced by deep sources
    Vxx2, Vxy2, Vxz2, Vyy2, Vyz2 = eqls.second_derivatives(P, S)

    Vxx2 = np.abs(np.sum(Vxx2, axis=1))
    Vxy2 = np.abs(np.sum(Vxy2, axis=1))
    Vxz2 = np.abs(np.sum(Vxz2, axis=1))
    Vyy2 = np.abs(np.sum(Vyy2, axis=1))
    Vyz2 = np.abs(np.sum(Vyz2, axis=1))

    assert np.all((Vxx1 - Vxx2) >= 0)
    assert np.all((Vxy1 - Vxy2) >= 0)
    assert np.all((Vxz1 - Vxz2) >= 0)
    assert np.all((Vyy1 - Vyy2) >= 0)
    assert np.all((Vyz1 - Vyz2) >= 0)


def test_inverse_distance_known_points():
    'verify results obtained for specific points'

    # single source
    S = np.array([[0],
                  [0],
                  [10]])

    # computation points
    P = np.vstack([[-10, -10,   0, 10,  0,  0],
                   [  0, -10,   0,  0, 10,  0],
                   [  0,   0, -10,  0,  0,  0]])

    V_ref = np.array([[1/np.sqrt(200)],
                      [1/np.sqrt(300)],
                      [1/np.sqrt(400)],
                      [1/np.sqrt(200)],
                      [1/np.sqrt(200)],
                      [0.1]])

    V = 1./np.sqrt(eqls.sedm(P,S))
    aae(V, V_ref, decimal=15)


def test_second_derivatives_known_points():
    'verify results obtained for specific points'

    # single source
    S = np.array([[0],
                  [0],
                  [10]])

    # computation points
    P = np.vstack([[-10, -10,  10,  10,  0,  0],
                   [  0, -10,   0, -10, 10,  0],
                   [  0,   0, -10,   0,  0,  0]])

    Vxx_ref = np.array([[3*(-10)*(-10)/(np.sqrt(200)**5) - 1/(np.sqrt(200)**3)],
                        [3*(-10)*(-10)/(np.sqrt(300)**5) - 1/(np.sqrt(300)**3)],
                        [3*( 10)*( 10)/(np.sqrt(500)**5) - 1/(np.sqrt(500)**3)],
                        [3*( 10)*( 10)/(np.sqrt(300)**5) - 1/(np.sqrt(300)**3)],
                        [3*(  0)*(  0)/(np.sqrt(200)**5) - 1/(np.sqrt(200)**3)],
                        [3*(  0)*(  0)/(          10**5) - 1/(          10**3)]])

    Vxy_ref = np.array([[3*(-10)*(  0)/(np.sqrt(200)**5)],
                        [3*(-10)*(-10)/(np.sqrt(300)**5)],
                        [3*( 10)*(  0)/(np.sqrt(500)**5)],
                        [3*( 10)*(-10)/(np.sqrt(300)**5)],
                        [3*(  0)*( 10)/(np.sqrt(200)**5)],
                        [3*(  0)*(  0)/(          10**5)]])

    Vxz_ref = np.array([[3*(-10)*(-10)/(np.sqrt(200)**5)],
                        [3*(-10)*(-10)/(np.sqrt(300)**5)],
                        [3*( 10)*(-20)/(np.sqrt(500)**5)],
                        [3*( 10)*(-10)/(np.sqrt(300)**5)],
                        [3*(  0)*(-10)/(np.sqrt(200)**5)],
                        [3*(  0)*(-10)/(          10**5)]])

    Vyy_ref = np.array([[3*(  0)*(  0)/(np.sqrt(200)**5) - 1/(np.sqrt(200)**3)],
                        [3*(-10)*(-10)/(np.sqrt(300)**5) - 1/(np.sqrt(300)**3)],
                        [3*(  0)*(  0)/(np.sqrt(500)**5) - 1/(np.sqrt(500)**3)],
                        [3*(-10)*(-10)/(np.sqrt(300)**5) - 1/(np.sqrt(300)**3)],
                        [3*( 10)*( 10)/(np.sqrt(200)**5) - 1/(np.sqrt(200)**3)],
                        [3*(  0)*(  0)/(          10**5) - 1/(          10**3)]])

    Vyz_ref = np.array([[3*(  0)*(-10)/(np.sqrt(200)**5)],
                        [3*(-10)*(-10)/(np.sqrt(300)**5)],
                        [3*(  0)*(-20)/(np.sqrt(500)**5)],
                        [3*(-10)*(-10)/(np.sqrt(300)**5)],
                        [3*( 10)*(-10)/(np.sqrt(200)**5)],
                        [3*(  0)*(-10)/(          10**5)]])

    Vxx, Vxy, Vxz, Vyy, Vyz = eqls.second_derivatives(P,S)
    aae(Vxx, Vxx_ref, decimal=15)
    aae(Vxy, Vxy_ref, decimal=15)
    aae(Vxz, Vxz_ref, decimal=15)
    aae(Vyy, Vyy_ref, decimal=15)
    aae(Vyz, Vyz_ref, decimal=15)


def test_first_derivatives_known_points():
    'verify results obtained for specific points'

    # single source
    S = np.array([[0],
                  [0],
                  [10]])

    # computation points
    P = np.vstack([[-10, -10,  10,  10,  0,  0],
                   [  0, -10,   0, -10, 10,  0],
                   [  0,   0, -10,   0,  0,  0]])

    Vx_ref = np.array([[-(-10)/(np.sqrt(200)**3)],
                       [-(-10)/(np.sqrt(300)**3)],
                       [-( 10)/(np.sqrt(500)**3)],
                       [-( 10)/(np.sqrt(300)**3)],
                       [-(  0)/(np.sqrt(200)**3)],
                       [-(  0)/(          10**3)]])

    Vy_ref = np.array([[-(  0)/(np.sqrt(200)**3)],
                       [-(-10)/(np.sqrt(300)**3)],
                       [-(  0)/(np.sqrt(500)**3)],
                       [-(-10)/(np.sqrt(300)**3)],
                       [-( 10)/(np.sqrt(200)**3)],
                       [-(  0)/(          10**3)]])

    Vz_ref = np.array([[-(-10)/(np.sqrt(200)**3)],
                       [-(-10)/(np.sqrt(300)**3)],
                       [-(-20)/(np.sqrt(500)**3)],
                       [-(-10)/(np.sqrt(300)**3)],
                       [-(-10)/(np.sqrt(200)**3)],
                       [-(-10)/(          10**3)]])

    Vx, Vy, Vz = eqls.first_derivatives(P,S)
    aae(Vx, Vx_ref, decimal=15)
    aae(Vy, Vy_ref, decimal=15)
    aae(Vz, Vz_ref, decimal=15)
