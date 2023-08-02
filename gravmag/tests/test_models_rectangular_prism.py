import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
import pytest
from ..models import rectangular_prism_numba as rp_nb
from ..models import rectangular_prism as rp
from .. import constants as cts


# def test_invalid_grav_field():
#     "Check if passing an invalid field raises an error"
#     model = np.array([[-100, 100, -100, 100, 100, 200]])
#     density = np.array([1000])
#     coordinates = np.array([[0], [0], [0]])
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="invalid field")


# def test_invalid_mag_field():
#     "Check if passing an invalid field raises an error"
#     model = np.array([[-100, 100, -100, 100, 100, 200]])
#     magnetization = np.array([[1], [1], [1]])
#     coordinates = np.array([[0], [0], [0]])
#     with pytest.raises(ValueError):
#         rp.mag(coordinates, model, magnetization, field="invalid field")


# def test_invalid_prism_boundaries():
#     "Check if passing an invalid prism boundaries raises an error"
#     density = np.array([1000])
#     magnetization = np.array([[1], [1], [1]])
#     coordinates = np.array([[0], [0], [0]])
#     # wrong x boundaries
#     model = np.array([[100, -100, -100, 100, 100, 200]])
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="x")
#         rp.mag(coordinates, model, magnetization, field="y")
#     # wrong y boundaries
#     model = np.array([[-100, 100, 100, -100, 100, 200]])
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="z")
#         rp.mag(coordinates, model, magnetization, field="z")
#     # wrong z boundaries
#     model = np.array([[-100, 100, -100, 100, 200, 100]])
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="potential")
#         rp.mag(coordinates, model, magnetization, field="x")


# def test_invalid_prism():
#     "Check if passing an invalid prism raises an error"
#     density = np.array([1000])
#     coordinates = np.array([[0], [0], [0]])
#     field = "potential"
#     # shape (1,)
#     model = np.array([100, -100, -100, 100, 100, 200])
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="potential")
#     # shape (2,4)
#     model = np.empty((2, 4))
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="x")
#     # shape (1,4)
#     model = np.empty((1, 5))
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="z")


# def test_invalid_coordinates():
#     "Check if passing an invalid coordinates raises an error"
#     model = np.array([[-100, 100, -100, 100, 100, 200]])
#     density = np.array([1000])
#     # shape (1,)
#     coordinates = np.array([0, 0, 0])
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="z")
#     # shape (4,3)
#     coordinates = np.zeros((4, 3))
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="z")


# def test_invalid_density():
#     "Check if density with shape[0] != 3 raises an error"
#     model = np.array([[-100, 100, -100, 100, 100, 200]])
#     density = np.array([1000])
#     coordinates = np.array([0, 0, 0])
#     with pytest.raises(ValueError):
#         rp.grav(coordinates, model, density, field="z")


# def test_field_decreases_with_distance():
#     "Check if field decreases with distance"
#     model = np.array([[-100, 100, -100, 100, 100, 200]])
#     density = np.array([1000])
#     close = np.array([[20], [0], [0]])
#     far = np.array([[20], [0], [-100]])
#     # potentia
#     potential_close = rp.grav(close, model, density, field="potential")
#     potential_far = rp.grav(far, model, density, field="potential")
#     # gz
#     gz_close = rp.grav(close, model, density, field="z")
#     gz_far = rp.grav(far, model, density, field="z")
#     # gx
#     gx_close = rp.grav(close, model, density, field="x")
#     gx_far = rp.grav(far, model, density, field="x")
#     diffs = np.array(
#         [
#             np.abs(potential_far) < np.abs(potential_close),
#             np.abs(gz_far) < np.abs(gz_close),
#             np.abs(gx_far) < np.abs(gx_close),
#         ]
#     )
#     npt.assert_allclose(diffs, np.ones((3, 1), dtype=bool))


# def test_Laplace_equation():
#     "Sum of derivatives xx, yy and zz must be zero outside the prism"
#     model = np.array([[-130, 100, -100, 100, 100, 213]])
#     coords = np.array(
#         [
#             [0, -130, 100, 50, 400],
#             [0, -100, 100, -100, 400],
#             [0, -100, -100, -100, -100],
#         ]
#     )
#     rho = np.array([1300])
#     gxx = rp.grav(coordinates=coords, prisms=model, density=rho, field="xx")
#     gyy = rp.grav(coordinates=coords, prisms=model, density=rho, field="yy")
#     gzz = rp.grav(coordinates=coords, prisms=model, density=rho, field="zz")
#     npt.assert_almost_equal(
#         gxx + gyy + gzz, np.zeros(coords.shape[1]), decimal=12
#     )


# def test_Poisson_equation():
#     "Sum of derivatives xx, yy and zz must -4*pi*rho*G inside the prism"
#     model = np.array([[-130, 100, -100, 100, 100, 213]])
#     coords = np.array([[0, 30, -62.1], [0, -10, 80], [150, 110, 200]])
#     rho = np.array([1300])
#     gxx = rp.grav(coordinates=coords, prisms=model, density=rho, field="xx")
#     gyy = rp.grav(coordinates=coords, prisms=model, density=rho, field="yy")
#     gzz = rp.grav(coordinates=coords, prisms=model, density=rho, field="zz")
#     reference_value = (
#         -4 * np.pi * 1300 * cts.GRAVITATIONAL_CONST * cts.SI2EOTVOS
#     )
#     npt.assert_almost_equal(
#         gxx + gyy + gzz, np.zeros(coords.shape[1]) + reference_value, decimal=12
#     )


##### kernels

def test_kernel_potential_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_inverse_r(X2, Y2, Z2)
                - rp_nb.kernel_inverse_r(X2, Y2, Z1)
                - rp_nb.kernel_inverse_r(X1, Y2, Z2)
                + rp_nb.kernel_inverse_r(X1, Y2, Z1)
                - rp_nb.kernel_inverse_r(X2, Y1, Z2)
                + rp_nb.kernel_inverse_r(X2, Y1, Z1)
                + rp_nb.kernel_inverse_r(X1, Y1, Z2)
                - rp_nb.kernel_inverse_r(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_potential)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_x_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dx(X2, Y2, Z2)
                - rp_nb.kernel_dx(X2, Y2, Z1)
                - rp_nb.kernel_dx(X1, Y2, Z2)
                + rp_nb.kernel_dx(X1, Y2, Z1)
                - rp_nb.kernel_dx(X2, Y1, Z2)
                + rp_nb.kernel_dx(X2, Y1, Z1)
                + rp_nb.kernel_dx(X1, Y1, Z2)
                - rp_nb.kernel_dx(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_x)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_y_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dy(X2, Y2, Z2)
                - rp_nb.kernel_dy(X2, Y2, Z1)
                - rp_nb.kernel_dy(X1, Y2, Z2)
                + rp_nb.kernel_dy(X1, Y2, Z1)
                - rp_nb.kernel_dy(X2, Y1, Z2)
                + rp_nb.kernel_dy(X2, Y1, Z1)
                + rp_nb.kernel_dy(X1, Y1, Z2)
                - rp_nb.kernel_dy(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_y)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_z_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dz(X2, Y2, Z2)
                - rp_nb.kernel_dz(X2, Y2, Z1)
                - rp_nb.kernel_dz(X1, Y2, Z2)
                + rp_nb.kernel_dz(X1, Y2, Z1)
                - rp_nb.kernel_dz(X2, Y1, Z2)
                + rp_nb.kernel_dz(X2, Y1, Z1)
                + rp_nb.kernel_dz(X1, Y1, Z2)
                - rp_nb.kernel_dz(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_z)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_xx_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dxx(X2, Y2, Z2)
                - rp_nb.kernel_dxx(X2, Y2, Z1)
                - rp_nb.kernel_dxx(X1, Y2, Z2)
                + rp_nb.kernel_dxx(X1, Y2, Z1)
                - rp_nb.kernel_dxx(X2, Y1, Z2)
                + rp_nb.kernel_dxx(X2, Y1, Z1)
                + rp_nb.kernel_dxx(X1, Y1, Z2)
                - rp_nb.kernel_dxx(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_xx)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_xy_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dxy(X2, Y2, Z2)
                - rp_nb.kernel_dxy(X2, Y2, Z1)
                - rp_nb.kernel_dxy(X1, Y2, Z2)
                + rp_nb.kernel_dxy(X1, Y2, Z1)
                - rp_nb.kernel_dxy(X2, Y1, Z2)
                + rp_nb.kernel_dxy(X2, Y1, Z1)
                + rp_nb.kernel_dxy(X1, Y1, Z2)
                - rp_nb.kernel_dxy(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_xy)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_xz_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dxz(X2, Y2, Z2)
                - rp_nb.kernel_dxz(X2, Y2, Z1)
                - rp_nb.kernel_dxz(X1, Y2, Z2)
                + rp_nb.kernel_dxz(X1, Y2, Z1)
                - rp_nb.kernel_dxz(X2, Y1, Z2)
                + rp_nb.kernel_dxz(X2, Y1, Z1)
                + rp_nb.kernel_dxz(X1, Y1, Z2)
                - rp_nb.kernel_dxz(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_xz)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_yy_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dyy(X2, Y2, Z2)
                - rp_nb.kernel_dyy(X2, Y2, Z1)
                - rp_nb.kernel_dyy(X1, Y2, Z2)
                + rp_nb.kernel_dyy(X1, Y2, Z1)
                - rp_nb.kernel_dyy(X2, Y1, Z2)
                + rp_nb.kernel_dyy(X2, Y1, Z1)
                + rp_nb.kernel_dyy(X1, Y1, Z2)
                - rp_nb.kernel_dyy(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_yy)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_yz_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dyz(X2, Y2, Z2)
                - rp_nb.kernel_dyz(X2, Y2, Z1)
                - rp_nb.kernel_dyz(X1, Y2, Z2)
                + rp_nb.kernel_dyz(X1, Y2, Z1)
                - rp_nb.kernel_dyz(X2, Y1, Z2)
                + rp_nb.kernel_dyz(X2, Y1, Z1)
                + rp_nb.kernel_dyz(X1, Y1, Z2)
                - rp_nb.kernel_dyz(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_yz)
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_zz_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        'x1' : np.array([-130]),
        'x2' : np.array([ 100]),
        'y1' : np.array([-100]),
        'y2' : np.array([ 100]),
        'z1' : np.array([ 100]),
        'z2' : np.array([ 213])
    }
    coords = {
        'x' : np.array([0, 30, -62.1]),
        'y' : np.array([0, -10, 80]),
        'z' : np.array([-1, 0, -2])
    }
    rho = np.array([1300])
    D = 3 # number of computation points
    P = 1 # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype='float')
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model['x1'][p] - coords['x'][d]
            X2 = model['x2'][p] - coords['x'][d]
            Y1 = model['y1'][p] - coords['y'][d]
            Y2 = model['y2'][p] - coords['y'][d]
            Z1 = model['z1'][p] - coords['z'][d]
            Z2 = model['z2'][p] - coords['z'][d]
            # Compute the field
            result_numba[d] += rho[p] * (
                  rp_nb.kernel_dzz(X2, Y2, Z2)
                - rp_nb.kernel_dzz(X2, Y2, Z1)
                - rp_nb.kernel_dzz(X1, Y2, Z2)
                + rp_nb.kernel_dzz(X1, Y2, Z1)
                - rp_nb.kernel_dzz(X2, Y1, Z2)
                + rp_nb.kernel_dzz(X2, Y1, Z1)
                + rp_nb.kernel_dzz(X1, Y1, Z2)
                - rp_nb.kernel_dzz(X1, Y1, Z1)
            )
    # compute with numpy
    result_numpy = rp.iterate_over_vertices(coords, model, rho, rp.kernel_zz)
    aae(result_numba, result_numpy, decimal=8)