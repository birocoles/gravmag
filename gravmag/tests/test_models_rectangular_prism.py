import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
import pytest
from ..models import rectangular_prism_numba as rp_nb
from ..models import rectangular_prism as rp
from .. import constants as cts


def test_rectangular_prism_invalid_grav_field():
    "Check if passing an invalid field raises an error"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="invalid field")

def test_rectangular_prism_invalid_mag_field():
    "Check if passing an invalid field raises an error"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    mx = np.array([1.])
    my = np.array([1.])
    mz = np.array([1.])
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="invalid field")


def test_rectangular_prism_grav_invalid_prism_boundaries():
    "Check if passing an invalid prism boundaries raises an error"
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    # wrong x boundaries
    model = {
        "x1": np.array([130]),
        "x2": np.array([-100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="x")
    # wrong y boundaries
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([100]),
        "y2": np.array([-100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="z")
    # wrong z boundaries
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([-213]),
    }
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="potential")

def test_rectangular_prism_mag_invalid_prism_boundaries():
    "Check if passing an invalid prism boundaries raises an error"
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    mx = np.array([1.])
    my = np.array([1.])
    mz = np.array([1.])
    # wrong x boundaries
    model = {
        "x1": np.array([130]),
        "x2": np.array([-100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="x")
    # wrong y boundaries
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([100]),
        "y2": np.array([-100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="z")
    # wrong z boundaries
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([-213]),
    }
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="potential")


def test_rectangular_prism_grav_invalid_prism():
    "Check if passing a non-dictionaty prism raises an error"
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    field = "potential"
    # array
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="potential")
    # list
    model = [2, 4]
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="x")
    # array shape (1,4)
    model = (1, 5)
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="z")


def test_rectangular_prism_mag_invalid_prism():
    "Check if passing a non-dictionaty prism raises an error"
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    mx = np.array([1.])
    my = np.array([1.])
    mz = np.array([1.])
    field = "potential"
    # array
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="potential")
    # list
    model = [2, 4]
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="x")
    # array shape (1,4)
    model = (1, 5)
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="z")


def test_rectangular_prism_invalid_coordinates():
    "Check if passing an invalid coordinates raises an error"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    rho = np.array([1300])
    mx = np.array([1.])
    my = np.array([1.])
    mz = np.array([1.])
    # array
    coords = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="z")
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="z")
    # tuple
    coords = (4, 3)
    with pytest.raises(ValueError):
        rp.grav(coords, model, rho, field="z")
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="z")


def test_rectangular_prism_grav_incompatible_density_prisms():
    "Check if passing incompatible density and prisms raises an error"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    density = np.ones(2)
    with pytest.raises(ValueError):
        rp.grav(coords, model, density, field="z")


def test_rectangular_prism_mag_incompatible_magnetization_prisms():
    "Check if passing incompatible magnetization and prisms raises an error"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    mx = np.ones(2)
    my = np.array([1.])
    mz = np.array([1.])
    with pytest.raises(ValueError):
        rp.mag(coords, model, mx, my, mz, field="potential")


def test_grav_field_decreases_with_distance():
    "Check if grav field decreases with distance"
    model = {
        "x1": np.array([-100]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([200]),
    }
    density = np.array([1000])
    close = {
        "x": np.array([20]),
        "y": np.array([0]),
        "z": np.array([0]),
    }
    far = {
        "x": np.array([20]),
        "y": np.array([0]),
        "z": np.array([-100]),
    }
    # potentia
    potential_close = rp.grav(close, model, density, field="potential")
    potential_far = rp.grav(far, model, density, field="potential")
    # gz
    gz_close = rp.grav(close, model, density, field="z")
    gz_far = rp.grav(far, model, density, field="z")
    # gx
    gx_close = rp.grav(close, model, density, field="x")
    gx_far = rp.grav(far, model, density, field="x")
    diffs = [
        np.abs(potential_far) < np.abs(potential_close),
        np.abs(gz_far) < np.abs(gz_close),
        np.abs(gx_far) < np.abs(gx_close),
    ]
    ae(diffs, [True, True, True])


def test_mag_field_decreases_with_distance():
    "Check if mag field decreases with distance"
    model = {
        "x1": np.array([-100]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([200]),
    }
    mx = np.array([8.])
    my = np.array([8.])
    mz = np.array([8.])
    close = {
        "x": np.array([20]),
        "y": np.array([0]),
        "z": np.array([0]),
    }
    far = {
        "x": np.array([20]),
        "y": np.array([0]),
        "z": np.array([-100]),
    }
    # potential
    potential_close = rp.mag(close, model, mx, my, mz, field="potential")
    potential_far = rp.mag(far, model, mx, my, mz, field="potential")
    # bz
    bz_close = rp.mag(close, model, mx, my, mz, field="z")
    bz_far = rp.mag(far, model, mx, my, mz, field="z")
    # bx
    bx_close = rp.mag(close, model, mx, my, mz, field="x")
    bx_far = rp.mag(far, model, mx, my, mz, field="x")
    diffs = [
        np.abs(potential_far) < np.abs(potential_close),
        np.abs(bz_far) < np.abs(bz_close),
        np.abs(bx_far) < np.abs(bx_close),
    ]
    ae(diffs, [True, True, True])


def test_Laplace_equation():
    "Sum of derivatives xx, yy and zz must be zero outside the prism"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, -130, 100, 50, 400]),
        "y": np.array([0, -100, 100, -100, 400]),
        "z": np.array([0, -100, -100, -100, -100]),
    }
    rho = np.array([1300])
    gxx = rp.grav(coordinates=coords, prisms=model, density=rho, field="xx")
    gyy = rp.grav(coordinates=coords, prisms=model, density=rho, field="yy")
    gzz = rp.grav(coordinates=coords, prisms=model, density=rho, field="zz")
    aae(gxx + gyy + gzz, np.zeros(5), decimal=12)


def test_Poisson_equation():
    "Sum of derivatives xx, yy and zz must be equal to -4*pi*rho*G inside the prism"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([150, 110, 200]),
    }
    rho = np.array([1300])
    gxx = rp.grav(coordinates=coords, prisms=model, density=rho, field="xx")
    gyy = rp.grav(coordinates=coords, prisms=model, density=rho, field="yy")
    gzz = rp.grav(coordinates=coords, prisms=model, density=rho, field="zz")
    reference_value = (
        -4 * np.pi * 1300 * cts.GRAVITATIONAL_CONST * cts.SI2EOTVOS
    )
    aae(gxx + gyy + gzz, np.zeros(3) + reference_value, decimal=12)


def test_rectangular_prism_symmetric_points():
    "Check if computed values are consisten with literature"
    model = {
        "x1": np.array([-10.0]),
        "x2": np.array([10.0]),
        "y1": np.array([-10.0]),
        "y2": np.array([10.0]),
        "z1": np.array([-10.0]),
        "z2": np.array([10.0]),
    }
    coords = {
        "x": np.array([0.0, 0.0, -10.0, 10.0, 0.0, 0.0]),
        "y": np.array([0.0, 0.0, 0.0, 0.0, -10.0, 10.0]),
        "z": np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0]),
    }
    rho = np.array([1000])
    computed = rp.grav(coordinates=coords, prisms=model, density=rho, field="z")
    # symmmetry along z
    aae(computed[0], -computed[1], decimal=6)
    # symmmetries along x and y
    aae(computed[2], computed[3], decimal=6)
    aae(computed[4], computed[5], decimal=6)


def test_rectangular_prism_compare_reference_values():
    "Check if computed values are consistent with literature"
    model = {
        "x1": np.array([-10.0]),
        "x2": np.array([10.0]),
        "y1": np.array([-10.0]),
        "y2": np.array([10.0]),
        "z1": np.array([-10.0]),
        "z2": np.array([10.0]),
    }
    coords = {
        "x": np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 100.0]),
        "y": np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0]),
        "z": np.array([10.0, -10.0, 0.0, 10.0, 100.0, 1000.0, 10.0]),
    }
    rho = np.array([1000])
    # reference values presented by Li and Chouteau (1998), Three-dimensional gravity modeling in all space
    reference = np.array(
        [-0.346426, 0.346426, 0.0, -0.129316, -0.005335, -0.000053, -0.000518]
    )
    computed = rp.grav(coordinates=coords, prisms=model, density=rho, field="z")
    aae(reference, computed, decimal=3)


##### kernels: comparison Numba x Numpy


def test_kernel_potential_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
    result_numpy = rp.iterate_over_vertices(
        coords, model, rho, rp.kernel_potential_grav
    )
    aae(result_numba, result_numpy, decimal=8)


def test_kernel_x_numbaXnumpy():
    "Verify if results obtained with numba and numpy are equal to each other"
    model = {
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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
        "x1": np.array([-130]),
        "x2": np.array([100]),
        "y1": np.array([-100]),
        "y2": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    D = 3  # number of computation points
    P = 1  # number of prisms
    # compute with numba
    result_numba = np.zeros(D, dtype="float")
    for d in range(D):
        # Iterate over prisms
        for p in range(P):
            # Change coordinates
            X1 = model["x1"][p] - coords["x"][d]
            X2 = model["x2"][p] - coords["x"][d]
            Y1 = model["y1"][p] - coords["y"][d]
            Y2 = model["y2"][p] - coords["y"][d]
            Z1 = model["z1"][p] - coords["z"][d]
            Z2 = model["z2"][p] - coords["z"][d]
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