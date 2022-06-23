import numpy as np
import numpy.testing as npt
import pytest
from ..models import rectangular_prism as rp
from .. import constants as cts


def test_invalid_grav_field():
    "Check if passing an invalid field raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    coordinates = np.array([[0], [0], [0]])
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="invalid field")


def test_invalid_mag_field():
    "Check if passing an invalid field raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    magnetization = np.array([[1], [1], [1]])
    coordinates = np.array([[0], [0], [0]])
    with pytest.raises(ValueError):
        rp.mag(coordinates, model, magnetization, field="invalid field")


def test_invalid_prism_boundaries():
    "Check if passing an invalid prism boundaries raises an error"
    density = np.array([1000])
    magnetization = np.array([[1], [1], [1]])
    coordinates = np.array([[0], [0], [0]])
    # wrong x boundaries
    model = np.array([[100, -100, -100, 100, 100, 200]])
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_x")
        rp.mag(coordinates, model, magnetization, field="b_y")
    # wrong y boundaries
    model = np.array([[-100, 100, 100, -100, 100, 200]])
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_z")
        rp.mag(coordinates, model, magnetization, field="b_z")
    # wrong z boundaries
    model = np.array([[-100, 100, -100, 100, 200, 100]])
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_potential")
        rp.mag(coordinates, model, magnetization, field="b_x")


def test_invalid_prism():
    "Check if passing an invalid prism raises an error"
    density = np.array([1000])
    coordinates = np.array([[0], [0], [0]])
    field = "g_potential"
    # shape (1,)
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_potential")
    # shape (2,4)
    model = np.empty((2, 4))
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_x")
    # shape (1,4)
    model = np.empty((1, 5))
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_z")


def test_invalid_coordinates():
    "Check if passing an invalid coordinates raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    # shape (1,)
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_z")
    # shape (4,3)
    coordinates = np.zeros((4, 3))
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_z")


def test_invalid_density():
    "Check if density with shape[0] != 3 raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        rp.grav(coordinates, model, density, field="g_z")


def test_field_decreases_with_distance():
    "Check if field decreases with distance"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    close = np.array([[20], [0], [0]])
    far = np.array([[20], [0], [-100]])
    # potentia
    potential_close = rp.grav(close, model, density, field="g_potential")
    potential_far = rp.grav(far, model, density, field="g_potential")
    # gz
    gz_close = rp.grav(close, model, density, field="g_z")
    gz_far = rp.grav(far, model, density, field="g_z")
    # gx
    gx_close = rp.grav(close, model, density, field="g_x")
    gx_far = rp.grav(far, model, density, field="g_x")
    diffs = np.array(
        [
            np.abs(potential_far) < np.abs(potential_close),
            np.abs(gz_far) < np.abs(gz_close),
            np.abs(gx_far) < np.abs(gx_close),
        ]
    )
    npt.assert_allclose(diffs, np.ones((3, 1), dtype=bool))


def test_Laplace_equation():
    "Sum of derivatives xx, yy and zz must be zero outside the prism"
    model = np.array([[-130, 100, -100, 100, 100, 213]])
    coords = np.array(
        [
            [0, -130, 100, 50, 400],
            [0, -100, 100, -100, 400],
            [0, -100, -100, -100, -100],
        ]
    )
    rho = np.array([1300])
    gxx = rp.grav(coordinates=coords, prisms=model, density=rho, field="g_xx")
    gyy = rp.grav(coordinates=coords, prisms=model, density=rho, field="g_yy")
    gzz = rp.grav(coordinates=coords, prisms=model, density=rho, field="g_zz")
    npt.assert_almost_equal(
        gxx + gyy + gzz, np.zeros(coords.shape[1]), decimal=12
    )


def test_Poisson_equation():
    "Sum of derivatives xx, yy and zz must -4*pi*rho*G inside the prism"
    model = np.array([[-130, 100, -100, 100, 100, 213]])
    coords = np.array([[0, 30, -62.1], [0, -10, 80], [150, 110, 200]])
    rho = np.array([1300])
    gxx = rp.grav(coordinates=coords, prisms=model, density=rho, field="g_xx")
    gyy = rp.grav(coordinates=coords, prisms=model, density=rho, field="g_yy")
    gzz = rp.grav(coordinates=coords, prisms=model, density=rho, field="g_zz")
    reference_value = (
        -4 * np.pi * 1300 * cts.GRAVITATIONAL_CONST * cts.SI2EOTVOS
    )
    npt.assert_almost_equal(
        gxx + gyy + gzz, np.zeros(coords.shape[1]) + reference_value, decimal=12
    )
