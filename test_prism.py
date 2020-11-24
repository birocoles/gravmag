import numpy as np
import numpy.testing as npt
import pytest
import prism


def test_invalid_grav_field():
    "Check if passing an invalid field raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    coordinates = np.array([[0], [0], [0]])
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="invalid field")


def test_invalid_mag_field():
    "Check if passing an invalid field raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    magnetization = np.array([[1], [1], [1]])
    coordinates = np.array([[0], [0], [0]])
    with pytest.raises(ValueError):
        prism.mag(coordinates, model, magnetization, field="invalid field")


def test_invalid_prism_boundaries():
    "Check if passing an invalid prism boundaries raises an error"
    density = np.array([1000])
    magnetization = np.array([[1], [1], [1]])
    coordinates = np.array([[0], [0], [0]])
    # wrong x boundaries
    model = np.array([[100, -100, -100, 100, 100, 200]])
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_x")
        prism.mag(coordinates, model, magnetization, field="b_y")
    # wrong y boundaries
    model = np.array([[-100, 100, 100, -100, 100, 200]])
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_z")
        prism.mag(coordinates, model, magnetization, field="b_z")
    # wrong z boundaries
    model = np.array([[-100, 100, -100, 100, 200, 100]])
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_potential")
        prism.mag(coordinates, model, magnetization, field="b_x")


def test_invalid_prism():
    "Check if passing an invalid prism raises an error"
    density = np.array([1000])
    coordinates = np.array([[0], [0], [0]])
    field = "g_potential"
    # shape (1,)
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_potential")
    # shape (2,4)
    model = np.empty((2, 4))
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_x")
    # shape (1,4)
    model = np.empty((1, 5))
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_z")


def test_invalid_coordinates():
    "Check if passing an invalid coordinates raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    # shape (1,)
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_z")
    # shape (4,3)
    coordinates = np.zeros((4,3))
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_z")


def test_invalid_density():
    "Check if density with shape[0] != 3 raises an error"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    coordinates = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        prism.grav(coordinates, model, density, field="g_z")


def test_safe_atan2():
    "Test the safe_atan2 function"
    # Test safe_atan2 for one point per quadrant
    # First quadrant
    x, y = 1, 1
    npt.assert_allclose(prism.safe_atan2(y, x), np.pi / 4)
    # Second quadrant
    x, y = -1, 1
    npt.assert_allclose(prism.safe_atan2(y, x), -np.pi / 4)
    # Third quadrant
    x, y = -1, -1
    npt.assert_allclose(prism.safe_atan2(y, x), np.pi / 4)
    # Forth quadrant
    x, y = 1, -1
    npt.assert_allclose(prism.safe_atan2(y, x), -np.pi / 4)
    # Test safe_atan2 if the denominator is equal to zero
    npt.assert_allclose(prism.safe_atan2(1, 0), np.pi / 2)
    npt.assert_allclose(prism.safe_atan2(-1, 0), -np.pi / 2)
    # Test safe_atan2 if both numerator and denominator are equal to zero
    npt.assert_allclose(prism.safe_atan2(0, 0), 0)


def test_safe_log():
    "Test the safe_log function"
    # Check if safe_log function satisfies safe_log(0) == 0
    npt.assert_allclose(prism.safe_log(0), 0)
    # Check if safe_log behaves like the natural logarithm in case that x != 0
    x = np.linspace(1, 100, 101)
    for x_i in x:
        npt.assert_allclose(prism.safe_log(x_i), np.log(x_i))


def test_field_decreases_with_distance():
    "Check if field decreases with distance"
    model = np.array([[-100, 100, -100, 100, 100, 200]])
    density = np.array([1000])
    close = np.array([[0], [20], [0]])
    far = np.array([[0], [20], [-100]])
    # potentia
    potential_close = prism.grav(close, model, density, field="g_potential")
    potential_far = prism.grav(far, model, density, field="g_potential")
    # gz
    gz_close = prism.grav(close, model, density, field="g_z")
    gz_far = prism.grav(far, model, density, field="g_z")
    # gx
    gx_close = prism.grav(close, model, density, field="g_x")
    gx_far = prism.grav(far, model, density, field="g_x")
    diffs = np.array([np.abs(potential_far) < np.abs(potential_close),
                      np.abs(gz_far) < np.abs(gz_close),
                      np.abs(gx_far) < np.abs(gx_close)])
    npt.assert_allclose(diffs, np.ones((3,1), dtype=bool))


def test_Laplace_equation():
    "Sum of derivatives xx, yy and zz must be zero outside the prism"
    model = np.array([[-100, 100, -130, 100, 100, 213]])
    coords = np.array([[0, -100, 100, -100, 400],
                       [0, -130, 100, 50, 400],
                       [0, -100, -100, -100, -100]])
    rho = np.array([1300])
    gxx = prism.grav(coordinates=coords, prisms=model, density=rho, field="g_xx")
    gyy = prism.grav(coordinates=coords, prisms=model, density=rho, field="g_yy")
    gzz = prism.grav(coordinates=coords, prisms=model, density=rho, field="g_zz")
    npt.assert_almost_equal(gxx+gyy+gzz, np.zeros(coords.shape[1]), decimal=12)


def test_Poisson_equation():
    "Sum of derivatives xx, yy and zz must -4*pi*rho*G inside the prism"
    model = np.array([[-100, 100, -130, 100, 100, 213]])
    coords = np.array([[0, -10, 80],
                       [0, 30, -62.1],
                       [150, 110, 200]])
    rho = np.array([1300])
    gxx = prism.grav(coordinates=coords, prisms=model, density=rho, field="g_xx")
    gyy = prism.grav(coordinates=coords, prisms=model, density=rho, field="g_yy")
    gzz = prism.grav(coordinates=coords, prisms=model, density=rho, field="g_zz")
    reference_value = -4*np.pi*1300*prism.GRAVITATIONAL_CONST*1e9
    npt.assert_almost_equal(gxx+gyy+gzz,
                            np.zeros(coords.shape[1]) + reference_value,
                            decimal=12)
