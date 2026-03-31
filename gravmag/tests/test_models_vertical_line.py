import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
import pytest
from ..models import vertical_line as vl
from .. import constants as cts

def test_vertical_line_invalid_grav_field():
    "Check if passing an invalid field raises an error"
    model = {
        "x": np.array([-130]),
        "y": np.array([100]),
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
        vl.grav(coords, model, rho, field="invalid field")


def test_vertical_line_grav_invalid_boundaries():
    "Check if passing an invalid line boundaries raises an error"
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    # wrong z boundaries
    model = {
        "x": np.array([3.]),
        "y": np.array([-1]),
        "z1": np.array([100]),
        "z2": np.array([99]),
    }
    with pytest.raises(ValueError):
        vl.grav(coords, model, rho, field="z")


def test_vertical_line_grav_invalid_line():
    "Check if passing a non-dictionaty line raises an error"
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    rho = np.array([1300])
    field = "z"
    # array
    model = np.array([100, -100, -100, 100, 100, 200])
    with pytest.raises(ValueError):
        vl.grav(coords, model, rho, field=field)
    # array
    model = np.vstack([coords['x'], coords['y'], coords['z']+1, coords['z']+2])
    with pytest.raises(ValueError):
        vl.grav(coords, model, rho, field=field)
    # list
    model = [2, 4]
    with pytest.raises(ValueError):
        vl.grav(coords, model, rho, field=field)
    # tuple
    model = (1, 5)
    with pytest.raises(ValueError):
        vl.grav(coords, model, rho, field=field)


def test_vertical_line_invalid_coordinates():
    "Check if passing an invalid coordinates raises an error"
    model = {
        "x": np.array([-130]),
        "y": np.array([100]),
        "z1": np.array([100]),
        "z2": np.array([213]),
    }
    rho = np.array([1300])
    # array
    coords = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        vl.grav(coords, model, rho, field="z")
    # tuple
    coords = (4, 3)
    with pytest.raises(ValueError):
        vl.grav(coords, model, rho, field="z")


def test_vertical_line_grav_incompatible_density_lines():
    "Check if passing incompatible density and lines raises an error"
    model = {
        "x": np.array([-30]),
        "y": np.array([10]),
        "z1": np.array([10]),
        "z2": np.array([15.1]),
    }
    coords = {
        "x": np.array([0, 30, -62.1]),
        "y": np.array([0, -10, 80]),
        "z": np.array([-1, 0, -2]),
    }
    # array
    density = np.ones(2)
    with pytest.raises(ValueError):
        vl.grav(coords, model, density, field="z")
    # list
    density = [1000.4,]
    with pytest.raises(ValueError):
        vl.grav(coords, model, density, field="z")
    # tuple
    density = (1000.4,)
    with pytest.raises(ValueError):
        vl.grav(coords, model, density, field="z")


def test_grav_field_decreases_with_distance():
    "Check if grav field decreases with distance"
    model = {
        "x": np.array([-30]),
        "y": np.array([10]),
        "z1": np.array([10]),
        "z2": np.array([305.1]),
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
    # gz
    gz_close = vl.grav(close, model, density, field="z")
    gz_far = vl.grav(far, model, density, field="z")
    diffs = [
        np.abs(gz_far) < np.abs(gz_close),
    ]
    ae(diffs, [True,])


def test_vertical_line_symmetric_points():
    "Check if computed values satisfies symmetries"
    model = {
        "x": np.array([0]),
        "y": np.array([0]),
        "z1": np.array([20]),
        "z2": np.array([305.1]),
    }
    coords = {
        "x": np.array([-10.0, 10.0, 0.0, 0.0]),
        "y": np.array([0.0, 0.0, -10.0, 10.0]),
        "z": np.array([0.0, 0.0, 0.0, 0.0]),
    }
    rho = np.array([1000])
    computed = vl.grav(coordinates=coords, lines=model, density=rho, field="z")
    # symmetries along x and y
    aae(computed[0], computed[1], decimal=6)
    aae(computed[2], computed[3], decimal=6)


def test_vertical_line_reference():
    "Check if computed values are consistent with reference values"
    model = {
        "x": np.array([0]),
        "y": np.array([0]),
        "z1": np.array([20]),
        "z2": np.array([30]),
    }
    coords = {
        "x": np.array([0.0, -10.0, 13.0, 0.0, 0.0]),
        "y": np.array([0.0, 0.0, 0.0, -8.0, 11.0]),
        "z": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    rho = 1000
    reference = cts.GRAVITATIONAL_CONST * cts.SI2miliGAL * rho * np.array([
        np.log(30 + 30) - np.log(20 + 20),
        np.log(30 + np.sqrt(10 ** 2 + 30 ** 2)) - np.log(20 + np.sqrt(10 ** 2 + 20 ** 2)),
        np.log(30 + np.sqrt(13 ** 2 + 30 ** 2)) - np.log(20 + np.sqrt(13 ** 2 + 20 ** 2)),
        np.log(30 + np.sqrt(8 ** 2 + 30 ** 2)) - np.log(20 + np.sqrt(8 ** 2 + 20 ** 2)),
        np.log(30 + np.sqrt(11 ** 2 + 30 ** 2)) - np.log(20 + np.sqrt(11 ** 2 + 20 ** 2)),
    ])
    computed = vl.grav(coordinates=coords, lines=model, density=rho, field="z")
    aae(computed, reference, decimal=10)