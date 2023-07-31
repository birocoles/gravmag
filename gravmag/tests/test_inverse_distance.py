import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from pytest import raises
from .. import inverse_distance as idist
from .. import convolve as conv

##### SEDM


def test_sedm_known_points():
    "verify results obtained for specific points"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([10.0])}

    # computation points
    P = {
        "x": np.array([-10, -10, 0, 10, 0, 0]),
        "y": np.array([0, -10, 0, 0, 10, 0]),
        "z": np.array([0, 0, -10, 0, 0, 0]),
    }

    SEDM_reference = np.array(
        [
            [200.0],
            [300.0],
            [400.0],
            [200.0],
            [200.0],
            [100.0],
        ]
    )

    SEDM_computed = idist.sedm(P, S)
    aae(SEDM_computed, SEDM_reference, decimal=10)


def test_sedm_symmetric_points():
    "verify results obtained for symmetrically positioned sources"

    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}

    # computation points
    P = {
        "x": np.array([-10, 0, 0, 10, 0, 0, 10]),
        "y": np.array([0, -10, 0, 0, 10, 0, 10]),
        "z": np.array([0, 0, -10, 0, 0, 10, 0]),
    }
    SEDM_computed = idist.sedm(P, S)
    SEDM_reference = np.zeros((7, 1)) + 100.0
    SEDM_reference[-1, 0] += 100.0
    aae(SEDM_computed, SEDM_reference, decimal=10)

    # multiple sources
    np.random.seed(10)
    np.random.rand(34)
    S = {
        "x": -50 + 100 * np.random.rand(123),
        "y": -50 + 100 * np.random.rand(123),
        "z": np.zeros(123),
    }

    # computation points
    P_up = S.copy()
    P_up["z"] -= 64
    P_down = S.copy()
    P_down["z"] += 64

    SEDM_up = idist.sedm(P_up, S)
    SEDM_down = idist.sedm(P_down, S)
    aae(SEDM_up, SEDM_down, decimal=15)


##### SEDM BTTB


def test_sedm_BTTB_compare_sedm_xy():
    "verify if sedm_BTTB produces the same result as sedm for a xy grid"
    # cordinates of the grid
    x = np.linspace(1.3, 5.7, 5)
    y = np.linspace(100.0, 104.3, 4)
    Dz = 15.8
    # define points with 'ordering'='xy'
    xp, yp = np.meshgrid(x, y, indexing="xy")
    zp = np.zeros_like(xp) + 30.0
    data_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel()}
    source_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel() + Dz}
    grid = {"x": x[:, np.newaxis], "y": y, "z": 30.0, "ordering": "xy"}
    # compute the SEDM's
    SEDM = idist.sedm(data_points=data_points, source_points=source_points)
    SEDM_BTTB_1st_col = idist.sedm_BTTB(data_grid=grid, delta_z=Dz)
    SEDM_BTTB = conv.general_BTTB(
        num_blocks=y.size,
        columns_blocks=np.reshape(
            a=SEDM_BTTB_1st_col, newshape=(y.size, x.size)
        ),
        rows_blocks=None,
    )
    aae(SEDM, SEDM_BTTB, decimal=10)


def test_sedm_BTTB_compare_sedm_yx():
    "verify if sedm_BTTB produces the same result as sedm for a yx grid"
    # cordinates of the grid
    x = np.linspace(1.3, 5.7, 5)
    y = np.linspace(100.0, 104.3, 4)
    Dz = 15.8
    # define points with 'ordering'='yx'
    xp, yp = np.meshgrid(x, y, indexing="ij")
    zp = np.zeros_like(xp) + 30.0
    data_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel()}
    source_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel() + Dz}
    grid = {"x": x[:, np.newaxis], "y": y, "z": 30.0, "ordering": "yx"}
    # compute the SEDM's
    SEDM = idist.sedm(data_points=data_points, source_points=source_points)
    SEDM_BTTB_1st_col = idist.sedm_BTTB(data_grid=grid, delta_z=Dz)
    SEDM_BTTB = conv.general_BTTB(
        num_blocks=x.size,
        columns_blocks=np.reshape(
            a=SEDM_BTTB_1st_col, newshape=(x.size, y.size)
        ),
        rows_blocks=None,
    )
    aae(SEDM, SEDM_BTTB, decimal=10)


#### grad


def test_grad_single_versus_joint_computation():
    "verify if components computed separately are the same as that computed simultaneously"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    R2 = idist.sedm(P, S)
    # compute separated components
    X0 = idist.grad(P, S, R2, ["x"])[0]
    Y0 = idist.grad(P, S, R2, ["y"])[0]
    Z0 = idist.grad(P, S, R2, ["z"])[0]

    # compute x, y and z components
    X, Y, Z = idist.grad(P, S, R2, ["x", "y", "z"])
    ae(X0, X)
    ae(Y0, Y)
    ae(Z0, Z)
    # compute x and z components
    X, Z = idist.grad(P, S, R2, ["x", "z"])
    ae(X0, X)
    ae(Z0, Z)
    # compute y and z components
    Y, Z = idist.grad(P, S, R2, ["y", "z"])
    ae(Y0, Y)
    ae(Z0, Z)


def test_grad_repeated_components():
    "verify if repeated are equal to each other"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    R2 = idist.sedm(P, S)

    # repeat x component
    computed = idist.grad(P, S, R2, ["x", "x"])
    ae(computed[0], computed[1])
    # repeat y component
    computed = idist.grad(P, S, R2, ["y", "y"])
    ae(computed[0], computed[1])
    # repeat z component
    computed = idist.grad(P, S, R2, ["z", "z"])
    ae(computed[0], computed[1])


def test_grad_invalid_component():
    "must raise ValueError for invalid components"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    R2 = idist.sedm(P, S)
    # float
    components = ["x", 13, "z"]
    with raises(ValueError):
        idist.grad(P, S, R2, components)
    # invalid string
    components = ["x", "h"]
    with raises(ValueError):
        idist.grad(P, S, R2, components)


def test_grad_invalid_SEDM():
    "must raise ValueError for invalid SEDM"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    components = ["x", "y", "z"]
    # SEDM with shape different from (1,1)
    with raises(ValueError):
        idist.grad(P, S, np.ones((2, 2)), components)


def test_grad_known_points():
    "verify results obtained for specific points"

    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([10.0])}

    # computation points
    P = {
        "x": np.array([-10, -10, 10, 10, 0, 0]),
        "y": np.array([0, -10, 0, -10, 10, 0]),
        "z": np.array([0, 0, -10, 0, 0, 0]),
    }

    Vx_ref = np.array(
        [
            [-(-10) / (np.sqrt(200) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(10) / (np.sqrt(500) ** 3)],
            [-(10) / (np.sqrt(300) ** 3)],
            [-(0) / (np.sqrt(200) ** 3)],
            [-(0) / (10**3)],
        ]
    )

    Vy_ref = np.array(
        [
            [-(0) / (np.sqrt(200) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(0) / (np.sqrt(500) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(10) / (np.sqrt(200) ** 3)],
            [-(0) / (10**3)],
        ]
    )

    Vz_ref = np.array(
        [
            [-(-10) / (np.sqrt(200) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(-20) / (np.sqrt(500) ** 3)],
            [-(-10) / (np.sqrt(300) ** 3)],
            [-(-10) / (np.sqrt(200) ** 3)],
            [-(-10) / (10**3)],
        ]
    )

    R2 = idist.sedm(P, S)

    # all components
    Vx, Vy, Vz = idist.grad(P, S, R2)
    aae(Vx, Vx_ref, decimal=15)
    aae(Vy, Vy_ref, decimal=15)
    aae(Vz, Vz_ref, decimal=15)

    # x and y components
    Vx, Vy = idist.grad(P, S, R2, ["x", "y"])
    aae(Vx, Vx_ref, decimal=15)
    aae(Vy, Vy_ref, decimal=15)

    # x and z components
    Vx, Vz = idist.grad(P, S, R2, ["x", "z"])
    aae(Vx, Vx_ref, decimal=15)
    aae(Vz, Vz_ref, decimal=15)

    # z and y components
    Vz, Vy = idist.grad(P, S, R2, ["z", "y"])
    aae(Vz, Vz_ref, decimal=15)
    aae(Vy, Vy_ref, decimal=15)


##### grad BTTB


def test_grad_BTTB_compare_grad_xy():
    "verify if grad_BTTB produces the same result as grad for a xy grid"
    # cordinates of the grid
    x = np.linspace(1.3, 5.7, 5) * 1e3
    y = np.linspace(100.0, 104.3, 4) * 1e3
    Dz = 15.8 * 1e3
    # define points with 'ordering'='xy'
    xp, yp = np.meshgrid(x, y, indexing="xy")
    zp = np.zeros_like(xp) + 30.0
    data_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel()}
    source_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel() + Dz}
    grid = {"x": x[:, np.newaxis], "y": y, "z": 30.0, "ordering": "xy"}
    # compute the SEDM's
    SEDM = idist.sedm(data_points=data_points, source_points=source_points)
    SEDM_BTTB = idist.sedm_BTTB(data_grid=grid, delta_z=Dz)
    # compute grad's
    GRAD = idist.grad(
        data_points=data_points,
        source_points=source_points,
        SEDM=SEDM,
        components=["x", "y", "z"],
    )
    GRAD_BTTB_1st_col = idist.grad_BTTB(
        data_grid=grid, delta_z=Dz, SEDM=SEDM_BTTB, components=["x", "y", "z"]
    )
    for element, element_BTTB_1st_col in zip(GRAD, GRAD_BTTB_1st_col):
        element_BTTB = conv.general_BTTB(
            num_blocks=y.size,
            columns_blocks=np.reshape(
                a=element_BTTB_1st_col, newshape=(y.size, x.size)
            ),
            rows_blocks=None,
        )
        aae(element, element_BTTB, decimal=8)


def test_grad_BTTB_compare_grad_yx():
    "verify if grad_BTTB produces the same result as grad for a yx grid"
    # cordinates of the grid
    x = np.linspace(1.3, 5.7, 5) * 1e3
    y = np.linspace(100.0, 104.3, 4) * 1e3
    Dz = 15.8 * 1e3
    # define points with 'ordering'='yx'
    xp, yp = np.meshgrid(x, y, indexing="ij")
    zp = np.zeros_like(xp) + 30.0
    data_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel()}
    source_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel() + Dz}
    grid = {"x": x[:, np.newaxis], "y": y, "z": 30.0, "ordering": "yx"}
    # compute the SEDM's
    SEDM = idist.sedm(data_points=data_points, source_points=source_points)
    SEDM_BTTB = idist.sedm_BTTB(data_grid=grid, delta_z=Dz)
    # compute grad's
    GRAD = idist.grad(
        data_points=data_points,
        source_points=source_points,
        SEDM=SEDM,
        components=["x", "y", "z"],
    )
    GRAD_BTTB_1st_col = idist.grad_BTTB(
        data_grid=grid, delta_z=Dz, SEDM=SEDM_BTTB, components=["x", "y", "z"]
    )
    for element, element_BTTB_1st_col in zip(GRAD, GRAD_BTTB_1st_col):
        element_BTTB = conv.general_BTTB(
            num_blocks=x.size,
            columns_blocks=np.reshape(
                a=element_BTTB_1st_col, newshape=(x.size, y.size)
            ),
            rows_blocks=None,
        )
        aae(element, element_BTTB, decimal=8)


#### grad tensor


def test_grad_tensor_single_versus_joint_computation():
    "verify if components computed separately are the same as that computed simultaneously"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    R2 = idist.sedm(P, S)
    # compute separated components
    XX0 = idist.grad_tensor(P, S, R2, ["xx"])[0]
    XY0 = idist.grad_tensor(P, S, R2, ["xy"])[0]
    XZ0 = idist.grad_tensor(P, S, R2, ["xz"])[0]
    YY0 = idist.grad_tensor(P, S, R2, ["yy"])[0]
    YZ0 = idist.grad_tensor(P, S, R2, ["yz"])[0]
    ZZ0 = idist.grad_tensor(P, S, R2, ["zz"])[0]

    # compute x, y and z components
    XX, XY, XZ, YY, YZ, ZZ = idist.grad_tensor(
        P, S, R2, ["xx", "xy", "xz", "yy", "yz", "zz"]
    )
    ae(XX0, XX)
    ae(XY0, XY)
    ae(XZ0, XZ)
    ae(YY0, YY)
    ae(YZ0, YZ)
    ae(ZZ0, ZZ)
    # compute xy and yz components
    XY, YZ = idist.grad_tensor(P, S, R2, ["xy", "yz"])
    ae(XY0, XY)
    ae(YZ0, YZ)
    # compute yy and xz components
    YY, XZ = idist.grad_tensor(P, S, R2, ["yy", "xz"])
    ae(YY0, YY)
    ae(XZ0, XZ)


def test_grad_tensor_repeated_components():
    "verify if repeated are equal to each other"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    R2 = idist.sedm(P, S)

    # repeat xx component
    computed = idist.grad_tensor(P, S, R2, ["xx", "xx"])
    ae(computed[0], computed[1])
    # repeat yz component
    computed = idist.grad_tensor(P, S, R2, ["yz", "yz"])
    ae(computed[0], computed[1])
    # repeat xy component
    computed = idist.grad_tensor(P, S, R2, ["xy", "xy"])
    ae(computed[0], computed[1])
    # repeat zz component
    computed = idist.grad_tensor(P, S, R2, ["zz", "zz"])
    ae(computed[0], computed[1])


def test_grad_tensor_invalid_component():
    "must raise ValueError for invalid components"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    R2 = idist.sedm(P, S)
    # int
    components = ["xx", "xy", "xz", "yy", 45, "zz"]
    with raises(ValueError):
        idist.grad_tensor(P, S, R2, components)
    # invalid string
    components = ["xx", "xh"]
    with raises(ValueError):
        idist.grad_tensor(P, S, R2, components)


def test_grad_tensor_invalid_SEDM():
    "must raise ValueError for invalid SEDM"
    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    # singe data point
    P = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([-10.0])}
    # SEDM with shape different from (1,1)
    with raises(ValueError):
        idist.grad_tensor(P, S, np.ones((2, 2)))


def test_grad_tensor_xx_symmetric_points():
    "verify results obtained for symmetrically positioned sources"
    # sources
    S = {
        "x": np.array([0, 0]),
        "y": np.array([-100, 100]),
        "z": np.array([0, 0]),
    }

    # computation points
    P = {
        "x": np.array([-140, 140]),
        "y": np.array([0, 0]),
        "z": np.array([0, 0]),
    }
    R2 = idist.sedm(P, S)
    Vxx = idist.grad_tensor(P, S, R2, ["xx"])

    aae(Vxx[0][0, :], Vxx[0][1, :], decimal=15)


def test_grad_tensor_yy_symmetric_points():
    "verify results obtained for symmetrically positioned sources"
    # sources
    S = {
        "x": np.array([-100, 100]),
        "y": np.array([0, 0]),
        "z": np.array([0, 0]),
    }

    # computation points
    P = {
        "x": np.array([0, 0]),
        "y": np.array([-140, 140]),
        "z": np.array([0, 0]),
    }
    R2 = idist.sedm(P, S)
    Vyy = idist.grad_tensor(P, S, R2, ["yy"])

    aae(Vyy[0][0, :], Vyy[0][1, :], decimal=15)


def test_grad_tensor_zz_symmetric_points():
    "verify results obtained for symmetrically positioned sources"

    # sources
    S = {
        "x": np.array([0, 0]),
        "y": np.array([0, 0]),
        "z": np.array([100, 200]),
    }
    # computation points
    P = {
        "x": np.array([0, 140]),
        "y": np.array([-140, 0]),
        "z": np.array([0, 0]),
    }
    R2 = idist.sedm(P, S)
    Vzz = idist.grad_tensor(P, S, R2, ["zz"])

    aae(Vzz[0][0, :], Vzz[0][1, :], decimal=15)


def test_grad_tensor_Laplace():
    "abs values of second derivatives must decay with distance"

    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([10.0])}

    # computation points
    P = {
        "x": np.array([-10, -10, 0, 10, 0, 0]),
        "y": np.array([0, -10, 0, 0, 10, 0]),
        "z": np.array([0, 0, -10, 0, 0, 0]),
    }
    # second derivatives produced by shallow sources
    R2 = idist.sedm(P, S)
    Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = idist.grad_tensor(P, S, R2)
    aae(Vzz, -Vxx - Vyy, decimal=15)


def test_grad_tensor_known_points():
    "verify results obtained for specific points"

    # single source
    S = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([10.0])}

    # computation points
    P = {
        "x": np.array([-10, -10, 10, 10, 0, 0]),
        "y": np.array([0, -10, 0, -10, 10, 0]),
        "z": np.array([0, 0, -10, 0, 0, 0]),
    }

    Vxx_ref = np.array(
        [
            [3 * (-10) * (-10) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (10) * (10) / (np.sqrt(500) ** 5) - 1 / (np.sqrt(500) ** 3)],
            [3 * (10) * (10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (0) * (0) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (0) * (0) / (10**5) - 1 / (10**3)],
        ]
    )

    Vxy_ref = np.array(
        [
            [3 * (-10) * (0) / (np.sqrt(200) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (10) * (0) / (np.sqrt(500) ** 5)],
            [3 * (10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (0) * (10) / (np.sqrt(200) ** 5)],
            [3 * (0) * (0) / (10**5)],
        ]
    )

    Vxz_ref = np.array(
        [
            [3 * (-10) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (10) * (-20) / (np.sqrt(500) ** 5)],
            [3 * (10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (0) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (0) * (-10) / (10**5)],
        ]
    )

    Vyy_ref = np.array(
        [
            [3 * (0) * (0) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (0) * (0) / (np.sqrt(500) ** 5) - 1 / (np.sqrt(500) ** 3)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5) - 1 / (np.sqrt(300) ** 3)],
            [3 * (10) * (10) / (np.sqrt(200) ** 5) - 1 / (np.sqrt(200) ** 3)],
            [3 * (0) * (0) / (10**5) - 1 / (10**3)],
        ]
    )

    Vyz_ref = np.array(
        [
            [3 * (0) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (0) * (-20) / (np.sqrt(500) ** 5)],
            [3 * (-10) * (-10) / (np.sqrt(300) ** 5)],
            [3 * (10) * (-10) / (np.sqrt(200) ** 5)],
            [3 * (0) * (-10) / (10**5)],
        ]
    )

    R2 = idist.sedm(P, S)
    Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = idist.grad_tensor(P, S, R2)
    aae(Vxx, Vxx_ref, decimal=15)
    aae(Vxy, Vxy_ref, decimal=15)
    aae(Vxz, Vxz_ref, decimal=15)
    aae(Vyy, Vyy_ref, decimal=15)
    aae(Vyz, Vyz_ref, decimal=15)


##### grad tensor BTTB


def test_grad_tensor_BTTB_compare_grad_xy():
    "verify if grad_tensor_BTTB produces the same result as grad_tensor for a xy grid"
    # cordinates of the grid
    x = np.linspace(1.3, 5.7, 5) * 1e3
    y = np.linspace(100.0, 104.3, 4) * 1e3
    Dz = 15.8 * 1e3
    # define points with 'ordering'='xy'
    xp, yp = np.meshgrid(x, y, indexing="xy")
    zp = np.zeros_like(xp) + 30.0
    data_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel()}
    source_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel() + Dz}
    grid = {"x": x[:, np.newaxis], "y": y, "z": 30.0, "ordering": "xy"}
    # compute the SEDM's
    SEDM = idist.sedm(data_points=data_points, source_points=source_points)
    SEDM_BTTB = idist.sedm_BTTB(data_grid=grid, delta_z=Dz)
    # compute grad's
    GRAD = idist.grad_tensor(
        data_points=data_points,
        source_points=source_points,
        SEDM=SEDM,
        components=["xx", "xy", "xz", "yy", "yz", "zz"],
    )
    GRAD_BTTB_1st_col = idist.grad_tensor_BTTB(
        data_grid=grid,
        delta_z=Dz,
        SEDM=SEDM_BTTB,
        components=["xx", "xy", "xz", "yy", "yz", "zz"],
    )
    for element, element_BTTB_1st_col in zip(GRAD, GRAD_BTTB_1st_col):
        element_BTTB = conv.general_BTTB(
            num_blocks=y.size,
            columns_blocks=np.reshape(
                a=element_BTTB_1st_col, newshape=(y.size, x.size)
            ),
            rows_blocks=None,
        )
        aae(element, element_BTTB, decimal=8)


def test_grad_tensor_BTTB_compare_grad_yx():
    "verify if grad_tensor_BTTB produces the same result as grad_tensor for a yx grid"
    # cordinates of the grid
    x = np.linspace(1.3, 5.7, 5) * 1e3
    y = np.linspace(100.0, 104.3, 4) * 1e3
    Dz = 15.8 * 1e3
    # define points with 'ordering'='yx'
    xp, yp = np.meshgrid(x, y, indexing="ij")
    zp = np.zeros_like(xp) + 30.0
    data_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel()}
    source_points = {"x": xp.ravel(), "y": yp.ravel(), "z": zp.ravel() + Dz}
    grid = {"x": x[:, np.newaxis], "y": y, "z": 30.0, "ordering": "yx"}
    # compute the SEDM's
    SEDM = idist.sedm(data_points=data_points, source_points=source_points)
    SEDM_BTTB = idist.sedm_BTTB(data_grid=grid, delta_z=Dz)
    # compute grad's
    GRAD = idist.grad_tensor(
        data_points=data_points,
        source_points=source_points,
        SEDM=SEDM,
        components=["xx", "xy", "xz", "yy", "yz", "zz"],
    )
    GRAD_BTTB_1st_col = idist.grad_tensor_BTTB(
        data_grid=grid,
        delta_z=Dz,
        SEDM=SEDM_BTTB,
        components=["xx", "xy", "xz", "yy", "yz", "zz"],
    )
    for element, element_BTTB_1st_col in zip(GRAD, GRAD_BTTB_1st_col):
        element_BTTB = conv.general_BTTB(
            num_blocks=x.size,
            columns_blocks=np.reshape(
                a=element_BTTB_1st_col, newshape=(x.size, y.size)
            ),
            rows_blocks=None,
        )
        aae(element, element_BTTB, decimal=8)
