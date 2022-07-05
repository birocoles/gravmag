import numpy as np
from scipy.spatial import distance
from . import inverse_distance as id
from . import check, utils


def kernel_matrix_monopoles(data_points, z0, field=["z"], check_input=True):
    """
    Compute the kernel matrix produced by a planar layer of monopoles.

    parameters
    ----------
    data_points : numpy array 2d
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N data points. The ith column contains the
        coordinates of the ith data point.
    z0 : int or float
        Scalar defining the constant vertical coordinate of the layer.
    field : string
        Defines the field produced by the layer. The available options are:
        "potential", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz".
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    G: numpy array 2d
        N x M matrix defined by the kernel of the equivalent layer integral.
    """

    if check_input is True:
        data_points = np.asarray(data_points)
        check.coordinates(data_points)
        assert isinstance(z0, (int, float)), "z0 must be int or float"
        assert np.all(data_points[2] < z0), "all data points must be above z0"
        # check if field is valid
        if field not in [
            "potential",
            "x",
            "y",
            "z",
            "xx",
            "xy",
            "xz",
            "yy",
            "yz",
            "zz",
        ]:
            raise ValueError("invalid field {}".format(field))

    # define source points
    source_points = data_points.copy()
    source_points[2] = z0
    # compute Squared Euclidean Distance Matrix (SEDM)
    R2 = distance.cdist(data_points.T, source_points.T, "sqeuclidean")

    # compute the kernel matrix according to "field"
    if field is "potential":
        G = 1.0 / np.sqrt(R2)
    elif field in ["x", "y", "z"]:
        G = id.grad(data_points, source_points, R2, [field], False)[0]
    else:  # field is in ["xx", "xy", "xz", "yy", "yz", "zz"]
        G = id.grad_tensor(data_points, source_points, R2, [field], False)[0]

    return G


def kernel_matrix_dipoles(
    data_points, z0, inc, dec, field=["z"], check_input=True
):
    """
    Compute the kernel matrix produced by a planar layer of dipoles with
    constant magnetization direction.

    parameters
    ----------
    data_points : numpy array 2d
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N data points. The ith column contains the
        coordinates of the ith data point.
    z0 : int or float
        Scalar defining the constant vertical coordinate of the layer.
    inc, dec : ints or floats
        Scalars defining the constant inclination and declination of the
        dipoles magnetization.
    field : string
        Defines the field produced by the layer. The available options are:
        "potential", "x", "y", "z".
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    G: numpy array 2d
        N x M matrix defined by the kernel of the equivalent layer integral.
    """

    if check_input is True:
        data_points = np.asarray(data_points)
        check.coordinates(data_points)
        assert isinstance(z0, (int, float)), "z0 must be int or float"
        assert np.all(data_points[2] < z0), "all data points must be above z0"
        # check if field is valid
        if field not in ["potential", "x", "y", "z"]:
            raise ValueError("invalid field {}".format(field))

    # define source points
    source_points = data_points.copy()
    source_points[2] = z0
    # compute Squared Euclidean Distance Matrix (SEDM)
    R2 = distance.cdist(data_points.T, source_points.T, "sqeuclidean")
    # compute the unit vector defined by inc and dec
    u = utils.unit_vector(inc, dec)

    # compute the kernel matrix according to "field"
    if field is "potential":
        Gx, Gy, Gz = id.grad(
            data_points, source_points, R2, ["x", "y", "z"], False
        )
        G = -(u[0] * Gx + u[1] * Gy + u[2] * Gz)
    elif field is "x":
        Gxx, Gxy, Gxz = id.grad_tensor(
            data_points, source_points, R2, ["xx", "xy", "xz"], False
        )
        G = u[0] * Gxx + u[1] * Gxy + u[2] * Gxz
    elif field is "y":
        Gxy, Gyy, Gyz = id.grad_tensor(
            data_points, source_points, R2, ["xy", "yy", "yz"], False
        )
        G = u[0] * Gxy + u[1] * Gyy + u[2] * Gyz
    else:  # field is "z"
        Gxz, Gyz, Gzz = id.grad_tensor(
            data_points, source_points, R2, ["xz", "yz", "zz"], False
        )
        G = u[0] * Gxz + u[1] * Gyz + u[2] * Gzz

    return G
