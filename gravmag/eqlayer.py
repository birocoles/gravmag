import numpy as np
from scipy.spatial import distance
from . import inverse_distance as id
from . import check


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

    # dictionary translating field into kernel matrix
    kernel = {
        "potential": 1.0 / R2,
        "x": id.grad(data_points, source_points, R2, ["x"], False)[0],
        "y": id.grad(data_points, source_points, R2, ["y"], False)[0],
        "z": id.grad(data_points, source_points, R2, ["z"], False)[0],
        "xx": id.grad_tensor(data_points, source_points, R2, ["xx"], False)[0],
        "xy": id.grad_tensor(data_points, source_points, R2, ["xy"], False)[0],
        "xz": id.grad_tensor(data_points, source_points, R2, ["xz"], False)[0],
        "yy": id.grad_tensor(data_points, source_points, R2, ["yy"], False)[0],
        "yz": id.grad_tensor(data_points, source_points, R2, ["yz"], False)[0],
        "zz": id.grad_tensor(data_points, source_points, R2, ["zz"], False)[0],
    }

    return kernel[field]
