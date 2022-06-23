"""
Algoritmhs for computing the derivatives of the inverse distance function
between a set of data points and a set of source points.
"""

import numpy as np
from scipy.spatial import distance
from . import check


def sedm(data_points, source_points, check_input=True):
    """
    Compute Squared Euclidean Distance Matrix (SEDM) between the data points
    and the source points.

    parameters
    ----------
    data_points: numpy array 2d
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N data points. The ith column contains the
        coordinates of the ith data point.
    source_points: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    SEDM: numpy array 2d
        N x M SEDM between data points and source points.
    """

    if check_input is True:
        # check shape and ndim of points
        check.coordinates(data_points)
        check.coordinates(source_points)

    # compute de SEDM by using scipy.spatial.distance.cdist
    SEDM = distance.cdist(data_points.T, source_points.T, "sqeuclidean")

    return SEDM


def grad(
    data_points,
    source_points,
    SEDM,
    components=["x", "y", "z"],
    check_input=True,
):
    """
    Compute the partial derivatives of first order of the inverse distance
    function between the data points and the source points.

    parameters
    ----------
    data_points: numpy array 2d
        3 x N matrix containing the coordinates
        x (1rt row), y (2nd row), z (3rd row) of N data points.
        The ith column contains the coordinates of the ith data point.
    source_points: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.
    SEDM: numpy array 2d
        Squared Euclidean Distance Matrix (SEDM) between the N data
        points and the M sources computed according to function 'sedm'.
    components : list of strings
        List of strings defining the Cartesian components to be computed.
        Default is ['x', 'y', 'z'], which contains all possible components.
        Repeated components are ignored.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Ka: list of numpy arrays 2d
        List of N x M matrices containing the partial derivatives of first order
        along x, y and z directions.
    """

    if check_input is True:
        # check shape and ndim of points
        check.coordinates(data_points)
        check.coordinates(source_points)
        # check number of elements in components
        if len(components) > 3:
            raise ValueError("components must have at most 3 elements")
        # convert components to array of strings
        # repeated components are ignored
        # the code below results in an unsorted array _components
        _, _indices = np.unique(
            np.asarray(components, dtype=str), return_index=True
        )
        _components = np.array(components)[np.sort(_indices)]
        # check if components are valid
        for component in _components:
            if component not in ["x", "y", "z"]:
                raise ValueError("component {} invalid".format(component))
        # check if SEDM match data_points and source_points
        if SEDM.shape != (data_points.shape[1], source_points.shape[1]):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )

    # define a dictionary for component indices
    component_index = {"x": 0, "y": 1, "z": 2}

    # compute the cube of inverse distance function from the SEDM
    R3 = SEDM * np.sqrt(SEDM)

    # compute the gradient components defined in _components
    Ka = []
    for component in _components:
        index = component_index[component]
        delta = data_points[index][:, np.newaxis] - source_points[index]
        Ka.append(-delta / R3)

    return Ka


def grad_tensor(
    data_points,
    source_points,
    SEDM,
    components=["xx", "xy", "xz", "yy", "yz", "zz"],
    check_input=True,
):
    """
    Compute the partial derivatives of second order xx, xy, xz, yy and yz of
    the inverse distance function between the data points and the source
    points.

    parameters
    ----------
    data_points: numpy array 2d
        3 x N matrix containing the coordinates
        x (1rt row), y (2nd row), z (3rd row) of N data points.
        The ith column contains the coordinates of the ith data point.
    source_points: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.
    SEDM: numpy array 2d
        Squared Euclidean Distance Matrix (SEDM) between the N data
        points and the M sources computed according to function 'sedm'.
    components : list of strings
        List of strings defining the tensor components to be computed.
        Default is ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'], which contains all
        possible components. Repeated components are ignored.
    check_input : boolean
        If True, verify if the input is valid. Default is True.


    returns
    -------
    Kab: list numpy arrays 2d
        List of N x M matrices containing the computed partial derivatives of
        second order.
    """

    if check_input is True:
        # check shape and ndim of points
        check.coordinates(data_points)
        check.coordinates(source_points)
        # check number of elements in components
        if len(components) > 6:
            raise ValueError("components must have at most 6 elements")
        # convert components to array of strings
        # repeated components are ignored
        # the code below results in an unsorted array _components
        _, _indices = np.unique(
            np.asarray(components, dtype=str), return_index=True
        )
        _components = np.array(components)[np.sort(_indices)]
        # check if components are valid
        for component in _components:
            if component not in ["xx", "xy", "xz", "yy", "yz", "zz"]:
                raise ValueError("component {} invalid".format(component))
        # check if SEDM match data_points and source_points
        if SEDM.shape != (data_points.shape[1], source_points.shape[1]):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )

    # define a dictionary for component indices
    component_indices = {
        "xx": (0, 0),
        "xy": (0, 1),
        "xz": (0, 2),
        "yy": (1, 1),
        "yz": (1, 2),
        "zz": (2, 2),
    }

    # compute the inverse distance function to the powers 3 and 5
    R3 = SEDM * np.sqrt(SEDM)
    R5 = R3 * SEDM

    # compute the gradient tensor components defined in _components
    Kab = []
    if ("xx" in _components) or ("yy" in _components) or ("zz" in _components):
        aux = 1 / R3  # compute this term only if it is necessary
    else:
        aux = 0
    for component in _components:
        index1, index2 = component_indices[component]
        delta1 = data_points[index1][:, np.newaxis] - source_points[index1]
        delta2 = data_points[index2][:, np.newaxis] - source_points[index2]
        if component in ["xx", "yy", "zz"]:
            Kab.append((3 * delta1 * delta2) / R5 - aux)
        else:
            Kab.append((3 * delta1 * delta2) / R5)

    return Kab


def Dv(v, Kx, Ky, Kz, check_input=True):
    """
    Compute a directional derivative of first order along the unit vector v.

    of the inverse distance
    function between the data points and the source points.

    parameters
    ----------
    v: numpy array 1d
        Unit vector defining the derivative direction.
    Kx, Ky, Kz: numpy arrays 2d
        Matrices containing the partial derivatives of the inverse distance
        function along x, y and z directions computed according to
        function 'grad'.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Kv: numpy array 2d
        Matrix containing the directional derivative along v.
    """
    if check_input is True:
        # check if v is a vector
        if (v.ndim != 1) or (v.size != 3):
            raise ValueError("v must be a vector with 3 elements")
        # check if v is a unit vector
        if np.sum(v * v) != 1:
            raise ValueError("v must be a unit vector")
        # check if Kx, Ky and Kz are matrices
        if Kx.ndim != 2:
            raise ValueError("Kx must be a matrix")
        if Ky.ndim != 2:
            raise ValueError("Ky must be a matrix")
        if Kz.ndim != 2:
            raise ValueError("Kz must be a matrix")
        # check if Kx, Ky and Kz have the same shape
        shape_x = Kx.shape
        if Ky.shape != shape_x:
            raise ValueError("Ky and Kx must have the same shape")
        if Kz.shape != shape_x:
            raise ValueError("Kz and Kx must have the same shape")

    Kv = v[0] * Kx + v[1] * Ky + v[2] * Kz

    return Kv


def grad_Dv(v, Kxx, Kxy, Kxz, Kyy, Kyz, check_input=True):
    """
    Compute the gradient components of the directional derivative of first
    order of the inverse distance function along the direction defined by
    vector v.

    parameters
    ----------
    v: numpy array 1d
        Unit vector defining the derivative direction.
    Kxx, Kxy, Kxz, Kyy, Kyz: numpyy arrays 2d
        N x M matrices containing the second derivatives xx, xy, xz, yy and yz
        computed according to function 'Dv'.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Kvx, Kvy, Kvz: numpy array 2d
        Matrix containing the Cartesian components of the gradient of the
        directional derivative of the inverse distance along v.
    """
    if check_input is True:
        # check if v is a vector
        if (v.ndim != 1) or (v.size != 3):
            raise ValueError("v must be a vector with 3 elements")
        # check if v is a unit vector
        if np.sum(v * v) != 1:
            raise ValueError("v must be a unit vector")
        # check if Kxx, Kxy, Kxz, Kyy and Kyz are matrices
        if Kxx.ndim != 2:
            raise ValueError("Kxx must be a matrix")
        if Kxy.ndim != 2:
            raise ValueError("Kxy must be a matrix")
        if Kxz.ndim != 2:
            raise ValueError("Kxz must be a matrix")
        if Kyy.ndim != 2:
            raise ValueError("Kyy must be a matrix")
        if Kyz.ndim != 2:
            raise ValueError("Kyz must be a matrix")
        # check if Kxx, Kxy, Kxz, Kyy and Kyz have the same shape
        shape_xx = Kxx.shape
        if Kxy.shape != shape_xx:
            raise ValueError("Kxy and Kxx must have the same shape")
        if Kxz.shape != shape_xx:
            raise ValueError("Kxz and Kxx must have the same shape")
        if Kyy.shape != shape_xx:
            raise ValueError("Kyy and Kxx must have the same shape")
        if Kyz.shape != shape_xx:
            raise ValueError("Kyz and Kxx must have the same shape")

    Kvx = v[0] * Kxx + v[1] * Kxy + v[2] * Kxz
    Kvy = v[0] * Kxy + v[1] * Kyy + v[2] * Kyz
    Kvz = v[0] * Kxz + v[1] * Kyz - v[2] * (Kxx + Kyy)

    return Kvx, Kvy, Kvz
