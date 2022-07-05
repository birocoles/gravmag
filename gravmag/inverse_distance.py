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
        # the code below removes possibly duplicated components in components
        _, _indices = np.unique(
            np.asarray(components, dtype=str), return_index=True
        )
        components = np.array(components)[np.sort(_indices)]
        # check if components are valid
        for component in components:
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
    for component in components:
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
        # the code below removes possibly duplicated components in components
        _, _indices = np.unique(
            np.asarray(components, dtype=str), return_index=True
        )
        components = np.array(components)[np.sort(_indices)]
        # check if components are valid
        for component in components:
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

    # compute the gradient tensor components defined in components
    Kab = []
    if ("xx" in components) or ("yy" in components) or ("zz" in components):
        aux = 1 / R3  # compute this term only if it is necessary
    else:
        aux = 0
    for component in components:
        index1, index2 = component_indices[component]
        delta1 = data_points[index1][:, np.newaxis] - source_points[index1]
        delta2 = data_points[index2][:, np.newaxis] - source_points[index2]
        if component in ["xx", "yy", "zz"]:
            Kab.append((3 * delta1 * delta2) / R5 - aux)
        else:
            Kab.append((3 * delta1 * delta2) / R5)

    return Kab
