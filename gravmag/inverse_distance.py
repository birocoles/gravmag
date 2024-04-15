"""
Algoritmhs for computing the derivatives of the inverse distance function
between a set of data points and a set of source points.
"""

import numpy as np
from scipy.spatial import distance
from . import check


def sedm(data_points, source_points, check_input=True):
    """
    Compute the full Squared Euclidean Distance Matrix (SEDM) between the data points
    and the source points.

    parameters
    ----------
    data_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    source_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    SEDM: numpy array 2d
        N x M SEDM between data points and source points.
    """

    if check_input is True:
        # check shape and ndim of points
        check.are_coordinates(data_points)
        check.are_coordinates(source_points)

    # compute the SEDM using numpy
    D1 = (
        data_points["x"] * data_points["x"]
        + data_points["y"] * data_points["y"]
        + data_points["z"] * data_points["z"]
    )
    D2 = (
        source_points["x"] * source_points["x"]
        + source_points["y"] * source_points["y"]
        + source_points["z"] * source_points["z"]
    )
    D3 = 2 * (
        np.outer(data_points["x"], source_points["x"])
        + np.outer(data_points["y"], source_points["y"])
        + np.outer(data_points["z"], source_points["z"])
    )

    # use broadcasting rules to add D1, D2 and D3
    D = D1[:, np.newaxis] + D2[np.newaxis, :] - D3

    return D


def sedm_BTTB(data_grid, delta_z, check_input=True):
    """
    Compute the first column of the Squared Euclidean Distance Matrix (SEDM) between
    a horizontal regular grid of Nx x Ny data points and a grid of source points having the
    same shape, but dislocated by a constant and positive vertical distance.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'.

        Each key is a numpy array 1d containing only the non-repeating data
        coordinates in ascending order along the axes x, y and z. The nodes
        of the Nx x Ny grid of data points have indices i = 0, 1, ..., Nx-1 and j = 0, ..., Ny-1
        along the x and y axes, respectively, and all nodes have the same vertical coordinate z_0.
        Consider for example, a grid formed by Nx = 3 and Ny = 2. In this case, there
        are 3 non-repeating coordinates (x_0, x_1, x_2) along the x-axis and
        2 non-repeating coordinates (y_0, y_1) along the y-axis so that the coordinates
        of the nodes are arranged in the following matrix:

        (x_0, y_0, z_0) (x_0, y_1, z_0)
        (x_1, y_0, z_0) (x_1, y_1, z_0)  .
        (x_2, y_0, z_0) (x_2, y_1, z_0)

        Note that the non-repeating x and y coordinates form vectors (numpy arrays 1d)
        with elements x_i and y_j, respectively, where i = 0, ..., Nx-1 and j = 0, ..., Ny-1.

        It is also important noting that the nodes may be indexed by following two
        different schemes:
        (1) 0 - (x_0, y_0, z_0), 1 - (x_0, y_1, z_0), 2 - (x_1, y_0, z_0), ..., 5 - (x_2, y_1, z_0)
        (2) 0 - (x_0, y_0, z_0), 1 - (x_1, y_0, z_0), 2 - (x_2, y_0, z_0), ..., 5 - (x_2, y_1, z_0)
        In scheme (1), the nodes are indexed along the y-axis and then along the x-axis.
        In scheme (2), the nodes are indexed along the x-axis and then along the y-axis.

        Then, the data_grid dictionary must be formed by the following keys:
        'x' - numpy array 2d with Nx elements containing the x coordinates of the grid points and a single column;
        'y' - numpy array 1d with Ny elements containing the y coordinates of the grid points;
        'z' - scalar (float or int) defining the constant vertical coordinates of the grid points and
        'ordering' - string 'xy' or 'yx' defining how the grid points are indexed.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    SEDM: dictionary
        Returns a dictionary containing the metadata associated with the full matrix 
        (see input of function 'check.BTTB_metadata').
        The dictionary contains the first column of the N x N SEDM between data points 
        and source points, where N = Nx x Ny is the total number of data (and source) points.
    """

    if check_input is True:
        # check shape and ndim of points
        check.is_planar_grid(coordinates=data_grid)
        check.is_scalar(x=delta_z, positive=True)

    Nx = data_grid['x'].size
    Ny = data_grid['y'].size

    # compute the SEDM using numpy
    DX = (
        data_grid["x"] * (data_grid["x"] - 2 * data_grid["x"][0])
        + data_grid["x"][0] * data_grid["x"][0]
    )

    DY = (
        data_grid["y"] * (data_grid["y"] - 2 * data_grid["y"][0])
        + data_grid["y"][0] * data_grid["y"][0]
    )

    DZ = delta_z * delta_z

    # use broadcasting rules to add DX, DY and DZ
    D = DX + DY[np.newaxis, :] + DZ

    D = D.ravel()

    # dictionary containing metadata associated with the full SEDM
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "rows": None,
    }

    if data_grid["ordering"] == "xy":
        BTTB["nblocks"] = Ny
        BTTB["columns"] = np.reshape(a=D, newshape=(Ny, Nx))
    else:  # data_grid['ordering'] == 'yx'
        BTTB["nblocks"] = Nx
        BTTB["columns"] = np.reshape(a=D, newshape=(Nx, Ny))

    return BTTB


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
    data_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    source_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    SEDM: numpy array 2d
        Squared Euclidean Distance Matrix (SEDM) between the N data
        points and the M sources computed according to function 'sedm'.
    components : list of strings
        List of strings defining the Cartesian components to be computed.
        Default is ['x', 'y', 'z'], which contains all possible components.
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
        D = check.are_coordinates(data_points)
        P = check.are_coordinates(source_points)
        # check if components are valid
        for component in components:
            if component not in ["x", "y", "z"]:
                raise ValueError("component {} invalid".format(component))
        # check if SEDM match data_points and source_points
        if type(SEDM) != np.ndarray:
            raise ValueError("SEDM must be a numpy array")
        if SEDM.ndim != 2:
            raise ValueError("SEDM must be have ndim = 2")
        if SEDM.shape != (D, P):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )

    # compute the cube of inverse distance function from the SEDM
    R3 = SEDM * np.sqrt(SEDM)

    # compute the gradient components defined in components
    Ka = []
    for component in components:
        delta = data_points[component][:, np.newaxis] - source_points[component]
        Ka.append(-delta / R3)

    return Ka


def grad_BTTB(
    data_grid,
    delta_z,
    SEDM,
    components=["x", "y", "z"],
    check_input=True,
):
    """
    Compute the partial derivatives of first order of the inverse distance
    function between a horizontal regular grid of Nx x Ny data points and a
    grid of source points having the same shape, but dislocated by a constant
    and positive vertical distance.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'.

        Each key is a numpy array 1d containing only the non-repeating data
        coordinates in ascending order along the axes x, y and z. The nodes
        of the Nx x Ny grid of data points have indices i = 0, 1, ..., Nx-1 and j = 0, ..., Ny-1
        along the x and y axes, respectively, and all nodes have the same vertical coordinate z_0.
        Consider for example, a grid formed by Nx = 3 and Ny = 2. In this case, there
        are 3 non-repeating coordinates (x_0, x_1, x_2) along the x-axis and
        2 non-repeating coordinates (y_0, y_1) along the y-axis so that the coordinates
        of the nodes are arranged in the following matrix:

        (x_0, y_0, z_0) (x_0, y_1, z_0)
        (x_1, y_0, z_0) (x_1, y_1, z_0)  .
        (x_2, y_0, z_0) (x_2, y_1, z_0)

        Note that the non-repeating x and y coordinates form vectors (numpy arrays 1d)
        with elements x_i and y_j, respectively, where i = 0, ..., Nx-1 and j = 0, ..., Ny-1.

        It is also important noting that the nodes may be indexed by following two
        different schemes:
        (1) 0 - (x_0, y_0, z_0), 1 - (x_0, y_1, z_0), 2 - (x_1, y_0, z_0), ..., 5 - (x_2, y_1, z_0)
        (2) 0 - (x_0, y_0, z_0), 1 - (x_1, y_0, z_0), 2 - (x_2, y_0, z_0), ..., 5 - (x_2, y_1, z_0)
        In scheme (1), the nodes are indexed along the y-axis and then along the x-axis.
        In scheme (2), the nodes are indexed along the x-axis and then along the y-axis.

        Then, the data_grid dictionary must be formed by the following keys:
        'x' - numpy array 2d with Nx elements containing the x coordinates of the grid points and a single column;
        'y' - numpy array 1d with Ny elements containing the y coordinates of the grid points;
        'z' - scalar (float or int) defining the constant vertical coordinates of the grid points and
        'ordering' - string 'xy' or 'yx' defining how the grid points are indexed.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    SEDM: dictionary
        Output of the function 'sedm_BTTB'.
    components : list of strings
        List of strings defining the Cartesian components to be computed.
        Default is ['x', 'y', 'z'], which contains all possible components.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Ka: list of numpy arrays 1d
        List of vectors defining the first columns of N x M matrices containing the
        partial derivatives of first order along x, y and z directions.
    """

    if check_input is True:
        # check shape and ndim of points
        D = check.is_planar_grid(data_grid)
        check.is_scalar(x=delta_z, positive=True)
        # check if components are valid
        for component in components:
            if component not in ["x", "y", "z"]:
                raise ValueError("component {} invalid".format(component))
        # check the SEDM
        check.BTTB_metadata(BTTB=SEDM)
        if SEDM["columns"].size != D:
            raise ValueError("SEDM does not match data_points")

    # compute the cube of inverse distance function from the SEDM
    R3 = SEDM["columns"] * np.sqrt(SEDM["columns"]).ravel()

    # dictionary setting parameters for broadcast_to
    broadcast_to_args = {
        "x": (data_grid["x"] - data_grid["x"][0]),
        "y": (data_grid["y"] - data_grid["y"][0]),
        "z": -delta_z,
    }

    PAREI AQUI

    # compute the gradient components defined in components
    Ka = []
    if data_grid["ordering"] == "xy":
        for component in components:
            delta = np.broadcast_to(
                array=broadcast_to_args[component],
                shape=(data_grid["x"].size, data_grid["y"].size),
            )
            Ka.append(-(delta.T).ravel() / R3)
    else:  # data_grid['ordering'] == 'yx'
        for component in components:
            delta = np.broadcast_to(
                array=broadcast_to_args[component],
                shape=(data_grid["x"].size, data_grid["y"].size),
            )
            Ka.append(-delta.ravel() / R3)

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
    data_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    source_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    SEDM: numpy array 2d
        Squared Euclidean Distance Matrix (SEDM) between the N data
        points and the M sources computed according to function 'sedm'.
    components : list of strings
        List of strings defining the tensor components to be computed.
        Default is ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'], which contains all
        possible components.
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
        D = check.are_coordinates(data_points)
        P = check.are_coordinates(source_points)
        # check if components are valid
        for component in components:
            if component not in ["xx", "xy", "xz", "yy", "yz", "zz"]:
                raise ValueError("component {} invalid".format(component))
        # check if SEDM match data_points and source_points
        if type(SEDM) != np.ndarray:
            raise ValueError("SEDM must be a numpy array")
        if SEDM.ndim != 2:
            raise ValueError("SEDM must be have ndim = 2")
        if SEDM.shape != (D, P):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )

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
        delta1 = (
            data_points[component[0]][:, np.newaxis]
            - source_points[component[0]]
        )
        delta2 = (
            data_points[component[1]][:, np.newaxis]
            - source_points[component[1]]
        )
        if component in ["xx", "yy", "zz"]:
            Kab.append((3 * delta1 * delta2) / R5 - aux)
        else:
            Kab.append((3 * delta1 * delta2) / R5)

    return Kab


def grad_tensor_BTTB(
    data_grid,
    delta_z,
    SEDM,
    components=["xx", "xy", "xz", "yy", "yz", "zz"],
    check_input=True,
):
    """
    Compute the partial derivatives of second order of the inverse distance
    function between a horizontal regular grid of Nx x Ny data points and a
    grid of source points having the same shape, but dislocated by a constant
    and positive vertical distance.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'.

        Each key is a numpy array 1d containing only the non-repeating data
        coordinates in ascending order along the axes x, y and z. The nodes
        of the Nx x Ny grid of data points have indices i = 0, 1, ..., Nx-1 and j = 0, ..., Ny-1
        along the x and y axes, respectively, and all nodes have the same vertical coordinate z_0.
        Consider for example, a grid formed by Nx = 3 and Ny = 2. In this case, there
        are 3 non-repeating coordinates (x_0, x_1, x_2) along the x-axis and
        2 non-repeating coordinates (y_0, y_1) along the y-axis so that the coordinates
        of the nodes are arranged in the following matrix:

        (x_0, y_0, z_0) (x_0, y_1, z_0)
        (x_1, y_0, z_0) (x_1, y_1, z_0)  .
        (x_2, y_0, z_0) (x_2, y_1, z_0)

        Note that the non-repeating x and y coordinates form vectors (numpy arrays 1d)
        with elements x_i and y_j, respectively, where i = 0, ..., Nx-1 and j = 0, ..., Ny-1.

        It is also important noting that the nodes may be indexed by following two
        different schemes:
        (1) 0 - (x_0, y_0, z_0), 1 - (x_0, y_1, z_0), 2 - (x_1, y_0, z_0), ..., 5 - (x_2, y_1, z_0)
        (2) 0 - (x_0, y_0, z_0), 1 - (x_1, y_0, z_0), 2 - (x_2, y_0, z_0), ..., 5 - (x_2, y_1, z_0)
        In scheme (1), the nodes are indexed along the y-axis and then along the x-axis.
        In scheme (2), the nodes are indexed along the x-axis and then along the y-axis.

        Then, the data_grid dictionary must be formed by the following keys:
        'x' - numpy array 2d with Nx elements containing the x coordinates of the grid points and a single column;
        'y' - numpy array 1d with Ny elements containing the y coordinates of the grid points;
        'z' - scalar (float or int) defining the constant vertical coordinates of the grid points and
        'ordering' - string 'xy' or 'yx' defining how the grid points are indexed.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    SEDM: numpy array 1d
        First column of the N x N SEDM between data points and source points,
        where N = Nx x Ny is the total number of data (and source) points.
        Computed according to function 'sedm_BTTB'.
    components : list of strings
        List of strings defining the tensor components to be computed.
        Default is ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'], which contains all
        possible components.
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
        D = check.is_planar_grid(data_grid)
        check.is_scalar(x=delta_z, positive=True)
        # check if components are valid
        for component in components:
            if component not in ["xx", "xy", "xz", "yy", "yz", "zz"]:
                raise ValueError("component {} invalid".format(component))
        # check if SEDM match data_points and source_points
        if type(SEDM) != np.ndarray:
            raise ValueError("SEDM must be a numpy array")
        if SEDM.ndim != 1:
            raise ValueError("SEDM must be have ndim = 1")
        if SEDM.size != D:
            raise ValueError("SEDM does not match data_points")

    # dictionary setting parameters for broadcast_to
    broadcast_to_args = {
        "x": (data_grid["x"] - data_grid["x"][0]),
        "y": (data_grid["y"] - data_grid["y"][0]),
        "z": -delta_z,
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
    if data_grid["ordering"] == "xy":
        for component in components:
            delta1 = np.broadcast_to(
                array=broadcast_to_args[component[0]],
                shape=(data_grid["x"].size, data_grid["y"].size),
            )
            delta2 = np.broadcast_to(
                array=broadcast_to_args[component[1]],
                shape=(data_grid["x"].size, data_grid["y"].size),
            )
            if component in ["xx", "yy", "zz"]:
                Kab.append((3 * (delta1 * delta2).T.ravel()) / R5 - aux)
            else:
                Kab.append((3 * (delta1 * delta2).T.ravel()) / R5)
    else:  # data_grid['ordering'] == 'yx'
        for component in components:
            delta1 = np.broadcast_to(
                array=broadcast_to_args[component[0]],
                shape=(data_grid["x"].size, data_grid["y"].size),
            )
            delta2 = np.broadcast_to(
                array=broadcast_to_args[component[1]],
                shape=(data_grid["x"].size, data_grid["y"].size),
            )
            if component in ["xx", "yy", "zz"]:
                Kab.append((3 * (delta1 * delta2).ravel()) / R5 - aux)
            else:
                Kab.append((3 * (delta1 * delta2).ravel()) / R5)

    return Kab
