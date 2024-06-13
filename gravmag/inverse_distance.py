"""
Algoritmhs for computing the derivatives of the inverse distance function
between a set of data points and a set of source points.
"""

import numpy as np
from scipy.spatial import distance
from . import check, utils
from . import convolve as cv


def sedm(data_points, source_points, check_input=True):
    """
    Compute the full Squared Euclidean Distance Matrix (SEDM) between the data points
    and the source points (Dokmanic et. al, 2015).

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


def sedm_BTTB(data_grid, delta_z, ordering, check_input=True):
    """
    Compute the first column of the Squared Euclidean Distance Matrix (SEDM) between
    a horizontal regular grid of Nx x Ny data points and a grid of source points having the
    same shape, but dislocated by a constant and positive vertical distance (Dokmanic et. al, 2015).
    This function optimizes `sedm` to deal with BTTB matrices (Chan and Jin, 2007, p. 67).

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    SEDM: dictionary
        Returns a dictionary containing the metadata associated with the full matrix
        (see input of function 'check.BTTB_metadata').
    """

    if check_input is True:
        # check shape and ndim of points
        check.is_grid_xy(grid=data_grid)
        check.is_scalar(x=delta_z, positive=True)
        check.is_ordering(ordering)

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute auxiliary variables SEDMx, SEDMy and SEDMz
    SEDMx = (
        data_grid["x"] * (data_grid["x"] - 2 * data_grid["x"][0])
        + data_grid["x"][0] * data_grid["x"][0]
    )
    SEDMy = (
        data_grid["y"] * (data_grid["y"] - 2 * data_grid["y"][0])
        + data_grid["y"][0] * data_grid["y"][0]
    )
    SEDMz = delta_z * delta_z

    if ordering == "xy":
        # compute the auxiliary vector associated with SEDM
        SEDM = np.tile(SEDMx, Ny) + np.repeat(SEDMy, Nx) + SEDMz
        # define shape
        shape = data_grid["shape"][::-1]
    else:  # data_grid['ordering'] == 'yx'
        # use broadcasting rules to add DX, DY and DZ
        SEDM = np.repeat(SEDMx, Ny) + np.tile(SEDMy, Nx) + SEDMz
        # define shape
        shape = data_grid["shape"]
    # dictionary containing metadata associated with the full SEDM
    BTTB = {
        "ordering": ordering,
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": shape[0],
        "columns": np.reshape(SEDM, shape),
        "rows": None,
    }

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
    Ka: Dictionary of numpy arrays 2d
        Dictionary of N x M matrices containing the partial derivatives of first order
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
        check.is_array(x=SEDM, ndim=2)
        if SEDM.shape != (D, P):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )

    # compute the cube of inverse distance function from the SEDM
    R3 = SEDM * np.sqrt(SEDM)

    # compute the gradient components defined in components
    Ka = dict()
    Ka["header"] = (
        "1st-order partial derivative(s) of the inverse distance function computed at scattered points"
    )
    for component in components:
        delta = data_points[component][:, np.newaxis] - source_points[component]
        Ka[component] = -delta / R3

    return Ka


def grad_BTTB(
    data_grid,
    delta_z,
    SEDM,
    ordering,
    components=["x", "y", "z"],
    check_input=True,
):
    """
    Compute the partial derivatives of first order of the inverse distance
    function between a horizontal regular grid of Nx x Ny data points and a
    grid of source points having the same shape, but dislocated by a constant
    and positive vertical distance.
    This function optimizes `grad` to deal with BTTB matrices (Chan and Jin, 2007, p. 67).

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    SEDM: dictionary
        Dictionary containing the metadata associated with the full matrix
        (output of function 'inverse_distance.sedm_BTTB').
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    components : list of strings
        List of strings defining the Cartesian components to be computed.
        Default is ['x', 'y', 'z'], which contains all possible components.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Ka: Dictionary of dictionaries
        Dictionary of dictionaries containing the metadata associated with the full matrix
        (see input of function 'check.BTTB_metadata').
    """

    if check_input is True:
        # check shape and ndim of points
        D = check.is_grid_xy(data_grid)
        check.is_scalar(x=delta_z, positive=True)
        # check if components are valid
        for component in components:
            if component not in ["x", "y", "z"]:
                raise ValueError("component {} invalid".format(component))
        # check the SEDM
        check.BTTB_metadata(SEDM)
        check.is_ordering(ordering)

    # compute the cube of inverse distance function from the SEDM
    R3 = SEDM["columns"] * np.sqrt(SEDM["columns"])

    delta_func = {"x": _delta_x, "y": _delta_y, "z": _delta_z}

    # compute the gradient components defined in components
    Ka = dict()
    Ka["header"] = (
        "1st-order partial derivative(s) of the inverse distance function computed at gridded points"
    )
    for component in components:
        # get the parameters of the BTTB matrix
        symmetries, shape, delta = delta_func[component](
            data_grid, delta_z, ordering
        )
        # dictionary containing metadata associated with the full BTTB
        BTTB = {
            "ordering": ordering,
            "symmetry_structure": symmetries[0],
            "symmetry_blocks": symmetries[1],
            "nblocks": shape[0],
            "columns": delta / R3,
            "rows": None,
        }
        Ka[component] = BTTB

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
    Kab: Dictionary numpy arrays 2d
        Dictionary of N x M matrices containing the computed partial derivatives of
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
        check.is_array(x=SEDM, ndim=2)
        if SEDM.shape != (D, P):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )

    # compute the inverse distance function to the powers 3 and 5
    R3 = SEDM * np.sqrt(SEDM)
    R5 = R3 * SEDM

    # compute the gradient tensor components defined in components
    Kab = dict()
    Kab["header"] = (
        "2nd-order partial derivative(s) of the inverse distance function computed at scattered points"
    )
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
            Kab[component] = ((3 * delta1 * delta2) / R5) - aux
        else:
            Kab[component] = (3 * delta1 * delta2) / R5

    return Kab


def grad_tensor_BTTB(
    data_grid,
    delta_z,
    SEDM,
    ordering,
    components=["xx", "xy", "xz", "yy", "yz", "zz"],
    check_input=True,
):
    """
    Compute the partial derivatives of second order of the inverse distance
    function between a horizontal regular grid of Nx x Ny data points and a
    grid of source points having the same shape, but dislocated by a constant
    and positive vertical distance.
    This function optimizes `grad_tensor` to deal with BTTB matrices (Chan and Jin, 2007, p. 67).

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    SEDM: dictionary
        Dictionary containing the metadata associated with the full matrix
        (output of function 'inverse_distance.sedm_BTTB').
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    components : list of strings
        List of strings defining the tensor components to be computed.
        Default is ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'], which contains all
        possible components.
    check_input : boolean
        If True, verify if the input is valid. Default is True.


    returns
    -------
    Kab: Dictionary of numpy arrays 2d
        Dictionary of N x M matrices containing the computed partial derivatives of
        second order.
    """

    if check_input is True:
        # check shape and ndim of points
        D = check.is_grid_xy(data_grid)
        check.is_scalar(x=delta_z, positive=True)
        # check if components are valid
        for component in components:
            if component not in ["xx", "xy", "xz", "yy", "yz", "zz"]:
                raise ValueError("component {} invalid".format(component))
        # check the SEDM
        check.BTTB_metadata(SEDM)
        check.is_ordering(ordering)

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the inverse distance function to the powers 3 and 5
    R3 = SEDM["columns"] * np.sqrt(SEDM["columns"])
    R5 = R3 * SEDM["columns"]

    delta_func = {
        "xx": _delta_xx,
        "xy": _delta_xy,
        "xz": _delta_xz,
        "yy": _delta_yy,
        "yz": _delta_yz,
        "zz": _delta_zz,
    }

    # compute the gradient tensor components defined in components
    Kab = dict()
    Kab["header"] = (
        "2nd-order partial derivative(s) of the inverse distance function computed at gridded points"
    )
    for component in components:
        # get the parameters of the BTTB matrix
        symmetries, shape, delta = delta_func[component](
            data_grid, delta_z, ordering
        )
        # dictionary containing metadata associated with the full BTTB
        BTTB = {
            "ordering": ordering,
            "symmetry_structure": symmetries[0],
            "symmetry_blocks": symmetries[1],
            "nblocks": shape[0],
            "columns": (delta / R5),
            "rows": None,
        }
        if component in ["xx", "yy", "zz"]:
            BTTB["columns"] -= 1 / R3
        Kab[component] = BTTB

    return Kab


def directional_1st_order(
    data_points,
    source_points,
    SEDM,
    inc,
    dec,
    check_input=True,
):
    """
    Compute the partial directional derivative of first order of the inverse distance
    function between the data points and the source points. The direcional derivative
    is computed along a direction defined by a given inclination and declination.

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
    inc, dec : ints or floats
        Scalars defining the constant inclination and declination of the
        direction along which the derivative will be computed.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Kt: Dictionary
        Dictionary containing the x, y and z components of the computed 1st-orfer directional derivative
        along a constant direction with predefined inclination and declination.
        The x, y and z components are stored at the keys 'tx', 'ty' and 'tz', respectively.
        Inclination and declination values at keys 'inclination' and 'declination', respectively.
    """

    if check_input is True:
        # check shape and ndim of points
        D = check.are_coordinates(data_points)
        P = check.are_coordinates(source_points)
        # check if SEDM match data_points and source_points
        check.is_array(x=SEDM, ndim=2)
        if SEDM.shape != (D, P):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )
        check.is_scalar(x=inc, positive=False)
        check.is_scalar(x=dec, positive=False)

    # compute the gradient components along x, y and z directions
    Grad = grad(
        data_points=data_points,
        source_points=source_points,
        SEDM=SEDM,
        components=["x", "y", "z"],
        check_input=False,
    )

    # compute unit vector with direction defined by 'inc' and 'dec'
    t = utils.unit_vector(inc=inc, dec=dec, check_input=False)

    # compute the directional derivative of 1st order
    Kt = dict()
    Kt["header"] = (
        "1st-order directional derivative of the inverse distance function computed at scattered points"
    )
    Kt["tx"] = t[0] * Grad["x"]
    Kt["ty"] = t[1] * Grad["y"]
    Kt["tz"] = t[2] * Grad["z"]
    Kt["inclination"] = inc
    Kt["declination"] = dec

    return Kt


def directional_1st_order_BTTB(
    data_grid,
    delta_z,
    SEDM,
    ordering,
    inc,
    dec,
    check_input=True,
):
    """
    Compute the partial directional derivative of first order of the inverse distance
    function between a horizontal regular grid of Nx x Ny data points and a
    grid of source points having the same shape, but dislocated by a constant
    and positive vertical distance.
    The direcional derivative is computed along a direction defined by a given inclination and declination.
    This function optimizes `directional_1st_order` to deal with BTTB matrices (Chan and Jin, 2007, p. 67).

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    SEDM: dictionary
        Dictionary containing the metadata associated with the full matrix
        (output of function 'inverse_distance.sedm_BTTB').
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    inc, dec : ints or floats
        Scalars defining the constant inclination and declination of the
        direction along which the derivative will be computed.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Kt: Dictionary
        Dictionary containing the x, y and z components of the computed 1st-orfer directional derivative
        along a constant direction with predefined inclination and declination.
        The x, y and z components are stored at the keys 'tx', 'ty' and 'tz', respectively,
        in the form of BTTB matrices (see input of function 'check.BTTB_metadata').
        Inclination and declination values at keys 'inclination' and 'declination', respectively.
        .
    """

    if check_input is True:
        # check shape and ndim of points
        D = check.is_grid_xy(data_grid)
        check.is_scalar(x=delta_z, positive=True)
        # check the SEDM
        check.BTTB_metadata(SEDM)
        check.is_ordering(ordering)
        check.is_scalar(x=inc, positive=False)
        check.is_scalar(x=dec, positive=False)

    # compute the gradient components along x, y and z directions
    Grad = grad_BTTB(
        data_grid=data_grid,
        delta_z=delta_z,
        SEDM=SEDM,
        ordering=ordering,
        components=["x", "y", "z"],
        check_input=False,
    )

    # compute unit vector with direction defined by 'inc' and 'dec'
    t = utils.unit_vector(inc=inc, dec=dec, check_input=False)

    # compute the gradient components defined in components
    Kt = dict()
    Kt["header"] = (
        "1st-order partial derivative(s) of the inverse distance function computed at gridded points"
    )
    Kt["tx"] = Grad["x"].copy()
    Kt["ty"] = Grad["y"].copy()
    Kt["tz"] = Grad["z"].copy()
    Kt["tx"]["columns"] *= t[0]
    Kt["ty"]["columns"] *= t[1]
    Kt["tz"]["columns"] *= t[2]
    Kt["inclination"] = inc
    Kt["declination"] = dec

    return Kt


def directional_2nd_order(
    data_points,
    source_points,
    SEDM,
    inc0,
    dec0,
    inc,
    dec,
    check_input=True,
):
    """
    Compute the partial directional derivative of second order of the inverse distance
    function between the data points and the source points. The direcional derivative
    is computed along a directions defined by the given inclination/declinations pairs
    (inc0, dec0) and (inc, dec).

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
    inc0, dec0, inc, dec : ints or floats
        Scalars defining the constant inclinations and declinations of the
        directions along which the derivative will be computed.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Ktu: Dictionary
        Dictionary containing the xx, xy, xz, yy and yz components of the computed 2nd-order directional derivative
        along a constant directions with predefined inclinations and declinations.
        The xx, xy, xz, yy and yz components are stored at the keys 'xx', 'xy', 'xz', 'yy' and 'yz', respectively.
        Inclinations and declinations values at keys 'inclination', 'declination0', 'inclination' and 'declination'.
    """

    if check_input is True:
        # check shape and ndim of points
        D = check.are_coordinates(data_points)
        P = check.are_coordinates(source_points)
        # check if SEDM match data_points and source_points
        check.is_array(x=SEDM, ndim=2)
        if SEDM.shape != (D, P):
            raise ValueError(
                "SEDM does not match data_points and source_points"
            )
        check.is_scalar(x=inc0, positive=False)
        check.is_scalar(x=dec0, positive=False)
        check.is_scalar(x=inc, positive=False)
        check.is_scalar(x=dec, positive=False)

    # compute the tensor components along x, y and z directions
    Tensor = grad_tensor(
        data_points=data_points,
        source_points=source_points,
        SEDM=SEDM,
        components=["xx", "xy", "xz", "yy", "yz"],
        check_input=False,
    )

    # compute unit vector with direction defined by 'inc' and 'dec'
    t = utils.unit_vector(inc=inc, dec=dec, check_input=False)

    # compute unit vector with direction defined by 'inc0' and 'dec0'
    u = utils.unit_vector(inc=inc0, dec=dec0, check_input=False)

    # compute the directional factors
    a = utils.directional_factors(t, u, check_input=False)

    # compute the directional derivative of 1st order
    Ktu = dict()
    Ktu["header"] = (
        "2nd-order directional derivative of the inverse distance function computed at scattered points"
    )
    Ktu["xx"] = a['xx'] * Tensor["xx"]
    Ktu["xy"] = a['xy'] * Tensor["xy"]
    Ktu["xz"] = a['xz'] * Tensor["xz"]
    Ktu["yy"] = a['yy'] * Tensor["yy"]
    Ktu["yz"] = a['yz'] * Tensor["yz"]
    Ktu["inclination0"] = inc0
    Ktu["declination0"] = dec0
    Ktu["inclination"] = inc
    Ktu["declination"] = dec

    return Ktu


def directional_2nd_order_BTTB(
    data_grid,
    delta_z,
    SEDM,
    ordering,
    inc0,
    dec0,
    inc,
    dec,
    check_input=True,
):
    """
    Compute the partial directional derivative of second order of the inverse distance
    function between a horizontal regular grid of Nx x Ny data points and a
    grid of source points having the same shape, but dislocated by a constant
    and positive vertical distance.
    The direcional derivative is computed along a directions defined by the given
    inclination/declinations pairs (inc0, dec0) and (inc, dec).
    This function optimizes `directional_2nd_order` to deal with BTTB matrices (Chan and Jin, 2007, p. 67).

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    SEDM: dictionary
        Dictionary containing the metadata associated with the full matrix
        (output of function 'inverse_distance.sedm_BTTB').
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
    inc0, dec0, inc, dec : ints or floats
        Scalars defining the constant inclinations and declinations of the
        directions along which the derivative will be computed.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    Ktu: Dictionary
        Dictionary containing the xx, xy, xz, yy and yz components of the computed 2nd-orfer directional derivative
        along the constant directions with predefined inclinations and declinations.
        The xx, xy, xz, yy and yz components are stored at the keys 'xx', 'xy', 'xz', 'yy' and 'yz', respectively,
        in the form of BTTB matrices (see input of function 'check.BTTB_metadata').
        Inclination and declination values at keys 'inclination0', 'declination0', 'inclination' and 'declination'.
        .
    """

    if check_input is True:
        # check shape and ndim of points
        D = check.is_grid_xy(data_grid)
        check.is_scalar(x=delta_z, positive=True)
        # check the SEDM
        check.BTTB_metadata(SEDM)
        check.is_ordering(ordering)
        check.is_scalar(x=inc0, positive=False)
        check.is_scalar(x=dec0, positive=False)
        check.is_scalar(x=inc, positive=False)
        check.is_scalar(x=dec, positive=False)

    # compute the gradient components along x, y and z directions
    Tensor = grad_tensor_BTTB(
        data_grid=data_grid,
        delta_z=delta_z,
        SEDM=SEDM,
        ordering=ordering,
        components=["xx", "xy", "xz", "yy", "yz"],
        check_input=False,
    )

    # compute unit vector with direction defined by 'inc' and 'dec'
    t = utils.unit_vector(inc=inc, dec=dec, check_input=False)

    # compute unit vector with direction defined by 'inc0' and 'dec0'
    u = utils.unit_vector(inc=inc0, dec=dec0, check_input=False)

    # compute the directional factors
    a = utils.directional_factors(t, u, check_input=False)

    # compute the gradient components defined in components
    Ktu = dict()
    Ktu["header"] = (
        "2nd-order partial derivative(s) of the inverse distance function computed at gridded points"
    )
    Ktu["xx"] = Tensor["xx"].copy()
    Ktu["xy"] = Tensor["xy"].copy()
    Ktu["xz"] = Tensor["xz"].copy()
    Ktu["yy"] = Tensor["yy"].copy()
    Ktu["yz"] = Tensor["yz"].copy()
    Ktu["xx"]["columns"] *= a['xx']
    Ktu["xy"]["columns"] *= a['xy']
    Ktu["xz"]["columns"] *= a['xz']
    Ktu["yy"]["columns"] *= a['yy']
    Ktu["yz"]["columns"] *= a['yz']
    Ktu["inclination0"] = inc0
    Ktu["declination0"] = dec0
    Ktu["inclination"] = inc
    Ktu["declination"] = dec

    return Ktu


def _delta_x(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component x.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : numpy array 2d
        Term -(x_i - x_j) arranged in a matrix.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the auxiliary variable
    aux = -(data_grid["x"] - data_grid["x"][0])

    if ordering == "xy":
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("symm", "skew")
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
        # term -(x_i - x_j)
        delta = np.reshape(np.tile(aux, Ny), shape)
    else:  # ordering == "yx"
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("skew", "symm")
        # shape (Nx, Ny)
        shape = data_grid["shape"]
        # term -(x_i - x_j)
        delta = np.reshape(np.repeat(aux, Ny), shape)

    return symmetries, shape, delta


def _delta_y(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component y.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : numpy array 2d
        Term -(y_i - y_j) arranged in a matrix.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the auxiliary variable
    aux = -(data_grid["y"] - data_grid["y"][0])

    if ordering == "xy":
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("skew", "symm")
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
        # term -(y_i - y_j)
        delta = np.reshape(np.repeat(aux, Nx), shape)
    else:  # ordering == "yx"
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("symm", "skew")
        # shape (Nx, Ny)
        shape = data_grid["shape"]
        # term -(y_i - y_j)
        delta = np.reshape(np.tile(aux, Nx), shape)

    return symmetries, shape, delta


def _delta_z(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component z.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : scalar
        Term -(z_i - z_j).
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # (symmetry_structure, symmetry_blocks)
    symmetries = ("symm", "symm")
    # term -(z_i - z_j)
    delta = delta_z
    if ordering == "xy":
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
    else:  # ordering == "yx"
        # shape (Nx, Ny)
        shape = data_grid["shape"]

    return symmetries, shape, delta


def _delta_xx(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component xx.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : numpy array 2d
        Term 3 * (x_i - x_j)**2 arranged in a matrix.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the auxiliary variable
    aux = 3 * (data_grid["x"] - data_grid["x"][0]) ** 2

    # (symmetry_structure, symmetry_blocks)
    symmetries = ("symm", "symm")
    if ordering == "xy":
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
        # term 3 * (x_i - x_j)**2
        delta = np.reshape(np.tile(aux, Ny), shape)
    else:  # ordering == "yx"
        # shape (Nx, Ny)
        shape = data_grid["shape"]
        # term 3 * (x_i - x_j)**2
        delta = np.reshape(np.repeat(aux, Ny), shape)

    return symmetries, shape, delta


def _delta_xy(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component xy.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : numpy array 2d
        Term 3 * (x_i - x_j) * (y_i - y_j) arranged in a matrix.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the auxiliary variables
    aux_x = -(data_grid["x"] - data_grid["x"][0])
    aux_y = -(data_grid["y"] - data_grid["y"][0])

    # (symmetry_structure, symmetry_blocks)
    symmetries = ("skew", "skew")
    if ordering == "xy":
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
        # terms (x_i - x_j) and (y_i - y_j)
        delta_x = np.reshape(np.tile(aux_x, Ny), shape)
        delta_y = np.reshape(np.repeat(aux_y, Nx), shape)
    else:  # ordering == "yx"
        # shape (Nx, Ny)
        shape = data_grid["shape"]
        # terms (x_i - x_j) and (y_i - y_j)
        delta_x = np.reshape(np.repeat(aux_x, Ny), shape)
        delta_y = np.reshape(np.tile(aux_y, Nx), shape)

    # term 3 * (x_i - x_j) * (y_i - y_j)
    delta = 3 * delta_x * delta_y

    return symmetries, shape, delta


def _delta_xz(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component xz.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : numpy array 2d
        Term 3 * (x_i - x_j) * (z_i - z_j) arranged in a matrix.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the auxiliary variable
    aux_x = -(data_grid["x"] - data_grid["x"][0])

    if ordering == "xy":
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("symm", "skew")
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
        # term (x_i - x_j)
        delta_x = np.reshape(np.tile(aux_x, Ny), shape)
    else:  # ordering == "yx"
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("skew", "symm")
        # shape (Nx, Ny)
        shape = data_grid["shape"]
        # term (x_i - x_j)
        delta_x = np.reshape(np.repeat(aux_x, Ny), shape)

    # term 3 * (x_i - x_j) * (z_i - z_j)
    delta = 3 * delta_x * delta_z

    return symmetries, shape, delta


def _delta_yy(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component yy.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : numpy array 2d
        Term 3 * (y_i - y_j)**2 arranged in a matrix.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the auxiliary variable
    aux = 3 * (data_grid["y"] - data_grid["y"][0]) ** 2

    # (symmetry_structure, symmetry_blocks)
    symmetries = ("symm", "symm")
    if ordering == "xy":
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
        # term 3* (y_i - y_j)**2
        delta = np.reshape(np.repeat(aux, Nx), shape)
    else:  # ordering == "yx"
        # shape (Nx, Ny)
        shape = data_grid["shape"]
        # term 3* (y_i - y_j)**2
        delta = np.reshape(np.tile(aux, Nx), shape)

    return symmetries, shape, delta


def _delta_yz(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component yz.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : numpy array 2d
        Term 3 * (y_i - y_j) * (z_i - z_j) arranged in a matrix.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # compute the auxiliary variable
    aux_y = -(data_grid["y"] - data_grid["y"][0])

    if ordering == "xy":
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("skew", "symm")
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
        # term (y_i - y_j)
        delta_y = np.reshape(np.repeat(aux_y, Nx), shape)
    else:  # ordering == "yx"
        # (symmetry_structure, symmetry_blocks)
        symmetries = ("symm", "skew")
        # shape (Nx, Ny)
        shape = data_grid["shape"]
        # term (y_i - y_j)
        delta_y = np.reshape(np.tile(aux_y, Nx), shape)

    # term 3 * (y_i - y_j) * (z_i - z_j)
    delta = 3 * delta_y * delta_z

    return symmetries, shape, delta


def _delta_zz(data_grid, delta_z, ordering):
    """
    Parameters associated with the BTTB defined by field component zz.

    parameters
    ----------
    data_grid : dictionary
        Dictionary containing the x, y and z coordinates of the grid points (or nodes)
        at the keys 'x', 'y' and 'z', respectively, and the scheme for indexing the
        points at the key 'ordering'. See function 'data_structures.regular_grid_xy'.
    delta_z : float or int
        Positive scalar defining the constant vertical distance between the data and
        source grids of points.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.

    returns
    -------
    symmetries : tuple
        Strings defining the symmetries of the correponding BTTB (symmetry_structure, symmetry_blocks)
    shape : tuple
        Tuple defining the number of blocks and number of points per blocks
    delta : scalar
        Term 3 * (z_i - z_j)**2.
    """

    # number of points along x and y directions
    Nx = data_grid["x"].size
    Ny = data_grid["y"].size

    # (symmetry_structure, symmetry_blocks)
    symmetries = ("symm", "symm")
    # term 3 * (z_i - z_j)**2
    delta = 3 * delta_z**2
    if ordering == "xy":
        # shape (Ny, Nx)
        shape = data_grid["shape"][::-1]
    else:  # ordering == "yx"
        # shape (Nx, Ny)
        shape = data_grid["shape"]

    return symmetries, shape, delta
