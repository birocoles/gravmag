import numpy as np


def rectangular_prisms(prisms):
    """
    Check if prisms is a 2d numpy array with 6 columns.

    parameters
    ----------
    prisms : generic object 
        Python object to be verified.

    returns
    -------
    P : int
        Number of rows in prisms (total number of prisms).
    """
    if type(prisms) != np.ndarray:
        raise ValueError("prisms must be a numpy array")
    if prisms.ndim != 2:
        raise ValueError(
            "prisms ndim ({}) ".format(prisms.ndim) + "not equal to 2"
        )
    if prisms.shape[1] != 6:
        raise ValueError(
            "Number of columns in prisms ({}) ".format(prisms.shape[1])
            + "not equal to 6"
        )
    # check the x lower and upper limits
    if np.any(prisms[:,1] <= prisms[:,0]):
        raise ValueError(
            "all x2 values (2nd column) must be greater than x1 values (1st column)."
            )
    # check the y lower and upper limits
    if np.any(prisms[:,3] <= prisms[:,2]):
        raise ValueError(
            "all y2 values (4th column) must be greater than y1 values (3rd column)."
            )
    # check the z lower and upper limits
    if np.any(prisms[:,5] <= prisms[:,4]):
        raise ValueError(
            "all bottom (z2) values (5th column) must be greater than top (z1) values (6th column)."
            )

    # number of rows in prisms
    P = prisms.shape[0]
    return P


def coordinates(coordinates):
    """
    Check if coordinates is a 2d numpy array with 3 rows.

    parameters
    ----------
    coordinates : generic object 
        Python object to be verified.

    returns
    -------
    D : int
        Number of columns in coordinates (total number of data points).
    """
    if type(coordinates) != np.ndarray:
        raise ValueError("coordinates must be a numpy array")
    if coordinates.ndim != 2:
        raise ValueError(
            "coordinates ndim ({}) ".format(coordinates.ndim) + "not equal to 2"
        )
    if coordinates.shape[0] != 3:
        raise ValueError(
            "Number of lines in coordinates ({}) ".format(coordinates.shape[0])
            + "not equal to 3"
        )
    # number of columns in coordinates
    D = coordinates.shape[1]
    return D


def scalar_prop(prop, P):
    """
    Check if prop is a 1d numpy array having a previously defined number of elements P.

    parameters
    ----------
    prop : generic object 
        Python object to be verified.
    P : int
        Positive integer defining the desired numbe of element is prop.
    """
    if (type(P) != int) and (P <= 0):
        raise ValueError("P must be a positive integer")
    if type(prop) != np.ndarray:
        raise ValueError("prop must be a numpy array")
    if prop.ndim != 1:
        raise ValueError(
            "prop ndim ({}) ".format(prop.ndim) + "not equal to 1"
        )
    if prop.size != P:
        raise ValueError(
            "Number of elements in prop ({}) ".format(prop.size)
            + "mismatch P ({})".format(P)
        )


def vector_prop(prop, P):
    """
    Check if prop is a 2d numpy array having 3 columns and 
    a previously defined number of rows elements P.

    parameters
    ----------
    prop : generic object 
        Python object to be verified.
    P : int
        Positive integer defining the desired numbe of element is prop.
    """
    if (type(P) != int) and (P <= 0):
        raise ValueError("P must be a positive integer")
    if type(prop) != np.ndarray:
        raise ValueError("prop must be a numpy array")
    if prop.ndim != 2:
        raise ValueError(
            "prop ndim ({}) ".format(prop.ndim)
            + "not equal to 2"
        )
    if prop.shape[1] != 3:
        raise ValueError(
            "prop ndim ({}) ".format(prop.shape[1])
            + "not equal to 3"
        )
    if prop.shape[0] != P:
        raise ValueError(
            "Number of elements in prop ({}) ".format(
                prop.size
            )
            + "mismatch P ({})".format(P)
        )


def wavenumbers(kx, ky, kz):
    """
    Check if kx, ky and kz are 2d numpy array having the same shape 
    and specific properties (See function 'gravmag.filters.wavenumbers').

    parameters
    ----------
    kx, ky, kz: generic objects
        Python objects to be verified.
    """
    if type(kx) != np.ndarray:
        raise ValueError("kx must be a numpy array")
    if type(ky) != np.ndarray:
        raise ValueError("ky must be a numpy array")
    if type(kz) != np.ndarray:
        raise ValueError("kz must be a numpy array")
    if kx.ndim != 2:
        raise ValueError("kx must be a matrix")
    if ky.ndim != 2:
        raise ValueError("ky must be a matrix")
    if kz.ndim != 2:
        raise ValueError("kz must be a matrix")
    common_shape = kx.shape
    if ky.shape != common_shape:
       raise ValueError("ky shape mismatch kx shape")
    if kz.shape != common_shape:
       raise ValueError("kz shape mismatch kx shape")
    if np.any(kx[0, :] != 0):
        raise ValueError("first line of kx must be 0")
    if np.any(ky[:, 0] != 0):
        raise ValueError("first column of ky must be 0")
    if np.any(kz < 0):
        raise ValueError("all elements of kz must be positive or zero")


def scalar(x, positive=True):
    """
    Check if x is a float or int.

    parameters
    ----------
    x : generic object
        Python object to be verified.
    positive : boolean
        If True, impose that x must be positive.
    """

    if (type(x) not in [float, int]):
        raise ValueError(
            "x must be a float or int"
        )

    if positive == True:
        if x <= 0:
            raise ValueError(
                "x must be positive"
            )


def sensibility_matrix_and_data(G, data):
    """
    Check if G and data are consistent numpy arrays.

    parameters
    ----------
    G , data : generic objects
        Python objects to be verified.
    """
    if type(G) != np.ndarray:
        raise ValueError("G must be a numpy array")
    if type(data) != np.ndarray:
        raise ValueError("data must be a numpy array")
    if G.ndim != 2:
        raise ValueError("G must be a matrix")
    if data.ndim != 1:
        raise ValueError("data must be a vector")
    if G.shape[0] != data.size:
        raise ValueError("Sensibility matrix rows mismatch data size")