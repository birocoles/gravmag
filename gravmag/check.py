import numpy as np


def rectangular_prisms(prisms):
    """
    Check if rectangular prisms are well defined

    parameters
    ----------
    prisms : 2d-array
        Array containing the boundaries of the prisms in the following order:
        ``s``, ``n``, ``w``, ``e``, ``top``, ``bottom``.
        The array must have the following shape: (``n_prisms``, 6), where
        ``n_prisms`` is the total number of prisms.
        This array of prisms must have valid boundaries.
    """
    prisms = np.asarray(prisms)
    if prisms.ndim != 2:
        raise ValueError(
            "prisms ndim ({}) ".format(prisms.ndim) + "not equal to 2"
        )
    if prisms.shape[1] != 6:
        raise ValueError(
            "Number of columns in prisms ({}) ".format(prisms.shape[1])
            + "not equal to 6"
        )
    south, north, west, east, top, bottom = tuple(
        prisms[:, i] for i in range(6)
    )
    err_msg = "Invalid rectangular prism(s). "
    bad_sn = south > north
    bad_we = west > east
    bad_bt = top > bottom
    if bad_sn.any():
        err_msg += "The south boundary can't be greater than the north one.\n"
        for prism in prisms[bad_sn]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if bad_we.any():
        err_msg += "The west boundary can't be greater than the east one.\n"
        for prism in prisms[bad_we]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if bad_bt.any():
        err_msg += "The top boundary can't be greater than the bottom one.\n"
        for prism in prisms[bad_bt]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)


def coordinates(coordinates):
    """
    Check if coordinates are well defined

    parameters
    ----------
    coordinates : 2d-array
        2d-array containing x (first line), y (second line), and z (third line)
        of the computation points. All coordinates should be in meters.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2:
        raise ValueError(
            "coordinates ndim ({}) ".format(coordinates.ndim) + "not equal to 2"
        )
    if coordinates.shape[0] != 3:
        raise ValueError(
            "Number of lines in coordinates ({}) ".format(coordinates.shape[0])
            + "not equal to 3"
        )


def scalar_prop(prop, sources):
    """
    Check if sources scalar physical property are well defined.
    Check the ``sources`` before.

    parameters
    ----------
    prop : 1d-array
        1d-array containing the scalar physical property of each source.
    sources : 2d-array
        2d-array containing the coordinates of the sources.
        Each line must contain the coordinates of a single source.
        All coordinates should be in meters.
    """
    prop = np.asarray(prop)
    if prop.ndim != 1:
        raise ValueError(
            "prop ndim ({}) ".format(prop.ndim) + "not equal to 1"
        )
    if prop.size != sources.shape[0]:
        raise ValueError(
            "Number of elements in prop ({}) ".format(prop.size)
            + "mismatch the number of sources ({})".format(sources.shape[0])
        )


def vector_prop(prop, sources):
    """
    Check if sources vector physical property are well defined.
    Check the ``sources`` before.

    parameters
    ----------
    prop : 1d-array
        2d-array containing the vector physical property components of 
        the prisms. Each line must contain the x, y and z components of 
        the physical property of a single source. All values should be in A/m.
    sources : 2d-array
        2d-array containing the coordinates of the sources.
        Each line must contain the coordinates of a single source.
        All coordinates should be in meters.
    """
    prop = np.asarray(prop)
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
    if prop.shape[0] != sources.shape[0]:
        raise ValueError(
            "Number of elements in prop ({}) ".format(
                prop.size
            )
            + "mismatch the number of sources ({})".format(sources.shape[0])
        )


def wavenumbers(kx, ky, kz):
    """
    Check if wavenumbers are well defined.
    See function 'gravmag.filters.wavenumbers'.

    parameters
    ----------
    kx, ky, kz: numpy arrays 2D
        Wavenumbers in x, y and z directions computed according to
        function 'wavenumbers'.
    """
    # Convert the wavenumbers to arrays
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    kz = np.asarray(kz)

    assert kx.ndim == ky.ndim == kz.ndim == 2, "kx, ky and kz must be matrices"
    common_shape = kx.shape
    assert (
        ky.shape == kz.shape == common_shape
    ), "kx, ky and kz must have the same shape"
    assert np.all(kx[0, :] == 0), "first line of kx must be 0"
    assert np.all(ky[:, 0] == 0), "first column of ky must be 0"
    assert np.all(kz >= 0), "elements of kz must be >= 0"
