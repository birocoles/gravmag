import numpy as np


def are_rectangular_prisms(prisms):
    """
    Check if prisms is a dictionary containing the x, y and z coordinates of the
    corners of each prism in prisms.
    The corners south (x1), north (x2), west (y1), east (y2), top (z1) and bottom (z2) of each
    prism are arranged in the keys 'x1', 'x2', 'y1', 'y2', 'z1' and 'z2', respectively.
    All keys must be numpy arrays 1d having the same number of elements.

    parameters
    ----------
    prisms : generic object
        Python object to be verified.

    returns
    -------
    P : int
        Total number of prisms.
    """
    if type(prisms) != dict:
        raise ValueError("prisms must be a dictionary")
    if list(prisms.keys()) != ["x1", "x2", "y1", "y2", "z1", "z2"]:
        raise ValueError(
            "prisms must have the following 6 keys: 'x1', 'x2', 'y1', 'y2', 'z1', 'z2'"
        )
    for key in prisms.keys():
        if type(prisms[key]) != np.ndarray:
            raise ValueError("all keys in prisms must be numpy arrays")
    for key in prisms.keys():
        if prisms[key].ndim != 1:
            raise ValueError("all keys in prisms must be a numpy array 1d")
    P = prisms["x1"].size
    for key in prisms.keys():
        if prisms[key].size != P:
            raise ValueError(
                "all keys in prisms must have the same number of elements"
            )
    # check the x lower and upper limits
    if np.any(prisms["x2"] <= prisms["x1"]):
        raise ValueError("all 'x2' values must be greater than 'x1' values.")
    # check the y lower and upper limits
    if np.any(prisms["y2"] <= prisms["y1"]):
        raise ValueError("all 'y2' values must be greater than 'y1' values.")
    # check the z lower and upper limits
    if np.any(prisms["z2"] <= prisms["z1"]):
        raise ValueError("all 'z2' values must be greater than 'z1' values.")

    return P


def are_coordinates(coordinates):
    """
    Check if coordinates is a dictionary containing the x, y and z
    coordinates at the keys 'x', 'y' and 'z', respectively. All keys
    must be numpy arrays 1d having the same number of elements.

    parameters
    ----------
    coordinates : generic object
        Python object to be verified.

    returns
    -------
    D : int
        Total number of points.
    """
    if type(coordinates) != dict:
        raise ValueError("coordinates must be a dictionary")
    if list(coordinates.keys()) != ["x", "y", "z"]:
        raise ValueError(
            "coordinates must have the following 3 keys: 'x', 'y', 'z'"
        )
    for key in coordinates.keys():
        if type(coordinates[key]) != np.ndarray:
            raise ValueError("all keys in coordinates must be numpy arrays")
    for key in coordinates.keys():
        if coordinates[key].ndim != 1:
            raise ValueError("all keys in coordinates must be a numpy array 1d")
    D = coordinates["x"].size
    if (coordinates["y"].size != D) or (coordinates["z"].size != D):
        raise ValueError(
            "all keys in coordinates must have the same number of elements"
        )

    return D


def is_regular_grid_xy(coordinates):
    """
    Check if coordinates is a dictionary containing the x, y and z
    coordinates at the keys 'x', 'y' and 'z', respectively, and a key 'ordering'
    defining how the points are ordered after the first point (min x, min y).
    If 'ordering' = 'xy', the points vary first along x and then along y.
    If 'ordering' = 'yx', the points vary first along y and then along x.
    Key 'x' must be a numpy array 2d with a single column, i.e., with shape = (N, 1),
    where Nx is the number of data along x-axis.
    Key 'y' must be a numpy array 1d with shape = (Ny, ), where Ny is the number of
    data long y-axis.
    Key 'z' must be a scalar (float or int) defining the constant vertical coordinate of the grid.

    parameters
    ----------
    coordinates : generic object
        Python object to be verified.

    returns
    -------
    D : int
        Total number of points forming the grid.
    """
    if type(coordinates) != dict:
        raise ValueError("coordinates must be a dictionary")
    if list(coordinates.keys()) != ["x", "y", "z", "ordering", "area", "shape"]:
        raise ValueError(
            "coordinates must have the following 6 keys: 'x', 'y', 'z', 'ordering', 'area', 'shape'"
        )
    for key in ["x", "y"]:
        if type(coordinates[key]) != np.ndarray:
            raise ValueError(
                "'x' and 'y' keys in coordinates must be numpy arrays"
            )
    if coordinates["x"].ndim != 2:
        raise ValueError("'x' key must have ndim = 2")
    if coordinates["x"].shape[1] != 1:
        raise ValueError("'x' key must have shape[1] = 1")
    if coordinates["y"].ndim != 1:
        raise ValueError("'y' key must have ndim = 1")
    is_scalar(coordinates["z"], positive=False)
    is_ordering(coordinates["ordering"])
    is_shape(coordinates["shape"])
    is_area(coordinates["area"])
    if (coordinates["x"].size, coordinates["y"].size) != coordinates["shape"]:
        raise ValueError("number of elements in 'x' and 'y' keys must must be consistent with shape key")
    D = (coordinates["x"].size) * (coordinates["y"].size)

    return D


def is_regular_grid_wavenumbers(wavenumbers):
    """
    Check if wavenumbers is a dictionary containing the x, y and z
    wavenumbers at the keys 'x', 'y' and 'z', respectively, and the keys 
    'shape,' 'spacing', 'ordering'. See docstring of function 
    'data_structures.regular_grid_wavenumbers'.

    parameters
    ----------
    wavenumbers : dictionaty
        Dictionary containing metadata associated with a wavenumbers grid.

    """
    if type(wavenumbers) != dict:
        raise ValueError("wavenumbers must be a dictionary")
    if list(wavenumbers.keys()) != ["x", "y", "z", "ordering", "shape", "spacing"]:
        raise ValueError(
            "wavenumbers must have the following 6 keys: 'x', 'y', 'z', 'ordering', 'shape', 'spacing'"
        )
    for key in ["x", "y", "z"]:
        if type(wavenumbers[key]) != np.ndarray:
            raise ValueError(
                "'x', 'y' and 'z' keys of wavenumbers must be numpy arrays"
            )
    if wavenumbers["x"].ndim != 2:
        raise ValueError("'x' key must have ndim = 2")
    if wavenumbers["x"].shape[1] != 1:
        raise ValueError("'x' key must have shape[1] = 1")
    if wavenumbers["y"].ndim != 1:
        raise ValueError("'y' key must have ndim = 1")
    if wavenumbers["z"].ndim != 2:
        raise ValueError("'z' key must have ndim = 2")
    if wavenumbers["z"].shape != (wavenumbers["x"].size, wavenumbers["y"].size):
        raise ValueError("shape of 'z' key must consistent with sizes of keys 'x' and 'y'")
    is_scalar(wavenumbers["z"], positive=True)
    is_shape(wavenumbers['shape'])
    is_spacing(wavenumbers['spacing'])
    is_ordering(wavenumbers["ordering"])
    if (wavenumbers["x"].size, wavenumbers["y"].size) != wavenumbers["shape"]:
        raise ValueError("number of elements in 'x' and 'y' keys must must be consistent with shape key")


def is_scalar(x, positive=True):
    """
    Check if x is a float or int.

    parameters
    ----------
    x : generic object
        Python object to be verified.
    positive : boolean
        If True, impose that x must be positive.
    """
    if isinstance(x, (float, int)) is False:
        raise ValueError("x must be in float or int")
    if positive == True:
        if x <= 0:
            raise ValueError("x must be positive")


def is_integer(x, positive=True):
    """
    Check if x is an int.

    parameters
    ----------
    x : generic object
        Python object to be verified.
    positive : boolean
        If True, impose that x must be positive.
    """
    if isinstance(x, int) is False:
        raise ValueError("x must be an int")
    if positive == True:
        if x <= 0:
            raise ValueError("x must be positive")


def is_array(x, ndim=None, shape=None):
    """
    Check if x is a numpy array having specific ndim and shape.

    parameters
    ----------
    prop : generic object
        Python object to be verified.
    ndim : int
        Positive integer defining the dimension of x.
        If None, ndim is ignored. Default is None.
    shape : tuple
        Tuple defining the shape of x.
        If None, shape is ignored. Default is None.
    """
    if type(x) != np.ndarray:
        raise ValueError("x must be a numpy array")
    if ndim != None:
        if (type(ndim) != int) and (ndim <= 0):
            raise ValueError("'ndim' must be a positive integer")
        if x.ndim != ndim:
            raise ValueError(
                "x.ndim ({}) ".format(x.ndim)
                + "not equal to the predefined ndim {}".format(ndim)
            )
    if shape != None:
        if (type(shape) != tuple) or (len(shape) != ndim):
            raise ValueError("'shape' must be a tuple of 'ndim' elements")
        for item in shape:
            if (type(item) != int) and (item <= 0):
                raise ValueError("'shape' must be formed by positive integers")
        if x.shape != shape:
            raise ValueError(
                "x.shape ({}) ".format(x.shape)
                + "not equal to the predefined shape {}".format(shape)
            )


def is_area(area):
    """
    Check if area is a list containing min x, max x, min y and max y 
    coordinates of a given region.

    parameters
    ----------
    area : generic object
        Python object to be verified.
    """
    if type(area) != list:
        raise ValueError("'area' must be a list")
    if len(area) != 4:
        raise ValueError("'area' must have 4 elements")
    if (area[0] >= area[1]) or (area[2] >= area[3]):
        raise ValueError("'area[0]' must be smaller than 'area[1]' and 'area[2]' must be smaller than 'area[3]'")


def is_shape(shape):
    '''
    Check is shape is a tuple containing two positive integers.

    parameters
    ----------
    shape : generic object
        Python object to be verified.
    '''
    if type(shape) != tuple:
        raise ValueError("'shape' must be a tuple")
    if len(shape) != 2:
        raise ValueError("'shape' must have 2 elements")
    is_integer(x=shape[0], positive=True)
    is_integer(x=shape[1], positive=True)


def is_spacing(spacing):
    '''
    Check is spacing is a tuple containing two positive scalar.

    parameters
    ----------
    spacing : generic object
        Python object to be verified.
    '''
    if type(spacing) != tuple:
        raise ValueError("spacing must be a tuple")
    if len(spacing) != 2:
        raise ValueError("spacing must have 2 elements")
    is_scalar(x=spacing[0], positive=True)
    is_scalar(x=spacing[1], positive=True)


def is_ordering(ordering):
    '''
    Check if ordering is a string 'xy' or 'yx'.

    parameters
    ----------
    ordering : generic object
        Python object to be verified.
    '''
    if type(ordering) != str:
        raise ValueError("ordering must be a string")
    if len(ordering) != 2:
        raise ValueError("ordering must have 2 elements")
    if ordering not in ["xy", "yx"]:
        raise ValueError("invalid ordering {}".format(ordering))


def sensitivity_matrix_and_data(matrix, data):
    """
    Check if the given matrix and data are formed by consistent numpy arrays.

    parameters
    ----------
    matrix , vector : generic objects
        Lists of Python objects to be verified.
    """
    if type(matrix) != np.ndarray:
        raise ValueError("matrix must be a numpy array")
    if type(data) != np.ndarray:
        raise ValueError("data must be a numpy array")
    if matrix.ndim != 2:
        raise ValueError("matrix must be a matrix")
    if data.ndim != 1:
        raise ValueError("data must be a vector")
    if matrix.shape[0] != data.size:
        raise ValueError("matrix rows mismatch data size")


def Toeplitz_metadata(Toeplitz):
    """
    All information to generate the circulant Circulant matrix C which embbeds a Toeplitz matrix T.

    The Toeplitz matrix T has P x P elements. The embedding circulant matrix C has 2P x 2P elements.

    Matrix T is represented as follows:

        |t11 t12 ... t1P|
        |t21            |
    T = |.              | .
        |:              |
        |tP1            |


    We consider that matrix T may have three symmetry types:
    * gene - it denotes 'generic' and it means that there is no symmetry.
    * symm - it denotes 'symmetric' and it means that there is a perfect symmetry.
    * skew - it denotes 'skew-symmetric' and it means that the elements above the main diagonal
        have opposite signal with respect to those below the main diagonal.

    parameters
    ----------
    Toeplitz : dictionary containing the following keys
        symmetry : string
            Defines the type of symmetry between elements above and below the main diagonal.
            It can be 'gene', 'symm' or 'skew' (see the explanation above).
        column : numpy array 1D
            First column of T.
        row : None or numpy array 1D
            If not None, it is the first row of T, without the diagonal element. In
            this case, T does not have the assumed symmetries (see the text above).
            If None, matrix T is symmetric or skew-symmetric. Default is None.
    """
    if type(Toeplitz) != dict:
        raise ValueError("'Toeplitz' must be a dictionary")
    if list(Toeplitz.keys()) != [
        "symmetry",
        "column",
        "row",
    ]:
        raise ValueError(
            "'Toeplitz' must have the following keys: 'symmetry', 'column', 'row'"
        )

    # get the parameters defining the Toeplitz matrix
    symmetry = Toeplitz["symmetry"]
    column = Toeplitz["column"]
    row = Toeplitz["row"]
    if symmetry not in ["symm", "skew", "gene"]:
        raise ValueError("invalid {} symmetry".format(symmetry))
    is_array(x=column, ndim=1)
    if symmetry == "gene":
        is_array(x=row, ndim=1, shape=(column.size - 1,))
    else:  # symmetry in ["symm", "skew"]
        if row is not None:
            raise ValueError(
                "symmetry {} requires row to be None".format(symmetry)
            )


def BTTB_metadata(BTTB):
    """
    All information required to generate a Block Toeplitz formed by Toeplitz Blocks (BTTB)
    matrix T from the first columns and first rows of its non-repeating blocks.

    The matrix T has nblocks x nblocks blocks, each one with npoints_per_block x npoints_per_block elements.

    The first column and row of blocks forming the BTTB matrix T are
    represented as follows:

        |T11 T12 ... T1Q|
        |T21            |
    T = |.              | .
        |:              |
        |TQ1            |


    There are two symmetries:
    * symmetry_structure - between all blocks above and below the main block diagonal.
    * symmetry_blocks    - between all elements above and below the main diagonal within each block.
    Each symmetry pattern have three possible types:
    * gene - it denotes 'generic' and it means that there is no symmetry.
    * symm - it denotes 'symmetric' and it means that there is a perfect symmetry.
    * skew - it denotes 'skew-symmetric' and it means that the elements above the main diagonal
        have opposite signal with respect to those below the main diagonal.
    Hence, we consider that the BTTB matrix T has nine possible symmetry patterns:
    * 'symm-symm' - Symmetric Block Toeplitz formed by Symmetric Toeplitz Blocks
    * 'symm-skew' - Symmetric Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
    * 'symm-gene' - Symmetric Block Toeplitz formed by Generic Toeplitz Blocks
    * 'skew-symm' - Skew-Symmetric Block Toeplitz formed by Symmetric Toeplitz Blocks
    * 'skew-skew' - Skew-Symmetric Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
    * 'skew-gene' - Skew-Symmetric Block Toeplitz formed by Generic Toeplitz Blocks
    * 'gene-symm' - Generic Block Toeplitz formed by Symmetric Toeplitz Blocks
    * 'gene-skew' - Generic Block Toeplitz formed by Skew-Symmetric Toeplitz Blocks
    * 'gene-gene' - Generic Block Toeplitz formed by Generic Toeplitz Blocks

    parameters
    ----------
    BTTB : dictionary containing the following keys:
        symmetry_structure : string
            Defines the type of symmetry between all blocks above and below the main block diagonal.
            It can be 'gene', 'symm' or 'skew' (see the explanation above).
        symmetry_blocks : string
            Defines the type of symmetry between elements above and below the main diagonal within all blocks.
            It can be 'gene', 'symm' or 'skew' (see the explanation above).
        nblocks : int
            Number of blocks (nblocks) of T along column and row.
        columns : numpy array
            Matrix whose rows are the first columns cij of the non-repeating blocks
            Tij of T. They must be ordered as follows: c11, c21, ..., cQ1,
            c12, ..., c1Q.
        rows : None or numpy array 2D
            If not None, it is a matrix whose rows are the first rows rij of the
            non-repeating blocks Tij of T, without the diagonal term. They must
            be ordered as follows: r11, r21, ..., rM1, r12, ..., r1M.
    """
    if type(BTTB) != dict:
        raise ValueError("'BTTB' must be a dictionary")
    if list(BTTB.keys()) != [
        "symmetry_structure",
        "symmetry_blocks",
        "nblocks",
        "columns",
        "rows",
    ]:
        raise ValueError(
            "'Toeplitz' must have the following keys: 'symmetry_structure', 'symmetry_blocks', 'nblocks', 'columns', 'rows'"
        )

    # get the parameters defining the BTTB matrix
    symmetry_structure = BTTB["symmetry_structure"]
    symmetry_blocks = BTTB["symmetry_blocks"]
    nblocks = BTTB["nblocks"]
    columns = BTTB["columns"]
    rows = BTTB["rows"]

    if symmetry_structure not in ["symm", "skew", "gene"]:
        raise ValueError("invalid {} symmetry".format(symmetry_structure))
    if symmetry_blocks not in ["symm", "skew", "gene"]:
        raise ValueError("invalid {} symmetry".format(symmetry_blocks))
    is_integer(x=nblocks, positive=True)
    is_array(x=columns, ndim=2)

    # verify symmetry between blocks
    if (symmetry_structure == "symm") or (symmetry_structure == "skew"):
        # check consistency between 'symmetry_structure', 'nblocks' and number of rows in 'columns'
        if columns.shape[0] != nblocks:
            raise ValueError(
                "'symmetry_structure' ({}) requires the number of rows in 'columns' ({}) to be equal to 'nblocks' ({})".format(
                    symmetry_structure, columns.shape[0], nblocks
                )
            )
    else:  # symmetry_structure == 'gene'
        # check consistency between 'symmetry_structure', 'nblocks' and number of rows in 'columns'
        if columns.shape[0] != (2 * nblocks - 1):
            raise ValueError(
                "'symmetry_structure' ({}) requires the number of rows in 'columns' ({}) to be equal to 2*'nblocks'-1 ({})".format(
                    symmetry_structure, columns.shape[0], 2 * nblocks - 1
                )
            )

    # verify symmetry between elements within blocks
    if (symmetry_blocks == "symm") or (symmetry_blocks == "skew"):
        # check consistency between 'symmetry_blocks', 'nblocks' and number of rows in 'rows'
        if rows is not None:
            raise ValueError(
                "'symmetry_blocks' ({}) requires 'rows' to be None".format(
                    symmetry_blocks
                )
            )
    else:  # symmetry_blocks == 'gene'
        # check consistency between 'symmetry_blocks', 'nblocks' and number of rows in 'rows'
        is_array(x=rows, ndim=2)
        if rows.shape[0] != columns.shape[0]:
            raise ValueError(
                "'symmetry_blocks' ({}) requires number of rows in 'rows' ({}) to be equal to that in 'columns' ({})".format(
                    symmetry_blocks, rows.shape[0], columns.shape[0]
                )
            )
        if rows.shape[1] != (columns.shape[1] - 1):
            raise ValueError(
                "'symmetry_blocks' ({}) requires number of columns in 'rows' ({}) to be equal to that in 'columns' minus 1 ({})".format(
                    symmetry_blocks, rows.shape[1], columns.shape[1] - 1
                )
            )
