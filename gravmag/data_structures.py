import numpy as np
from . import check


def regular_grid_xy(area, shape, z0, ordering="xy", check_input=True):
    """
    Define the data structure for a horizontal grid of points x and y.

    parameters
    ----------
    area : list
        List of min x, max x, min y and max y.
    shape : tuple
        Tuple defining the total number of points along x and y directions, respectively.
    z0 : scalar
        Constant vertical coordinate of the grid.
    ordering : string
        Defines how the points are ordered after the first point (min x, min y).
        If 'xy', the points vary first along x and then along y.
        If 'yx', the points vary first along y and then along x.
        Default is 'xy'.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    grid : dictionary containing the following keys
        'x' : numpy array 2d with a single column, i.e., with shape = (N, 1),
            where Nx is the number of data along x-axis.
        'y' : numpy array 1d with shape = (Ny, ), where Ny is the number of
            data long y-axis.
        'z' : scalar (float or int) defining the constant vertical coordinate 
            of the grid.
        'ordering' : string
            The input parameter 'ordering'
    """
    if check_input == True:
        if type(area) != list:
            raise ValueError("'area' must be a list")
        if len(area) != 4:
            raise ValueError("'area' must have 4 elements")
        if (area[0] >= area[1]) or (area[2] >= area[3]):
            raise ValueError("'area[0]' must be smaller than 'area[1]' and 'area[2]' must be smaller than 'area[3]'")
        if type(shape) != tuple:
            raise ValueError("'shape' must be a tuple")
        if len(shape) != 2:
            raise ValueError("'shape' must have 2 elements")
        check.is_integer(x=shape[0], positive=True)
        check.is_integer(x=shape[1], positive=True)
        check.is_scalar(x=z0)
        if ordering not in ["xy", "yx"]:
            raise ValueError("invalid ordering {}".format(ordering))

    grid = {
        'x' : np.linspace(area[0], area[1], shape[0])[:,np.newaxis],
        'y' : np.linspace(area[2], area[3], shape[1]),
        'z' : z0,
        'ordering' : ordering
    }

    return grid


def BTTB_transpose(BTTB, check_input=True):
    """
    Return the data structure for the transposed BTTB.

    parameters
    ----------
    BTTB : dictionary
        See the function 'check.BTTB_metadata'.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    BTTB_T : dictionary
        Data structure similar to the input 'BTTB', but for its transposed.
    """
    if check_input == True:
        check.BTTB_metadata(BTTB=BTTB)

    # get the columns of the BTTB matrix
    columns = np.copy(BTTB["columns"])
    if BTTB["rows"] is None:
        rows = None
    else: # BTTB["rows"] is a numpy array 2D
        rows = np.copy(rows)

    # modify the column
    if BTTB["symmetry_structure"] == "symm":
        if BTTB["symmetry_blocks"] == "symm":
            pass
        elif BTTB["symmetry_blocks"] == "skew":
            columns[:,1:] *= -1
        else: # BTTB["symmetry_blocks"] == "gene"

    elif BTTB["symmetry_structure"] == "skew":
        if BTTB["symmetry_blocks"] == "symm":
            columns[1:,:] *= -1
        elif BTTB["symmetry_blocks"] == "skew":
            columns[:,1:] *= -1
            columns[1:,:] *= -1
        else: # BTTB["symmetry_blocks"] == "gene"

    else: # BTTB["symmetry_structure"] == "gene"
        if BTTB["symmetry_blocks"] == "symm":
            
        elif BTTB["symmetry_blocks"] == "skew":
            
        else: # BTTB["symmetry_blocks"] == "gene"

    BTTB_T = {
        "symmetry_structure" : BTTB["symmetry_structure"],
        "symmetry_blocks" : BTTB["symmetry_blocks"],
        "nblocks": BTTB["nblocks"],
        "columns": columns,
        "rows": None,
    }

    return BTTB_T
