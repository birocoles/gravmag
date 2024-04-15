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
        check.is_scalar(x=z0, positive=False)
        if ordering not in ["xy", "yx"]:
            raise ValueError("invalid ordering {}".format(ordering))

    grid = {
        'x' : np.linspace(area[0], area[1], shape[0])[:,np.newaxis],
        'y' : np.linspace(area[2], area[3], shape[1]),
        'z' : z0,
        'ordering' : ordering
    }

    return grid


def BTTB_transposed_metadata(BTTB_metadata, check_input=True):
    """
    Return the data structure for the transposed BTTB.

    parameters
    ----------
    BTTB_metadata : dictionary
        See the function 'convolve.generic_BTTB'.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    BTTB_T : dictionary
        Data structure similar to the input of 'check.BTTB_metadata', but for its transposed.
    """
    if check_input == True:
        check.BTTB_metadata(BTTB=BTTB_metadata)

    # the transposed of a BTTB inherits the symmetry structure between blocks,
    # within blocks, number of blocks and also order of blocks.
    BTTB_T_metadata = {
        "symmetry_structure" : BTTB_metadata["symmetry_structure"],
        "symmetry_blocks" : BTTB_metadata["symmetry_blocks"],
        "nblocks": BTTB_metadata["nblocks"]
    }

    # get data and perform the required changes
    if BTTB_metadata["symmetry_structure"] == "symm":
        if BTTB_metadata["symmetry_blocks"] == "symm":
            BTTB_T_metadata["columns"] = np.copy(BTTB_metadata["columns"])
            BTTB_T_metadata["rows"] = None
        elif BTTB_metadata["symmetry_blocks"] == "skew":
            BTTB_T_metadata["columns"] = np.copy(BTTB_metadata["columns"])
            BTTB_T_metadata["columns"][:,1:] *= -1
            BTTB_T_metadata["rows"] = None
        else: # BTTB_metadata["symmetry_blocks"] == "gene"
            BTTB_T_metadata["columns"] = np.hstack([
                BTTB_metadata["columns"][:,0][:,np.newaxis], 
                BTTB_metadata["rows"]
                ])
            BTTB_T_metadata["rows"] = BTTB_metadata["columns"][:,1:]

    elif BTTB_metadata["symmetry_structure"] == "skew":
        if BTTB_metadata["symmetry_blocks"] == "symm":
            # get the elements forming the columns and rows
            BTTB_T_metadata["columns"] = np.copy(BTTB_metadata["columns"])
            BTTB_T_metadata["columns"][1:,:] *= -1
            BTTB_T_metadata["rows"] = None
        elif BTTB_metadata["symmetry_blocks"] == "skew":
            # get the elements forming the columns and rows
            BTTB_T_metadata["columns"] = np.copy(BTTB_metadata["columns"])
            BTTB_T_metadata["columns"][:,1:] *= -1
            BTTB_T_metadata["columns"][1:,:] *= -1
            BTTB_T_metadata["rows"] = None
        else: # BTTB_metadata["symmetry_blocks"] == "gene"
            # get the elements forming the columns and rows
            BTTB_T_metadata["columns"] = np.hstack(
                [
                BTTB_metadata["columns"][:,0][:,np.newaxis], 
                BTTB_metadata["rows"]
                ])
            BTTB_T_metadata["rows"] = BTTB_metadata["columns"][:,1:]
            # change signal
            BTTB_T_metadata["columns"][1:] *= -1
            BTTB_T_metadata["rows"][1:] *= -1
    else: # BTTB_metadata["symmetry_structure"] == "gene"
        if BTTB_metadata["symmetry_blocks"] == "symm":
            # get the elements forming the columns and rows
            BTTB_T_metadata["columns"] = np.copy(BTTB_metadata["columns"])
            BTTB_T_metadata["rows"] = None
            # get the number of blocks along a column/row
            nblocks = BTTB_T_metadata["nblocks"]
            # permute elements with respect to the main diagonal
            permutation_indices = [i for i in range(2*nblocks - 1)]
            (
                permutation_indices[1:nblocks], 
                permutation_indices[nblocks:]
            ) = (
                permutation_indices[nblocks:], 
                permutation_indices[1:nblocks]
            )
            BTTB_T_metadata["columns"] = BTTB_T_metadata["columns"][permutation_indices]
        elif BTTB_metadata["symmetry_blocks"] == "skew":
            # get the columns
            BTTB_T_metadata["columns"] = np.copy(BTTB_metadata["columns"])
            BTTB_T_metadata["rows"] = None
            # get the number of blocks along a column/row
            nblocks = BTTB_T_metadata["nblocks"]
            # permute the elements with respect to the main diagonal
            permutation_indices = [i for i in range(2*nblocks - 1)]
            (
                permutation_indices[1:nblocks], 
                permutation_indices[nblocks:]
            ) = (
                permutation_indices[nblocks:], 
                permutation_indices[1:nblocks]
            )
            BTTB_T_metadata["columns"] = BTTB_T_metadata["columns"][permutation_indices]
            # change signal
            BTTB_T_metadata["columns"][:,1:] *= -1
        else: # BTTB_metadata["symmetry_blocks"] == "gene"
            # get the elements forming the columns and rows
            BTTB_T_metadata["columns"] = np.hstack(
                [
                BTTB_metadata["columns"][:,0][:,np.newaxis], 
                BTTB_metadata["rows"]
                ])
            BTTB_T_metadata["rows"] = BTTB_metadata["columns"][:,1:]
            # get the number of blocks along a column/row
            nblocks = BTTB_T_metadata["nblocks"]
            # permute the elements with respect to the main diagonal
            permutation_indices = [i for i in range(2*nblocks - 1)]
            (
                permutation_indices[1:nblocks], 
                permutation_indices[nblocks:]
            ) = (
                permutation_indices[nblocks:], 
                permutation_indices[1:nblocks]
            )
            BTTB_T_metadata["columns"] = BTTB_T_metadata["columns"][permutation_indices]
            BTTB_T_metadata["rows"] = BTTB_T_metadata["rows"][permutation_indices]

    return BTTB_T_metadata