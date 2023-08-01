import numpy as np
from scipy.spatial import distance
import warnings
from . import inverse_distance as idist
from . import check, utils, constants, convolve


# def kernel_grav(data_points, source_points, fields=["z"], check_input=True):
#     """
#     Compute the kernel matrix produced by a planar layer of monopoles.

#     parameters
#     ----------
#     data_points: dictionary
#         Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
#         respectively. Each key is a numpy array 1d having the same number of elements.
#     source_points: dictionary
#         Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
#         respectively. Each key is a numpy array 1d having the same number of elements.
#     field : list of strings
#         Defines the fields produced by the layer. The available options are:
#         "potential", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz".
#     check_input : boolean
#         If True, verify if the input is valid. Default is True.

#     returns
#     -------
#     G: list of numpy array 2d
#         List of N x M matrices defined by the kernels of the equivalent layer integral.
#     """

#     if check_input is True:
#         check.are_coordinates(data_points)
#         check.are_coordinates(source_points)
#         if np.max(data_points['z']) >= np.min(source_points['z']):
#             warnings.warn("verify if the surface containing data cross the equivalent layer")
#         # check if field is valid
#         for field in fields:
#             if field not in [
#                 "potential",
#                 "x",
#                 "y",
#                 "z",
#                 "xx",
#                 "xy",
#                 "xz",
#                 "yy",
#                 "yz",
#                 "zz",
#             ]:
#                 raise ValueError("invalid field {}".format(field))

#     # compute Squared Euclidean Distance Matrix (SEDM)
#     R2 = idist.sedm(data_points, source_points, check_input=False)

#     # compute the kernel matrices according to "fields"
#     G = []
#     for field in fields:
#         if field == "potential":
#             G.append(1.0 / np.sqrt(R2))
#         elif field in ["x", "y", "z"]:
#             G.append(idist.grad(data_points, source_points, R2, [field], False)[0])
#         else:  # field is in ["xx", "xy", "xz", "yy", "yz", "zz"]
#             G.append(idist.grad_tensor(data_points, source_points, R2, [field], False)[0])

#     return G


def kernel_matrix_dipoles(
    data_points,
    source_points,
    inc,
    dec,
    field="z",
    inct=None,
    dect=None,
    check_input=True,
):
    """
    Compute the kernel matrix produced by a planar layer of dipoles with
    constant magnetization direction.

    parameters
    ----------
    data_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    source_points: dictionary
        Dictionary containing the x, y and z coordinates at the keys 'x', 'y' and 'z',
        respectively. Each key is a numpy array 1d having the same number of elements.
    inc, dec : ints or floats
        Scalars defining the constant inclination and declination of the
        dipoles magnetization.
    field : string
        Defines the field produced by the layer. The available options are:
            - "potential" : magnetic scalar potential
            - "x", "y", "z" : Cartesian components of the magnetic induction
            - "t" : Component of magnetic induction along a constant direction
                with inclination and declination defined by "inct" and "dect",
                respectively. It approximates the total-field anomaly when
                "inct" and "dect" define the constant inclination and
                declination of the main field at the study area.
    inct, dect : ints or floats
        Scalars defining the constant inclination and declination of the
        magnetic field component. They must be not None if "field" is "t".
        Otherwise, they are ignored.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    G: numpy array 2d
        N x M matrix defined by the kernel of the equivalent layer integral.
    """

    if check_input is True:
        check.are_coordinates(data_points)
        check.are_coordinates(source_points)
        if np.max(data_points["z"]) >= np.min(source_points["z"]):
            warnings.warn(
                "verify if the surface containing data cross the equivalent layer"
            )
        if type(inc) not in [float, int]:
            raise ValueError("inc must be a scalar")
        if type(dec) not in [float, int]:
            raise ValueError("dec must be a scalar")
        # check if field is valid
        if field not in ["potential", "x", "y", "z", "t"]:
            raise ValueError("invalid field {}".format(field))
        if field == "t":
            if type(inct) not in [float, int]:
                raise ValueError("inct must be a scalar because field is 't'")
            if type(dect) not in [float, int]:
                raise ValueError("dect must be a scalar because field is 't'")

    # compute Squared Euclidean Distance Matrix (SEDM)
    R2 = idist.sedm(data_points, source_points, check_input=False)
    # compute the unit vector defined by inc and dec
    u = utils.unit_vector(inc, dec, check_input=False)

    # compute the kernel matrix according to "field"
    if field == "potential":
        Gx, Gy, Gz = idist.grad(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["x", "y", "z"],
            check_input=False,
        )
        G = -(u[0] * Gx + u[1] * Gy + u[2] * Gz)
    elif field == "x":
        Gxx, Gxy, Gxz = idist.grad_tensor(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["xx", "xy", "xz"],
            check_input=False,
        )
        G = u[0] * Gxx + u[1] * Gxy + u[2] * Gxz
    elif field == "y":
        Gxy, Gyy, Gyz = idist.grad_tensor(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["xy", "yy", "yz"],
            check_input=False,
        )
        G = u[0] * Gxy + u[1] * Gyy + u[2] * Gyz
    elif field == "z":
        Gxz, Gyz, Gzz = idist.grad_tensor(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["xz", "yz", "zz"],
            check_input=False,
        )
        G = u[0] * Gxz + u[1] * Gyz + u[2] * Gzz
    else:  # field is "t"
        # compute the unit vector defined by inct and dect
        t = utils.unit_vector(inct, dect, check_input=False)
        Gxx, Gxy, Gxz, Gyy, Gyz = idist.grad_tensor(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["xx", "xy", "xz", "yy", "yz"],
            check_input=False,
        )
        axx = u[0] * t[0] - u[2] * t[2]
        axy = u[0] * t[1] + u[1] * t[0]
        axz = u[0] * t[2] + u[2] * t[0]
        ayy = u[1] * t[1] - u[2] * t[2]
        ayz = u[1] * t[2] + u[2] * t[1]
        G = axx * Gxx + axy * Gxy + axz * Gxz + ayy * Gyy + ayz * Gyz

    return G


def method_CGLS(
    sensibility_matrices, data_vectors, epsilon, ITMAX=50, check_input=True
):
    """
    Solves the unconstrained overdetermined problem to estimate the physical-property
    distribution on the equivalent layer via conjugate gradient normal equation residual
    (CGNR) (Golub and Van Loan, 2013, sec. 11.3) or conjugate gradient least squares (CGLS)
    (Aster et al., 2019, p. 165) method.

    parameters
    ----------
    sensibility_matrices: list of numpy arrays 2d
        List of matrices with same number of columns defining the kernel of the equivalent layer integral.
    data_vectors : list of numpy arrays 1d
        List of potential-field data.
    epsilon : float
        Tolerance for evaluating convergence criterion.
    ITMAX : int
        Maximum number of iterations. Default is 50.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    deltas : list of floats
        List of ratios of Euclidean norm of the residuals and number of data.
    parameters : numpy array 1d
        Physical property distribution on the equivalent layer.
    """

    if check_input == True:
        # check if G and data are consistent numpy arrays
        check.sensibility_matrix_and_data(
            matrices=sensibility_matrices, vectors=data_vectors
        )
        # check if epsilon is a positive scalar
        check.is_scalar(x=epsilon, positive=True)
        # check if ITMAX is a positive integer
        check.is_integer(x=ITMAX, positive=True)

    # get number of data for each dataset and initialize residuals list
    number_of_data = []
    residuals = []
    for data in data_vectors:
        number_of_data.append(data.size)
        residuals.append(np.copy(data))

    # compute the first delta and initialize the deltas list
    deltas = []
    delta = 0.0
    for res in residuals:
        delta += np.sum(res * res)
    delta = np.sqrt(delta) / np.sum(number_of_data)
    deltas.append(delta)

    # initialize the parameter vector
    parameters = np.zeros(sensibility_matrices[0].shape[1])

    # initialize auxiliary variables
    vartheta = np.zeros_like(parameters)
    for sensibility_matrix, res in zip(sensibility_matrices, residuals):
        vartheta[:] += sensibility_matrix.T @ res
    rho0 = np.sum(vartheta * vartheta)
    tau = 0.
    eta = np.zeros_like(parameters)
    nus = []
    for ndata in number_of_data:
        nus.append(np.zeros_like(parameters))
    m = 1

    # updates
    while (delta > epsilon) and (m < ITMAX):
        print(residuals[0])
        eta[:] = vartheta + tau * eta
        aux = 0.0
        for sensibility_matrix, nu in zip(sensibility_matrices, nus):
            nu[:] = sensibility_matrix @ eta
            aux += np.sum(nu * nu)
        upsilon = rho0 / aux
        parameters[:] += upsilon * eta
        delta = 0.0
        for res, nu in zip(residuals, nus):
            res[:] -= upsilon * nu
            delta += np.sum(res * res)
        delta = np.sqrt(delta) / np.sum(number_of_data)
        deltas.append(delta)
        vartheta[:] = 0.0  # remember that vartheta in an array like parameters
        for sensibility_matrix, res in zip(sensibility_matrices, residuals):
            vartheta[:] += sensibility_matrix.T @ res
        rho = np.sum(vartheta * vartheta)
        tau = rho / rho0
        rho0 = rho
        m += 1

    return deltas, parameters


def method_column_action_C92(
    G, data, data_points, zlayer, scale, epsilon, ITMAX, check_input=True
):
    """
    Estimates the physical-property distribution on the equivalent layer via column-action approach proposed by Cordell (1992).

    parameters
    ----------
    G: numpy array 2d
        N x M matrix defined by the kernel of the equivalent layer integral.
    data : numpy array 1d
        Potential-field data.
    data_points : numpy array 2d
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N data points. The ith column contains the
        coordinates of the ith data point.
    zlayer : float
        Constant defining the vertical position for all equivalent sources.
    epsilon : float
        Tolerance for evaluating convergence criterion.
    ITMAX : int
        Maximum number of iterations. Default is 50.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    rmax_list : list of floats
        List of maximum absolute residuals.
    parameters : numpy array 1d
        Physical property distribution on the equivalent layer.
    """

    if check_input == True:
        # check if data and G are consistent numpy arrays
        check.sensibility_matrix_and_data(G=G, data=data)
        # check data points
        check.are_coordinates(coordinates=data_points)
        # check if zlayer result in a layer below the data points
        if (type(zlayer) != float) and (np.any(zlayer <= data_points[2])):
            raise ValueError(
                "zlayer must be a scalar greater than the z coordinate of all data points"
            )
        # check if epsilon is a positive scalar
        check.is_scalar(x=epsilon, positive=True)
        # check if ITMAX is a positive integer
        check.is_integer(x=ITMAX, positive=True)

        # initializations
        residuals = np.copy(data)
        parameters = data * scale
        imax = np.argmax(np.abs(residuals))
        rmax = residuals[imax]
        rmax_list = []
        rmax_list.append(rmax)
        m = 1
        # updates
        while (rmax > epsilon) and (m < ITMAX):
            xmax, ymax, zmax = data_points[:, imax]
            parameters[imax] += rmax * scale * np.abs(zlayer - zmax)
            residuals -= G[:, imax] * rmax
            imax = np.argmax(np.abs(residuals))
            rmax = residuals[imax]
            rmax_list.append(rmax)
            m += 1

        return rmax_list, parameters


def method_iterative_SOB17(
    G, data, factor, epsilon, ITMAX=50, check_input=True
):
    """
    Solves the unconstrained problem to estimate the physical-property
    distribution on the equivalent layer via iterative method.

    parameters
    ----------
    G: numpy array 2d
        N x M matrix defined by the kernel of the equivalent layer integral.
    data : numpy array 1d
        Potential-field data.
    factor : float
        Positive scalar controlling the convergence.
    epsilon : float
        Tolerance for evaluating convergence criterion.
    ITMAX : int
        Maximum number of iterations. Default is 50.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    delta_list : list of floats
        List of ratios of Euclidean norm of the residuals and number of data.
    parameters : numpy array 1d
        Physical property distribution on the equivalent layer.
    """

    if check_input == True:
        # check if data and G are consistent numpy arrays
        check.sensibility_matrix_and_data(G=G, data=data)
        # check if factor and epsilon are positive scalars
        check.is_scalar(x=factor, positive=True)
        check.is_scalar(x=epsilon, positive=True)
        # check if ITMAX is a positive integer
        check.is_integer(x=ITMAX, positive=True)

    # initializations
    D = data.size
    parameters = np.zeros_like(data)
    residuals = np.copy(data)
    delta_list = []
    delta = np.sqrt(np.sum(residuals * residuals)) / D
    delta_list.append(delta)
    m = 1
    # updates
    while (delta > epsilon) and (m < ITMAX):
        dp = factor * residuals
        parameters += dp
        nu = G @ dp
        residuals -= nu
        delta = np.sqrt(np.sum(residuals * residuals)) / D
        delta_list.append(delta)
        m += 1

    return delta_list, parameters
