import numpy as np
from scipy.spatial import distance
from . import inverse_distance as id
from . import check, utils, constants


def kernel_matrix_monopoles(data_points, source_points, field="z", check_input=True):
    """
    Compute the kernel matrix produced by a planar layer of monopoles.

    parameters
    ----------
    data_points : numpy array 2d
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N data points. The ith column contains the
        coordinates of the ith data point.
    source_points: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.
    field : string
        Defines the field produced by the layer. The available options are:
        "potential", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz".
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    returns
    -------
    G: numpy array 2d
        N x M matrix defined by the kernel of the equivalent layer integral.
    """

    if check_input is True:
        check.coordinates(data_points)
        check.coordinates(source_points)
        if np.any(data_points[2] >= source_points[2]):
            raise ValueError("all data points must be above source points")
        # check if field is valid
        if field not in [
            "potential",
            "x",
            "y",
            "z",
            "xx",
            "xy",
            "xz",
            "yy",
            "yz",
            "zz",
        ]:
            raise ValueError("invalid field {}".format(field))

    # compute Squared Euclidean Distance Matrix (SEDM)
    R2 = id.sedm(data_points, source_points, check_input=False)

    # compute the kernel matrix according to "field"
    if field == "potential":
        G = 1.0 / np.sqrt(R2)
    elif field in ["x", "y", "z"]:
        G = id.grad(data_points, source_points, R2, [field], False)[0]
    else:  # field is in ["xx", "xy", "xz", "yy", "yz", "zz"]
        G = id.grad_tensor(data_points, source_points, R2, [field], False)[0]

    return G


def kernel_matrix_dipoles(
    data_points, source_points, inc, dec, field="t", inct=None, dect=None, check_input=True
):
    """
    Compute the kernel matrix produced by a planar layer of dipoles with
    constant magnetization direction.

    parameters
    ----------
    data_points : numpy array 2d
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N data points. The ith column contains the
        coordinates of the ith data point.
    source_points: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M sources. The jth column contains the coordinates of
        the jth source.
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
        check.coordinates(data_points)
        check.coordinates(source_points)
        if np.any(data_points[2] >= source_points[2]):
            raise ValueError("all data points must be above source points")
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
    R2 = id.sedm(data_points, source_points, check_input=False)
    # compute the unit vector defined by inc and dec
    u = utils.unit_vector(inc, dec, check_input=False)

    # compute the kernel matrix according to "field"
    if field == "potential":
        Gx, Gy, Gz = id.grad(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["x", "y", "z"],
            check_input=False,
        )
        G = -(u[0] * Gx + u[1] * Gy + u[2] * Gz)
    elif field == "x":
        Gxx, Gxy, Gxz = id.grad_tensor(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["xx", "xy", "xz"],
            check_input=False,
        )
        G = u[0] * Gxx + u[1] * Gxy + u[2] * Gxz
    elif field == "y":
        Gxy, Gyy, Gyz = id.grad_tensor(
            data_points=data_points,
            source_points=source_points,
            SEDM=R2,
            components=["xy", "yy", "yz"],
            check_input=False,
        )
        G = u[0] * Gxy + u[1] * Gyy + u[2] * Gyz
    elif field == "z":
        Gxz, Gyz, Gzz = id.grad_tensor(
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
        Gxx, Gxy, Gxz, Gyy, Gyz = id.grad_tensor(
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


def method_CGLS(G, data, epsilon, ITMAX=50, check_input=True):
    """
    Solves the unconstrained overdetermined problem to estimate the physical-property
    distribution on the equivalent layer via conjugate gradient normal equation residual 
    (CGNR) (Golub and Van Loan, 2013, sec. 11.3) or conjugate gradient least squares (CGLS) 
    (Aster et al., 2019, p. 165) method.

    parameters
    ----------
    G: numpy array 2d
        N x M matrix defined by the kernel of the equivalent layer integral.
    data : numpy array 1d
        Potential-field data.
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
    
    if check_input is True:
        # check if G and data are consistent numpy arrays
        check.sensibility_matrix_and_data(G=G, data=data)
        # check if epsilon is a positive scalar
        check.scalar(x=epsilon, positive=True)
        # check if ITMAX is a positive integer
        check.integer(x=ITMAX, positive=True)


    # initializations
    D = data.size
    residuals = np.copy(data)
    delta_list = []
    delta = np.sqrt(np.sum(residuals*residuals))/D
    delta_list.append(delta)
    vartheta = G.T@residuals
    rho0 = np.sum(vartheta*vartheta)
    parameters = np.zeros(G.shape[1])
    tau = 0
    eta = np.zeros_like(parameters)
    m = 1
    # updates
    while (delta > epsilon) and (m < ITMAX):
        eta = vartheta + tau*eta
        nu = G@eta
        upsilon = rho0/np.sum(nu*nu)
        parameters += upsilon*eta
        residuals -= upsilon*nu
        delta = np.sqrt(np.sum(residuals*residuals))/D
        delta_list.append(delta)
        vartheta = G.T@residuals
        rho = np.sum(vartheta*vartheta)
        tau = rho/rho0
        rho0 = rho
        m += 1

    return delta_list, parameters


def method_column_action_C92(G, data, data_points, zlayer, scale, epsilon, ITMAX, check_input=True):
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

    if check_input is True:
        # check if data and G are consistent numpy arrays
        check.sensibility_matrix_and_data(G=G, data=data)
        # check data points
        check.coordinates(coordinates=data_points)
        # check if zlayer result in a layer below the data points
        if (type(zlayer) != float) and (np.any(zlayer <= data_points[2])):
            raise ValueError("zlayer must be a scalar greater than the z coordinate of all data points")
        # check if epsilon is a positive scalar
        check.scalar(x=epsilon, positive=True)
        # check if ITMAX is a positive integer
        check.integer(x=ITMAX, positive=True)

        # initializations
        residuals = np.copy(data)
        parameters = data*scale
        imax = np.argmax(np.abs(residuals))
        rmax = residuals[imax]
        rmax_list = []
        rmax_list.append(rmax)
        m = 1
        # updates
        while (rmax > epsilon) and (m < ITMAX):
            xmax, ymax, zmax = data_points[:,imax]
            parameters[imax] += rmax*scale*np.abs(zlayer - zmax)
            residuals -= G[:,imax]*rmax
            imax = np.argmax(np.abs(residuals))
            rmax = residuals[imax]
            rmax_list.append(rmax)
            m += 1

        return rmax_list, parameters


def method_iterative_SOB17(G, data, factor, epsilon, ITMAX=50, check_input=True):
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
    
    if check_input is True:
        # check if data and G are consistent numpy arrays
        check.sensibility_matrix_and_data(G=G, data=data)
        # check if factor and epsilon are positive scalars
        check.scalar(x=factor, positive=True)
        check.scalar(x=epsilon, positive=True)
        # check if ITMAX is a positive integer
        check.integer(x=ITMAX, positive=True)

    # initializations
    D = data.size
    parameters = np.zeros_like(data)
    residuals = np.copy(data)
    delta_list = []
    delta = np.sqrt(np.sum(residuals*residuals))/D
    delta_list.append(delta)
    m = 1
    # updates
    while (delta > epsilon) and (m < ITMAX):
        dp = factor*residuals
        parameters += dp
        nu = G@dp
        residuals -= nu
        delta = np.sqrt(np.sum(residuals*residuals))/D
        delta_list.append(delta)
        m += 1

    return delta_list, parameters