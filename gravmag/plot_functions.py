"""
This code presents some routines for data and model visualization
using Matplotlib and PyVista.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import pyvista as pv
from . import check


def plot_panels(
    size,
    nrows,
    ncols,
    X,
    Y,
    Z,
    bounds,
    titles,
    shape,
    area,
    mask,
    masked_shape,
    save=None,
):
    """
    Plot a regular grid of maps.
    """
    plt.figure(figsize=size)

    npanels = len(Z)

    for i in range(npanels):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis("scaled")
        plt.title(titles[i], fontsize=14)
        plt.contourf(
            0.001 * Y.ravel()[mask].reshape(masked_shape),
            0.001 * X.ravel()[mask].reshape(masked_shape),
            Z[i].ravel()[mask].reshape(masked_shape),
            vmin=-bounds[i],
            vmax=bounds[i],
            cmap="seismic",
        )
        plt.colorbar(shrink=0.52)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("y (km)", fontsize=14)
        plt.ylabel("x (km)", fontsize=14)
        plt.xlim(0.001 * area[2], 0.001 * area[3])
        plt.ylim(0.001 * area[0], 0.001 * area[1])

    if save is not None:
        plt.savefig(save, dpi=300, facecolor="w")

    plt.show()


def define_bound(X, Y, Z, area):
    """
    Define maximum absolute bounds for a list of data Z
    defined at the coordinates X and Y on the area "area".
    """
    mask = (
        (X.ravel() >= area[0])
        & (X.ravel() <= area[1])
        & (Y.ravel() >= area[2])
        & (Y.ravel() <= area[3])
    )
    Z_clipped = []
    for Zi in Z:
        Z_clipped.append(Zi.ravel()[mask])
    bound = np.max(np.abs(Z_clipped))
    return bound


def define_mask(total, shape, clip):
    """
    Defined a mask on a grid a regular of points
    with shape "shape", on the area "total". The
    area to be selected by the mask is defined by
    "clip".
    """
    xp = np.linspace(total[0], total[1], shape[0])
    yp = np.linspace(total[2], total[3], shape[1])
    maskx = (xp.ravel() >= clip[0]) & (xp.ravel() <= clip[1])
    masky = (yp.ravel() >= clip[2]) & (yp.ravel() <= clip[3])
    masked_shape = (xp[maskx].size, yp[masky].size)

    yp, xp = np.meshgrid(yp, xp)  # y-oriented grid
    xp = np.ravel(xp)
    yp = np.ravel(yp)
    mask = (
        (xp.ravel() >= clip[0])
        & (xp.ravel() <= clip[1])
        & (yp.ravel() >= clip[2])
        & (yp.ravel() <= clip[3])
    )
    return mask, masked_shape


def model_boundaries(model, color="k", style="--", width="2", m2km=True):
    """
    Plot the projection of the model boundaries on plane xy.
    """
    P = check.are_rectangular_prisms(model)
    for i in range(P):
        x1 = model["x1"][i]
        x2 = model["x2"][i]
        y1 = model["y1"][i]
        y2 = model["y2"][i]
        x = np.array([x1, x2, x2, x1, x1])
        y = np.array([y1, y1, y2, y2, y1])
        if m2km is True:
            plt.plot(
                0.001 * y,
                0.001 * x,
                color=color,
                linestyle=style,
                linewidth=width,
            )
        else:
            plt.plot(y, x, color=color, linestyle=style, linewidth=width)


def draw_region(
    ax, xmin, xmax, ymin, ymax, zmin, zmax, label_size=14, ticks_size=12
):
    """
    Draw the 3D region where the objects will be plotted.

    Parameters:

    * ax: axes of a matplotlib figure.
    * xmin, xmax, ymin, ymax, zmin, zmax: floats
        Lower and upper limites along the x-, y- and z- axes.
    """

    # x = [xmin, xmax, xmin, xmin]
    # y = [ymin, ymin, ymax, ymin]
    # z = [zmin, zmin, zmin, zmax]
    # ax.scatter(x, y, z, s=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.grid()
    ax.xaxis.set_tick_params(labelsize=ticks_size)
    ax.yaxis.set_tick_params(labelsize=ticks_size)
    ax.zaxis.set_tick_params(labelsize=ticks_size)
    ax.set_xlabel("x (m)", fontsize=label_size)
    ax.set_ylabel("y (m)", fontsize=label_size)
    ax.set_zlabel("z (m)", fontsize=label_size)


def bounds_diffs(computed, true):
    assert len(computed) == len(true)
    bounds = []
    diffs = []
    for c, t in zip(computed, true):
        bound_fields = np.max(np.abs(t))
        diffs.append(c - t)
        bound_diff = np.max(np.abs(c - t))
        bounds.append(bound_fields)
        bounds.append(bound_fields)
        bounds.append(bound_diff)

    return bounds, diffs


def fields_list(computed, true, diffs):
    assert len(computed) == len(true) == len(diffs)
    fields = []
    for c, t, d in zip(computed, true, diffs):
        fields.append(c)
        fields.append(t)
        fields.append(d)
    return fields


def prisms_to_pyvista(prisms, prop):
    """
    This function creates a pyvista.UnstructuredGrid of
    rectangular prisms from a Numpy array acoording to:

    https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html

    parameters
    ----------
    prisms : dictionary
        Dictionary containing the x, y and z coordinates of the corners of each prism in prisms.
        The corners south (x1), north (x2), west (y1), east (y2), top (z1) and bottom (z2) of each
        prism are arranged in the keys 'x1', 'x2', 'y1', 'y2', 'z1' and 'z2', respectively.
        Each key is a numpy array 1d having the same number of elements.
    prop : 1d-array
        1d-array containing the scalar physical property of each prism.

    returns
    -------
    model_mesh: pyvista.UnstructuredGrid
    """

    # Verify the input parameters
    nprisms = check.are_rectangular_prisms(prisms=prisms)
    check.is_array(x=prop, ndim=1, shape=(nprisms,))

    # define indices of the cells composing the pyvista.UnstructuredGrid
    cells = np.empty((nprisms, 9), dtype=int)
    cells[:, 0] = 8
    cells[:, 1:] = np.reshape(np.arange(8 * nprisms), (nprisms, 8))
    cells = cells.ravel()

    # define cell types of the pyvista.UnstructuredGrid
    cell_type = np.tile(np.array([pv.CellType.HEXAHEDRON]), nprisms)

    # define the cells forming the pyvista.UnstructuredGrid
    points = []
    for i in range(nprisms):
        points.append(
            np.array(
                [
                    [prisms["x1"][i], prisms["y1"][i], prisms["z1"][i]],
                    [prisms["x2"][i], prisms["y1"][i], prisms["z1"][i]],
                    [prisms["x2"][i], prisms["y2"][i], prisms["z1"][i]],
                    [prisms["x1"][i], prisms["y2"][i], prisms["z1"][i]],
                    [prisms["x1"][i], prisms["y1"][i], prisms["z2"][i]],
                    [prisms["x2"][i], prisms["y1"][i], prisms["z2"][i]],
                    [prisms["x2"][i], prisms["y2"][i], prisms["z2"][i]],
                    [prisms["x1"][i], prisms["y2"][i], prisms["z2"][i]],
                ],
                dtype=float,
            )
        )
    points = np.vstack(points)

    # create the model mesh (pyvista.UnstructuredGrid)
    model_mesh = pv.UnstructuredGrid(cells, cell_type, points)

    # add physical property
    model_mesh.cell_data["prop"] = prop.flatten(order="F")

    return model_mesh


def data_to_surface_pyvista(coordinates, data):
    """
    This function creates a connected surface from a PyVista point cloud.
    The surface is created by using delaunay_2d triangulation.

    parameters
    ----------
    coordinates : 2d-array
        2d-array containing x (first line), y (second line), and z (third line) of
        the computation points. All coordinates should be in meters.
    data : numpy array 1D
        Vector containing the potential-field data.

    returns
    -------
    data_mesh: pyvista.PolyData
    """
    D = check.are_coordinates(coordinates)
    assert data.size == D, "data size and coordinates must match"

    # create a PyVista point cloud
    data_mesh = pv.PolyData(
        np.vstack([coordinates["x"], coordinates["y"], coordinates["z"]]).T
    )

    # add point data to the point cloud
    data_mesh.point_data["data"] = data

    # transform the point cloud in a connected surface
    _ = data_mesh.delaunay_2d(inplace=True)

    return data_mesh
