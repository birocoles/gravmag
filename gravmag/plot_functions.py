import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


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


def model_boundaries(model, m2km=True):
    """
    Plot the projection of the model boundaries on plane xy.
    """
    for prism in model:
        x1, x2, y1, y2 = prism[:4]
        x = np.array([x1, x2, x2, x1, x1])
        y = np.array([y1, y1, y2, y2, y1])
        if m2km is True:
            plt.plot(0.001 * y, 0.001 * x, "k--", linewidth=2)
        else:
            plt.plot(y, x, "k--", linewidth=2)


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


def draw_surface_prisms(
    ax, prisms, color, alpha, edges_width, edges_color, check_prisms=True
):
    """
    Plot the surface of rectangular prisms.

    Parameters:

    * ax: axes of a matplotlib figure.
    * prisms : 2d-array
        2d-array containing the coordinates of the prisms. Each line must contain
        the coordinates of a single prism in the following order:
        south (x1), north (x2), west (y1), east (y2), top (z1) and bottom (z2).
        All coordinates should be in meters.
    * color: RGB matplotlib tuple
        Color of the body.
    * alpha: float
        Transparency of the body.
    * edges_width: float
        Thickness of the edges.
    * edges_color: RGB matplotlib tuple
        Color of the edges.
    * check_prisms: boolean
        If True, call function prism_functions._check_prisms to verify if prisms
        are well defined.
    """

    if check_prisms is True:
        prf._check_prisms(prisms)

    for prism in prisms:
        # get the coordinates
        x1, x2, y1, y2, z1, z2 = prism
        # plot top surface
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        z = [z1, z1, z1, z1, z1]
        surface = Poly3DCollection(
            [list(zip(x, y, z))], color=color, alpha=alpha
        )
        ax.add_collection3d(surface)
        edges = Line3DCollection(
            [list(zip(x, y, z))], color=edges_color, linewidth=edges_width
        )
        ax.add_collection3d(edges)
        # plot bottom surface
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        z = [z2, z2, z2, z2, z2]
        surface = Poly3DCollection(
            [list(zip(x, y, z))], color=color, alpha=alpha
        )
        ax.add_collection3d(surface)
        edges = Line3DCollection(
            [list(zip(x, y, z))], color=edges_color, linewidth=edges_width
        )
        ax.add_collection3d(edges)
        # plot x1 surface
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        z = [z2, z2, z2, z2, z2]
        surface = Poly3DCollection(
            [list(zip(x, y, z))], color=color, alpha=alpha
        )
        ax.add_collection3d(surface)
        edges = Line3DCollection(
            [list(zip(x, y, z))], color=edges_color, linewidth=edges_width
        )
        ax.add_collection3d(edges)


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
