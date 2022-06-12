import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


def model_boundaries(model, m2km=True):
    '''
    Plot the projection of the model boundaries on plane xy.
    '''
    for prism in model:
        x1, x2, y1, y2 = prism[:4]
        x = np.array([x1, x2, x2, x1, x1])
        y = np.array([y1, y1, y2, y2, y1])
        if m2km is True:
            plt.plot(0.001*y, 0.001*x, 'k--', linewidth=2)
        else:
            plt.plot(y, x, 'k--', linewidth=2)


def draw_region(ax, xmin, xmax, ymin, ymax, zmin, zmax,
                label_size = 14, ticks_size = 12):
    '''
    Draw the 3D region where the objects will be plotted.

    Parameters:

    * ax: axes of a matplotlib figure.
    * xmin, xmax, ymin, ymax, zmin, zmax: floats
        Lower and upper limites along the x-, y- and z- axes.
    '''

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
    ax.set_xlabel('x (m)', fontsize=label_size)
    ax.set_ylabel('y (m)', fontsize=label_size)
    ax.set_zlabel('z (m)', fontsize=label_size)


def draw_surface_prisms(ax, prisms, color, alpha,
                        edges_width, edges_color, check_prisms=True):
    '''
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
    '''

    if check_prisms is True:
        prf._check_prisms(prisms)

    for prism in prisms:
        # get the coordinates
        x1, x2, y1, y2, z1, z2 = prism
        # plot top surface
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        z = [z1, z1, z1, z1, z1]
        surface = Poly3DCollection([list(zip(x,y,z))], color=color, alpha=alpha)
        ax.add_collection3d(surface)
        edges = Line3DCollection([list(zip(x,y,z))],
                                 color=edges_color, linewidth=edges_width)
        ax.add_collection3d(edges)
        # plot bottom surface
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        z = [z2, z2, z2, z2, z2]
        surface = Poly3DCollection([list(zip(x,y,z))], color=color, alpha=alpha)
        ax.add_collection3d(surface)
        edges = Line3DCollection([list(zip(x,y,z))],
                                 color=edges_color, linewidth=edges_width)
        ax.add_collection3d(edges)
        # plot x1 surface
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        z = [z2, z2, z2, z2, z2]
        surface = Poly3DCollection([list(zip(x,y,z))], color=color, alpha=alpha)
        ax.add_collection3d(surface)
        edges = Line3DCollection([list(zip(x,y,z))],
                                 color=edges_color, linewidth=edges_width)
        ax.add_collection3d(edges)
