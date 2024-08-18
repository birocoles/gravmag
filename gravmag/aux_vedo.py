"""
This code contains auxiliary routines for data and model visualization
using Vedo.
"""

import numpy as np
import vedo
from . import check

def custom_axes(area):
    '''
    Function form creating the axes in Vedo.
    '''
    axes = vedo.Axes(
        obj=None,
        xtitle='x',
        ytitle='y',
        ztitle='z',
        xrange=(area[0], area[1]),
        yrange=(area[2], area[3]),
        zrange=(area[4], area[5]),
        c=None,
        number_of_divisions=None,
        digits=None,
        limit_ratio=0.04,
        title_depth=0,
        title_font='',
        text_scale=1.0,
        x_values_and_labels=None,
        y_values_and_labels=None,
        z_values_and_labels=None,
        htitle='',
        htitle_size=0.03,
        htitle_font=None,
        htitle_italic=False,
        htitle_color=None,
        htitle_backface_color=None,
        htitle_justify='bottom-left',
        htitle_rotation=90,
        htitle_offset=(0, 0.01, 0),
        xtitle_position=0.95,
        ytitle_position=0.95,
        ztitle_position=0.95,
        xtitle_offset=0.025,
        ytitle_offset=-0.0875,
        ztitle_offset=-0.025,
        xtitle_justify=None,
        ytitle_justify=None,
        ztitle_justify=None,
        xtitle_rotation=(180,0,180),
        ytitle_rotation=(90, 0, 180),
        ztitle_rotation=(0, 0, 180),
        xtitle_box=False,
        ytitle_box=False,
        xtitle_size=0.025,
        ytitle_size=0.025,
        ztitle_size=0.025,
        xtitle_color='k',
        ytitle_color='k',
        ztitle_color='k',
        xtitle_backface_color='k',
        ytitle_backface_color='k',
        ztitle_backface_color='k',
        xtitle_italic=0,
        ytitle_italic=0,
        ztitle_italic=0,
        grid_linewidth=1,
        xygrid=False,
        yzgrid=False,
        zxgrid=True,
        xygrid2=True,
        yzgrid2=True,
        zxgrid2=False,
        xygrid_transparent=True,
        yzgrid_transparent=False,
        zxgrid_transparent=True,
        xygrid2_transparent=False,
        yzgrid2_transparent=True,
        zxgrid2_transparent=False,
        xyplane_color=None,
        yzplane_color=None,
        zxplane_color=None,
        xygrid_color=None,
        yzgrid_color=None,
        zxgrid_color=None,
        xyalpha=0.075,
        yzalpha=0.075,
        zxalpha=0.075,
        xyframe_line=None,
        yzframe_line=None,
        zxframe_line=None,
        xyframe_color=None,
        yzframe_color=None,
        zxframe_color=None,
        axes_linewidth=1,
        xline_color=None,
        yline_color=None,
        zline_color=None,
        xhighlight_zero=False,
        yhighlight_zero=False,
        zhighlight_zero=False,
        xhighlight_zero_color='red4',
        yhighlight_zero_color='green4',
        zhighlight_zero_color='blue4',
        show_ticks=True,
        xtick_length=0.015,
        ytick_length=0.015,
        ztick_length=0.015,
        xtick_thickness=0.0025,
        ytick_thickness=0.0025,
        ztick_thickness=0.0025,
        xminor_ticks=1,
        yminor_ticks=1,
        zminor_ticks=None,
        tip_size=None,
        label_font='',
        xlabel_color='k',
        ylabel_color='k',
        zlabel_color='k',
        xlabel_backface_color='k',
        ylabel_backface_color='k',
        zlabel_backface_color='k',
        xlabel_size=0.016,
        ylabel_size=0.016,
        zlabel_size=0.016,
        xlabel_offset=0.8,
        ylabel_offset=-1,
        zlabel_offset=1,
        xlabel_justify=None,
        ylabel_justify=None,
        zlabel_justify=None,
        xlabel_rotation=(180,0,180),
        ylabel_rotation=(90, 0, 180),
        zlabel_rotation=(0, 0, 180),
        xaxis_rotation=0,
        yaxis_rotation=0,
        zaxis_rotation=0,
        xyshift=0,
        yzshift=0,
        zxshift=0,
        xshift_along_y=0,
        xshift_along_z=0,
        yshift_along_x=1,
        yshift_along_z=0,
        zshift_along_x=0,
        zshift_along_y=0,
        x_use_bounds=False,
        y_use_bounds=False,
        z_use_bounds=False,
        x_inverted=False,
        y_inverted=False,
        z_inverted=False,
        use_global=False,
        tol=0.001
    )
    return axes


def gravmag2vedo_prisms(gravmag_prisms, scalar_props=None, cmap='jet', vmin=None, vmax=None):
    '''
    Receive a gravmag model and return a Vedo model.

    parameters
    ----------
    gravmag_prisms : dictionary
        Dictionary containing the x, y and z coordinates of the corners of each prism in prisms.
        The corners south (x1), north (x2), west (y1), east (y2), top (z1) and bottom (z2) of each
        prism are arranged in the keys 'x1', 'x2', 'y1', 'y2', 'z1' and 'z2', respectively.
        Each key is a numpy array 1d having the same number of elements.
    scalar_props : None or numpy array 1d
        If not None, it must be a numpy array 1d containing the scalar physical
        property of each prism forming the gravmag_model.
    cmap : Matplotib colormap
        A Matplotlib colormap (https://matplotlib.org/stable/users/explain/colors/colormaps.html).
        This parameter is ignored if scalar_props=None.
    vmin, vmax : scalars
        Lower and upper limits for the color scale.
        This is ignored if scalar_props=None.

    returns
    -------
    vedo_prisms : vedo.mesh.Mesh
        Vedo object containing a collection of rectangular prisms.
    '''
    P = check.are_rectangular_prisms(prisms=gravmag_prisms)
    if scalar_props is not None:
        check.is_array(x=scalar_props, ndim=1, shape=(P,))
        check.is_scalar(x=vmin, positive=False)
        check.is_scalar(x=vmax, positive=False)
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax")
    
    # iterate iver prisms to create vedo.Box objects
    vedo_prisms = []
    for (x1, x2, y1, y2, z1, z2) in zip(
        gravmag_prisms['x1'], 
        gravmag_prisms['x2'], 
        gravmag_prisms['y1'], 
        gravmag_prisms['y2'], 
        gravmag_prisms['z1'], 
        gravmag_prisms['z2']
    ):
        vedo_prisms.append(
            vedo.Box(
                pos=(x1, x2, y1, y2, z1, z2),
                c='blue4',
                alpha=1
            )
        )

    # create a unified mesh
    vedo_prisms = vedo.merge(vedo_prisms).force_opaque()
    
    if scalar_props is not None:
        # colorize the prisms
        # create colors for the model
        # the color values must be defined for the faces of each prism
        # we need to repeat the color values for each prism 6 times because they are
        # associated to the prisms faces
        vedo_prisms.cmap(
            input_cmap=cmap, 
            input_array=np.repeat(scalar_props, 6), 
            on='cells', 
            vmin=vmin, 
            vmax=vmax,
            alpha=1
        )

    return vedo_prisms


def points(data, scalar_props=None, cmap=None, vmin=None, vmax=None):
    '''
    Receive a set of points (x, y, z) and return a colorized Vedo pointcloud.

    parameters
    ----------
    data : numpy array 2d or vedo.pointcloud.Points
        N x 3 matrix containing the coorinates x, y and z at the first, second and third columns,
        respectively, or a Vedo object vedo.pointcloud.Points.
    scalar_props : None or numpy array 1d
        If not None, it must be a numpy array 1d containing the scalar physical
        property of each point in data.
    cmap : Matplotib colormap
        A Matplotlib colormap (https://matplotlib.org/stable/users/explain/colors/colormaps.html).
        This is ignored if scalar_props=None.
    vmin, vmax : scalars
        Lower and upper limits for the color scale.
        This is ignored if scalar_props=None.

    returns
    -------
    vedo_pointcloud or data : vedo.pointcloud.Points
        Vedo object defining a colorized pointcloud. If data is a numpy array 2d,
        returns a new Vedo pointcloud. Otherwise, returns the colorized data.
    '''
    if isinstance(data, np.ndarray):
        check.is_array(x=data, ndim=2)
        if data.shape[1] != 3:
            raise ValueError("data must have three columns")
        N = data.shape[0]
    elif isinstance(data, vedo.pointcloud.Points):
        if data.vertices.shape[1] != 3:
            raise ValueError("data must have three columns")
        N = data.vertices.shape[0]
    else:
        raise ValueError(
            "data must be a numpy array or vedo.pointcloud.Points"
            )

    if scalar_props is not None:
        check.is_array(x=scalar_props, ndim=1, shape=(N,))
        check.is_scalar(x=vmin, positive=False)
        check.is_scalar(x=vmax, positive=False)
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax")

    # colorize the poincloud
    if scalar_props is not None:
        # colorize the original data
        try:
            data.cmap(
                input_cmap=cmap, 
                input_array=scalar_props,
                vmin=vmin, 
                vmax=vmax,
                alpha=1
            )
            print("The original data were colorized")
            return data
        except:
            # generate a Vedo pointcloud if data is a numpy array 2d
            vedo_pointcloud = vedo.Points(
                inputobj=data, 
                r=4, 
                c='blue4', 
                alpha=1
                )
            vedo_pointcloud.cmap(
                input_cmap=cmap, 
                input_array=scalar_props,
                vmin=vmin, 
                vmax=vmax,
                alpha=1
            )
            print("A colorized Vedo pointcloud were created")
            return vedo_pointcloud


def surface(data, scalar_props=None, cmap=None, vmin=None, vmax=None):
    '''
    Receive a numpy array 2d, a vedo.pointcloud.Points or a Vedo surface and 
    return a colorized Vedo surface.

    parameters
    ----------
    data : numpy array 2d, vedo.pointcloud.Points or vedo.mesh.Mesh
        N x 3 matrix containing the coorinates x, y and z at the first, second and third columns,
        respectively, or a Vedo object vedo.pointcloud.Points of a Vedo object vedo.mesh.Mesh 
        defining a surface.
    scalar_props : None or numpy array 1d
        If not None, it must be a numpy array 1d containing the scalar physical
        property of obj.
    cmap : Matplotib colormap
        A Matplotlib colormap (https://matplotlib.org/stable/users/explain/colors/colormaps.html).
        This is ignored if scalar_props=None.
    vmin, vmax : scalars
        Lower and upper limits for the color scale.
        This is ignored if scalar_props=None.

    returns
    -------
    vedo_surface : vedo.mesh.Mesh
        Vedo object defining a surface. If data is a numpy array 2d or a Vedo pointcloud,
        returns a new Vedo surface. Otherwise, returns the colorized data.
    '''
    if isinstance(data, np.ndarray):
        check.is_array(x=data, ndim=2)
        if data.shape[1] != 3:
            raise ValueError("data must have three columns")
        N = data.shape[0]
    elif isinstance(data, (vedo.pointcloud.Points, vedo.mesh.Mesh)):
        if data.vertices.shape[1] != 3:
            raise ValueError("data must have three columns")
        N = data.vertices.shape[0]
    else:
        raise ValueError(
            "data must be a numpy array, vedo.pointcloud.Points or Vedo surface vedo.mesh.Mesh"
            )

    if scalar_props is not None:
        check.is_array(x=scalar_props, ndim=1, shape=(N,))
        check.is_scalar(x=vmin, positive=False)
        check.is_scalar(x=vmax, positive=False)
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax")

    # colorize the surface
    if scalar_props is not None:
        # colorize the original data
        if isinstance(data, vedo.mesh.Mesh):
            data.cmap(
                input_cmap=cmap, 
                input_array=scalar_props,
                vmin=vmin, 
                vmax=vmax,
                alpha=1
            )
            print("The original data were colorized")
            return data
        # generate a surface using 2D Delaunay triangulation
        # if data is a Vedo pointcloud or numpy array 
        else:
            if isinstance(data, vedo.pointcloud.Points):
                vedo_surface = data.generate_delaunay2d()
            else: # data is an np.ndarray
                vedo_surface = vedo.Points(data).generate_delaunay2d()
            vedo_surface.cmap(
                input_cmap=cmap, 
                input_array=scalar_props,
                vmin=vmin, 
                vmax=vmax,
                alpha=1
            )
            print("A colorized Vedo surface were created")
            return vedo_surface
            