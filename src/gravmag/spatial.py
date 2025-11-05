import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from warnings import warn


class Spatial2D:
    """
    Represents 2D spatial coordinates (x, y) that can be scattered or gridded.
    Gridded data store only 1D coordinate vectors internally (no redundancy).
    """

    def __init__(self, x, y, is_gridded=False, azimuth=0.0):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.is_gridded = bool(is_gridded)
        self.azimuth = float(azimuth)

        # Expect 1D coordinate vectors
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be 1D coordinate vectors.")

        # Ensure matching shapes for scattered data
        if self.is_gridded is False:
            if self.x.shape != self.y.shape:
                raise ValueError("Scattered coordinates require x and y with identical shapes.")

    @classmethod
    def from_area(cls, area, shape, azimuth):
        """
        Create a gridded Spatial2D from area (bounding box) and shape.

        Parameters
        ----------
        area : tuple of 4 scalars
            [xmin, xmax, ymin, ymax]
        shape : tuple of 2 ints
            (nx, ny) = number of points along x and y
        azimuth : float
            Clockwise angle (in degrees) from geographic north to grid x direction
        """
        
        area = tuple(area)
        shape = tuple(shape)
        azimuth = float(azimuth)

        if len(area) != 4:
            raise ValueError("area must have four elements (xmin, xmax, ymin, ymax).")
        if len(shape) != 2:
            raise ValueError("shape must have two elements (nx, ny).")

        xmin, xmax, ymin, ymax = map(float, area)
        nx, ny = map(int, shape)
        cos = np.cos(np.deg2rad(azimuth))
        sin = np.sin(np.deg2rad(azimuth))

        # 1D coordinate vectors
        _x = np.linspace(xmin, xmax, nx)
        _y = np.linspace(ymin, ymax, ny)
        x = cos * _x - sin * _y
        y = sin * _x + cos * _y

        is_gridded = True
        return cls(x, y, is_gridded, azimuth)
    
    @property
    def shape(self):
        return (len(self.x), len(self.y)) if self.is_gridded else self.x.shape

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def area(self):
        return [self.x.min(), self.x.max(), self.y.min(), self.y.max()]
    

    def full_grid_xy_coordinates(self):
        """Return (X, Y) as 2D arrays views."""
        if self.is_gridded:
            X = np.broadcast_to(self.x[:, None], (len(self.x), len(self.y)))
            Y = np.broadcast_to(self.y[None, :], (len(self.x), len(self.y)))
            return X, Y
        else:
            raise ValueError("Cannot return full grid coordinates from scattered data.")
    
    def stack(self):
        """Return (N, 2) coordinate array."""
        if self.is_gridded:
            raise ValueError("Cannot stack gridded coordinates.")
        else:
            # The stacked coordinates are copies of the original x and y
            return np.column_stack((self.x, self.y))
    
    def __repr__(self):
        t = "gridded" if self.is_gridded else "scattered"
        return f"<Spatial2D: {t}, shape={self.shape}, size={self.size}, azimuth={self.azimuth:.2f}>"


class Spatial3D(Spatial2D):
    """
    Extends Spatial2D by adding a vertical coordinate z (scalar or array).
    In gridded mode, z can be scalar or 2D array of shape (ny, nx).
    """

    def __init__(self, x, y, z=0.0, is_gridded=None, azimuth=0.0):
        super().__init__(x, y, is_gridded, azimuth)
        
        if isinstance(z, (float, int)):
            self.z = float(z)
        else:
            self.z = np.asarray(z, dtype=float)
            if self.is_gridded:
                nx, ny = len(self.x), len(self.y)
                if self.z.shape != (nx, ny):
                    raise ValueError(f"For gridded mode, z must have shape {(nx, ny)}.")
            else:
                if self.z.shape != self.x.shape:
                    raise ValueError("For scattered mode, z must have same shape as x and y.")
    
    def stack(self):
        """Return (N, 3) coordinate array."""
        if self.is_gridded:
            raise ValueError("Cannot stack gridded coordinates.")
        else:
            if self.z.ndim == 0: # z is scalar
                # create a view of z
                z_array = np.broadcast_to(self.z, self.x.shape)
            else: # z is array
                z_array = self.z
            return np.column_stack((self.x, self.y, z_array))
    
    def __repr__(self):
        t = "gridded" if self.is_gridded else "scattered"
        return f"<Spatial3D: {t}, shape={self.shape}, size={self.size}, azimuth={self.azimuth:.2f}>"
