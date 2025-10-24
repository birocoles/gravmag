import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
import warnings


class SpatialData:
    """
    Lightweight container for spatial data (scattered or gridded).
    """

    def __init__(self, x, y, values, grid=False, metadata=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.values = np.asarray(values)
        self.grid = grid
        self.metadata = metadata or {}

        # Basic validation
        if not grid and len(self.x) != len(self.y):
            raise ValueError("x and y must have same length for scattered data.")
        if grid and self.values.ndim != 2:
            raise ValueError("For gridded data, values must be 2D.")

    def __repr__(self):
        t = "Gridded" if self.grid else "Scattered"
        return f"<SpatialData ({t}) with {self.values.size} values>"

    def to_geodataframe(self, crs="EPSG:4326"):
        if self.grid:
            raise TypeError("GeoDataFrame conversion valid only for scattered data.")
        geometry = [Point(xy) for xy in zip(self.x, self.y)]
        return gpd.GeoDataFrame({"value": self.values}, geometry=geometry, crs=crs)

    # ---- I/O shortcuts ----
    def to_file(self, path, **kwargs):
        if self.grid:
            return save_grid(self, path, **kwargs)
        else:
            return save_scattered(self, path, **kwargs)

    @classmethod
    def from_file(cls, path, **kwargs):
        ext = path.lower().split(".")[-1]
        if ext in {"csv", "parquet", "gpkg", "geojson"}:
            return load_scattered(path, **kwargs)
        elif ext in {"tif", "tiff"}:
            return load_grid(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


def save_scattered(data: SpatialData, path, driver=None, crs="EPSG:4326"):
    """
    Save scattered data to CSV, GeoPackage, or GeoParquet.
    """
    gdf = data.to_geodataframe(crs=crs)
    ext = path.lower().split(".")[-1]

    if ext == "csv":
        gdf.drop(columns="geometry").to_csv(path, index=False)
    elif ext == "parquet":
        gdf.to_parquet(path, index=False)
    elif ext == "gpkg":
        gdf.to_file(path, driver="GPKG")
    elif ext == "geojson":
        gdf.to_file(path, driver="GeoJSON")
    else:
        raise ValueError(f"Unsupported scattered output format: {ext}")
    return path


def load_scattered(path, crs=None):
    """
    Load scattered data from CSV, Parquet, GeoPackage, or GeoJSON.
    """
    ext = path.lower().split(".")[-1]
    if ext == "csv":
        df = gpd.read_file(path) if "geometry" in open(path).read() else None
        import pandas as pd
        df = pd.read_csv(path)
        x, y, values = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    elif ext == "parquet":
        df = gpd.read_parquet(path)
        x, y, values = df.geometry.x, df.geometry.y, df["value"]
    elif ext in {"gpkg", "geojson"}:
        gdf = gpd.read_file(path)
        x, y, values = gdf.geometry.x, gdf.geometry.y, gdf["value"]
    else:
        raise ValueError(f"Unsupported scattered input format: {ext}")

    return SpatialData(x, y, values, grid=False, metadata={"source": path})


def save_grid(data: SpatialData, path, crs="EPSG:4326", transform=None, dtype=None):
    """
    Save gridded data as GeoTIFF.
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise ImportError("rasterio is required to save gridded data.")

    nx, ny = len(data.x), len(data.y)
    dx = (data.x[-1] - data.x[0]) / (nx - 1)
    dy = (data.y[-1] - data.y[0]) / (ny - 1)
    transform = transform or from_origin(data.x[0], data.y[-1], dx, dy)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.values.shape[0],
        width=data.values.shape[1],
        count=1,
        dtype=dtype or data.values.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data.values, 1)
    return path


def load_grid(path):
    """
    Load gridded data (GeoTIFF).
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required to read gridded data.")

    with rasterio.open(path) as src:
        values = src.read(1)
        transform = src.transform
        width, height = src.width, src.height
        x = np.arange(width) * transform.a + transform.c
        y = np.arange(height) * transform.e + transform.f
        crs = src.crs

    return SpatialData(x, y, values, grid=True, metadata={"crs": crs, "source": path})


def save_numpy(data: SpatialData, path):
    np.savez_compressed(
        path,
        x=data.x,
        y=data.y,
        values=data.values,
        grid=data.grid,
        metadata=data.metadata,
    )
    return path


def load_numpy(path):
    d = np.load(path, allow_pickle=True)
    return SpatialData(
        d["x"], d["y"], d["values"],
        grid=bool(d["grid"]),
        metadata=dict(d["metadata"].item())
    )