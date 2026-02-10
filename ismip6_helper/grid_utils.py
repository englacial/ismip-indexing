"""
Utility functions for correcting ISMIP6 grid coordinates.

The ISMIP6 standard grid is a polar stereographic projection (EPSG:3031):
- Standard parallel: 71° S
- Central meridian: 0° W
- Datum: WGS84
- Domain: (-3,040,000 m, -3,040,000 m) to (3,040,000 m, 3,040,000 m)
- Resolutions: 32 km, 16 km, 8 km, 4 km, 2 km, or 1 km
"""

import logging
import numpy as np
import xarray as xr
from typing import Tuple, Optional
import warnings
import pyproj

logger = logging.getLogger(__name__)


# ISMIP6 standard grid parameters (EPSG:3031 - Antarctic Polar Stereographic)
GRID_BOUNDS = {
    'x_min': -3040000.0,  # meters
    'x_max': 3040000.0,   # meters
    'y_min': -3040000.0,  # meters
    'y_max': 3040000.0,   # meters
}

# Standard resolutions in meters
STANDARD_RESOLUTIONS = [32000, 16000, 8000, 4000, 2000, 1000]


def detect_grid_resolution(nx: int, ny: int) -> Tuple[float, float]:
    """
    Detect the grid resolution based on the number of grid points.

    Parameters
    ----------
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction

    Returns
    -------
    dx, dy : float, float
        Grid spacing in meters
    """
    # Calculate grid extent
    x_extent = GRID_BOUNDS['x_max'] - GRID_BOUNDS['x_min']
    y_extent = GRID_BOUNDS['y_max'] - GRID_BOUNDS['y_min']

    # Calculate spacing
    dx = x_extent / (nx - 1) if nx > 1 else x_extent
    dy = y_extent / (ny - 1) if ny > 1 else y_extent

    # Try to match to standard resolutions
    for res in STANDARD_RESOLUTIONS:
        if abs(dx - res) < res * 0.1:  # Within 10% tolerance
            dx = res
            break

    for res in STANDARD_RESOLUTIONS:
        if abs(dy - res) < res * 0.1:
            dy = res
            break

    return dx, dy


def create_coordinates(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create x and y coordinate arrays for ISMIP6 grid.

    Parameters
    ----------
    nx, ny : int
        Number of grid points
    dx, dy : float
        Grid spacing in meters

    Returns
    -------
    x, y : np.ndarray
        Coordinate arrays in EPSG:3031
    """
    # Create coordinates centered on the ISMIP6 domain
    x = np.linspace(GRID_BOUNDS['x_min'], GRID_BOUNDS['x_max'], nx)
    y = np.linspace(GRID_BOUNDS['y_min'], GRID_BOUNDS['y_max'], ny)

    # Adjust to match detected spacing if needed
    actual_dx = (x[-1] - x[0]) / (nx - 1) if nx > 1 else 0
    actual_dy = (y[-1] - y[0]) / (ny - 1) if ny > 1 else 0

    if abs(actual_dx - dx) > 1.0:  # More than 1m difference
        x = GRID_BOUNDS['x_min'] + np.arange(nx) * dx
    if abs(actual_dy - dy) > 1.0:
        y = GRID_BOUNDS['y_min'] + np.arange(ny) * dy

    return x, y


def verify_latlon_consistency(x: np.ndarray, y: np.ndarray,
                               lat: Optional[np.ndarray] = None,
                               lon: Optional[np.ndarray] = None,
                               tolerance: float = 1.0) -> bool:
    """
    Verify that projected coordinates are consistent with lat/lon if provided.

    Parameters
    ----------
    x, y : np.ndarray
        Projected coordinates (EPSG:3031)
    lat, lon : np.ndarray, optional
        Latitude and longitude arrays
    tolerance : float
        Tolerance in degrees for lat/lon matching

    Returns
    -------
    consistent : bool
        True if coordinates are consistent (or if lat/lon not provided)
    """
    if lat is None or lon is None:
        return True  # Can't verify without lat/lon

    try:
        from pyproj import Transformer

        # Create transformer from EPSG:3031 to WGS84 (lat/lon)
        transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=False)

        # Sample a few points for verification (avoid loading full arrays)
        sample_indices = [
            (0, 0),  # Lower left
            (-1, -1),  # Upper right
            (len(x)//2, len(y)//2),  # Center
        ]

        for i, j in sample_indices:
            if i < len(x) and j < len(y):
                # Transform projected to lat/lon
                est_lat, est_lon = transformer.transform(x[i], y[j])

                # Get actual lat/lon from data
                actual_lat = lat[j, i] if lat.ndim == 2 else lat[j]
                actual_lon = lon[j, i] if lon.ndim == 2 else lon[i]

                # Check consistency
                lat_diff = abs(est_lat - actual_lat)
                lon_diff = abs(est_lon - actual_lon)

                if lat_diff > tolerance or lon_diff > tolerance:
                    warnings.warn(
                        f"Coordinate mismatch at grid point ({i},{j}): "
                        f"Estimated (lat={est_lat:.2f}, lon={est_lon:.2f}), "
                        f"Actual (lat={actual_lat:.2f}, lon={actual_lon:.2f})"
                    )
                    return False

        return True

    except ImportError:
        warnings.warn("pyproj not available, skipping lat/lon verification")
        return True
    except Exception as e:
        warnings.warn(f"Error during lat/lon verification: {e}")
        return True


def correct_grid_coordinates(ds: xr.Dataset, data_var: Optional[str] = None) -> xr.Dataset:
    """
    Correct grid coordinates for ISMIP6 datasets.

    If the dataset has x and y coordinates, returns it unmodified.
    If missing, attempts to add EPSG:3031 coordinates based on data dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    data_var : str, optional
        Name of the main data variable to use for dimension detection.
        If None, uses the first 2D or 3D variable found.

    Returns
    -------
    ds_corrected : xr.Dataset
        Dataset with corrected coordinates

    Examples
    --------
    >>> ds = xr.open_dataset('some_file.nc')
    >>> ds_corrected = correct_grid_coordinates(ds)
    """
    # Check if x and y coordinates already exist
    has_x = 'x' in ds.coords
    has_y = 'y' in ds.coords

    if has_x and has_y:
        # Coordinates exist -- check for duplicates which break xarray indexing
        x_vals = ds.coords['x'].values
        y_vals = ds.coords['y'].values
        x_has_dupes = len(x_vals) != len(np.unique(x_vals))
        y_has_dupes = len(y_vals) != len(np.unique(y_vals))
        if not x_has_dupes and not y_has_dupes:
            return ds
        # Fall through to regenerate coordinates from the ISMIP6 grid spec
        logger.warning(
            "Duplicate coordinate values detected (x_dupes=%s, y_dupes=%s), "
            "regenerating from ISMIP6 grid spec",
            x_has_dupes, y_has_dupes,
        )

    # Find the data variable to use for dimension detection
    if data_var is None:
        # Find first 2D or 3D variable
        for var_name in ds.data_vars:
            var = ds[var_name]
            if var.ndim >= 2:
                data_var = var_name
                break

    if data_var is None or data_var not in ds:
        warnings.warn("No suitable data variable found for grid detection")
        return ds

    var = ds[data_var]
    dims = var.dims

    # Identify spatial dimensions
    # Common patterns: (time, y, x), (y, x), (time, lat, lon), etc.
    # Exclude time and also exclude common non-spatial dimensions like 'bnds', 'nv', 'nv4', etc.
    spatial_dims = []
    for dim in dims:
        if dim not in ['time', 't', 'bnds', 'nv', 'nv4', 'nb2', 'vertices']:
            spatial_dims.append(dim)

    if len(spatial_dims) < 2:
        warnings.warn(f"Cannot identify spatial dimensions for {data_var}")
        return ds

    # Assume last two dimensions are (y, x) or (lat, lon)
    # But only if they are actual spatial grid dimensions (reasonably large)
    y_dim = spatial_dims[-2]
    x_dim = spatial_dims[-1]

    # Sanity check: spatial dimensions should have reasonable sizes (> 10)
    if y_dim in ds.sizes and x_dim in ds.sizes:
        if ds.sizes[y_dim] < 10 or ds.sizes[x_dim] < 10:
            warnings.warn(f"Detected dimensions {y_dim}={ds.sizes[y_dim]}, {x_dim}={ds.sizes[x_dim]} seem too small to be spatial dimensions")
            return ds

    ny = ds.sizes[y_dim]
    nx = ds.sizes[x_dim]

    print(f"⚠️  Grid correction: Dataset missing x/y coordinates for '{data_var}'")
    print(f"   Detected dimensions: {y_dim}={ny}, {x_dim}={nx}")

    # Detect resolution
    dx, dy = detect_grid_resolution(nx, ny)
    print(f"   Estimated resolution: dx={dx/1000:.1f} km, dy={dy/1000:.1f} km")

    # Create coordinates
    x, y = create_coordinates(nx, ny, dx, dy)
    print(f"   Creating coordinates: x=[{x[0]/1000:.1f}, {x[-1]/1000:.1f}] km, "
          f"y=[{y[0]/1000:.1f}, {y[-1]/1000:.1f}] km")

    # Create a copy of the dataset
    ds_corrected = ds.copy()

    # Check for existing lat/lon coordinates
    has_lat = 'lat' in ds.coords or 'latitude' in ds.coords
    has_lon = 'lon' in ds.coords or 'longitude' in ds.coords

    lat_data = None
    lon_data = None

    if has_lat:
        lat_data = ds.coords.get('lat', ds.coords.get('latitude'))
    if has_lon:
        lon_data = ds.coords.get('lon', ds.coords.get('longitude'))

    # Verify consistency if lat/lon exist
    if lat_data is not None and lon_data is not None:
        lat_array = lat_data.values
        lon_array = lon_data.values

        print("   Verifying consistency with existing lat/lon coordinates...")
        consistent = verify_latlon_consistency(x, y, lat_array, lon_array)

        if consistent:
            print("   ✓ Coordinates are consistent with lat/lon")
        else:
            print("   ⚠️  Warning: Coordinates may not match lat/lon exactly")

    # Add x and y coordinates
    # Map the dimensional names to x and y
    # Only rename if the dimension names are different from 'x' and 'y'
    rename_dict = {}
    if x_dim != 'x':
        rename_dict[x_dim] = 'x'
    if y_dim != 'y':
        rename_dict[y_dim] = 'y'

    if rename_dict:
        # Before renaming, drop any problematic bound variables that might cause conflicts
        # These often have dimensions like (y, x, nv4) which can cause issues
        vars_to_drop = []
        for var_name in ds_corrected.data_vars:
            var = ds_corrected[var_name]
            # Check if this variable uses the dimensions we're about to rename
            if any(dim in rename_dict for dim in var.dims):
                # If it's a bounds variable or has extra dimensions beyond the spatial ones,
                # it might cause conflicts
                if '_bnds' in var_name or len(var.dims) > 3:
                    vars_to_drop.append(var_name)

        if vars_to_drop:
            print(f"   Dropping problematic variables before rename: {vars_to_drop}")
            ds_corrected = ds_corrected.drop_vars(vars_to_drop)

        ds_corrected = ds_corrected.rename(rename_dict)

    # Assign coordinate values
    ds_corrected = ds_corrected.assign_coords({
        'x': ('x', x, {'units': 'm', 'long_name': 'x coordinate (EPSG:3031)',
               'standard_name': 'projection_x_coordinate'}),
        'y': ('y', y, {'units': 'm', 'long_name': 'y coordinate (EPSG:3031)',
               'standard_name': 'projection_y_coordinate'})
    })

    # Add grid mapping information
    ds_corrected.attrs['grid_mapping'] = 'polar_stereographic'

    # Add projection information as a coordinate
    crs = xr.DataArray(
        data=np.int32(0),
        attrs={
            'grid_mapping_name': 'polar_stereographic',
            'latitude_of_projection_origin': -90.0,
            'standard_parallel': -71.0,
            'straight_vertical_longitude_from_pole': 0.0,
            'false_easting': 0.0,
            'false_northing': 0.0,
            'semi_major_axis': 6378137.0,
            'semi_minor_axis': 6356752.314245,
            'inverse_flattening': 298.257223563,
            'spatial_ref': 'EPSG:3031',
            'crs_wkt': 'PROJCS["WGS 84 / Antarctic Polar Stereographic",'
                       'GEOGCS["WGS 84",DATUM["WGS_1984",'
                       'SPHEROID["WGS 84",6378137,298.257223563]],'
                       'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
                       'PROJECTION["Polar_Stereographic"],'
                       'PARAMETER["latitude_of_origin",-71],'
                       'PARAMETER["central_meridian",0],'
                       'PARAMETER["false_easting",0],'
                       'PARAMETER["false_northing",0],'
                       'UNIT["metre",1]]'
        }
    )
    ds_corrected.coords['polar_stereographic'] = crs

    print(f"   ✓ Grid correction complete\n")

    return ds_corrected


