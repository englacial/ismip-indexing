"""
Utility functions for correcting ISMIP6 grid coordinates.

Both ISMIP6 standard grids are polar stereographic on WGS84, and a source file
that lacks x/y is told apart by its array shape alone (no shape is shared):

- Antarctica (AIS), EPSG:3031: standard parallel 71° S, central meridian 0°,
  domain (-3,040 km, -3,040 km) to (3,040 km, 3,040 km), square;
  32/16/8/4/2/1 km give 191/381/761/1521/3041/6081 points per side.
- Greenland (GrIS), EPSG:3413: standard parallel 70° N, central meridian 45° W,
  domain (-720 km, -3,450 km) to (960 km, -570 km), so nx != ny;
  32/16/8/5/4/2/1 km give 53x91, 106x181, 211x361, 337x577, 421x721,
  841x1441, 1681x2881 (nx x ny). At 32 km the x extent is not a whole number
  of cells and the last column sits at 944 km.

The tables are the ISMIP grid description files vendored in
``ismip7_helper.conventions`` (``grid_registry()``), which list AIS 2-32 km and
GrIS 1-16 km; AIS 1 km and GrIS 32 km follow from the same origins and extents.
"""

import logging
import numpy as np
import xarray as xr
from typing import Tuple, Optional
import warnings

logger = logging.getLogger(__name__)


# ISMIP6 standard grid parameters (EPSG:3031 - Antarctic Polar Stereographic)
GRID_BOUNDS = {
    'x_min': -3040000.0,  # meters
    'x_max': 3040000.0,   # meters
    'y_min': -3040000.0,  # meters
    'y_max': 3040000.0,   # meters
}

#: Domain of each ice sheet's standard grid, in metres of its own projection.
GRID_BOUNDS_BY_REGION = {
    'AIS': GRID_BOUNDS,
    'GrIS': {'x_min': -720000.0, 'x_max': 960000.0, 'y_min': -3450000.0, 'y_max': -570000.0},
}

# Standard resolutions in meters
STANDARD_RESOLUTIONS = [32000, 16000, 8000, 4000, 2000, 1000]

#: Resolutions (m) each ice sheet's standard grid comes in, coarsest first.
STANDARD_RESOLUTIONS_BY_REGION = {
    'AIS': STANDARD_RESOLUTIONS,
    'GrIS': [32000, 16000, 8000, 5000, 4000, 2000, 1000],
}


def _grid_shape(region: str, res: int) -> Tuple[int, int]:
    """(nx, ny) of a standard grid: one point per cell edge, truncated when the extent is not a whole number of cells."""
    b = GRID_BOUNDS_BY_REGION[region]
    return (int((b['x_max'] - b['x_min']) // res) + 1,
            int((b['y_max'] - b['y_min']) // res) + 1)


#: ``(nx, ny) -> (region, resolution in m)`` for every standard ISMIP6 grid; no shape is shared.
GRID_SHAPES = {
    _grid_shape(region, res): (region, res)
    for region, resolutions in STANDARD_RESOLUTIONS_BY_REGION.items()
    for res in resolutions
}


#: EPSG:3031 (WGS 84 / Antarctic Polar Stereographic) as CF grid-mapping attributes.
#: The projection scalar every ISMIP6 group carries; viewers read ``grid_mapping``
#: attributes that point at it.
EPSG_3031_ATTRS = {
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
               'UNIT["metre",1]]',
}

#: EPSG:3413 (WGS 84 / NSIDC Sea Ice Polar Stereographic North), the ISMIP6 Greenland grid.
EPSG_3413_ATTRS = {
    'grid_mapping_name': 'polar_stereographic',
    'latitude_of_projection_origin': 90.0,
    'standard_parallel': 70.0,
    'straight_vertical_longitude_from_pole': -45.0,
    'false_easting': 0.0,
    'false_northing': 0.0,
    'semi_major_axis': 6378137.0,
    'semi_minor_axis': 6356752.314245,
    'inverse_flattening': 298.257223563,
    'spatial_ref': 'EPSG:3413',
    'crs_wkt': 'PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",'
               'GEOGCS["WGS 84",DATUM["WGS_1984",'
               'SPHEROID["WGS 84",6378137,298.257223563]],'
               'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
               'PROJECTION["Polar_Stereographic"],'
               'PARAMETER["latitude_of_origin",70],'
               'PARAMETER["central_meridian",-45],'
               'PARAMETER["false_easting",0],'
               'PARAMETER["false_northing",0],'
               'UNIT["metre",1]]',
}

#: The projection of each ISMIP ice sheet, by the region token of the file names.
CRS_ATTRS_BY_REGION = {'AIS': EPSG_3031_ATTRS, 'GrIS': EPSG_3413_ATTRS}

#: Names a source file's projection scalar goes by, in the order we look for them.
CRS_NAMES = ('crs', 'polar_stereographic', 'mapping', 'spatial_ref')

#: Coordinate-like variables that never get a ``grid_mapping`` attribute.
COORD_NAMES = {'x', 'y', 'time', 'lat', 'lon', 'latitude', 'longitude', 'time_bnds', 'time_bounds'}


def detect_grid_resolution(nx: int, ny: int) -> Tuple[float, float, str]:
    """
    Detect the ice sheet and grid resolution from the number of grid points.

    Parameters
    ----------
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction

    Returns
    -------
    dx, dy, region : float, float, str
        Grid spacing in meters and the ice sheet, ``'AIS'`` or ``'GrIS'``

    Raises
    ------
    ValueError
        If ``(nx, ny)`` is not a standard ISMIP6 grid of either ice sheet. The
        shape alone cannot say where such a grid sits, so nothing is guessed.
    """
    try:
        region, res = GRID_SHAPES[(int(nx), int(ny))]
    except KeyError:
        known = ', '.join(
            f"{region} {res // 1000} km {sx}x{sy}"
            for (sx, sy), (region, res) in sorted(GRID_SHAPES.items(), key=lambda kv: kv[1]))
        raise ValueError(
            f"Grid shape nx={nx}, ny={ny} is not a standard ISMIP6 grid of either ice sheet; "
            f"cannot reconstruct x/y coordinates. Known shapes (nx x ny): {known}"
        ) from None
    return float(res), float(res), region


def create_coordinates(nx: int, ny: int, dx: float, dy: float,
                       region: str = 'AIS') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create x and y coordinate arrays for an ISMIP6 grid.

    Parameters
    ----------
    nx, ny : int
        Number of grid points
    dx, dy : float
        Grid spacing in meters
    region : str
        ``'AIS'`` (EPSG:3031, the default) or ``'GrIS'`` (EPSG:3413)

    Returns
    -------
    x, y : np.ndarray
        Coordinate arrays in the region's projection, from its lower-left corner
    """
    bounds = GRID_BOUNDS_BY_REGION[region]

    # Span the ISMIP6 domain
    x = np.linspace(bounds['x_min'], bounds['x_max'], nx)
    y = np.linspace(bounds['y_min'], bounds['y_max'], ny)

    # Adjust to match detected spacing if needed
    actual_dx = (x[-1] - x[0]) / (nx - 1) if nx > 1 else 0
    actual_dy = (y[-1] - y[0]) / (ny - 1) if ny > 1 else 0

    if abs(actual_dx - dx) > 1.0:  # More than 1m difference
        x = bounds['x_min'] + np.arange(nx) * dx
    if abs(actual_dy - dy) > 1.0:
        y = bounds['y_min'] + np.arange(ny) * dy

    return x, y


def verify_latlon_consistency(x: np.ndarray, y: np.ndarray,
                               lat: Optional[np.ndarray] = None,
                               lon: Optional[np.ndarray] = None,
                               tolerance: float = 1.0,
                               crs: str = 'EPSG:3031') -> bool:
    """
    Verify that projected coordinates are consistent with lat/lon if provided.

    Parameters
    ----------
    x, y : np.ndarray
        Projected coordinates in ``crs``
    lat, lon : np.ndarray, optional
        Latitude and longitude arrays
    tolerance : float
        Tolerance in degrees for lat/lon matching
    crs : str
        The projection of ``x`` and ``y``; EPSG:3031 (AIS) or EPSG:3413 (GrIS)

    Returns
    -------
    consistent : bool
        True if coordinates are consistent (or if lat/lon not provided)
    """
    if lat is None or lon is None:
        return True  # Can't verify without lat/lon

    try:
        from pyproj import Transformer

        # Create transformer from the grid's projection to WGS84 (lat/lon)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=False)

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
    If missing, tells the ice sheet from the array shape and adds its standard
    grid's coordinates (EPSG:3031 for AIS, EPSG:3413 for GrIS) and projection
    scalar. A shape that is not a standard ISMIP6 grid raises ``ValueError``.

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

    # Detect ice sheet and resolution
    dx, dy, region = detect_grid_resolution(nx, ny)
    crs_attrs = CRS_ATTRS_BY_REGION[region]
    epsg = crs_attrs['spatial_ref']
    print(f"   Detected grid: {region} ({epsg}), dx={dx/1000:.1f} km, dy={dy/1000:.1f} km")

    # Create coordinates
    x, y = create_coordinates(nx, ny, dx, dy, region)
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
        consistent = verify_latlon_consistency(x, y, lat_array, lon_array, crs=epsg)

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
        'x': ('x', x, {'units': 'm', 'long_name': f'x coordinate ({epsg})',
               'standard_name': 'projection_x_coordinate'}),
        'y': ('y', y, {'units': 'm', 'long_name': f'y coordinate ({epsg})',
               'standard_name': 'projection_y_coordinate'})
    })

    # Add grid mapping information
    ds_corrected.attrs['grid_mapping'] = 'polar_stereographic'

    # Add the region's projection as a coordinate
    crs = xr.DataArray(data=np.int32(0), attrs=dict(crs_attrs))
    ds_corrected.coords['polar_stereographic'] = crs

    print("   ✓ Grid correction complete\n")

    return ds_corrected


def region_of_crs(attrs: dict) -> Optional[str]:
    """``'AIS'`` or ``'GrIS'`` from a projection scalar's own parameters, else ``None``."""
    lat0 = attrs.get('latitude_of_projection_origin')
    if lat0 is None:
        code = str(attrs.get('spatial_ref', attrs.get('epsg_code', '')))
        return {'EPSG:3031': 'AIS', 'EPSG:3413': 'GrIS'}.get(code.upper())
    return 'AIS' if float(np.ravel(lat0)[0]) < 0 else 'GrIS'


def gets_grid_mapping(name: str, dims) -> bool:
    """Whether a variable is a field on the (y, x) grid that should point at the projection."""
    dims = tuple(dims or ())
    if name in COORD_NAMES or name in CRS_NAMES or name.endswith(('_bnds', '_bounds')):
        return False
    return 'x' in dims and 'y' in dims


def ensure_grid_mapping(ds: xr.Dataset, region: str = 'AIS') -> xr.Dataset:
    """Give every data variable a ``grid_mapping`` pointing at the group's projection scalar.

    ``region`` ('AIS' or 'GrIS') picks the projection written when the source has
    no scalar; an existing scalar's own parameters win over it for the WKT added.

    A viewer finds the CRS through the variable's ``grid_mapping`` attribute, then
    the group's, then an array named ``crs``. The source files carry a projection
    scalar under several names (or none), and merging variables into one dataset
    drops dataset attributes, so this runs last, after the merge:

    - the first scalar named as in :data:`CRS_NAMES` is the group's projection
      scalar (moved to the coordinates, given ``crs_wkt`` if it lacks one);
    - without one, a ``crs`` coordinate with the region's attributes is created;
    - every data variable on the (y, x) grid and the dataset get ``grid_mapping = <name>``;
      bounds, 1-D coordinates and other projection scalars do not, and any other
      scalar named as in :data:`CRS_NAMES` is moved to the coordinates so it is not
      mistaken for data.
    The input dataset is not modified.
    """
    if region not in CRS_ATTRS_BY_REGION:
        raise ValueError(f"region must be one of {sorted(CRS_ATTRS_BY_REGION)}, got {region!r}")
    ds = ds.copy()
    present = [n for n in CRS_NAMES if n in ds.variables]
    if present:
        name = present[0]
        ds = ds.set_coords([n for n in present if n in ds.data_vars])
        own = region_of_crs(ds[name].attrs) or region
        ds[name].attrs.setdefault('crs_wkt', CRS_ATTRS_BY_REGION[own]['crs_wkt'])
    else:
        name = 'crs'
        ds = ds.assign_coords({name: xr.DataArray(np.int32(0), attrs=dict(CRS_ATTRS_BY_REGION[region]))})
    for v in ds.data_vars:
        if gets_grid_mapping(v, ds[v].dims):
            ds[v].attrs['grid_mapping'] = name
    ds.attrs['grid_mapping'] = name
    return ds
