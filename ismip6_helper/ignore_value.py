"""
Detect sentinel / ignore values in ISMIP6 virtual datasets.

ISMIP6 data arrays are zero-filled in regions outside the ice sheet model
domain (ocean, ice-free areas).  These zeros are not NaN or fill values —
they are literal 0.0 written by the model.  This module detects such
sentinels by reading corners of the spatial grid (which are guaranteed to
be ocean for the Antarctic grid) and optionally computes CF-compliant
``valid_min`` / ``valid_max`` attributes so downstream consumers can mask
them without custom logic.

The detection runs as a post-write annotation step: after a batch is
written to icechunk, the virtual chunks are resolvable through the store,
so we can read actual data values without extra S3 downloads.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Coordinate / dimension names to skip during annotation
COORD_NAMES = frozenset([
    "x", "y", "lat", "lon", "latitude", "longitude",
    "time", "t", "bnds", "bounds", "time_bnds", "time_bounds",
    "x_bnds", "y_bnds", "lat_bnds", "lon_bnds", "nv4",
    "mapping", "crs", "spatial_ref",
])


def _read_corner_values(data: np.ndarray) -> np.ndarray:
    """Extract a small set of corner values from a 2-D spatial array.

    Reads the four 2x2 corners of the grid.  For a 3-D array (time, y, x)
    the first time slice is used.

    Returns a 1-D array of the corner values (up to 16 values).
    """
    if data.ndim == 3:
        data = data[0]  # first time slice
    if data.ndim != 2:
        return np.array([])

    ny, nx = data.shape
    if ny < 2 or nx < 2:
        return np.array([])

    # Four 2x2 corners
    corners = np.concatenate([
        data[:2, :2].ravel(),      # top-left
        data[:2, -2:].ravel(),     # top-right
        data[-2:, :2].ravel(),     # bottom-left
        data[-2:, -2:].ravel(),    # bottom-right
    ])
    return corners


def detect_ignore_value(corner_values: np.ndarray) -> Optional[float]:
    """Determine whether corners contain a consistent sentinel value.

    Parameters
    ----------
    corner_values : np.ndarray
        Values extracted from the grid corners.

    Returns
    -------
    The sentinel value if all corners are finite and identical, else None.
    """
    if corner_values.size == 0:
        return None

    # If corners are already NaN -> properly encoded, nothing to do
    if np.all(np.isnan(corner_values)):
        return None

    # Remove NaN/Inf before checking for a consistent sentinel
    finite = corner_values[np.isfinite(corner_values)]
    if finite.size == 0:
        return None

    # All finite corner values must be the same value
    candidate = finite[0]
    if np.all(finite == candidate):
        return float(candidate)

    return None


def compute_valid_range(
    data: np.ndarray,
    sentinel: float,
    fill_value: Optional[float] = None,
) -> Optional[tuple]:
    """Compute (valid_min, valid_max) after masking the sentinel.

    Returns the range only if the sentinel falls outside it — meaning
    CF ``valid_min`` / ``valid_max`` is sufficient to exclude it.
    Returns None if the sentinel is inside the valid range (in which case
    only ``ignore_value`` can handle it).

    Parameters
    ----------
    data : np.ndarray
        The full data array (2-D or 3-D).
    sentinel : float
        The detected sentinel value.
    fill_value : float, optional
        Additional fill value to exclude from range computation.
    """
    flat = data.ravel()

    # Build a mask: exclude NaN, Inf, sentinel, and fill_value
    mask = np.isfinite(flat) & (flat != sentinel)
    if fill_value is not None and np.isfinite(fill_value):
        mask &= (flat != fill_value)

    valid = flat[mask]
    if valid.size == 0:
        return None

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))

    # Sentinel must fall outside the valid range for CF attrs to work
    if vmin <= sentinel <= vmax:
        return None

    return (vmin, vmax)


def annotate_store_group(repo, group_path: str) -> bool:
    """Detect and annotate ignore values for all variables in a written group.

    Opens a writable session on the icechunk repo, reads corner pixels of
    each 2-D+ data variable via zarr, detects sentinel values, and updates
    the variable attrs with ``ignore_value`` and optionally ``valid_min`` /
    ``valid_max``.  Commits only if any annotations were added.

    This is designed to run immediately after ``batch_write_func`` succeeds.

    Parameters
    ----------
    repo : icechunk.Repository
        The icechunk repository (already opened).
    group_path : str
        Path to the group just written (e.g. "combined/AWI_PISM1/exp05").

    Returns
    -------
    True if annotations were committed, False if nothing to annotate.
    """
    import icechunk
    import zarr

    session = repo.writable_session("main")
    store = session.store
    root = zarr.open(store, mode="a")

    try:
        group = root[group_path]
    except KeyError:
        logger.warning("[ignore_value] Group %s not found in store", group_path)
        return False

    annotated = []

    for var_name in list(group.keys()):
        if var_name.lower() in COORD_NAMES:
            continue

        try:
            arr = group[var_name]
        except Exception as e:
            logger.warning("[ignore_value] Could not open %s/%s: %s", group_path, var_name, e)
            continue

        # Only process 2-D and 3-D spatial arrays
        if arr.ndim < 2:
            continue

        # Read corner pixels only (small slice, resolves through virtual chunks)
        try:
            if arr.ndim == 3:
                # Read first time slice, full spatial extent would be wasteful.
                # Instead read just the four 2x2 corners.
                ny, nx = arr.shape[1], arr.shape[2]
                tl = np.asarray(arr[0, :2, :2])
                tr = np.asarray(arr[0, :2, -2:])
                bl = np.asarray(arr[0, -2:, :2])
                br = np.asarray(arr[0, -2:, -2:])
                corner_values = np.concatenate([tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()])
            elif arr.ndim == 2:
                ny, nx = arr.shape
                tl = np.asarray(arr[:2, :2])
                tr = np.asarray(arr[:2, -2:])
                bl = np.asarray(arr[-2:, :2])
                br = np.asarray(arr[-2:, -2:])
                corner_values = np.concatenate([tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()])
            else:
                continue
        except Exception as e:
            logger.warning("[ignore_value] Could not read corners of %s/%s: %s", group_path, var_name, e)
            continue

        sentinel = detect_ignore_value(corner_values)
        if sentinel is None:
            logger.debug("[ignore_value] %s/%s: no sentinel detected", group_path, var_name)
            continue

        logger.info("[ignore_value] %s/%s: detected sentinel = %s", group_path, var_name, sentinel)

        # Set the custom ignore_value attribute
        arr.attrs['ignore_value'] = sentinel

        # Read a full time slice to compute valid range
        # (this loads one 2-D slice — for 761x761 float32 that's ~2.3 MB)
        try:
            if arr.ndim == 3:
                full_slice = np.asarray(arr[0, :, :])
            else:
                full_slice = np.asarray(arr[:, :])
        except Exception as e:
            logger.warning("[ignore_value] Could not read full slice of %s/%s for range: %s",
                           group_path, var_name, e)
            annotated.append(var_name)
            continue

        # Get fill_value from zarr metadata if available
        fv = arr.fill_value
        fill_value = float(fv) if fv is not None and np.isfinite(float(fv)) else None

        valid_range = compute_valid_range(full_slice, sentinel, fill_value)
        if valid_range is not None:
            vmin, vmax = valid_range
            arr.attrs['valid_min'] = vmin
            arr.attrs['valid_max'] = vmax
            logger.info(
                "[ignore_value] %s/%s: sentinel %.1f outside valid range [%.4g, %.4g], "
                "set valid_min/valid_max",
                group_path, var_name, sentinel, vmin, vmax,
            )
        else:
            logger.info(
                "[ignore_value] %s/%s: sentinel %.1f is inside valid range, "
                "using ignore_value attr only",
                group_path, var_name, sentinel,
            )

        annotated.append(var_name)

    if not annotated:
        return False

    try:
        commit_msg = f"Annotate ignore_value for {group_path}: {', '.join(annotated)}"
        session.commit(
            commit_msg,
            rebase_with=icechunk.BasicConflictSolver(),
            rebase_tries=5,
        )
        logger.info("[ignore_value] Committed annotations for %s (%d variables)",
                     group_path, len(annotated))
        return True
    except Exception as e:
        logger.warning("[ignore_value] Failed to commit annotations for %s: %s", group_path, e)
        return False
