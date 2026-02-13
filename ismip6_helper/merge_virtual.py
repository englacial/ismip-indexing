"""
Merge virtual datasets with different time axes.

When merging ISMIP6 virtual datasets for different variables, flux and state
variables can have different time axes. VirtualiZarr can't reindex ManifestArrays
(fancy indexing not supported), so xr.merge(join='outer') fails.

This module works around the limitation by:
1. Binning all time values to Jan 1 of each year (eliminates flux/state date offset)
2. Computing the union time axis across all variables in a group
3. Padding each variable's manifest with empty-path chunks for missing timesteps
4. Merging (now safe because all variables share the same time axis)

Missing chunks resolve to fill_value (set to NaN) at read time in Zarr/icechunk.

This module can be deleted when VirtualiZarr upstream fixes land:
- https://github.com/zarr-developers/VirtualiZarr/issues/51
- https://github.com/zarr-developers/VirtualiZarr/issues/382
"""

import dataclasses
import logging

import cftime
import numpy as np
import xarray as xr
from virtualizarr.manifests import ChunkManifest, ManifestArray
from zarr.core.chunk_grids import RegularChunkGrid

from .rechunk_virtual import rechunk_contiguous_time_axis
from .time_utils import STANDARD_TIME_UNITS, STANDARD_TIME_CALENDAR

logger = logging.getLogger(__name__)


def bin_time_to_year(ds: xr.Dataset) -> xr.Dataset:
    """
    Round each time value to Jan 1 of its year.

    After normalization, time values are float64 "days since 2000-01-01" in
    proleptic_gregorian. This converts each to a cftime date, extracts the year,
    and replaces the value with the Jan 1 date for that year.

    After binning:
    - State variable at Jan 1 2015 -> Jan 1 2015 (unchanged)
    - Flux variable at Jul 1 2015 -> Jan 1 2015 (rounded to year start)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with normalized time encoding.

    Returns
    -------
    xr.Dataset with time values binned to Jan 1 of each year.
    """
    if 'time' not in ds.variables:
        return ds

    # Check that time encoding was actually normalized. normalize_time_encoding
    # can silently return the dataset unchanged on failure (e.g. overflow).
    # If the attrs don't match, binning with the standard constants would produce
    # garbage or overflow.
    time_attrs = ds['time'].attrs
    if time_attrs.get('units') != STANDARD_TIME_UNITS or \
       time_attrs.get('calendar') != STANDARD_TIME_CALENDAR:
        logger.warning(
            "Time encoding not normalized (units=%r, calendar=%r), skipping year binning",
            time_attrs.get('units'), time_attrs.get('calendar'),
        )
        return ds

    ds = ds.copy()
    time_values = ds['time'].values

    try:
        dates = cftime.num2date(
            time_values,
            units=STANDARD_TIME_UNITS,
            calendar=STANDARD_TIME_CALENDAR,
        )
    except Exception as e:
        logger.warning("Failed to decode time values for year binning: %s", e)
        return ds

    jan1_dates = [
        cftime.datetime(dt.year, 1, 1, calendar=STANDARD_TIME_CALENDAR)
        for dt in np.asarray(dates).flat
    ]

    new_values = cftime.date2num(
        jan1_dates,
        units=STANDARD_TIME_UNITS,
        calendar=STANDARD_TIME_CALENDAR,
    )
    new_values = np.asarray(new_values, dtype=np.float64).reshape(time_values.shape)

    ds['time'] = xr.Variable(
        dims=ds['time'].dims,
        data=new_values,
        attrs=ds['time'].attrs,
    )

    return ds


def compute_union_time_axis(vdatasets: list, tolerance_days: float = 0.5) -> np.ndarray:
    """
    Compute the sorted union of time values across all datasets.

    Collects time values from all datasets that have a 'time' coordinate,
    then deduplicates within a tolerance (0.5 days handles floating-point
    noise after year-binning).

    Parameters
    ----------
    vdatasets : list of xr.Dataset
        Virtual datasets, each potentially with a 'time' coordinate.
    tolerance_days : float
        Values within this many days are considered duplicates.

    Returns
    -------
    np.ndarray of float64, sorted unique time values.
    """
    all_times = []
    for ds in vdatasets:
        if 'time' in ds.variables:
            # Only include time from datasets where normalization succeeded.
            # Un-normalized datasets (e.g. BISICLES lim/limnsw with garbage
            # time values) still have their original encoding attrs, so their
            # values are in unknown units and must not be mixed into the union.
            time_attrs = ds['time'].attrs
            if time_attrs.get('units') != STANDARD_TIME_UNITS or \
               time_attrs.get('calendar') != STANDARD_TIME_CALENDAR:
                logger.warning(
                    "Skipping non-normalized time from union axis (units=%r, calendar=%r)",
                    time_attrs.get('units'), time_attrs.get('calendar'),
                )
                continue
            all_times.append(ds['time'].values.ravel())

    if not all_times:
        return np.array([], dtype=np.float64)

    combined = np.concatenate(all_times)
    combined = np.sort(combined)

    # Deduplicate within tolerance
    if len(combined) == 0:
        return combined

    unique = [combined[0]]
    for v in combined[1:]:
        if v - unique[-1] > tolerance_days:
            unique.append(v)

    return np.array(unique, dtype=np.float64)


def _pad_manifest_to_union(
    var: xr.Variable,
    union_time: np.ndarray,
    ds_time: np.ndarray,
    tolerance: float = 0.5,
) -> xr.Variable:
    """
    Pad a ManifestArray variable to match the union time axis.

    For each position in the union time axis, finds the matching position in
    the dataset's time axis (within tolerance). Matched positions copy the
    original manifest entry; missing positions get empty-path chunks that
    resolve to fill_value at read time.

    Preserves the spatial chunk grid from the original manifest. For example,
    if the source has chunk shape (1, 381, 381) with a 2x2 spatial grid,
    the padded manifest retains the same spatial chunking.

    Parameters
    ----------
    var : xr.Variable
        Variable backed by a ManifestArray with per-timestep chunks.
    union_time : np.ndarray
        The target union time axis.
    ds_time : np.ndarray
        The dataset's original time values.
    tolerance : float
        Tolerance in days for matching time values.

    Returns
    -------
    xr.Variable with padded ManifestArray.
    """
    marr = var.data
    old_shape = marr.shape
    n_union = len(union_time)
    n_old_time = old_shape[0]
    spatial_shape = old_shape[1:]  # (ny, nx) or similar

    # Preserve the spatial chunk grid from the original manifest.
    # For contiguous arrays this is (1, 1, ...), but for chunked HDF5 files
    # (e.g. LSCE_GRISLI2 with chunk shape (1, 381, 381) on a 761x761 grid)
    # this can be (2, 2) or similar.
    old_chunk_grid_shape = marr.manifest.shape_chunk_grid  # e.g. (87, 2, 2)
    spatial_chunk_grid = old_chunk_grid_shape[1:]  # e.g. (2, 2)
    n_spatial_chunks = int(np.prod(spatial_chunk_grid))

    # Build index mapping: for each union timestep, find matching dataset timestep
    # Returns -1 for missing positions
    mapping = np.full(n_union, -1, dtype=np.intp)
    for i, ut in enumerate(union_time):
        diffs = np.abs(ds_time - ut)
        best = np.argmin(diffs)
        if diffs[best] <= tolerance:
            mapping[i] = best

    # New chunk grid preserves spatial chunking
    chunk_grid_shape = (n_union,) + tuple(spatial_chunk_grid)

    paths = np.empty(chunk_grid_shape, dtype=np.dtypes.StringDType())
    # Initialize to empty string (missing chunk)
    paths[:] = ""

    offsets = np.zeros(chunk_grid_shape, dtype=np.dtype("uint64"))
    lengths = np.zeros(chunk_grid_shape, dtype=np.dtype("uint64"))

    # Build a lookup of original entries indexed by chunk grid position.
    # entries are in C-order (last index varies fastest), matching np.ndindex.
    old_entries = list(marr.manifest.values())
    old_entries_by_idx = {}
    for idx, entry in zip(np.ndindex(old_chunk_grid_shape), old_entries):
        old_entries_by_idx[idx] = entry

    # Copy entries from original manifest for matched positions,
    # preserving all spatial chunks per timestep.
    for i, src_idx in enumerate(mapping):
        if src_idx >= 0:
            for spatial_idx in np.ndindex(tuple(spatial_chunk_grid)):
                old_key = (src_idx,) + spatial_idx
                entry = old_entries_by_idx.get(old_key)
                if entry is not None:
                    new_key = (i,) + spatial_idx
                    paths[new_key] = entry["path"]
                    offsets[new_key] = entry["offset"]
                    lengths[new_key] = entry["length"]

    new_manifest = ChunkManifest.from_arrays(
        paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
    )

    # Update metadata: new shape, fill_value=NaN, preserve original chunk shape
    new_shape = (n_union,) + spatial_shape
    original_chunk_shape = tuple(marr.metadata.chunk_grid.chunk_shape)
    new_chunk_grid = RegularChunkGrid(chunk_shape=original_chunk_shape)
    new_metadata = dataclasses.replace(
        marr.metadata,
        shape=new_shape,
        fill_value=np.float64('nan'),
        chunk_grid=new_chunk_grid,
    )

    new_marr = ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)

    return xr.Variable(dims=var.dims, data=new_marr, attrs=var.attrs, encoding=var.encoding)


def _pad_manifest_chunks_to_union(
    var: xr.Variable,
    union_time: np.ndarray,
    ds_time: np.ndarray,
    tolerance: float = 0.5,
) -> xr.Variable:
    """
    Pad a ManifestArray with multi-timestep chunks to match the union time axis.

    Unlike _pad_manifest_to_union (which works per-timestep), this handles
    variables where each time chunk covers multiple timesteps (e.g. BISICLES
    with chunk_shape (18, 153, 153) and chunk_grid (5, 5, 5) covering 90 steps).

    We can't split compressed multi-timestep chunks, but we CAN do chunk-level
    padding: keep all existing time chunks and append empty chunks for new
    time-chunk positions at the end of the union axis.

    Parameters
    ----------
    var : xr.Variable
        Variable backed by a ManifestArray with multi-timestep chunks.
    union_time : np.ndarray
        The target union time axis.
    ds_time : np.ndarray
        The dataset's original time values.
    tolerance : float
        Tolerance in days for matching time values.

    Returns
    -------
    xr.Variable with padded ManifestArray.
    """
    import math

    marr = var.data
    old_shape = marr.shape
    spatial_shape = old_shape[1:]
    n_union = len(union_time)

    chunk_shape = tuple(marr.metadata.chunk_grid.chunk_shape)
    time_chunk_size = chunk_shape[0]

    old_chunk_grid = marr.manifest.shape_chunk_grid
    n_orig_time_chunks = old_chunk_grid[0]
    spatial_chunk_grid = old_chunk_grid[1:]
    n_spatial_chunks = int(np.prod(spatial_chunk_grid))

    n_union_time_chunks = math.ceil(n_union / time_chunk_size)
    new_chunk_grid_shape = (n_union_time_chunks,) + tuple(spatial_chunk_grid)

    paths = np.empty(new_chunk_grid_shape, dtype=np.dtypes.StringDType())
    paths[:] = ""
    offsets = np.zeros(new_chunk_grid_shape, dtype=np.dtype("uint64"))
    lengths = np.zeros(new_chunk_grid_shape, dtype=np.dtype("uint64"))

    # Build lookup of original entries by chunk grid index
    old_entries = list(marr.manifest.values())
    old_entries_by_idx = {}
    for idx, entry in zip(np.ndindex(tuple(old_chunk_grid)), old_entries):
        old_entries_by_idx[idx] = entry

    # For each new time-chunk position, check if the corresponding timestep
    # range overlaps with the original dataset's time axis. If the first
    # timestep of that chunk range matches an original chunk, copy all spatial
    # entries for that time-chunk position.
    for tc in range(n_union_time_chunks):
        first_step = tc * time_chunk_size
        if first_step >= n_union:
            break

        union_val = union_time[first_step]

        # Find matching original time-chunk
        matched_orig_tc = None
        for orig_tc in range(n_orig_time_chunks):
            orig_first_step = orig_tc * time_chunk_size
            if orig_first_step < len(ds_time):
                if abs(ds_time[orig_first_step] - union_val) <= tolerance:
                    matched_orig_tc = orig_tc
                    break

        if matched_orig_tc is not None:
            for spatial_idx in np.ndindex(tuple(spatial_chunk_grid)):
                old_key = (matched_orig_tc,) + spatial_idx
                entry = old_entries_by_idx.get(old_key)
                if entry is not None:
                    new_key = (tc,) + spatial_idx
                    paths[new_key] = entry["path"]
                    offsets[new_key] = entry["offset"]
                    lengths[new_key] = entry["length"]

    new_manifest = ChunkManifest.from_arrays(
        paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
    )

    new_shape = (n_union,) + spatial_shape
    new_chunk_grid = RegularChunkGrid(chunk_shape=chunk_shape)
    new_metadata = dataclasses.replace(
        marr.metadata,
        shape=new_shape,
        fill_value=np.float64('nan'),
        chunk_grid=new_chunk_grid,
    )

    new_marr = ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)
    return xr.Variable(dims=var.dims, data=new_marr, attrs=var.attrs, encoding=var.encoding)


def pad_dataset_to_union_time(
    vds: xr.Dataset,
    union_time: np.ndarray,
    tolerance_days: float = 0.5,
) -> xr.Dataset:
    """
    Pad all eligible variables in a dataset to match the union time axis.

    Variable handling:
    - Loaded data (time, x, y, lat, lon): skipped; time replaced at end
    - ManifestArray without time dimension: left as-is
    - ManifestArray with per-timestep chunks: padded
    - ManifestArray not per-timestep (compressed): left unchanged with warning

    Parameters
    ----------
    vds : xr.Dataset
        Virtual dataset (already rechunked to per-timestep).
    union_time : np.ndarray
        The target union time axis.
    tolerance_days : float
        Tolerance for matching time values.

    Returns
    -------
    xr.Dataset with all ManifestArray variables padded to the union time axis.
    """
    if 'time' not in vds.variables:
        return vds

    # Skip time-dependent variables from datasets with un-normalized time.
    # Their time values are in unknown units and can't be matched against
    # the union axis. Writing them would produce arrays with empty manifests
    # (all NaN) that pollute the store. Return only non-time-dependent
    # variables so the downstream merge doesn't try to reindex them.
    time_attrs = vds['time'].attrs
    if time_attrs.get('units') != STANDARD_TIME_UNITS or \
       time_attrs.get('calendar') != STANDARD_TIME_CALENDAR:
        logger.warning(
            "Skipping padding for dataset with non-normalized time (units=%r, calendar=%r)",
            time_attrs.get('units'), time_attrs.get('calendar'),
        )
        # Keep only variables without a time dimension
        keep_vars = {
            name: var for name, var in vds.data_vars.items()
            if 'time' not in var.dims
        }
        keep_coords = {
            name: coord for name, coord in vds.coords.items()
            if name != 'time' and 'time' not in coord.dims
        }
        return xr.Dataset(keep_vars, coords=keep_coords)

    ds_time = vds['time'].values.ravel()
    new_vars = {}

    for name in vds.data_vars:
        var = vds[name]

        if not isinstance(var.data, ManifestArray):
            # Loaded data variable (e.g. lat, lon) -- skip
            continue

        if 'time' not in var.dims:
            # No time dimension (e.g. hfgeoubed) -- leave as-is
            continue

        marr = var.data

        # Check for per-timestep chunks vs multi-timestep chunks
        if marr.manifest.shape_chunk_grid[0] != marr.shape[0]:
            import math
            time_chunk_size = marr.metadata.chunk_grid.chunk_shape[0]
            n_orig_chunks = math.ceil(marr.shape[0] / time_chunk_size)
            n_union_chunks = math.ceil(len(union_time) / time_chunk_size)
            if n_union_chunks >= n_orig_chunks:
                # Union is a superset — can append empty chunks at the end
                new_vars[name] = _pad_manifest_chunks_to_union(
                    var, union_time, ds_time, tolerance_days
                )
            else:
                logger.warning(
                    "Variable %s: multi-timestep chunks and union is shorter, skipping",
                    name,
                )
            continue

        new_vars[name] = _pad_manifest_to_union(var, union_time, ds_time, tolerance_days)

    # Build a new dataset: start with the union time coordinate, then add
    # padded data vars and any unchanged variables. We can't use vds.assign()
    # because the padded variables have a different time dimension size than
    # the original time coordinate.
    new_time = xr.Variable(
        dims=vds['time'].dims,
        data=union_time.copy(),
        attrs=vds['time'].attrs,
    )

    # Collect all variables for the new dataset.
    # Variables that couldn't be padded (compressed chunks) and have a
    # mismatched time dimension must be excluded -- xarray would try to
    # reindex them against the union time, triggering VirtualiZarr's
    # "fancy indexing not supported" error.
    n_union = len(union_time)
    all_vars = {}
    for name in vds.data_vars:
        if name in new_vars:
            all_vars[name] = new_vars[name]
        else:
            var = vds[name]
            if 'time' in var.dims and var.shape[0] != n_union:
                logger.warning(
                    "Dropping variable %s: time dimension %d != union %d and cannot be padded",
                    name, var.shape[0], n_union,
                )
                continue
            all_vars[name] = var

    # Collect non-time coordinates (x, y, lat, lon, etc.)
    # Skip ManifestArray-backed coordinates (e.g. 'bnds') -- xr.Dataset
    # tries to build a pandas index from every coordinate, which calls
    # to_numpy() → get_chunked_array_type() and fails without dask.
    coords = {"time": new_time}
    for name in vds.coords:
        if name != "time":
            coord = vds.coords[name]
            if isinstance(coord.data, ManifestArray):
                logger.warning("Skipping ManifestArray coordinate %s -- add to loadable_variables", name)
                continue
            coords[name] = coord

    return xr.Dataset(all_vars, coords=coords)


def _collect_datasets(datasets: list) -> xr.Dataset:
    """
    Combine multiple datasets by directly collecting variables and coordinates.

    This avoids xr.merge() which triggers xarray's chunk manager lookup for
    ManifestArray, failing in environments without dask (e.g. AWS Lambda).
    All datasets must already share compatible dimensions.

    ManifestArray-backed coordinates (e.g. 'bnds') are skipped because the
    xr.Dataset constructor tries to build pandas indexes from coordinates,
    which calls to_numpy() and triggers get_chunked_array_type().
    """
    all_vars = {}
    all_coords = {}
    for ds in datasets:
        for name, var in ds.data_vars.items():
            all_vars[name] = var
        for name, coord in ds.coords.items():
            if name not in all_coords:
                if isinstance(coord.data, ManifestArray):
                    continue
                all_coords[name] = coord

    # Safety net: detect and drop variables whose time dimension doesn't
    # match the time coordinate. This prevents xarray from trying to
    # reindex ManifestArrays (which triggers "fancy indexing not supported")
    # when datasets with mismatched time sizes are combined without padding
    # (e.g. empty union fallback with BISICLES state=90 vs flux=91 steps).
    if 'time' in all_coords:
        n_time = all_coords['time'].shape[0]
        to_drop = []
        for name, var in all_vars.items():
            if 'time' in var.dims and var.shape[var.dims.index('time')] != n_time:
                logger.warning(
                    "Dropping variable %s from collect: time dim %d != coord %d",
                    name, var.shape[var.dims.index('time')], n_time,
                )
                to_drop.append(name)
        for name in to_drop:
            del all_vars[name]

    return xr.Dataset(all_vars, coords=all_coords)


def merge_virtual_datasets(vdatasets: list) -> xr.Dataset:
    """
    Merge virtual datasets with potentially different time axes.

    Replaces the bare xr.merge() + rechunk_contiguous_time_axis() calls.

    1. Rechunk each dataset to per-timestep chunks
    2. Compute union time axis
    3. If all datasets share the same axis -> fast path, skip padding
    4. Otherwise, pad each dataset to the union axis
    5. Collect variables into a single dataset

    Parameters
    ----------
    vdatasets : list of xr.Dataset
        Virtual datasets to merge (one per variable file).

    Returns
    -------
    xr.Dataset, merged with aligned time axes.
    """
    # Step 1: rechunk each dataset to per-timestep chunks
    rechunked = [rechunk_contiguous_time_axis(ds) for ds in vdatasets]

    # Step 2: compute union time axis
    union_time = compute_union_time_axis(rechunked)

    if len(union_time) == 0:
        # No time dimension in any dataset
        return _collect_datasets(rechunked)

    # Step 3: check if all datasets already share the same time axis
    all_same = True
    for ds in rechunked:
        if 'time' in ds.variables:
            ds_time = ds['time'].values.ravel()
            if len(ds_time) != len(union_time) or not np.allclose(ds_time, union_time, atol=0.5):
                all_same = False
                break

    if all_same:
        logger.debug("All datasets share the same time axis, skipping padding")
        return _collect_datasets(rechunked)

    # Step 4: pad each dataset to match the union time axis
    padded = [pad_dataset_to_union_time(ds, union_time) for ds in rechunked]

    # Step 5: combine into a single dataset
    return _collect_datasets(padded)
