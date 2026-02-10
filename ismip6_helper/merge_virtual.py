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

    ds = ds.copy()
    time_values = ds['time'].values

    dates = cftime.num2date(
        time_values,
        units=STANDARD_TIME_UNITS,
        calendar=STANDARD_TIME_CALENDAR,
    )

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

    # Build index mapping: for each union timestep, find matching dataset timestep
    # Returns -1 for missing positions
    mapping = np.full(n_union, -1, dtype=np.intp)
    for i, ut in enumerate(union_time):
        diffs = np.abs(ds_time - ut)
        best = np.argmin(diffs)
        if diffs[best] <= tolerance:
            mapping[i] = best

    # Extract original manifest entries
    chunk_grid_shape = (n_union,) + (1,) * len(spatial_shape)

    paths = np.empty(chunk_grid_shape, dtype=np.dtypes.StringDType())
    # Initialize to empty string (missing chunk)
    paths[:] = ""

    offsets = np.zeros(chunk_grid_shape, dtype=np.dtype("uint64"))
    lengths = np.zeros(chunk_grid_shape, dtype=np.dtype("uint64"))

    # Copy entries from original manifest for matched positions
    old_entries = list(marr.manifest.values())
    for i, src_idx in enumerate(mapping):
        if src_idx >= 0:
            entry = old_entries[src_idx]
            idx = (i,) + (0,) * len(spatial_shape)
            paths[idx] = entry["path"]
            offsets[idx] = entry["offset"]
            lengths[idx] = entry["length"]

    new_manifest = ChunkManifest.from_arrays(
        paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
    )

    # Update metadata: new shape, fill_value=NaN
    new_shape = (n_union,) + spatial_shape
    chunk_shape = (1,) + spatial_shape
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

        # Check for per-timestep chunks
        if marr.manifest.shape_chunk_grid[0] != marr.shape[0]:
            logger.warning(
                "Variable %s has compressed time chunks (%s), skipping padding",
                name, marr.manifest.shape_chunk_grid
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
    coords = {"time": new_time}
    for name in vds.coords:
        if name != "time":
            coords[name] = vds.coords[name]

    return xr.Dataset(all_vars, coords=coords)


def _collect_datasets(datasets: list) -> xr.Dataset:
    """
    Combine multiple datasets by directly collecting variables and coordinates.

    This avoids xr.merge() which triggers xarray's chunk manager lookup for
    ManifestArray, failing in environments without dask (e.g. AWS Lambda).
    All datasets must already share compatible dimensions.
    """
    all_vars = {}
    all_coords = {}
    for ds in datasets:
        for name, var in ds.data_vars.items():
            all_vars[name] = var
        for name, coord in ds.coords.items():
            if name not in all_coords:
                all_coords[name] = coord

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
