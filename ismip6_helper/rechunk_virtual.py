"""
Virtual rechunking for contiguous uncompressed HDF5 data.

When source NetCDF files store 3D arrays (time, y, x) as a single contiguous
uncompressed block, VirtualiZarr creates one huge virtual chunk covering the
entire array. This module splits that single chunk into per-time-slice virtual
references, so each chunk is (1, ny, nx) and can be fetched independently.
"""

import dataclasses

import numpy as np
import xarray as xr
from virtualizarr.manifests import ChunkManifest, ManifestArray
from zarr.core.chunk_grids import RegularChunkGrid


def rechunk_contiguous_time_axis(vds: xr.Dataset) -> xr.Dataset:
    """
    Rechunk contiguous uncompressed 3D variables along the time axis.

    For each data variable backed by a ManifestArray with a single chunk
    covering the full array, split into per-time-slice chunks by computing
    byte offsets into the contiguous data block.

    Variables that are already multi-chunked, non-3D, or compressed are
    left unchanged.

    Parameters
    ----------
    vds : xr.Dataset
        Virtual dataset from VirtualiZarr.

    Returns
    -------
    xr.Dataset with rechunked ManifestArrays where applicable.
    """
    new_vars = {}
    for name in vds.data_vars:
        var = vds[name]
        new_vars[name] = _rechunk_variable(var)

    return vds.assign(new_vars)


def _rechunk_variable(var: xr.Variable) -> xr.Variable:
    """Rechunk a single variable if it meets the criteria."""
    if not isinstance(var.data, ManifestArray):
        return var

    marr = var.data

    # Only handle 3D arrays (time, y, x)
    if len(marr.shape) != 3:
        return var

    # Only handle single-chunk (contiguous) arrays
    if marr.manifest.shape_chunk_grid != (1, 1, 1):
        return var

    n_time, ny, nx = marr.shape
    itemsize = marr.dtype.itemsize
    expected_length = n_time * ny * nx * itemsize
    slice_bytes = ny * nx * itemsize

    # Get the single chunk entry
    entry = list(marr.manifest.values())[0]
    base_path = entry["path"]
    base_offset = entry["offset"]
    total_length = entry["length"]

    # Verify uncompressed: byte length must match shape * dtype size
    if total_length != expected_length:
        return var

    # Build per-time-slice manifest arrays
    chunk_grid_shape = (n_time, 1, 1)

    paths = np.empty(chunk_grid_shape, dtype=np.dtypes.StringDType())
    paths[:] = base_path

    offsets = np.empty(chunk_grid_shape, dtype=np.dtype("uint64"))
    lengths = np.empty(chunk_grid_shape, dtype=np.dtype("uint64"))

    for t in range(n_time):
        offsets[t, 0, 0] = base_offset + t * slice_bytes
        lengths[t, 0, 0] = slice_bytes

    new_manifest = ChunkManifest.from_arrays(
        paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
    )

    # Update metadata with new chunk shape via dataclasses.replace
    new_chunk_grid = RegularChunkGrid(chunk_shape=(1, ny, nx))
    new_metadata = dataclasses.replace(marr.metadata, chunk_grid=new_chunk_grid)

    new_marr = ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)

    return xr.Variable(dims=var.dims, data=new_marr, attrs=var.attrs, encoding=var.encoding)
