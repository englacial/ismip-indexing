# Icechunk Store Documentation

## Overview

This document describes the Icechunk store created by the `virtualize_with_lithops.py` script for the ISMIP6 dataset. The store contains virtualized references to NetCDF files in Google Cloud Storage, allowing efficient access to the dataset without duplicating the underlying data.

## Store Location

- **Bucket**: `gs://ismip6-icechunk`
- **Source Data**: `gs://ismip6/`

## What is Icechunk and how is it being used

Icechunk is a transactional storage system for chunked array data. Here we are using it to store metadata that references chunks in remote object storage. In this case, the Google Cloud Storage bucket gs://ismip6, enabling efficient access to large scientific datasets without copying the data.

## Store Structure

The Icechunk store is organized hierarchically using the same structure as the `gs://ismip6` source bucket, i.e.:

```
{institution}_{model_name}/{experiment}/{variable}
```

For example:
- `AWI_ISSM/ctrl/lithk`
- `JPL_ISSM/expAE01/ivol`

Each path contains a virtual Zarr dataset with metadata pointing to chunks in the original NetCDF files.

## How virtualize_with_lithops Creates the Store

### Processing Strategy

The script uses a three-step approach to build the Icechunk store:

1. **Build File Index**: Scans the `gs://ismip6/` bucket to identify all available NetCDF files
2. **Group Files**: Groups files by model name and experiment to create logical batches
3. **Process Batches**: For each batch:
   - Virtualizes files in parallel using Lithops serverless functions
   - Writes successful virtualizations to Icechunk in a single commit

### Virtualization Process

#### Step 1: File Virtualization

For each NetCDF file, the `virtualize_file()` function:

1. Opens the file using `virtualizarr.open_virtual_dataset()`
2. Loads coordinate variables (`time`, `x`, `y`, `lat`, `lon`, etc.) into memory
3. References data variables virtually (no data is copied)
4. Applies ISMIP6-specific fixes:
   - `fix_time_encoding()`: Corrects time coordinate encoding
   - `correct_grid_coordinates()`: Fixes grid coordinate metadata
5. Returns the virtual dataset with its hierarchical path

#### Step 2: Batch Writing

The `write_batch_to_icechunk()` function:

1. Opens or creates the Icechunk repository
2. Creates a writable session on the `main` branch
3. Writes all virtual datasets in the batch using `virtual_dataset_to_icechunk()`
4. Commits all changes in a single transaction

This approach minimizes the number of commits and reduces the risk of conflicts.

#### Step 3: Parallel Processing (lines 104-217)

The `process_all_files()`:

1. Groups files by model and experiment to create batches
2. Uses Lithops to parallelize virtualization within each batch
3. Processes batches sequentially to avoid Icechunk write conflicts
4. Logs any failures to `gs://ismip6-icechunk/failures/`

## Accessing the Store

See [notebooks/open_icechunk.ipynb](./notebooks/open_icechunk.ipynb)

## Commit History

Each batch write creates a commit with a message like:
```
Added 5 datasets: AWI_ISSM/ctrl/lithk, AWI_ISSM/ctrl/ivol, AWI_ISSM/ctrl/base, ...
```

The commit ID is logged during processing for traceability.

## Failure Handling

If virtualization or writing fails for any files, failures are logged to:
```
gs://ismip6-icechunk/failures/virtualization_failures_YYYYMMDD_HHMMSS.json
```

Each failure record includes:
- URL of the failed file
- Error message
- Model name and experiment
- Timestamp

## Improvements

* Virtualization is still a bit slow for the entire archive (~6 hours) grouping by model and experiment. Could this be sped up, perhaps virtualizing all files first and / or grouping only by model?
* Reconciliation: If writing fails for some subset of files, what would be a good reconciliation process? That is, compare what is in the zarr store with the `gs://ismip6` index parquet file.
