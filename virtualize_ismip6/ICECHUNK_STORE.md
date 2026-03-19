# Icechunk Store Documentation

## Overview

The ingest pipeline (`virtualize_with_lithops_combine_variables.py`) creates an [Icechunk](https://icechunk.io/) store containing virtualized references to ISMIP6 NetCDF files on source.coop. The store enables efficient Zarr-based access to the full dataset without duplicating the underlying data.

## Store Location

- **Icechunk store**: `s3://us-west-2.opendata.source.coop/englacial/ismip6/icechunk-ais/`
- **Source data**: `s3://us-west-2.opendata.source.coop/englacial/ismip6/`

Both are publicly readable with anonymous S3 access (region: `us-west-2`).

## Store Structure

The store is a single Icechunk repository with three top-level groups, each serving a different use case:

```
combined/{institution}_{model_name}/{experiment}/   # all variables, year-binned time
state/{institution}_{model_name}/{experiment}/      # state variables only, native time
flux/{institution}_{model_name}/{experiment}/       # flux variables only, native time
```

For example:
- `combined/AWI_PISM1/exp05/` -- all variables for AWI/PISM1 experiment 05, time binned to Jan 1 of each year
- `state/DOE_MALI/ctrl_proj_std/` -- state variables (lithk, orog, base, etc.) at native time resolution
- `flux/NCAR_CISM/expD11/` -- flux variables (acabf, dlithkdt, etc.) at native time resolution

Each experiment group contains the merged variables as Zarr arrays, plus shared coordinate arrays (`time`, `x`, `y`). Data variable chunks are virtual references to byte ranges in the source NetCDF files; coordinate arrays are stored inline.

### Store types

| Store type | Group prefix | Time handling | Variables |
|------------|-------------|---------------|-----------|
| `combined` | `combined/` | Binned to Jan 1 of each year | All (state + flux) |
| `state` | `state/` | Native resolution | State only (ST): lithk, orog, base, etc. |
| `flux` | `flux/` | Native resolution | Flux only (FL): acabf, dlithkdt, etc. |

The `combined` store bins time to annual resolution so that state variables (sampled at Jan 1) and flux variables (sampled at Jul 1) share a common time axis. The `state` and `flux` stores preserve the original time values.

## How the Pipeline Works

The pipeline is implemented in `virtualize_with_lithops_combine_variables.py` and uses Lithops to run virtualization on AWS Lambda.

### Step 1: Build File Index

Scans the source bucket to identify all NetCDF files. Results are cached locally as a Parquet file.

### Step 2: Group and Filter

Files are grouped by `(institution, model_name, experiment)`. Each group becomes a batch. Depending on `--store-type`:

- **combined**: All variables included, time will be year-binned
- **state/flux**: Only variables matching the type filter are included

A skip list (`skip_list.txt`) excludes known-bad files (0 bytes, malformed time, etc.).

For incremental updates, the pipeline checks which groups already exist in the store and only processes groups with missing variables. If the new variables have incompatible time axes, the pipeline automatically falls back to rewriting the entire group.

### Step 3: Virtualize (on Lambda)

For each batch, a Lithops serverless function:

1. Opens each file with `virtualizarr.open_virtual_dataset()` (HDF5/NetCDF4) or falls back to kerchunk (NetCDF3)
2. Loads coordinate variables (`time`, `x`, `y`, `lat`, `lon`) into memory
3. Keeps data variables as virtual references (no data copied)
4. Applies ISMIP6-specific fixes:
   - `fix_time_encoding()` -- corrects malformed time attributes
   - `normalize_time_encoding()` -- converts to standard epoch/calendar
   - `bin_time_to_year()` -- (combined store only) rounds time to Jan 1
   - `correct_grid_coordinates()` -- fixes grid metadata
5. Merges all variables in the batch via `merge_virtual_datasets()`, which:
   - Rechunks to per-timestep chunks
   - Computes the union time axis across all variables
   - Pads each variable's manifest to the union axis (missing timesteps resolve to NaN)

### Step 4: Write to Icechunk

Writes are parallelized by model (experiments within each model are sequential to avoid rebase conflicts). Each write:

1. Opens a writable session on the `main` branch
2. Writes the virtual dataset to the group path
3. Commits with automatic rebase handling

### Step 5: Annotate Ignore Values

A post-write pass (`--annotate-only`) detects sentinel values in data arrays. ISMIP6 models zero-fill regions outside the ice sheet domain, but these zeros are literal values, not NaN or fill values. The annotation step:

1. Reads corner pixels of each spatial array (guaranteed ocean for the Antarctic grid)
2. If all corners share a consistent value, records it as `ignore_value` in the array attributes
3. Computes `valid_min`/`valid_max` when the sentinel falls outside the data range

## Accessing the Store

```python
import icechunk
import xarray as xr

SOURCE_BUCKET = "s3://us-west-2.opendata.source.coop/englacial/ismip6"

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="englacial/ismip6/icechunk-ais",
    region="us-west-2",
    anonymous=True,
)

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        SOURCE_BUCKET + "/",
        store=icechunk.s3_store(region="us-west-2", anonymous=True),
    )
)
credentials = icechunk.containers_credentials({SOURCE_BUCKET + "/": None})

repo = icechunk.Repository.open(
    storage=storage,
    config=config,
    authorize_virtual_chunk_access=credentials,
)

session = repo.readonly_session(branch="main")
store = session.store
ds = xr.open_zarr(store, group="combined/AWI_PISM1/exp05", consolidated=False)
```

See also: [notebooks/open_icechunk.ipynb](./notebooks/open_icechunk.ipynb)

## Pipeline Invocation

```bash
# Build all three store types sequentially
python virtualize_with_lithops_combine_variables.py \
    --config lithops_aws.yaml \
    --write-creds sc_creds.json

# Build a specific store type
python virtualize_with_lithops_combine_variables.py \
    --config lithops_aws.yaml \
    --write-creds sc_creds.json \
    --store-type flux

# Run only the ignore-value annotation pass
python virtualize_with_lithops_combine_variables.py \
    --config lithops_aws.yaml \
    --write-creds sc_creds.json \
    --annotate-only

# Test with a single model
python virtualize_with_lithops_combine_variables.py \
    --config lithops_aws.yaml \
    --write-creds sc_creds.json \
    --store-type state \
    --test-model PISM1
```

The Lambda runtime must be built and deployed before running:

```bash
lithops runtime build ismip6-icechunk -f Dockerfile.lithops --config lithops_aws.yaml
lithops runtime deploy ismip6-icechunk --config lithops_aws.yaml
```
