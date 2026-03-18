# ISMIP6 Data Indexing and Ingest Pipeline

TL;DR: Browse the data at [englacial.org/static/models/](https://englacial.org/static/models/), or read more at [englacial.org/models.html](https://englacial.org/models.html).

## Overview

This repository contains tools for indexing, ingesting, and serving [ISMIP6](https://www.ismip.org/) Antarctic ice sheet model outputs. We are not associated with ISMIP. These are tools for publicly-available data that we hope are interesting and useful to the scientific community.

There are three main components:

1. **Ingest pipeline** -- Virtualizes NetCDF source files into an [Icechunk](https://icechunk.io/) store using [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr) and [Lithops](https://lithops-cloud.github.io/) serverless functions on AWS Lambda.
2. **Python library** (`ismip6_helper`) -- Handles file indexing, grid correction, time encoding normalization, and ignore-value detection.
3. **Static indexing site** -- Catalogs available outputs at [docs.englacial.org/ismip-indexing/](https://docs.englacial.org/ismip-indexing/).

The interactive web viewer lives in a separate repository: [englacial/ismip-viewer](https://github.com/englacial/ismip-viewer).

## Data

### Source files

A copy of the ISMIP6 outputs (originally [available through Globus](https://theghub.org/accessing-data-with-globus)) is hosted on source.coop:

```
s3://us-west-2.opendata.source.coop/englacial/ismip6/
```

Public, anonymous read access. No authentication required. For citation guidance, see the [ISMIP wiki](https://theghub.org/groups/ismip6/wiki/PublicationsCitationGuidance).

### Icechunk store

The ingest pipeline writes a virtualized Icechunk store to:

```
s3://us-west-2.opendata.source.coop/englacial/ismip6/icechunk-ais/
```

This store contains virtual references to chunks in the source NetCDF files -- no data is duplicated. It is organized into three top-level groups:

- **`combined/`** -- All variables merged per model+experiment, with time binned to annual resolution
- **`state/`** -- State variables only (e.g. `lithk`, `orog`, `base`), native time resolution
- **`flux/`** -- Flux variables only (e.g. `acabf`, `dlithkdt`), native time resolution

See [ICECHUNK_STORE.md](ICECHUNK_STORE.md) for details on the store structure and how the pipeline works.

### Data overview

- **10,034 files** (~1.1 TB total)
- **17 models** from 14 institutions
- **94 experiments**
- **37 variables**
- All Antarctic ice sheet (AIS) data

## Developers

### Setup

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v -m "not integration"
```

### Running the ingest pipeline

The pipeline virtualizes source NetCDF files and writes them to the Icechunk store using Lithops on AWS Lambda:

```bash
# Build all three store types (combined, state, flux)
python virtualize_with_lithops_combine_variables.py \
    --config lithops_aws.yaml \
    --write-creds sc_creds.json

# Or build a specific store type
python virtualize_with_lithops_combine_variables.py \
    --config lithops_aws.yaml \
    --write-creds sc_creds.json \
    --store-type flux
```

See `python virtualize_with_lithops_combine_variables.py --help` for all options, and [lithops_aws.md](lithops_aws.md) for AWS infrastructure setup.

### Python API

The `ismip6_helper` library provides utilities for working with ISMIP6 data:

```python
from ismip6_helper import get_file_index

# Get file index (cached locally)
df = get_file_index()

# Force rebuild from source bucket
df = get_file_index(force_rebuild=True)
```

Key modules:

- `index` -- File indexing and path parsing
- `grid_utils` -- Grid coordinate correction
- `time_utils` -- Time encoding normalization
- `merge_virtual` -- Union time axis computation and manifest padding
- `variable_classification` -- State/flux variable classification
- `ignore_value` -- Sentinel value detection and annotation
