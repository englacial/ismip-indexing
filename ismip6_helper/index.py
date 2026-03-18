#!/usr/bin/env python3
"""
ISMIP6 file indexing and caching library.

This module provides functionality to index ISMIP6 files from cloud storage
and cache the results locally for faster subsequent access.
"""

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import pandas as pd
import fsspec


def parse_ismip6_path(path: str) -> Optional[dict]:
    """
    Parse an ISMIP6 storage path to extract metadata from directory structure.

    Matches the segment ``ismip6/Projection-{ICE_SHEET}/{INSTITUTION}/{MODEL}/{EXPERIMENT}/{VARIABLE}_....nc``
    anywhere in *path* and reconstructs a full ``s3://`` URL using the canonical
    :data:`~ismip6_helper.SOURCE_DATA_URL`.

    NOTE: UCIJPL/ISSM has a naming issue where the experiment name is incorrectly
    prepended to the variable name. This function automatically detects and corrects it.

    Example of incorrect naming (UCIJPL):
      .../UCIJPL/ISSM/exp13/exp13acabf_AIS_UCIJPL_ISSM_exp13.nc
      Variable extracted from filename: "exp13acabf"
      Corrected to: "acabf" (by stripping the "exp13" prefix)

    Example of correct naming (other models):
      .../AWI/PISM1/exp13/acabf_AIS_AWI_PISM1_exp13.nc
      Variable: "acabf" (no correction needed)

    Affected files: 200 files from UCIJPL/ISSM in experiments:
      exp13, expA5, expA6, expA7, expA8, expB6, expB7, expB8, expB9, expB10

    Args:
        path: Storage path to the file (any scheme or bare path)

    Returns:
        Dictionary with parsed metadata or None if path doesn't match expected format
    """
    from . import SOURCE_DATA_URL

    # Pattern extracts from directory structure and filename
    # Matches: ...ismip6/Projection-{ICE_SHEET}/{INSTITUTION}/{MODEL}/{EXPERIMENT}/{FILENAME}.nc
    pattern = r'ismip6/Projection-([A-Z]+)/([^/]+)/([^/]+)/([^/]+)/([^_]+).*\.nc$'

    match = re.search(pattern, path)
    if match:
        ice_sheet, institution, model, experiment, variable = match.groups()

        # Fix UCIJPL naming issue: variable name incorrectly contains experiment prefix
        # Pattern: variable starts with experiment name (e.g., "exp13acabf" should be "acabf")
        if variable.startswith(experiment):
            corrected_variable = variable[len(experiment):]
            # Only apply correction if it results in a valid variable name (starts with lowercase)
            if corrected_variable and corrected_variable[0].islower():
                variable = corrected_variable

        # Extract the relative path after "ismip6/"
        rel_path = match.group(0)  # e.g. "ismip6/Projection-AIS/AWI/PISM1/exp13/acabf_....nc"
        # Strip the leading "ismip6/" since SOURCE_DATA_URL already ends with /ismip6
        rel_after_ismip6 = rel_path[len("ismip6/"):]
        url = f'{SOURCE_DATA_URL}/{rel_after_ismip6}'

        return {
            'variable': variable,
            'ice_sheet': ice_sheet,
            'institution': institution,
            'model_name': model,
            'experiment': experiment,
            'url': url,
        }

    return None


def build_file_index(source_url: Optional[str] = None,
                     cache_path: Optional[str] = None,
                     force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build an index of all ISMIP6 files in the source bucket.

    Args:
        source_url: S3 URL of the source data (default: :data:`~ismip6_helper.SOURCE_DATA_URL`)
        cache_path: Path to cache file (default: ./.cache/ismip6_index.parquet)
        force_rebuild: If True, ignore cache and rebuild index

    Returns:
        DataFrame with columns: variable, ice_sheet, institution, model_name, experiment, url, size_bytes
    """
    from . import SOURCE_DATA_URL

    if source_url is None:
        source_url = SOURCE_DATA_URL

    # Set up cache path
    if cache_path is None:
        cache_path = Path(".cache/ismip6_index.parquet")
    else:
        cache_path = Path(cache_path)

    # Try to load from cache first
    if not force_rebuild and cache_path.exists():
        print(f"Loading index from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    # Parse the source URL to get protocol, bucket, and prefix
    parsed = urlparse(source_url)
    protocol = parsed.scheme  # e.g. "s3"
    bucket = parsed.netloc     # e.g. "us-west-2.opendata.source.coop"
    prefix = parsed.path.lstrip("/")  # e.g. "englacial/ismip6"
    root_path = f"{bucket}/{prefix}" if prefix else bucket

    print(f"Building file index from {source_url}...")

    # Initialize filesystem
    fs = fsspec.filesystem(protocol, anon=True)

    # Collect all .nc files with their metadata
    all_files = []

    # Walk through the bucket structure
    # Expected: {root_path}/Projection-{ICE_SHEET}/{INSTITUTION}/{MODEL}/{EXPERIMENT}/*.nc
    try:
        # Get all projection directories (Projection-AIS, Projection-GIS, etc.)
        projection_dirs = fs.ls(root_path, detail=False)

        for proj_dir in projection_dirs:
            if not proj_dir.endswith('/') and 'Projection-' in proj_dir:
                print(f"  Scanning {proj_dir}...")

                # Get all institutions
                institutions = fs.ls(proj_dir, detail=False)

                for inst_dir in institutions:
                    # Get all models
                    try:
                        models = fs.ls(inst_dir, detail=False)

                        for model_dir in models:
                            # Get all experiments
                            try:
                                experiments = fs.ls(model_dir, detail=False)

                                for exp_dir in experiments:
                                    # Get all .nc files with details (including size)
                                    try:
                                        files = fs.ls(exp_dir, detail=True)
                                        nc_files = [f for f in files if f['name'].endswith('.nc')]
                                        all_files.extend(nc_files)
                                    except Exception as e:
                                        print(f"    Warning: Error reading {exp_dir}: {e}")

                            except Exception as e:
                                print(f"    Warning: Error reading {model_dir}: {e}")

                    except Exception as e:
                        print(f"    Warning: Error reading {inst_dir}: {e}")

    except Exception as e:
        print(f"Error scanning bucket: {e}")
        raise

    print(f"  Found {len(all_files)} .nc files")
    print("  Parsing file paths...")

    # Parse all file paths
    records = []
    for file_info in all_files:
        file_path = file_info['name']
        file_size = file_info.get('size', 0)

        parsed = parse_ismip6_path(file_path)
        if parsed:
            parsed['size_bytes'] = file_size
            records.append(parsed)
        else:
            print(f"    Warning: Could not parse path: {file_path}")

    print(f"  Successfully parsed {len(records)} files")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Sort for consistency
    df = df.sort_values(['ice_sheet', 'institution', 'model_name', 'experiment', 'variable']).reset_index(drop=True)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Saved index to cache: {cache_path}")

    return df


def get_file_index(source_url: Optional[str] = None,
                   cache_path: Optional[str] = None,
                   force_rebuild: bool = False) -> pd.DataFrame:
    """
    Get the ISMIP6 file index, using cache if available.

    Args:
        source_url: S3 URL of the source data (default: :data:`~ismip6_helper.SOURCE_DATA_URL`)
        cache_path: Path to cache file (default: ./.cache/ismip6_index.parquet)
        force_rebuild: If True, ignore cache and rebuild index

    Returns:
        DataFrame with columns: variable, ice_sheet, institution, model_name, experiment, url, size_bytes
    """
    return build_file_index(source_url=source_url, cache_path=cache_path, force_rebuild=force_rebuild)


if __name__ == "__main__":
    # Example usage
    df = get_file_index()
    print(f"\nIndexed {len(df)} files")
    print("\nFirst few entries:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
