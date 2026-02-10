"""
ISMIP6 Helper Package

A utility package for working with ISMIP6 (Ice Sheet Model Intercomparison Project for CMIP6) data.

This package provides:
- Grid utilities for correcting ISMIP6 grid coordinates
- File indexing and caching for ISMIP6 datasets on Google Cloud Storage
- Time encoding fixes for CF-compliance

Main modules:
- grid_utils: Functions for handling ISMIP6 grid coordinates and projections
- index: Functions for indexing and caching ISMIP6 file metadata
- time_utils: Functions for fixing time encoding issues in ISMIP6 files
"""

from .grid_utils import (
    correct_grid_coordinates,
    detect_grid_resolution,
    create_coordinates,
    verify_latlon_consistency,
    GRID_BOUNDS,
    STANDARD_RESOLUTIONS,
)

from .index import (
    get_file_index,
    parse_ismip6_path,
    build_file_index,
)

from .time_utils import (
    fix_time_encoding,
    normalize_time_encoding,
    open_ismip6_dataset,
    STANDARD_TIME_UNITS,
    STANDARD_TIME_CALENDAR,
)

from .rechunk_virtual import (
    rechunk_contiguous_time_axis,
)

from .merge_virtual import (
    bin_time_to_year,
    merge_virtual_datasets,
)

__version__ = "0.1.0"

__all__ = [
    # Grid utils
    "correct_grid_coordinates",
    "detect_grid_resolution",
    "create_coordinates",
    "verify_latlon_consistency",
    "GRID_BOUNDS",
    "STANDARD_RESOLUTIONS",
    # Index utils
    "get_file_index",
    "parse_ismip6_path",
    "build_file_index",
    # Time utils
    "fix_time_encoding",
    "normalize_time_encoding",
    "open_ismip6_dataset",
    "STANDARD_TIME_UNITS",
    "STANDARD_TIME_CALENDAR",
    # Rechunking
    "rechunk_contiguous_time_axis",
    # Merge virtual
    "bin_time_to_year",
    "merge_virtual_datasets",
]
