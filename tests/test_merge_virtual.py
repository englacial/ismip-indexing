"""Unit tests for ismip6_helper.merge_virtual.

Tests use synthetic ManifestArrays following patterns from rechunk_virtual.py.
"""

import dataclasses
import logging

import numpy as np
import pytest
import xarray as xr
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import create_v3_array_metadata

from ismip6_helper.merge_virtual import (
    bin_time_to_year,
    compute_union_time_axis,
    merge_virtual_datasets,
    pad_dataset_to_union_time,
    _pad_manifest_to_union,
)
from ismip6_helper.time_utils import STANDARD_TIME_UNITS, STANDARD_TIME_CALENDAR


# ---------------------------------------------------------------------------
# Helpers to build synthetic ManifestArray-backed datasets
# ---------------------------------------------------------------------------

def _make_manifest_var(n_time, ny=3, nx=3, per_timestep=True):
    """Create a ManifestArray variable with per-timestep chunks.

    Returns an xr.Variable backed by a ManifestArray with shape (n_time, ny, nx).
    Each chunk points to a fake path with distinct offset/length.
    """
    itemsize = 8  # float64
    slice_bytes = ny * nx * itemsize

    if per_timestep:
        chunk_grid_shape = (n_time, 1, 1)
        paths = np.empty(chunk_grid_shape, dtype=np.dtypes.StringDType())
        offsets = np.empty(chunk_grid_shape, dtype=np.dtype("uint64"))
        lengths = np.empty(chunk_grid_shape, dtype=np.dtype("uint64"))

        for t in range(n_time):
            paths[t, 0, 0] = "s3://bucket/data.nc"
            offsets[t, 0, 0] = 1000 + t * slice_bytes
            lengths[t, 0, 0] = slice_bytes

        manifest = ChunkManifest.from_arrays(
            paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
        )
        chunk_shape = (1, ny, nx)
    else:
        # Single contiguous chunk (compressed / not rechunkable)
        chunk_grid_shape = (1, 1, 1)
        paths = np.empty(chunk_grid_shape, dtype=np.dtypes.StringDType())
        offsets = np.empty(chunk_grid_shape, dtype=np.dtype("uint64"))
        lengths = np.empty(chunk_grid_shape, dtype=np.dtype("uint64"))

        paths[0, 0, 0] = "s3://bucket/data.nc"
        offsets[0, 0, 0] = 1000
        # Use a length that doesn't match n_time * slice_bytes (simulating compression)
        lengths[0, 0, 0] = n_time * slice_bytes // 2

        manifest = ChunkManifest.from_arrays(
            paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
        )
        chunk_shape = (n_time, ny, nx)

    metadata = create_v3_array_metadata(
        shape=(n_time, ny, nx),
        data_type=np.dtype("float64"),
        chunk_shape=chunk_shape,
    )

    marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    return xr.Variable(dims=("time", "y", "x"), data=marr)


def _make_manifest_2d(ny=3, nx=3):
    """Create a 2D ManifestArray variable with no time dimension (e.g. hfgeoubed)."""
    itemsize = 8
    total_bytes = ny * nx * itemsize

    paths = np.empty((1, 1), dtype=np.dtypes.StringDType())
    offsets = np.empty((1, 1), dtype=np.dtype("uint64"))
    lengths = np.empty((1, 1), dtype=np.dtype("uint64"))

    paths[0, 0] = "s3://bucket/geo.nc"
    offsets[0, 0] = 500
    lengths[0, 0] = total_bytes

    manifest = ChunkManifest.from_arrays(
        paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
    )

    metadata = create_v3_array_metadata(
        shape=(ny, nx),
        data_type=np.dtype("float64"),
        chunk_shape=(ny, nx),
    )

    marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    return xr.Variable(dims=("y", "x"), data=marr)


def _days_since_2000(year, month=1, day=1):
    """Compute days since 2000-01-01 for a given date."""
    import cftime
    dt = cftime.datetime(year, month, day, calendar=STANDARD_TIME_CALENDAR)
    return cftime.date2num(dt, units=STANDARD_TIME_UNITS, calendar=STANDARD_TIME_CALENDAR)


def _make_dataset(var_name, time_values, ny=3, nx=3, per_timestep=True):
    """Build a virtual dataset with one ManifestArray data variable and a time coord."""
    n_time = len(time_values)
    var = _make_manifest_var(n_time, ny, nx, per_timestep=per_timestep)
    time_coord = xr.Variable(
        dims=("time",),
        data=np.array(time_values, dtype=np.float64),
        attrs={"units": STANDARD_TIME_UNITS, "calendar": STANDARD_TIME_CALENDAR},
    )
    return xr.Dataset(
        {var_name: var},
        coords={"time": time_coord},
    )


# ---------------------------------------------------------------------------
# Tests: bin_time_to_year
# ---------------------------------------------------------------------------

class TestBinTimeToYear:
    def test_flux_times_binned_to_jan1(self):
        """Flux times at Jul 1 should map to Jan 1 of the same year."""
        flux_times = [_days_since_2000(y, 7, 1) for y in range(2015, 2101)]
        ds = _make_dataset("acabf", flux_times)
        result = bin_time_to_year(ds)

        expected = np.array([_days_since_2000(y, 1, 1) for y in range(2015, 2101)])
        np.testing.assert_allclose(result['time'].values, expected, atol=0.01)

    def test_state_times_unchanged(self):
        """State times already at Jan 1 should be unchanged."""
        state_times = [_days_since_2000(y, 1, 1) for y in range(2015, 2102)]
        ds = _make_dataset("lithk", state_times)
        result = bin_time_to_year(ds)

        np.testing.assert_allclose(result['time'].values, ds['time'].values, atol=0.01)

    def test_no_time_passthrough(self):
        """Dataset without time should pass through unchanged."""
        ds = xr.Dataset({"hfgeoubed": _make_manifest_2d()})
        result = bin_time_to_year(ds)
        assert "time" not in result.variables

    def test_unnormalized_time_skipped(self):
        """Time with non-standard encoding should be skipped, not overflow."""
        # Simulate normalize_time_encoding failure: attrs still have original encoding
        time_values = np.array([5475.0, 5840.0, 6205.0])  # days since some other epoch
        time_coord = xr.Variable(
            dims=("time",),
            data=time_values,
            attrs={"units": "days since 1995-01-01", "calendar": "365_day"},
        )
        var = _make_manifest_var(3)
        ds = xr.Dataset({"acabf": var}, coords={"time": time_coord})

        result = bin_time_to_year(ds)
        # Should return unchanged -- not crash with overflow
        np.testing.assert_array_equal(result['time'].values, time_values)


# ---------------------------------------------------------------------------
# Tests: compute_union_time_axis
# ---------------------------------------------------------------------------

class TestComputeUnionTimeAxis:
    def test_identical_axes(self):
        """Same axes should produce the same union."""
        times = [_days_since_2000(y) for y in range(2015, 2101)]
        ds1 = _make_dataset("acabf", times)
        ds2 = _make_dataset("lithk", times)

        union = compute_union_time_axis([ds1, ds2])
        np.testing.assert_allclose(union, times, atol=0.01)

    def test_different_ranges(self):
        """Variables with different start/end years produce correct union."""
        # State: 2015-2101 (87 steps)
        state_times = [_days_since_2000(y) for y in range(2015, 2102)]
        # Flux: 2015-2100 (86 steps) -- already binned to Jan 1
        flux_times = [_days_since_2000(y) for y in range(2015, 2101)]

        ds_state = _make_dataset("lithk", state_times)
        ds_flux = _make_dataset("acabf", flux_times)

        union = compute_union_time_axis([ds_state, ds_flux])
        assert len(union) == 87  # union spans 2015-2101

    def test_no_time_datasets(self):
        """Datasets without time should be ignored."""
        ds_no_time = xr.Dataset({"hfgeoubed": _make_manifest_2d()})
        times = [_days_since_2000(y) for y in range(2015, 2020)]
        ds_with_time = _make_dataset("lithk", times)

        union = compute_union_time_axis([ds_no_time, ds_with_time])
        assert len(union) == 5

    def test_empty_input(self):
        """Empty input should return empty array."""
        union = compute_union_time_axis([])
        assert len(union) == 0

    def test_tolerance_deduplication(self):
        """Values within tolerance should be deduplicated."""
        base = _days_since_2000(2015)
        times1 = [base, base + 365.0]
        times2 = [base + 0.01, base + 365.01]  # within 0.5 day tolerance

        ds1 = _make_dataset("a", times1)
        ds2 = _make_dataset("b", times2)

        union = compute_union_time_axis([ds1, ds2])
        assert len(union) == 2


# ---------------------------------------------------------------------------
# Tests: _pad_manifest_to_union
# ---------------------------------------------------------------------------

class TestPadManifestToUnion:
    def test_pad_with_gaps(self):
        """Variable with fewer timesteps is padded with empty-path chunks."""
        # Variable has 3 timesteps: 2015, 2016, 2017
        ds_time = np.array([_days_since_2000(y) for y in [2015, 2016, 2017]])
        var = _make_manifest_var(3)

        # Union has 5 timesteps: 2014, 2015, 2016, 2017, 2018
        union_time = np.array([_days_since_2000(y) for y in [2014, 2015, 2016, 2017, 2018]])

        padded = _pad_manifest_to_union(var, union_time, ds_time)
        marr = padded.data
        assert marr.shape[0] == 5

        # Empty-path chunks are omitted from manifest.values() (they become
        # missing chunks that resolve to fill_value). Only the 3 real data
        # entries should be present.
        entries = list(marr.manifest.values())
        assert len(entries) == 3
        for e in entries:
            assert e["path"] != ""

    def test_fill_value_is_nan(self):
        """Padded metadata should have fill_value=NaN."""
        ds_time = np.array([_days_since_2000(2015)])
        var = _make_manifest_var(1)
        union_time = np.array([_days_since_2000(y) for y in [2015, 2016]])

        padded = _pad_manifest_to_union(var, union_time, ds_time)
        assert np.isnan(padded.data.metadata.fill_value)

    def test_shape_updated(self):
        """Metadata shape should reflect the new time dimension."""
        ds_time = np.array([_days_since_2000(y) for y in [2015, 2016]])
        var = _make_manifest_var(2, ny=5, nx=5)
        union_time = np.array([_days_since_2000(y) for y in [2015, 2016, 2017]])

        padded = _pad_manifest_to_union(var, union_time, ds_time)
        assert padded.data.shape == (3, 5, 5)


# ---------------------------------------------------------------------------
# Tests: pad_dataset_to_union_time
# ---------------------------------------------------------------------------

class TestPadDatasetToUnionTime:
    def test_no_time_variable_passthrough(self):
        """2D variables (no time dim) should be unchanged."""
        times = [_days_since_2000(y) for y in [2015, 2016]]
        ds = _make_dataset("lithk", times)
        ds["hfgeoubed"] = _make_manifest_2d()

        union_time = np.array([_days_since_2000(y) for y in [2015, 2016, 2017]])
        padded = pad_dataset_to_union_time(ds, union_time)

        # hfgeoubed should still be 2D
        assert padded["hfgeoubed"].dims == ("y", "x")

    def test_compressed_dropped_with_warning(self, caplog):
        """Variables with non-per-timestep chunks are dropped when time mismatches."""
        times = [_days_since_2000(y) for y in [2015, 2016, 2017]]
        ds = _make_dataset("compressed_var", times, per_timestep=False)

        union_time = np.array([_days_since_2000(y) for y in [2015, 2016, 2017, 2018]])

        with caplog.at_level(logging.WARNING, logger="ismip6_helper.merge_virtual"):
            padded = pad_dataset_to_union_time(ds, union_time)

        # Variable can't be padded and has mismatched time dim, so it's dropped
        assert "compressed_var" not in padded.data_vars
        assert "compressed time chunks" in caplog.text

    def test_compressed_kept_when_time_matches(self):
        """Compressed variables are kept if their time dim matches the union."""
        times = [_days_since_2000(y) for y in [2015, 2016, 2017]]
        ds = _make_dataset("compressed_var", times, per_timestep=False)

        # Same time axis as the variable -- no mismatch
        union_time = np.array(times)
        padded = pad_dataset_to_union_time(ds, union_time)

        assert "compressed_var" in padded.data_vars


# ---------------------------------------------------------------------------
# Tests: merge_virtual_datasets (end-to-end)
# ---------------------------------------------------------------------------

class TestMergeVirtualDatasets:
    def test_merge_same_axis_fast_path(self):
        """Identical time axes should skip padding (fast path)."""
        times = [_days_since_2000(y) for y in range(2015, 2101)]
        ds1 = _make_dataset("acabf", times)
        ds2 = _make_dataset("lithk", times)

        merged = merge_virtual_datasets([ds1, ds2])
        assert "acabf" in merged
        assert "lithk" in merged
        assert merged["acabf"].shape[0] == 86

    def test_merge_end_to_end(self):
        """Full pipeline: state (87 steps) + flux (86 steps) merge correctly."""
        state_times = [_days_since_2000(y) for y in range(2015, 2102)]  # 87
        flux_times = [_days_since_2000(y) for y in range(2015, 2101)]   # 86

        ds_state = _make_dataset("lithk", state_times)
        ds_flux = _make_dataset("acabf", flux_times)

        merged = merge_virtual_datasets([ds_state, ds_flux])

        assert "lithk" in merged
        assert "acabf" in merged
        # Both should now have 87 timesteps (union)
        assert merged["lithk"].shape[0] == 87
        assert merged["acabf"].shape[0] == 87
        # Flux variable should have NaN fill_value (it was padded)
        assert np.isnan(merged["acabf"].data.metadata.fill_value)

    def test_merge_with_2d_variable(self):
        """Merge with a 2D variable (no time dim) should work."""
        times = [_days_since_2000(y) for y in range(2015, 2101)]
        ds1 = _make_dataset("lithk", times)
        ds2 = xr.Dataset({"hfgeoubed": _make_manifest_2d()})

        merged = merge_virtual_datasets([ds1, ds2])
        assert "lithk" in merged
        assert "hfgeoubed" in merged
        assert merged["hfgeoubed"].dims == ("y", "x")

    def test_merge_different_start_years(self):
        """Variables starting at different years get padded correctly."""
        # var1: 2005-2100 (96 years)
        times1 = [_days_since_2000(y) for y in range(2005, 2101)]
        # var2: 2015-2100 (86 years)
        times2 = [_days_since_2000(y) for y in range(2015, 2101)]

        ds1 = _make_dataset("early_var", times1)
        ds2 = _make_dataset("late_var", times2)

        merged = merge_virtual_datasets([ds1, ds2])

        # Union should span 2005-2100 (96 years)
        assert merged["early_var"].shape[0] == 96
        assert merged["late_var"].shape[0] == 96

    def test_merge_single_dataset(self):
        """Single dataset should pass through cleanly."""
        times = [_days_since_2000(y) for y in range(2015, 2101)]
        ds = _make_dataset("lithk", times)

        merged = merge_virtual_datasets([ds])
        assert "lithk" in merged
        assert merged["lithk"].shape[0] == 86


# ---------------------------------------------------------------------------
# Tests: skip list loading and filtering
# ---------------------------------------------------------------------------

class TestSkipList:
    def test_load_skip_list(self, tmp_path):
        """Skip list should load entries and ignore comments/blanks."""
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from virtualize_with_lithops_combine_variables import load_skip_list

        skip_file = tmp_path / "skip.txt"
        skip_file.write_text(
            "# comment\n"
            "\n"
            "NCAR/CISM/expD12/acabf_AIS_NCAR_CISM_expD12.nc\n"
            "ULB/fETISh_16km/expA6/xvelbase_AIS_ULB_fETISh_expA6.nc\n"
        )
        result = load_skip_list(str(skip_file))
        assert len(result) == 2
        assert "NCAR/CISM/expD12/acabf_AIS_NCAR_CISM_expD12.nc" in result

    def test_filter_urls(self):
        """filter_urls_by_skip_list should match substrings."""
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from virtualize_with_lithops_combine_variables import filter_urls_by_skip_list

        skip_list = {"NCAR/CISM/expD12/acabf"}
        urls = [
            "s3://bucket/Projection-AIS/NCAR/CISM/expD12/acabf_AIS_NCAR_CISM_expD12.nc",
            "s3://bucket/Projection-AIS/NCAR/CISM/expD12/lithk_AIS_NCAR_CISM_expD12.nc",
            "s3://bucket/Projection-AIS/AWI/PISM1/exp05/acabf_AIS_AWI_PISM1_exp05.nc",
        ]
        kept, skipped = filter_urls_by_skip_list(urls, skip_list)
        assert len(kept) == 2
        assert len(skipped) == 1
        assert "expD12/acabf" in skipped[0]

    def test_missing_skip_list(self, tmp_path):
        """Missing file should return empty set with warning."""
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from virtualize_with_lithops_combine_variables import load_skip_list

        result = load_skip_list(str(tmp_path / "nonexistent.txt"))
        assert len(result) == 0
