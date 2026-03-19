"""Unit tests for ismip6_helper.time_utils."""

import numpy as np
import pytest
import xarray as xr

from ismip6_helper.time_utils import (
    normalize_time_encoding,
    STANDARD_TIME_UNITS,
    STANDARD_TIME_CALENDAR,
)


def _make_raw_ds(time_values, units="days since 2000-01-01", calendar="proleptic_gregorian"):
    """Build a minimal dataset with raw (undecoded) time values."""
    time_coord = xr.Variable(
        dims=("time",),
        data=np.array(time_values, dtype=np.float64),
        attrs={"units": units, "calendar": calendar},
    )
    data = xr.Variable(
        dims=("time",),
        data=np.zeros(len(time_values)),
    )
    return xr.Dataset({"var": data}, coords={"time": time_coord})


# ---------------------------------------------------------------------------
# Tests: garbage value rejection in normalize_time_encoding
# ---------------------------------------------------------------------------

class TestNormalizeGarbageRejection:
    def test_extreme_values_rejected(self):
        """Values with magnitude > 1e15 should cause normalization to be skipped."""
        ds = _make_raw_ds([1e219, 2.31e-310, 5475.0])
        result = normalize_time_encoding(ds)
        # Should return unchanged — attrs still have original encoding
        assert result['time'].attrs['units'] == "days since 2000-01-01"
        np.testing.assert_array_equal(result['time'].values, ds['time'].values)

    def test_nan_values_rejected(self):
        """NaN in time values should cause normalization to be skipped."""
        ds = _make_raw_ds([np.nan, 5475.0, 5840.0])
        result = normalize_time_encoding(ds)
        assert np.isnan(result['time'].values[0])

    def test_normal_values_pass(self):
        """Standard values should be normalized correctly."""
        # Use non-uniform intervals to avoid calendar mislabel detection
        # 5479 = 2015-01-01, 5844 = 2016-01-01, 6575 = 2018-01-01
        ds = _make_raw_ds([5479.0, 5844.0, 6575.0])
        result = normalize_time_encoding(ds)
        # Should have been re-encoded to standard target
        assert result['time'].attrs['units'] == STANDARD_TIME_UNITS
        assert result['time'].attrs['calendar'] == STANDARD_TIME_CALENDAR
        # Values should be approximately the same (same epoch + calendar)
        np.testing.assert_allclose(result['time'].values, [5479.0, 5844.0, 6575.0], atol=1.0)

    def test_inf_values_rejected(self):
        """Inf values should cause normalization to be skipped."""
        ds = _make_raw_ds([np.inf, 5475.0])
        result = normalize_time_encoding(ds)
        assert np.isinf(result['time'].values[0])


# ---------------------------------------------------------------------------
# Tests: packed date format validation
# ---------------------------------------------------------------------------

class TestPackedDateValidation:
    def test_valid_packed_dates_decoded(self):
        """Real packed YYYYMMDD values (>= 18500101) should be decoded normally."""
        ds = _make_raw_ds(
            [20150101.0, 20160101.0, 20170101.0],
            units="day as %Y%m%d.%f",
            calendar="proleptic_gregorian",
        )
        result = normalize_time_encoding(ds)
        assert result['time'].attrs['units'] == STANDARD_TIME_UNITS
        # Values should correspond to 2015, 2016, 2017
        import cftime
        dates = cftime.num2date(
            result['time'].values,
            units=STANDARD_TIME_UNITS,
            calendar=STANDARD_TIME_CALENDAR,
        )
        years = [d.year for d in np.asarray(dates).flat]
        assert years == [2015, 2016, 2017]

    def test_small_values_use_epoch_detection(self):
        """NCAR-style: values 0–31301 with packed date format → days since 2015-01-01."""
        # 87 years × 365.25 ≈ 31,776 days; use 87 annual steps
        raw_values = [i * 365.0 for i in range(87)]  # 0, 365, 730, ..., 31390
        ds = _make_raw_ds(
            raw_values,
            units="day as %Y%m%d.%f",
            calendar="proleptic_gregorian",
        )
        result = normalize_time_encoding(ds)
        assert result['time'].attrs['units'] == STANDARD_TIME_UNITS

        # Should have matched "days since 2015-01-01" epoch
        import cftime
        dates = cftime.num2date(
            result['time'].values,
            units=STANDARD_TIME_UNITS,
            calendar=STANDARD_TIME_CALENDAR,
        )
        years = sorted(set(d.year for d in np.asarray(dates).flat))
        # First year should be 2015, last should be ~2101
        assert years[0] == 2015
        assert years[-1] >= 2100

    def test_guard_threshold(self):
        """Values at the boundary (18500101) should use packed date decode."""
        ds = _make_raw_ds(
            [18500101.0, 18510101.0],
            units="day as %Y%m%d.%f",
            calendar="proleptic_gregorian",
        )
        result = normalize_time_encoding(ds)
        import cftime
        dates = cftime.num2date(
            result['time'].values,
            units=STANDARD_TIME_UNITS,
            calendar=STANDARD_TIME_CALENDAR,
        )
        years = [d.year for d in np.asarray(dates).flat]
        assert years == [1850, 1851]

    def test_zero_value_epoch_detection(self):
        """A single value of 0.0 with packed format should detect epoch correctly."""
        ds = _make_raw_ds(
            [0.0],
            units="day as %Y%m%d.%f",
            calendar="proleptic_gregorian",
        )
        result = normalize_time_encoding(ds)
        import cftime
        dates = cftime.num2date(
            result['time'].values,
            units=STANDARD_TIME_UNITS,
            calendar=STANDARD_TIME_CALENDAR,
        )
        # With "days since 2015-01-01", 0.0 → 2015-01-01
        assert dates[0].year == 2015
