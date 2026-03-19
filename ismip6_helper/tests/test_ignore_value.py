"""Unit tests for ismip6_helper.ignore_value."""

import numpy as np
import pytest

from ismip6_helper.ignore_value import (
    _read_corner_values,
    detect_ignore_value,
    compute_valid_range,
)


# ---------------------------------------------------------------------------
# _read_corner_values
# ---------------------------------------------------------------------------

class TestReadCornerValues:
    def test_2d_array(self):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        corners = _read_corner_values(data)
        # 4 corners x 2x2 = 16 values
        assert corners.shape == (16,)

    def test_3d_array_uses_first_slice(self):
        data = np.zeros((5, 10, 10), dtype=np.float32)
        data[0] = 42.0  # first slice is all 42
        data[1:] = 99.0
        corners = _read_corner_values(data)
        assert np.all(corners == 42.0)

    def test_1d_returns_empty(self):
        data = np.ones(10, dtype=np.float32)
        corners = _read_corner_values(data)
        assert corners.size == 0

    def test_small_grid(self):
        data = np.ones((2, 2), dtype=np.float32) * 7.0
        corners = _read_corner_values(data)
        # All four corners overlap on the same 2x2 grid
        assert np.all(corners == 7.0)


# ---------------------------------------------------------------------------
# detect_ignore_value
# ---------------------------------------------------------------------------

class TestDetectIgnoreValue:
    def test_all_nan_returns_none(self):
        corners = np.full(16, np.nan)
        assert detect_ignore_value(corners) is None

    def test_consistent_zero(self):
        corners = np.zeros(16, dtype=np.float32)
        assert detect_ignore_value(corners) == 0.0

    def test_consistent_nonzero(self):
        corners = np.full(16, -9999.0)
        assert detect_ignore_value(corners) == -9999.0

    def test_mixed_values_returns_none(self):
        corners = np.array([0.0, 0.0, 1.0, 0.0])
        assert detect_ignore_value(corners) is None

    def test_empty_returns_none(self):
        corners = np.array([])
        assert detect_ignore_value(corners) is None

    def test_all_inf_returns_none(self):
        corners = np.full(16, np.inf)
        assert detect_ignore_value(corners) is None

    def test_nan_mixed_with_consistent_sentinel(self):
        # Some NaN, rest all same finite value → still detected
        corners = np.array([np.nan, 0.0, 0.0, 0.0, np.nan, 0.0])
        assert detect_ignore_value(corners) == 0.0


# ---------------------------------------------------------------------------
# compute_valid_range
# ---------------------------------------------------------------------------

class TestComputeValidRange:
    def test_sentinel_outside_range(self):
        """Temperature in Kelvin: valid ~200-280, sentinel 0."""
        data = np.zeros((10, 10), dtype=np.float32)
        # Fill interior with realistic temperature
        data[2:8, 2:8] = 250.0
        result = compute_valid_range(data, sentinel=0.0)
        assert result is not None
        vmin, vmax = result
        assert vmin == 250.0
        assert vmax == 250.0

    def test_sentinel_inside_range(self):
        """If sentinel could be a valid value, returns None."""
        data = np.array([-1.0, 0.0, 1.0, 2.0])
        result = compute_valid_range(data, sentinel=0.0)
        assert result is None

    def test_fill_value_excluded(self):
        data = np.array([0.0, 200.0, 250.0, 1e20])
        result = compute_valid_range(data, sentinel=0.0, fill_value=1e20)
        assert result is not None
        vmin, vmax = result
        assert vmin == 200.0
        assert vmax == 250.0

    def test_all_sentinel_returns_none(self):
        data = np.zeros((10, 10), dtype=np.float32)
        result = compute_valid_range(data, sentinel=0.0)
        assert result is None

    def test_sentinel_above_range(self):
        """Sentinel is above valid data (e.g. -9999 as sentinel for negative data)."""
        data = np.full((10, 10), -9999.0, dtype=np.float32)
        data[2:8, 2:8] = -50.0  # valid range is around -50
        result = compute_valid_range(data, sentinel=-9999.0)
        assert result is not None
        vmin, vmax = result
        assert vmin == -50.0
        assert vmax == -50.0


# ---------------------------------------------------------------------------
# Integration: realistic ISMIP6 pattern
# ---------------------------------------------------------------------------

class TestISMIP6Pattern:
    def test_temperature_grid_with_zero_ocean(self):
        """Simulate litempbotgr: 761x761 grid, corners are ocean (0.0),
        interior is temperature in Kelvin (~200-280)."""
        data = np.zeros((761, 761), dtype=np.float32)
        # Fill a circular-ish interior with temperature data
        yy, xx = np.mgrid[:761, :761]
        center = 380
        dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        ice_mask = dist < 300
        data[ice_mask] = 240.0 + np.random.default_rng(42).normal(0, 10, ice_mask.sum()).astype(np.float32)

        corners = _read_corner_values(data)
        sentinel = detect_ignore_value(corners)
        assert sentinel == 0.0

        valid_range = compute_valid_range(data, sentinel)
        assert valid_range is not None
        vmin, vmax = valid_range
        # Zero should be outside the range (temperatures are ~200+ K)
        assert vmin > 0.0
        assert 0.0 < vmin  # sentinel is below valid_min

    def test_nan_ocean_no_sentinel(self):
        """Properly encoded data with NaN ocean → no sentinel detected."""
        data = np.full((100, 100), np.nan, dtype=np.float32)
        data[20:80, 20:80] = 250.0

        corners = _read_corner_values(data)
        sentinel = detect_ignore_value(corners)
        assert sentinel is None
