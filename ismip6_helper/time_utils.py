"""
Time utilities for ISMIP6 datasets.

This module provides functions to fix common time encoding issues in ISMIP6 files
to ensure CF-compliance and compatibility with xarray's time decoding.
"""

import logging
import re
import warnings
import numpy as np
import xarray as xr
from typing import Optional

logger = logging.getLogger(__name__)

# Standard encoding that all time arrays are normalized to.
# proleptic_gregorian is the default for numpy datetime64 and works
# natively with xarray.open_zarr(decode_times=True).
STANDARD_TIME_UNITS = "days since 2000-01-01"
STANDARD_TIME_CALENDAR = "proleptic_gregorian"


def fix_time_encoding(ds: xr.Dataset, verbose: bool = False) -> xr.Dataset:
    """
    Fix common time encoding issues in ISMIP6 files.

    This function modifies time variable attributes in-place to fix:
    1. 'unit' typo → 'units' (CF-compliant)
    2. Missing timestamps → add ' 00:00:00'
    3. Invalid dates with day 0 → change to day 1
    4. Missing calendar attribute → add '365_day'

    Parameters
    ----------
    ds : xr.Dataset
        Dataset opened with decode_cf=False, decode_times=False
    verbose : bool, optional
        If True, print information about fixes applied

    Returns
    -------
    xr.Dataset
        Dataset with fixed time encoding attributes

    Examples
    --------
    >>> import xarray as xr
    >>> import ismip6_helper
    >>>
    >>> # Open with decoding disabled
    >>> ds_raw = xr.open_dataset(url, engine='h5netcdf', decode_cf=False, decode_times=False)
    >>>
    >>> # Fix time encoding issues
    >>> ds_fixed = ismip6_helper.fix_time_encoding(ds_raw, verbose=True)
    >>>
    >>> # Now decode with xarray
    >>> ds = xr.decode_cf(ds_fixed, use_cftime=True)
    """
    # Work on a copy to avoid modifying the original
    ds = ds.copy()

    # Find time variables
    if 'time' not in ds.variables:
        return ds  # No time variable to fix
    

    attrs = ds['time'].attrs

    # Fix 1: Rename 'unit' to 'units' (CF-compliance)
    if 'unit' in attrs and 'units' not in attrs:
        if verbose:
            print(f"  - Fixing typo: 'unit' → 'units'")
        attrs['units'] = attrs.pop('unit')

    # Fix 2: Correct MM-DD-YYYY to YYYY-MM-DD
    if 'units' in attrs:
        units_str = str(attrs['units'])
        original_units = units_str

        # Fix pattern like "2000-31-12" or "YYYY-DD-MM"
        if re.search(r'\d{1,2}-\d{1,2}-\d{4}', units_str):
            # Swap day and month
            units_str = re.sub(r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: f"{m.group(3)}-{m.group(1)}-{m.group(2)}", units_str)

            if units_str != original_units:
                if verbose:
                    print(f"  - Fixing date format: MM-DD-YYYY → YYYY-MM-DD")
                attrs['units'] = units_str

    # Fix 3: Correct invalid dates (day 0 → day 1)
    if 'units' in attrs:
        units_str = str(attrs['units'])
        original_units = units_str

        # Fix pattern like "2000-1-0" or "YYYY-M-0"
        if re.search(r'-\d+-0\s', units_str) or re.search(r'-\d+-0$', units_str):
            # Replace month-0 with month-1
            units_str = re.sub(r'(-\d+)-0(\s|$)', r'\1-1\2', units_str)

            if units_str != original_units:
                if verbose:
                    print(f"  - Fixing invalid date: day 0 → day 1")
                attrs['units'] = units_str

    # Fix 4: Add calendar if missing (for cftime compatibility)
    if 'units' in attrs and 'calendar' not in attrs:
        if verbose:
            print(f"  - Adding missing calendar attribute: '365_day'")
        attrs['calendar'] = '365_day'

    # Update the variable attributes
    ds['time'].attrs = attrs

    return ds


def normalize_time_encoding(
    ds: xr.Dataset,
    target_units: str = STANDARD_TIME_UNITS,
    target_calendar: str = STANDARD_TIME_CALENDAR,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Normalize time values and encoding to a standard epoch and calendar.

    This goes beyond fix_time_encoding: it decodes the raw time values using
    the original (possibly non-standard) encoding, converts to real dates,
    and re-encodes as float64 values relative to a standard epoch and calendar.

    After normalization, xarray.open_zarr(decode_times=True) will work without
    any custom fixers, and the JS viewer can use a single decoding path.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset opened with decode_cf=False, decode_times=False, ideally
        after fix_time_encoding has been applied to clean up malformed attrs.
    target_units : str
        CF units string for the normalized encoding (default: "days since 2000-01-01").
    target_calendar : str
        CF calendar for the normalized encoding (default: "proleptic_gregorian").
    verbose : bool
        If True, print details about the conversion.

    Returns
    -------
    xr.Dataset
        Dataset with time values re-encoded to the target units/calendar.
        The time variable will be float64 with the new units and calendar attrs.
    """
    import cftime

    ds = ds.copy()

    if 'time' not in ds.variables:
        return ds

    time_var = ds['time']
    attrs = dict(time_var.attrs)
    original_units = attrs.get('units')
    original_calendar = attrs.get('calendar', '365_day')

    if not original_units:
        if verbose:
            print("  - No units on time variable, skipping normalization")
        return ds

    raw_values = time_var.values

    # Early-reject variables with garbage raw time values (e.g. BISICLES
    # lim/limnsw with uninitialized memory: 2.31e-310, 1.71e+219).
    raw_flat = np.asarray(raw_values, dtype=np.float64).ravel()
    if np.any(~np.isfinite(raw_flat)):
        logger.warning("Time values contain NaN/Inf, skipping normalization")
        return ds
    if len(raw_flat) > 0 and np.max(np.abs(raw_flat)) > 1e15:
        logger.warning(
            "Time values have extreme magnitude (max=%.2e), skipping normalization",
            np.max(np.abs(raw_flat)),
        )
        return ds

    # Handle "day as %Y%m%d.%f" packed format
    if 'as %Y' in str(original_units):
        # Valid packed YYYYMMDD dates are 8-digit numbers (>= 18500101).
        # If max(raw_values) < 18500101, the values are day offsets, not
        # packed dates (e.g. NCAR_CISM exp07: 0–31301 with units
        # "day as %Y%m%d.%f" but actually meaning days since some epoch).
        max_val = float(np.max(np.abs(raw_flat)))

        if max_val >= 18500101:
            # Real packed YYYYMMDD dates
            if verbose:
                print(f"  - Decoding packed date format: {original_units}")
            dates = []
            for v in raw_values.flat:
                date_int = int(v)
                y = date_int // 10000
                m = (date_int % 10000) // 100
                d = date_int % 100
                if d < 1:
                    d = 1
                if m < 1:
                    m = 1
                if m > 12:
                    y += (m - 1) // 12
                    m = (m - 1) % 12 + 1
                dates.append(cftime.datetime(y, m, d, calendar=target_calendar))
        else:
            # Values are day offsets, not packed YYYYMMDD.
            # Try common ISMIP6 reference epochs to find one that
            # produces dates in the plausible 1850–2200 range.
            if verbose:
                print(f"  - Packed date format with small values (max={max_val:.0f}), trying epoch detection")
            candidate_epochs = [
                "days since 2015-01-01",  # projection start
                "days since 2006-01-01",  # historical run start
                "days since 1950-01-01",  # climate convention
                "days since 0001-01-01",  # generic fallback
            ]
            dates = None
            for epoch in candidate_epochs:
                try:
                    trial = cftime.num2date(raw_values, units=epoch, calendar=target_calendar)
                    years = [d.year for d in np.asarray(trial).flat]
                    if all(1850 <= y <= 2200 for y in years):
                        dates = trial
                        logger.info("Packed date format with small values; matched epoch '%s'", epoch)
                        break
                except Exception:
                    continue
            if dates is None:
                logger.warning(
                    "Could not determine epoch for '%s' (max=%.0f)",
                    original_units, max_val,
                )
                return ds
    else:
        # Detect mislabeled calendars: if the calendar claims to be
        # standard/gregorian/proleptic_gregorian but the time steps are
        # constant 365-day intervals, the data was actually produced on a
        # 365_day (noleap) calendar.
        decode_calendar = original_calendar
        vals = np.asarray(raw_values, dtype=np.float64).ravel()
        if original_calendar in ('standard', 'gregorian', 'proleptic_gregorian') and len(vals) > 2:
            diffs = np.diff(vals)
            if np.allclose(diffs, 365.0, atol=0.5):
                if verbose:
                    print(f"  - Calendar '{original_calendar}' has constant 365-day steps, "
                          f"overriding to '365_day' for decoding")
                decode_calendar = '365_day'

        # Standard CF decode: use cftime.num2date with original encoding
        try:
            dates = cftime.num2date(
                raw_values,
                units=original_units,
                calendar=decode_calendar,
            )
        except Exception as e:
            logger.warning("Failed to decode time (units=%r, calendar=%r): %s",
                           original_units, decode_calendar, e)
            if verbose:
                print(f"  - Failed to decode time: {e}")
            return ds

    # Re-encode to target units/calendar
    # Convert cftime dates to target calendar dates, then to numeric
    target_dates = []
    for dt in np.asarray(dates).flat:
        try:
            target_dates.append(
                cftime.datetime(dt.year, dt.month, dt.day, calendar=target_calendar)
            )
        except ValueError:
            # Handle impossible dates from non-standard calendars:
            # - Invalid month (e.g. month 14) → roll into next year
            # - Invalid day (e.g. Feb 30 from 360_day) → clamp to month max
            import calendar as cal_mod
            month = dt.month
            year = dt.year
            if month > 12:
                year += (month - 1) // 12
                month = (month - 1) % 12 + 1
            elif month < 1:
                month = 1
            if target_calendar in ('proleptic_gregorian', 'standard', 'gregorian'):
                max_day = cal_mod.monthrange(year, month)[1]
            else:
                max_day = 28  # safe fallback
            clamped_day = min(max(dt.day, 1), max_day)
            target_dates.append(
                cftime.datetime(year, month, clamped_day, calendar=target_calendar)
            )

    new_values = cftime.date2num(
        target_dates,
        units=target_units,
        calendar=target_calendar,
    )

    new_values = np.asarray(new_values, dtype=np.float64).reshape(raw_values.shape)

    if verbose:
        print(f"  - Normalized time: {original_units} ({original_calendar})")
        print(f"    → {target_units} ({target_calendar})")
        if len(new_values) > 0:
            print(f"    values: [{new_values[0]:.2f} ... {new_values[-1]:.2f}]")

    # Update the dataset
    new_attrs = {k: v for k, v in attrs.items() if k not in ('units', 'calendar')}
    new_attrs['units'] = target_units
    new_attrs['calendar'] = target_calendar

    ds['time'] = xr.Variable(
        dims=time_var.dims,
        data=new_values,
        attrs=new_attrs,
    )

    return ds


def open_ismip6_dataset(
    url: str,
    engine: Optional[str] = None,
    chunks: Optional[dict] = None,
    use_cftime: bool = True,
    fix_time: bool = True,
    convert_cftime_to_datetime: bool = True,
    **kwargs
) -> xr.Dataset:
    """
    Open an ISMIP6 dataset with automatic time encoding fixes.

    This is a convenience wrapper around xr.open_dataset that:
    1. Tries h5netcdf engine first, falls back to scipy for NetCDF3 files
    2. Automatically fixes time encoding issues before decoding
    3. Decodes times to cftime by default for consistency

    Parameters
    ----------
    url : str
        Path or URL to the NetCDF file
    engine : str, optional
        NetCDF engine to use. If None, tries h5netcdf first, then scipy
    chunks : dict, optional
        Chunk sizes for dask arrays (e.g., {'time': 1})
    use_cftime : bool, default True
        Whether to decode times to cftime objects (recommended for ISMIP6)
    fix_time : bool, default True
        Whether to apply time encoding fixes before decoding
    **kwargs
        Additional arguments passed to xr.open_dataset

    Returns
    -------
    xr.Dataset
        Opened dataset with properly decoded times

    Examples
    --------
    >>> import ismip6_helper
    >>>
    >>> ds = ismip6_helper.open_ismip6_dataset(
    ...     'gs://ismip6/path/to/file.nc',
    ...     chunks={'time': 1}
    ... )
    """
    # Default chunks if not specified
    if chunks is None:
        chunks = {'time': 1}

    # Try to determine the engine if not specified
    engines_to_try = [engine] if engine else ['h5netcdf', 'scipy']

    last_error = None
    for eng in engines_to_try:
        try:
            if fix_time:
                # Open without decoding
                ds_raw = xr.open_dataset(
                    url,
                    engine=eng,
                    decode_cf=False,
                    decode_times=False,
                    chunks=chunks,
                    **kwargs
                )

                # Fix time encoding
                ds_fixed = fix_time_encoding(ds_raw)

                # Now decode with proper settings
                ds = xr.decode_cf(ds_fixed, use_cftime=use_cftime)
            else:
                # Open with standard decoding
                ds = xr.open_dataset(
                    url,
                    engine=eng,
                    decode_cf=True,
                    decode_times=True,
                    chunks=chunks,
                    **kwargs
                )
                if use_cftime:
                    # Convert to cftime if needed
                    ds = xr.decode_cf(ds, use_cftime=True)

            if convert_cftime_to_datetime:
                # Suppress warning about converting from non-standard calendars
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                          message='Converting a CFTimeIndex with dates from a non-standard calendar',
                                          category=RuntimeWarning)
                    ds['time'] = ds.indexes['time'].to_datetimeindex(time_unit='ns')

            return ds

        except Exception as e:
            last_error = e
            continue

    # If we get here, all engines failed
    raise last_error
