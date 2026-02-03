"""
Normalize time encoding across all groups in an existing Icechunk store.

Reads each group's time array and its CF attrs (units, calendar), decodes
the raw values via cftime, then re-encodes to a standard epoch/calendar
("days since 2000-01-01", proleptic_gregorian) and writes the updated
values + attrs back.  A single commit captures all changes.

The pre-normalization snapshot is logged so it remains accessible.

Unlike rechunking, this modifies actual data (the small time coordinate
arrays, not the large virtual data arrays), so no virtual ref surgery
is needed.

Usage:
    python normalize_time_existing_store.py [--dry-run] [--store {s3,gcs}]
"""

import argparse
import warnings

import cftime
import icechunk
import ismip6_helper
import numpy as np
import zarr

warnings.filterwarnings("ignore", module="zarr")

STORE_CONFIGS = {
    "s3": {
        "description": "Production S3 store (combined-variables-v3)",
        "storage_fn": lambda: icechunk.s3_storage(
            bucket="ismip6-icechunk",
            prefix="combined-variables-v3",
            from_env=True,
            region="us-west-2",
        ),
    },
    "gcs": {
        "description": "Dev GCS store (12-07-2025)",
        "storage_fn": lambda: icechunk.gcs_storage(
            bucket="ismip6-icechunk",
            prefix="12-07-2025",
            from_env=True,
        ),
    },
}


def get_repo(store_key: str):
    store_cfg = STORE_CONFIGS[store_key]
    storage = store_cfg["storage_fn"]()
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store(),
        )
    )
    config.max_concurrent_requests = 3
    credentials = icechunk.containers_credentials({"gs://ismip6/": None})

    repo = icechunk.Repository.open(
        storage, config=config, authorize_virtual_chunk_access=credentials
    )
    print(f"Opened: {store_cfg['description']}")
    return repo


def normalize_time_in_group(root, group_path: str, dry_run: bool = False) -> bool:
    """
    Normalize the time array in a single group (model/experiment).

    Returns True if the time array was modified, False if skipped.
    """
    time_path = f"{group_path}/time"
    try:
        time_arr = root[time_path]
    except KeyError:
        return False

    attrs = dict(time_arr.attrs)
    original_units = attrs.get("units")
    original_calendar = attrs.get("calendar")

    # Already normalized?
    if (
        original_units == ismip6_helper.STANDARD_TIME_UNITS
        and original_calendar == ismip6_helper.STANDARD_TIME_CALENDAR
    ):
        return False

    if not original_units:
        print(f"  SKIP {group_path}: no units on time variable")
        return False

    raw_values = time_arr[:]

    # Handle packed date format
    if "as %Y" in str(original_units):
        dates = []
        for v in np.asarray(raw_values).flat:
            date_int = int(v)
            y = date_int // 10000
            m = (date_int % 10000) // 100
            d = date_int % 100
            if y < 1 or m < 1 or m > 12 or d < 1 or d > 31:
                print(f"  ERROR {group_path}: invalid packed date value {v} -> y={y} m={m} d={d}")
                return False
            dates.append(
                cftime.datetime(y, m, d, calendar=ismip6_helper.STANDARD_TIME_CALENDAR)
            )
    else:
        # Fix common attr issues before decoding
        fixed_units = original_units
        fixed_calendar = original_calendar or "365_day"

        # Apply the same fixes as fix_time_encoding
        import re
        if re.search(r"\d{1,2}-\d{1,2}-\d{4}", fixed_units):
            fixed_units = re.sub(
                r"(\d{1,2})-(\d{1,2})-(\d{4})",
                lambda m: f"{m.group(3)}-{m.group(1)}-{m.group(2)}",
                fixed_units,
            )
        if re.search(r"-\d+-0(\s|$)", fixed_units):
            fixed_units = re.sub(r"(-\d+)-0(\s|$)", r"\1-1\2", fixed_units)

        # Detect mislabeled calendars: if the calendar claims to be
        # standard/gregorian/proleptic_gregorian but the time steps are
        # constant 365-day intervals, the data was actually produced on a
        # 365_day (noleap) calendar.  Decoding with the wrong calendar
        # causes cumulative leap-year drift (~1 day every 4 years).
        decode_calendar = fixed_calendar
        vals = np.asarray(raw_values, dtype=np.float64)
        if fixed_calendar in ("standard", "gregorian", "proleptic_gregorian") and len(vals) > 2:
            diffs = np.diff(vals)
            # Check if all steps are exactly 365 days (noleap signature)
            if np.allclose(diffs, 365.0, atol=0.5):
                print(f"  NOTE {group_path}: calendar '{fixed_calendar}' has constant "
                      f"365-day steps — overriding to '365_day' for decoding")
                decode_calendar = "365_day"

        try:
            dates = cftime.num2date(
                vals,
                units=fixed_units,
                calendar=decode_calendar,
            )
        except Exception as e:
            print(f"  ERROR {group_path}: could not decode time: {e}")
            return False

    # Convert to target calendar dates
    import calendar as cal_mod
    target_dates = []
    for dt in np.asarray(dates).flat:
        try:
            target_dates.append(
                cftime.datetime(
                    dt.year, dt.month, dt.day,
                    calendar=ismip6_helper.STANDARD_TIME_CALENDAR,
                )
            )
        except ValueError:
            max_day = cal_mod.monthrange(dt.year, dt.month)[1]
            target_dates.append(
                cftime.datetime(
                    dt.year, dt.month, min(dt.day, max_day),
                    calendar=ismip6_helper.STANDARD_TIME_CALENDAR,
                )
            )

    new_values = cftime.date2num(
        target_dates,
        units=ismip6_helper.STANDARD_TIME_UNITS,
        calendar=ismip6_helper.STANDARD_TIME_CALENDAR,
    )
    new_values = np.asarray(new_values, dtype=np.float64).reshape(raw_values.shape)

    # Format first/last for display
    first_date = target_dates[0] if target_dates else "?"
    last_date = target_dates[-1] if target_dates else "?"

    if dry_run:
        print(
            f"  {group_path}: {original_units} ({original_calendar}) "
            f"-> {first_date}..{last_date} [{len(new_values)} steps]"
        )
        return True

    # Write normalized values and attrs
    time_arr[:] = new_values

    new_attrs = {k: v for k, v in attrs.items() if k not in ("units", "calendar")}
    new_attrs["units"] = ismip6_helper.STANDARD_TIME_UNITS
    new_attrs["calendar"] = ismip6_helper.STANDARD_TIME_CALENDAR
    time_arr.attrs.clear()
    time_arr.attrs.update(new_attrs)

    print(
        f"  {group_path}: {original_units} ({original_calendar}) "
        f"-> {first_date}..{last_date} [{len(new_values)} steps]"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Normalize time encoding in icechunk store"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without writing"
    )
    parser.add_argument(
        "--store",
        choices=["s3", "gcs"],
        default="s3",
        help="Which store to normalize (default: s3)",
    )
    args = parser.parse_args()

    print(f"Opening icechunk repository ({args.store})...")
    repo = get_repo(args.store)

    # Log pre-normalization snapshot
    ro_session = repo.readonly_session("main")
    pre_snapshot = ro_session.snapshot_id
    print(f"Pre-normalization snapshot: {pre_snapshot}")

    if not args.dry_run:
        session = repo.writable_session("main")
    else:
        session = repo.readonly_session("main")

    root = zarr.open(session.store, mode="r" if args.dry_run else "r+")

    total_normalized = 0
    total_skipped = 0

    # Walk the hierarchy: model_groups -> experiment_groups
    for model_name in sorted(root.group_keys()):
        print(f"\n{model_name}:")
        model_group = root[model_name]
        for exp_name in sorted(model_group.group_keys()):
            group_path = f"{model_name}/{exp_name}"
            normalized = normalize_time_in_group(root, group_path, dry_run=args.dry_run)
            if normalized:
                total_normalized += 1
            else:
                total_skipped += 1

    print(f"\n{'Would normalize' if args.dry_run else 'Normalized'} {total_normalized} groups.")
    print(f"Skipped {total_skipped} groups (already normalized or no time variable).")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")
        return

    if total_normalized == 0:
        print("\nNo changes to commit.")
        return

    commit_id = session.commit(
        f"Normalized time encoding to '{ismip6_helper.STANDARD_TIME_UNITS}' "
        f"({ismip6_helper.STANDARD_TIME_CALENDAR}) for {total_normalized} groups"
    )
    print(f"\nDone!")
    print(f"  Pre-normalization snapshot:  {pre_snapshot}")
    print(f"  Post-normalization commit:   {commit_id}")


if __name__ == "__main__":
    main()
