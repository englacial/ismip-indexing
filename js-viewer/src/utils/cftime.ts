/**
 * CF-convention time decoder for zarr/netCDF data.
 *
 * Parses the CF `units` attribute (e.g. "days since 2005-1-1 00:00:00")
 * and `calendar` attribute, then converts raw numeric time values to
 * human-readable date strings.
 *
 * Supports calendars: standard, gregorian, proleptic_gregorian, julian,
 * 365_day (noleap), 360_day.  For display purposes the non-standard
 * calendars are approximated to real dates (off by at most a few days
 * for annual-resolution ISMIP6 data).
 */

// ---------- types -----------------------------------------------------------

export type Calendar =
  | "standard"
  | "gregorian"
  | "proleptic_gregorian"
  | "julian"
  | "365_day"
  | "noleap"
  | "360_day";

export interface TimeEncoding {
  /** Multiplier to convert the unit to days (e.g. seconds→days = 1/86400) */
  toDays: number;
  /** Epoch components */
  epochYear: number;
  epochMonth: number;
  epochDay: number;
  /** Calendar type (normalised) */
  calendar: Calendar;
}

// ---------- units string parser ---------------------------------------------

const UNITS_RE =
  /^(days|hours|minutes|seconds|milliseconds)\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})/i;

const UNIT_TO_DAYS: Record<string, number> = {
  days: 1,
  hours: 1 / 24,
  minutes: 1 / 1440,
  seconds: 1 / 86400,
  milliseconds: 1 / 86400000,
};

/**
 * Special-case: "day as %Y%m%d.%f" — the values themselves encode the date
 * as e.g. 20050115.5.  Return null to signal this format.
 */
const PACKED_DATE_RE = /^day\s+as\s+%Y%m%d/i;

export function parseTimeUnits(
  units: string | undefined | null,
  calendar: string | undefined | null,
): TimeEncoding | "packed_date" | null {
  if (!units) return null;

  if (PACKED_DATE_RE.test(units)) return "packed_date";

  const m = UNITS_RE.exec(units);
  if (!m) return null;

  const toDays = UNIT_TO_DAYS[m[1].toLowerCase()];
  if (toDays === undefined) return null;

  const cal = normalizeCalendar(calendar);

  return {
    toDays,
    epochYear: parseInt(m[2], 10),
    epochMonth: parseInt(m[3], 10),
    epochDay: parseInt(m[4], 10),
    calendar: cal,
  };
}

function normalizeCalendar(cal: string | undefined | null): Calendar {
  if (!cal) return "365_day"; // ISMIP6 default when missing
  const lc = cal.toLowerCase().replace(/[^a-z0-9_]/g, "");
  if (lc === "noleap" || lc === "365_day") return "365_day";
  if (lc === "360_day") return "360_day";
  if (lc === "julian") return "julian";
  // standard, gregorian, proleptic_gregorian all map to JS Date
  return "standard";
}

// ---------- calendar-aware date math ----------------------------------------

/** Days per month for noleap (365_day) calendar */
const NOLEAP_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/** Is `year` a leap year in the standard (Gregorian) calendar? */
function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

/** Days in month for a given calendar */
function daysInMonth(year: number, month: number, cal: Calendar): number {
  if (cal === "360_day") return 30;
  if (cal === "365_day" || cal === "noleap") return NOLEAP_DAYS[month - 1];
  // standard/gregorian/julian
  const std = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (month === 2 && isLeapYear(year)) return 29;
  return std[month - 1];
}

/** Convert an epoch + fractional days to {year, month, day}. */
function addDaysToDate(
  year: number,
  month: number,
  day: number,
  totalDays: number,
  cal: Calendar,
): { year: number; month: number; day: number } {
  // For standard calendars, use JS Date arithmetic (handles leap years etc.)
  if (cal === "standard" || cal === "gregorian" || cal === "proleptic_gregorian" || cal === "julian") {
    const d = new Date(Date.UTC(year, month - 1, day));
    // Date constructor treats 0-99 as 1900-1999; fix it
    if (year >= 0 && year < 100) d.setUTCFullYear(year);
    d.setUTCDate(d.getUTCDate() + Math.floor(totalDays));
    return {
      year: d.getUTCFullYear(),
      month: d.getUTCMonth() + 1,
      day: d.getUTCDate(),
    };
  }

  // Non-standard calendars: manual walk
  let y = year;
  let m = month;
  let d = day;
  let remaining = Math.floor(totalDays);

  // Handle negative offsets
  while (remaining < 0) {
    d += remaining;
    while (d < 1) {
      m -= 1;
      if (m < 1) { m = 12; y -= 1; }
      d += daysInMonth(y, m, cal);
    }
    remaining = 0;
  }

  // Handle positive offsets
  while (remaining > 0) {
    const dim = daysInMonth(y, m, cal);
    const daysLeftInMonth = dim - d;
    if (remaining <= daysLeftInMonth) {
      d += remaining;
      remaining = 0;
    } else {
      remaining -= daysLeftInMonth + 1;
      d = 1;
      m += 1;
      if (m > 12) { m = 1; y += 1; }
    }
  }

  return { year: y, month: m, day: d };
}

// ---------- public API ------------------------------------------------------

function pad2(n: number): string {
  return n < 10 ? `0${n}` : `${n}`;
}

/**
 * Decode a single raw time value to a date string.
 */
export function decodeTimeValue(
  value: number,
  encoding: TimeEncoding | "packed_date",
): string {
  if (encoding === "packed_date") {
    // value is e.g. 20050115.5
    const dateInt = Math.floor(value);
    const y = Math.floor(dateInt / 10000);
    const m = Math.floor((dateInt % 10000) / 100);
    const d = dateInt % 100;
    return `${y}-${pad2(m)}-${pad2(d)}`;
  }

  const totalDays = value * encoding.toDays;
  const { year, month, day } = addDaysToDate(
    encoding.epochYear,
    encoding.epochMonth,
    encoding.epochDay,
    totalDays,
    encoding.calendar,
  );
  return `${year}-${pad2(month)}-${pad2(day)}`;
}

/**
 * Decode an array of raw time values to date strings.
 */
export function decodeTimeArray(
  values: ArrayLike<number>,
  encoding: TimeEncoding | "packed_date",
): string[] {
  const result: string[] = [];
  for (let i = 0; i < values.length; i++) {
    result.push(decodeTimeValue(values[i], encoding));
  }
  return result;
}

/**
 * Extract year from a date label string like "2035-07-02".
 */
export function yearFromLabel(label: string): number {
  return parseInt(label.split("-")[0], 10);
}

/**
 * Find the index in timeLabels whose year is closest to the target year.
 * Returns null if the target year is outside the range of valid (non-NaN) years,
 * or if all labels have NaN years.
 */
export function findIndexForYear(
  timeLabels: string[],
  targetYear: number,
): number | null {
  if (timeLabels.length === 0) return null;

  // Compute actual min/max years, skipping NaN labels
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < timeLabels.length; i++) {
    const y = yearFromLabel(timeLabels[i]);
    if (isNaN(y)) continue;
    if (y < lo) lo = y;
    if (y > hi) hi = y;
  }

  // No valid years at all
  if (!isFinite(lo) || !isFinite(hi)) return null;

  // Out of range — no data for this year
  if (targetYear < lo || targetYear > hi) return null;

  let bestIdx = 0;
  let bestDiff = Infinity;
  for (let i = 0; i < timeLabels.length; i++) {
    const y = yearFromLabel(timeLabels[i]);
    if (isNaN(y)) continue;
    const diff = Math.abs(y - targetYear);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Get the min and max years across an array of time label arrays.
 */
export function yearRange(
  allTimeLabels: (string[] | null)[],
): { minYear: number; maxYear: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (const labels of allTimeLabels) {
    if (!labels || labels.length === 0) continue;
    for (let i = 0; i < labels.length; i++) {
      const y = yearFromLabel(labels[i]);
      if (isNaN(y)) continue;
      if (y < min) min = y;
      if (y > max) max = y;
    }
  }
  if (!isFinite(min)) return null;
  return { minYear: min, maxYear: max };
}
