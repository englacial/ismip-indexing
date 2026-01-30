/**
 * Format a numeric value for display, using scientific notation for
 * very small or very large numbers.
 */
export function formatValue(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs < 0.01 || abs >= 1e6) return v.toExponential(2);
  if (abs < 1) return v.toPrecision(3);
  return v.toFixed(2);
}
