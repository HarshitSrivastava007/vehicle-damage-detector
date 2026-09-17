export function formatINR(value: string | number): string {
  return `₹${Number(value).toFixed(2)}`
}
