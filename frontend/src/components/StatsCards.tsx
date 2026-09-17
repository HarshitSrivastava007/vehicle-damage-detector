import type { UsageSummary } from "../api/types"
import { formatINR } from "../utils/currency"

function Card({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="panel" style={{ flex: 1, minWidth: 160 }}>
      <div style={{ color: "var(--muted)", fontSize: "0.85rem", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: "1.6rem", fontWeight: 600 }}>{value}</div>
    </div>
  )
}

export function StatsCards({ summary }: { summary: UsageSummary }) {
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
      <Card label="Calls this month" value={summary.calls_this_month} />
      <Card label="Cost this month" value={formatINR(summary.cost_this_month)} />
      <Card label="Calls all-time" value={summary.calls_all_time} />
      <Card label="Cost all-time" value={formatINR(summary.cost_all_time)} />
    </div>
  )
}
