import { useQuery } from "@tanstack/react-query"
import { api } from "../api/client"
import type { UsageSummary } from "../api/types"
import { DetectionHistoryTable } from "../components/DetectionHistoryTable"
import { StatsCards } from "../components/StatsCards"

export function DashboardPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["usage-summary"],
    queryFn: () => api.get<UsageSummary>("/usage/summary"),
  })

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Dashboard</h2>
      {isLoading ? <div className="panel">Loading…</div> : data && <StatsCards summary={data} />}
      <div style={{ marginTop: 20 }}>
        <DetectionHistoryTable />
      </div>
    </div>
  )
}
