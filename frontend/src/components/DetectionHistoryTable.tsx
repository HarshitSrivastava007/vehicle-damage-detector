import { useQuery } from "@tanstack/react-query"
import { useState } from "react"
import { api } from "../api/client"
import type { DetectionLogOut } from "../api/types"
import { formatINR } from "../utils/currency"
import { Badge } from "./Badge"

const PAGE_SIZE = 20

export function DetectionHistoryTable() {
  const [offset, setOffset] = useState(0)

  const { data, isLoading } = useQuery({
    queryKey: ["detections", offset],
    queryFn: () => api.get<DetectionLogOut[]>(`/detections?limit=${PAGE_SIZE}&offset=${offset}`),
  })

  if (isLoading) {
    return <div className="panel">Loading history…</div>
  }

  const rows = data ?? []

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>Detection History</h3>
      {rows.length === 0 ? (
        <div style={{ color: "var(--muted)" }}>No detections yet.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Filename</th>
              <th>Classes</th>
              <th>Count</th>
              <th>Cost</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.filename}</td>
                <td>
                  {Object.entries(row.class_counts).map(([name, count]) => (
                    <Badge key={name}>
                      {name} ×{count}
                    </Badge>
                  ))}
                </td>
                <td>{row.detection_count}</td>
                <td>{formatINR(row.cost)}</td>
                <td>{new Date(row.created_at + "Z").toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button
          className="secondary"
          disabled={offset === 0}
          onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
        >
          Prev
        </button>
        <button className="secondary" disabled={rows.length < PAGE_SIZE} onClick={() => setOffset(offset + PAGE_SIZE)}>
          Next
        </button>
      </div>
    </div>
  )
}
