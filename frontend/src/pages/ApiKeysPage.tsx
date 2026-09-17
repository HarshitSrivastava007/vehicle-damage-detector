import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { ApiError, api } from "../api/client"
import type { ApiKeyCreateResponse, ApiKeyOut } from "../api/types"
import { StatusMessage } from "../components/StatusMessage"

export function ApiKeysPage() {
  const queryClient = useQueryClient()
  const [newKey, setNewKey] = useState<string | null>(null)
  const [error, setError] = useState("")

  const { data, isLoading } = useQuery({
    queryKey: ["keys"],
    queryFn: () => api.get<ApiKeyOut[]>("/auth/keys"),
  })

  const createKey = useMutation({
    mutationFn: () => api.post<ApiKeyCreateResponse>("/auth/keys"),
    onSuccess: (data) => {
      setNewKey(data.api_key)
      setError("")
      queryClient.invalidateQueries({ queryKey: ["keys"] })
    },
    onError: (err) => setError(err instanceof ApiError ? err.message : "Something went wrong"),
  })

  const revokeKey = useMutation({
    mutationFn: (id: number) => api.del(`/auth/keys/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["keys"] }),
    onError: (err) => setError(err instanceof ApiError ? err.message : "Something went wrong"),
  })

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>API Keys</h2>

      {newKey && (
        <div className="panel" style={{ borderColor: "var(--accent)" }}>
          <strong>New key created — copy it now, it won't be shown again:</strong>
          <div style={{ fontFamily: "monospace", marginTop: 8, wordBreak: "break-all" }}>{newKey}</div>
          <button className="secondary" style={{ marginTop: 12 }} onClick={() => setNewKey(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="panel">
        <button onClick={() => createKey.mutate()} disabled={createKey.isPending}>
          Create new key
        </button>
        <StatusMessage message={error} error />

        {isLoading ? (
          <div style={{ marginTop: 16 }}>Loading…</div>
        ) : (
          <table style={{ marginTop: 16 }}>
            <thead>
              <tr>
                <th>Key</th>
                <th>Created</th>
                <th>Last used</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(data ?? []).map((key) => (
                <tr key={key.id}>
                  <td style={{ fontFamily: "monospace" }}>{key.prefix}…</td>
                  <td>{new Date(key.created_at + "Z").toLocaleDateString()}</td>
                  <td>{key.last_used_at ? new Date(key.last_used_at + "Z").toLocaleString() : "never"}</td>
                  <td>{key.revoked_at ? "revoked" : "active"}</td>
                  <td>
                    {!key.revoked_at && (
                      <button className="danger" onClick={() => revokeKey.mutate(key.id)}>
                        Revoke
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
