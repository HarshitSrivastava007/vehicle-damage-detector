import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState, type FormEvent } from "react"
import { ApiError, api } from "../api/client"
import type { AdminCreateUserResponse, AdminUserOut } from "../api/types"
import { StatusMessage } from "../components/StatusMessage"
import { useAuth } from "../auth/useAuth"
import { formatINR } from "../utils/currency"

function CostEditor({ user }: { user: AdminUserOut }) {
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(user.cost_per_call)
  const [error, setError] = useState("")

  const setCost = useMutation({
    mutationFn: () => api.patch(`/admin/users/${user.id}/cost`, { cost_per_call: value }),
    onSuccess: () => {
      setEditing(false)
      setError("")
      queryClient.invalidateQueries({ queryKey: ["admin-users"] })
    },
    onError: (err) => setError(err instanceof ApiError ? err.message : "Something went wrong"),
  })

  if (!editing) {
    return (
      <button className="secondary" onClick={() => setEditing(true)}>
        {formatINR(user.cost_per_call)}
      </button>
    )
  }

  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      <span style={{ color: "var(--muted)" }}>₹</span>
      <input
        type="number"
        step="0.01"
        min="0"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        style={{ width: 90 }}
      />
      <button onClick={() => setCost.mutate()} disabled={setCost.isPending}>
        Save
      </button>
      <button className="secondary" onClick={() => setEditing(false)}>
        Cancel
      </button>
      {error && <StatusMessage message={error} error />}
    </div>
  )
}

function StatusToggle({ user, isSelf }: { user: AdminUserOut; isSelf: boolean }) {
  const queryClient = useQueryClient()
  const [error, setError] = useState("")

  const setStatus = useMutation({
    mutationFn: (is_active: boolean) => api.patch(`/admin/users/${user.id}/status`, { is_active }),
    onSuccess: () => {
      setError("")
      queryClient.invalidateQueries({ queryKey: ["admin-users"] })
    },
    onError: (err) => setError(err instanceof ApiError ? err.message : "Something went wrong"),
  })

  return (
    <div>
      <button
        className={user.is_active ? "danger" : "secondary"}
        disabled={setStatus.isPending || isSelf}
        title={isSelf ? "You can't deactivate your own account" : undefined}
        onClick={() => setStatus.mutate(!user.is_active)}
      >
        {user.is_active ? "Deactivate" : "Activate"}
      </button>
      {error && <StatusMessage message={error} error />}
    </div>
  )
}

function CreateUserForm() {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [isAdmin, setIsAdmin] = useState(false)
  const [costPerCall, setCostPerCall] = useState("0")
  const [createdKey, setCreatedKey] = useState<string | null>(null)

  const createUser = useMutation({
    mutationFn: () =>
      api.post<AdminCreateUserResponse>("/admin/users", {
        email,
        password,
        is_admin: isAdmin,
        cost_per_call: costPerCall,
      }),
    onSuccess: (data) => {
      setCreatedKey(data.api_key)
      setEmail("")
      setPassword("")
      setIsAdmin(false)
      setCostPerCall("0")
      queryClient.invalidateQueries({ queryKey: ["admin-users"] })
    },
  })

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    createUser.mutate()
  }

  if (createdKey) {
    return (
      <div className="panel" style={{ borderColor: "var(--accent)" }}>
        <strong>User created — API key (shown once):</strong>
        <div style={{ fontFamily: "monospace", marginTop: 8, wordBreak: "break-all" }}>{createdKey}</div>
        <button className="secondary" style={{ marginTop: 12 }} onClick={() => setCreatedKey(null)}>
          Dismiss
        </button>
      </div>
    )
  }

  if (!open) {
    return (
      <button onClick={() => setOpen(true)} style={{ marginBottom: 20 }}>
        Create user
      </button>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="panel">
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--muted)", marginBottom: 4 }}>
            Email
          </label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--muted)", marginBottom: 4 }}>
            Password
          </label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--muted)", marginBottom: 4 }}>
            Cost / call (₹)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={costPerCall}
            onChange={(e) => setCostPerCall(e.target.value)}
            style={{ width: 90 }}
          />
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
          <input type="checkbox" checked={isAdmin} onChange={(e) => setIsAdmin(e.target.checked)} />
          Admin
        </label>
        <button type="submit" disabled={createUser.isPending}>
          Create
        </button>
        <button type="button" className="secondary" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
      {createUser.isError && (
        <StatusMessage
          message={createUser.error instanceof ApiError ? createUser.error.message : "Something went wrong"}
          error
        />
      )}
    </form>
  )
}

export function AdminUsersPage() {
  const { user: me } = useAuth()
  const { data, isLoading } = useQuery({
    queryKey: ["admin-users"],
    queryFn: () => api.get<AdminUserOut[]>("/admin/users"),
  })

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Admin — Partners</h2>

      <CreateUserForm />

      <div className="panel">
        {isLoading ? (
          <div>Loading…</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Status</th>
                <th>Detections</th>
                <th>Cost / call</th>
                <th>Joined</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(data ?? []).map((user) => (
                <tr key={user.id} style={{ opacity: user.is_active ? 1 : 0.6 }}>
                  <td>
                    {user.email} {user.is_admin && <span className="badge">admin</span>}
                  </td>
                  <td>{user.is_active ? "active" : "inactive"}</td>
                  <td>{user.total_detections}</td>
                  <td>
                    <CostEditor user={user} />
                  </td>
                  <td>{new Date(user.created_at + "Z").toLocaleDateString()}</td>
                  <td>
                    <StatusToggle user={user} isSelf={user.id === me?.id} />
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
