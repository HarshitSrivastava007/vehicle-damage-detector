import { useMutation, useQueryClient } from "@tanstack/react-query"
import { NavLink, Outlet, useNavigate } from "react-router-dom"
import { api } from "../api/client"
import { useAuth } from "../auth/useAuth"

export function AppShell() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const logout = useMutation({
    mutationFn: () => api.post("/auth/logout"),
    onSuccess: () => {
      queryClient.setQueryData(["session"], null)
      navigate("/login")
    },
  })

  const linkStyle = ({ isActive }: { isActive: boolean }) => ({
    color: isActive ? "var(--text)" : "var(--muted)",
    textDecoration: "none",
    fontWeight: isActive ? 600 : 400,
  })

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px 16px" }}>
      <nav style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 24 }}>
        <strong>Veridex.ai</strong>
        <NavLink to="/" end style={linkStyle}>
          Dashboard
        </NavLink>
        <NavLink to="/keys" style={linkStyle}>
          API Keys
        </NavLink>
        <NavLink to="/detect" style={linkStyle}>
          Try Detector
        </NavLink>
        {user?.is_admin && (
          <NavLink to="/admin" style={linkStyle}>
            Partners
          </NavLink>
        )}
        <div style={{ flex: 1 }} />
        <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>{user?.email}</span>
        <button className="secondary" onClick={() => logout.mutate()}>
          Logout
        </button>
      </nav>
      <Outlet />
    </div>
  )
}
