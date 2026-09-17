import { Navigate, Outlet } from "react-router-dom"
import { useAuth } from "./useAuth"

export function RequireAdmin() {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return <div style={{ padding: 32 }}>Loading…</div>
  }
  if (!user?.is_admin) {
    return <Navigate to="/" replace />
  }
  return <Outlet />
}
