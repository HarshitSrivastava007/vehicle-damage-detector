import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useState, type FormEvent } from "react"
import { Navigate, useLocation, useNavigate } from "react-router-dom"
import { ApiError, api } from "../api/client"
import type { SessionUser } from "../api/types"
import { useAuth } from "../auth/useAuth"

export function LoginPage() {
  const { user, isLoading } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  const login = useMutation({
    mutationFn: () => api.post<SessionUser>("/auth/login", { email, password }),
    onSuccess: (data) => {
      queryClient.setQueryData(["session"], data)
      const from = (location.state as { from?: string } | null)?.from ?? "/"
      navigate(from, { replace: true })
    },
  })

  if (!isLoading && user) {
    return <Navigate to="/" replace />
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    login.mutate()
  }

  return (
    <div style={{ position: "relative", minHeight: "100vh", overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: "url(/veridex-login-banner.png)",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      />
      {/* Uniform dark shade over the whole cover, plus a left-to-right
          gradient on top so the form side is darker still — keeps the
          form readable regardless of viewport width, while the image
          overall reads as darker/moodier rather than washed out. */}
      <div style={{ position: "absolute", inset: 0, background: "rgba(15,17,21,0.45)" }} />
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "linear-gradient(90deg, rgba(15,17,21,0.9) 0%, rgba(15,17,21,0.7) 35%, rgba(15,17,21,0.45) 70%, rgba(15,17,21,0.3) 100%)",
        }}
      />

      <div
        style={{
          position: "relative",
          zIndex: 1,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          padding: "0 6%",
        }}
      >
        <div style={{ width: "100%", maxWidth: 360 }}>
          <h1 style={{ fontSize: "1.6rem", marginBottom: 4 }}>Veridex.ai</h1>
          <p style={{ color: "var(--muted)", fontSize: "0.9rem", marginTop: 0, marginBottom: 24 }}>
            AI-powered vehicle damage detection
          </p>
          <form
            onSubmit={handleSubmit}
            className="panel"
            style={{ backdropFilter: "blur(6px)", background: "rgba(23, 26, 33, 0.85)" }}
          >
            <div style={{ marginBottom: 12 }}>
              <label style={{ display: "block", marginBottom: 6, fontSize: "0.85rem", color: "var(--muted)" }}>
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={{ width: "100%" }}
              />
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: "block", marginBottom: 6, fontSize: "0.85rem", color: "var(--muted)" }}>
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{ width: "100%" }}
              />
            </div>
            <button type="submit" disabled={login.isPending} style={{ width: "100%" }}>
              Log in
            </button>
            {login.isError && (
              <div style={{ marginTop: 12, color: "var(--danger)", fontSize: "0.9rem" }}>
                {login.error instanceof ApiError ? login.error.message : "Something went wrong"}
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  )
}
