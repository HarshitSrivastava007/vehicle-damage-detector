import { Link } from "react-router-dom"

export function NotFoundPage() {
  return (
    <div style={{ maxWidth: 480, margin: "80px auto", textAlign: "center" }}>
      <h1>404</h1>
      <p style={{ color: "var(--muted)" }}>Page not found.</p>
      <Link to="/">Back to dashboard</Link>
    </div>
  )
}
