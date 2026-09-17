export function StatusMessage({ message, error }: { message: string; error?: boolean }) {
  if (!message) return null
  return (
    <div style={{ marginTop: 12, fontSize: "0.9rem", color: error ? "var(--danger)" : "var(--muted)" }}>
      {message}
    </div>
  )
}
