import { useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { runDetection } from "../api/detect"
import { ApiError } from "../api/client"
import { Badge } from "./Badge"
import { StatusMessage } from "./StatusMessage"

export function DetectorTestPanel() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [status, setStatus] = useState("")
  const [statusError, setStatusError] = useState(false)
  const [running, setRunning] = useState(false)
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null)
  const [detections, setDetections] = useState<
    { class_name: string; confidence: number; bbox: { x1: number; y1: number; x2: number; y2: number } }[]
  >([])
  const queryClient = useQueryClient()

  function setFile(file: File | null | undefined) {
    if (!file) return
    setSelectedFile(file)
    setStatus("")
    setStatusError(false)
  }

  async function handleRun() {
    if (!selectedFile) return
    setRunning(true)
    setStatus("Running inference…")
    setStatusError(false)

    try {
      const result = await runDetection(selectedFile)
      setAnnotatedUrl(result.annotatedImageUrl)
      setDetections(result.detections.detections)
      setStatus(`Found ${result.detections.count} detection(s) in ${result.detections.filename}.`)
      queryClient.invalidateQueries({ queryKey: ["detections"] })
      queryClient.invalidateQueries({ queryKey: ["usage-summary"] })
    } catch (err) {
      setStatus(err instanceof ApiError ? err.message : "Something went wrong")
      setStatusError(true)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="panel">
      <label
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          setFile(e.dataTransfer.files[0])
        }}
        style={{
          display: "block",
          border: `2px dashed ${dragOver ? "var(--accent)" : "var(--border)"}`,
          borderRadius: 10,
          padding: 32,
          textAlign: "center",
          cursor: "pointer",
        }}
      >
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          style={{ display: "none" }}
          onChange={(e) => setFile(e.target.files?.[0])}
        />
        <div>{selectedFile ? selectedFile.name : "Click to choose an image, or drag one here"}</div>
        <div style={{ color: "var(--muted)", fontSize: "0.85rem", marginTop: 6 }}>JPEG, PNG, or WEBP</div>
      </label>

      <button disabled={!selectedFile || running} onClick={handleRun} style={{ marginTop: 16 }}>
        Run Detection
      </button>
      <StatusMessage message={status} error={statusError} />

      {(annotatedUrl || detections.length > 0) && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 20 }}>
          <div>
            <h3>Annotated</h3>
            {annotatedUrl && <img src={annotatedUrl} alt="Annotated result" style={{ width: "100%", borderRadius: 8 }} />}
          </div>
          <div>
            <h3>Detections</h3>
            {detections.length === 0 ? (
              <div style={{ color: "var(--muted)" }}>No damage detected.</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>BBox</th>
                  </tr>
                </thead>
                <tbody>
                  {detections.map((d, i) => (
                    <tr key={i}>
                      <td>
                        <Badge>{d.class_name}</Badge>
                      </td>
                      <td>{(d.confidence * 100).toFixed(1)}%</td>
                      <td>
                        {d.bbox.x1.toFixed(0)}, {d.bbox.y1.toFixed(0)}, {d.bbox.x2.toFixed(0)}, {d.bbox.y2.toFixed(0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
