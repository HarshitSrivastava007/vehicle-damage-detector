import { ApiError } from "./client"
import type { DetectionResponse } from "./types"

export interface DetectResult {
  detections: DetectionResponse
  annotatedImageUrl: string
}

export async function runDetection(file: File): Promise<DetectResult> {
  const form = new FormData()
  form.append("file", file)

  // One request, not two: /detect?include_annotated=true returns both the
  // JSON detections and the annotated PNG (base64) together. Calling the
  // endpoint twice (once for JSON, once with ?annotate=true) would log two
  // DetectionLog rows for what's a single user action.
  const res = await fetch("/api/v1/detect?include_annotated=true", {
    method: "POST",
    credentials: "include",
    body: form,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail ?? `Request failed (${res.status})`)
  }

  const detections: DetectionResponse = await res.json()
  if (!detections.annotated_image_base64) {
    throw new ApiError(500, "server did not return an annotated image")
  }

  return {
    detections,
    annotatedImageUrl: `data:image/png;base64,${detections.annotated_image_base64}`,
  }
}
