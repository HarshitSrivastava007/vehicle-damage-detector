export interface BBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Detection {
  class_name: string
  confidence: number
  bbox: BBox
  polygon: number[][] | null
}

export interface DetectionResponse {
  filename: string
  image_width: number
  image_height: number
  count: number
  detections: Detection[]
  annotated_image_base64: string | null
}

export interface DetectionLogOut {
  id: number
  filename: string
  detection_count: number
  class_counts: Record<string, number>
  cost: string
  created_at: string
}

export interface SessionUser {
  id: number
  email: string
  is_admin: boolean
  is_active: boolean
  cost_per_call: string
  created_at: string
}

export interface AdminUserOut extends SessionUser {
  total_detections: number
}

export interface AdminCreateUserResponse extends AdminUserOut {
  api_key: string
}

export interface UsageSummary {
  calls_this_month: number
  cost_this_month: string
  calls_all_time: number
  cost_all_time: string
}

export interface ApiKeyOut {
  id: number
  prefix: string
  created_at: string
  last_used_at: string | null
  revoked_at: string | null
}

export interface ApiKeyCreateResponse {
  id: number
  api_key: string
  created_at: string
}
