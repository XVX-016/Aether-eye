export type BBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type AircraftDetection = {
  bbox: BBox;
  confidence: number;
  class_id: number;
};

export type AircraftDetectionsResponse = {
  detections: AircraftDetection[];
  inference_time_ms?: number | null;
  model_name?: string | null;
  device_used?: string | null;
};

export type ChangeDetectionResponse = {
  change_score: number;
  change_mask_base64?: string | null;
  inference_time_ms?: number | null;
  model_name?: string | null;
  device_used?: string | null;
};

export type MainSection =
  | "aircraft-detection"
  | "change-detection"
  | "aircraft-classification"
  | "metrics-dashboard";

export type SystemMetricsSnapshot = {
  inference_time_ms?: number | null;
  model_name?: string | null;
  device_used?: string | null;
  confidence?: number | null;
};


