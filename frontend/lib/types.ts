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
    regions?: { type: string; bbox: [number, number, number, number] }[] | null;
    change_mask_base64?: string | null;
    overlay_base64?: string | null;
    changed_pixels?: number | null;
    debug?: Record<string, number> | null;
    inference_time_ms?: number | null;
    model_name?: string | null;
    device_used?: string | null;
};

export type AircraftClassificationResponse = {
    class_id: number;
    class_name: string;
    confidence: number;
    origin_country: string;
    friend_or_foe: "FRIEND" | "FOE" | "NEUTRAL" | string;
    inference_time_ms?: number | null;
    model_name?: string | null;
    device_used?: string | null;
};

export type AircraftGradCamResponse = {
    class_id: number;
    class_name: string;
    confidence: number;
    origin_country: string;
    heatmap_base64_png: string;
    inference_time_ms?: number | null;
    model_name?: string | null;
    device_used?: string | null;
};

export type BBoxDetection = {
    bbox: { x1: number; y1: number; x2: number; y2: number };
    detection_confidence: number;
    class_name: string;
    class_confidence: number;
    origin_country: string;
    friend_or_foe: "FRIEND" | "FOE" | "NEUTRAL" | string;
    crop_base64: string;
};

export type DetectClassifyResponse = {
    detections: BBoxDetection[];
    annotated_image_base64: string;
    total_aircraft: number;
    model_used: string;
};

export type VideoFrameResult = {
    frame_number: number;
    timestamp_sec: number;
    detections?: BBoxDetection[];
    annotated_frame_base64?: string;
    total_aircraft?: number;
    source_type?: string;
    error?: string;
};

export type AircraftDetectVideoResponse = {
    frames: VideoFrameResult[];
    total_frames_processed: number;
    source_type: string;
};

export type MainSection =
    | "aircraft-intelligence"
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

export type CountResponse = {
    count: number;
};
