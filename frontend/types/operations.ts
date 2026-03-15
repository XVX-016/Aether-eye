export interface OperationsEvent {
    event_id: string;
    event_type: string;
    lat: number;
    lon: number;
    confidence: number;
    priority: string;
    timestamp: string;
    aoi_name?: string;
}

export interface SiteStatus {
    id: string;
    name: string;
    type: string;
    priority: string;
    country: string;
    today_count?: number | null;
    baseline?: number | null;
    anomaly_factor?: number | null;
    status: "normal" | "elevated" | "anomalous" | string;
}
