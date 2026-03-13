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

