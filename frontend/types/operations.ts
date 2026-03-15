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

export interface SiteProperties {
    id: string;
    name: string;
    country: string;
    type: string;
    lat: number;
    lon: number;
    bbox: [number, number, number, number];
    priority: string;
    tags?: string[];
}

export interface SiteFeature {
    type: "Feature";
    geometry: {
        type: "Point";
        coordinates: [number, number];
    };
    properties: SiteProperties;
}

export interface SiteGeoJson {
    type: "FeatureCollection";
    features: SiteFeature[];
}

export interface IntelArticle {
    title?: string;
    url?: string;
    source?: string;
    source_tier?: number;
    published_at?: string;
    site_id?: string | null;
}

export interface SiteIntelResponse {
    site_id: string;
    articles: IntelArticle[];
    hours: number;
}
