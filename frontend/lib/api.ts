import axios from "axios";
import type {
    AircraftClassificationResponse,
    AircraftDetectionsResponse,
    AircraftGradCamResponse,
    ChangeDetectionResponse,
    CountResponse,
} from "./types";
import type { AirbaseStatus, OperationsEvent } from "@/types/operations";

const api = axios.create({
    // Use Next.js same-origin API proxy to avoid browser CORS issues.
    baseURL: "/api",
});

async function postWithFallback<T>(paths: string[], form: FormData): Promise<T> {
    let lastErr: unknown = null;
    for (const path of paths) {
        try {
            const res = await api.post<T>(path, form, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            return res.data;
        } catch (err: any) {
            const status = err?.response?.status;
            if (status === 404) {
                lastErr = err;
                continue;
            }
            throw err;
        }
    }
    throw lastErr ?? new Error("No API endpoint matched the request.");
}

async function postMultipart<T>(path: string, form: FormData): Promise<T> {
    const res = await api.post<T>(path, form, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
}

export async function runAircraftDetection(file: File, country?: string) {
    const form = new FormData();
    form.append("image", file);

    const countryParam = country ? `?country=${encodeURIComponent(country)}` : "";
    return postWithFallback<AircraftDetectionsResponse>(
        [
            `/v1/aircraft-detect${countryParam}`,
            `/v1/aircraft-detection${countryParam}`,
        ],
        form,
    );
}

export async function runChangeDetection(
    before: File,
    after: File,
    includeMask = false,
    country?: string,
    semantic = false,
) {
    const form = new FormData();
    form.append("before_image", before);
    form.append("after_image", after);

    const includeMaskParam = `include_mask=${includeMask ? "true" : "false"}`;
    const maybeCountryQuery = country ? `country=${encodeURIComponent(country)}` : "";
    const maybeSemanticQuery = `semantic=${semantic ? "true" : "false"}`;
    const endpoint = `/v1/change-detection?${includeMaskParam}${maybeCountryQuery ? `&${maybeCountryQuery}` : ""}${maybeSemanticQuery ? `&${maybeSemanticQuery}` : ""}&debug=true`;
    const primary = await postMultipart<ChangeDetectionResponse>(endpoint, form);

    return {
        change_score: primary.change_score,
        regions: primary.regions ?? [],
        change_mask_base64: primary.change_mask_base64 ?? null,
        overlay_base64: primary.overlay_base64 ?? null,
        changed_pixels: primary.changed_pixels ?? null,
        debug: primary.debug ?? null,
        inference_time_ms: primary.inference_time_ms ?? null,
        model_name: primary.model_name ?? null,
        device_used: primary.device_used ?? null,
    } satisfies ChangeDetectionResponse;
}

export async function runAircraftClassification(file: File, country?: string) {
    const form = new FormData();
    form.append("image", file);
    const countryParam = country ? `?country=${encodeURIComponent(country)}` : "";
    return postWithFallback<AircraftClassificationResponse>(
        [
            `/v1/aircraft-classify${countryParam}`,
            `/v1/predict/aircraft${countryParam}${countryParam ? "&" : "?"}use_onnx=true`,
        ],
        form,
    );
}

export async function runAircraftGradCam(file: File, country?: string) {
    const form = new FormData();
    form.append("image", file);
    const countryParam = country ? `?country=${encodeURIComponent(country)}` : "";
    return postWithFallback<AircraftGradCamResponse>(
        [`/v1/aircraft-gradcam${countryParam}`],
        form,
    );
}

export async function fetchImagePreview(file: File) {
    const form = new FormData();
    form.append("image", file);
    return postWithFallback<{ width: number; height: number; png_base64: string }>(
        ["/v1/change/preview-image"],
        form,
    );
}

export async function fetchOperationsEvents(params?: {
    hours?: number;
    limit?: number;
    start?: string;
    end?: string;
}) {
    const query = new URLSearchParams();
    if (params?.hours != null) {
        query.set("hours", String(params.hours));
    }
    if (params?.limit != null) {
        query.set("limit", String(params.limit));
    }
    if (params?.start) {
        query.set("start", params.start);
    }
    if (params?.end) {
        query.set("end", params.end);
    }
    const suffix = query.toString() ? `?${query.toString()}` : "";
    const res = await api.get<OperationsEvent[]>(`/events${suffix}`);
    return res.data;
}

export async function fetchOperationsCount(path: "/scenes/count" | "/detections/count" | "/alerts/count") {
    const res = await api.get<CountResponse>(path);
    return res.data;
}

export async function fetchAirbaseStatus(days = 30) {
    const res = await api.get<AirbaseStatus[]>(`/airbase-status?days=${days}`);
    return res.data;
}
