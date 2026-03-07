import axios from "axios";
import type {
    AircraftClassificationResponse,
    AircraftDetectionsResponse,
    AircraftGradCamResponse,
    ChangeDetectionResponse,
} from "./types";

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

    const countryParam = country ? `&country=${encodeURIComponent(country)}` : "";
    const semanticParam = `&semantic=${semantic ? "true" : "false"}`;
    const includeMaskParam = `include_mask=${includeMask ? "true" : "false"}`;
    const maybeCountryQuery = country ? `country=${encodeURIComponent(country)}` : "";
    const maybeSemanticQuery = `semantic=${semantic ? "true" : "false"}`;

    const primary = await postWithFallback<any>(
        [
            `/v1/change-detection?${includeMaskParam}${countryParam}${semanticParam}`,
            `/v1/predict/change?include_overlay=${includeMask ? "true" : "false"}`,
            `/v1/change-detection?${includeMaskParam}${maybeCountryQuery ? `&${maybeCountryQuery}` : ""}${maybeSemanticQuery ? `&${maybeSemanticQuery}` : ""}`,
        ],
        form,
    );

    // Normalize legacy/new schema variants into the frontend type.
    if (typeof primary?.change_score === "number") {
        return primary as ChangeDetectionResponse;
    }

    return {
        change_score:
            typeof primary?.change_ratio === "number"
                ? primary.change_ratio
                : 0,
        regions: primary?.regions ?? [],
        change_mask_base64: primary?.change_mask_base64 ?? primary?.mask_base64 ?? null,
        inference_time_ms: primary?.inference_time_ms ?? null,
        model_name: primary?.model_name ?? null,
        device_used: primary?.device_used ?? null,
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
