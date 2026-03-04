import axios from "axios";
import type {
    AircraftClassificationResponse,
    AircraftDetectionsResponse,
    AircraftGradCamResponse,
    ChangeDetectionResponse,
} from "./types";

export async function runAircraftDetection(file: File, country?: string) {
    const form = new FormData();
    form.append("image", file);

    const countryParam = country ? `?country=${encodeURIComponent(country)}` : "";
    const res = await axios.post<AircraftDetectionsResponse>(
        `/api/v1/aircraft-detect${countryParam}`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } },
    );
    return res.data;
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
    const res = await axios.post<ChangeDetectionResponse>(
        `/api/v1/change-detection?include_mask=${includeMask ? "true" : "false"}${countryParam}${semanticParam}`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } },
    );
    return res.data;
}

export async function runAircraftClassification(file: File, country?: string) {
    const form = new FormData();
    form.append("image", file);
    const countryParam = country ? `?country=${encodeURIComponent(country)}` : "";
    const res = await axios.post<AircraftClassificationResponse>(
        `/api/v1/aircraft-classify${countryParam}`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } },
    );
    return res.data;
}

export async function runAircraftGradCam(file: File, country?: string) {
    const form = new FormData();
    form.append("image", file);
    const countryParam = country ? `?country=${encodeURIComponent(country)}` : "";
    const res = await axios.post<AircraftGradCamResponse>(
        `/api/v1/aircraft-gradcam${countryParam}`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } },
    );
    return res.data;
}
