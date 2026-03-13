export type OperationsAoi = {
    id: string;
    name: string;
    bbox: [number, number, number, number];
    center: [number, number];
};

export const OPERATIONS_AOIS: OperationsAoi[] = [
    {
        id: "dubai_airport",
        name: "Dubai Airport AOI",
        bbox: [55.33, 25.23, 55.4, 25.27],
        center: [55.36, 25.25],
    },
];

export function resolveAoiName(
    lat: number,
    lon: number,
    fallback?: string | null,
) {
    if (fallback) {
        return fallback;
    }

    const match = OPERATIONS_AOIS.find((aoi) => {
        const [minLon, minLat, maxLon, maxLat] = aoi.bbox;
        return lon >= minLon && lon <= maxLon && lat >= minLat && lat <= maxLat;
    });
    return match?.name ?? null;
}

