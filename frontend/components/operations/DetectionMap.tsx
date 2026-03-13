"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import maplibregl, { type GeoJSONSource, type LngLatLike, type Map, type MapMouseEvent } from "maplibre-gl";

import type { OperationsEvent } from "@/types/operations";

type Props = {
    events: OperationsEvent[];
    onReady?: (map: DetectionMapHandle | null) => void;
};

export type DetectionMapHandle = {
    flyTo: (lon: number, lat: number, zoom?: number) => void;
};

const EVENT_COLORS: Record<string, string> = {
    ACTIVITY_SURGE: "#F59E0B",
    NEW_OBJECT: "#06B6D4",
    CHANGE_DETECTED: "#EF4444",
};

const DUBAI_AOI_BBOX: [number, number, number, number] = [55.33, 25.23, 55.4, 25.27];
const DEFAULT_CENTER: [number, number] = [55.36, 25.25];

function buildEventCollection(events: OperationsEvent[]) {
    return {
        type: "FeatureCollection" as const,
        features: events.map((event) => ({
            type: "Feature" as const,
            geometry: {
                type: "Point" as const,
                coordinates: [event.lon, event.lat],
            },
            properties: {
                event_id: event.event_id,
                event_type: event.event_type,
                confidence: event.confidence ?? 0,
                priority: event.priority,
                timestamp: event.timestamp,
                color: EVENT_COLORS[event.event_type] ?? "#ef4444",
            },
        })),
    };
}

function buildAoiCollection() {
    const [minLon, minLat, maxLon, maxLat] = DUBAI_AOI_BBOX;
    return {
        type: "FeatureCollection" as const,
        features: [{
            type: "Feature" as const,
            geometry: {
                type: "Polygon" as const,
                coordinates: [[
                    [minLon, minLat],
                    [maxLon, minLat],
                    [maxLon, maxLat],
                    [minLon, maxLat],
                    [minLon, minLat],
                ]],
            },
            properties: {
                id: "dubai_airport",
                name: "Dubai Airport",
            },
        }],
    };
}

function relativeTimestamp(iso: string) {
    const diffMs = Date.now() - new Date(iso).getTime();
    const mins = Math.max(0, Math.floor(diffMs / 60000));
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

export const DetectionMap = forwardRef<DetectionMapHandle, Props>(function DetectionMap(
    { events, onReady },
    ref,
) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const mapRef = useRef<Map | null>(null);
    const popupRef = useRef<maplibregl.Popup | null>(null);
    const eventsRef = useRef<OperationsEvent[]>(events);

    useEffect(() => {
        eventsRef.current = events;
    }, [events]);

    useImperativeHandle(ref, () => ({
        flyTo(lon: number, lat: number, zoom = 14) {
            mapRef.current?.flyTo({ center: [lon, lat] as LngLatLike, zoom, essential: true });
        },
    }));

    useEffect(() => {
        if (!containerRef.current || mapRef.current) {
            return;
        }

        const map = new maplibregl.Map({
            container: containerRef.current,
            style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            center: DEFAULT_CENTER,
            zoom: 10,
        });

        map.addControl(new maplibregl.NavigationControl(), "top-right");

        map.on("load", () => {
            map.addSource("ops-events", {
                type: "geojson",
                data: buildEventCollection([]),
            });
            map.addLayer({
                id: "ops-events-circle",
                type: "circle",
                source: "ops-events",
                paint: {
                    "circle-radius": 7,
                    "circle-color": ["get", "color"],
                    "circle-stroke-width": 1.5,
                    "circle-stroke-color": "#020408",
                    "circle-opacity": 0.95,
                },
            });

            map.addSource("ops-aois", {
                type: "geojson",
                data: buildAoiCollection(),
            });
            map.addLayer({
                id: "ops-aois-line",
                type: "line",
                source: "ops-aois",
                paint: {
                    "line-color": "#3b82f6",
                    "line-width": 2,
                    "line-opacity": 0.8,
                },
            });

            map.on("click", "ops-events-circle", (event: MapMouseEvent & { features?: any[] }) => {
                const feature = event.features?.[0];
                if (!feature || feature.geometry.type !== "Point") {
                    return;
                }
                const [lon, lat] = feature.geometry.coordinates as [number, number];
                const eventId = String(feature.properties?.event_id ?? "");
                const selected = eventsRef.current.find((item) => item.event_id === eventId);
                popupRef.current?.remove();
                popupRef.current = new maplibregl.Popup({ offset: 18 })
                    .setLngLat([lon, lat])
                    .setHTML(
                        `
                        <div class="ops-map-popup">
                          <div class="ops-map-popup-title">${String(feature.properties?.event_type ?? "").replaceAll("_", " ")}</div>
                          <div>Confidence: ${Math.round(Number(feature.properties?.confidence ?? 0) * 100)}%</div>
                          <div>Time: ${relativeTimestamp(String(feature.properties?.timestamp ?? ""))}</div>
                          <div>Lat/Lon: ${lat.toFixed(3)}, ${lon.toFixed(3)}</div>
                        </div>
                        `,
                    )
                    .addTo(map);
                if (selected) {
                    map.flyTo({ center: [selected.lon, selected.lat], zoom: 14, essential: true });
                }
            });

            map.on("mouseenter", "ops-events-circle", () => {
                map.getCanvas().style.cursor = "pointer";
            });
            map.on("mouseleave", "ops-events-circle", () => {
                map.getCanvas().style.cursor = "";
            });
        });

        mapRef.current = map;
        onReady?.({
            flyTo(lon: number, lat: number, zoom = 14) {
                map.flyTo({ center: [lon, lat], zoom, essential: true });
            },
        });

        return () => {
            popupRef.current?.remove();
            map.remove();
            mapRef.current = null;
            onReady?.(null);
        };
    }, [onReady]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map || !map.isStyleLoaded()) {
            return;
        }
        const source = map.getSource("ops-events") as GeoJSONSource | undefined;
        source?.setData(buildEventCollection(events));
    }, [events]);

    return <div ref={containerRef} className="ops-map-canvas" />;
});
