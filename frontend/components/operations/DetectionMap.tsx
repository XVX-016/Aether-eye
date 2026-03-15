"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import maplibregl, { type GeoJSONSource, type LngLatLike, type Map, type MapMouseEvent } from "maplibre-gl";

import { fetchSitesGeoJson } from "@/lib/api";
import type { OperationsEvent, SiteFeature, SiteGeoJson, SiteProperties } from "@/types/operations";

type Props = {
    events: OperationsEvent[];
    onReady?: (map: DetectionMapHandle | null) => void;
    onSiteClick?: (siteProperties: SiteProperties) => void;
    selectedSiteId?: string | null;
};

export type DetectionMapHandle = {
    flyTo: (lon: number, lat: number, zoom?: number) => void;
};

const EVENT_COLORS: Record<string, string> = {
    ACTIVITY_SURGE: "#F59E0B",
    NEW_OBJECT: "#06B6D4",
    CHANGE_DETECTED: "#EF4444",
    ELEVATED_ACTIVITY: "#F59E0B",
};

const PRIORITY_COLORS: Record<string, string> = {
    critical: "#EF4444",
    high: "#F59E0B",
    medium: "#6B7280",
};

const PRIORITY_RADIUS: Record<string, number> = {
    critical: 10,
    high: 8,
    medium: 6,
};

const DEFAULT_CENTER: [number, number] = [20, 20];

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
                color: EVENT_COLORS[event.event_type] ?? "#EF4444",
            },
        })),
    };
}

function buildSiteBboxCollection(sites: SiteGeoJson | null) {
    return {
        type: "FeatureCollection" as const,
        features: (sites?.features ?? []).map((feature) => {
            const [minLon, minLat, maxLon, maxLat] = feature.properties.bbox;
            const color = PRIORITY_COLORS[feature.properties.priority] ?? "#6B7280";
            return {
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
                    ...feature.properties,
                    color,
                },
            };
        }),
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

function formatSiteType(value: string) {
    return value.replaceAll("_", " ");
}

export const DetectionMap = forwardRef<DetectionMapHandle, Props>(function DetectionMap(
    { events, onReady, onSiteClick, selectedSiteId = null },
    ref,
) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const mapRef = useRef<Map | null>(null);
    const popupRef = useRef<maplibregl.Popup | null>(null);
    const eventsRef = useRef<OperationsEvent[]>(events);
    const siteGeoJsonRef = useRef<SiteGeoJson | null>(null);

    useEffect(() => {
        eventsRef.current = events;
    }, [events]);

    useImperativeHandle(ref, () => ({
        flyTo(lon: number, lat: number, zoom = 14) {
            mapRef.current?.flyTo({ center: [lon, lat] as LngLatLike, zoom, essential: true });
        },
    }));

    useEffect(() => {
        let cancelled = false;
        const load = async () => {
            try {
                const data = await fetchSitesGeoJson();
                if (!cancelled) {
                    siteGeoJsonRef.current = data;
                    const map = mapRef.current;
                    if (map?.isStyleLoaded()) {
                        const siteSource = map.getSource("ops-sites") as GeoJSONSource | undefined;
                        siteSource?.setData(data);
                        const bboxSource = map.getSource("ops-site-bboxes") as GeoJSONSource | undefined;
                        bboxSource?.setData(buildSiteBboxCollection(data));
                    }
                }
            } catch (error) {
                console.error("Failed to load sites geojson", error);
            }
        };
        void load();
        return () => {
            cancelled = true;
        };
    }, []);

    useEffect(() => {
        if (!containerRef.current || mapRef.current) {
            return;
        }

        const map = new maplibregl.Map({
            container: containerRef.current,
            style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            center: DEFAULT_CENTER,
            zoom: 2.5,
        });

        map.addControl(new maplibregl.NavigationControl(), "top-right");

        map.on("load", () => {
            map.addSource("ops-sites", {
                type: "geojson",
                data: siteGeoJsonRef.current ?? { type: "FeatureCollection", features: [] },
            });
            map.addLayer({
                id: "ops-sites-circle",
                type: "circle",
                source: "ops-sites",
                paint: {
                    "circle-radius": [
                        "match",
                        ["get", "priority"],
                        "critical", PRIORITY_RADIUS.critical,
                        "high", PRIORITY_RADIUS.high,
                        "medium", PRIORITY_RADIUS.medium,
                        6,
                    ],
                    "circle-color": [
                        "match",
                        ["get", "priority"],
                        "critical", PRIORITY_COLORS.critical,
                        "high", PRIORITY_COLORS.high,
                        "medium", PRIORITY_COLORS.medium,
                        "#6B7280",
                    ],
                    "circle-stroke-width": 1.25,
                    "circle-stroke-color": "#020408",
                    "circle-opacity": 0.9,
                },
            });

            map.addSource("ops-site-bboxes", {
                type: "geojson",
                data: buildSiteBboxCollection(siteGeoJsonRef.current),
            });
            map.addLayer({
                id: "ops-site-bboxes-fill",
                type: "fill",
                source: "ops-site-bboxes",
                minzoom: 6,
                paint: {
                    "fill-color": ["get", "color"],
                    "fill-opacity": 0.3,
                },
            });
            map.addLayer({
                id: "ops-site-bboxes-line",
                type: "line",
                source: "ops-site-bboxes",
                minzoom: 6,
                paint: {
                    "line-color": ["get", "color"],
                    "line-width": 2,
                    "line-opacity": 1,
                },
            });

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
                    "circle-opacity": 0.7,
                },
            });

            map.on("mouseenter", "ops-sites-circle", () => {
                map.getCanvas().style.cursor = "pointer";
            });
            map.on("mouseleave", "ops-sites-circle", () => {
                map.getCanvas().style.cursor = "";
            });
            map.on("mouseenter", "ops-events-circle", () => {
                map.getCanvas().style.cursor = "pointer";
            });
            map.on("mouseleave", "ops-events-circle", () => {
                map.getCanvas().style.cursor = "";
            });

            map.on("click", "ops-sites-circle", (event: MapMouseEvent & { features?: any[] }) => {
                const feature = event.features?.[0] as SiteFeature | undefined;
                if (!feature || feature.geometry.type !== "Point") {
                    return;
                }
                const props = feature.properties;
                popupRef.current?.remove();
                popupRef.current = new maplibregl.Popup({ offset: 16 })
                    .setLngLat(feature.geometry.coordinates)
                    .setHTML(
                        `
                        <div class="ops-map-popup">
                          <div class="ops-map-popup-title">${props.name}</div>
                          <div>${formatSiteType(props.type)}</div>
                          <div>${props.country}</div>
                        </div>
                        `,
                    )
                    .addTo(map);
                onSiteClick?.(props);
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
                    map.flyTo({ center: [selected.lon, selected.lat], zoom: 10, essential: true });
                }
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
    }, [onReady, onSiteClick]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map || !map.isStyleLoaded()) {
            return;
        }
        const source = map.getSource("ops-events") as GeoJSONSource | undefined;
        source?.setData(buildEventCollection(events));
    }, [events]);

    useEffect(() => {
        if (!selectedSiteId || !siteGeoJsonRef.current || !mapRef.current) {
            return;
        }
        const feature = siteGeoJsonRef.current.features.find((item) => item.properties.id === selectedSiteId);
        if (!feature) {
            return;
        }
        const [lon, lat] = feature.geometry.coordinates;
        mapRef.current.flyTo({ center: [lon, lat], zoom: 10, essential: true });
    }, [selectedSiteId]);

    return <div ref={containerRef} className="ops-map-canvas" />;
});
