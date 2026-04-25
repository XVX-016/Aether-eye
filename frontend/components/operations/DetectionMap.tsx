"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import maplibregl, { type GeoJSONSource, type LayerSpecification, type LngLatLike, type Map, type MapMouseEvent } from "maplibre-gl";

import { fetchGlobalFlights, fetchSitesGeoJson } from "@/lib/api";
import type { FlightState, OperationsEvent, SiteFeature, SiteGeoJson, SiteProperties } from "@/types/operations";

type Props = {
    events: OperationsEvent[];
    sitesGeoJson?: SiteGeoJson | null;
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

function buildFlightCollection(flights: FlightState[]) {
    return {
        type: "FeatureCollection" as const,
        features: flights.map((f) => ({
            type: "Feature" as const,
            geometry: {
                type: "Point" as const,
                coordinates: [f.lon ?? 0, f.lat ?? 0],
            },
            properties: {
                icao24: f.icao24,
                callsign: f.callsign ?? "UNKNOWN",
                altitude: f.altitude_m ?? 0,
                on_ground: f.on_ground,
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
    { events, sitesGeoJson, onReady, onSiteClick, selectedSiteId = null },
    ref,
) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const mapRef = useRef<Map | null>(null);
    const popupRef = useRef<maplibregl.Popup | null>(null);
    const [mapError, setMapError] = useState(false);
    const [flights, setFlights] = useState<FlightState[]>([]);

    const displayEvents = useMemo(() => events.slice(-200), [events]);
    const eventsRef = useRef<OperationsEvent[]>(displayEvents);
    const siteGeoJsonRef = useRef<SiteGeoJson | null>(sitesGeoJson ?? null);

    useEffect(() => {
        eventsRef.current = displayEvents;
    }, [displayEvents]);

    useEffect(() => {
        let active = true;
        const loadFlights = async () => {
            try {
                const data = await fetchGlobalFlights(1);
                if (active) setFlights(data);
            } catch (e) {
                console.warn("Failed to fetch global flights for map", e);
            }
        };
        loadFlights();
        const timer = setInterval(loadFlights, 30_000);
        return () => {
            active = false;
            clearInterval(timer);
        };
    }, []);

    useImperativeHandle(ref, () => ({
        flyTo(lon: number, lat: number, zoom = 14) {
            mapRef.current?.flyTo({ center: [lon, lat] as LngLatLike, zoom, essential: true });
        },
    }));

    useEffect(() => {
        siteGeoJsonRef.current = sitesGeoJson ?? null;
        const map = mapRef.current;
        if (map?.isStyleLoaded() && sitesGeoJson) {
            const siteSource = map.getSource("ops-sites") as GeoJSONSource | undefined;
            siteSource?.setData(sitesGeoJson);
            const bboxSource = map.getSource("ops-site-bboxes") as GeoJSONSource | undefined;
            bboxSource?.setData(buildSiteBboxCollection(sitesGeoJson));
        }
    }, [sitesGeoJson]);

    useEffect(() => {
        if (!containerRef.current || mapRef.current || mapError) {
            return;
        }

        const STYLES = [
            "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            "https://demotiles.maplibre.org/style.json",
        ];
        let styleIndex = 0;

        // Firefox compatibility: check WebGL support explicitly
        const canvas = document.createElement("canvas");
        const gl = canvas.getContext("webgl2") || canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        if (!gl) {
            console.error("WebGL not supported, map cannot initialize.");
            setMapError(true);
            return;
        }

        let map: Map;
        try {
            map = new maplibregl.Map({
                container: containerRef.current,
                style: STYLES[0],
                center: DEFAULT_CENTER,
                zoom: 2.5,
                antialias: false, // Performance fix for Firefox
                preserveDrawingBuffer: true, // Required for certain Firefox privacy modes
            } as any);
        } catch (err) {
            console.error("Map initialization failed:", err);
            setMapError(true);
            return;
        }

        // Firefox fallback: retry with a different style if the first one is blocked (ETP)
        map.on("error", (e) => {
            if (e.error?.message?.includes("style") && styleIndex === 0) {
                console.warn("Primary map style blocked/failed, falling back to MapLibre demo tiles");
                styleIndex = 1;
                map.setStyle(STYLES[1]);
            }
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
                minzoom: 7,
                paint: {
                    "fill-color": ["get", "color"],
                    "fill-opacity": 0.3,
                },
            });
            map.addLayer({
                id: "ops-site-bboxes-line",
                type: "line",
                source: "ops-site-bboxes",
                minzoom: 7,
                paint: {
                    "line-color": ["get", "color"],
                    "line-width": 2,
                    "line-opacity": 1,
                },
            });

            map.addSource("flights-source", {
                type: "geojson",
                data: buildFlightCollection([]),
            });
            map.addLayer({
                id: "flights-layer",
                type: "circle",
                source: "flights-source",
                minzoom: 5,
                paint: {
                    "circle-radius": 4,
                    "circle-color": "#22D3EE",
                    "circle-stroke-width": 1,
                    "circle-stroke-color": "#020408",
                    "circle-opacity": 0.8,
                },
            });

            map.addSource("ops-events", {
                type: "geojson",
                data: buildEventCollection(eventsRef.current),
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

            map.on("zoom", () => {
                const zoom = map.getZoom();
                if (map.getLayer("flights-layer")) {
                    map.setLayoutProperty(
                        "flights-layer",
                        "visibility",
                        zoom >= 5 ? "visible" : "none",
                    );
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
            try {
                popupRef.current?.remove();
                if (mapRef.current) {
                    mapRef.current.remove();
                    mapRef.current = null;
                }
            } catch (e) {
                console.warn("Map cleanup error:", e);
            }
            onReady?.(null);
        };
    }, [onReady, onSiteClick]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map || !map.isStyleLoaded()) {
            return;
        }
        const source = map.getSource("ops-events") as GeoJSONSource | undefined;
        source?.setData(buildEventCollection(displayEvents));
    }, [displayEvents]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map || !map.isStyleLoaded()) {
            return;
        }
        const source = map.getSource("flights-source") as GeoJSONSource | undefined;
        source?.setData(buildFlightCollection(flights));
    }, [flights]);

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

    if (mapError) {
        return (
            <div style={{
                height: "60vh",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "#0a0a0a",
                color: "#4B5563",
                fontFamily: "monospace",
                fontSize: "0.75rem",
                letterSpacing: "0.1em",
            }}
            >
                MAP UNAVAILABLE — RELOAD TO RETRY
            </div>
        );
    }

    return <div ref={containerRef} className="ops-map-canvas" />;
});
