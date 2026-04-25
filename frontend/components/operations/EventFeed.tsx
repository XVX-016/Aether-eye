"use client";

import { useMemo, useState } from "react";

import type { OperationsEvent } from "@/types/operations";

const DUBAI_AIRPORT_BBOX: [number, number, number, number] = [55.33, 25.23, 55.4, 25.27];

export type EventFilter = "ALL" | "SURGE" | "NEW_OBJECT" | "CHANGE";

type Props = {
    events: OperationsEvent[];
    onEventClick: (event: OperationsEvent) => void;
    loading: boolean;
};

const FILTERS: EventFilter[] = ["ALL", "SURGE", "NEW_OBJECT", "CHANGE"];

function relativeTime(iso: string) {
    const now = Date.now();
    const ts = new Date(iso).getTime();
    const diffMs = Math.max(0, now - ts);
    const diffMinutes = Math.floor(diffMs / 60_000);
    if (diffMinutes < 1) {
        return "just now";
    }
    if (diffMinutes < 60) {
        return `${diffMinutes}m ago`;
    }
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) {
        return `${diffHours}h ago`;
    }
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
}

function badgeClass(event: OperationsEvent) {
    if (event.event_type === "ACTIVITY_SURGE") {
        return { label: "HIGH", className: "ops-badge-high" };
    }
    if (event.event_type === "ELEVATED_ACTIVITY" || event.event_type === "NEW_OBJECT") {
        return { label: "MED", className: "ops-badge-medium" };
    }
    if ((event.confidence ?? 0) >= 0.7) {
        return { label: "HIGH", className: "ops-badge-high" };
    }
    if ((event.confidence ?? 0) >= 0.4) {
        return { label: "MED", className: "ops-badge-medium" };
    }
    return { label: "LOW", className: "ops-badge-low" };
}

function formatLocation(event: OperationsEvent) {
    const [minLon, minLat, maxLon, maxLat] = DUBAI_AIRPORT_BBOX;
    if (event.lon >= minLon && event.lon <= maxLon && event.lat >= minLat && event.lat <= maxLat) {
        return "Dubai Airport";
    }
    if (event.aoi_name) {
        return event.aoi_name
            .split("_")
            .filter(Boolean)
            .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
            .join(" ");
    }
    return `${event.lat.toFixed(3)}, ${event.lon.toFixed(3)}`;
}

function matchesFilter(event: OperationsEvent, filter: EventFilter) {
    if (filter === "ALL") {
        return true;
    }
    if (filter === "SURGE") {
        return event.event_type === "ACTIVITY_SURGE";
    }
    if (filter === "NEW_OBJECT") {
        return event.event_type === "NEW_OBJECT";
    }
    return event.event_type.includes("CHANGE");
}

function formatType(eventType: string) {
    return eventType.replaceAll("_", " ");
}

export function EventFeed({ events, onEventClick, loading }: Props) {
    const [filter, setFilter] = useState<EventFilter>("ALL");

    const filteredEvents = useMemo(
        () => events.filter((event) => matchesFilter(event, filter)),
        [events, filter],
    );

    return (
        <>
            <div className="ops-filter-row">
                {FILTERS.map((value) => (
                    <button
                        key={value}
                        type="button"
                        className={filter === value ? "ops-filter-btn ops-filter-btn-active" : "ops-filter-btn"}
                        onClick={() => setFilter(value)}
                    >
                        {value}
                    </button>
                ))}
            </div>

            <div className="ops-feed-list">
                {loading ? (
                    <div className="empty-state small empty-state-plain">Loading events…</div>
                ) : filteredEvents.length === 0 ? (
                    <div className="empty-state small empty-state-plain">No events in the selected window.</div>
                ) : (
                    filteredEvents.map((event) => {
                        const badge = badgeClass(event);
                        return (
                            <button
                                key={event.event_id}
                                type="button"
                                className="ops-event-card"
                                onClick={() => onEventClick(event)}
                            >
                                <div className="ops-event-card-top">
                                    <span className={`ops-priority-badge mono ${badge.className}`}>
                                        {badge.label}
                                    </span>
                                    <span className="ops-event-time mono">{relativeTime(event.timestamp)}</span>
                                </div>
                                <div className="ops-event-title mono">{formatType(event.event_type)}</div>
                                <div className="ops-event-location">{formatLocation(event)}</div>
                                <div className="ops-event-meta mono">
                                    <span>{(event.confidence * 100).toFixed(1)}%</span>
                                    <span>
                                        {event.lat.toFixed(3)}, {event.lon.toFixed(3)}
                                    </span>
                                </div>
                            </button>
                        );
                    })
                )}
            </div>
        </>
    );
}
