"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";

import { Navbar } from "@/components/Navbar";
import { EventFeed } from "@/components/operations/EventFeed";
import { SiteStatusPanel } from "@/components/operations/SiteStatus";
import { StatCards } from "@/components/operations/StatCards";
import { TimelineSlider } from "@/components/operations/TimelineSlider";
import { fetchOperationsEvents } from "@/lib/api";
import type { DetectionMapHandle } from "@/components/operations/DetectionMap";
import type { OperationsEvent } from "@/types/operations";

const DetectionMap = dynamic(
    () => import("@/components/operations/DetectionMap").then((mod) => mod.DetectionMap),
    { ssr: false },
);

function startOfDay(date: Date) {
    const next = new Date(date);
    next.setHours(0, 0, 0, 0);
    return next;
}

function endOfDay(date: Date) {
    const next = new Date(date);
    next.setHours(23, 59, 59, 999);
    return next;
}

function defaultRange() {
    const today = new Date();
    const start = new Date(today);
    start.setDate(today.getDate() - 13);
    return [startOfDay(start), endOfDay(today)] as const;
}

export default function OperationsPage() {
    const mapRef = useRef<DetectionMapHandle | null>(null);
    const [events, setEvents] = useState<OperationsEvent[]>([]);
    const [startDate, setStartDate] = useState<Date>(() => defaultRange()[0]);
    const [endDate, setEndDate] = useState<Date>(() => defaultRange()[1]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadEvents = useCallback(async (start: Date, end: Date) => {
        setLoading(true);
        setError(null);
        try {
            const payload = await fetchOperationsEvents({
                start: start.toISOString(),
                end: end.toISOString(),
                limit: 200,
            });
            setEvents(payload);
        } catch (err) {
            console.error("Failed to load operations events", err);
            setEvents([]);
            setError("Unable to load operations events.");
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        void loadEvents(startDate, endDate);
    }, [loadEvents]);

    const handleEventClick = useCallback((event: OperationsEvent) => {
        mapRef.current?.flyTo(event.lon, event.lat, 14);
    }, []);

    const handleMapReady = useCallback((handle: DetectionMapHandle | null) => {
        mapRef.current = handle;
    }, []);

    const handleTimelineChange = useCallback((start: Date, end: Date) => {
        const nextStart = startOfDay(start);
        const nextEnd = endOfDay(end);

        setStartDate((current) => (current.getTime() === nextStart.getTime() ? current : nextStart));
        setEndDate((current) => (current.getTime() === nextEnd.getTime() ? current : nextEnd));
        void loadEvents(nextStart, nextEnd);
    }, [loadEvents]);

    const displayedRange = useMemo(
        () => ({
            start: startDate,
            end: endDate,
        }),
        [endDate, startDate],
    );

    return (
        <div className="app">
            <Navbar />
            <div className="home-body">
                <header className="capability-header">
                    <p className="capability-kicker mono">AETHER EYE</p>
                    <h1 className="capability-title">OPERATIONS DASHBOARD</h1>
                    <p className="capability-description">
                        Live intelligence events, ingestion status, and activity signals across monitored AOIs.
                    </p>
                </header>

                <div className="ops-dashboard">
                    <StatCards />
                    <SiteStatusPanel />

                    <div className="ops-content-grid">
                        <section className="ops-map-panel glass-panel">
                            <div className="ops-panel-header">
                                <div>
                                    <div className="ops-kicker mono">Operations View</div>
                                    <h2 className="ops-panel-title mono">Global Activity Map</h2>
                                </div>
                                <div className="ops-range-chip mono">{events.length} events</div>
                            </div>

                            {error ? <div className="alert alert-error">{error}</div> : null}

                            <DetectionMap
                                ref={mapRef}
                                events={events}
                                onReady={handleMapReady}
                            />

                            <TimelineSlider
                                onChange={handleTimelineChange}
                                value={displayedRange}
                            />
                        </section>

                        <EventFeed
                            events={events}
                            onEventClick={handleEventClick}
                            loading={loading}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
