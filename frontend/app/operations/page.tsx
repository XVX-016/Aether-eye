"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";

import { Navbar } from "@/components/Navbar";
import { EventFeed } from "@/components/operations/EventFeed";
import { SiteDetailPanel } from "@/components/operations/SiteDetailPanel";
import { SiteStatusPanel } from "@/components/operations/SiteStatus";
import { StatCards } from "@/components/operations/StatCards";
import { TimelineSlider } from "@/components/operations/TimelineSlider";
import { fetchGlobalIntel, fetchOperationsEvents, fetchSiteStatus, fetchSitesGeoJson } from "@/lib/api";
import type { DetectionMapHandle } from "@/components/operations/DetectionMap";
import type { IntelArticle, OperationsEvent, SiteGeoJson, SiteProperties, SiteStatus } from "@/types/operations";

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

function relativeTime(iso?: string) {
    if (!iso) return "--";
    const diffMs = Date.now() - new Date(iso).getTime();
    const mins = Math.max(0, Math.floor(diffMs / 60000));
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

function intelTierBadge(tier?: number) {
    if (tier === 1) {
        return { label: "WIRE", style: { border: "1px solid #164E63", color: "#22D3EE", borderRadius: "2px" } };
    }
    if (tier === 2) {
        return { label: "MAJOR", style: { border: "1px solid #451A03", color: "#F59E0B", borderRadius: "2px" } };
    }
    return { label: "SPEC", style: { border: "1px solid #1F2937", color: "#6B7280", borderRadius: "2px" } };
}

export default function OperationsPage() {
    const mapRef = useRef<DetectionMapHandle | null>(null);
    const [events, setEvents] = useState<OperationsEvent[]>([]);
    const [selectedSite, setSelectedSite] = useState<SiteProperties | null>(null);
    const [sitesGeoJson, setSitesGeoJson] = useState<SiteGeoJson | null>(null);
    const [siteStatuses, setSiteStatuses] = useState<SiteStatus[]>([]);
    const [globalIntel, setGlobalIntel] = useState<IntelArticle[]>([]);
    const [globalIntelLoading, setGlobalIntelLoading] = useState(false);
    const [feedTab, setFeedTab] = useState<"GLOBAL_TRENDS" | "EVENTS" | "GLOBAL_INTEL">("GLOBAL_TRENDS");
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

    useEffect(() => {
        let alive = true;
        const loadSiteData = async () => {
            try {
                const [geo, statuses] = await Promise.all([fetchSitesGeoJson(), fetchSiteStatus(30)]);
                if (alive) {
                    setSitesGeoJson(geo);
                    setSiteStatuses(statuses);
                }
            } catch (err) {
                console.error("Failed to load site registry data", err);
            }
        };
        void loadSiteData();
        const timer = window.setInterval(() => void loadSiteData(), 60_000);
        return () => {
            alive = false;
            window.clearInterval(timer);
        };
    }, []);

    useEffect(() => {
        let alive = true;
        const loadGlobalIntel = async () => {
            setGlobalIntelLoading(true);
            try {
                const articles = await fetchGlobalIntel(48);
                if (alive) {
                    setGlobalIntel(articles);
                }
            } catch (err) {
                console.error("Failed to load global intel", err);
                if (alive) {
                    setGlobalIntel([]);
                }
            } finally {
                if (alive) {
                    setGlobalIntelLoading(false);
                }
            }
        };
        void loadGlobalIntel();
        const timer = window.setInterval(() => void loadGlobalIntel(), 30 * 60_000);
        return () => {
            alive = false;
            window.clearInterval(timer);
        };
    }, []);

    const handleEventClick = useCallback((event: OperationsEvent) => {
        mapRef.current?.flyTo(event.lon, event.lat, 14);
    }, []);

    const handleSiteSelect = useCallback((site: SiteProperties | null) => {
        setSelectedSite(site);
    }, []);

    const handleSiteSelectById = useCallback((siteId: string) => {
        const site = sitesGeoJson?.features.find((feature) => feature.properties.id === siteId)?.properties ?? null;
        setSelectedSite(site);
    }, [sitesGeoJson]);

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

    const topSites = useMemo(
        () => [...siteStatuses].sort((a, b) => (b.anomaly_factor ?? 0) - (a.anomaly_factor ?? 0)),
        [siteStatuses],
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
                                sitesGeoJson={sitesGeoJson}
                                onReady={handleMapReady}
                                onSiteClick={handleSiteSelect}
                                selectedSiteId={selectedSite?.id ?? null}
                            />
                        </section>

                        <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
                            {selectedSite ? (
                                <SiteDetailPanel site={selectedSite} onClose={() => setSelectedSite(null)} />
                            ) : (
                                <aside className="ops-feed glass-panel" style={{ height: "100%" }}>
                                <div className="ops-feed-header">
                                    <div>
                                        <div className="ops-kicker mono">Intelligence Feed</div>
                                        <h2 className="ops-feed-title mono">
                                            {feedTab === "GLOBAL_TRENDS" ? "Global Trends" : feedTab === "GLOBAL_INTEL" ? "Global Intel" : "Event Feed"}
                                        </h2>
                                    </div>
                                    <div className="ops-feed-count mono">
                                        {feedTab === "GLOBAL_TRENDS" ? topSites.length : feedTab === "GLOBAL_INTEL" ? globalIntel.length : events.length}
                                    </div>
                                </div>

                                <div className="ops-filter-row" style={{ marginBottom: "0.9rem" }}>
                                    <button
                                        type="button"
                                        className={feedTab === "GLOBAL_TRENDS" ? "ops-filter-btn ops-filter-btn-active" : "ops-filter-btn"}
                                        onClick={() => setFeedTab("GLOBAL_TRENDS")}
                                    >
                                        GLOBAL TRENDS
                                    </button>
                                    <button
                                        type="button"
                                        className={feedTab === "EVENTS" ? "ops-filter-btn ops-filter-btn-active" : "ops-filter-btn"}
                                        onClick={() => setFeedTab("EVENTS")}
                                    >
                                        EVENTS
                                    </button>
                                    <button
                                        type="button"
                                        className={feedTab === "GLOBAL_INTEL" ? "ops-filter-btn ops-filter-btn-active" : "ops-filter-btn"}
                                        onClick={() => setFeedTab("GLOBAL_INTEL")}
                                    >
                                        GLOBAL INTEL
                                    </button>
                                </div>

                                {feedTab === "GLOBAL_TRENDS" ? (
                                    <div className="ops-feed-list">
                                        {topSites.map((site) => (
                                            <button
                                                key={site.id}
                                                type="button"
                                                className="ops-event-card"
                                                onClick={() => handleSiteSelectById(site.id)}
                                            >
                                                <div className="ops-event-card-top">
                                                    <span className={`ops-priority-badge mono ${site.status === "anomalous" ? "ops-badge-high" : site.status === "elevated" ? "ops-badge-medium" : "ops-badge-low"}`}>
                                                        {site.status.toUpperCase()}
                                                    </span>
                                                </div>
                                                <div className="ops-event-title mono">{site.name}</div>
                                                <div className="ops-event-meta mono">
                                                    <span>Today {site.today_count ?? "--"}</span>
                                                    <span>{site.baseline ? site.baseline.toFixed(1) : "--"}</span>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                ) : feedTab === "GLOBAL_INTEL" ? (
                                    <div className="ops-feed-list">
                                        {globalIntelLoading ? (
                                            Array.from({ length: 3 }).map((_, index) => (
                                                <div
                                                    key={`global-intel-skeleton-${index}`}
                                                    className="ops-stat-pulse"
                                                    style={{
                                                        padding: "0.85rem",
                                                        borderRadius: "2px",
                                                        background: "rgba(255,255,255,0.03)",
                                                        border: "1px solid rgba(255,255,255,0.06)",
                                                        marginBottom: "0.65rem",
                                                    }}
                                                >
                                                    <div style={{ width: "42%", height: "0.7rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.2)", marginBottom: "0.6rem" }} />
                                                    <div style={{ width: "100%", height: "0.8rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.16)", marginBottom: "0.35rem" }} />
                                                    <div style={{ width: "84%", height: "0.8rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.16)", marginBottom: "0.6rem" }} />
                                                    <div style={{ width: "28%", height: "0.7rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.14)" }} />
                                                </div>
                                            ))
                                        ) : globalIntel.length === 0 ? (
                                            <div style={{ color: "var(--text-muted)" }}>
                                                <div style={{ marginBottom: "0.35rem" }}>No recent global intelligence</div>
                                                <div className="mono" style={{ fontSize: "0.72rem", color: "rgba(148, 163, 184, 0.8)" }}>Intel updates every 30 minutes</div>
                                            </div>
                                        ) : (
                                            globalIntel.map((article, index) => {
                                                const tier = intelTierBadge(article.source_tier);
                                                return (
                                                    <a
                                                        key={`${article.url ?? "global-intel"}-${index}`}
                                                        href={article.url}
                                                        target="_blank"
                                                        rel="noreferrer"
                                                        style={{
                                                            display: "block",
                                                            textDecoration: "none",
                                                            color: "inherit",
                                                            padding: "0.85rem",
                                                            borderRadius: "2px",
                                                            background: "rgba(255,255,255,0.025)",
                                                            border: "1px solid rgba(255,255,255,0.06)",
                                                            marginBottom: "0.65rem",
                                                            transition: "background 140ms ease, border-color 140ms ease",
                                                        }}
                                                        onMouseEnter={(event) => {
                                                            event.currentTarget.style.background = "rgba(255,255,255,0.05)";
                                                            event.currentTarget.style.borderColor = "rgba(255,255,255,0.11)";
                                                        }}
                                                        onMouseLeave={(event) => {
                                                            event.currentTarget.style.background = "rgba(255,255,255,0.025)";
                                                            event.currentTarget.style.borderColor = "rgba(255,255,255,0.06)";
                                                        }}
                                                    >
                                                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.75rem", marginBottom: "0.55rem" }}>
                                                            <span className="mono" style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                                                                {article.source ?? "Unknown source"}
                                                            </span>
                                                            <span className="mono" style={{ padding: "0.16rem 0.45rem", fontSize: "0.6rem", letterSpacing: "0.12em", ...tier.style }}>
                                                                {tier.label}
                                                            </span>
                                                        </div>
                                                        <div
                                                            style={{
                                                                fontWeight: 500,
                                                                lineHeight: 1.45,
                                                                marginBottom: "0.55rem",
                                                                display: "-webkit-box",
                                                                WebkitLineClamp: 2,
                                                                WebkitBoxOrient: "vertical",
                                                                overflow: "hidden",
                                                            }}
                                                        >
                                                            {article.title ?? "Untitled article"}
                                                        </div>
                                                        <div className="mono" style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                                                            {relativeTime(article.published_at)}
                                                        </div>
                                                    </a>
                                                );
                                            })
                                        )}
                                    </div>
                                ) : (
                                    <EventFeed
                                        events={events}
                                        onEventClick={handleEventClick}
                                        loading={loading}
                                    />
                                )}
                            </aside>
                        )}
                        </div>
                    </div>

                    <div className="glass-panel" style={{ padding: "0.9rem 1.1rem", marginTop: "1rem" }}>
                        <TimelineSlider
                            onChange={handleTimelineChange}
                            value={displayedRange}
                        />
                    </div>

                    <SiteStatusPanel onSiteClick={handleSiteSelectById} />
                </div>
            </div>
        </div>
    );
}
