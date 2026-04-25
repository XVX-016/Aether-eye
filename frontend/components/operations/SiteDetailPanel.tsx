"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchSiteFlights, fetchSiteIntel, fetchSiteStatus } from "@/lib/api";
import type { FlightActivity, SiteIntelResponse, SiteProperties, SiteStatus } from "@/types/operations";

type Props = {
    site: SiteProperties | null;
    onClose: () => void;
};

function formatLabel(value: string) {
    return value.replaceAll("_", " ").toUpperCase();
}

function statusStyle(value: string) {
    const normalized = value.toLowerCase();
    if (normalized === "anomalous") {
        return { border: "1px solid #7F1D1D", color: "#EF4444", padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.1em", borderRadius: 2 };
    }
    if (normalized === "elevated") {
        return { border: "1px solid #92400E", color: "#D97706", padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.1em", borderRadius: 2 };
    }
    return { border: "1px solid #374151", color: "#6B7280", padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.1em", borderRadius: 2 };
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

function getTierBadge(tier?: number) {
    if (tier === 1) {
        return { label: "WIRE", style: { border: "1px solid #164E63", color: "#22D3EE", borderRadius: 2 } };
    }
    if (tier === 2) {
        return { label: "MAJOR", style: { border: "1px solid #451A03", color: "#F59E0B", borderRadius: 2 } };
    }
    return { label: "SPEC", style: { border: "1px solid #1F2937", color: "#6B7280", borderRadius: 2 } };
}

export function SiteDetailPanel({ site, onClose }: Props) {
    const [statusRows, setStatusRows] = useState<SiteStatus[]>([]);
    const [intel, setIntel] = useState<SiteIntelResponse | null>(null);
    const [intelLoading, setIntelLoading] = useState(false);
    const [flights, setFlights] = useState<FlightActivity | null>(null);
    const [flightLoading, setFlightLoading] = useState(false);

    useEffect(() => {
        if (!site) {
            return;
        }
        let alive = true;
        const load = async () => {
            setIntelLoading(true);
            setFlightLoading(true);
            try {
                const [statusData, intelData, flightData] = await Promise.all([
                    fetchSiteStatus(30),
                    fetchSiteIntel(site.id, 48),
                    fetchSiteFlights(site.id, 24),
                ]);
                if (alive) {
                    setStatusRows(statusData);
                    setIntel(intelData);
                    setFlights(flightData);
                }
            } catch (error) {
                console.error("Failed to load site detail", error);
                if (alive) {
                    setIntel({ site_id: site.id, articles: [], hours: 48 });
                    setFlights({
                        site_id: site.id,
                        recent_count: 0,
                        unique_aircraft: 0,
                        on_ground_count: 0,
                        airborne_count: 0,
                        latest_states: [],
                    });
                }
            } finally {
                if (alive) {
                    setIntelLoading(false);
                    setFlightLoading(false);
                }
            }
        };
        void load();
        return () => {
            alive = false;
        };
    }, [site]);

    const status = useMemo(
        () => statusRows.find((row) => row.id === site?.id) ?? null,
        [site?.id, statusRows],
    );

    if (!site) {
        return null;
    }

    return (
        <aside className="ops-feed glass-panel" style={{ height: "100%", overflowY: "auto", padding: "1.1rem", borderRadius: 2 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "1rem", marginBottom: "1rem" }}>
                <div>
                    <div className="ops-kicker mono">Site Detail</div>
                    <h2 className="ops-panel-title" style={{ marginBottom: "0.45rem" }}>{site.name}</h2>
                    <div className="mono" style={{ color: "#4B5563", fontSize: "0.65rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                        {formatLabel(site.type)}  ·  {formatLabel(site.priority)}  ·  {site.country}
                    </div>
                </div>
                <button type="button" className="ops-filter-btn" onClick={onClose}>Close</button>
            </div>

            <div className="glass-panel" style={{ padding: "0.9rem", marginBottom: "1rem", borderRadius: 2 }}>
                <div className="ops-kicker mono">Current Status</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginTop: "0.65rem" }}>
                    <div>
                        <div className="mono" style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Today</div>
                        <div style={{ fontSize: "1.1rem", fontWeight: 700 }}>{status?.today_count ?? "--"}</div>
                    </div>
                    <div>
                        <div className="mono" style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Baseline</div>
                        <div style={{ fontSize: "1.1rem", fontWeight: 700 }}>{status?.baseline ? status.baseline.toFixed(1) : "--"}</div>
                    </div>
                    <div>
                        <div className="mono" style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Anomaly Factor</div>
                        <div style={{ fontSize: "1.1rem", fontWeight: 700 }}>{status?.anomaly_factor ? status.anomaly_factor.toFixed(2) : "--"}</div>
                    </div>
                    <div>
                        <div className="mono" style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Status</div>
                        <span className="mono" style={{ display: "inline-block", marginTop: "0.3rem", textTransform: "uppercase", ...statusStyle(status?.status ?? "normal") }}>
                            {status?.status ?? "normal"}
                        </span>
                    </div>
                </div>
            </div>

            <div className="glass-panel" style={{ padding: "0.9rem", marginBottom: "1rem", borderRadius: 2 }}>
                <div className="ops-kicker mono">Flight Activity</div>
                {flightLoading ? (
                    <div className="ops-stat-pulse" style={{ marginTop: "0.75rem" }}>
                        <div style={{ width: "35%", height: "0.7rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.18)", marginBottom: "0.7rem" }} />
                        <div style={{ width: "22%", height: "1.6rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.16)", marginBottom: "0.85rem" }} />
                        <div style={{ width: "100%", height: "2.8rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.12)", marginBottom: "0.85rem" }} />
                        <div style={{ width: "100%", height: "4.6rem", borderRadius: 2, background: "rgba(148, 163, 184, 0.1)" }} />
                    </div>
                ) : !flights || flights.recent_count === 0 ? (
                    <div style={{ marginTop: "0.75rem", color: "var(--text-muted)" }}>
                        No flight activity detected in last 24h
                    </div>
                ) : (
                    <div style={{ marginTop: "0.75rem" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "end", marginBottom: "0.75rem", gap: "0.75rem" }}>
                            <div>
                                <div className="mono" style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Recent Flights</div>
                                <div style={{ fontSize: "1.5rem", fontWeight: 700, lineHeight: 1.1 }}>{flights.recent_count}</div>
                            </div>
                            {status?.flight_baseline != null && status.flight_baseline > 0 ? (
                                <div style={{ textAlign: "right" }}>
                                    <div className="mono" style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
                                        30-Day Avg: {status.flight_baseline.toFixed(1)}
                                    </div>
                                    <span className="mono" style={{ display: "inline-block", marginTop: "0.3rem", textTransform: "uppercase", ...statusStyle(status.flight_anomaly ?? "normal") }}>
                                        {status.flight_anomaly ?? "normal"}
                                    </span>
                                </div>
                            ) : null}
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "0.75rem", marginBottom: "0.9rem" }}>
                            <div>
                                <div className="mono" style={{ fontSize: "0.62rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Unique Aircraft</div>
                                <div style={{ fontWeight: 700, marginTop: "0.18rem" }}>{flights.unique_aircraft}</div>
                            </div>
                            <div>
                                <div className="mono" style={{ fontSize: "0.62rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>On Ground</div>
                                <div style={{ fontWeight: 700, marginTop: "0.18rem" }}>{flights.on_ground_count}</div>
                            </div>
                            <div>
                                <div className="mono" style={{ fontSize: "0.62rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Airborne</div>
                                <div style={{ fontWeight: 700, marginTop: "0.18rem" }}>{flights.airborne_count}</div>
                            </div>
                        </div>

                        <div style={{ overflowX: "auto" }}>
                            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.75rem" }}>
                                <thead>
                                    <tr className="mono" style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.58rem", letterSpacing: "0.12em" }}>
                                        <th style={{ textAlign: "left", padding: "0.35rem 0.25rem" }}>ICAO24</th>
                                        <th style={{ textAlign: "left", padding: "0.35rem 0.25rem" }}>Callsign</th>
                                        <th style={{ textAlign: "left", padding: "0.35rem 0.25rem" }}>Country</th>
                                        <th style={{ textAlign: "right", padding: "0.35rem 0.25rem" }}>Alt</th>
                                        <th style={{ textAlign: "right", padding: "0.35rem 0.25rem" }}>Speed</th>
                                        <th style={{ textAlign: "right", padding: "0.35rem 0.25rem" }}>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {flights.latest_states.slice(0, 5).map((flight) => (
                                        <tr
                                            key={`${flight.icao24}-${flight.timestamp}`}
                                            style={{
                                                borderTop: "1px solid rgba(255,255,255,0.06)",
                                                color: flight.on_ground ? "rgba(148, 163, 184, 0.72)" : "var(--text-primary)",
                                            }}
                                        >
                                            <td className="mono" style={{ padding: "0.45rem 0.25rem" }}>{flight.icao24.toUpperCase()}</td>
                                            <td className="mono" style={{ padding: "0.45rem 0.25rem" }}>{flight.callsign?.trim() || "--"}</td>
                                            <td style={{ padding: "0.45rem 0.25rem" }}>{flight.origin_country || "--"}</td>
                                            <td style={{ padding: "0.45rem 0.25rem", textAlign: "right" }}>{flight.altitude_m != null ? `${Math.round(flight.altitude_m)}m` : "--"}</td>
                                            <td style={{ padding: "0.45rem 0.25rem", textAlign: "right" }}>{flight.velocity_ms != null ? `${Math.round(flight.velocity_ms)}m/s` : "--"}</td>
                                            <td style={{ padding: "0.45rem 0.25rem", textAlign: "right" }}>
                                                <span className="mono" style={{ display: "inline-block", textTransform: "uppercase", ...statusStyle(flight.on_ground ? "normal" : "elevated") }}>
                                                    {flight.on_ground ? "Ground" : "Airborne"}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>

            <div className="glass-panel" style={{ padding: "0.9rem", marginBottom: "1rem", borderRadius: 2 }}>
                <div className="ops-kicker mono">Intelligence Feed</div>
                <div style={{ marginTop: "0.75rem" }}>
                    {intelLoading ? (
                        Array.from({ length: 3 }).map((_, index) => (
                            <div
                                key={`intel-skeleton-${index}`}
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
                    ) : !intel || intel.articles.length === 0 ? (
                        <div style={{ color: "var(--text-muted)" }}>
                            <div style={{ marginBottom: "0.35rem" }}>No recent intelligence for this site</div>
                            <div className="mono" style={{ fontSize: "0.72rem", color: "rgba(148, 163, 184, 0.8)" }}>Intel updates every 30 minutes</div>
                        </div>
                    ) : intel.articles.map((article, index) => {
                        const tierBadge = getTierBadge(article.source_tier);
                        return (
                            <a
                                key={`${article.url ?? "article"}-${index}`}
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
                                    <span className="mono" style={{ padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.12em", ...tierBadge.style }}>
                                        {tierBadge.label}
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
                    })}
                </div>
            </div>

            <div className="glass-panel" style={{ padding: "0.9rem", borderRadius: 2 }}>
                <div className="ops-kicker mono">Detection Timeline</div>
                <div style={{ marginTop: "0.65rem", color: "var(--text-muted)" }}>Detection history will populate as satellite scenes are processed for this site.</div>
            </div>
        </aside>
    );
}
