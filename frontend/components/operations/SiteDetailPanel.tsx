"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchSiteIntel, fetchSiteStatus } from "@/lib/api";
import type { SiteIntelResponse, SiteProperties, SiteStatus } from "@/types/operations";

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

    useEffect(() => {
        if (!site) {
            return;
        }
        let alive = true;
        const load = async () => {
            setIntelLoading(true);
            try {
                const [statusData, intelData] = await Promise.all([
                    fetchSiteStatus(30),
                    fetchSiteIntel(site.id, 48),
                ]);
                if (alive) {
                    setStatusRows(statusData);
                    setIntel(intelData);
                }
            } catch (error) {
                console.error("Failed to load site detail", error);
                if (alive) {
                    setIntel({ site_id: site.id, articles: [], hours: 48 });
                }
            } finally {
                if (alive) {
                    setIntelLoading(false);
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
                <div style={{ marginTop: "0.65rem", color: "var(--text-muted)" }}>Detection history - coming in Stage 5.3</div>
            </div>
        </aside>
    );
}
