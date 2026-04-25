"use client";

import { useEffect, useState } from "react";

import { fetchSiteStatus } from "@/lib/api";
import type { SiteStatus } from "@/types/operations";

const POLL_INTERVAL_MS = 60_000;

function formatLabel(value: string) {
    return value.replaceAll("_", " ").toUpperCase();
}

function statusStyle(status: string) {
    if (status === "anomalous") {
        return { border: "1px solid #7F1D1D", color: "#EF4444", padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.1em", borderRadius: 2 };
    }
    if (status === "elevated") {
        return { border: "1px solid #92400E", color: "#D97706", padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.1em", borderRadius: 2 };
    }
    return { border: "1px solid #374151", color: "#6B7280", padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.1em", borderRadius: 2 };
}

function priorityStyle(priority: string) {
    if (priority === "critical") {
        return { border: "1px solid #7F1D1D", color: "#9CA3AF" };
    }
    if (priority === "high") {
        return { border: "1px solid #78350F", color: "#9CA3AF" };
    }
    return { border: "1px solid #374151", color: "#6B7280" };
}

type Props = {
    onSiteClick?: (siteId: string) => void;
};

export function SiteStatusPanel({ onSiteClick }: Props) {
    const [rows, setRows] = useState<SiteStatus[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let alive = true;

        const load = async () => {
            try {
                const data = await fetchSiteStatus(30);
                if (alive) {
                    setRows(data);
                }
            } catch {
                if (alive) {
                    setRows([]);
                }
            } finally {
                if (alive) {
                    setLoading(false);
                }
            }
        };

        void load();
        const timer = window.setInterval(() => void load(), POLL_INTERVAL_MS);
        return () => {
            alive = false;
            window.clearInterval(timer);
        };
    }, []);

    return (
        <section className="glass-panel" style={{ padding: "1rem 1.1rem", marginTop: "1rem", borderRadius: 2 }}>
            <div className="ops-panel-header" style={{ marginBottom: "0.75rem" }}>
                <div>
                    <div className="ops-kicker mono">Site Monitoring</div>
                    <h2 className="ops-panel-title mono" style={{ fontSize: "1rem" }}>Site Status</h2>
                </div>
                <div className="ops-range-chip mono">{rows.length}</div>
            </div>

            <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
                    <thead>
                        <tr className="mono" style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.65rem", letterSpacing: "0.18em" }}>
                            <th style={{ textAlign: "left", padding: "0.45rem 0.35rem" }}>Name</th>
                            <th style={{ textAlign: "left", padding: "0.45rem 0.35rem" }}>Type</th>
                            <th style={{ textAlign: "left", padding: "0.45rem 0.35rem" }}>Priority</th>
                            <th style={{ textAlign: "right", padding: "0.45rem 0.35rem" }}>Today</th>
                            <th style={{ textAlign: "right", padding: "0.45rem 0.35rem" }}>Flights</th>
                            <th style={{ textAlign: "right", padding: "0.45rem 0.35rem" }}>Baseline</th>
                            <th style={{ textAlign: "right", padding: "0.45rem 0.35rem" }}>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            <tr>
                                <td colSpan={7} style={{ padding: "0.85rem 0.35rem", color: "var(--text-muted)" }}>Loading site status...</td>
                            </tr>
                        ) : rows.length === 0 ? (
                            <tr>
                                <td colSpan={7} style={{ padding: "0.85rem 0.35rem", color: "var(--text-muted)" }}>No site status available.</td>
                            </tr>
                        ) : rows.map((row) => (
                            <tr
                                key={row.id}
                                style={{ borderTop: "1px solid rgba(255,255,255,0.06)", cursor: onSiteClick ? "pointer" : "default" }}
                                onClick={() => onSiteClick?.(row.id)}
                            >
                                <td style={{ padding: "0.65rem 0.35rem", color: "var(--text-primary)", fontWeight: 600 }}>{row.name}</td>
                                <td className="mono" style={{ padding: "0.65rem 0.35rem", color: "#4B5563", fontSize: "0.6rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>{formatLabel(row.type)}</td>
                                <td style={{ padding: "0.65rem 0.35rem" }}>
                                    <span className="mono" style={{ display: "inline-block", padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.1em", textTransform: "uppercase", borderRadius: 2, ...priorityStyle(row.priority) }}>
                                        {row.priority}
                                    </span>
                                </td>
                                <td style={{ padding: "0.65rem 0.35rem", textAlign: "right", color: "var(--text-primary)" }}>{row.today_count ?? "--"}</td>
                                <td style={{ padding: "0.65rem 0.35rem", textAlign: "right", color: "var(--text-primary)" }}>{row.today_flights != null ? row.today_flights : "--"}</td>
                                <td style={{ padding: "0.65rem 0.35rem", textAlign: "right", color: "var(--text-primary)" }}>{row.baseline != null && row.baseline > 0 ? row.baseline.toFixed(1) : "--"}</td>
                                <td style={{ padding: "0.65rem 0.35rem", textAlign: "right" }}>
                                    <span className="mono" style={{ display: "inline-block", textTransform: "uppercase", ...statusStyle(row.status) }}>
                                        {row.status}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}
