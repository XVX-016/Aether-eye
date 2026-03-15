"use client";

import { useEffect, useState } from "react";

import { fetchSiteStatus } from "@/lib/api";
import type { SiteStatus } from "@/types/operations";

const POLL_INTERVAL_MS = 60_000;

function formatTitle(value: string) {
    return value
        .split("_")
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function statusStyle(status: string) {
    if (status === "anomalous") {
        return { background: "rgba(239, 68, 68, 0.18)", color: "#fca5a5", border: "1px solid rgba(239, 68, 68, 0.45)" };
    }
    if (status === "elevated") {
        return { background: "rgba(245, 158, 11, 0.18)", color: "#fcd34d", border: "1px solid rgba(245, 158, 11, 0.45)" };
    }
    return { background: "rgba(34, 197, 94, 0.18)", color: "#86efac", border: "1px solid rgba(34, 197, 94, 0.45)" };
}

export function SiteStatusPanel() {
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
        <section className="glass-panel" style={{ padding: "1rem 1.1rem", marginTop: "1rem" }}>
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
                            <th style={{ textAlign: "right", padding: "0.45rem 0.35rem" }}>Baseline</th>
                            <th style={{ textAlign: "right", padding: "0.45rem 0.35rem" }}>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            <tr>
                                <td colSpan={6} style={{ padding: "0.85rem 0.35rem", color: "var(--text-muted)" }}>Loading site status...</td>
                            </tr>
                        ) : rows.length === 0 ? (
                            <tr>
                                <td colSpan={6} style={{ padding: "0.85rem 0.35rem", color: "var(--text-muted)" }}>No site status available.</td>
                            </tr>
                        ) : rows.map((row) => (
                            <tr key={row.id} style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                                <td style={{ padding: "0.65rem 0.35rem", color: "var(--text-primary)", fontWeight: 600 }}>{row.name}</td>
                                <td style={{ padding: "0.65rem 0.35rem", color: "var(--text-secondary)" }}>{formatTitle(row.type)}</td>
                                <td style={{ padding: "0.65rem 0.35rem", color: "var(--text-secondary)" }}>{formatTitle(row.priority)}</td>
                                <td style={{ padding: "0.65rem 0.35rem", textAlign: "right", color: "var(--text-primary)" }}>{row.today_count ?? "--"}</td>
                                <td style={{ padding: "0.65rem 0.35rem", textAlign: "right", color: "var(--text-primary)" }}>{row.baseline != null && row.baseline > 0 ? row.baseline.toFixed(1) : "--"}</td>
                                <td style={{ padding: "0.65rem 0.35rem", textAlign: "right" }}>
                                    <span className="mono" style={{ display: "inline-block", padding: "0.18rem 0.55rem", borderRadius: 999, fontSize: "0.65rem", letterSpacing: "0.14em", textTransform: "uppercase", ...statusStyle(row.status) }}>
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
