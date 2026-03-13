"use client";

import { useEffect, useMemo, useState } from "react";

type StatCard = {
    key: string;
    label: string;
    accent: string;
    value: string;
    loading: boolean;
};

const STATIC_AOI_COUNT = 1;
const POLL_INTERVAL_MS = 30_000;
const API_BASE = "http://localhost:8000/api";

export function StatCards() {
    const [counts, setCounts] = useState<Record<string, { value: number | null; loading: boolean }>>({
        scenes: { value: null, loading: true },
        detections: { value: null, loading: true },
        alerts: { value: null, loading: true },
    });

    useEffect(() => {
        let alive = true;

        const fetchCount = async (path: string) => {
            const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
            if (!response.ok) {
                throw new Error(`Failed to fetch ${path}`);
            }
            const payload = (await response.json()) as { count?: number };
            return payload.count ?? 0;
        };

        const load = async () => {
            if (alive) {
                setCounts((current) => ({
                    scenes: { value: current.scenes.value, loading: true },
                    detections: { value: current.detections.value, loading: true },
                    alerts: { value: current.alerts.value, loading: true },
                }));
            }

            try {
                const [scenes, detections, alerts] = await Promise.all([
                    fetchCount("/scenes/count"),
                    fetchCount("/detections/count"),
                    fetchCount("/alerts/count"),
                ]);
                if (!alive) {
                    return;
                }
                setCounts({
                    scenes: { value: scenes, loading: false },
                    detections: { value: detections, loading: false },
                    alerts: { value: alerts, loading: false },
                });
            } catch {
                if (!alive) {
                    return;
                }
                setCounts({
                    scenes: { value: null, loading: false },
                    detections: { value: null, loading: false },
                    alerts: { value: null, loading: false },
                });
            }
        };

        void load();
        const timer = window.setInterval(() => void load(), POLL_INTERVAL_MS);
        return () => {
            alive = false;
            window.clearInterval(timer);
        };
    }, []);

    const cards = useMemo<StatCard[]>(
        () => [
            {
                key: "scenes",
                label: "Scenes Ingested",
                accent: "var(--ops-accent-scenes)",
                value: counts.scenes.value != null ? counts.scenes.value.toLocaleString() : "--",
                loading: counts.scenes.loading,
            },
            {
                key: "detections",
                label: "Detections",
                accent: "var(--ops-accent-detections)",
                value: counts.detections.value != null ? counts.detections.value.toLocaleString() : "--",
                loading: counts.detections.loading,
            },
            {
                key: "alerts",
                label: "Active Alerts",
                accent: "var(--ops-accent-alerts)",
                value: counts.alerts.value != null ? counts.alerts.value.toLocaleString() : "--",
                loading: counts.alerts.loading,
            },
            {
                key: "aois",
                label: "AOIs Monitored",
                accent: "var(--ops-accent-aois)",
                value: STATIC_AOI_COUNT.toLocaleString(),
                loading: false,
            },
        ],
        [counts],
    );

    return (
        <div
            style={{
                display: "flex",
                width: "100%",
                gap: "1rem",
                alignItems: "stretch",
                flexWrap: "wrap",
            }}
        >
            {cards.map((card) => (
                <div
                    key={card.key}
                    className="glass-panel"
                    style={{
                        flex: "1 1 220px",
                        minWidth: 0,
                        padding: "1.15rem 1.2rem",
                        borderLeft: `4px solid ${card.accent}`,
                        background: "rgba(255,255,255,0.02)",
                    }}
                >
                    <div
                        className={`mono ${card.loading ? "ops-stat-pulse" : ""}`}
                        style={{
                            fontSize: "2.5rem",
                            fontWeight: 700,
                            color: "var(--text-primary)",
                            letterSpacing: "0.08em",
                            lineHeight: 1,
                        }}
                    >
                        {card.value}
                    </div>
                    <div
                        className="mono"
                        style={{
                            marginTop: "0.45rem",
                            fontSize: "0.65rem",
                            textTransform: "uppercase",
                            color: "var(--text-muted)",
                            letterSpacing: "0.22em",
                        }}
                    >
                        {card.label}
                    </div>
                </div>
            ))}
        </div>
    );
}
