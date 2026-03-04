"use client";

import React from "react";
import type { SystemMetricsSnapshot } from "@/lib/types";
import { ContentPanel } from "./ContentPanel";

type Props = {
    metrics: SystemMetricsSnapshot | null;
};

export const SystemMetrics: React.FC<Props> = ({ metrics }) => {
    if (!metrics) {
        return (
            <ContentPanel title="System Metrics" subtitle="No inference runs yet.">
                <div className="empty-state small">Trigger an inference to populate metrics.</div>
            </ContentPanel>
        );
    }

    const { inference_time_ms, model_name, device_used, confidence } = metrics;

    return (
        <ContentPanel title="System Metrics" subtitle="Last inference run summary.">
            <div className="kpis kpis-compact">
                <div className="kpi">
                    <div className="kpi-label">INFERENCE TIME</div>
                    <div className="kpi-value mono">
                        {inference_time_ms == null ? "--" : `${inference_time_ms.toFixed(1)} ms`}
                    </div>
                </div>
                <div className="kpi">
                    <div className="kpi-label">MODEL</div>
                    <div className="kpi-value mono">{model_name && model_name.length > 0 ? model_name : "UNKNOWN"}</div>
                </div>
                <div className="kpi">
                    <div className="kpi-label">DEVICE</div>
                    <div className="kpi-value mono">
                        {device_used && device_used.length > 0 ? device_used.toUpperCase() : "N/A"}
                    </div>
                </div>
                <div className="kpi">
                    <div className="kpi-label">CONFIDENCE</div>
                    <div className="kpi-value mono">
                        {confidence == null ? "--" : `${(confidence * 100).toFixed(1)}%`}
                    </div>
                </div>
            </div>
        </ContentPanel>
    );
};
