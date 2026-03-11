"use client";

import React from "react";
import type { MainSection } from "@/lib/types";
import { Navbar } from "./Navbar";
import { DashboardShell } from "./DashboardShell";

type CapabilityPageProps = {
    section: MainSection;
};

const PAGE_META: Record<MainSection, { title: string; description: string }> = {
    "aircraft-intelligence": {
        title: "AIRCRAFT INTELLIGENCE",
        description: "Operational target detection pipeline for aircraft imagery with ONNX acceleration.",
    },
    "aircraft-detection": {
        title: "AIRCRAFT SURVEILLANCE",
        description: "Operational target detection pipeline for aircraft imagery with ONNX acceleration.",
    },
    "aircraft-classification": {
        title: "AIRCRAFT RECOGNITION",
        description: "Fine-grained airframe recognition with origin attribution and explainability overlays.",
    },
    "change-detection": {
        title: "CHANGE INTELLIGENCE",
        description: "Before/after satellite analysis for segmentation masks, semantic regions, and temporal insight.",
    },
    "metrics-dashboard": {
        title: "OPERATIONS DASHBOARD",
        description: "Inference telemetry, performance snapshots, and system output monitoring.",
    },
};

export const CapabilityPage: React.FC<CapabilityPageProps> = ({ section }) => {
    const meta = PAGE_META[section];

    return (
        <div className="app">
            <Navbar />
            <div className="home-body">
                <header className="capability-header">
                    <p className="capability-kicker mono">AETHER EYE</p>
                    <h1 className="capability-title">{meta.title}</h1>
                    <p className="capability-description">{meta.description}</p>
                </header>
                <DashboardShell initialSection={section} consoleMode={false} />
            </div>
        </div>
    );
};
