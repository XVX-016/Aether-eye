"use client";

import React from "react";

type Capability = {
  title: string;
  description: string;
  points: string[];
};

const CAPABILITIES: Capability[] = [
  {
    title: "AIRCRAFT INTELLIGENCE",
    description: "Unified aircraft pipeline: ONNX target detection followed by fine-grained recognition.",
    points: [
      "CONVNEXT SMALL CLASSIFIER",
      "100 AIRCRAFT CLASSES",
      "GRAD-CAM EXPLAINABILITY",
      "FRIEND/FOE ATTRIBUTION",
      "CPU/GPU AUTO ACCELERATION",
    ],
  },
  {
    title: "CHANGE INTELLIGENCE",
    description: "Before/after satellite change analysis with semantic region labeling.",
    points: [
      "BINARY CHANGE MASK",
      "CONSTRUCTION/TRACK/TERRAIN",
      "MASK OVERLAY CONTROLS",
      "ONNX CUDA INFERENCE",
    ],
  },
  {
    title: "OPERATIONS DASHBOARD",
    description: "Central telemetry for inference runs, latency snapshots, and model health.",
    points: [
      "18 GLOBALLY MONITORED SITES",
      "SENTINEL-2 CHANGE DETECTION",
      "LIVE ADS-B FLIGHT TRACKING",
      "OSINT INTELLIGENCE FEED",
      "TEMPORAL ANOMALY DETECTION",
    ],
  },
];

export const CapabilitiesSection: React.FC = () => {
  return (
    <section className="home-capabilities">
      <h2 className="home-section-title mono">SYSTEM CAPABILITIES</h2>
      <div className="home-cap-grid">
        {CAPABILITIES.map((cap) => (
          <article key={cap.title} className="home-cap-card">
            <h3 className="home-cap-title mono">{cap.title}</h3>
            <p className="home-cap-desc">{cap.description}</p>
            <ul className="home-cap-list">
              {cap.points.map((point) => (
                <li key={point} className="mono">
                  {point}
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
};
