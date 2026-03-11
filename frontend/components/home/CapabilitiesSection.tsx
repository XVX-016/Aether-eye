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
      "YOLOV8 INFERENCE",
      "BOUNDING BOX EXTRACTION",
      "VIT CLASS PREDICTION",
      "ORIGIN COUNTRY ATTRIBUTION",
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
      "LIVE CONFIDENCE PANELS",
      "SYSTEM METRICS",
      "MODEL/DEVICE VISIBILITY",
      "OPS-READY OUTPUT SUMMARIES",
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
