"use client";

import React from "react";

type Capability = {
  title: string;
  description: string;
  points: string[];
};

const CAPABILITIES: Capability[] = [
  {
    title: "AIRCRAFT SURVEILLANCE",
    description: "ONNX-powered target detection and tracking-ready aircraft surveillance.",
    points: [
      "YOLOV8 INFERENCE",
      "BOUNDING BOX EXTRACTION",
      "CONFIDENCE AGGREGATION",
      "CPU/GPU AUTO ACCELERATION",
    ],
  },
  {
    title: "AIRCRAFT RECOGNITION",
    description: "Fine-grained airframe recognition with origin metadata and explainability.",
    points: [
      "VIT CLASS PREDICTION",
      "ORIGIN COUNTRY ATTRIBUTION",
      "FRIEND/FOE BACKEND LOGIC",
      "GRAD-CAM VISUALIZATION",
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
