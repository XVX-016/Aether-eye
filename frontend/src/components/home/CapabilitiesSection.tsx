import React from "react";

type Capability = {
  title: string;
  description: string;
  points: string[];
};

const CAPABILITIES: Capability[] = [
  {
    title: "Aircraft Detection",
    description: "ONNX-based tactical aircraft detection from still imagery.",
    points: [
      "YOLOv8 inference runtime",
      "Confidence and bbox extraction",
      "Operational metrics tracking",
      "CPU/GPU auto acceleration",
    ],
  },
  {
    title: "Aircraft Classification",
    description: "ViT class prediction with explainability and origin metadata.",
    points: [
      "Class + confidence output",
      "Origin country attribution",
      "Friend/Foe backend reasoning",
      "Grad-CAM heatmap overlay",
    ],
  },
  {
    title: "Satellite Change Intelligence",
    description: "Binary and semantic change analysis across temporal captures.",
    points: [
      "Before/after change score",
      "Mask overlay with opacity control",
      "Semantic region extraction",
      "Construction/track/terrain labels",
    ],
  },
];

export const CapabilitiesSection: React.FC = () => {
  return (
    <section className="home-capabilities">
      <h2 className="home-section-title mono">System Capabilities</h2>
      <div className="home-cap-grid">
        {CAPABILITIES.map((cap) => (
          <article className="home-cap-card" key={cap.title}>
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
