import React from "react";

export const PhilosophySection: React.FC = () => {
  return (
    <section className="home-philosophy">
      <h2 className="home-section-title mono">Engineering Philosophy</h2>
      <p className="home-philosophy-copy">
        Aether Eye prioritizes deterministic inference pipelines, auditable model output, and explainable intelligence
        products for aerospace and geospatial monitoring workflows.
      </p>
      <div className="home-philosophy-tags mono">
        <span>Deterministic Core</span>
        <span>Modular Pipelines</span>
        <span>Operational Explainability</span>
      </div>
    </section>
  );
};
