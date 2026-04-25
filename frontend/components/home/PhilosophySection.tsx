"use client";

import React from "react";

export const PhilosophySection: React.FC = () => {
    return (
        <section className="home-philosophy">
            <h2 className="home-section-title mono">SYSTEM PHILOSOPHY</h2>
            <p className="home-philosophy-copy">
                Aether-Eye is built on verifiable intelligence principles.
                Every detection is traceable to a satellite scene. Every alert
                references a statistical baseline. Every classification produces
                an explainable attention map. No black-box outputs — only
                auditable, evidence-backed intelligence.
            </p>
            <div className="home-philosophy-tags mono">
                <span>OPEN DATA SOURCES</span>
                <span className="home-philosophy-dot">×</span>
                <span>AUDITABLE ML</span>
                <span className="home-philosophy-dot">×</span>
                <span>ON-PREMISES DEPLOYMENT</span>
            </div>
        </section>
    );
};
