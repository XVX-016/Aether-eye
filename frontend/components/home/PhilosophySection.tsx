"use client";

import React from "react";

export const PhilosophySection: React.FC = () => {
    return (
        <section className="home-philosophy">
            <h2 className="home-section-title mono">ENGINEERING PHILOSOPHY</h2>
            <p className="home-philosophy-copy">
                This system is built on deterministic modeling principles, emphasizing traceable state propagation,
                mathematically consistent linearization, and physically meaningful control synthesis.
                It rejects "black box" behavior in favor of transparent, equation-based validation.
            </p>
            <div className="home-philosophy-tags mono">
                <span>DETERMINISTIC CORE</span>
                <span className="home-philosophy-dot">*</span>
                <span>MODULAR DATABASE</span>
                <span className="home-philosophy-dot">*</span>
                <span>ACADEMIC VALIDATION MODE</span>
            </div>
        </section>
    );
};
