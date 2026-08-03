"use client";

import Link from "next/link";
import React from "react";

export const AetherHero: React.FC = () => {
  return (
    <section className="hero-section">
      <div className="hero-streaks" aria-hidden="true">
        {Array.from({ length: 14 }).map((_, idx) => (
          <span key={idx} className={`hero-streak hero-streak-${(idx % 6) + 1}`} />
        ))}
      </div>

      <div className="hero-content">
        <p className="hero-kicker mono">AETHER EYE INTELLIGENCE SYSTEM</p>

        <div className="hero-typography">
          <h1 className="hero-main-title">
            <span className="hero-word-bold">AETHER EYE</span>
          </h1>
          <h2 className="hero-sub-title">INTELLIGENCE DRIVEN</h2>
          <p className="hero-support mono" style={{ marginTop: "1rem", marginBottom: "3rem" }}>
            DETECTION &amp; ANALYSIS
          </p>
        </div>

        <div className="hero-actions">
          <Link href="/operations" className="hero-cta glass">
            ENTER CONSOLE
          </Link>
        </div>
      </div>

      <div className="hero-fade" />
    </section>
  );
};
