"use client";

import Link from "next/link";
import React from "react";

export const AetherHero = () => {
  return (
    <section className="hero-section">
      <div className="hero-content">
        <p className="hero-kicker mono">AETHER EYE INTELLIGENCE SYSTEM</p>

        <div className="hero-typography">
          <h1 className="hero-main-title">
            <span className="hero-word-bold">Intelligence</span>
            <span className="hero-word-bold">Driven</span>
          </h1>
          <h2 className="hero-sub-title hero-outline-title">Detection &amp; Analysis</h2>
        </div>

        <p className="hero-support mono">AETHER EYE INTELLIGENCE OPERATIONS SURFACE</p>

        <div className="hero-actions">
          <Link href="/aircraft-intelligence" className="hero-cta glass">
            ENTER CONSOLE
          </Link>
        </div>
      </div>

      <div className="hero-fade" />
    </section>
  );
};
