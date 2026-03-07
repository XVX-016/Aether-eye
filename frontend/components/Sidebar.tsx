"use client";

import React from "react";
import type { MainSection } from "@/lib/types";

type Props = {
    active: MainSection;
    onChange: (section: MainSection) => void;
};

const NAV_ITEMS: { key: MainSection; label: string; short: string }[] = [
    { key: "aircraft-intelligence", label: "Aircraft Intelligence", short: "Air Intel" },
    { key: "change-detection", label: "Change Intelligence", short: "Change Intel" },
    { key: "metrics-dashboard", label: "Operations Dashboard", short: "Operations" },
];

export const Sidebar: React.FC<Props> = ({ active, onChange }) => {
    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <div className="sidebar-brand-mark">AE</div>
                <div className="sidebar-brand-text">
                    <div className="sidebar-brand-title">Aether Eye</div>
                    <div className="sidebar-brand-subtitle">INTELLIGENCE SYSTEM</div>
                </div>
            </div>

            <nav className="sidebar-nav" aria-label="Primary navigation">
                {NAV_ITEMS.map((item) => {
                    const isActive = item.key === active;
                    return (
                        <button
                            key={item.key}
                            type="button"
                            className={isActive ? "sidebar-link sidebar-link-active" : "sidebar-link"}
                            onClick={() => onChange(item.key)}
                        >
                            <span className="sidebar-link-dot" aria-hidden="true" />
                            <span className="sidebar-link-text">
                                <span className="sidebar-link-main">{item.label}</span>
                                <span className="sidebar-link-sub">{item.short}</span>
                            </span>
                        </button>
                    );
                })}
            </nav>
        </aside>
    );
};
