"use client";

import React from "react";
import type { MainSection } from "@/lib/types";
import { Navbar } from "./Navbar";
import { DashboardShell } from "./DashboardShell";

type CapabilityPageProps = {
    section: MainSection;
};

export const CapabilityPage: React.FC<CapabilityPageProps> = ({ section }) => {
    return (
        <div className="app">
            <Navbar />
            <div className="home-body">
                <DashboardShell initialSection={section} consoleMode={false} />
            </div>
        </div>
    );
};

