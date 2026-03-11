"use client";

import React from "react";
import { Navbar } from "./Navbar";
import { AetherHero } from "./AetherHero";
import { CapabilitiesSection } from "./home/CapabilitiesSection";
import { PhilosophySection } from "./home/PhilosophySection";
import { HomeFooter } from "./home/HomeFooter";

export const HomePage: React.FC = () => {
    return (
        <div className="app">
            <Navbar />
            <AetherHero />
            <div className="home-body">
                <CapabilitiesSection />
                <PhilosophySection />
                <HomeFooter />
            </div>
        </div>
    );
};
