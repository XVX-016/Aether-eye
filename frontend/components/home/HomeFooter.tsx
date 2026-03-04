"use client";

import React from "react";

export const HomeFooter: React.FC = () => {
    return (
        <footer className="home-footer">
            <div className="home-footer-grid">
                <div className="home-footer-brand-wrap">
                    <h3 className="home-footer-brand mono">AETHER EYE</h3>
                    <p className="home-footer-copy">
                        Advanced intelligence platform for aircraft detection, classification,
                        and satellite change monitoring.
                    </p>
                </div>

                <div className="home-footer-section">
                    <h4 className="home-footer-head mono">PLATFORM</h4>
                    <ul className="home-footer-links">
                        <li>Detection Console</li>
                        <li>Classification Pipeline</li>
                        <li>Change Intelligence</li>
                    </ul>
                </div>

                <div className="home-footer-section">
                    <h4 className="home-footer-head mono">RESOURCES</h4>
                    <ul className="home-footer-links">
                        <li>
                            <a
                                href="https://github.com/XVX-016/Aether-eye"
                                target="_blank"
                                rel="noreferrer"
                            >
                                GitHub Repository
                            </a>
                        </li>
                        <li>API Documentation</li>
                        <li>Model Registry</li>
                    </ul>
                </div>

                <div className="home-footer-section">
                    <h4 className="home-footer-head mono">CONTACT</h4>
                    <ul className="home-footer-links">
                        <li>
                            <a href="mailto:xvx016xc@gmail.com">xvx016xc@gmail.com</a>
                        </li>
                        <li>Collaboration Inquiries</li>
                    </ul>
                </div>
            </div>

            <div className="home-footer-bottom mono">
                <div className="home-footer-copyright">(c) 2026 AETHER EYE. ALL RIGHTS RESERVED.</div>
                <div className="home-footer-motto">BUILT FOR AEROSPACE INTELLIGENCE OPERATIONS</div>
            </div>
        </footer>
    );
};
