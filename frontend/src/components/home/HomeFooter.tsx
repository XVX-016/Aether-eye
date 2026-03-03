import React from "react";

export const HomeFooter: React.FC = () => {
  return (
    <footer className="home-footer">
      <div className="home-footer-grid">
        <div>
          <h3 className="home-footer-brand mono">AETHER-EYE</h3>
          <p className="home-footer-copy">
            Multi-model intelligence platform for aircraft detection, geopolitical classification, and satellite change
            analytics.
          </p>
        </div>
        <div>
          <h4 className="home-footer-head mono">Platform</h4>
          <ul className="home-footer-links">
            <li>Detection Console</li>
            <li>Classification Pipeline</li>
            <li>Change Intelligence</li>
          </ul>
        </div>
        <div>
          <h4 className="home-footer-head mono">Resources</h4>
          <ul className="home-footer-links">
            <li>
              <a href="https://github.com/XVX-016/Aether-eye" target="_blank" rel="noreferrer">
                GitHub Repository
              </a>
            </li>
            <li>API Documentation</li>
            <li>Model Registry</li>
          </ul>
        </div>
        <div>
          <h4 className="home-footer-head mono">Contact</h4>
          <ul className="home-footer-links">
            <li>ops@aether-eye.local</li>
            <li>Collaboration Inquiries</li>
          </ul>
        </div>
      </div>
      <div className="home-footer-bottom mono">
        <span>(c) 2026 AETHER-EYE. ALL RIGHTS RESERVED.</span>
        <span>BUILT FOR AEROSPACE & GEOSPATIAL INTELLIGENCE</span>
      </div>
    </footer>
  );
};
