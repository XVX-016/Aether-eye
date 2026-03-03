import React from "react";

const ITEMS = ["HOME", "DETECTION", "CLASSIFICATION", "CHANGE", "METRICS"];

export const HomepageNav: React.FC = () => {
  return (
    <div className="home-nav">
      <div className="home-nav-brand mono">AETHER EYE</div>
      <div className="home-nav-links">
        {ITEMS.map((item, idx) => (
          <span
            key={item}
            className={idx === 0 ? "home-nav-link home-nav-link-active mono" : "home-nav-link mono"}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
};
