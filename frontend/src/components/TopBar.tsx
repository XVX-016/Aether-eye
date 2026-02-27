import React from "react";
import type { MainSection } from "../lib/types";
import { useCountry } from "../context/CountryContext";

type Props = {
  activeSection: MainSection;
  onRunAircraft: () => void;
  onRunChange: () => void;
  canRunAircraft: boolean;
  canRunChange: boolean;
  loadingAircraft: boolean;
  loadingChange: boolean;
  error: string | null;
};

const SECTION_LABEL: Record<MainSection, string> = {
  "aircraft-detection": "Aircraft Detection",
  "change-detection": "Change Detection",
  "aircraft-classification": "Aircraft Classification",
  "metrics-dashboard": "Metrics Dashboard",
};

export const Topbar: React.FC<Props> = ({
  activeSection,
  onRunAircraft,
  onRunChange,
  canRunAircraft,
  canRunChange,
  loadingAircraft,
  loadingChange,
  error,
}) => {
  const { country, setCountry, options } = useCountry();
  const busy = loadingAircraft || loadingChange;
  const systemLabel = error ? "DEGRADED" : busy ? "ACTIVE" : "IDLE";
  const systemClass = error
    ? "status-pill status-pill-error"
    : busy
      ? "status-pill status-pill-warn"
      : "status-pill";

  return (
    <header className="topbar">
      <div className="topbar-left">
        <div className="topbar-kicker">AETHER EYE</div>
        <div className="topbar-title">{SECTION_LABEL[activeSection]}</div>
        <div className="topbar-subtitle">Aerospace intelligence operations surface</div>
      </div>

      <div className="topbar-center">
        <div className="status-stack">
          <div className="status-block">
            <span className="status-label">SYSTEM</span>
            <span className={systemClass}>{systemLabel}</span>
          </div>
          <div className="status-block">
            <span className="status-label">BACKEND</span>
            <span className="status-pill">API /api/v1</span>
          </div>
          <div className="status-block">
            <span className="status-label">ACCEL</span>
            <span className="status-pill">AUTO (CPU/GPU)</span>
          </div>
        </div>
      </div>

      <div className="topbar-right">
        <label className="country-select">
          <span className="country-label">COUNTRY</span>
          <select
            value={country}
            onChange={(e) => setCountry(e.target.value as typeof country)}
            aria-label="Select country"
          >
            {options.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </label>
        <button
          className="btn btn-primary"
          type="button"
          onClick={onRunAircraft}
          disabled={!canRunAircraft || loadingAircraft}
        >
          {loadingAircraft ? "DETECTING..." : "RUN DETECTION"}
        </button>
        <button
          className="btn btn-outline"
          type="button"
          onClick={onRunChange}
          disabled={!canRunChange || loadingChange}
        >
          {loadingChange ? "ANALYZING..." : "RUN CHANGE"}
        </button>
      </div>
    </header>
  );
};
