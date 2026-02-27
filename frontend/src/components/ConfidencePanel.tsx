import React, { useMemo } from "react";
import type { AircraftDetection } from "../lib/types";
import { ContentPanel } from "./ContentPanel";

type Props = {
  detections: AircraftDetection[];
  changeScore: number | null;
  loadingAircraft?: boolean;
  loadingChange?: boolean;
  error?: string | null;
};

export const ConfidencePanel: React.FC<Props> = ({
  detections,
  changeScore,
  loadingAircraft = false,
  loadingChange = false,
  error,
}) => {
  const sorted = useMemo(
    () => [...detections].sort((a, b) => b.confidence - a.confidence),
    [detections],
  );

  const top = sorted.slice(0, 8);
  const maxConf = sorted.length ? sorted[0].confidence : null;

  return (
    <ContentPanel title="Confidence Panel" subtitle="Tactical summary of model outputs.">
      {error && <div className="alert alert-error">{error}</div>}

      <div className="kpis">
        <div className="kpi">
          <div className="kpi-label">DETECTIONS</div>
          <div className="kpi-value">{loadingAircraft ? "..." : String(detections.length)}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">TOP CONF</div>
          <div className="kpi-value">
            {loadingAircraft ? "..." : maxConf == null ? "--" : `${(maxConf * 100).toFixed(1)}%`}
          </div>
        </div>
        <div className="kpi">
          <div className="kpi-label">CHANGE SCORE</div>
          <div className="kpi-value">
            {loadingChange ? "..." : changeScore == null ? "--" : changeScore.toFixed(4)}
          </div>
        </div>
      </div>

      <div className="section-title">Top detections</div>
      {top.length === 0 ? (
        <div className="empty-state small">
          {loadingAircraft ? "Running detection..." : "No detections yet."}
        </div>
      ) : (
        <div className="table">
          <div className="table-row table-head">
            <div>Class</div>
            <div>Confidence</div>
            <div>Box</div>
          </div>
          {top.map((d, i) => (
            <div key={i} className="table-row">
              <div className="mono">AIRCRAFT ({d.class_id})</div>
              <div className="mono">{(d.confidence * 100).toFixed(1)}%</div>
              <div className="mono">
                {Math.round(d.bbox.x1)},{Math.round(d.bbox.y1)} to {Math.round(d.bbox.x2)},
                {Math.round(d.bbox.y2)}
              </div>
            </div>
          ))}
        </div>
      )}
    </ContentPanel>
  );
};
