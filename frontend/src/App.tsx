import React, { useEffect, useMemo, useState } from "react";
import { BeforeAfterSlider } from "./components/BeforeAfterSlider";
import { ConfidencePanel } from "./components/ConfidencePanel";
import { DetectionCanvas } from "./components/DetectionCanvas";
import { ImageUploadPanel } from "./components/ImageUploadPanel";
import { MainLayout } from "./components/MainLayout";
import { SystemMetrics } from "./components/SystemMetrics";
import { ContentPanel } from "./components/ContentPanel";
import { runAircraftDetection, runChangeDetection } from "./lib/api";
import type { AircraftDetection, MainSection, SystemMetricsSnapshot } from "./lib/types";
import { useCountry } from "./context/CountryContext";

export const App: React.FC = () => {
  const { country } = useCountry();
  const [activeSection, setActiveSection] = useState<MainSection>("aircraft-detection");
  const [aircraftFile, setAircraftFile] = useState<File | null>(null);
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);

  const [aircraftDetections, setAircraftDetections] = useState<AircraftDetection[]>([]);
  const [changeScore, setChangeScore] = useState<number | null>(null);

  const [loadingAircraft, setLoadingAircraft] = useState(false);
  const [loadingChange, setLoadingChange] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetricsSnapshot | null>(null);

  const aircraftUrl = useMemo(
    () => (aircraftFile ? URL.createObjectURL(aircraftFile) : null),
    [aircraftFile],
  );
  const beforeUrl = useMemo(
    () => (beforeFile ? URL.createObjectURL(beforeFile) : null),
    [beforeFile],
  );
  const afterUrl = useMemo(
    () => (afterFile ? URL.createObjectURL(afterFile) : null),
    [afterFile],
  );

  useEffect(() => {
    return () => {
      if (aircraftUrl) URL.revokeObjectURL(aircraftUrl);
      if (beforeUrl) URL.revokeObjectURL(beforeUrl);
      if (afterUrl) URL.revokeObjectURL(afterUrl);
    };
  }, [aircraftUrl, beforeUrl, afterUrl]);

  const runAircraft = async () => {
    if (!aircraftFile) return;
    setError(null);
    setLoadingAircraft(true);
    try {
      const res = await runAircraftDetection(aircraftFile, country);
      setAircraftDetections(res.detections ?? []);

      const bestConf =
        res.detections && res.detections.length
          ? res.detections.reduce(
              (max, d) => (d.confidence > max ? d.confidence : max),
              res.detections[0].confidence,
            )
          : null;

      setSystemMetrics({
        inference_time_ms: res.inference_time_ms ?? undefined,
        model_name: res.model_name ?? undefined,
        device_used: res.device_used ?? undefined,
        confidence: bestConf,
      });
    } catch (err: any) {
      setAircraftDetections([]);
      setError(err?.response?.data?.detail ?? err?.message ?? "Aircraft detection failed.");
    } finally {
      setLoadingAircraft(false);
    }
  };

  const runChange = async () => {
    if (!beforeFile || !afterFile) return;
    setError(null);
    setLoadingChange(true);
    try {
      const res = await runChangeDetection(beforeFile, afterFile, false, country);
      setChangeScore(res.change_score);
      setSystemMetrics({
        inference_time_ms: res.inference_time_ms ?? undefined,
        model_name: res.model_name ?? undefined,
        device_used: res.device_used ?? undefined,
        confidence: res.change_score,
      });
    } catch (err: any) {
      setChangeScore(null);
      setError(err?.response?.data?.detail ?? err?.message ?? "Change detection failed.");
    } finally {
      setLoadingChange(false);
    }
  };

  return (
    <div className="app">
      <MainLayout
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        onRunAircraft={runAircraft}
        onRunChange={runChange}
        canRunAircraft={!!aircraftFile}
        canRunChange={!!beforeFile && !!afterFile}
        loadingAircraft={loadingAircraft}
        loadingChange={loadingChange}
        error={error}
      >
        <div className="grid">
          {activeSection === "aircraft-detection" && (
            <>
              <div className="col col-left">
                <ImageUploadPanel
                  label="Upload: detection image"
                  helpText="Used for aircraft detection (YOLOv8 ONNX)."
                  file={aircraftFile}
                  onChange={(f) => {
                    setAircraftFile(f);
                    setAircraftDetections([]);
                  }}
                />

                <DetectionCanvas
                  imageUrl={aircraftUrl}
                  detections={aircraftDetections}
                  loading={loadingAircraft}
                />
              </div>

              <div className="col col-right">
                <div className="stack">
                  <ConfidencePanel
                    detections={aircraftDetections}
                    changeScore={null}
                    loadingAircraft={loadingAircraft}
                    loadingChange={false}
                    error={error}
                  />
                  <SystemMetrics metrics={systemMetrics} />
                </div>
              </div>
            </>
          )}

          {activeSection === "change-detection" && (
            <>
              <div className="col col-left">
                <div className="stack">
                  <ImageUploadPanel
                    label="Upload: before image"
                    helpText="Aligned capture at (t0)."
                    file={beforeFile}
                    onChange={(f) => {
                      setBeforeFile(f);
                      setChangeScore(null);
                    }}
                  />
                  <ImageUploadPanel
                    label="Upload: after image"
                    helpText="Aligned capture at (t1)."
                    file={afterFile}
                    onChange={(f) => {
                      setAfterFile(f);
                      setChangeScore(null);
                    }}
                  />
                </div>

                <BeforeAfterSlider beforeUrl={beforeUrl} afterUrl={afterUrl} />
              </div>

              <div className="col col-right">
                <div className="stack">
                  <ConfidencePanel
                    detections={[]}
                    changeScore={changeScore}
                    loadingAircraft={false}
                    loadingChange={loadingChange}
                    error={error}
                  />
                  <SystemMetrics metrics={systemMetrics} />
                </div>
              </div>
            </>
          )}

          {activeSection === "aircraft-classification" && (
            <>
              <div className="col col-left">
                <ContentPanel
                  title="Aircraft Classification"
                  subtitle="ViT-based FGVC Aircraft classification (UI wiring placeholder)."
                >
                  <div className="empty-state small">
                    Backend endpoints are available; hook Grad-CAM and class outputs here as needed.
                  </div>
                </ContentPanel>
              </div>

              <div className="col col-right">
                <ContentPanel
                  title="Classification Metrics"
                  subtitle="Reserved panel for top-1/top-5 and variant labels."
                >
                  <div className="empty-state small">No classification runs yet.</div>
                </ContentPanel>
              </div>
            </>
          )}

          {activeSection === "metrics-dashboard" && (
            <>
              <div className="col col-left">
                <div className="stack">
                  <ConfidencePanel
                    detections={aircraftDetections}
                    changeScore={changeScore}
                    loadingAircraft={loadingAircraft}
                    loadingChange={loadingChange}
                    error={error}
                  />
                  <SystemMetrics metrics={systemMetrics} />
                </div>
              </div>

              <div className="col col-right">
                <ContentPanel title="Ops Notes" subtitle="Quick operational guidance">
                  <ul className="notes">
                    <li>Keep before/after images aligned for best change results.</li>
                    <li>Confidence is class-score based; tune thresholds server-side.</li>
                    <li>Use the overlay to validate boxes before downstream actions.</li>
                  </ul>
                </ContentPanel>
              </div>
            </>
          )}
        </div>
      </MainLayout>
    </div>
  );
};

