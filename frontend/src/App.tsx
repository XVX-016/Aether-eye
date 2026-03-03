import React, { useEffect, useMemo, useState } from "react";
import { BeforeAfterSlider } from "./components/BeforeAfterSlider";
import { ConfidencePanel } from "./components/ConfidencePanel";
import { DetectionCanvas } from "./components/DetectionCanvas";
import { HeroSection } from "./components/hero/HeroSection";
import { CapabilitiesSection } from "./components/home/CapabilitiesSection";
import { HomeFooter } from "./components/home/HomeFooter";
import { PhilosophySection } from "./components/home/PhilosophySection";
import { ImageUploadPanel } from "./components/ImageUploadPanel";
import { MainLayout } from "./components/MainLayout";
import { SystemMetrics } from "./components/SystemMetrics";
import { ContentPanel } from "./components/ContentPanel";
import {
  runAircraftClassification,
  runAircraftDetection,
  runAircraftGradCam,
  runChangeDetection,
} from "./lib/api";
import type {
  AircraftClassificationResponse,
  AircraftDetection,
  MainSection,
  SystemMetricsSnapshot,
} from "./lib/types";
import { useCountry } from "./context/CountryContext";

export const App: React.FC = () => {
  const { country } = useCountry();
  const [activeSection, setActiveSection] = useState<MainSection>("aircraft-detection");
  const [aircraftFile, setAircraftFile] = useState<File | null>(null);
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);
  const [classificationFile, setClassificationFile] = useState<File | null>(null);

  const [aircraftDetections, setAircraftDetections] = useState<AircraftDetection[]>([]);
  const [changeScore, setChangeScore] = useState<number | null>(null);
  const [changeMaskBase64, setChangeMaskBase64] = useState<string | null>(null);
  const [semanticRegions, setSemanticRegions] = useState<
    { type: string; bbox: [number, number, number, number] }[]
  >([]);
  const [afterImageDims, setAfterImageDims] = useState<{ w: number; h: number } | null>(null);
  const [changeMaskOpacity, setChangeMaskOpacity] = useState<number>(45);
  const [classificationResult, setClassificationResult] = useState<AircraftClassificationResponse | null>(null);
  const [gradCamBase64, setGradCamBase64] = useState<string | null>(null);
  const [gradCamOpacity, setGradCamOpacity] = useState<number>(45);

  const [loadingAircraft, setLoadingAircraft] = useState(false);
  const [loadingChange, setLoadingChange] = useState(false);
  const [loadingClassification, setLoadingClassification] = useState(false);
  const [loadingGradCam, setLoadingGradCam] = useState(false);
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
  const classificationUrl = useMemo(
    () => (classificationFile ? URL.createObjectURL(classificationFile) : null),
    [classificationFile],
  );
  const changeMaskUrl = useMemo(
    () => (changeMaskBase64 ? `data:image/png;base64,${changeMaskBase64}` : null),
    [changeMaskBase64],
  );
  const gradCamUrl = useMemo(
    () => (gradCamBase64 ? `data:image/png;base64,${gradCamBase64}` : null),
    [gradCamBase64],
  );

  useEffect(() => {
    return () => {
      if (aircraftUrl) URL.revokeObjectURL(aircraftUrl);
      if (beforeUrl) URL.revokeObjectURL(beforeUrl);
      if (afterUrl) URL.revokeObjectURL(afterUrl);
      if (classificationUrl) URL.revokeObjectURL(classificationUrl);
    };
  }, [aircraftUrl, beforeUrl, afterUrl, classificationUrl]);

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
      const res = await runChangeDetection(beforeFile, afterFile, true, country, true);
      setChangeScore(res.change_score);
      setChangeMaskBase64(res.change_mask_base64 ?? null);
      setSemanticRegions(res.regions ?? []);
      setSystemMetrics({
        inference_time_ms: res.inference_time_ms ?? undefined,
        model_name: res.model_name ?? undefined,
        device_used: res.device_used ?? undefined,
        confidence: res.change_score,
      });
    } catch (err: any) {
      setChangeScore(null);
      setChangeMaskBase64(null);
      setSemanticRegions([]);
      setError(err?.response?.data?.detail ?? err?.message ?? "Change detection failed.");
    } finally {
      setLoadingChange(false);
    }
  };

  const runClassification = async () => {
    if (!classificationFile) return;
    setError(null);
    setLoadingClassification(true);
    try {
      const res = await runAircraftClassification(classificationFile, country);
      setClassificationResult(res);
      setSystemMetrics({
        inference_time_ms: res.inference_time_ms ?? undefined,
        model_name: res.model_name ?? undefined,
        device_used: res.device_used ?? undefined,
        confidence: res.confidence,
      });
    } catch (err: any) {
      setClassificationResult(null);
      setError(err?.response?.data?.detail ?? err?.message ?? "Aircraft classification failed.");
    } finally {
      setLoadingClassification(false);
    }
  };

  const runGradCam = async () => {
    if (!classificationFile) return;
    setError(null);
    setLoadingGradCam(true);
    try {
      const res = await runAircraftGradCam(classificationFile, country);
      setGradCamBase64(res.heatmap_base64_png ?? null);
    } catch (err: any) {
      setGradCamBase64(null);
      setError(err?.response?.data?.detail ?? err?.message ?? "Grad-CAM failed.");
    } finally {
      setLoadingGradCam(false);
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
        {activeSection === "aircraft-detection" && <HeroSection />}
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
                      setChangeMaskBase64(null);
                      setSemanticRegions([]);
                      setAfterImageDims(null);
                    }}
                  />
                  <ImageUploadPanel
                    label="Upload: after image"
                    helpText="Aligned capture at (t1)."
                    file={afterFile}
                    onChange={(f) => {
                      setAfterFile(f);
                      setChangeScore(null);
                      setChangeMaskBase64(null);
                      setSemanticRegions([]);
                      setAfterImageDims(null);
                    }}
                  />
                </div>

                <BeforeAfterSlider beforeUrl={beforeUrl} afterUrl={afterUrl} />
                <ContentPanel
                  title="Change Mask Overlay"
                  subtitle="Change mask rendered over the after image."
                >
                  {!afterUrl ? (
                    <div className="empty-state small">Upload and analyze images to view the mask overlay.</div>
                  ) : (
                    <div className="overlay-stage">
                      <img
                        className="overlay-base-image"
                        src={afterUrl}
                        alt="After image"
                        onLoad={(e) => {
                          const target = e.currentTarget;
                          setAfterImageDims({ w: target.naturalWidth, h: target.naturalHeight });
                        }}
                      />
                      {changeMaskUrl && (
                        <img
                          className="overlay-mask-image overlay-mask-red"
                          src={changeMaskUrl}
                          alt="Change mask overlay"
                          style={{ opacity: changeMaskOpacity / 100 }}
                        />
                      )}
                      {semanticRegions.map((region, idx) => {
                        const [x1, y1, x2, y2] = region.bbox;
                        const w = Math.max(1, x2 - x1);
                        const h = Math.max(1, y2 - y1);
                        const denomW = afterImageDims?.w ?? 1;
                        const denomH = afterImageDims?.h ?? 1;
                        const lineColor =
                          region.type === "Construction"
                            ? "#f59e0b"
                            : region.type === "Vehicle Track"
                              ? "#3b82f6"
                              : "#22c55e";
                        return (
                          <div
                            key={`${region.type}-${idx}`}
                            className="semantic-box"
                            style={{
                              left: `${(x1 / denomW) * 100}%`,
                              top: `${(y1 / denomH) * 100}%`,
                              width: `${(w / denomW) * 100}%`,
                              height: `${(h / denomH) * 100}%`,
                              borderColor: lineColor,
                            }}
                          >
                            <span className="semantic-label" style={{ background: lineColor }}>
                              {region.type}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  <div className="overlay-controls">
                    <label htmlFor="changeMaskOpacity">Mask opacity: {changeMaskOpacity}%</label>
                    <input
                      id="changeMaskOpacity"
                      type="range"
                      min={0}
                      max={100}
                      value={changeMaskOpacity}
                      onChange={(e) => setChangeMaskOpacity(Number(e.target.value))}
                    />
                  </div>
                  <div className="semantic-legend">
                    <div className="legend-item">
                      <span className="legend-swatch legend-construction" />
                      <span>Construction</span>
                    </div>
                    <div className="legend-item">
                      <span className="legend-swatch legend-track" />
                      <span>Vehicle Track</span>
                    </div>
                    <div className="legend-item">
                      <span className="legend-swatch legend-terrain" />
                      <span>Terrain Change</span>
                    </div>
                  </div>
                </ContentPanel>
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
                  subtitle="Upload image, run classification, then generate Grad-CAM."
                >
                  <div className="stack stack-single">
                    <ImageUploadPanel
                      label="Upload: classification image"
                      helpText="Used for ViT aircraft classification."
                      file={classificationFile}
                      onChange={(f) => {
                        setClassificationFile(f);
                        setClassificationResult(null);
                        setGradCamBase64(null);
                      }}
                    />

                    <div className="button-row">
                      <button
                        className="btn btn-primary"
                        type="button"
                        onClick={runClassification}
                        disabled={!classificationFile || loadingClassification}
                      >
                        {loadingClassification ? "CLASSIFYING..." : "RUN CLASSIFICATION"}
                      </button>
                      <button
                        className="btn btn-outline"
                        type="button"
                        onClick={runGradCam}
                        disabled={!classificationFile || loadingGradCam}
                      >
                        {loadingGradCam ? "GENERATING..." : "SHOW GRAD-CAM"}
                      </button>
                    </div>

                    {!classificationUrl ? (
                      <div className="empty-state small">Upload an image to preview classification and Grad-CAM.</div>
                    ) : (
                      <div className="overlay-stage">
                        <img className="overlay-base-image" src={classificationUrl} alt="Classification input" />
                        {gradCamUrl && (
                          <img
                            className="overlay-mask-image"
                            src={gradCamUrl}
                            alt="Grad-CAM overlay"
                            style={{ opacity: gradCamOpacity / 100 }}
                          />
                        )}
                      </div>
                    )}

                    <div className="overlay-controls">
                      <label htmlFor="gradCamOpacity">Grad-CAM opacity: {gradCamOpacity}%</label>
                      <input
                        id="gradCamOpacity"
                        type="range"
                        min={0}
                        max={100}
                        value={gradCamOpacity}
                        onChange={(e) => setGradCamOpacity(Number(e.target.value))}
                      />
                    </div>
                  </div>
                </ContentPanel>
              </div>

              <div className="col col-right">
                <ContentPanel
                  title="Classification Metrics"
                  subtitle="Class identity and geopolitical relation output."
                >
                  {!classificationResult ? (
                    <div className="empty-state small">No classification runs yet.</div>
                  ) : (
                    <div className="classification-result">
                      <div className="classification-row">
                        <span className="classification-key">Aircraft class</span>
                        <span className="classification-value">{classificationResult.class_name}</span>
                      </div>
                      <div className="classification-row">
                        <span className="classification-key">Confidence</span>
                        <span className="classification-value">
                          {(classificationResult.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="classification-row">
                        <span className="classification-key">Origin country</span>
                        <span className="classification-value">{classificationResult.origin_country}</span>
                      </div>
                      <div className="classification-row">
                        <span className="classification-key">Friend/Foe</span>
                        <span
                          className={`ff-badge ${
                            classificationResult.friend_or_foe === "FRIEND"
                              ? "ff-friend"
                              : classificationResult.friend_or_foe === "FOE"
                                ? "ff-foe"
                                : "ff-neutral"
                          }`}
                        >
                          {classificationResult.friend_or_foe}
                        </span>
                      </div>
                    </div>
                  )}
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
        {activeSection === "aircraft-detection" && (
          <>
            <CapabilitiesSection />
            <PhilosophySection />
            <HomeFooter />
          </>
        )}
      </MainLayout>
    </div>
  );
};

