"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { AetherHero } from "./AetherHero";
import { ImageUploadPanel } from "./ImageUploadPanel";
import { DetectionCanvas } from "./DetectionCanvas";
import { ConfidencePanel } from "./ConfidencePanel";
import { SystemMetrics } from "./SystemMetrics";
import { ContentPanel } from "./ContentPanel";
import { BeforeAfterSlider } from "./BeforeAfterSlider";
import { CapabilitiesSection } from "./home/CapabilitiesSection";
import { PhilosophySection } from "./home/PhilosophySection";
import { HomeFooter } from "./home/HomeFooter";
import {
    runAircraftClassification,
    runAircraftDetection,
    runAircraftGradCam,
    runChangeDetection,
} from "@/lib/api";
import type {
    AircraftClassificationResponse,
    AircraftDetection,
    MainSection,
    SystemMetricsSnapshot,
} from "@/lib/types";
import { useCountry } from "@/context/CountryContext";

type DashboardShellProps = {
    initialSection?: MainSection;
    consoleMode?: boolean;
};

export const DashboardShell: React.FC<DashboardShellProps> = ({
    initialSection = "aircraft-intelligence",
    consoleMode = true,
}) => {
    const { country } = useCountry();
    const [activeSection, setActiveSection] = useState<MainSection>(initialSection);
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
    const [classificationResult, setClassificationResult] =
        useState<AircraftClassificationResponse | null>(null);
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

    useEffect(() => {
        if (!consoleMode) {
            setActiveSection(initialSection);
        }
    }, [consoleMode, initialSection]);

    /* ── API Handlers ── */

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

    /* ── Render ── */

    const isAircraftIntelligence =
        activeSection === "aircraft-intelligence" || activeSection === "aircraft-detection";

    const content = (
        <main className="main-content">
                    {consoleMode && isAircraftIntelligence && <AetherHero />}

                    <div
                        className={`grid ${!consoleMode &&
                                (isAircraftIntelligence ||
                                    activeSection === "aircraft-classification" ||
                                    activeSection === "change-detection")
                                ? "grid-single-column"
                                : ""
                            }`}
                        id="ops-grid"
                    >
                        {/* ── Aircraft Detection ── */}
                        {isAircraftIntelligence && (
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
                                    {!consoleMode && (
                                        <div className="button-row">
                                            <button
                                                className="btn btn-primary"
                                                type="button"
                                                onClick={runAircraft}
                                                disabled={!aircraftFile || loadingAircraft}
                                            >
                                                {loadingAircraft ? "DETECTING..." : "RUN DETECTION"}
                                            </button>
                                            <button
                                                className="btn btn-outline"
                                                type="button"
                                                onClick={async () => {
                                                    if (!aircraftFile) return;
                                                    setError(null);
                                                    setLoadingClassification(true);
                                                    try {
                                                        const res = await runAircraftClassification(aircraftFile, country);
                                                        setClassificationResult(res);
                                                        setSystemMetrics({
                                                            inference_time_ms: res.inference_time_ms ?? undefined,
                                                            model_name: res.model_name ?? undefined,
                                                            device_used: res.device_used ?? undefined,
                                                            confidence: res.confidence,
                                                        });
                                                    } catch (err: any) {
                                                        setClassificationResult(null);
                                                        setError(
                                                            err?.response?.data?.detail ??
                                                                err?.message ??
                                                                "Aircraft recognition failed.",
                                                        );
                                                    } finally {
                                                        setLoadingClassification(false);
                                                    }
                                                }}
                                                disabled={!aircraftFile || loadingClassification}
                                            >
                                                {loadingClassification ? "RECOGNIZING..." : "RUN RECOGNITION"}
                                            </button>
                                        </div>
                                    )}
                                    {aircraftUrl && (
                                        <DetectionCanvas
                                            imageUrl={aircraftUrl}
                                            detections={aircraftDetections}
                                            loading={loadingAircraft}
                                        />
                                    )}

                                    {!consoleMode && (
                                        <>
                                            <div className="detection-metrics-row">
                                                <ConfidencePanel
                                                    detections={aircraftDetections}
                                                    changeScore={null}
                                                    loadingAircraft={loadingAircraft}
                                                    loadingChange={false}
                                                    error={error}
                                                />
                                                <SystemMetrics metrics={systemMetrics} />
                                            </div>
                                            <ContentPanel
                                                title="Recognition Output"
                                                subtitle="Detected class identity from uploaded airframe."
                                            >
                                                {!classificationResult ? (
                                                    <div className="empty-state small">
                                                        Run recognition to populate classification output.
                                                    </div>
                                                ) : (
                                                    <div className="classification-result">
                                                        <div className="classification-row">
                                                            <span className="classification-key">Aircraft class</span>
                                                            <span className="classification-value">
                                                                {classificationResult.class_name}
                                                            </span>
                                                        </div>
                                                        <div className="classification-row">
                                                            <span className="classification-key">Confidence</span>
                                                            <span className="classification-value">
                                                                {(classificationResult.confidence * 100).toFixed(2)}%
                                                            </span>
                                                        </div>
                                                        <div className="classification-row">
                                                            <span className="classification-key">Origin country</span>
                                                            <span className="classification-value">
                                                                {classificationResult.origin_country}
                                                            </span>
                                                        </div>
                                                    </div>
                                                )}
                                            </ContentPanel>
                                        </>
                                    )}
                                </div>
                                {consoleMode && (
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
                                )}
                            </>
                        )}

                        {/* ── Change Detection ── */}
                        {activeSection === "change-detection" && (
                            <>
                                <div className="col col-left">
                                    <div className="stack change-upload-grid">
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
                                        {!consoleMode && (
                                            <div className="button-row button-row-span">
                                                <button
                                                    className="btn btn-primary"
                                                    type="button"
                                                    onClick={runChange}
                                                    disabled={!beforeFile || !afterFile || loadingChange}
                                                >
                                                    {loadingChange ? "ANALYZING..." : "RUN CHANGE ANALYSIS"}
                                                </button>
                                            </div>
                                        )}
                                    </div>

                                    <BeforeAfterSlider beforeUrl={beforeUrl} afterUrl={afterUrl} />

                                    <ContentPanel
                                        title="Change Mask Overlay"
                                        subtitle="Change mask rendered over the after image."
                                    >
                                        {!afterUrl ? (
                                            <div className="empty-state small">
                                                Upload and analyze images to view the mask overlay.
                                            </div>
                                        ) : (
                                            <div className="overlay-stage">
                                                <img
                                                    className="overlay-base-image"
                                                    src={afterUrl}
                                                    alt="After image"
                                                    onLoad={(e) => {
                                                        const target = e.currentTarget;
                                                        setAfterImageDims({
                                                            w: target.naturalWidth,
                                                            h: target.naturalHeight,
                                                        });
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
                                            <label htmlFor="changeMaskOpacity">
                                                Mask opacity: {changeMaskOpacity}%
                                            </label>
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

                                    {!consoleMode && (
                                        <div className="detection-metrics-row">
                                            <ConfidencePanel
                                                detections={[]}
                                                changeScore={changeScore}
                                                loadingAircraft={false}
                                                loadingChange={loadingChange}
                                                error={error}
                                            />
                                            <SystemMetrics metrics={systemMetrics} />
                                        </div>
                                    )}
                                </div>

                                {consoleMode && (
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
                                )}
                            </>
                        )}

                        {/* ── Aircraft Classification ── */}
                        {activeSection === "aircraft-classification" && (
                            <>
                                <div className="col col-left">
                                    <ContentPanel
                                        title="Aircraft Recognition"
                                        subtitle="Upload image, run recognition, then generate Grad-CAM."
                                    >
                                        <div className="stack stack-single">
                                            <ImageUploadPanel
                                                label="Upload: recognition image"
                                                helpText="Used for ConvNeXt aircraft recognition."
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
                                                    {loadingClassification ? "RECOGNIZING..." : "RUN RECOGNITION"}
                                                </button>
                                                <button
                                                    className="btn btn-outline"
                                                    type="button"
                                                    onClick={runGradCam}
                                                    disabled={!classificationFile || loadingGradCam}
                                                >
                                                    {loadingGradCam ? "GENERATING..." : "SHOW EXPLAINABILITY"}
                                                </button>
                                            </div>

                                            {!classificationUrl ? (
                                                <div className="empty-state small">
                                                    Upload an image to preview recognition and Grad-CAM.
                                                </div>
                                            ) : (
                                                <div className="overlay-stage">
                                                    <img
                                                        className="overlay-base-image"
                                                        src={classificationUrl}
                                                        alt="Recognition input"
                                                    />
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
                                                <label htmlFor="gradCamOpacity">
                                                    Grad-CAM opacity: {gradCamOpacity}%
                                                </label>
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
                                    {!consoleMode && (
                                        <div className="detection-metrics-row">
                                            <ContentPanel
                                                title="Recognition Output"
                                                subtitle="Class identity and geopolitical relation output."
                                            >
                                                {!classificationResult ? (
                                                    <div className="empty-state small">No recognition runs yet.</div>
                                                ) : (
                                                    <div className="classification-result">
                                                        <div className="classification-row">
                                                            <span className="classification-key">Aircraft class</span>
                                                            <span className="classification-value">
                                                                {classificationResult.class_name}
                                                            </span>
                                                        </div>
                                                        <div className="classification-row">
                                                            <span className="classification-key">Confidence</span>
                                                            <span className="classification-value">
                                                                {(classificationResult.confidence * 100).toFixed(2)}%
                                                            </span>
                                                        </div>
                                                        <div className="classification-row">
                                                            <span className="classification-key">Origin country</span>
                                                            <span className="classification-value">
                                                                {classificationResult.origin_country}
                                                            </span>
                                                        </div>
                                                        <div className="classification-row">
                                                            <span className="classification-key">Friend/Foe</span>
                                                            <span
                                                                className={`ff-badge ${classificationResult.friend_or_foe === "FRIEND"
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
                                            <SystemMetrics metrics={systemMetrics} />
                                        </div>
                                    )}
                                </div>

                                {consoleMode && (
                                    <div className="col col-right">
                                        <ContentPanel
                                            title="Recognition Output"
                                            subtitle="Class identity and geopolitical relation output."
                                        >
                                            {!classificationResult ? (
                                                <div className="empty-state small">No recognition runs yet.</div>
                                            ) : (
                                                <div className="classification-result">
                                                    <div className="classification-row">
                                                        <span className="classification-key">Aircraft class</span>
                                                        <span className="classification-value">
                                                            {classificationResult.class_name}
                                                        </span>
                                                    </div>
                                                    <div className="classification-row">
                                                        <span className="classification-key">Confidence</span>
                                                        <span className="classification-value">
                                                            {(classificationResult.confidence * 100).toFixed(2)}%
                                                        </span>
                                                    </div>
                                                    <div className="classification-row">
                                                        <span className="classification-key">Origin country</span>
                                                        <span className="classification-value">
                                                            {classificationResult.origin_country}
                                                        </span>
                                                    </div>
                                                    <div className="classification-row">
                                                        <span className="classification-key">Friend/Foe</span>
                                                        <span
                                                            className={`ff-badge ${classificationResult.friend_or_foe === "FRIEND"
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
                                )}
                            </>
                        )}

                        {/* ── Metrics Dashboard ── */}
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
                                            <li>
                                                Confidence is class-score based; tune thresholds server-side.
                                            </li>
                                            <li>
                                                Use the overlay to validate boxes before downstream actions.
                                            </li>
                                        </ul>
                                    </ContentPanel>
                                </div>
                            </>
                        )}
                    </div>

                    {consoleMode && isAircraftIntelligence && (
                        <>
                            <CapabilitiesSection />
                            <PhilosophySection />
                            <HomeFooter />
                        </>
                    )}
                </main>
    );

    if (!consoleMode) {
        return <div className="main-area main-area-standalone">{content}</div>;
    }

    return (
        <div className="main-layout">
            <Sidebar active={activeSection} onChange={setActiveSection} />

            <div className="main-area main-area-console">
                <Topbar
                    activeSection={activeSection}
                    onRunAircraft={runAircraft}
                    onRunChange={runChange}
                    canRunAircraft={!!aircraftFile}
                    canRunChange={!!beforeFile && !!afterFile}
                    loadingAircraft={loadingAircraft}
                    loadingChange={loadingChange}
                    error={error}
                />

                {content}
            </div>
        </div>
    );
};
