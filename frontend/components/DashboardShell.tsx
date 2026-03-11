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
    fetchImagePreview,
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
    const [changeOverlayBase64, setChangeOverlayBase64] = useState<string | null>(null);
    const [changeDebug, setChangeDebug] = useState<Record<string, number> | null>(null);
    const [semanticRegions, setSemanticRegions] = useState<
        { type: string; bbox: [number, number, number, number] }[]
    >([]);
    const [afterImageDims, setAfterImageDims] = useState<{ w: number; h: number } | null>(null);
    const [changeMaskOpacity, setChangeMaskOpacity] = useState<number>(45);
    const [changedPixels, setChangedPixels] = useState<number | null>(null);
    const [classificationResult, setClassificationResult] =
        useState<AircraftClassificationResponse | null>(null);
    const [aircraftClasses, setAircraftClasses] = useState<
        { detection: AircraftDetection; class_name: string; confidence: number }[]
    >([]);
    const [gradCamBase64, setGradCamBase64] = useState<string | null>(null);
    const [gradCamOpacity, setGradCamOpacity] = useState<number>(45);
    const [aircraftPreviewUrl, setAircraftPreviewUrl] = useState<string | null>(null);
    const [beforePreviewUrl, setBeforePreviewUrl] = useState<string | null>(null);
    const [afterPreviewUrl, setAfterPreviewUrl] = useState<string | null>(null);

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
    const changeOverlayUrl = useMemo(
        () => (changeOverlayBase64 ? `data:image/png;base64,${changeOverlayBase64}` : null),
        [changeOverlayBase64],
    );
    const gradCamUrl = useMemo(
        () => (gradCamBase64 ? `data:image/png;base64,${gradCamBase64}` : null),
        [gradCamBase64],
    );

    const aircraftDisplayUrl = aircraftPreviewUrl ?? aircraftUrl;
    const beforeDisplayUrl = beforePreviewUrl ?? beforeUrl;
    const afterDisplayUrl = afterPreviewUrl ?? afterUrl;

    const isTiff = (file: File | null) => {
        if (!file) return false;
        const name = file.name.toLowerCase();
        const type = file.type.toLowerCase();
        return name.endsWith(".tif") || name.endsWith(".tiff") || type.includes("tiff");
    };

    const formatMetric = (value: number | null | undefined) => {
        if (value == null || !Number.isFinite(value)) return "--";
        const abs = Math.abs(value);
        if (abs > 0 && abs < 0.0001) return value.toExponential(2);
        return value.toFixed(4);
    };

    useEffect(() => {
        let active = true;
        if (!aircraftFile || !isTiff(aircraftFile)) {
            setAircraftPreviewUrl(null);
            return;
        }
        fetchImagePreview(aircraftFile)
            .then((res) => {
                if (!active) return;
                setAircraftPreviewUrl(`data:image/png;base64,${res.png_base64}`);
            })
            .catch(() => {
                if (!active) return;
                setAircraftPreviewUrl(null);
            });
        return () => {
            active = false;
        };
    }, [aircraftFile]);

    useEffect(() => {
        let active = true;
        if (!beforeFile || !isTiff(beforeFile)) {
            setBeforePreviewUrl(null);
            return;
        }
        fetchImagePreview(beforeFile)
            .then((res) => {
                if (!active) return;
                setBeforePreviewUrl(`data:image/png;base64,${res.png_base64}`);
            })
            .catch(() => {
                if (!active) return;
                setBeforePreviewUrl(null);
            });
        return () => {
            active = false;
        };
    }, [beforeFile]);

    useEffect(() => {
        let active = true;
        if (!afterFile || !isTiff(afterFile)) {
            setAfterPreviewUrl(null);
            return;
        }
        fetchImagePreview(afterFile)
            .then((res) => {
                if (!active) return;
                setAfterPreviewUrl(`data:image/png;base64,${res.png_base64}`);
            })
            .catch(() => {
                if (!active) return;
                setAfterPreviewUrl(null);
            });
        return () => {
            active = false;
        };
    }, [afterFile]);

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
            setAircraftClasses([]);
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

    const cropDetectionToFile = async (file: File, det: AircraftDetection): Promise<File | null> => {
        try {
            const bitmap = await createImageBitmap(file);
            const x1 = Math.max(0, Math.floor(det.bbox.x1));
            const y1 = Math.max(0, Math.floor(det.bbox.y1));
            const x2 = Math.min(bitmap.width, Math.ceil(det.bbox.x2));
            const y2 = Math.min(bitmap.height, Math.ceil(det.bbox.y2));
            const w = Math.max(1, x2 - x1);
            const h = Math.max(1, y2 - y1);
            const canvas = document.createElement("canvas");
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext("2d");
            if (!ctx) return null;
            ctx.drawImage(bitmap, x1, y1, w, h, 0, 0, w, h);
            const blob = await new Promise<Blob | null>((resolve) =>
                canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92),
            );
            if (!blob) return null;
            return new File([blob], "crop.jpg", { type: "image/jpeg" });
        } catch {
            return null;
        }
    };

    const runAircraftRecognition = async () => {
        if (!aircraftFile) return;
        setError(null);
        setLoadingClassification(true);
        try {
            const detectionRes = await runAircraftDetection(aircraftFile, country);
            const detections = detectionRes.detections ?? [];
            setAircraftDetections(detections);
            if (!detections.length) {
                setAircraftClasses([]);
                setClassificationResult(null);
                setSystemMetrics({
                    inference_time_ms: detectionRes.inference_time_ms ?? undefined,
                    model_name: detectionRes.model_name ?? undefined,
                    device_used: detectionRes.device_used ?? undefined,
                    confidence: null,
                });
                return;
            }

            const results: { detection: AircraftDetection; class_name: string; confidence: number }[] = [];
            for (const det of detections) {
                const cropFile = await cropDetectionToFile(aircraftFile, det);
                if (!cropFile) continue;
                const cls = await runAircraftClassification(cropFile, country);
                results.push({
                    detection: det,
                    class_name: cls.class_name,
                    confidence: cls.confidence,
                });
            }
            setAircraftClasses(results);
            setClassificationResult(null);
            const bestConf =
                detections.length
                    ? detections.reduce(
                        (max, d) => (d.confidence > max ? d.confidence : max),
                        detections[0].confidence,
                    )
                    : null;
            setSystemMetrics({
                inference_time_ms: detectionRes.inference_time_ms ?? undefined,
                model_name: detectionRes.model_name ?? undefined,
                device_used: detectionRes.device_used ?? undefined,
                confidence: bestConf,
            });
        } catch (err: any) {
            setAircraftDetections([]);
            setAircraftClasses([]);
            setError(err?.response?.data?.detail ?? err?.message ?? "Aircraft recognition failed.");
        } finally {
            setLoadingClassification(false);
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
            setChangeOverlayBase64(res.overlay_base64 ?? null);
            setChangeDebug(res.debug ?? null);
            setSemanticRegions(res.regions ?? []);
            setChangedPixels(res.changed_pixels ?? null);
            setSystemMetrics({
                inference_time_ms: res.inference_time_ms ?? undefined,
                model_name: res.model_name ?? undefined,
                device_used: res.device_used ?? undefined,
                confidence: res.change_score,
            });
        } catch (err: any) {
            setChangeScore(null);
            setChangeMaskBase64(null);
            setChangeOverlayBase64(null);
            setChangeDebug(null);
            setSemanticRegions([]);
            setChangedPixels(null);
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

    const runAircraftAction = isAircraftIntelligence ? runAircraftRecognition : runAircraft;
    const aircraftBusy = isAircraftIntelligence ? loadingClassification : loadingAircraft;
    const canRunAircraftAction = !!aircraftFile;
    const aircraftActionLabel = isAircraftIntelligence
        ? "RUN DETECTION + RECOGNITION"
        : "RUN DETECTION";
    const aircraftBusyLabel = isAircraftIntelligence
        ? "ANALYZING..."
        : "DETECTING...";

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
                                        label="Upload: recognition image"
                                        helpText="Supports PNG/JPG/GeoTIFF for recognition."
                                        file={aircraftFile}
                                        onChange={(f) => {
                                            setAircraftFile(f);
                                            setAircraftDetections([]);
                                            setAircraftClasses([]);
                                        }}
                                    />
                                    {!consoleMode && (
                                        <div className="button-row">
                                            <button
                                                className="btn btn-primary"
                                                type="button"
                                                onClick={runAircraftRecognition}
                                                disabled={!aircraftFile || loadingClassification}
                                            >
                                                {loadingClassification ? "ANALYZING..." : "RUN DETECTION + RECOGNITION"}
                                            </button>
                                        </div>
                                    )}
                                    {aircraftDisplayUrl && (
                                        <DetectionCanvas
                                            imageUrl={aircraftDisplayUrl}
                                            detections={aircraftDetections}
                                            loading={loadingAircraft}
                                            title="Multi-Aircraft Detection"
                                            subtitle="Detection results used for per-aircraft recognition."
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
                                                subtitle="Per-aircraft recognition from detection crops."
                                            >
                                                {aircraftDetections.length === 0 ? (
                                                    <div className="empty-state small">
                                                        {loadingClassification
                                                            ? "Analyzing detections..."
                                                            : "No aircraft detected."}
                                                    </div>
                                                ) : aircraftClasses.length === 0 ? (
                                                    <div className="empty-state small">
                                                        {loadingClassification
                                                            ? "Classifying detected aircraft..."
                                                            : "No recognition results yet."}
                                                    </div>
                                                ) : (
                                                    <div className="table">
                                                        <div className="table-row table-head">
                                                            <div>Aircraft</div>
                                                            <div>Confidence</div>
                                                            <div>Box</div>
                                                        </div>
                                                        {aircraftClasses.map((item, idx) => (
                                                            <div key={idx} className="table-row">
                                                                <div className="mono">{item.class_name}</div>
                                                                <div className="mono">
                                                                    {(item.confidence * 100).toFixed(1)}%
                                                                </div>
                                                                <div className="mono">
                                                                    {Math.round(item.detection.bbox.x1)},
                                                                    {Math.round(item.detection.bbox.y1)} to{" "}
                                                                    {Math.round(item.detection.bbox.x2)},
                                                                    {Math.round(item.detection.bbox.y2)}
                                                                </div>
                                                            </div>
                                                        ))}
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
                                                setChangeOverlayBase64(null);
                                                setChangeDebug(null);
                                                setSemanticRegions([]);
                                                setChangedPixels(null);
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
                                                setChangeOverlayBase64(null);
                                                setChangeDebug(null);
                                                setSemanticRegions([]);
                                                setChangedPixels(null);
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

                                    <BeforeAfterSlider beforeUrl={beforeDisplayUrl} afterUrl={afterDisplayUrl} />

                                    <ContentPanel
                                        title="Change Mask Overlay"
                                        subtitle="Change mask rendered over the after image."
                                    >
                                        {!afterDisplayUrl ? (
                                            <div className="empty-state small">
                                                Upload and analyze images to view the mask overlay.
                                            </div>
                                        ) : (
                                            <div
                                                className="overlay-stage"
                                                style={
                                                    afterImageDims
                                                        ? { aspectRatio: `${afterImageDims.w} / ${afterImageDims.h}` }
                                                        : undefined
                                                }
                                            >
                                                <img
                                                    className="overlay-base-image"
                                                    src={afterDisplayUrl}
                                                    alt="After image"
                                                    onLoad={(e) => {
                                                        const target = e.currentTarget;
                                                        setAfterImageDims({
                                                            w: target.naturalWidth,
                                                            h: target.naturalHeight,
                                                        });
                                                    }}
                                                />
                                                {(changeOverlayUrl || changeMaskUrl) && (
                                                    <img
                                                        className={`overlay-mask-image ${changeOverlayUrl ? "" : "overlay-mask-red"}`}
                                                        src={changeOverlayUrl ?? changeMaskUrl ?? undefined}
                                                        alt="Change mask overlay"
                                                        style={{ opacity: changeMaskOpacity / 100 }}
                                                    />
                                                )}
                                                {(changeScore != null || changedPixels != null) && (
                                                    <div className="overlay-hud">
                                                        {changeScore != null && (
                                                            <div className="overlay-hud-row">
                                                                <span className="overlay-hud-key">Change score</span>
                                                                <span className="overlay-hud-value">
                                                                    {formatMetric(changeScore)}
                                                                </span>
                                                            </div>
                                                        )}
                                                        {changeDebug?.prob_max != null && (
                                                            <div className="overlay-hud-row">
                                                                <span className="overlay-hud-key">Prob max</span>
                                                                <span className="overlay-hud-value">
                                                                    {formatMetric(changeDebug.prob_max)}
                                                                </span>
                                                            </div>
                                                        )}
                                                        {changedPixels != null && (
                                                            <div className="overlay-hud-row">
                                                                <span className="overlay-hud-key">Changed pixels</span>
                                                                <span className="overlay-hud-value">{changedPixels}</span>
                                                            </div>
                                                        )}
                                                    </div>
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

                                    <ContentPanel
                                        title="Change Analysis"
                                        subtitle="Computed change score and detected regions."
                                    >
                                        {changeScore == null ? (
                                            <div className="empty-state small">
                                                Run change analysis to populate results.
                                            </div>
                                        ) : (
                                            <div className="classification-result">
                                                <div className="classification-row">
                                                    <span className="classification-key">Change score</span>
                                                    <span className="classification-value">
                                                        {formatMetric(changeScore)}
                                                    </span>
                                                </div>
                                                <div className="classification-row">
                                                    <span className="classification-key">Changed pixels</span>
                                                    <span className="classification-value">
                                                        {changedPixels == null ? "--" : changedPixels}
                                                    </span>
                                                </div>
                                                <div className="classification-row">
                                                    <span className="classification-key">Semantic regions</span>
                                                    <span className="classification-value">
                                                        {semanticRegions.length}
                                                    </span>
                                                </div>
                                                <div className="classification-row">
                                                    <span className="classification-key">Prob max</span>
                                                    <span className="classification-value">
                                                        {formatMetric(changeDebug?.prob_max)}
                                                    </span>
                                                </div>
                                                <div className="classification-row">
                                                    <span className="classification-key">Prob mean</span>
                                                    <span className="classification-value">
                                                        {formatMetric(changeDebug?.prob_mean)}
                                                    </span>
                                                </div>
                                            </div>
                                        )}
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
                    onRunAircraft={runAircraftAction}
                    onRunChange={runChange}
                    canRunAircraft={canRunAircraftAction}
                    canRunChange={!!beforeFile && !!afterFile}
                    loadingAircraft={aircraftBusy}
                    loadingChange={loadingChange}
                    error={error}
                    aircraftActionLabel={aircraftActionLabel}
                    aircraftBusyLabel={aircraftBusyLabel}
                />

                {content}
            </div>
        </div>
    );
};
