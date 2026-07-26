"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Navbar } from "@/components/Navbar";
import {
    runAircraftClassification,
    runAircraftDetectClassify,
    runAircraftDetectVideo,
    runAircraftGradCam,
    streamAircraftDetect,
} from "@/lib/api";
import type {
    AircraftClassificationResponse,
    AircraftDetectVideoResponse,
    AircraftGradCamResponse,
    BBoxDetection,
    DetectClassifyResponse,
    VideoFrameResult,
} from "@/lib/types";

const COUNTRIES = ["USA", "China", "Russia", "India", "UK", "France", "Germany", "Japan"] as const;
const VIDEO_ACCEPT = "video/mp4,video/avi,video/quicktime,video/x-matroska,.mp4,.avi,.mov,.mkv";
type InputTab = "image" | "video" | "stream";
type StreamStatus = "DISCONNECTED" | "CONNECTING" | "LIVE";

function friendFoeStyle(value: string) {
    if (value === "FRIEND") {
        return { border: "1px solid #22D3EE", color: "#22D3EE" };
    }
    if (value === "FOE") {
        return { border: "1px solid #EF4444", color: "#EF4444" };
    }
    return { border: "1px solid #6B7280", color: "#9CA3AF" };
}

function confidenceStyle(confidence: number) {
    if (confidence >= 0.8) {
        return "#22C55E";
    }
    if (confidence >= 0.6) {
        return "#F59E0B";
    }
    return "#EF4444";
}

function modelStatusStyle(online: boolean) {
    return online
        ? { border: "1px solid #164E63", color: "#22D3EE" }
        : { border: "1px solid #7F1D1D", color: "#EF4444" };
}

function readHealthUrl() {
    if (typeof window === "undefined") {
        return "http://127.0.0.1:8000/health/models";
    }
    return `${window.location.protocol}//${window.location.hostname}:8000/health/models`;
}

function tabButtonStyle(active: boolean): React.CSSProperties {
    return {
        flex: 1,
        border: "none",
        borderBottom: active ? "2px solid #22D3EE" : "2px solid transparent",
        background: "transparent",
        color: active ? "var(--text-primary)" : "#6B7280",
        padding: "0.75rem 0.5rem",
        textTransform: "uppercase",
        letterSpacing: "0.18em",
        fontSize: "0.62rem",
        cursor: "pointer",
        borderRadius: 0,
    };
}

function actionButtonStyle(disabled: boolean): React.CSSProperties {
    return {
        width: "100%",
        border: "1px solid rgba(255,255,255,0.18)",
        background: "rgba(255,255,255,0.03)",
        color: "var(--text-primary)",
        padding: "0.9rem 1rem",
        textTransform: "uppercase",
        letterSpacing: "0.22em",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        borderRadius: 0,
    };
}

function FriendFoeBadge({ value }: { value: string }) {
    return (
        <div
            className="mono"
            style={{
                padding: "2px 6px",
                fontSize: "0.58rem",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                borderRadius: 0,
                ...friendFoeStyle(value),
            }}
        >
            {value}
        </div>
    );
}

function DetectionCard({ detection }: { detection: BBoxDetection }) {
    return (
        <div
            style={{
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(0,0,0,0.35)",
                borderRadius: 0,
                overflow: "hidden",
            }}
        >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
                src={`data:image/png;base64,${detection.crop_base64}`}
                alt={detection.class_name}
                style={{ width: "100%", height: "120px", objectFit: "cover", display: "block", background: "#000" }}
            />
            <div style={{ padding: "0.75rem" }}>
                <div style={{ fontWeight: 600, fontSize: "0.88rem" }}>{detection.class_name}</div>
                <div className="mono" style={{ marginTop: "0.35rem", fontSize: "0.62rem", color: "var(--text-muted)" }}>
                    {(detection.class_confidence * 100).toFixed(1)}% confidence
                </div>
                <div style={{ marginTop: "0.65rem" }}>
                    <FriendFoeBadge value={detection.friend_or_foe} />
                </div>
            </div>
        </div>
    );
}

function computeVideoSummary(frames: VideoFrameResult[]) {
    const validFrames = frames.filter((frame) => !frame.error);
    const peakAircraft = validFrames.reduce((max, frame) => Math.max(max, frame.total_aircraft ?? 0), 0);
    const classCounts = new Map<string, number>();

    for (const frame of validFrames) {
        for (const detection of frame.detections ?? []) {
            classCounts.set(detection.class_name, (classCounts.get(detection.class_name) ?? 0) + 1);
        }
    }

    let mostCommonClass = "--";
    let topCount = 0;
    for (const [className, count] of classCounts.entries()) {
        if (count > topCount) {
            topCount = count;
            mostCommonClass = className;
        }
    }

    return {
        totalFrames: validFrames.length,
        peakAircraft,
        mostCommonClass,
    };
}

export default function AircraftIntelligencePage() {
    const [activeTab, setActiveTab] = useState<InputTab>("image");

    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [country, setCountry] = useState<(typeof COUNTRIES)[number]>("USA");
    const [singleResult, setSingleResult] = useState<AircraftClassificationResponse | null>(null);
    const [detectResult, setDetectResult] = useState<DetectClassifyResponse | null>(null);
    const [imageResultMode, setImageResultMode] = useState<"none" | "single" | "detect">("none");
    const [gradcam, setGradcam] = useState<AircraftGradCamResponse | null>(null);
    const [classifying, setClassifying] = useState(false);
    const [detecting, setDetecting] = useState(false);
    const [gradcamLoading, setGradcamLoading] = useState(false);
    const [gradcamError, setGradcamError] = useState(false);
    const [imageError, setImageError] = useState<string | null>(null);
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [classifierOnline, setClassifierOnline] = useState(false);

    const [videoFile, setVideoFile] = useState<File | null>(null);
    const [videoResult, setVideoResult] = useState<AircraftDetectVideoResponse | null>(null);
    const [videoProcessing, setVideoProcessing] = useState(false);
    const [videoProgress, setVideoProgress] = useState(0);
    const [videoError, setVideoError] = useState<string | null>(null);

    const [streamUrl, setStreamUrl] = useState("");
    const [streamStatus, setStreamStatus] = useState<StreamStatus>("DISCONNECTED");
    const [currentStreamFrame, setCurrentStreamFrame] = useState<VideoFrameResult | null>(null);
    const [streamLog, setStreamLog] = useState<Array<{ id: string; timestamp: number; summary: string }>>([]);
    const [streamError, setStreamError] = useState<string | null>(null);
    const streamAbortRef = useRef<AbortController | null>(null);

    useEffect(() => {
        let active = true;
        fetch(readHealthUrl())
            .then((response) => {
                if (!response.ok) {
                    throw new Error("offline");
                }
                return response.json();
            })
            .then((payload) => {
                if (active) {
                    setClassifierOnline(payload?.status === "ok");
                }
            })
            .catch(() => {
                if (active) {
                    setClassifierOnline(false);
                }
            });
        return () => {
            active = false;
        };
    }, []);

    useEffect(() => {
        if (!file) {
            setPreviewUrl(null);
            return;
        }
        const objectUrl = URL.createObjectURL(file);
        setPreviewUrl(objectUrl);
        return () => URL.revokeObjectURL(objectUrl);
    }, [file]);

    useEffect(() => {
        return () => {
            streamAbortRef.current?.abort();
        };
    }, []);

    const heatmapSrc = useMemo(() => {
        if (!gradcam?.heatmap_base64_png) {
            return null;
        }
        return `data:image/png;base64,${gradcam.heatmap_base64_png}`;
    }, [gradcam]);

    const videoSummary = useMemo(() => {
        if (!videoResult?.frames?.length) {
            return null;
        }
        return computeVideoSummary(videoResult.frames);
    }, [videoResult]);

    const liveDetectionCount = currentStreamFrame?.total_aircraft ?? 0;

    async function classifySingle() {
        if (!file) {
            return;
        }
        setClassifying(true);
        setDetectResult(null);
        setSingleResult(null);
        setImageResultMode("none");
        setGradcam(null);
        setImageError(null);
        try {
            const classification = await runAircraftClassification(file, country);
            setSingleResult(classification);
            setImageResultMode("single");
            setGradcamLoading(true);
            setGradcamError(false);
            try {
                const heatmap = await runAircraftGradCam(file, country);
                setGradcam(heatmap);
            } catch (gcError: any) {
                setGradcamError(true);
                if (gcError?.response?.status !== 400) {
                    console.warn("Grad-CAM generation failed:", gcError);
                }
            } finally {
                setGradcamLoading(false);
            }
        } catch (classificationError: any) {
            setImageError(classificationError?.response?.data?.detail ?? "Classification failed.");
        } finally {
            setClassifying(false);
        }
    }

    async function detectAllAircraft() {
        if (!file) {
            return;
        }
        setDetecting(true);
        setDetectResult(null);
        setSingleResult(null);
        setImageResultMode("none");
        setGradcam(null);
        setImageError(null);
        try {
            const result = await runAircraftDetectClassify(file, country);
            setDetectResult(result);
            setImageResultMode("detect");
        } catch (detectError: any) {
            setImageError(detectError?.response?.data?.detail ?? "Detection failed.");
        } finally {
            setDetecting(false);
        }
    }

    async function analyseVideo() {
        if (!videoFile) {
            return;
        }
        setVideoProcessing(true);
        setVideoProgress(8);
        setVideoResult(null);
        setVideoError(null);

        const progressTimer = window.setInterval(() => {
            setVideoProgress((value) => (value >= 92 ? value : value + 4));
        }, 500);

        try {
            const result = await runAircraftDetectVideo(videoFile, country);
            setVideoResult(result);
            setVideoProgress(100);
        } catch (err: any) {
            setVideoError(err?.response?.data?.detail ?? "Video analysis failed.");
            setVideoProgress(0);
        } finally {
            window.clearInterval(progressTimer);
            setVideoProcessing(false);
        }
    }

    function disconnectStream() {
        streamAbortRef.current?.abort();
        streamAbortRef.current = null;
        setStreamStatus("DISCONNECTED");
    }

    async function connectStream() {
        if (!streamUrl.trim()) {
            return;
        }
        disconnectStream();
        setStreamError(null);
        setStreamLog([]);
        setCurrentStreamFrame(null);
        setStreamStatus("CONNECTING");

        const controller = new AbortController();
        streamAbortRef.current = controller;

        try {
            await streamAircraftDetect(streamUrl.trim(), {
                country,
                maxFrames: 60,
                signal: controller.signal,
                onFrame: (frame) => {
                    setStreamStatus("LIVE");
                    if (frame.error) {
                        setStreamError(frame.error);
                        return;
                    }
                    setCurrentStreamFrame(frame);
                    const names = (frame.detections ?? []).map((d) => d.class_name).join(", ") || "No aircraft";
                    setStreamLog((prev) => [
                        {
                            id: `${frame.frame_number}-${frame.timestamp_sec}`,
                            timestamp: frame.timestamp_sec,
                            summary: `${frame.total_aircraft ?? 0} aircraft — ${names}`,
                        },
                        ...prev,
                    ].slice(0, 50));
                },
                onDone: () => {
                    if (!controller.signal.aborted) {
                        setStreamStatus("DISCONNECTED");
                    }
                },
                onError: (error) => {
                    setStreamError(error.message);
                    setStreamStatus("DISCONNECTED");
                },
            });
        } catch (err: any) {
            if (!controller.signal.aborted) {
                setStreamError(err?.message ?? "Stream connection failed.");
                setStreamStatus("DISCONNECTED");
            }
        }
    }

    function onImageSelected(nextFile: File | null) {
        setFile(nextFile);
        setSingleResult(null);
        setDetectResult(null);
        setImageResultMode("none");
        setGradcam(null);
        setGradcamError(false);
        setImageError(null);
    }

    function onImageDrop(event: React.DragEvent<HTMLLabelElement>) {
        event.preventDefault();
        const dropped = event.dataTransfer.files?.[0] ?? null;
        if (dropped) {
            onImageSelected(dropped);
        }
    }

    function onVideoSelected(nextFile: File | null) {
        setVideoFile(nextFile);
        setVideoResult(null);
        setVideoError(null);
        setVideoProgress(0);
    }

    function onVideoDrop(event: React.DragEvent<HTMLLabelElement>) {
        event.preventDefault();
        const dropped = event.dataTransfer.files?.[0] ?? null;
        if (dropped) {
            onVideoSelected(dropped);
        }
    }

    return (
        <div className="app">
            <Navbar />
            <div className="home-body">
                <header
                    className="capability-header"
                    style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "flex-start", flexWrap: "wrap" }}
                >
                    <div>
                        <p className="capability-kicker mono">AETHER EYE</p>
                        <h1 className="capability-title">AIRCRAFT INTELLIGENCE</h1>
                        <p className="capability-description">
                            Fine-grained aircraft classification across 100 aircraft types with explainability
                        </p>
                        <div
                            className="mono"
                            style={{ marginTop: "0.75rem", color: "#4B5563", fontSize: "0.65rem", letterSpacing: "0.12em", textTransform: "uppercase" }}
                        >
                            YOLOv8 Detection + ConvNeXt Small — 100 classes — 72.5% top-1
                        </div>
                    </div>
                    <div
                        className="mono"
                        style={{
                            padding: "1px 6px",
                            fontSize: "0.6rem",
                            letterSpacing: "0.12em",
                            textTransform: "uppercase",
                            borderRadius: 0,
                            ...modelStatusStyle(classifierOnline),
                        }}
                    >
                        {classifierOnline ? "Classifier Online" : "Classifier Offline"}
                    </div>
                </header>

                <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 0.95fr) minmax(380px, 1.15fr)", gap: "1.5rem" }}>
                    <section className="glass-panel" style={{ padding: "1.25rem" }}>
                        <div style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.1)", marginBottom: "1rem" }}>
                            <button type="button" className="mono" style={tabButtonStyle(activeTab === "image")} onClick={() => setActiveTab("image")}>
                                Image Upload
                            </button>
                            <button type="button" className="mono" style={tabButtonStyle(activeTab === "video")} onClick={() => setActiveTab("video")}>
                                Video Upload
                            </button>
                            <button type="button" className="mono" style={tabButtonStyle(activeTab === "stream")} onClick={() => setActiveTab("stream")}>
                                Live Stream
                            </button>
                        </div>

                        {activeTab === "image" ? (
                            <>
                                <div className="ops-kicker mono">Upload Image</div>
                                <h2 className="ops-panel-title mono" style={{ marginBottom: "1rem" }}>Classification Input</h2>
                                <label
                                    onDrop={onImageDrop}
                                    onDragOver={(event) => event.preventDefault()}
                                    style={{
                                        display: "grid",
                                        placeItems: "center",
                                        minHeight: "280px",
                                        border: "1px dashed rgba(255,255,255,0.14)",
                                        background: "rgba(255,255,255,0.02)",
                                        cursor: "pointer",
                                        overflow: "hidden",
                                        borderRadius: 0,
                                    }}
                                >
                                    <input
                                        type="file"
                                        accept="image/jpeg,image/png"
                                        style={{ display: "none" }}
                                        onChange={(event) => onImageSelected(event.target.files?.[0] ?? null)}
                                    />
                                    {previewUrl ? (
                                        // eslint-disable-next-line @next/next/no-img-element
                                        <img src={previewUrl} alt="Aircraft preview" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                                    ) : (
                                        <div style={{ textAlign: "center", padding: "1rem" }}>
                                            <div className="mono" style={{ fontSize: "0.72rem", color: "var(--text-primary)", textTransform: "uppercase" }}>
                                                Drag and drop or click to upload
                                            </div>
                                            <div className="mono" style={{ marginTop: "0.65rem", fontSize: "0.6rem", color: "#4B5563", textTransform: "uppercase" }}>
                                                Accepts JPEG and PNG
                                            </div>
                                        </div>
                                    )}
                                </label>

                                <div style={{ marginTop: "1rem" }}>
                                    <label className="mono" style={{ display: "block", marginBottom: "0.45rem", color: "#4B5563", fontSize: "0.62rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                                        Operator country (for friend/foe context)
                                    </label>
                                    <select
                                        value={country}
                                        onChange={(event) => setCountry(event.target.value as (typeof COUNTRIES)[number])}
                                        style={{
                                            width: "100%",
                                            background: "rgba(255,255,255,0.03)",
                                            border: "1px solid rgba(255,255,255,0.12)",
                                            color: "var(--text-primary)",
                                            padding: "0.75rem",
                                            borderRadius: 0,
                                        }}
                                    >
                                        {COUNTRIES.map((option) => (
                                            <option key={option} value={option} style={{ color: "#000000" }}>
                                                {option}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginTop: "1rem" }}>
                                    <button
                                        type="button"
                                        onClick={classifySingle}
                                        disabled={!file || classifying || detecting}
                                        className="mono"
                                        style={actionButtonStyle(!file || classifying || detecting)}
                                    >
                                        {classifying ? "Classifying..." : "Classify Single"}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={detectAllAircraft}
                                        disabled={!file || classifying || detecting}
                                        className="mono"
                                        style={actionButtonStyle(!file || classifying || detecting)}
                                    >
                                        {detecting ? "Detecting..." : "Detect All Aircraft"}
                                    </button>
                                </div>

                                {imageError ? (
                                    <div style={{ marginTop: "1rem", border: "1px solid #7F1D1D", color: "#FCA5A5", padding: "0.8rem", fontSize: "0.82rem", borderRadius: 0 }}>
                                        {typeof imageError === "string" ? imageError : JSON.stringify(imageError)}
                                    </div>
                                ) : null}
                            </>
                        ) : null}

                        {activeTab === "video" ? (
                            <>
                                <div className="ops-kicker mono">Upload Video</div>
                                <h2 className="ops-panel-title mono" style={{ marginBottom: "1rem" }}>Video Analysis Input</h2>
                                <label
                                    onDrop={onVideoDrop}
                                    onDragOver={(event) => event.preventDefault()}
                                    style={{
                                        display: "grid",
                                        placeItems: "center",
                                        minHeight: "220px",
                                        border: "1px dashed rgba(255,255,255,0.14)",
                                        background: "rgba(255,255,255,0.02)",
                                        cursor: "pointer",
                                        borderRadius: 0,
                                    }}
                                >
                                    <input
                                        type="file"
                                        accept={VIDEO_ACCEPT}
                                        style={{ display: "none" }}
                                        onChange={(event) => onVideoSelected(event.target.files?.[0] ?? null)}
                                    />
                                    <div style={{ textAlign: "center", padding: "1rem" }}>
                                        <div className="mono" style={{ fontSize: "0.72rem", color: "var(--text-primary)", textTransform: "uppercase" }}>
                                            {videoFile ? videoFile.name : "Drag and drop or click to upload video"}
                                        </div>
                                        <div className="mono" style={{ marginTop: "0.65rem", fontSize: "0.6rem", color: "#4B5563", textTransform: "uppercase" }}>
                                            Accepts MP4, AVI, MOV, MKV
                                        </div>
                                        <div className="mono" style={{ marginTop: "0.45rem", fontSize: "0.58rem", color: "#F59E0B", textTransform: "uppercase" }}>
                                            Large files may take time to process
                                        </div>
                                    </div>
                                </label>

                                <div style={{ marginTop: "1rem" }}>
                                    <label className="mono" style={{ display: "block", marginBottom: "0.45rem", color: "#4B5563", fontSize: "0.62rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                                        Operator country (for friend/foe context)
                                    </label>
                                    <select
                                        value={country}
                                        onChange={(event) => setCountry(event.target.value as (typeof COUNTRIES)[number])}
                                        style={{
                                            width: "100%",
                                            background: "rgba(255,255,255,0.03)",
                                            border: "1px solid rgba(255,255,255,0.12)",
                                            color: "var(--text-primary)",
                                            padding: "0.75rem",
                                            borderRadius: 0,
                                        }}
                                    >
                                        {COUNTRIES.map((option) => (
                                            <option key={option} value={option} style={{ color: "#000000" }}>
                                                {option}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <button
                                    type="button"
                                    onClick={analyseVideo}
                                    disabled={!videoFile || videoProcessing}
                                    className="mono"
                                    style={{ ...actionButtonStyle(!videoFile || videoProcessing), marginTop: "1rem" }}
                                >
                                    {videoProcessing ? "Analysing Video..." : "Analyse Video"}
                                </button>

                                {videoProcessing ? (
                                    <div style={{ marginTop: "1rem" }}>
                                        <div className="mono" style={{ fontSize: "0.6rem", color: "#4B5563", textTransform: "uppercase", marginBottom: "0.35rem" }}>
                                            Processing frames...
                                        </div>
                                        <div style={{ height: "8px", background: "rgba(255,255,255,0.08)" }}>
                                            <div style={{ width: `${videoProgress}%`, height: "100%", background: "#22D3EE", transition: "width 0.3s ease" }} />
                                        </div>
                                    </div>
                                ) : null}

                                {videoError ? (
                                    <div style={{ marginTop: "1rem", border: "1px solid #7F1D1D", color: "#FCA5A5", padding: "0.8rem", fontSize: "0.82rem", borderRadius: 0 }}>
                                        {typeof videoError === "string" ? videoError : JSON.stringify(videoError)}
                                    </div>
                                ) : null}
                            </>
                        ) : null}

                        {activeTab === "stream" ? (
                            <>
                                <div className="ops-kicker mono">Live Stream</div>
                                <h2 className="ops-panel-title mono" style={{ marginBottom: "1rem" }}>Stream Input</h2>
                                <label className="mono" style={{ display: "block", marginBottom: "0.45rem", color: "#4B5563", fontSize: "0.62rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                                    Stream URL
                                </label>
                                <input
                                    type="text"
                                    value={streamUrl}
                                    onChange={(event) => setStreamUrl(event.target.value)}
                                    placeholder="YouTube URL, RTSP stream, or HTTP stream"
                                    style={{
                                        width: "100%",
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.12)",
                                        color: "var(--text-primary)",
                                        padding: "0.75rem",
                                        borderRadius: 0,
                                    }}
                                />

                                <div style={{ marginTop: "1rem" }}>
                                    <label className="mono" style={{ display: "block", marginBottom: "0.45rem", color: "#4B5563", fontSize: "0.62rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                                        Operator country (for friend/foe context)
                                    </label>
                                    <select
                                        value={country}
                                        onChange={(event) => setCountry(event.target.value as (typeof COUNTRIES)[number])}
                                        style={{
                                            width: "100%",
                                            background: "rgba(255,255,255,0.03)",
                                            border: "1px solid rgba(255,255,255,0.12)",
                                            color: "var(--text-primary)",
                                            padding: "0.75rem",
                                            borderRadius: 0,
                                        }}
                                    >
                                        {COUNTRIES.map((option) => (
                                            <option key={option} value={option} style={{ color: "#000000" }}>
                                                {option}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <div style={{ display: "grid", gridTemplateColumns: streamStatus === "LIVE" || streamStatus === "CONNECTING" ? "1fr 1fr" : "1fr", gap: "0.75rem", marginTop: "1rem" }}>
                                    <button
                                        type="button"
                                        onClick={connectStream}
                                        disabled={!streamUrl.trim() || streamStatus === "CONNECTING" || streamStatus === "LIVE"}
                                        className="mono"
                                        style={actionButtonStyle(!streamUrl.trim() || streamStatus === "CONNECTING" || streamStatus === "LIVE")}
                                    >
                                        Connect
                                    </button>
                                    {streamStatus === "LIVE" || streamStatus === "CONNECTING" ? (
                                        <button
                                            type="button"
                                            onClick={disconnectStream}
                                            className="mono"
                                            style={{
                                                ...actionButtonStyle(false),
                                                border: "1px solid #7F1D1D",
                                                color: "#FCA5A5",
                                            }}
                                        >
                                            Disconnect
                                        </button>
                                    ) : null}
                                </div>

                                <div className="mono" style={{ marginTop: "1rem", fontSize: "0.62rem", letterSpacing: "0.14em", textTransform: "uppercase", color: streamStatus === "LIVE" ? "#22D3EE" : streamStatus === "CONNECTING" ? "#F59E0B" : "#6B7280" }}>
                                    Status: {streamStatus}
                                </div>

                                {streamError ? (
                                    <div style={{ marginTop: "1rem", border: "1px solid #7F1D1D", color: "#FCA5A5", padding: "0.8rem", fontSize: "0.82rem", borderRadius: 0 }}>
                                        {streamError}
                                    </div>
                                ) : null}
                            </>
                        ) : null}
                    </section>

                    <section className="glass-panel" style={{ padding: "1.25rem" }}>
                        <div className="ops-kicker mono">Results</div>
                        <h2 className="ops-panel-title mono" style={{ marginBottom: "1rem" }}>
                            {activeTab === "image" ? "Aircraft Assessment" : activeTab === "video" ? "Video Timeline" : "Live Feed"}
                        </h2>

                        {activeTab === "image" && imageResultMode === "none" ? (
                            <div style={{ minHeight: "520px", display: "grid", placeItems: "center", border: "1px dashed rgba(255,255,255,0.12)", color: "#4B5563" }}>
                                <div className="mono" style={{ fontSize: "0.68rem", textTransform: "uppercase" }}>
                                    {classifying || detecting ? "Analysis in progress..." : "Awaiting image upload"}
                                </div>
                            </div>
                        ) : null}

                        {activeTab === "image" && imageResultMode === "detect" && detectResult ? (
                            <div style={{ display: "grid", gap: "1rem" }}>
                                <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "1rem", borderRadius: 0 }}>
                                    <div className="mono" style={{ color: "#4B5563", fontSize: "0.6rem", textTransform: "uppercase" }}>
                                        Detected Aircraft: {detectResult.total_aircraft} — {detectResult.model_used}
                                    </div>
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                        src={`data:image/jpeg;base64,${detectResult.annotated_image_base64}`}
                                        alt="Annotated detections"
                                        style={{ width: "100%", marginTop: "0.75rem", display: "block", border: "1px solid rgba(255,255,255,0.08)" }}
                                    />
                                </div>

                                {detectResult.detections.length > 0 ? (
                                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "0.75rem" }}>
                                        {detectResult.detections.map((detection, index) => (
                                            <DetectionCard key={`${detection.class_name}-${index}`} detection={detection} />
                                        ))}
                                    </div>
                                ) : (
                                    <div className="mono" style={{ fontSize: "0.65rem", color: "#6B7280", textTransform: "uppercase" }}>
                                        No aircraft detected in image.
                                    </div>
                                )}
                            </div>
                        ) : null}

                        {activeTab === "image" && imageResultMode === "single" && singleResult ? (
                            <div style={{ display: "grid", gap: "1rem" }}>
                                <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "1rem", borderRadius: 0 }}>
                                    <div className="mono" style={{ color: "#4B5563", fontSize: "0.6rem", textTransform: "uppercase" }}>Predicted Class</div>
                                    <div style={{ marginTop: "0.5rem", fontSize: "1.85rem", fontWeight: 700 }}>{singleResult.class_name}</div>

                                    <div style={{ marginTop: "0.9rem" }}>
                                        <div className="mono" style={{ marginBottom: "0.35rem", fontSize: "0.6rem", color: "#4B5563", textTransform: "uppercase" }}>Confidence</div>
                                        <div style={{ height: "8px", background: "rgba(255,255,255,0.08)", position: "relative" }}>
                                            <div style={{ width: `${(singleResult.confidence * 100).toFixed(1)}%`, height: "100%", background: confidenceStyle(singleResult.confidence) }} />
                                        </div>
                                        <div className="mono" style={{ marginTop: "0.35rem", fontSize: "0.62rem", color: "var(--text-muted)" }}>
                                            {(singleResult.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>

                                    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.8rem", marginTop: "1rem", alignItems: "center" }}>
                                        <div className="mono" style={{ fontSize: "0.66rem", color: "var(--text-muted)" }}>
                                            {singleResult.origin_country} - {singleResult.friend_or_foe}
                                        </div>
                                        <FriendFoeBadge value={singleResult.friend_or_foe} />
                                    </div>

                                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
                                        {[
                                            ["Inference Time", singleResult.inference_time_ms ? `${singleResult.inference_time_ms.toFixed(1)} ms` : "--"],
                                            ["Model", singleResult.model_name ?? "convnext_small"],
                                            ["Device", singleResult.device_used ?? "--"],
                                        ].map(([label, value]) => (
                                            <div key={label as string} style={{ border: "1px solid rgba(255,255,255,0.08)", padding: "0.65rem", borderRadius: 0 }}>
                                                <div className="mono" style={{ fontSize: "0.56rem", color: "#4B5563", textTransform: "uppercase" }}>{label}</div>
                                                <div className="mono" style={{ marginTop: "0.35rem", fontSize: "0.66rem", color: "var(--text-primary)" }}>{value}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "1rem", borderRadius: 0 }}>
                                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", marginBottom: "0.8rem" }}>
                                        <div>
                                            <div className="ops-kicker mono">Attention Map</div>
                                            <div className="mono" style={{ fontSize: "0.62rem", color: "#4B5563", textTransform: "uppercase" }}>Grad-CAM explainability</div>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => setShowHeatmap((value) => !value)}
                                            className="mono"
                                            style={{ border: "1px solid rgba(255,255,255,0.12)", background: "transparent", color: "var(--text-primary)", padding: "0.4rem 0.65rem", cursor: "pointer", textTransform: "uppercase", fontSize: "0.58rem", borderRadius: 0 }}
                                        >
                                            {showHeatmap ? "Hide Heatmap" : "Show Heatmap"}
                                        </button>
                                    </div>
                                    <div style={{ position: "relative", minHeight: "300px", border: "1px solid rgba(255,255,255,0.08)", background: "rgba(0,0,0,0.3)", overflow: "hidden" }}>
                                        {previewUrl ? (
                                            <>
                                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                                <img src={previewUrl} alt="Original aircraft" style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }} />
                                                {showHeatmap && heatmapSrc ? (
                                                    // eslint-disable-next-line @next/next/no-img-element
                                                    <img
                                                        src={heatmapSrc}
                                                        alt="Grad-CAM heatmap"
                                                        style={{
                                                            position: "absolute",
                                                            inset: 0,
                                                            width: "100%",
                                                            height: "100%",
                                                            objectFit: "contain",
                                                            opacity: 0.7,
                                                            mixBlendMode: "multiply",
                                                            filter: "sepia(1) saturate(7) hue-rotate(-35deg) brightness(1.05)",
                                                        }}
                                                    />
                                                ) : null}
                                            </>
                                        ) : null}
                                        {gradcamLoading ? (
                                            <div style={{ position: "absolute", inset: 0, display: "grid", gap: "0.65rem", padding: "1rem" }}>
                                                {[0, 1, 2].map((key) => (
                                                    <div key={key} className="ops-stat-pulse" style={{ height: key === 0 ? "55%" : "14px", background: "rgba(255,255,255,0.06)" }} />
                                                ))}
                                            </div>
                                        ) : gradcamError ? (
                                            <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#4B5563" }}>
                                                <div className="mono" style={{ fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                                                    Attention map unavailable
                                                </div>
                                            </div>
                                        ) : null}
                                    </div>
                                </div>
                            </div>
                        ) : null}

                        {activeTab === "video" && !videoResult && !videoProcessing ? (
                            <div style={{ minHeight: "520px", display: "grid", placeItems: "center", border: "1px dashed rgba(255,255,255,0.12)", color: "#4B5563" }}>
                                <div className="mono" style={{ fontSize: "0.68rem", textTransform: "uppercase" }}>
                                    Awaiting video upload
                                </div>
                            </div>
                        ) : null}

                        {activeTab === "video" && videoProcessing ? (
                            <div style={{ minHeight: "520px", display: "grid", placeItems: "center", border: "1px dashed rgba(255,255,255,0.12)", color: "#4B5563" }}>
                                <div className="mono" style={{ fontSize: "0.68rem", textTransform: "uppercase" }}>
                                    Analysing video frames...
                                </div>
                            </div>
                        ) : null}

                        {activeTab === "video" && videoResult ? (
                            <div style={{ display: "grid", gap: "1rem" }}>
                                {videoSummary ? (
                                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "0.75rem" }}>
                                        {[
                                            ["Frames Analyzed", String(videoSummary.totalFrames)],
                                            ["Peak Aircraft Count", String(videoSummary.peakAircraft)],
                                            ["Most Common Class", videoSummary.mostCommonClass],
                                        ].map(([label, value]) => (
                                            <div key={label} style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "0.85rem", borderRadius: 0 }}>
                                                <div className="mono" style={{ color: "#4B5563", fontSize: "0.58rem", textTransform: "uppercase" }}>{label}</div>
                                                <div style={{ marginTop: "0.45rem", fontSize: "0.95rem", lineHeight: 1.35 }}>{value}</div>
                                            </div>
                                        ))}
                                    </div>
                                ) : null}

                                <div style={{ maxHeight: "680px", overflowY: "auto", display: "grid", gap: "0.85rem", paddingRight: "0.25rem" }}>
                                    {videoResult.frames.map((frame) => (
                                        <div
                                            key={`${frame.frame_number}-${frame.timestamp_sec}`}
                                            style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "0.85rem", borderRadius: 0 }}
                                        >
                                            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "center", marginBottom: "0.65rem" }}>
                                                <div className="mono" style={{ fontSize: "0.62rem", color: "#22D3EE", textTransform: "uppercase" }}>
                                                    t={frame.timestamp_sec.toFixed(2)}s — frame {frame.frame_number}
                                                </div>
                                                <div className="mono" style={{ fontSize: "0.62rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                                                    {frame.error ? "Error" : `${frame.total_aircraft ?? 0} aircraft`}
                                                </div>
                                            </div>
                                            {frame.annotated_frame_base64 ? (
                                                // eslint-disable-next-line @next/next/no-img-element
                                                <img
                                                    src={`data:image/jpeg;base64,${frame.annotated_frame_base64}`}
                                                    alt={`Frame ${frame.frame_number}`}
                                                    style={{ width: "100%", maxHeight: "220px", objectFit: "contain", background: "#000", display: "block" }}
                                                />
                                            ) : null}
                                            {frame.error ? (
                                                <div style={{ color: "#FCA5A5", fontSize: "0.82rem", marginTop: "0.5rem" }}>{frame.error}</div>
                                            ) : (
                                                <div style={{ marginTop: "0.65rem", display: "grid", gap: "0.35rem" }}>
                                                    {(frame.detections ?? []).map((detection, index) => (
                                                        <div key={`${frame.frame_number}-${index}`} className="mono" style={{ fontSize: "0.62rem", color: "var(--text-muted)" }}>
                                                            {detection.class_name} ({(detection.class_confidence * 100).toFixed(0)}%) — {detection.friend_or_foe}
                                                        </div>
                                                    ))}
                                                    {(frame.detections ?? []).length === 0 ? (
                                                        <div className="mono" style={{ fontSize: "0.62rem", color: "#6B7280", textTransform: "uppercase" }}>
                                                            No aircraft detected
                                                        </div>
                                                    ) : null}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : null}

                        {activeTab === "stream" ? (
                            <div style={{ display: "grid", gap: "1rem" }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem" }}>
                                    <div className="mono" style={{ fontSize: "0.62rem", letterSpacing: "0.14em", textTransform: "uppercase", color: streamStatus === "LIVE" ? "#22D3EE" : streamStatus === "CONNECTING" ? "#F59E0B" : "#6B7280" }}>
                                        {streamStatus}
                                    </div>
                                    <div className="mono" style={{ padding: "2px 8px", border: "1px solid rgba(255,255,255,0.18)", fontSize: "0.62rem", textTransform: "uppercase", borderRadius: 0 }}>
                                        Detections: {liveDetectionCount}
                                    </div>
                                </div>

                                <div style={{ height: "480px", background: "#000", border: "1px solid rgba(255,255,255,0.08)", display: "grid", placeItems: "center", overflow: "hidden", borderRadius: 0 }}>
                                    {currentStreamFrame?.annotated_frame_base64 ? (
                                        // eslint-disable-next-line @next/next/no-img-element
                                        <img
                                            src={`data:image/jpeg;base64,${currentStreamFrame.annotated_frame_base64}`}
                                            alt="Live stream frame"
                                            style={{ width: "100%", height: "100%", objectFit: "contain" }}
                                        />
                                    ) : (
                                        <div className="mono" style={{ fontSize: "0.68rem", color: "#4B5563", textTransform: "uppercase" }}>
                                            {streamStatus === "CONNECTING" ? "Connecting to stream..." : "No live frame yet"}
                                        </div>
                                    )}
                                </div>

                                <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "0.85rem", borderRadius: 0 }}>
                                    <div className="ops-kicker mono">Detection Log</div>
                                    <div style={{ marginTop: "0.75rem", maxHeight: "220px", overflowY: "auto", display: "grid", gap: "0.45rem" }}>
                                        {streamLog.length === 0 ? (
                                            <div className="mono" style={{ fontSize: "0.62rem", color: "#6B7280", textTransform: "uppercase" }}>
                                                Awaiting stream events
                                            </div>
                                        ) : (
                                            streamLog.map((entry) => (
                                                <div key={entry.id} className="mono" style={{ fontSize: "0.62rem", color: "var(--text-muted)", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "0.35rem" }}>
                                                    [{entry.timestamp.toFixed(2)}s] {entry.summary}
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </div>
                            </div>
                        ) : null}
                    </section>
                </div>

                <section className="glass-panel" style={{ marginTop: "1.5rem", padding: "1.25rem" }}>
                    <div className="ops-kicker mono">Model Information</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: "0.75rem", marginTop: "0.9rem" }}>
                        {[
                            ["Architecture", "ConvNeXt Small + YOLOv8n"],
                            ["Training Data", "FGVC Aircraft (100 classes)"],
                            ["Validation Accuracy", "72.5% top-1"],
                            ["Explainability", "Grad-CAM attention maps"],
                            ["Classes Include", "F-16A/B, F/A-18, Eurofighter Typhoon, C-130, Il-76, Tornado + 94 more"],
                        ].map(([label, value]) => (
                            <div key={label as string} style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "0.85rem", borderRadius: 0 }}>
                                <div className="mono" style={{ color: "#4B5563", fontSize: "0.58rem", textTransform: "uppercase" }}>{label}</div>
                                <div style={{ marginTop: "0.45rem", fontSize: "0.88rem", lineHeight: 1.45 }}>{value}</div>
                            </div>
                        ))}
                    </div>
                </section>
            </div>
        </div>
    );
}
