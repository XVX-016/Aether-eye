"use client";

import { useEffect, useMemo, useState } from "react";

import { Navbar } from "@/components/Navbar";
import { runAircraftClassification, runAircraftGradCam } from "@/lib/api";
import type { AircraftClassificationResponse, AircraftGradCamResponse } from "@/lib/types";

const COUNTRIES = ["USA", "China", "Russia", "India", "UK", "France", "Germany", "Japan"] as const;

function friendFoeStyle(value: string) {
    if (value === "FRIEND") {
        return { border: "1px solid #164E63", color: "#22D3EE" };
    }
    if (value === "FOE") {
        return { border: "1px solid #7F1D1D", color: "#EF4444" };
    }
    return { border: "1px solid #374151", color: "#6B7280" };
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

export default function AircraftIntelligencePage() {
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [country, setCountry] = useState<(typeof COUNTRIES)[number]>("USA");
    const [result, setResult] = useState<AircraftClassificationResponse | null>(null);
    const [gradcam, setGradcam] = useState<AircraftGradCamResponse | null>(null);
    const [classifying, setClassifying] = useState(false);
    const [gradcamLoading, setGradcamLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [classifierOnline, setClassifierOnline] = useState(false);

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

    const heatmapSrc = useMemo(() => {
        if (!gradcam?.heatmap_base64_png) {
            return null;
        }
        return `data:image/png;base64,${gradcam.heatmap_base64_png}`;
    }, [gradcam]);

    async function analyse() {
        if (!file) {
            return;
        }
        setClassifying(true);
        setGradcam(null);
        setResult(null);
        setError(null);
        try {
            const classification = await runAircraftClassification(file, country);
            setResult(classification);
            setGradcamLoading(true);
            try {
                const heatmap = await runAircraftGradCam(file, country);
                setGradcam(heatmap);
            } catch (gradcamError: any) {
                setError(gradcamError?.response?.data?.detail ?? "Grad-CAM generation failed.");
            } finally {
                setGradcamLoading(false);
            }
        } catch (classificationError: any) {
            setError(classificationError?.response?.data?.detail ?? "Classification failed.");
        } finally {
            setClassifying(false);
        }
    }

    function onFileSelected(nextFile: File | null) {
        setFile(nextFile);
        setResult(null);
        setGradcam(null);
        setError(null);
    }

    function onDrop(event: React.DragEvent<HTMLLabelElement>) {
        event.preventDefault();
        const dropped = event.dataTransfer.files?.[0] ?? null;
        if (dropped) {
            onFileSelected(dropped);
        }
    }

    return (
        <div className="app">
            <Navbar />
            <div className="home-body">
                <header className="capability-header" style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "flex-start", flexWrap: "wrap" }}>
                    <div>
                        <p className="capability-kicker mono">AETHER EYE</p>
                        <h1 className="capability-title">AIRCRAFT INTELLIGENCE</h1>
                        <p className="capability-description">
                            Fine-grained aircraft classification across 100 aircraft types with explainability
                        </p>
                        <div className="mono" style={{ marginTop: "0.75rem", color: "#4B5563", fontSize: "0.65rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                            ConvNeXt Small - 100 classes - 72.5% top-1
                        </div>
                    </div>
                    <div className="mono" style={{ padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.12em", textTransform: "uppercase", borderRadius: 2, ...modelStatusStyle(classifierOnline) }}>
                        {classifierOnline ? "Classifier Online" : "Classifier Offline"}
                    </div>
                </header>

                <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 0.95fr) minmax(380px, 1.15fr)", gap: "1.5rem" }}>
                    <section className="glass-panel" style={{ padding: "1.25rem" }}>
                        <div className="ops-kicker mono">Upload Image</div>
                        <h2 className="ops-panel-title mono" style={{ marginBottom: "1rem" }}>Classification Input</h2>
                        <label
                            onDrop={onDrop}
                            onDragOver={(event) => event.preventDefault()}
                            style={{
                                display: "grid",
                                placeItems: "center",
                                minHeight: "340px",
                                border: "1px dashed rgba(255,255,255,0.14)",
                                background: "rgba(255,255,255,0.02)",
                                cursor: "pointer",
                                overflow: "hidden",
                            }}
                        >
                            <input
                                type="file"
                                accept="image/jpeg,image/png"
                                style={{ display: "none" }}
                                onChange={(event) => onFileSelected(event.target.files?.[0] ?? null)}
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
                                    borderRadius: 2,
                                }}
                            >
                                {COUNTRIES.map((option) => (
                                    <option key={option} value={option}>
                                        {option}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <button
                            type="button"
                            onClick={analyse}
                            disabled={!file || classifying}
                            className="mono"
                            style={{
                                width: "100%",
                                marginTop: "1rem",
                                border: "1px solid rgba(255,255,255,0.18)",
                                background: "rgba(255,255,255,0.03)",
                                color: "var(--text-primary)",
                                padding: "0.9rem 1rem",
                                textTransform: "uppercase",
                                letterSpacing: "0.22em",
                                cursor: !file || classifying ? "not-allowed" : "pointer",
                                opacity: !file || classifying ? 0.5 : 1,
                            }}
                        >
                            {classifying ? "Analysing..." : "Analyse"}
                        </button>

                        {error ? (
                            <div style={{ marginTop: "1rem", border: "1px solid #7F1D1D", color: "#FCA5A5", padding: "0.8rem", fontSize: "0.82rem" }}>
                                {error}
                            </div>
                        ) : null}
                    </section>

                    <section className="glass-panel" style={{ padding: "1.25rem" }}>
                        <div className="ops-kicker mono">Results</div>
                        <h2 className="ops-panel-title mono" style={{ marginBottom: "1rem" }}>Aircraft Assessment</h2>

                        {!result ? (
                            <div style={{ minHeight: "520px", display: "grid", placeItems: "center", border: "1px dashed rgba(255,255,255,0.12)", color: "#4B5563" }}>
                                <div className="mono" style={{ fontSize: "0.68rem", textTransform: "uppercase" }}>
                                    {classifying ? "Classification in progress..." : "Awaiting image upload"}
                                </div>
                            </div>
                        ) : (
                            <div style={{ display: "grid", gap: "1rem" }}>
                                <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "1rem" }}>
                                    <div className="mono" style={{ color: "#4B5563", fontSize: "0.6rem", textTransform: "uppercase" }}>Predicted Class</div>
                                    <div style={{ marginTop: "0.5rem", fontSize: "1.85rem", fontWeight: 700 }}>{result.class_name}</div>

                                    <div style={{ marginTop: "0.9rem" }}>
                                        <div className="mono" style={{ marginBottom: "0.35rem", fontSize: "0.6rem", color: "#4B5563", textTransform: "uppercase" }}>Confidence</div>
                                        <div style={{ height: "8px", background: "rgba(255,255,255,0.08)", position: "relative" }}>
                                            <div style={{ width: `${(result.confidence * 100).toFixed(1)}%`, height: "100%", background: confidenceStyle(result.confidence) }} />
                                        </div>
                                        <div className="mono" style={{ marginTop: "0.35rem", fontSize: "0.62rem", color: "var(--text-muted)" }}>
                                            {(result.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>

                                    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.8rem", marginTop: "1rem", alignItems: "center" }}>
                                        <div className="mono" style={{ fontSize: "0.66rem", color: "var(--text-muted)" }}>
                                            {result.origin_country} - {result.friend_or_foe}
                                        </div>
                                        <div className="mono" style={{ padding: "1px 6px", fontSize: "0.6rem", letterSpacing: "0.12em", textTransform: "uppercase", borderRadius: 2, ...friendFoeStyle(result.friend_or_foe) }}>
                                            {result.friend_or_foe}
                                        </div>
                                    </div>

                                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
                                        {[
                                            ["Inference Time", result.inference_time_ms ? `${result.inference_time_ms.toFixed(1)} ms` : "--"],
                                            ["Model", result.model_name ?? "convnext_small"],
                                            ["Device", result.device_used ?? "--"],
                                        ].map(([label, value]) => (
                                            <div key={label as string} style={{ border: "1px solid rgba(255,255,255,0.08)", padding: "0.65rem" }}>
                                                <div className="mono" style={{ fontSize: "0.56rem", color: "#4B5563", textTransform: "uppercase" }}>{label}</div>
                                                <div className="mono" style={{ marginTop: "0.35rem", fontSize: "0.66rem", color: "var(--text-primary)" }}>{value}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "1rem" }}>
                                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", marginBottom: "0.8rem" }}>
                                        <div>
                                            <div className="ops-kicker mono">Attention Map</div>
                                            <div className="mono" style={{ fontSize: "0.62rem", color: "#4B5563", textTransform: "uppercase" }}>Grad-CAM explainability</div>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => setShowHeatmap((value) => !value)}
                                            className="mono"
                                            style={{ border: "1px solid rgba(255,255,255,0.12)", background: "transparent", color: "var(--text-primary)", padding: "0.4rem 0.65rem", cursor: "pointer", textTransform: "uppercase", fontSize: "0.58rem" }}
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
                                        ) : null}
                                    </div>
                                </div>

                                <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "1rem" }}>
                                    <div className="ops-kicker mono">Alternative Classifications</div>
                                    <div className="mono" style={{ marginTop: "0.55rem", color: "#4B5563", fontSize: "0.62rem", textTransform: "uppercase" }}>
                                        Top-5 breakdown available in next release
                                    </div>
                                </div>
                            </div>
                        )}
                    </section>
                </div>

                <section className="glass-panel" style={{ marginTop: "1.5rem", padding: "1.25rem" }}>
                    <div className="ops-kicker mono">Model Information</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: "0.75rem", marginTop: "0.9rem" }}>
                        {[
                            ["Architecture", "ConvNeXt Small"],
                            ["Training Data", "FGVC Aircraft (100 classes)"],
                            ["Validation Accuracy", "72.5% top-1"],
                            ["Explainability", "Grad-CAM attention maps"],
                            ["Classes Include", "F-16A/B, F/A-18, Eurofighter Typhoon, C-130, Il-76, Tornado + 94 more"],
                        ].map(([label, value]) => (
                            <div key={label as string} style={{ border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.02)", padding: "0.85rem" }}>
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
