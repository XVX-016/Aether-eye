"use client";

import React, { useEffect, useMemo, useRef } from "react";
import type { AircraftDetection } from "@/lib/types";
import { ContentPanel } from "./ContentPanel";

type Props = {
    imageUrl: string | null;
    detections: AircraftDetection[];
    loading?: boolean;
    title?: string;
};

function clamp(n: number, min: number, max: number) {
    return Math.max(min, Math.min(max, n));
}

export const DetectionCanvas: React.FC<Props> = ({
    imageUrl,
    detections,
    loading = false,
    title = "Aircraft Detection",
}) => {
    const imgRef = useRef<HTMLImageElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const sorted = useMemo(
        () => [...detections].sort((a, b) => b.confidence - a.confidence),
        [detections],
    );

    const draw = () => {
        const img = imgRef.current;
        const canvas = canvasRef.current;
        if (!img || !canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const dispW = img.clientWidth;
        const dispH = img.clientHeight;
        canvas.width = Math.max(1, Math.floor(dispW));
        canvas.height = Math.max(1, Math.floor(dispH));

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const natW = img.naturalWidth || dispW;
        const natH = img.naturalHeight || dispH;

        const sx = canvas.width / natW;
        const sy = canvas.height / natH;

        const fontSize = clamp(Math.round(canvas.width / 60), 11, 16);
        ctx.font = `600 ${fontSize}px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`;
        ctx.textBaseline = "top";

        for (const det of sorted) {
            const x1 = det.bbox.x1 * sx;
            const y1 = det.bbox.y1 * sy;
            const x2 = det.bbox.x2 * sx;
            const y2 = det.bbox.y2 * sy;

            const w = Math.max(1, x2 - x1);
            const h = Math.max(1, y2 - y1);

            ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
            ctx.lineWidth = 1.5;
            ctx.strokeRect(x1, y1, w, h);

            const label = `AIRCRAFT ${(det.confidence * 100).toFixed(1)}%`;
            const pad = 6;
            const textW = ctx.measureText(label).width;
            const boxH = fontSize + pad * 2;
            const boxW = textW + pad * 2;

            ctx.fillStyle = "rgba(8, 8, 8, 0.85)";
            ctx.fillRect(x1, Math.max(0, y1 - boxH), boxW, boxH);
            ctx.fillStyle = "rgba(248, 250, 252, 0.9)";
            ctx.fillText(label, x1 + pad, Math.max(0, y1 - boxH) + pad);
        }
    };

    useEffect(() => {
        draw();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imageUrl, detections]);

    return (
        <ContentPanel
            title={title}
            subtitle="Upload an image to run YOLOv8 ONNX aircraft detection."
        >
            {!imageUrl ? (
                <div className="empty-state">Upload an image to see detections.</div>
            ) : (
                <div className="detector-stage">
                    <img
                        ref={imgRef}
                        className="detector-image"
                        src={imageUrl}
                        alt="Detection input"
                        onLoad={draw}
                    />
                    <canvas ref={canvasRef} className="detector-canvas" />
                    {loading && <div className="stage-badge">ANALYZING...</div>}
                </div>
            )}
        </ContentPanel>
    );
};
