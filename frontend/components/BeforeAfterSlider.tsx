"use client";

import React, { useRef, useState } from "react";
import { ContentPanel } from "./ContentPanel";

type Props = {
    beforeUrl: string | null;
    afterUrl: string | null;
    value?: number;
    onChange?: (value: number) => void;
};

export const BeforeAfterSlider: React.FC<Props> = ({
    beforeUrl,
    afterUrl,
    value,
    onChange,
}) => {
    const [internal, setInternal] = useState(50);
    const [dragging, setDragging] = useState(false);
    const [aspectRatio, setAspectRatio] = useState<string | null>(null);
    const stageRef = useRef<HTMLDivElement | null>(null);
    const v = value ?? internal;

    const setValue = (next: number) => {
        const clamped = Math.max(0, Math.min(100, next));
        if (onChange) onChange(clamped);
        else setInternal(clamped);
    };

    const ready = Boolean(beforeUrl && afterUrl);

    const updateFromEvent = (event: React.PointerEvent<HTMLDivElement>) => {
        const rect = stageRef.current?.getBoundingClientRect();
        if (!rect) return;
        const next = ((event.clientX - rect.left) / rect.width) * 100;
        setValue(next);
    };

    return (
        <ContentPanel title="Before / After" subtitle="Drag the slider to compare captures.">
            {!ready ? (
                <div className="empty-state empty-state-plain">Upload both before and after images.</div>
            ) : (
                <div className="slider">
                    <div
                        className={`slider-stage ${dragging ? "slider-stage-dragging" : ""}`}
                        ref={stageRef}
                        style={aspectRatio ? { aspectRatio } : undefined}
                        onPointerDown={(event) => {
                            setDragging(true);
                            updateFromEvent(event);
                            event.currentTarget.setPointerCapture(event.pointerId);
                        }}
                        onPointerMove={(event) => {
                            if (!dragging) return;
                            updateFromEvent(event);
                        }}
                        onPointerUp={(event) => {
                            setDragging(false);
                            event.currentTarget.releasePointerCapture(event.pointerId);
                        }}
                        onPointerCancel={(event) => {
                            setDragging(false);
                            event.currentTarget.releasePointerCapture(event.pointerId);
                        }}
                    >
                        <img
                            className="slider-img"
                            src={afterUrl!}
                            alt="After"
                            onLoad={(event) => {
                                const img = event.currentTarget;
                                if (img.naturalWidth && img.naturalHeight) {
                                    setAspectRatio(`${img.naturalWidth} / ${img.naturalHeight}`);
                                }
                            }}
                        />
                        <div className="slider-before" style={{ width: `${v}%` }}>
                            <img className="slider-img" src={beforeUrl!} alt="Before" />
                        </div>

                        <div className="slider-label slider-label-before">Before</div>
                        <div className="slider-label slider-label-after">After</div>
                        <div className="slider-handle" style={{ left: `${v}%` }}>
                            <div className="slider-handle-line" />
                            <div className="slider-handle-knob" />
                            <div className="slider-handle-value">{Math.round(v)}%</div>
                        </div>
                    </div>

                    <input
                        className="slider-range"
                        type="range"
                        min={0}
                        max={100}
                        value={v}
                        onChange={(e) => setValue(Number(e.target.value))}
                        aria-label="Before-after slider"
                    />
                </div>
            )}
        </ContentPanel>
    );
};
