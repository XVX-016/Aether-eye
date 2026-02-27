import React, { useState } from "react";
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
  const v = value ?? internal;

  const setValue = (next: number) => {
    const clamped = Math.max(0, Math.min(100, next));
    if (onChange) onChange(clamped);
    else setInternal(clamped);
  };

  const ready = Boolean(beforeUrl && afterUrl);

  return (
    <ContentPanel title="Before / After" subtitle="Drag the slider to compare captures.">
      {!ready ? (
        <div className="empty-state">Upload both before and after images.</div>
      ) : (
        <div className="slider">
          <div className="slider-stage">
            <img className="slider-img" src={afterUrl!} alt="After" />
            <div className="slider-before" style={{ width: `${v}%` }}>
              <img className="slider-img" src={beforeUrl!} alt="Before" />
            </div>

            <div className="slider-handle" style={{ left: `${v}%` }}>
              <div className="slider-handle-line" />
              <div className="slider-handle-knob" />
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
