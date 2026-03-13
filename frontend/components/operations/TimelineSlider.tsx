"use client";

import { useEffect, useMemo, useState } from "react";

type Props = {
    onChange: (start: Date, end: Date) => void;
    value?: {
        start: Date;
        end: Date;
    };
};

const DAYS_WINDOW = 14;

function startOfDay(date: Date) {
    const next = new Date(date);
    next.setHours(0, 0, 0, 0);
    return next;
}

function endOfDay(date: Date) {
    const next = new Date(date);
    next.setHours(23, 59, 59, 999);
    return next;
}

function formatRange(start: Date, end: Date) {
    return `${start.toLocaleDateString(undefined, { month: "short", day: "numeric" })} -> ${end.toLocaleDateString(undefined, { month: "short", day: "numeric" })}`;
}

export function TimelineSlider({ onChange, value }: Props) {
    const dateScale = useMemo(() => {
        const today = startOfDay(new Date());
        return Array.from({ length: DAYS_WINDOW }, (_, index) => {
            const point = new Date(today);
            point.setDate(today.getDate() - (DAYS_WINDOW - 1 - index));
            return startOfDay(point);
        });
    }, []);

    const [startIndex, setStartIndex] = useState(0);
    const [endIndex, setEndIndex] = useState(DAYS_WINDOW - 1);

    useEffect(() => {
        if (!value) {
            return;
        }

        const nextStart = dateScale.findIndex((date) => date.getTime() === startOfDay(value.start).getTime());
        const nextEnd = dateScale.findIndex((date) => date.getTime() === startOfDay(value.end).getTime());

        if (nextStart >= 0 && nextStart !== startIndex) {
            setStartIndex(nextStart);
        }
        if (nextEnd >= 0 && nextEnd !== endIndex) {
            setEndIndex(nextEnd);
        }
    }, [dateScale, endIndex, startIndex, value]);

    const startDate = dateScale[startIndex];
    const endDate = dateScale[endIndex];

    return (
        <div className="ops-timeline">
            <div className="ops-timeline-label mono">{formatRange(startDate, endDate)}</div>
            <div className="ops-timeline-inputs">
                <input
                    type="range"
                    min={0}
                    max={DAYS_WINDOW - 1}
                    value={startIndex}
                    onChange={(event) => {
                        const next = Number(event.target.value);
                        const nextIndex = Math.min(next, endIndex);
                        setStartIndex(nextIndex);
                        onChange(dateScale[nextIndex], endOfDay(dateScale[endIndex]));
                    }}
                />
                <input
                    type="range"
                    min={0}
                    max={DAYS_WINDOW - 1}
                    value={endIndex}
                    onChange={(event) => {
                        const next = Number(event.target.value);
                        const nextIndex = Math.max(next, startIndex);
                        setEndIndex(nextIndex);
                        onChange(dateScale[startIndex], endOfDay(dateScale[nextIndex]));
                    }}
                />
            </div>
        </div>
    );
}
