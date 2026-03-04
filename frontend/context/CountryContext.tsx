"use client";

import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

export type CountryOption =
    | "USA"
    | "China"
    | "Russia"
    | "India"
    | "UK"
    | "France"
    | "Germany"
    | "Japan";

const COUNTRY_OPTIONS: CountryOption[] = [
    "USA",
    "China",
    "Russia",
    "India",
    "UK",
    "France",
    "Germany",
    "Japan",
];

type CountryContextValue = {
    country: CountryOption;
    setCountry: (country: CountryOption) => void;
    options: CountryOption[];
};

const CountryContext = createContext<CountryContextValue | undefined>(undefined);

function readSessionCountry(): CountryOption | null {
    if (typeof window === "undefined") return null;
    const raw = window.sessionStorage.getItem("aether-eye-country");
    if (!raw) return null;
    return COUNTRY_OPTIONS.includes(raw as CountryOption) ? (raw as CountryOption) : null;
}

export const CountryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [country, setCountry] = useState<CountryOption>(() => readSessionCountry() ?? "USA");

    useEffect(() => {
        if (typeof window === "undefined") return;
        window.sessionStorage.setItem("aether-eye-country", country);
    }, [country]);

    const value = useMemo(
        () => ({
            country,
            setCountry,
            options: COUNTRY_OPTIONS,
        }),
        [country],
    );

    return <CountryContext.Provider value={value}>{children}</CountryContext.Provider>;
};

export function useCountry(): CountryContextValue {
    const ctx = useContext(CountryContext);
    if (!ctx) {
        throw new Error("useCountry must be used within CountryProvider");
    }
    return ctx;
}
