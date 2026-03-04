"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
    { label: "HOME", href: "/" },
    { label: "AIRCRAFT SURVEILLANCE", href: "/detection" },
    { label: "AIRCRAFT RECOGNITION", href: "/recognition" },
    { label: "CHANGE INTELLIGENCE", href: "/change-intelligence" },
    { label: "OPERATIONS DASHBOARD", href: "/operations" },
];

export const Navbar: React.FC = () => {
    const pathname = usePathname();

    return (
        <nav className="home-nav">
            <div className="home-nav-inner">
                <div className="home-nav-brand mono">
                    AETHER EYE
                </div>
                <ul className="home-nav-links">
                    {NAV_ITEMS.map((item) => (
                        <li key={item.href}>
                            <Link
                                href={item.href}
                                className={
                                    pathname === item.href
                                        ? "home-nav-link home-nav-link-active"
                                        : "home-nav-link"
                                }
                            >
                                {item.label}
                            </Link>
                        </li>
                    ))}
                </ul>
            </div>
        </nav>
    );
};
