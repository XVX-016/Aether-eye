"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
    { label: "HOME", href: "/" },
    { label: "AIRCRAFT INTELLIGENCE", href: "/aircraft-intelligence" },
    { label: "CHANGE INTELLIGENCE", href: "/change-intelligence" },
    { label: "OPERATIONS DASHBOARD", href: "/operations" },
];

export const Navbar: React.FC = () => {
    const pathname = usePathname();

    return (
        <nav className="home-nav">
            <div className="home-nav-inner">
                <Link
                    href="/"
                    className="home-nav-brand mono home-nav-brand-button"
                >
                    AETHER EYE
                </Link>
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
                                aria-current={pathname === item.href ? "page" : undefined}
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
