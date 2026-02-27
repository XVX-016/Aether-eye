import React from "react";
import type { MainSection } from "../lib/types";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

type Props = {
  activeSection: MainSection;
  onSectionChange: (section: MainSection) => void;
  onRunAircraft: () => void;
  onRunChange: () => void;
  canRunAircraft: boolean;
  canRunChange: boolean;
  loadingAircraft: boolean;
  loadingChange: boolean;
  error: string | null;
  children: React.ReactNode;
};

export const Layout: React.FC<Props> = ({
  activeSection,
  onSectionChange,
  onRunAircraft,
  onRunChange,
  canRunAircraft,
  canRunChange,
  loadingAircraft,
  loadingChange,
  error,
  children,
}) => {
  return (
    <div className="shell">
      <Sidebar active={activeSection} onChange={onSectionChange} />

      <div className="shell-main">
        <TopBar
          activeSection={activeSection}
          onRunAircraft={onRunAircraft}
          onRunChange={onRunChange}
          canRunAircraft={canRunAircraft}
          canRunChange={canRunChange}
          loadingAircraft={loadingAircraft}
          loadingChange={loadingChange}
          error={error}
        />

        <main className="app-main">{children}</main>
      </div>
    </div>
  );
};

