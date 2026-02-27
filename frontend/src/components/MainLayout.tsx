import React from "react";
import type { MainSection } from "../lib/types";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

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

export const MainLayout: React.FC<Props> = ({
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
    <div className="main-layout">
      <Sidebar active={activeSection} onChange={onSectionChange} />
      <div className="main-area">
        <Topbar
          activeSection={activeSection}
          onRunAircraft={onRunAircraft}
          onRunChange={onRunChange}
          canRunAircraft={canRunAircraft}
          canRunChange={canRunChange}
          loadingAircraft={loadingAircraft}
          loadingChange={loadingChange}
          error={error}
        />
        <main className="main-content">{children}</main>
      </div>
    </div>
  );
};
