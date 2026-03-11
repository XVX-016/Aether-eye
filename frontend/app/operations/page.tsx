import { Navbar } from "@/components/Navbar";
import { OperationsDashboard } from "@/components/OperationsDashboard";

export default function OperationsPage() {
    return (
        <div className="app">
            <Navbar />
            <div className="home-body">
                <header className="capability-header">
                    <p className="capability-kicker mono">AETHER EYE</p>
                    <h1 className="capability-title">OPERATIONS DASHBOARD</h1>
                    <p className="capability-description">
                        Live intelligence events, ingestion status, and activity signals across monitored AOIs.
                    </p>
                </header>
                <OperationsDashboard />
            </div>
        </div>
    );
}
