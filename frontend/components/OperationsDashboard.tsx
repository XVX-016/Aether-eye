"use client";

import React, { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import axios from 'axios';

interface IntelligenceEvent {
  event_id: string;
  type: string;
  lat: number;
  lon: number;
  confidence: number;
  aircraft_class?: string;
  timestamp: string;
  priority: string;
}

export const OperationsDashboard: React.FC = () => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [events, setEvents] = useState<IntelligenceEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<IntelligenceEvent | null>(null);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const response = await axios.post('http://localhost:8000/api/events', {
          image_path: 'mock_satellite_image.png'
        });
        setEvents(response.data);
      } catch (error) {
        console.error('Error fetching intelligence events:', error);
        const mockRes = await fetch('/mocks/events.json');
        if (mockRes.ok) {
            setEvents(await mockRes.json());
        }
      }
    };
    fetchEvents();
  }, []);

  useEffect(() => {
    if (mapContainer.current && !map.current) {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: 'https://demotiles.maplibre.org/style.json',
        center: [55.2708, 25.2048],
        zoom: 12
      });

      map.current.addControl(new maplibregl.NavigationControl(), 'top-right');
    }

    if (map.current && events.length > 0) {
      events.forEach((event) => {
        const color = event.priority === 'HIGH' ? '#ff4d4d' : '#ff9f43';
        new maplibregl.Marker({ color })
          .setLngLat([event.lon, event.lat])
          .setPopup(new maplibregl.Popup({ offset: 25 }).setHTML(`
            <div style="color: #020408; font-family: 'JetBrains Mono', monospace; padding: 4px;">
              <div style="font-weight: 700; font-size: 10px; margin-bottom: 4px; border-bottom: 1px solid #ccc;">${event.type.toUpperCase()}</div>
              <div style="font-size: 9px;">CONFIDENCE: ${(event.confidence * 100).toFixed(1)}%</div>
            </div>
          `))
          .addTo(map.current!);
      });
    }
  }, [events]);

  return (
    <div className="app-shell flex h-[calc(100vh-80px)] mt-[80px] overflow-hidden">
      {/* Left Sidebar: Event List */}
      <div className="w-[320px] bg-stealth-charcoal border-r border-border-strong flex flex-col glass">
        <div className="p-6 border-b border-border-strong">
          <h1 className="text-xl font-bold tracking-[0.2em] text-white mono uppercase">INTELLIGENCE</h1>
          <p className="text-[10px] text-text-muted mt-1 uppercase tracking-[0.3em] font-medium">Active Operations</p>
        </div>
        <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-3">
          {events.length > 0 ? events.map((event) => (
            <div
              key={event.event_id}
              onClick={() => {
                setSelectedEvent(event);
                map.current?.flyTo({ center: [event.lon, event.lat], zoom: 15 });
              }}
              className={`p-4 border transition-all duration-300 relative overflow-hidden group ${
                selectedEvent?.event_id === event.event_id
                  ? 'bg-white/5 border-white/20'
                  : 'bg-transparent border-white/5 hover:border-white/10 hover:bg-white/[0.02]'
              }`}
            >
              {selectedEvent?.event_id === event.event_id && (
                <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-white shadow-[0_0_10px_rgba(255,255,255,0.5)]" />
              )}
              <div className="flex justify-between items-start mb-2">
                <span className={`text-[9px] px-2 py-0.5 font-bold uppercase mono border ${
                  event.priority === 'HIGH' ? 'border-red-500/50 text-red-500 bg-red-500/5' : 'border-orange-500/50 text-orange-500 bg-orange-500/5'
                }`}>
                  {event.priority}
                </span>
                <span className="text-[9px] text-text-dim mono">{new Date(event.timestamp).toLocaleTimeString()}</span>
              </div>
              <h3 className="text-xs font-bold text-text-primary tracking-wider mono uppercase">{event.type.replace('_', ' ')}</h3>
              <p className="text-[10px] text-text-muted mt-1 mono">{event.aircraft_class || 'SIGNAL DETECTED'}</p>
            </div>
          )) : (
            <div className="flex flex-col items-center justify-center h-40 opacity-30">
              <div className="w-8 h-8 border border-text-dim rounded-full flex items-center justify-center mb-3 mono text-xs">?</div>
              <p className="text-[10px] mono uppercase tracking-widest text-text-dim">No Active Signals</p>
            </div>
          )}
        </div>
      </div>

      {/* Center: Map Area */}
      <div className="flex-1 relative bg-stealth-black">
        <div ref={mapContainer} className="absolute inset-0 grayscale contrast-[1.2] brightness-[0.8]" />
        
        {/* Map Top-Left Overlay: Location Data */}
        <div className="absolute top-6 left-6 z-10 glass-panel p-4 border border-border-strong mono">
          <div className="text-[9px] text-text-dim uppercase tracking-widest mb-1">Global Precision Array</div>
          <div className="text-xs text-text-primary">MULTISPECTRAL_SYNTHESIS_ACTIVE</div>
          <div className="flex gap-4 mt-2">
            <div>
              <div className="text-[8px] text-text-dim uppercase">SAT_ORBIT</div>
              <div className="text-[10px] text-text-primary">GEO_STATIONARY_01</div>
            </div>
            <div>
              <div className="text-[8px] text-text-dim uppercase">SENSOR_MODE</div>
              <div className="text-[10px] text-text-primary">SAR_INFRARED</div>
            </div>
          </div>
        </div>

        {/* Bottom Timeline Overlay */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 w-[600px] z-10">
            <div className="glass-panel p-5 border border-border-strong shadow-2xl overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                <div className="flex justify-between text-[9px] text-text-muted mb-3 uppercase tracking-[0.4em] mono font-bold">
                    <span>Baseline</span>
                    <span className="text-white">Live Intelligence</span>
                </div>
                <div className="relative h-6 flex items-center">
                    <div className="absolute inset-0 bg-white/5 rounded-full" />
                    <input type="range" className="absolute inset-0 opacity-0 cursor-pointer z-20 w-full h-full" min="0" max="100" defaultValue="100" />
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 h-0.5 bg-white/20 w-full rounded-full" />
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 h-0.5 bg-white w-full rounded-full" />
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-[0_0_10px_white] z-10" />
                </div>
                <div className="flex justify-between mt-3 px-1">
                    {[0,1,2,3,4,5,6,7].map(i => (
                        <div key={i} className="w-[1px] h-1 bg-white/20" />
                    ))}
                </div>
            </div>
        </div>
      </div>

      {/* Right Sidebar: Inspector */}
      <div className="w-[400px] bg-stealth-charcoal border-l border-border-strong glass p-8 overflow-y-auto custom-scrollbar">
        {selectedEvent ? (
          <div className="animate-in fade-in slide-in-from-right-4 duration-500">
            <div className="flex items-center justify-between mb-8">
                <h2 className="text-sm font-bold text-white mono tracking-[0.3em] uppercase">Signal Inspector</h2>
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_red]" />
            </div>

            <div className="aspect-video bg-stealth-carbon rounded-sm flex items-center justify-center border border-border-strong mb-8 group relative overflow-hidden">
              <div className="absolute inset-0 opacity-20 pointer-events-none bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:16px_16px]" />
              <div className="z-10 text-center px-6">
                <div className="text-text-dim text-[8px] mono uppercase tracking-widest mb-2">Imagery Stream 01-X</div>
                <div className="text-[10px] text-text-primary mono border border-white/10 px-3 py-1 bg-black/40">
                    LAT: {selectedEvent.lat.toFixed(4)} LN: {selectedEvent.lon.toFixed(4)}
                </div>
              </div>
              <div className="absolute bottom-2 right-2 text-[8px] mono text-text-dim uppercase">ENCRYPTION: AES_256</div>
            </div>
            
            <div className="space-y-8">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-1">
                    <label className="text-[9px] text-text-dim uppercase tracking-widest mono">Identifier</label>
                    <p className="text-xs text-text-primary mono">{selectedEvent.event_id}</p>
                </div>
                <div className="space-y-1">
                    <label className="text-[9px] text-text-dim uppercase tracking-widest mono">Priority</label>
                    <p className={`text-xs font-bold mono ${selectedEvent.priority === 'HIGH' ? 'text-red-500' : 'text-orange-500'}`}>{selectedEvent.priority}</p>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[9px] text-text-dim uppercase tracking-widest mono">Classification Confidence</label>
                <div className="flex items-center gap-4">
                  <div className="flex-1 h-[2px] bg-white/5 relative">
                    <div className="absolute top-0 left-0 h-full bg-white shadow-[0_0_10px_white]" style={{ width: `${selectedEvent.confidence * 100}%` }} />
                  </div>
                  <span className="text-[10px] font-bold mono text-white">{(selectedEvent.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="bg-white/[0.03] p-4 border border-white/5 mono">
                 <div className="text-[9px] text-text-dim uppercase mb-3">Raw Metadata Buffer</div>
                 <pre className="text-[9px] text-text-muted leading-relaxed overflow-x-hidden">
                    {JSON.stringify(selectedEvent, null, 2)}
                 </pre>
              </div>
            </div>

            <button className="w-full mt-10 py-4 bg-white text-black font-bold tracking-[0.4em] transition-all hover:bg-white/90 active:scale-[0.98] mono text-[10px] uppercase">
              Authenticate & Export
            </button>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 border border-border-strong rounded-full flex items-center justify-center mb-6 opacity-20 relative">
                <div className="absolute inset-0 rounded-full border border-white/5 animate-ping" />
                <span className="mono text-xl">!</span>
            </div>
            <h3 className="text-xs font-bold text-white mono tracking-widest uppercase mb-2">No Signal Selected</h3>
            <p className="text-[10px] text-text-dim mono uppercase leading-loose max-w-[200px]">Intercept a packet from the primary datastream to initiate inspection protocol.</p>
          </div>
        )}
      </div>

      <style jsx global>{`
        .mono { font-family: 'JetBrains Mono', monospace; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
        
        .mapboxgl-popup-content, .maplibregl-popup-content {
          background: rgba(255, 255, 255, 0.9) !important;
          border-radius: 0 !important;
          padding: 10px !important;
        }
        .mapboxgl-popup-tip, .maplibregl-popup-tip {
          border-top-color: rgba(255, 255, 255, 0.9) !important;
        }
      `}</style>
    </div>
  );
};
