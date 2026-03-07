import React, { useEffect, useRef, useState } from 'react';
import Head from 'next/head';
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

const OperationsDashboard: React.FC = () => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [events, setEvents] = useState<IntelligenceEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<IntelligenceEvent | null>(null);

  useEffect(() => {
    // Load real events from API
    const fetchEvents = async () => {
      try {
        const response = await axios.post('http://localhost:8000/api/events', {
          image_path: 'mock_satellite_image.png' // Trigger intelligence synthesis
        });
        setEvents(response.data);
      } catch (error) {
        console.error('Error fetching intelligence events:', error);
        // Fallback to mock if API fails during dev
        const mockRes = await fetch('/mocks/events.json');
        setEvents(await mockRes.json());
      }
    };
    fetchEvents();
  }, []);

  useEffect(() => {
    if (mapContainer.current && !map.current) {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: 'https://demotiles.maplibre.org/style.json', // Basic open style
        center: [55.2708, 25.2048], // Dubai default
        zoom: 12,
      });

      map.current.addControl(new maplibregl.NavigationControl());
    }

    // Add markers for events
    if (map.current && events.length > 0) {
      events.forEach((event) => {
        new maplibregl.Marker({ color: event.priority === 'HIGH' ? '#ff0000' : '#ffa500' })
          .setLngLat([event.lon, event.lat])
          .setPopup(new maplibregl.Popup().setHTML(`<b>${event.type}</b><br/>Conf: ${event.confidence}`))
          .addTo(map.current!);
      });
    }
  }, [events]);

  return (
    <div className="flex h-screen bg-slate-900 text-slate-100 font-sans">
      <Head>
        <title>Aether-Eye | Operations Dashboard</title>
      </Head>

      {/* Left Sidebar: Event List */}
      <div className="w-80 border-r border-slate-700 flex flex-col">
        <div className="p-4 border-b border-slate-700">
          <h1 className="text-xl font-bold tracking-tight text-white">INTELLIGENCE EVENTS</h1>
          <p className="text-xs text-slate-400 mt-1 uppercase tracking-widest">Active Operations</p>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-2">
          {events.map((event) => (
            <div
              key={event.event_id}
              onClick={() => {
                setSelectedEvent(event);
                map.current?.flyTo({ center: [event.lon, event.lat], zoom: 15 });
              }}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                selectedEvent?.event_id === event.event_id
                  ? 'bg-blue-600/20 border-blue-500'
                  : 'bg-slate-800/50 border-slate-700 hover:border-slate-500'
              }`}
            >
              <div className="flex justify-between items-start">
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase ${
                  event.priority === 'HIGH' ? 'bg-red-500 text-white' : 'bg-orange-500 text-white'
                }`}>
                  {event.priority}
                </span>
                <span className="text-[10px] text-slate-500">{new Date(event.timestamp).toLocaleTimeString()}</span>
              </div>
              <h3 className="text-sm font-semibold mt-1 text-slate-200">{event.type.replace('_', ' ')}</h3>
              <p className="text-xs text-slate-400 mt-0.5">{event.aircraft_class || 'Static Infrastructure'}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Center: Map Area */}
      <div className="flex-1 relative">
        <div ref={mapContainer} className="absolute inset-0" />
        
        {/* Bottom Timeline Overlay */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-2/3 bg-slate-900/80 backdrop-blur-md border border-slate-700 p-4 rounded-xl shadow-2xl">
          <div className="flex justify-between text-[10px] text-slate-400 mb-2 uppercase tracking-widest font-bold">
            <span>Historical Baseline</span>
            <span>Current Intelligence</span>
          </div>
          <input type="range" className="w-full accent-blue-500" min="0" max="100" />
        </div>
      </div>

      {/* Right Sidebar: Inspector */}
      <div className="w-96 border-l border-slate-700 bg-slate-900/50 backdrop-blur-sm p-6">
        {selectedEvent ? (
          <div>
            <h2 className="text-lg font-bold text-white mb-4">EVENT INSPECTOR</h2>
            <div className="aspect-video bg-slate-800 rounded-lg flex items-center justify-center border border-slate-700 mb-4">
              <span className="text-slate-500 text-xs text-center px-4">Satellite Imagery Preview<br/>(Region: {selectedEvent.lat.toFixed(4)}, {selectedEvent.lon.toFixed(4)})</span>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-[10px] text-slate-500 uppercase font-bold">Event Type</label>
                <p className="text-sm text-slate-200">{selectedEvent.type}</p>
              </div>
              <div>
                <label className="text-[10px] text-slate-500 uppercase font-bold">Confidence Score</label>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500" style={{ width: `${selectedEvent.confidence * 100}%` }} />
                  </div>
                  <span className="text-xs font-mono">{(selectedEvent.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Latitude</label>
                  <p className="text-sm font-mono text-slate-200">{selectedEvent.lat.toFixed(6)}</p>
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Longitude</label>
                  <p className="text-sm font-mono text-slate-200">{selectedEvent.lon.toFixed(6)}</p>
                </div>
              </div>
            </div>

            <button className="w-full mt-8 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg transition-colors text-sm">
              GENERATE REPORT
            </button>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center opacity-50">
            <div className="w-12 h-12 border-2 border-slate-700 rounded-full flex items-center justify-center mb-4">
              <i className="text-xl">!</i>
            </div>
            <p className="text-sm text-slate-400">Select an event from the timeline<br/>to inspect details.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default OperationsDashboard;
