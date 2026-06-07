import { useState, useEffect } from 'react';
import { 
  Activity, 
  Battery, 
  CloudRain, 
  Cpu, 
  Power, 
  ShieldAlert, 
  Sun, 
  TrendingUp, 
  Wind, 
  Zap 
} from 'lucide-react';
import './index.css';
import AnalyticsDashboard from './AnalyticsDashboard';
import DecisionPanel from './DecisionPanel';

// ── Configuration ──────────────────────────────────────────
const GRAFANA_URL = "http://localhost/grafana/d-solo/ad6drfn/micro-grid";
const ORG_ID = 1;
const GRAFANA_FROM = "now-24h";
const GRAFANA_TO = "now";
const GRAFANA_REFRESH = "1s";

const PANELS: Record<string, string> = {
  solar: "panel-1",
  wind: "panel-2",
  battery: "panel-3",
  criticalLoad: "panel-4",
  nonCriticalLoad: "panel-5",
  gridSupply: "panel-6",
  inverterOutput: "panel-7"
};

import React from 'react';

class ErrorBoundary extends React.Component<any, any> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '20px', color: 'red', background: 'black', height: '100vh' }}>
          <h2>Dashboard Crash!</h2>
          <pre>{this.state.error?.toString()}</pre>
          <pre>{this.state.error?.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

function GrafanaPanel({ title, panelId, icon: Icon }: { title: string, panelId: string, icon: any }) {
  const src = `${GRAFANA_URL}?orgId=${ORG_ID}&from=${GRAFANA_FROM}&to=${GRAFANA_TO}&timezone=browser&refresh=${GRAFANA_REFRESH}&theme=dark&panelId=${panelId}`;
  return (
    <div className="glass-card fade-in">
      <div className="card-header">
        <Icon size={18} className="brand-icon" />
        {title}
      </div>
      <div className="grafana-embed">
        <iframe src={src} title={title} />
      </div>
    </div>
  );
}

function App() {
  const [telemetry, setTelemetry] = useState<any>(null);
  const [weather, setWeather] = useState<any>(null);
  const [forecast, setForecast] = useState<any>(null);

  // Poll for latest telemetry every 2 seconds
  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const res = await fetch('/api/telemetry/latest');
        if (res.ok) setTelemetry(await res.json());
      } catch (err) {
        console.error("Failed to fetch telemetry", err);
      }
    };
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2000);
    return () => clearInterval(interval);
  }, []);

  // Poll for weather and forecast every 10 seconds
  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const [weatherRes, forecastRes] = await Promise.all([
          fetch('/weather'),
          fetch('/forecast')
        ]);
        if (weatherRes.ok) setWeather(await weatherRes.json());
        if (forecastRes.ok) setForecast(await forecastRes.json());
      } catch (err) {
        console.error("Failed to fetch ML predictions", err);
      }
    };
    fetchPredictions();
    const interval = setInterval(fetchPredictions, 10000);
    return () => clearInterval(interval);
  }, []);

  const [currentView, setCurrentView] = useState<'control' | 'analytics' | 'decision'>('control');

  const info = telemetry?.info || {};
  const isAnomaly = forecast?.predictions?.theft_prediction?.anomaly === 1;
  const isTheftWarning = isAnomaly;

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="top-bar">
        <div className="brand">
          <Activity size={24} className="brand-icon" />
          Microgrid Control Center
        </div>
        <div className="nav-tabs" style={{ display: 'flex', gap: '1rem', marginLeft: 'auto', marginRight: '2rem' }}>
          <button 
            className={`nav-tab ${currentView === 'control' ? 'active' : ''}`} 
            onClick={() => setCurrentView('control')}
          >
            Control Center
          </button>
          { /*<button 
            className={`nav-tab ${currentView === 'analytics' ? 'active' : ''}`} 
            onClick={() => setCurrentView('analytics')}
          >
            Analytics
          </button> */}
          <button 
            className={`nav-tab ${currentView === 'decision' ? 'active' : ''}`} 
            onClick={() => setCurrentView('decision')}
          >
            Decision Panel
          </button>
        </div>
        <div className={`status-badge ${!telemetry ? 'offline' : ''}`}>
          <div className="status-dot"></div>
          {telemetry ? 'System Online' : 'Connecting...'}
        </div>
      </header>

      <main className="main-content">

        {currentView === 'control' ? (
          <>
            <div className="section-title" style={{ marginTop: '0px' }}>
              <TrendingUp size={20} /> Renewables & Storage
            </div>

            <div className="grid-grafana">
              <GrafanaPanel title="Solar Power" panelId={PANELS.solar} icon={Sun} />
              <GrafanaPanel title="Wind Power" panelId={PANELS.wind} icon={Wind} />
              <GrafanaPanel title="Battery SOC" panelId={PANELS.battery} icon={Battery} />
            </div>

            <div className="section-title" style={{ marginTop: '16px' }}>
              <Activity size={20} /> Loads & Inverter
            </div>

            <div className="grid-grafana-bottom">
              <GrafanaPanel title="Critical Load" panelId={PANELS.criticalLoad} icon={Cpu} />
              <GrafanaPanel title="Non-Critical Load" panelId={PANELS.nonCriticalLoad} icon={Power} />
              <GrafanaPanel title="Grid Supply" panelId={PANELS.gridSupply} icon={Zap} />
              <GrafanaPanel title="Inverter Output" panelId={PANELS.inverterOutput} icon={Activity} />
            </div>
            
            <div className="section-title" style={{ marginTop: '32px' }}>
              <Activity size={20} /> System Intelligence & Status
            </div>

            {/* Info Section moved to Bottom */}
            <div className="grid-systems">
              
              {/* Grid Details (from 'info') */}
              <div className="glass-card fade-in" style={{ animationDelay: '0.1s' }}>
                <div className="card-header">
                  <Cpu size={18} className="brand-icon" />
                  Controller States
                </div>
                <div className="details-list">
                  <div className="detail-item">
                    <div className="detail-label"><Power size={16} /> Grid Available</div>
                    <div className={`detail-value ${info.grid_available ? 'val-on' : 'val-off'}`}>
                      {info.grid_available ? 'YES' : 'NOT AVAILABLE'}
                    </div>
                  </div>
                  <div className="detail-item">
                    <div className="detail-label"><Zap size={16} /> Using Inverter</div>
                    <div className={`detail-value ${info.using_inverter ? 'val-warning' : 'val-off'}`}>
                      {info.using_inverter ? 'ACTIVE' : 'STANDBY'}
                    </div>
                  </div>
                  <div className="detail-item">
                    <div className="detail-label"><ShieldAlert size={16} /> Critical on Inverter</div>
                    <div className={`detail-value ${info.critical_on_inverter ? 'val-warning' : 'val-off'}`}>
                      {info.critical_on_inverter ? 'YES' : 'NO'}
                    </div>
                  </div>
                  <div className="detail-item">
                    <div className="detail-label"><Battery size={16} /> Non-Critical on Inverter</div>
                    <div className={`detail-value ${info.non_critical_on_inverter ? 'val-warning' : 'val-off'}`}>
                      {info.non_critical_on_inverter ? 'YES' : 'NO'}
                    </div>
                  </div>
                </div>
              </div>

              {/* AI Theft Detection */}
              <div className="glass-card fade-in" style={{ animationDelay: '0.2s' }}>
                <div className="card-header">
                  <ShieldAlert size={18} color={isTheftWarning ? "var(--danger-color)" : "var(--success-color)"} />
                  Theft Detection Model
                </div>
                {forecast ? (
                  <div className="value-block" style={{ marginTop: 'auto', marginBottom: 'auto', textAlign: 'center' }}>
                    <div className="val" style={{ 
                      color: isTheftWarning ? 'var(--danger-color)' : 'var(--success-color)',
                      fontSize: '2rem',
                      letterSpacing: '1px'
                    }}>
                      {isTheftWarning ? "SUSPICIOUS" : "SECURE"}
                    </div>
                    <div className="weather-desc">Grid Behavior Status</div>
                    <div className="metric-trend" style={{ justifyContent: 'center', marginTop: '12px' }}>
                      {isTheftWarning ? "⚠️ Warning: Potential Theft" : "✅ Grid is Secure"}
                    </div>
                  </div>
                ) : (
                  <div className="loading-state">Loading AI models...</div>
                )}
              </div>

              {/* Weather & Forecast */}
              <div className="glass-card fade-in" style={{ animationDelay: '0.3s' }}>
                <div className="card-header">
                  <CloudRain size={18} className="brand-icon" />
                  Weather Conditions
                </div>
                {weather ? (
                  <>
                    <div className="weather-main">
                      <div className="weather-details">
                        <div className="weather-temp">{typeof weather.temperature === 'object' ? weather.temperature.celsius : weather.temperature}°C</div>
                        <div className="weather-desc">{typeof weather.condition === 'object' ? JSON.stringify(weather.condition) : weather.condition}</div>
                      </div>
                      <Sun size={48} className="weather-icon-large" />
                    </div>
                    
                    {forecast && forecast.predictions && forecast.predictions.solar_prediction && (
                      <div className="details-list" style={{ marginTop: '12px' }}>
                        <div className="detail-item" style={{ padding: '8px 12px' }}>
                          <div className="detail-label">Solar Forecast</div>
                          <div className="detail-value val-on">
                            {((forecast.predictions?.solar_prediction?.power_output || 0) * 100).toFixed(1)} W
                          </div>
                        </div>
                        <div className="detail-item" style={{ padding: '8px 12px' }}>
                          <div className="detail-label">Wind Forecast</div>
                          <div className="detail-value val-on">
                            {(forecast.predictions?.wind_prediction?.power_output || 0).toFixed(1) } W
                          </div>
                        </div>
                        <div className="detail-item" style={{ padding: '8px 12px' }}>
                          <div className="detail-label">Power Cut Prob.</div>
                          <div className={`detail-value ${(forecast.predictions?.power_cut_prediction?.prob_cut || 0) > 0.5 ? 'val-warning' : ''}`}>
                            {((forecast.predictions?.power_cut_prediction?.prob_cut || 0) * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="loading-state">Fetching weather...</div>
                )}
              </div>
            </div>
          </>
        ) : currentView === 'analytics' ? (
          <AnalyticsDashboard />
        ) : (
          <DecisionPanel />
        )}

      </main>
    </div>
  );
}

export default function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}
