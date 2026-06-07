import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';
import { 
  Activity, 
  Zap, 
  Battery, 
  Clock, 
  Cpu, 
  CheckCircle2, 
  BarChart2
} from 'lucide-react';
import './AnalyticsDashboard.css';

// Mock Data
const successRateData = [
  { name: 'Success', value: 98.6 },
  { name: 'Failed', value: 1.4 },
];

const AnalyticsDashboard: React.FC = () => {
  const [energyUsage, setEnergyUsage] = useState<any[]>([]);
  const [sourceUtilization, setSourceUtilization] = useState<any[]>([]);
  const [aiAccuracy, setAiAccuracy] = useState<any[]>([]);
  const [kpiStats, setKpiStats] = useState<any>({
    grid_dependency: 0,
    renewable_usage: 0,
    battery_efficiency: 0,
    uptime: 0
  });
  const [eventStats, setEventStats] = useState<any>({
    total_decisions: 0,
    successful_switching: 0,
    grid_failover: 0,
    battery_protection: 0,
    manual_overrides: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const [energyRes, sourceRes, accuracyRes, kpiRes, eventRes] = await Promise.all([
          fetch('/api/analytics/energy-usage'),
          fetch('/api/analytics/source-utilization'),
          fetch('/api/analytics/ai-accuracy'),
          fetch('/api/analytics/kpi-stats'),
          fetch('/api/analytics/event-stats')
        ]);

        if (energyRes.ok) setEnergyUsage(await energyRes.json());
        if (sourceRes.ok) setSourceUtilization(await sourceRes.json());
        if (accuracyRes.ok) setAiAccuracy(await accuracyRes.json());
        if (kpiRes.ok) setKpiStats(await kpiRes.json());
        if (eventRes.ok) setEventStats(await eventRes.json());
      } catch (err) {
        console.error("Failed to fetch analytics data", err);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
    // Refresh every minute
    const interval = setInterval(fetchAnalytics, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading && energyUsage.length === 0) {
    return (
      <div className="analytics-container flex items-center justify-center min-height-100vh">
        <div className="loading-state">Loading Analytics...</div>
      </div>
    );
  }

  return (
    <div className="analytics-container fade-in">
      <header className="dashboard-header">
        <h1>System Performance Analytics Dashboard</h1>
      </header>

      {/* KPI Section */}
      <div className="kpi-grid">
        <div className="kpi-card" style={{ animationDelay: '0.1s' }}>
          <div className="kpi-label flex items-center gap-2">
            <Zap size={16} /> Grid Dependency
          </div>
          <div className="kpi-value">{kpiStats.grid_dependency}%</div>
        </div>
        <div className="kpi-card" style={{ animationDelay: '0.2s' }}>
          <div className="kpi-label flex items-center gap-2">
            <Activity size={16} /> Renewable Usage
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-green)' }}>{kpiStats.renewable_usage}%</div>
        </div>
        <div className="kpi-card" style={{ animationDelay: '0.3s' }}>
          <div className="kpi-label flex items-center gap-2">
            <Battery size={16} /> Battery Efficiency
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-magenta)' }}>{kpiStats.battery_efficiency}%</div>
        </div>
        <div className="kpi-card" style={{ animationDelay: '0.4s' }}>
          <div className="kpi-label flex items-center gap-2">
            <Clock size={16} /> Uptime
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-yellow)' }}>{kpiStats.uptime}%</div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="charts-grid">
        <div className="chart-card fade-in" style={{ animationDelay: '0.5s' }}>
          <div className="chart-title">
            <BarChart2 size={20} /> Daily Energy Usage (kWh)
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={energyUsage.length > 0 ? energyUsage : []}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Bar dataKey="kwh" fill="var(--accent-cyan)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card fade-in" style={{ animationDelay: '0.6s' }}>
          <div className="chart-title">
            <Activity size={20} /> Source Utilization (24 Hours)
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sourceUtilization.length > 0 ? sourceUtilization : []}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="solar" stackId="1" stroke="var(--accent-yellow)" fill="var(--accent-yellow)" fillOpacity={0.6} />
                <Area type="monotone" dataKey="wind" stackId="1" stroke="var(--accent-cyan)" fill="var(--accent-cyan)" fillOpacity={0.6} />
                <Area type="monotone" dataKey="battery" stackId="1" stroke="var(--accent-magenta)" fill="var(--accent-magenta)" fillOpacity={0.6} />
                <Area type="monotone" dataKey="grid" stackId="1" stroke="#475569" fill="#475569" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="charts-grid">
        <div className="chart-card fade-in" style={{ animationDelay: '0.7s' }}>
          <div className="chart-title">
            <Cpu size={20} /> AI Decision Accuracy
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={aiAccuracy.length > 0 ? aiAccuracy : []}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="predicted" stroke="var(--accent-cyan)" strokeWidth={2} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="actual" stroke="var(--accent-magenta)" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card fade-in" style={{ animationDelay: '0.8s' }}>
          <div className="chart-title">
            <CheckCircle2 size={20} /> Switching Success Rate
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={successRateData}
                  cx="50%"
                  cy="50%"
                  innerRadius={80}
                  outerRadius={120}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {successRateData.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? 'var(--accent-green)' : '#ef4444'} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', marginTop: '-100px', fontSize: '1.5rem', fontWeight: 'bold' }}>
              98.6%
            </div>
          </div>
        </div>
      </div>

      {/* Event Log Summary */}
      <div className="event-log-card fade-in" style={{ animationDelay: '0.9s' }}>
        <div className="event-log-title">Event Log Summary</div>
        <div className="event-stats">
          <div className="event-stat-item">
            <span className="event-stat-label">Total Decisions Today</span>
            <span className="event-stat-value">{eventStats.total_decisions}</span>
          </div>
          <div className="event-stat-item">
            <span className="event-stat-label">Successful Switching Events</span>
            <span className="event-stat-value" style={{ color: 'var(--accent-green)' }}>{eventStats.successful_switching}</span>
          </div>
          <div className="event-stat-item">
            <span className="event-stat-label">Grid Failover Events</span>
            <span className="event-stat-value highlight">{eventStats.grid_failover}</span>
          </div>
          <div className="event-stat-item">
            <span className="event-stat-label">Battery Protection Triggers</span>
            <span className="event-stat-value" style={{ color: '#ef4444' }}>{eventStats.battery_protection}</span>
          </div>
          <div className="event-stat-item">
            <span className="event-stat-label">Manual Overrides</span>
            <span className="event-stat-value">{eventStats.manual_overrides}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;
