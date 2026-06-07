import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  XCircle, 
  Settings, 
  Download, 
  Zap, 
  Battery, 
  Wind, 
  Sun,
  Activity
} from 'lucide-react';
import './DecisionPanel.css';

const DecisionPanel: React.FC = () => {
  const [decision, setDecision] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDecision = async () => {
      try {
        const res = await fetch('/api/decision/latest');
        if (res.ok) {
          const data = await res.json();
          if (Object.keys(data).length > 0) {
            setDecision(data);
          }
        }
      } catch (err) {
        console.error("Failed to fetch latest decision", err);
      } finally {
        setLoading(false);
      }
    };

    fetchDecision();
    const interval = setInterval(fetchDecision, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  if (loading && !decision) {
    return (
      <div className="decision-panel-container flex items-center justify-center min-h-[400px]">
        <div className="loading-state">Loading latest AI decision...</div>
      </div>
    );
  }

  if (!decision) {
    return (
      <div className="decision-panel-container flex items-center justify-center min-h-[400px]">
        <div className="loading-state">No decisions recorded in the last hour.</div>
      </div>
    );
  }

  const { timestamp, input, output } = decision;

  const inputParameters = [
    { label: "Solar Power", value: `${(input.solar_power).toFixed(1)} W`, icon: Sun },
    { label: "Wind Power", value: `${(input.wind_power).toFixed(1)} W`, icon: Wind },
    { label: "Battery SOC", value: `${(input.battery_soc).toFixed(1)}%`, icon: Battery, color: input.battery_soc > 20 ? "highlight-green" : "highlight-yellow" },
    { label: "Critical Load Demand", value: `${(input.critical_load).toFixed(1)} W`, icon: Zap },
    { label: "Non-Critical Load", value: `${(input.non_critical_load).toFixed(1)} W`, icon: Activity },
    { label: "Grid Available", value: input.grid_available ? "YES" : "NO", icon: ShieldCheck, color: input.grid_available ? "highlight-green" : "highlight-yellow" },
    { label: "Solar Forecast", value: `${(input.solar_forecast).toFixed(1)} W`, icon: Sun, color: "highlight-yellow" },
    { label: "Load Forecast", value: `${(input.load_forecast).toFixed(1)} W`, icon: Activity },
    { label: "Power Cut Probability", value: `${(input.power_cut_prob * 100).toFixed(1)}%`, icon: Zap, color: input.power_cut_prob < 0.3 ? "highlight-green" : "highlight-yellow" },
  ];

  const decisionOutput = [
    { label: "Critical Load Source", value: output.critical_source.toUpperCase() },
    { label: "Non-Critical Load Source", value: output.noncritical_source.toUpperCase() },
    { label: "Battery Action", value: output.battery_action },
    { label: "Grid Action", value: output.grid_action },
  ];

  const logicRules = [
    `Critical load prioritized to ${output.critical_source} based on system state`,
    `Non-critical load shifted to ${output.noncritical_source} to optimize balance`,
    `Battery status: ${output.battery_action} (SOC at ${input.battery_soc.toFixed(1)}%)`,
    input.grid_available ? "Grid retained as backup support" : "Grid unavailable - island mode active"
  ];

  return (
    <div className="decision-panel-container fade-in">
      <header className="panel-header">
        <h1>AI Decision Engine Response Panel</h1>
        <div className="timestamp-badge">
          {timestamp}
        </div>
      </header>

      <div className="panel-grid">
        {/* Input Parameters Section */}
        <div className="section-box">
          <div className="section-title">Input Parameters</div>
          <div className="parameter-list">
            {inputParameters.map((param, index) => (
              <div key={index} className="parameter-item">
                <div className="parameter-label flex items-center gap-2">
                  <param.icon size={14} /> {param.label}
                </div>
                <div className={`parameter-value ${param.color || ''}`}>
                  {param.value}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Decision Output Section */}
        <div className="flex flex-col gap-6">
          <div className="section-box">
            <div className="section-title">AI Decision Output</div>
            <div className="grid grid-cols-2 gap-4">
              {decisionOutput.map((output, index) => (
                <div key={index} className="decision-output-item">
                  <span className="decision-label">{output.label}</span>
                  <span className="decision-value">{output.value}</span>
                </div>
              ))}
            </div>
            
            <div className="confidence-bar-container">
              <div className="confidence-header">
                <span className="decision-label">Decision Confidence</span>
                <span className="parameter-value highlight-cyan">94.7%</span>
              </div>
              <div className="confidence-bar">
                <div className="confidence-fill" style={{ width: '94.7%' }}></div>
              </div>
            </div>
          </div>

          <div className="section-box">
            <div className="section-title">Decision Logic Trace</div>
            <div className="logic-trace-list">
              {logicRules.map((rule, index) => (
                <div key={index} className="logic-rule">
                  <span className="rule-number">RULE {index + 1}:</span>
                  <span>{rule}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Action Footer */}
      <div className="action-footer">
        <button className="action-btn btn-accept">
          <ShieldCheck size={18} /> Accepted
        </button>
        <button className="action-btn btn-reject">
          <XCircle size={18} /> Rejected
        </button>
        <button className="action-btn btn-override">
          <Settings size={18} /> Manual Override
        </button>
        <button className="action-btn btn-export">
          <Download size={18} /> Export Log
        </button>
      </div>
    </div>
  );
};

export default DecisionPanel;
