import React, { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Activity,
  Upload,
  FolderOpen,
  Cpu,
  Layers,
  Brain,
  GitCompare,
  Eye,
  Settings,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  FileText
} from 'lucide-react';
import { api } from '../../services/api';

const navItems = [
  { to: '/', label: 'Overview', icon: Activity },
  { to: '/upload', label: 'Patient Upload', icon: Upload },
  { to: '/records', label: 'Patient Records', icon: FolderOpen },
  { to: '/analysis/ml', label: 'ML Model', icon: Cpu, badge: 'XGBoost', color: 'text-blue-accent' },
  { to: '/analysis/cnn', label: 'CNN-BiLSTM', icon: Layers, badge: 'Raw EEG', color: 'text-cyan-accent' },
  { to: '/analysis/attention', label: 'Attention Model', icon: Brain, badge: 'Dynamic', color: 'text-purple-accent' },
  { to: '/comparison', label: 'Model Comparison', icon: GitCompare },
  { to: '/viewer', label: 'EEG Signal Viewer', icon: Eye },
  { to: '/settings', label: 'System Settings', icon: Settings },
];

export default function Sidebar() {
  const [backendOnline, setBackendOnline] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.getHealth();
        setBackendOnline(health?.status === 'healthy');
      } catch (err) {
        setBackendOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="w-64 bg-background-secondary border-r border-border flex flex-col h-screen select-none shrink-0 z-30">
      {/* Brand Header */}
      <div className="p-5 border-b border-border/80 flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-accent/20 to-blue-accent/20 border border-cyan-accent/40 flex items-center justify-center glow-cyan">
          <Brain className="w-5 h-5 text-cyan-accent" />
        </div>
        <div>
          <div className="flex items-center space-x-1.5">
            <span className="font-extrabold tracking-wider text-text-primary text-base">NEUROSCAN</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-accent/10 border border-cyan-accent/30 text-cyan-accent font-mono font-medium">v1.0</span>
          </div>
          <p className="text-[11px] text-text-muted tracking-tight truncate">EEG Intelligence Platform</p>
        </div>
      </div>

      {/* Nav Menu */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="text-[10px] font-semibold tracking-wider text-text-muted uppercase px-3 pb-2">
          Platform Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to));

          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={`flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all duration-150 group ${
                isActive
                  ? 'bg-background-panel text-cyan-accent border border-border shadow-sm'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-panel/50'
              }`}
            >
              <div className="flex items-center space-x-3 truncate">
                <Icon className={`w-4 h-4 shrink-0 transition-colors ${isActive ? 'text-cyan-accent' : 'text-text-muted group-hover:text-text-primary'}`} />
                <span className="truncate">{item.label}</span>
              </div>
              {item.badge && (
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
                  isActive ? 'bg-cyan-accent/15 text-cyan-accent border border-cyan-accent/30' : 'bg-background border border-border text-text-muted'
                }`}>
                  {item.badge}
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* System Status Footer */}
      <div className="p-3 border-t border-border/80 bg-background/50">
        <div className="p-2.5 rounded-lg bg-background-panel/80 border border-border flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-accent animate-pulse' : 'bg-rose-accent'}`} />
            <div>
              <div className="text-[11px] font-medium text-text-primary">
                {backendOnline ? 'Inference Online' : 'Connecting...'}
              </div>
              <div className="text-[9px] text-text-muted font-mono">
                {backendOnline ? 'FastAPI 127.0.0.1:8000' : 'Offline / Checking'}
              </div>
            </div>
          </div>
          {backendOnline ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-accent" />
          ) : (
            <AlertCircle className="w-3.5 h-3.5 text-rose-accent" />
          )}
        </div>
      </div>
    </aside>
  );
}
