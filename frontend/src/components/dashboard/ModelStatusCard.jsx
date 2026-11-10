import React from 'react';
import { Cpu, Layers, Brain, CheckCircle2, ShieldCheck, Zap } from 'lucide-react';

export default function ModelStatusCard({
  type,
  modelName,
  architecture,
  threshold,
  positiveWindows,
  totalWindows,
  status = 'Loaded & Active',
}) {
  const configs = {
    ml: {
      accent: 'blue',
      border: 'border-blue-accent/30',
      bgGlow: 'bg-blue-accent/5',
      badge: 'bg-blue-accent/15 text-blue-accent border-blue-accent/30',
      icon: Cpu,
      iconColor: 'text-blue-accent',
      thresholdLabel: 'Fixed F1 Threshold',
    },
    cnn: {
      accent: 'cyan',
      border: 'border-cyan-accent/30',
      bgGlow: 'bg-cyan-accent/5',
      badge: 'bg-cyan-accent/15 text-cyan-accent border-cyan-accent/30',
      icon: Layers,
      iconColor: 'text-cyan-accent',
      thresholdLabel: 'Logistic Cutoff',
    },
    attention: {
      accent: 'purple',
      border: 'border-purple-accent/30',
      bgGlow: 'bg-purple-accent/5',
      badge: 'bg-purple-accent/15 text-purple-accent border-purple-accent/30',
      icon: Brain,
      iconColor: 'text-purple-accent',
      thresholdLabel: 'Hybrid Recording Dynamic',
    },
  };

  const config = configs[type] || configs.ml;
  const Icon = config.icon;
  const percent = totalWindows > 0 ? ((positiveWindows / totalWindows) * 100).toFixed(1) : '0.0';

  return (
    <div className={`p-5 rounded-xl bg-background-panel border ${config.border} shadow-sm relative overflow-hidden flex flex-col justify-between`}>
      <div className={`absolute -right-8 -top-8 w-24 h-24 rounded-full ${config.bgGlow} blur-xl pointer-events-none`} />

      <div>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${config.badge}`}>
              <Icon className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-xs font-semibold text-text-primary tracking-tight">{modelName}</h3>
              <p className="text-[10px] text-text-muted truncate">{architecture}</p>
            </div>
          </div>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${config.badge}`}>
            {status}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 mt-4 pt-3 border-t border-border/60">
          <div>
            <span className="text-[10px] text-text-muted block">Operating Threshold</span>
            <span className="text-sm font-bold text-text-primary font-mono">{threshold}</span>
            <span className="text-[9px] text-text-muted block truncate">{config.thresholdLabel}</span>
          </div>
          <div>
            <span className="text-[10px] text-text-muted block">Seizure Detections</span>
            <span className="text-sm font-bold text-text-primary font-mono">
              {positiveWindows} <span className="text-[10px] text-text-muted font-normal">({percent}%)</span>
            </span>
            <span className="text-[9px] text-text-muted block">{totalWindows} windows evaluated</span>
          </div>
        </div>
      </div>

      <div className="mt-3 pt-2.5 border-t border-border/40 flex items-center justify-between text-[10px] text-text-muted">
        <span className="flex items-center space-x-1">
          <ShieldCheck className="w-3 h-3 text-emerald-accent" />
          <span>Inference Ready</span>
        </span>
        <span className="font-mono text-[9px] text-text-secondary">Artifact Verified</span>
      </div>
    </div>
  );
}
