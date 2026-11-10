import React, { useEffect, useState } from 'react';
import { Cpu, ShieldCheck, Clock, AlertTriangle, Layers, Calendar } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useAnalysis } from '../components/layout/MainLayout';

export default function MLAnalysis() {
  const { activeAnalysis, getActiveRecording, setHeaderInfo } = useAnalysis();
  const recording = getActiveRecording();

  useEffect(() => {
    setHeaderInfo({
      title: 'Model 1 — Tabular ML Analysis (XGBoost)',
      subtitle: 'Evaluation of 122 channel-order-invariant statistical, spectral, and synchrony features.',
    });
  }, [setHeaderInfo]);

  if (!activeAnalysis || !recording) {
    return (
      <div className="p-8 text-center bg-background-panel rounded-xl border border-border">
        <p className="text-xs text-text-muted">No patient recording loaded. Please upload or select an analysis session.</p>
      </div>
    );
  }

  const mlData = recording.models?.ml || {};
  const windows = recording.windows || [];
  const threshold = mlData.decision_threshold || 0.279;
  const events = mlData.events || [];

  // Prepare chart series
  const chartData = windows.map((w) => ({
    time: `${w.start_seconds.toFixed(0)}s`,
    startSec: w.start_seconds,
    prob: w.ml?.probability || 0,
    pred: w.ml?.prediction || 0,
    window_id: w.window_id,
  }));

  return (
    <div className="space-y-6">
      {/* Top Model Badge & Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-background-panel border border-blue-accent/30 flex items-center justify-between">
          <div>
            <span className="text-[10px] text-text-muted block">Algorithm</span>
            <span className="text-sm font-bold text-blue-accent font-mono">XGBoost Baseline</span>
            <span className="text-[10px] text-text-muted block">122 engineered features</span>
          </div>
          <Cpu className="w-6 h-6 text-blue-accent" />
        </div>

        <div className="p-4 rounded-xl bg-background-panel border border-border">
          <span className="text-[10px] text-text-muted block">Decision Cutoff Threshold</span>
          <span className="text-base font-bold text-text-primary font-mono">{threshold.toFixed(4)}</span>
          <span className="text-[10px] text-text-muted block">Trained Validation Max F1</span>
        </div>

        <div className="p-4 rounded-xl bg-background-panel border border-border">
          <span className="text-[10px] text-text-muted block">Positive 20s Windows</span>
          <span className="text-base font-bold text-blue-accent font-mono">
            {mlData.positive_window_count || 0} <span className="text-xs text-text-muted font-normal">/ {windows.length}</span>
          </span>
          <span className="text-[10px] text-text-muted block">
            {windows.length > 0 ? (((mlData.positive_window_count || 0) / windows.length) * 100).toFixed(1) : 0}% of recording
          </span>
        </div>

        <div className="p-4 rounded-xl bg-background-panel border border-border">
          <span className="text-[10px] text-text-muted block">Detected Seizure Episodes</span>
          <span className="text-base font-bold text-text-primary font-mono">{events.length}</span>
          <span className="text-[10px] text-text-muted block">Grouped contiguous windows</span>
        </div>
      </div>

      {/* Probability Chart */}
      <div className="p-5 rounded-xl bg-background-panel border border-border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xs font-semibold text-text-primary">Window Seizure Probability Timeline</h3>
            <p className="text-[11px] text-text-muted">
              Probability per 20-second non-overlapping window relative to decision threshold ({threshold.toFixed(4)})
            </p>
          </div>
          <div className="flex items-center space-x-2 text-xs font-mono">
            <span className="w-2.5 h-0.5 bg-blue-accent inline-block" />
            <span className="text-text-muted text-[11px]">Seizure Probability</span>
            <span className="w-2.5 h-0.5 bg-rose-accent inline-block ml-3" />
            <span className="text-text-muted text-[11px]">Threshold ({threshold.toFixed(4)})</span>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="mlProbGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4D8DFF" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#4D8DFF" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#23324B" vertical={false} />
              <XAxis dataKey="time" stroke="#64748B" fontSize={10} tickLine={false} />
              <YAxis domain={[0, 1]} stroke="#64748B" fontSize={10} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111D32', borderColor: '#23324B', borderRadius: '8px', fontSize: '11px' }}
                formatter={(val) => [typeof val === 'number' ? val.toFixed(4) : val, 'Probability']}
              />
              <ReferenceLine y={threshold} stroke="#FB7185" strokeDasharray="3 3" />
              <Area type="monotone" dataKey="prob" stroke="#4D8DFF" strokeWidth={2} fill="url(#mlProbGradient)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Seizure Event Episodes & Window Table */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Event List */}
        <div className="p-5 rounded-xl bg-background-panel border border-border">
          <h3 className="text-xs font-semibold text-text-primary mb-3">Grouped Seizure Episodes ({events.length})</h3>
          {events.length > 0 ? (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {events.map((evt) => (
                <div key={evt.event_id} className="p-3 rounded-lg bg-background-secondary border border-border text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-blue-accent">{evt.event_id}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-accent/10 border border-blue-accent/30 text-blue-accent font-mono">
                      Peak: {evt.max_probability.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-text-muted text-[11px]">
                    Interval: <span className="font-mono text-text-primary">{evt.start_seconds}s – {evt.end_seconds}s</span> ({evt.duration_seconds}s duration)
                  </div>
                  <div className="text-[10px] text-text-muted">
                    Contiguous Windows: {evt.window_count} (W#{evt.start_window_index} - W#{evt.end_window_index})
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-text-muted italic">No seizure event episodes detected by ML threshold.</p>
          )}
        </div>

        {/* Windows Table */}
        <div className="lg:col-span-2 p-5 rounded-xl bg-background-panel border border-border overflow-hidden flex flex-col">
          <h3 className="text-xs font-semibold text-text-primary mb-3">Window Results ({windows.length} windows)</h3>
          <div className="overflow-y-auto max-h-72 border border-border rounded-lg">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-background-secondary text-text-muted border-b border-border text-[11px] sticky top-0">
                <tr>
                  <th className="p-2">Window</th>
                  <th className="p-2">Start</th>
                  <th className="p-2">End</th>
                  <th className="p-2">Probability</th>
                  <th className="p-2">Threshold</th>
                  <th className="p-2">Classification</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                {windows.map((w) => {
                  const isPos = w.ml?.prediction === 1;
                  return (
                    <tr key={w.window_id} className={`hover:bg-background-secondary/40 ${isPos ? 'bg-blue-accent/5' : ''}`}>
                      <td className="p-2 text-text-primary">{w.window_id}</td>
                      <td className="p-2 text-text-muted">{w.start_seconds.toFixed(1)}s</td>
                      <td className="p-2 text-text-muted">{w.end_seconds.toFixed(1)}s</td>
                      <td className="p-2 font-bold text-blue-accent">{(w.ml?.probability || 0).toFixed(4)}</td>
                      <td className="p-2 text-text-muted">{threshold.toFixed(4)}</td>
                      <td className="p-2">
                        {isPos ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-accent/20 text-blue-accent border border-blue-accent/40">
                            SEIZURE
                          </span>
                        ) : (
                          <span className="text-[10px] text-text-muted">Non-seizure</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
