import React, { useEffect } from 'react';
import { Brain, Sliders, CheckCircle2, ShieldAlert, Sparkles, Activity } from 'lucide-react';
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useAnalysis } from '../components/layout/MainLayout';

export default function AttentionAnalysis() {
  const { activeAnalysis, getActiveRecording, setHeaderInfo } = useAnalysis();
  const recording = getActiveRecording();

  useEffect(() => {
    setHeaderInfo({
      title: 'Model 3 — Temporal Attention Classifier',
      subtitle: '4-head additive temporal attention with recording-specific dynamic threshold calibration.',
    });
  }, [setHeaderInfo]);

  if (!activeAnalysis || !recording) {
    return (
      <div className="p-8 text-center bg-background-panel rounded-xl border border-border">
        <p className="text-xs text-text-muted">No patient recording loaded. Please upload or select an analysis session.</p>
      </div>
    );
  }

  const attnData = recording.models?.attention || {};
  const windows = recording.windows || [];
  const thInfo = attnData.threshold_info || {};
  const calibratedTh = attnData.calibrated_threshold || thInfo.calibrated_threshold || 0.03;
  const events = attnData.events || [];

  const chartData = windows.map((w) => ({
    time: `${w.start_seconds.toFixed(0)}s`,
    startSec: w.start_seconds,
    prob: w.attention?.probability || 0,
    th: w.attention?.threshold || calibratedTh,
    pred: w.attention?.prediction || 0,
    window_id: w.window_id,
  }));

  return (
    <div className="space-y-6">
      {/* Top Model Badge & Configuration Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-background-panel border border-purple-accent/30 flex items-center justify-between">
          <div>
            <span className="text-[10px] text-text-muted block">Architecture</span>
            <span className="text-sm font-bold text-purple-accent font-mono">4-Head Attention</span>
            <span className="text-[10px] text-text-muted block">Gated Residual Fusion • 128d</span>
          </div>
          <Brain className="w-6 h-6 text-purple-accent" />
        </div>

        <div className="p-4 rounded-xl bg-background-panel border border-border">
          <span className="text-[10px] text-text-muted block">Recording Dynamic Threshold</span>
          <span className="text-base font-bold text-purple-accent font-mono">{calibratedTh.toFixed(4)}</span>
          <span className="text-[10px] text-text-muted block">
            Quantile {thInfo.quantile || 0.985} • Floor {thInfo.floor || 0.03}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-background-panel border border-border">
          <span className="text-[10px] text-text-muted block">Positive 20s Windows</span>
          <span className="text-base font-bold text-purple-accent font-mono">
            {attnData.positive_window_count || 0} <span className="text-xs text-text-muted font-normal">/ {windows.length}</span>
          </span>
          <span className="text-[10px] text-text-muted block">
            {windows.length > 0 ? (((attnData.positive_window_count || 0) / windows.length) * 100).toFixed(1) : 0}% of recording
          </span>
        </div>

        <div className="p-4 rounded-xl bg-background-panel border border-border">
          <span className="text-[10px] text-text-muted block">Persistent Alarm Episodes</span>
          <span className="text-base font-bold text-text-primary font-mono">{events.length}</span>
          <span className="text-[10px] text-text-muted block">Filtered by persistence</span>
        </div>
      </div>

      {/* Dynamic Calibration Diagnostics Block */}
      <div className="p-4 rounded-xl bg-background-panel border border-border/80 flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs font-mono">
        <div className="flex items-center space-x-2 text-text-secondary">
          <Sliders className="w-4 h-4 text-purple-accent" />
          <span className="font-semibold text-text-primary">Calibration Parameters:</span>
          <span className="text-text-muted">Method: {thInfo.method || 'hybrid_global_recording_quantile'}</span>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-text-muted">
          <span>Global Th: <b className="text-text-primary">{thInfo.global_threshold || 0.03}</b></span>
          <span>•</span>
          <span>Quantile: <b className="text-text-primary">{thInfo.quantile || 0.985}</b></span>
          <span>•</span>
          <span>Adaptive Blend: <b className="text-text-primary">{thInfo.adaptive_blend || 1.0}</b></span>
          <span>•</span>
          <span>Floor: <b className="text-text-primary">{thInfo.floor || 0.03}</b></span>
        </div>
      </div>

      {/* Probability & Dynamic Threshold Chart */}
      <div className="p-5 rounded-xl bg-background-panel border border-border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xs font-semibold text-text-primary">Attention Seizure Probability vs Dynamic Threshold</h3>
            <p className="text-[11px] text-text-muted">
              Raw attention probabilities with recording-specific calibrated threshold curve ({calibratedTh.toFixed(4)})
            </p>
          </div>
          <div className="flex items-center space-x-2 text-xs font-mono">
            <span className="w-2.5 h-0.5 bg-purple-accent inline-block" />
            <span className="text-text-muted text-[11px]">Attention Probability</span>
            <span className="w-2.5 h-0.5 bg-rose-accent inline-block ml-3" />
            <span className="text-text-muted text-[11px]">Recording Threshold ({calibratedTh.toFixed(4)})</span>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="attnProbGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#A78BFA" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#A78BFA" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#23324B" vertical={false} />
              <XAxis dataKey="time" stroke="#64748B" fontSize={10} tickLine={false} />
              <YAxis domain={[0, 1]} stroke="#64748B" fontSize={10} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111D32', borderColor: '#23324B', borderRadius: '8px', fontSize: '11px' }}
                formatter={(val, name) => [
                  typeof val === 'number' ? val.toFixed(4) : val,
                  name === 'prob' ? 'Probability' : 'Threshold',
                ]}
              />
              <Area type="monotone" dataKey="prob" stroke="#A78BFA" strokeWidth={2} fill="url(#attnProbGradient)" />
              <Line type="stepAfter" dataKey="th" stroke="#FB7185" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Alarm Episodes & Window Table */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Episodes */}
        <div className="p-5 rounded-xl bg-background-panel border border-border">
          <h3 className="text-xs font-semibold text-text-primary mb-3">Persistent Alarm Episodes ({events.length})</h3>
          {events.length > 0 ? (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {events.map((evt) => (
                <div key={evt.event_id} className="p-3 rounded-lg bg-background-secondary border border-border text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-purple-accent">{evt.event_id}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-accent/10 border border-purple-accent/30 text-purple-accent font-mono">
                      Peak: {evt.max_probability.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-text-muted text-[11px]">
                    Interval: <span className="font-mono text-text-primary">{evt.start_seconds}s – {evt.end_seconds}s</span> ({evt.duration_seconds}s)
                  </div>
                  <div className="text-[10px] text-text-muted">
                    Contiguous Windows: {evt.window_count} (W#{evt.start_window_index} - W#{evt.end_window_index})
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-text-muted italic">No persistent seizure alarm episodes formed under current threshold.</p>
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
                  <th className="p-2">Dynamic Th.</th>
                  <th className="p-2">Classification</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                {windows.map((w) => {
                  const isPos = w.attention?.prediction === 1;
                  return (
                    <tr key={w.window_id} className={`hover:bg-background-secondary/40 ${isPos ? 'bg-purple-accent/5' : ''}`}>
                      <td className="p-2 text-text-primary">{w.window_id}</td>
                      <td className="p-2 text-text-muted">{w.start_seconds.toFixed(1)}s</td>
                      <td className="p-2 text-text-muted">{w.end_seconds.toFixed(1)}s</td>
                      <td className="p-2 font-bold text-purple-accent">{(w.attention?.probability || 0).toFixed(4)}</td>
                      <td className="p-2 text-text-muted">{(w.attention?.threshold || calibratedTh).toFixed(4)}</td>
                      <td className="p-2">
                        {isPos ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-accent/20 text-purple-accent border border-purple-accent/40">
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
