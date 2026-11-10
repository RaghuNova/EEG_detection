import React, { useEffect, useState } from 'react';
import { GitCompare, Eye, Brain, Cpu, Layers, CheckCircle2, AlertCircle } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useAnalysis } from '../components/layout/MainLayout';
import { useNavigate } from 'react-router-dom';

export default function ModelComparison() {
  const { activeAnalysis, getActiveRecording, setHeaderInfo } = useAnalysis();
  const recording = getActiveRecording();
  const navigate = useNavigate();

  const [filterMode, setFilterMode] = useState('all'); // all, agreed, disagreed

  useEffect(() => {
    setHeaderInfo({
      title: 'Model Comparison & Concordance Analysis',
      subtitle: 'Side-by-side descriptive evaluation of ML, CNN-BiLSTM, and Attention predictions.',
    });
  }, [setHeaderInfo]);

  if (!activeAnalysis || !recording) {
    return (
      <div className="p-8 text-center bg-background-panel rounded-xl border border-border">
        <p className="text-xs text-text-muted">No patient recording loaded. Please upload or select an analysis session.</p>
      </div>
    );
  }

  const ml = recording.models?.ml || {};
  const cnn = recording.models?.cnn_bilstm || {};
  const attn = recording.models?.attention || {};
  const windows = recording.windows || [];

  // Metrics summary
  const mlProbs = ml.probabilities || [];
  const cnnProbs = cnn.probabilities || [];
  const attnProbs = attn.probabilities || [];

  const mlMean = mlProbs.length > 0 ? (mlProbs.reduce((a, b) => a + b, 0) / mlProbs.length).toFixed(4) : '0.0000';
  const cnnMean = cnnProbs.length > 0 ? (cnnProbs.reduce((a, b) => a + b, 0) / cnnProbs.length).toFixed(4) : '0.0000';
  const attnMean = attnProbs.length > 0 ? (attnProbs.reduce((a, b) => a + b, 0) / attnProbs.length).toFixed(4) : '0.0000';

  const mlMax = mlProbs.length > 0 ? Math.max(...mlProbs).toFixed(4) : '0.0000';
  const cnnMax = cnnProbs.length > 0 ? Math.max(...cnnProbs).toFixed(4) : '0.0000';
  const attnMax = attnProbs.length > 0 ? Math.max(...attnProbs).toFixed(4) : '0.0000';

  const chartData = windows.map((w) => ({
    time: `${w.start_seconds.toFixed(0)}s`,
    ML: w.ml?.probability || 0,
    CNN: w.cnn_bilstm?.probability || 0,
    Attention: w.attention?.probability || 0,
    window_id: w.window_id,
  }));

  // Filter windows based on concordance
  const filteredWindows = windows.filter((w) => {
    const p1 = w.ml?.prediction || 0;
    const p2 = w.cnn_bilstm?.prediction || 0;
    const p3 = w.attention?.prediction || 0;
    const allAgree = (p1 === p2) && (p2 === p3);

    if (filterMode === 'agreed') return allAgree;
    if (filterMode === 'disagreed') return !allAgree;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Metrics Comparison Table */}
      <div className="rounded-xl border border-border bg-background-panel overflow-hidden">
        <div className="px-5 py-3.5 border-b border-border bg-background-secondary/50 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <GitCompare className="w-4 h-4 text-cyan-accent" />
            <h2 className="text-xs font-semibold text-text-primary">Independent Model Performance Metrics</h2>
          </div>
          <span className="text-[10px] text-text-muted font-mono">{recording.filename}</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-background-secondary text-text-muted border-b border-border text-[11px]">
              <tr>
                <th className="p-3">Evaluation Metric</th>
                <th className="p-3 text-blue-accent font-bold">1. ML (XGBoost)</th>
                <th className="p-3 text-cyan-accent font-bold">2. CNN-BiLSTM</th>
                <th className="p-3 text-purple-accent font-bold">3. AttentionClassifier</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 font-mono">
              <tr>
                <td className="p-3 text-text-secondary font-sans font-medium">Positive Seizure Windows</td>
                <td className="p-3 font-bold text-text-primary">{ml.positive_window_count || 0}</td>
                <td className="p-3 font-bold text-text-primary">{cnn.positive_window_count || 0}</td>
                <td className="p-3 font-bold text-text-primary">{attn.positive_window_count || 0}</td>
              </tr>
              <tr>
                <td className="p-3 text-text-secondary font-sans font-medium">Applied Decision Threshold</td>
                <td className="p-3">{(ml.decision_threshold || 0.279).toFixed(4)}</td>
                <td className="p-3">{(cnn.decision_threshold || 0.50).toFixed(4)}</td>
                <td className="p-3 font-bold text-purple-accent">{(attn.calibrated_threshold || 0.03).toFixed(4)} <span className="text-[9px] text-text-muted font-normal">(Dynamic)</span></td>
              </tr>
              <tr>
                <td className="p-3 text-text-secondary font-sans font-medium">Mean Window Probability</td>
                <td className="p-3">{mlMean}</td>
                <td className="p-3">{cnnMean}</td>
                <td className="p-3">{attnMean}</td>
              </tr>
              <tr>
                <td className="p-3 text-text-secondary font-sans font-medium">Peak Window Probability</td>
                <td className="p-3">{mlMax}</td>
                <td className="p-3">{cnnMax}</td>
                <td className="p-3">{attnMax}</td>
              </tr>
              <tr>
                <td className="p-3 text-text-secondary font-sans font-medium">Formed Event Episodes</td>
                <td className="p-3">{(ml.events || []).length}</td>
                <td className="p-3">{(cnn.events || []).length}</td>
                <td className="p-3">{(attn.events || []).length}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Multi-Track Probability Comparison Chart */}
      <div className="p-5 rounded-xl bg-background-panel border border-border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xs font-semibold text-text-primary">Probability Evolution Over Time</h3>
            <p className="text-[11px] text-text-muted">Simultaneous overlay of ML, CNN, and Attention probability curves</p>
          </div>
          <div className="flex items-center space-x-4 text-[11px] font-mono">
            <span className="flex items-center space-x-1.5">
              <div className="w-2.5 h-0.5 bg-blue-accent" />
              <span className="text-text-muted">ML (XGB)</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <div className="w-2.5 h-0.5 bg-cyan-accent" />
              <span className="text-text-muted">CNN-BiLSTM</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <div className="w-2.5 h-0.5 bg-purple-accent" />
              <span className="text-text-muted">Attention</span>
            </span>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#23324B" vertical={false} />
              <XAxis dataKey="time" stroke="#64748B" fontSize={10} tickLine={false} />
              <YAxis domain={[0, 1]} stroke="#64748B" fontSize={10} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111D32', borderColor: '#23324B', borderRadius: '8px', fontSize: '11px' }}
                formatter={(val) => [typeof val === 'number' ? val.toFixed(4) : val, 'Probability']}
              />
              <Line type="monotone" dataKey="ML" stroke="#4D8DFF" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="CNN" stroke="#36C5E8" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="Attention" stroke="#A78BFA" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3-Track Event Timeline */}
      <div className="p-5 rounded-xl bg-background-panel border border-border space-y-4">
        <div>
          <h3 className="text-xs font-semibold text-text-primary">3-Track Seizure Detection Timeline</h3>
          <p className="text-[11px] text-text-muted">Visual alignment of positive classifications across the recording</p>
        </div>

        <div className="space-y-3 font-mono text-[11px]">
          {/* Track 1: ML */}
          <div>
            <div className="flex items-center justify-between text-text-muted mb-1 text-[10px]">
              <span className="text-blue-accent font-bold">1. ML Classifier Track</span>
              <span>{(ml.events || []).length} events</span>
            </div>
            <div className="h-6 rounded bg-background border border-border flex p-0.5 overflow-hidden">
              {windows.map((w, i) => (
                <div
                  key={i}
                  title={`ML W#${i}: prob=${(w.ml?.probability || 0).toFixed(3)} pred=${w.ml?.prediction}`}
                  className={`flex-1 h-full mx-[0.5px] rounded-sm transition-colors ${
                    w.ml?.prediction === 1 ? 'bg-blue-accent' : 'bg-transparent hover:bg-border/40'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Track 2: CNN */}
          <div>
            <div className="flex items-center justify-between text-text-muted mb-1 text-[10px]">
              <span className="text-cyan-accent font-bold">2. CNN-BiLSTM Track</span>
              <span>{(cnn.events || []).length} events</span>
            </div>
            <div className="h-6 rounded bg-background border border-border flex p-0.5 overflow-hidden">
              {windows.map((w, i) => (
                <div
                  key={i}
                  title={`CNN W#${i}: prob=${(w.cnn_bilstm?.probability || 0).toFixed(3)} pred=${w.cnn_bilstm?.prediction}`}
                  className={`flex-1 h-full mx-[0.5px] rounded-sm transition-colors ${
                    w.cnn_bilstm?.prediction === 1 ? 'bg-cyan-accent' : 'bg-transparent hover:bg-border/40'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Track 3: Attention */}
          <div>
            <div className="flex items-center justify-between text-text-muted mb-1 text-[10px]">
              <span className="text-purple-accent font-bold">3. Attention Classifier Track (Dynamic)</span>
              <span>{(attn.events || []).length} events</span>
            </div>
            <div className="h-6 rounded bg-background border border-border flex p-0.5 overflow-hidden">
              {windows.map((w, i) => (
                <div
                  key={i}
                  title={`Attention W#${i}: prob=${(w.attention?.probability || 0).toFixed(3)} pred=${w.attention?.prediction}`}
                  className={`flex-1 h-full mx-[0.5px] rounded-sm transition-colors ${
                    w.attention?.prediction === 1 ? 'bg-purple-accent' : 'bg-transparent hover:bg-border/40'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Window-by-Window Concordance Table */}
      <div className="p-5 rounded-xl bg-background-panel border border-border space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xs font-semibold text-text-primary">Window-by-Window Concordance Explorer</h3>
            <p className="text-[11px] text-text-muted">Inspect agreement and divergence across all three models</p>
          </div>

          <div className="flex items-center space-x-1.5 bg-background p-1 rounded-lg border border-border text-xs">
            <button
              onClick={() => setFilterMode('all')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                filterMode === 'all' ? 'bg-background-panel text-text-primary' : 'text-text-muted hover:text-text-primary'
              }`}
            >
              All ({windows.length})
            </button>
            <button
              onClick={() => setFilterMode('disagreed')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                filterMode === 'disagreed' ? 'bg-background-panel text-amber-accent' : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Divergent
            </button>
            <button
              onClick={() => setFilterMode('agreed')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                filterMode === 'agreed' ? 'bg-background-panel text-emerald-accent' : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Concordant
            </button>
          </div>
        </div>

        <div className="overflow-y-auto max-h-72 border border-border rounded-lg">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-background-secondary text-text-muted border-b border-border text-[11px] sticky top-0">
              <tr>
                <th className="p-2">Window</th>
                <th className="p-2">Interval</th>
                <th className="p-2 text-blue-accent">ML (p)</th>
                <th className="p-2 text-cyan-accent">CNN (p)</th>
                <th className="p-2 text-purple-accent">Attention (p)</th>
                <th className="p-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {filteredWindows.map((w) => {
                const p1 = w.ml?.prediction || 0;
                const p2 = w.cnn_bilstm?.prediction || 0;
                const p3 = w.attention?.prediction || 0;
                const concordant = (p1 === p2) && (p2 === p3);

                return (
                  <tr key={w.window_id} className="hover:bg-background-secondary/40">
                    <td className="p-2 text-text-primary">{w.window_id}</td>
                    <td className="p-2 text-text-muted">{w.start_seconds.toFixed(1)}s – {w.end_seconds.toFixed(1)}s</td>
                    <td className="p-2 font-bold text-blue-accent">{(w.ml?.probability || 0).toFixed(3)} ({p1})</td>
                    <td className="p-2 font-bold text-cyan-accent">{(w.cnn_bilstm?.probability || 0).toFixed(3)} ({p2})</td>
                    <td className="p-2 font-bold text-purple-accent">{(w.attention?.probability || 0).toFixed(3)} ({p3})</td>
                    <td className="p-2">
                      {concordant ? (
                        <span className="text-[10px] text-emerald-accent flex items-center space-x-1">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>Agreed ({p1 ? 'Seizure' : 'Normal'})</span>
                        </span>
                      ) : (
                        <span className="text-[10px] text-amber-accent flex items-center space-x-1">
                          <AlertCircle className="w-3 h-3" />
                          <span>Divergence</span>
                        </span>
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
  );
}
