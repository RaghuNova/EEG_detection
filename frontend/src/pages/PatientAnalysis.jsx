import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Activity, Brain, Cpu, FileDown, Waves } from 'lucide-react';
import { useAnalysis } from '../components/layout/MainLayout';
import { api } from '../services/api';

const MODEL_CARDS = [
  { key: 'ml', label: 'Existing ML model', field: 'ml_positive_windows', icon: Brain },
  { key: 'cnn_bilstm', label: 'CNN-BiLSTM', field: 'cnn_positive_windows', icon: Cpu },
  { key: 'attention', label: 'AttentionClassifier', field: 'attention_positive_windows', icon: Activity },
];

export default function PatientAnalysis() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { activeAnalysis, setActiveAnalysis, setActiveRecordingId, setHeaderInfo } = useAnalysis();
  const [loadedAnalysis, setLoadedAnalysis] = useState(null);
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    setHeaderInfo({ title: 'Patient Analysis', subtitle: 'Recording-level EEG inference results and model summaries.' });
  }, [setHeaderInfo]);

  useEffect(() => {
    if (activeAnalysis?.analysis_id === id) { setLoadedAnalysis(activeAnalysis); return; }
    let alive = true;
    api.getAnalysisResults(id).then((data) => {
      if (alive) { setLoadedAnalysis(data); setActiveAnalysis(data); }
    }).catch((error) => {
      if (alive) setLoadError(error.response?.data?.detail || error.message || 'Unable to load this analysis.');
    });
    return () => { alive = false; };
  }, [activeAnalysis, id, setActiveAnalysis]);

  const analysis = activeAnalysis?.analysis_id === id ? activeAnalysis : loadedAnalysis;
  if (loadError) return <div className="p-5 rounded-xl border border-amber-accent/30 bg-amber-accent/10 text-sm text-amber-accent">{loadError}</div>;
  if (!analysis) return <div className="p-8 rounded-xl border border-border bg-background-panel text-sm text-text-muted">Loading analysis results…</div>;

  const summary = analysis.summary || {};
  const recordings = analysis.recordings || [];
  const successCount = analysis.successful_recordings || 0;
  const failureCount = analysis.failed_recordings || 0;

  return <div className="space-y-6">
    <div className="flex flex-wrap justify-between items-start gap-4">
      <div>
        <p className="text-xs text-text-muted">Patient <b className="text-text-primary">{analysis.patient_id}</b> · Analysis <span className="font-mono">{id}</span></p>
        <h2 className="text-xl font-semibold mt-1">Analysis results</h2>
        <p className="text-xs text-text-muted mt-1">{successCount} completed, {failureCount} failed of {recordings.length} recordings</p>
      </div>
      <a href={api.getPdfReportUrl(id)} className="px-4 py-2 rounded-lg bg-cyan-accent text-background text-xs font-semibold flex items-center gap-2"><FileDown size={15}/> Download PDF</a>
    </div>

    <div className="grid md:grid-cols-3 gap-4">
      {MODEL_CARDS.map(({ key, label, field, icon: Icon }) => <div key={key} className="p-5 rounded-xl border border-border bg-background-panel">
        <div className="flex items-center gap-2 text-text-muted text-xs"><Icon size={16}/>{label}</div>
        <div className="text-3xl font-semibold mt-3">{summary[field] ?? 0}</div>
        <div className="text-[11px] text-text-muted mt-1">positive windows across completed recordings</div>
        <button onClick={() => navigate(`/analysis/${key === 'cnn_bilstm' ? 'cnn' : key}`)} className="mt-4 text-xs text-cyan-accent hover:underline">Open model analysis →</button>
      </div>)}
    </div>

    <div className="p-5 rounded-xl border border-border bg-background-panel">
      <h3 className="font-semibold text-sm mb-3">Recording results</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left"><thead className="text-text-muted border-b border-border"><tr>
          <th className="py-2 pr-3">Recording</th><th className="py-2 pr-3">Status</th><th className="py-2 pr-3">Duration</th><th className="py-2 pr-3">ML</th><th className="py-2 pr-3">CNN</th><th className="py-2 pr-3">Attention</th><th className="py-2">Explore</th>
        </tr></thead><tbody>
          {recordings.map((recording) => <tr key={recording.recording_id} className="border-b border-border/50">
            <td className="py-3 pr-3 font-mono">{recording.filename}</td>
            <td className={`py-3 pr-3 ${recording.status === 'completed' ? 'text-emerald-accent' : 'text-rose-accent'}`}>{recording.status}{recording.processing?.error ? `: ${recording.processing.error}` : ''}</td>
            <td className="py-3 pr-3">{recording.duration_seconds ? `${(recording.duration_seconds / 60).toFixed(1)} min` : '—'}</td>
            <td className="py-3 pr-3">{recording.models?.ml?.positive_window_count ?? '—'}</td>
            <td className="py-3 pr-3">{recording.models?.cnn_bilstm?.positive_window_count ?? '—'}</td>
            <td className="py-3 pr-3">{recording.models?.attention?.positive_window_count ?? '—'}</td>
            <td className="py-3"><button disabled={recording.status !== 'completed'} onClick={() => { setActiveRecordingId(recording.recording_id); navigate('/viewer'); }} className="text-cyan-accent disabled:opacity-40 flex items-center gap-1"><Waves size={14}/> EEG</button></td>
          </tr>)}
        </tbody></table>
      </div>
    </div>
    <div className="text-[11px] text-text-muted">Inference outputs are analysis support information and are not a standalone medical diagnosis.</div>
  </div>;
}
