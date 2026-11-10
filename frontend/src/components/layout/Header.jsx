import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileDown, Layers, ChevronRight, Activity } from 'lucide-react';
import { api } from '../../services/api';

export default function Header({
  title,
  subtitle,
  activeAnalysis,
  activeRecordingId,
  onSelectRecording,
}) {
  const navigate = useNavigate();

  return (
    <header className="h-16 border-b border-border bg-background-secondary/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Title & Context */}
      <div className="flex items-center space-x-3">
        <div>
          <h1 className="text-sm font-semibold text-text-primary tracking-tight">{title}</h1>
          {subtitle && <p className="text-[11px] text-text-muted">{subtitle}</p>}
        </div>

        {activeAnalysis && (
          <div className="hidden lg:flex items-center space-x-2 pl-4 border-l border-border/80">
            <span className="text-[11px] text-text-muted">Active:</span>
            <span className="text-xs px-2 py-0.5 rounded bg-cyan-accent/10 border border-cyan-accent/30 text-cyan-accent font-medium font-mono">
              {activeAnalysis.patient_id}
            </span>

            {activeAnalysis.recordings && activeAnalysis.recordings.length > 0 && onSelectRecording && (
              <select
                value={activeRecordingId || ''}
                onChange={(e) => onSelectRecording(e.target.value)}
                className="text-xs bg-background-panel border border-border rounded px-2.5 py-1 text-text-primary focus:outline-none focus:border-cyan-accent font-mono cursor-pointer"
              >
                {activeAnalysis.recordings.map((r) => (
                  <option key={r.recording_id} value={r.recording_id}>
                    {r.filename} ({r.total_windows}w)
                  </option>
                ))}
              </select>
            )}
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-3">
        {activeAnalysis && (
          <a
            href={api.getPdfReportUrl(activeAnalysis.analysis_id)}
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-background-panel border border-border text-text-secondary hover:text-text-primary hover:border-text-muted transition-colors"
          >
            <FileDown className="w-3.5 h-3.5 text-cyan-accent" />
            <span>Download Report</span>
          </a>
        )}

        <button
          onClick={() => navigate('/upload')}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-accent text-background font-semibold hover:bg-cyan-accent/90 transition-all shadow-sm"
        >
          <Upload className="w-3.5 h-3.5" />
          <span>Upload EEG</span>
        </button>
      </div>
    </header>
  );
}
